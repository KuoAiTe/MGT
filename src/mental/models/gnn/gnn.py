import torch
import torch.nn as nn
import torch.nn.functional as F
from .. import BaseModel, ModelOutput
from ...utils.dataprocessing import prepare_static_graph, prepare_batch_static_graph, prepare_dynamic_hyper_graphs

from torch_geometric.utils import add_self_loops,  coalesce, to_undirected
from typing import Any, Callable, Dict, List, Optional, Tuple, Union


from torch import Tensor
from torch.nn import ModuleList

from torch_geometric.nn.conv import (
    GATConv,
    GATv2Conv,
    GCNConv,
    MessagePassing,
    SAGEConv,
)

class GNNWrapper(BaseModel):
    def __init__(self, args, data_info):
        super().__init__()
        self.in_channels = data_info.num_features
        self.hidden_channels = args.gnn_hidden_channels
        self.num_layers = args.gnn_num_layers
        self.dropout = 0.0
        self.act = nn.ReLU()
        self.allowed_inputs = ['static_graph']
        self.convs = ModuleList()
        if self.num_layers > 1:
            self.convs.append(self.init_conv(self.in_channels, self.hidden_channels))
            in_channels = self.hidden_channels
        for _ in range(self.num_layers - 2):
            self.convs.append(self.init_conv(in_channels, self.hidden_channels))
            in_channels = self.hidden_channels
        self.convs.append(self.init_conv(in_channels, 1))
        self.reset_parameters()

    def init_conv(self, in_channels: Union[int, Tuple[int, int]],
                  out_channels: int, **kwargs) -> MessagePassing:
        raise NotImplementedError

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        for conv in self.convs:
            conv.reset_parameters()
        if hasattr(self, 'norms'):
            for norm in self.norms:
                norm.reset_parameters()
        if hasattr(self, 'feedfowards'):
            for lin in self.feedfowards:
                lin.reset_parameters()
        if hasattr(self, 'attn_convs'):
            for conv in self.attn_convs:
                conv.reset_parameters()
        
        if hasattr(self, 'lin'):
            self.lin.reset_parameters()

    def forward(self, graph,
        num_sampled_nodes_per_hop: Optional[List[int]] = None,
        num_sampled_edges_per_hop: Optional[List[int]] = None) -> Tensor:
        x = graph.features
        edge_index = graph.edge_index
        edge_weight = graph.edge_weight
        for i in range(self.num_layers):
            if i == self.num_layers - 1:
                logits = x[graph.label != -100]
            if num_sampled_nodes_per_hop is not None:
                x, edge_index, value = self._trim(
                    i,
                    num_sampled_nodes_per_hop,
                    num_sampled_edges_per_hop,
                    x,
                    edge_index,
                    edge_weight if edge_weight is not None else edge_attr,
                )
                if edge_weight is not None:
                    edge_weight = value
                else:
                    edge_attr = value

            # Tracing the module is not allowed with *args and **kwargs :(
            # As such, we rely on a static solution to pass optional edge
            # weights and edge attributes to the module.
            if self.supports_edge_weight:
                x = self.convs[i](x, edge_index, edge_weight = edge_weight)
            else:
                x = self.convs[i](x, edge_index)
            if i == self.num_layers - 1:
                break
            x = self.act(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        cls_logits = x[graph.label != -100]
        prediction_scores = torch.sigmoid(cls_logits)
        return ModelOutput(
            logits = logits,
            cls_logits = cls_logits,
            prediction_scores = prediction_scores,
        )
    

    def prepare_inputs(self, inputs):
        return prepare_static_graph(inputs)
    def prepare_batch_inputs(self, inputs):
        return prepare_batch_static_graph(inputs)


class GATWrapper(GNNWrapper):
    supports_edge_weight = False

    def init_conv(self, in_channels: Union[int, Tuple[int, int]],
                  out_channels: int, **kwargs) -> MessagePassing:
        return GATConv(in_channels, out_channels, heads=1, concat=True,
                    dropout=self.dropout, **kwargs)
    
class GCNWrapper(GNNWrapper):
    supports_edge_weight = True

    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:
        return GCNConv(in_channels, out_channels, **kwargs)
    
class GraphSAGEWrapper(GNNWrapper):
    supports_edge_weight = False

    def init_conv(self, in_channels: Union[int, Tuple[int, int]],
                  out_channels: int, **kwargs) -> MessagePassing:
        return SAGEConv(in_channels, out_channels, **kwargs)

class DynamicGNNWrapper(GNNWrapper):
    def __init__(self, args, data_info):
        super(DynamicGNNWrapper, self).__init__(args, data_info)
        self.allowed_inputs = ['hyper_graph']

        self.convs = ModuleList()
        if self.num_layers > 1:
            self.convs.append(self.init_conv(self.in_channels, self.hidden_channels))
            in_channels = self.hidden_channels
        for _ in range(self.num_layers - 1):
            self.convs.append(self.init_conv(in_channels, self.hidden_channels))
            in_channels = self.hidden_channels

        self.attn_convs = ModuleList()
        if self.num_layers > 1:
            self.attn_convs.append(self.init_temporal_conv(self.in_channels, self.hidden_channels))
            in_channels = self.hidden_channels
        for _ in range(self.num_layers - 1):
            self.attn_convs.append(self.init_temporal_conv(in_channels, self.hidden_channels))
            in_channels = self.hidden_channels
        
        self.act = nn.GELU()

        self.norms = nn.ModuleList(
            [nn.BatchNorm1d(self.hidden_channels, eps = 1e-12) for _ in range(self.num_layers)]
        )
        self.feedfowards = nn.ModuleList(
            [nn.Linear(self.hidden_channels, self.hidden_channels) for _ in range(self.num_layers)] + [nn.Linear(1, 1) ]
        )
        self.lin = nn.Linear(self.hidden_channels, 1)
    def prepare_inputs(self, inputs):
        return prepare_dynamic_hyper_graphs(inputs)
    
    def init_temporal_conv(self, in_channels: Union[int, Tuple[int, int]], out_channels: int, **kwargs) -> MessagePassing:
        return GATConv(in_channels, out_channels, **kwargs)
    
    def forward(self, graph,
            num_sampled_nodes_per_hop: Optional[List[int]] = None,
            num_sampled_edges_per_hop: Optional[List[int]] = None) -> Tensor:
            last_x = graph['user']['x']
            edge_index, edge_weight = to_undirected(graph.edge_index, graph.edge_weight)
            temporal_edge_index = graph.edge_index_dict['is']
            for i in range(self.num_layers):
                if num_sampled_nodes_per_hop is not None:
                    x, edge_index, value = self._trim(
                        i,
                        num_sampled_nodes_per_hop,
                        num_sampled_edges_per_hop,
                        x,
                        edge_index,
                        edge_weight if edge_weight is not None else edge_attr,
                    )
                    if edge_weight is not None:
                        edge_weight = value
                    else:
                        edge_attr = value

                # Tracing the module is not allowed with *args and **kwargs :(
                # As such, we rely on a static solution to pass optional edge
                # weights and edge attributes to the module.
                if self.supports_edge_weight:
                    gnn_x = self.convs[i](last_x, edge_index, edge_weight = edge_weight)
                else:
                    gnn_x = self.convs[i](last_x, edge_index)
                gnn_x = self.act(gnn_x)
                gnn_x = F.dropout(gnn_x, p=self.dropout, training = self.training)

                attn_x = self.attn_convs[i](last_x, temporal_edge_index)
                attn_x = self.act(attn_x)

                x = self.norms[i](gnn_x + attn_x)
                x = self.feedfowards[i](x)
                last_x = self.act(x)
            last_x = self.lin(last_x)
            logits = last_x
            cls_logits = logits[graph.last_snapshot]
            prediction_scores = torch.sigmoid(cls_logits)
            return ModelOutput(
                logits = logits,
                cls_logits = cls_logits,
                prediction_scores = prediction_scores,
            )
    
class DynamicGAT(DynamicGNNWrapper):
    supports_edge_weight = False

    def init_conv(self, in_channels: Union[int, Tuple[int, int]],
                  out_channels: int, **kwargs) -> MessagePassing:
        return GATConv(in_channels, out_channels, heads=1, concat=True,
                    dropout=self.dropout, **kwargs)
    
class DynamicGCN(DynamicGNNWrapper):
    supports_edge_weight = True

    def init_conv(self, in_channels: int, out_channels: int,
                  **kwargs) -> MessagePassing:
        return GCNConv(in_channels, out_channels, **kwargs)
    
class DynamicSAGE(DynamicGNNWrapper):
    supports_edge_weight = False

    def init_conv(self, in_channels: Union[int, Tuple[int, int]],
                  out_channels: int, **kwargs) -> MessagePassing:
        return SAGEConv(in_channels, out_channels, **kwargs)
