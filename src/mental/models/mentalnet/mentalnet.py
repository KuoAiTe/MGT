import numpy as np

import torch
import torch.nn as nn
from torch_geometric.nn import MLP
from torch_geometric.nn import SAGEConv, GATConv, GCNConv
from torch_geometric.nn import aggr
from torch_geometric.nn.conv.gatv2_conv import GATv2Conv
from torch_geometric.utils import add_self_loops
from .. import BaseModel, ModelOutput
from ...utils.dataprocessing import prepare_static_hetero_graph, prepare_batch_static_hetero_graph, prepare_static_hetero_graph_by_user


class MentalNet(BaseModel):
    def __init__(self, args, data_info):
        super(MentalNet, self).__init__()
        self.args = args
        config = args.mental_net
        self.config = config
        self.support_edge_weight = True
        heterognn_config = config.heterognn
        self.convs = nn.ModuleDict({
            relation: self.init_convs(heterognn_config) for relation in heterognn_config.hetero_relations
        })
        layer_size = heterognn_config.gnn_hidden_channel * heterognn_config.gnn_num_layer + 1
        
        self.conv1 = nn.Conv1d(1, config.conv1_channel, layer_size, layer_size)
        self.conv2 = nn.Conv1d(config.conv1_channel, config.conv2_channel, config.kernel_size, 1)
        self.lin_proj = nn.ModuleList([
            nn.Linear(heterognn_config.gnn_hidden_channel, heterognn_config.gnn_hidden_channel) for _ in range(heterognn_config.gnn_hidden_channel)
        ])
        self.maxpool = nn.MaxPool1d(2, 2)
        self.sort_aggr = aggr.SortAggregation(k = config.k)
        self.depression_prediction_head = nn.Linear(192, 1)
        # initialize parameters
        self.reset_parameters()

    
    def reset_parameters(self):
        # initialize parameters for each module
        for module_list in self.convs.values():
            for module in module_list:
                module.reset_parameters()
                    
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        for lin in self.lin_proj:
            lin.reset_parameters()
        self.sort_aggr.reset_parameters()
        self.depression_prediction_head.reset_parameters()

    def prepare_inputs(self, inputs):
        return prepare_static_hetero_graph(inputs)
        
    def prepare_batch_inputs(self, inputs):
        return prepare_batch_static_hetero_graph(inputs)
        
    def init_convs(self, heterognn_config):
        convs = torch.nn.ModuleList()
        for i in range(heterognn_config.gnn_num_layer):
            conv = self.init_conv(in_channels = -1, out_channels = heterognn_config.gnn_hidden_channel)
            convs.append(conv)
        convs.append(self.init_conv(in_channels = -1, out_channels = 1))
        return convs

    def init_conv(self, in_channels, out_channels):
        return GCNConv(in_channels = in_channels, out_channels = out_channels)
        
    def forward(self, graph):
        x_dict, edge_index_dict, edge_weight_dict, group = graph.x_dict, graph.edge_index_dict, graph.edge_weight_dict, graph.group
        features = x_dict['user']
        outputs = []

        inputs = {relation: features for relation in self.config.heterognn.hetero_relations}
        outputs = {relation: [] for relation in self.config.heterognn.hetero_relations}
        for i in range(self.config.heterognn.gnn_num_layer):
            for relation in self.config.heterognn.hetero_relations:
                if relation in edge_index_dict:
                    edge_index = edge_index_dict[relation]
                    edge_weight = edge_weight_dict[relation] 
                else:
                    edge_index = add_self_loops(torch.LongTensor([]), num_nodes = inputs[relation].shape[0])[0].to(features.device)
                    edge_weight = torch.ones(inputs[relation].shape[0], device = features.device)
                inputs[relation] = self.convs[relation][i](inputs[relation], edge_index, edge_weight = edge_weight)
                inputs[relation] = inputs[relation].tanh()
                outputs[relation].append(inputs[relation])
            
        out = []
        for i in range(self.config.heterognn.gnn_num_layer):
            layer_outputs = []
            for key in outputs.keys():
                layer_outputs.append(outputs[key][i])
            layer_outputs = torch.stack(layer_outputs, dim = -1)
            layer_outputs = torch.mean(layer_outputs, dim = -1)
            layer_outputs = self.lin_proj[i](layer_outputs)
            out.append(layer_outputs)
        outputs = torch.cat(out, dim = -1)
        
        x = self.sort_aggr(outputs, index = group)
        x = x.view(x.size(0), 1, x.size(-1))
        x = self.conv1(x).relu()
        x = self.maxpool(x)
        x = self.conv2(x).relu()
        x = x.flatten(1)
        out.append(x)
        logits = x.view(x.size(0), -1)
        
        cls_logits = self.depression_prediction_head(logits)
        prediction_scores = torch.sigmoid(cls_logits)
        return ModelOutput(
            logits = logits,
            cls_logits = cls_logits,
            prediction_scores = prediction_scores
        )

    

class MentalNet_Original(MentalNet):
    def __init__(self, args, data_info):
        super(MentalNet_Original, self).__init__(args, data_info)
    def forward(self, graphs):
        logits = []
        for graph in graphs:
            x_dict, edge_index_dict, edge_weight_dict = graph.x_dict, graph.edge_index_dict, graph.edge_weight_dict
            features = x_dict['user']
            outputs = {}
            for relation, edge_index in edge_index_dict.items():
                y = []
                x = features
                edge_weight = edge_weight_dict[relation]
                for conv in self.convs[relation]:
                    if self.support_edge_weight:
                        x = conv(x, edge_index, edge_weight = edge_weight).tanh()
                    else:
                        x = conv(x, edge_index).tanh()
                    y.append(x)
                x = torch.cat(y, dim = -1)
                outputs[relation] = x
            out = []
            for relation in edge_index_dict.keys():
                x = outputs[relation]
                x = self.sort_aggr(x)
                x = x.view(x.size(0), 1, x.size(-1))
                x = self.conv1(x).relu()
                x = self.maxpool(x)
                x = self.conv2(x).relu()
                x = x.view(x.size(0), -1)
                out.append(x)
            
            out = torch.cat(out, dim = 0)
            out = torch.sum(out, dim = 0).unsqueeze(dim = 0)
            logits.append(out)
        logits = torch.cat(logits)
        cls_logits = self.depression_prediction_head(logits)
        prediction_scores = torch.sigmoid(cls_logits)
        return ModelOutput(
            logits = logits,
            cls_logits = cls_logits,
            prediction_scores = prediction_scores
        )
    
    def prepare_inputs(self, inputs):
        return prepare_static_hetero_graph_by_user(inputs)

class MentalNetNaive(nn.Module):
    def __init__(self, args, data_info):
        super(MentalNetNaive, self).__init__()
        self.args = args
        config = args.mental_net
        self.config = config
        self.support_edge_weight = True
        self.convs = nn.ModuleDict({
            relation: self.init_convs() for relation in self.config.hetero_relations
        })
        self.depression_prediction_head = nn.Linear(64 * 3, 1)

    def init_conv(self, in_channels, out_channels):
        return GATConv(in_channels = in_channels, out_channels = out_channels)

    def init_convs(self, **kwargs):
        convs = torch.nn.ModuleList()
        for i in range(3):
            conv = self.init_conv(in_channels = -1, out_channels = self.config.gnn_hidden_channel)
            convs.append(conv)
        return convs

    def forward(self, inputs):
        x_dict, edge_index_dict, edge_weight_dict = inputs.x_dict, inputs.edge_index_dict, inputs.edge_weight_dict
        features = x_dict['user']
        outputs = []
        for relation, edge_index in edge_index_dict.items():
            x = features
            #edge_index = inputs.edge_index
            edge_weight = edge_weight_dict[relation]
            i = 0
            for conv in self.convs[relation]:
                i += 1
                x = conv(x, edge_index, edge_weight = edge_weight)
                if i != len(self.convs[relation]):
                    x = x.relu()
            outputs.append(x)
        x = torch.cat(outputs, dim = 1)

        return x

    def prepare_inputs(self, inputs):
        return prepare_static_hetero_graph(inputs)

class MentalNet_GAT2(MentalNet):
    def __init__(self, args, data_info):
        super(MentalNet_GAT2, self).__init__(args, data_info)
        
    def init_conv(self, in_channels, out_channels):
        self.support_edge_weight = False
        return GATv2Conv(in_channels = in_channels, out_channels = out_channels, heads = 1, concat = False, residual = True)
        
class MentalNet_GAT(MentalNet):
    def __init__(self, args, data_info):
        super(MentalNet_GAT, self).__init__(args, data_info)

    def init_conv(self, in_channels, out_channels):
        self.support_edge_weight = False
        return GATConv(in_channels = in_channels, out_channels = out_channels)

class MentalNet_SAGE(MentalNet):
    def __init__(self, args, data_info):
        super(MentalNet_SAGE, self).__init__(args, data_info)

    def init_conv(self, in_channels, out_channels):
        self.support_edge_weight = False
        return SAGEConv(in_channels = in_channels, out_channels = out_channels)
        