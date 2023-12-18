import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import mental
#from torch_geometric_temporal.nn.recurrent.evolvegcnh import EvolveGCNH

from torch.nn import GRU
from torch_geometric.nn import TopKPooling

#from torch_geometric_temporal.nn.recurrent.evolvegcno import glorot, GCNConv_Fixed_W


class EvolveGCNH(torch.nn.Module):
    r"""An implementation of the Evolving Graph Convolutional Hidden Layer.
    For details see this paper: `"EvolveGCN: Evolving Graph Convolutional
    Networks for Dynamic Graph." <https://arxiv.org/abs/1902.10191>`_

    Args:
        num_of_nodes (int): Number of vertices.
        in_channels (int): Number of filters.
        improved (bool, optional): If set to :obj:`True`, the layer computes
            :math:`\mathbf{\hat{A}}` as :math:`\mathbf{A} + 2\mathbf{I}`.
            (default: :obj:`False`)
        cached (bool, optional): If set to :obj:`True`, the layer will cache
            the computation of :math:`\mathbf{\hat{D}}^{-1/2} \mathbf{\hat{A}}
            \mathbf{\hat{D}}^{-1/2}` on first execution, and will use the
            cached version for further executions.
            This parameter should only be set to :obj:`True` in transductive
            learning scenarios. (default: :obj:`False`)
        normalize (bool, optional): Whether to add self-loops and apply
            symmetric normalization. (default: :obj:`True`)
        add_self_loops (bool, optional): If set to :obj:`False`, will not add
            self-loops to the input graph. (default: :obj:`True`)
    """

    def __init__(
        self,
        num_of_nodes: int,
        in_channels: int,
        improved: bool = False,
        cached: bool = False,
        normalize: bool = True,
        add_self_loops: bool = True,
    ):
        super(EvolveGCNH, self).__init__()

        self.num_of_nodes = num_of_nodes
        self.in_channels = in_channels
        self.improved = improved
        self.cached = cached
        self.normalize = normalize
        self.add_self_loops = add_self_loops
        self.weight = None
        self.initial_weight = torch.nn.Parameter(torch.Tensor(in_channels, in_channels))
        self._create_layers()
        self.reset_parameters()
    
    def reset_parameters(self):
        glorot(self.initial_weight)


    def _create_layers(self):

        self.ratio = self.in_channels / self.num_of_nodes

        self.pooling_layer = TopKPooling(self.in_channels, self.ratio)

        self.recurrent_layer = GRU(
            input_size=self.in_channels, hidden_size=self.in_channels, num_layers=1
        )

        self.conv_layer = GCNConv_Fixed_W(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            improved=self.improved,
            cached=self.cached,
            normalize=self.normalize,
            add_self_loops=self.add_self_loops
        )

    def forward(
        self,
        X: torch.FloatTensor,
        edge_index: torch.LongTensor,
        edge_weight: torch.FloatTensor = None,
    ) -> torch.FloatTensor:
        """
        Making a forward pass.

        Arg types:
            * **X** *(PyTorch Float Tensor)* - Node embedding.
            * **edge_index** *(PyTorch Long Tensor)* - Graph edge indices.
            * **edge_weight** *(PyTorch Float Tensor, optional)* - Edge weight vector.

        Return types:
            * **X** *(PyTorch Float Tensor)* - Output matrix for all nodes.
        """
        X_tilde = self.pooling_layer(X, edge_index)
        X_tilde = X_tilde[0][None, :, :]
        if self.weight is None:
            self.weight = self.initial_weight.data
        W = self.weight[None, :, :]
        X_tilde, W = self.recurrent_layer(X_tilde, W)
        X = self.conv_layer(W.squeeze(dim=0), X, edge_index, edge_weight)
        return X


class EvolveGCN(nn.Module):
    def __init__(self, args, data_info):
        super(EvolveGCN, self).__init__()
        self.evolve_gcn = EvolveGCNH(num_of_nodes = 1271, in_channels=768)
        self.depression_prediction_head = nn.Linear(768, 1)
        self.loss_fct = nn.BCEWithLogitsLoss()
    
    def prepare_data(self, data):
        return mental.utils.utilities.prepare_static_graph(data)


    def predict(self, inputs):
        graph = inputs['static_graph']
        outputs = self.forward(graph) # [N, T, F]
        indices = (graph.label != -100)
        prediction_scores = self.depression_prediction_head(outputs[indices]).flatten()
        labels = graph.label[indices]
        is_depressed = torch.zeros_like(labels)
        is_depressed[prediction_scores >= 0] = 1
        return labels, is_depressed

    def compute_loss(self, inputs):
        graph = inputs['static_graph']
        print("ffff")
        outputs = self.forward(graph)
        indices = (graph.label != -100)
        prediction_scores = self.depression_prediction_head(outputs[indices]).flatten()
        labels = graph.label[indices].float()
        loss = self.loss_fct(prediction_scores, labels)
        return outputs, loss

    def forward(self, graph):
        print("forward")
        node_embeddings, edge_index, edge_weight, edge_time = graph.features, graph.edge_index, graph.weight, graph.time
        edge_list = []
        for time in edge_time.unique():
            edge_list.append((edge_index[:, edge_time == time], edge_weight[edge_time == time]))
        print(len(edge_list))
        for (edge_index, edge_weight) in edge_list:
            print('???', node_embeddings.shape)
            node_embeddings =  self.evolve_gcn(node_embeddings, edge_index)
        return node_embeddings