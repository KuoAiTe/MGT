import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, SAGEConv
from torch_geometric.utils import add_self_loops, to_undirected, dropout_edge

from torch_geometric.nn.models.mlp import MLP

class SimpleAttentionLayer(nn.Module):
    def __init__(self, input_dim, activation = F.relu):
        super(SimpleAttentionLayer, self).__init__()
        self.input_dim = input_dim
        self.lin_proj = nn.Linear(input_dim, input_dim)
        self.attenion_weights = nn.Parameter(torch.Tensor(input_dim, 1))
        self.act = activation
        self.softmax = nn.Softmax(dim = 1)
        self.dropout = nn.Dropout(p=0.5)
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.BatchNorm1d(input_dim, eps=1e-12),
            nn.ReLU(inplace=True),
            nn.Linear(input_dim, input_dim),
        )
    
    def reset_parameters(self):
        self.lin_proj.reset_parameters()
        nn.init.xavier_uniform_(self.attenion_weights)
        for module in self.ffn:
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

    def forward(self, x):
        inputs = x
        inputs = self.dropout(inputs)
        x = self.lin_proj(x)
        x = self.act(x)
        
        # Compute the attention weights
        attention_weights = x @ self.attenion_weights
        attention_weights = F.softmax(attention_weights, dim = 1)
        output = torch.einsum('bij, bik -> bj', inputs, attention_weights)
        output = self.ffn(output)
    
        return output
    

class HomoGNN(nn.Module):
    supports_edge_weight = True
    def __init__(self, config):
        super(HomoGNN, self).__init__()
        self.config = config
        self.config.hetero_relations = ['homo']
        self.init_setup(self.config.hetero_relations, config.gnn_hidden_channel, config.gnn_num_layer)

        self.dropout = 0.1
        
        self.position_embeddings = nn.Embedding(11, 64)
        self.act = nn.ReLU()
        self.self_attn_conv = self.init_convs(self.init_gat_conv)
        self.selfnorms = nn.ModuleList(
            [nn.BatchNorm1d(config.gnn_hidden_channel, eps=1e-12) for _ in range(config.gnn_num_layer)]
        )
        self.edge_attention_block = nn.ModuleList(
            SimpleAttentionLayer(config.gnn_hidden_channel) for _ in range(config.gnn_num_layer)
        )
            
        self.posnorm = nn.BatchNorm1d(config.gnn_hidden_channel, eps=1e-12)
        
        self.gnn_final_projection = nn.Linear(config.gnn_hidden_channel, config.gnn_hidden_channel, bias = True)

    def init_setup(self, hetero_relations, hidden_channel, num_layers):
        self.convs = nn.ModuleDict({
            relation: self.init_convs(self.init_gat_conv) for relation in hetero_relations
        })

        self.skip_conntections = nn.ModuleDict({
            relation: nn.ModuleList(
            [
                nn.Sequential(
            nn.Linear(hidden_channel, hidden_channel),
            nn.BatchNorm1d(hidden_channel, eps=1e-12),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channel, hidden_channel),
        ) for _ in range(num_layers)
            ]
        ) for relation in hetero_relations
        })
        self.ffn = nn.Sequential(
            nn.Linear(hidden_channel, hidden_channel),
            nn.BatchNorm1d(hidden_channel, eps=1e-12),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_channel, hidden_channel),
        )






        self.norms = nn.ModuleDict({
            relation:  nn.ModuleList(
            [nn.BatchNorm1d(hidden_channel, eps=1e-12) for _ in range(num_layers)]
        ) for relation in hetero_relations
        })

    def reset_parameters(self):
        for convs in self.convs.values():
            for conv in convs:
                conv.reset_parameters()
                
        for conv in self.self_attn_conv:
            conv.reset_parameters()

        for norms in self.norms.values():
            for norm in norms:
                norm.reset_parameters()
        for norm in self.selfnorms:
            norm.reset_parameters()

        for skip_conntections in self.skip_conntections.values():
            for modules in skip_conntections:
                for module in modules:
                    if hasattr(module, 'reset_parameters'):
                        module.reset_parameters()
        self.posnorm.reset_parameters()

        self.position_embeddings.reset_parameters()
        self.gnn_final_projection.reset_parameters()
        for module in self.edge_attention_block:
            module.reset_parameters()
        
    def init_convs(self, init_conv, add_self_loops = True):
        convs = nn.ModuleList()
        for _ in range(self.config.gnn_num_layer - 1):
            convs.append(init_conv(in_channels = -1, out_channels = self.config.gnn_hidden_channel, add_self_loops = add_self_loops))
        convs.append(init_conv(in_channels = -1, out_channels = self.config.gnn_hidden_channel, add_self_loops = add_self_loops))
        return convs
    
    def init_gat_conv(self, in_channels, out_channels, add_self_loops = add_self_loops):
        return GATConv(in_channels = in_channels, out_channels = out_channels, edge_dim = self.config.gnn_hidden_channel, add_self_loops = add_self_loops)
    
    def init_sage_conv(self, in_channels, out_channels):
        return SAGEConv(in_channels = in_channels, out_channels = out_channels)

    def init_gcn_conv(self, in_channels, out_channels):
        return GCNConv(in_channels = in_channels, out_channels = out_channels)
    
    def forward(self, last_x, edge_index_dict, edge_weight_dict, graph = None,  interaction_cutout = True):
        tweet_attention_weights = None
        edge_index = graph.edge_index
        if self.training and interaction_cutout:
            edge_index = dropout_edge(edge_index, p = 0.2, force_undirected = True, training = self.training)[0]
            edge_index = add_self_loops(edge_index)[0]
        for i in range(self.config.gnn_num_layer):
            last_x = self.convs['homo'][i](last_x, edge_index)
            last_x = self.norms['homo'][i](last_x)
            last_x = self.act(last_x)
        return last_x, tweet_attention_weights

class HeteroGNN(HomoGNN):
    supports_edge_weight = True
    def __init__(self, config):
        super(HeteroGNN, self).__init__(config)
        self.config = config
        self.config.hetero_relations = ['mention', 'reply', 'quote']
        self.init_setup(self.config.hetero_relations, config.gnn_hidden_channel, config.gnn_num_layer)
        self.act = nn.ReLU()
        self.reset_parameters()

    
    def forward(self, last_x, edge_index_dict, edge_weight_dict, graph = None, interaction_cutout = True):
        identity = last_x
        inputs = {relation: last_x for relation in self.config.hetero_relations}
        graph_attention_weights = {relation: [[] for _ in range(self.config.gnn_num_layer)] for relation in self.config.hetero_relations}
        for relation in self.config.hetero_relations:
            edge_index_dict[relation] = edge_index_dict[relation] if relation in edge_index_dict else add_self_loops(torch.LongTensor([]), num_nodes = inputs[relation].shape[0])[0].to(last_x.device)
            if self.training and interaction_cutout:
                edge_index = edge_index_dict[relation]
                edge_index = dropout_edge(edge_index, p = 0.2, force_undirected = True, training = self.training)[0]
                edge_index = add_self_loops(edge_index)[0]
                edge_index_dict[relation] = edge_index
        #relation = 'mention'
        #jk = []
        for i in range(self.config.gnn_num_layer):
            output = []
            use_relation = 'quote'
            for j, relation in enumerate(self.config.hetero_relations):
                edge_index = edge_index_dict[relation]
                #edge_embeddings = self.position_embeddings(torch.ones((edge_index.shape[1]), device = edge_index.device).long() * j)
                #edge_attr = edge_embeddings,
                x, graph_attention_weights[relation][i] = self.convs[relation][i](last_x, edge_index,  return_attention_weights=True)
                #x = x + self.skip_conntections[relation][i](last_x)
                x = self.norms[relation][i](x)
                x = self.act(x)
                inputs[relation] = x

            x = torch.stack(list(inputs.values()), dim = 1)
            x = torch.sum(x, dim = 1)
            last_x = self.skip_conntections[relation][i](x + last_x)
        #x = self.posnorm(last_x + x)
        #x = self.act(x)
        print("shared last_x")
        #last_x = self.gnn_final_projection(last_x)
        return last_x, graph_attention_weights

class HeteroDynamicGNN(HomoGNN):
    supports_edge_weight = True
    def __init__(self, config):
        super(HeteroDynamicGNN, self).__init__(config)
        self.config = config
        self.config.hetero_relations = ['mention', 'reply', 'quote']
        self.init_setup(self.config.hetero_relations, config.gnn_hidden_channel, config.gnn_num_layer)
        self.reset_parameters()

    def forward(self, last_x, edge_index_dict, edge_weight_dict, graph = None, use_cnn = True):
        inputs = {}
        graph_attention_weights = {relation: [] for relation in self.config.hetero_relations}
        for relation in self.config.hetero_relations:
            edge_index_dict[relation] = edge_index_dict[relation] if relation in edge_index_dict else add_self_loops(torch.LongTensor([]), num_nodes = inputs[relation].shape[0])[0].to(last_x.device)
        
        for i in range(self.config.gnn_num_layer):
            identity = last_x
            for relation in self.config.hetero_relations:
                inputs[relation], graph_attention_weights[relation] = self.convs[relation][i](last_x, edge_index_dict[relation], return_attention_weights=True)
                inputs[relation] = self.act(inputs[relation])

            last_x = torch.stack(list(inputs.values()), dim = 1)
            last_x = torch.sum(last_x, dim = 1)
            last_x = self.norms[relation][i](identity + last_x)
        last_x = last_x + self.position_embeddings(graph.period_id)
        last_x = self.posnorm(last_x)
        
        for i in range(self.config.gnn_num_layer):
            attn_edge = to_undirected(dropout_edge(edge_index_dict['is'], p = 0.2, training = self.training)[0])
            identity = last_x
            last_x = self.self_attn_conv[i](last_x, attn_edge)
            last_x = self.act(last_x)
            last_x = self.selfnorms[i](identity + last_x)
        return last_x, None
