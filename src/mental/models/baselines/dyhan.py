import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.models import GAT
from .. import BaseModel, ModelOutput
from ...utils.dataprocessing import prepare_dynamic_hetero_graphs, prepare_batch_dynamic_hetero_graphs
from ..utils import TemporalAttentionLayer, SimpleAttentionLayer
from torch_geometric.utils import add_self_loops

class DyHAN(BaseModel):
    def __init__(self, args, dataset_info):
        super(DyHAN, self).__init__()
        self.args = args
        self.config = args.mental_net.heterognn
        self.dataset_info = dataset_info
        temporal_config = args.temporal_attention_config
        hidden_channels = args.gnn_hidden_channels
        num_layers = args.gnn_num_layers
        self.act = F.relu
        self.num_time_steps = int(dataset_info.max_period_length)
        self.register_buffer('position_ids', torch.arange(self.num_time_steps))
        self.position_embeddings = nn.Embedding(self.num_time_steps, hidden_channels)
        self.structural_self_attention = nn.ModuleDict({
            relation: GAT(in_channels = -1, hidden_channels = hidden_channels, out_channels = hidden_channels, num_layers = num_layers) for relation in self.config.hetero_relations
        })

        self.temporal_self_attention_block = nn.ModuleList(
             [TemporalAttentionLayer(d_in = hidden_channels, dim_feedforward = hidden_channels, nhead = temporal_config.num_attention_heads, activation = self.act) for _ in range(temporal_config.num_layers)]
        )
        self.edge_attention_block = SimpleAttentionLayer(hidden_channels)

        self.depression_prediction_head = nn.Linear(hidden_channels, 1)

        self.allowed_inputs = ['graphs']
    

    def reset_parameters(self):
        self.position_embeddings.reset_parameters()
        for module in self.structural_self_attention.values():
            module.reset_parameters()
        self.edge_attention_block.reset_parameters()
        for module in self.temporal_self_attention_block:
            module.reset_parameters()
        self.depression_prediction_head.reset_parameters()

    def prepare_inputs(self, inputs):
        return prepare_dynamic_hetero_graphs(inputs)
    def prepare_batch_inputs(self, inputs):
        return prepare_batch_dynamic_hetero_graphs(inputs)
    def tweet_aggr(self, x):
        return torch.mean(x, dim = 1)
    
    def forward(self, graphs):
        batch_temporal_inputs = []
        attention_mask = []
        last_snapshot_indices = []
        # Structural Attention forward
        for user_graphs in graphs:
            # We only focus on the ego nodes, so we only take the representation of the ego nodes over different timestamps.
            temporal_inputs = []
            for graph in user_graphs:
                features = self.tweet_aggr(graph['user'].x)

                structural_outputs = []
                for relation in self.config.hetero_relations:
                    edge_index = graph[relation] if relation in graph else add_self_loops(torch.LongTensor([]), num_nodes = features.shape[0])[0].to(features.device)
                    structural_out = self.structural_self_attention[relation](features, edge_index)[graph.label != -100]
                    structural_outputs.append(structural_out)
                structural_outputs = torch.stack(structural_outputs, dim = 1)
                edge_attention_outputs = self.edge_attention_block(structural_outputs)
                temporal_inputs.append(edge_attention_outputs)
            temporal_inputs = torch.cat(temporal_inputs, dim = 0)
            
            batch_temporal_inputs.append(temporal_inputs)
            attention_mask.append(torch.ones((temporal_inputs.shape[0]), device = temporal_inputs.device))
            last_snapshot_indices.append(temporal_inputs.shape[0] - 1)
        
        batch_temporal_inputs = torch.nn.utils.rnn.pad_sequence(batch_temporal_inputs, batch_first = True, padding_value = 0.0)
        attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask, batch_first = True, padding_value = False).bool()
        key_padding_mask = ~attention_mask
        attn_mask = ~(torch.ones(key_padding_mask.shape[-1], key_padding_mask.shape[-1], device = self.position_ids.device).tril().bool()) & attention_mask

        batch_temporal_inputs = batch_temporal_inputs + self.position_embeddings(self.position_ids[:batch_temporal_inputs.shape[1]])
        # We used the representation of the ego node from the last graph snapshot.
        for temporal_attention_block in self.temporal_self_attention_block:
            batch_temporal_inputs = temporal_attention_block(batch_temporal_inputs, attn_mask = attn_mask, key_padding_mask = key_padding_mask)
        logits = batch_temporal_inputs[:, 0]
        cls_logits = self.depression_prediction_head(logits)
        prediction_scores = torch.sigmoid(cls_logits)
        return ModelOutput(
            logits = logits,
            cls_logits = cls_logits,
            prediction_scores = prediction_scores,
        )
