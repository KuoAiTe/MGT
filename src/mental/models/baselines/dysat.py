import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.models import GAT
from .. import BaseModel, ModelOutput
from ...utils.dataprocessing import prepare_dynamic_homo_graphs, prepare_batch_dynamic_homo_graphs
from ..utils import TemporalAttentionLayer

class DySAT(BaseModel):
    def __init__(self, args, dataset_info):
        """[summary]

        Args:
            args ([type]): [description]
            time_length (int): Total timesteps in dataset.
        """
        super(DySAT, self).__init__()
        self.args = args
        self.dataset_info = dataset_info
        temporal_config = args.temporal_attention_config
        hidden_channels = args.gnn_hidden_channels
        num_layers = args.gnn_num_layers
        self.act = F.relu
        self.num_time_steps = int(dataset_info.max_period_length)
        self.register_buffer('position_ids', torch.arange(self.num_time_steps))
        self.position_embeddings = nn.Embedding(self.num_time_steps, hidden_channels)
        self.structural_self_attention = GAT(in_channels = -1, hidden_channels = hidden_channels, out_channels = hidden_channels, num_layers = num_layers)
        self.temporal_self_attention_block = nn.ModuleList(
             [TemporalAttentionLayer(d_in = hidden_channels, dim_feedforward = hidden_channels, nhead = temporal_config.num_attention_heads, activation = self.act) for _ in range(temporal_config.num_layers)]
        )

        self.depression_prediction_head = nn.Linear(hidden_channels, 1)
        self.allowed_inputs = ['graphs']

    def reset_parameters(self):
        self.position_embeddings.reset_parameters()
        self.structural_self_attention.reset_parameters()
        for module in self.temporal_self_attention_block:
            module.reset_parameters()
        self.depression_prediction_head.reset_parameters()

    def prepare_inputs(self, inputs):
        return prepare_dynamic_homo_graphs(inputs)
    
    def prepare_batch_inputs(self, inputs):
        return prepare_batch_dynamic_homo_graphs(inputs)
    
    def tweet_aggr(self, x):
        return torch.mean(x, dim = 1)
    
    def forward(self, graphs):
        # Structural Attention forward
        logits = []
        for user_graphs in graphs:
            # We only focus on the ego nodes, so we only take the representation of the ego nodes over different timestamps.
            structural_outputs = torch.stack([self.structural_self_attention(self.tweet_aggr(graph.features), graph.edge_index)[graph.label != -100] for graph in user_graphs], dim = 1)

            length = structural_outputs.size(1)
            temporal_inputs = structural_outputs + self.position_embeddings(self.position_ids[:length])
            # M ∈ RT ×T is a mask matrix with each entry Mi j ∈ {−∞, 0} to enforce the auto-regressive property.
            # To encode the temporal order, we define M as:
            attn_mask = ~(torch.ones(length, length, device = self.position_ids.device).tril().bool())

            # We used the representation of the ego node from the last graph snapshot.
            for temporal_attention_block in self.temporal_self_attention_block:
                temporal_inputs = temporal_attention_block(temporal_inputs, attn_mask = attn_mask)
            most_recent_logit = temporal_inputs[:, 0, :]
            logits.append(most_recent_logit)
        logits = torch.cat(logits, dim = 0)
        cls_logits = self.depression_prediction_head(logits)
        prediction_scores = torch.sigmoid(cls_logits)
        return ModelOutput(
            logits = logits,
            cls_logits = cls_logits,
            prediction_scores = prediction_scores,
        )
