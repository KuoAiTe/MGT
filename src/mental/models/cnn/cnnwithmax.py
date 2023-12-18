import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn.models import GAT, GCN, GraphSAGE
from .. import BaseModel, ModelOutput
from ...utils.dataprocessing import prepare_text_inputs, prepare_batch_text_inputs

from torch_geometric.utils import add_self_loops,  coalesce, to_undirected
import copy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

from torch import Tensor
from torch.nn import Linear, ModuleList
from tqdm import tqdm
from torch_geometric.utils import scatter

from torch_geometric.loader import NeighborLoader
from torch_geometric.nn.conv import (
    GATConv,
    GATv2Conv,
    GCNConv,
    MessagePassing,
    SAGEConv,
)

from torch_geometric.nn.models.mlp import MLP
from torch_geometric.typing import Adj


class CNNWithMax(BaseModel):
    def __init__(self, args, data_info):
        super().__init__()
        self.conv1d = nn.Conv1d(768, 64, kernel_size = 3, stride = 1)
        self.maxpool = nn.AdaptiveMaxPool1d(output_size = 1)
        self.activation = nn.ReLU()
        self.depression_prediction_head = nn.Linear(64, 1)
        self.allowed_inputs = ['text_data']
        self.reset_parameters()

    def reset_parameters(self):
        self.conv1d.reset_parameters()
        self.depression_prediction_head.reset_parameters()

    def forward(self, text_data) -> Tensor:
        inputs = [user['user_embeddings'] for user in text_data]
        inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first = True).permute(0, 2, 1)
        if inputs.shape[-1] < 3:
            inputs = F.pad(inputs, (1, 1), "constant", 0)
        conv_out = self.conv1d(inputs)
        conv_out = self.activation(conv_out)
        logits = self.maxpool(conv_out).squeeze(-1)
        cls_logits = self.depression_prediction_head(logits)
        prediction_scores = torch.sigmoid(cls_logits)

        return ModelOutput(
            logits = logits,
            cls_logits = cls_logits,
            prediction_scores = prediction_scores,
        )
    

    def prepare_inputs(self, inputs):
        return prepare_text_inputs(inputs)
    
    def prepare_batch_inputs(self, inputs):
        return prepare_batch_text_inputs(inputs)

class CNNWithMaxPlus(CNNWithMax):
    def forward(self, text_data) -> Tensor:
        inputs = [torch.cat([user['user_embeddings'], user['friend_embeddings']]) for user in text_data]
        inputs = torch.nn.utils.rnn.pad_sequence(inputs, batch_first = True).permute(0, 2, 1)
        if inputs.shape[-1] < 3:
            inputs = F.pad(inputs, (1, 1), "constant", 0)
        conv_out = self.conv1d(inputs)
        conv_out = self.activation(conv_out)
        logits = self.maxpool(conv_out).squeeze(-1)
        cls_logits = self.depression_prediction_head(logits)
        prediction_scores = torch.sigmoid(cls_logits)

        return ModelOutput(
            logits = logits,
            cls_logits = cls_logits,
            prediction_scores = prediction_scores,
        )
    