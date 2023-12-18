import torch
import torch.nn as nn
from .. import BaseModel, ModelOutput
from ...utils.dataprocessing import prepare_text_inputs, prepare_batch_text_inputs

import math
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
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

# We omit the TextGCN part as described in the paper to have a fair comparison.
class UsmanBiLSTM(BaseModel):
    def __init__(self, args, data_info):
        super().__init__()
        input_size = args.input_size
        config = args.rnn_config
        hidden_size = config.hidden_size
        self.allowed_inputs = ['text_data']
        self.lstm = nn.LSTM(input_size, hidden_size, bidirectional = True)
        self.tweet_history_weight = nn.Parameter(torch.Tensor(2 * hidden_size, 2 * hidden_size))
        self.tweet_history_bias = nn.Parameter(torch.Tensor(1, 2 * hidden_size))
        self.attenion_weights = nn.Parameter(torch.Tensor(2 * hidden_size, 1))
        self.depression_prediction_head = nn.Linear(2 * hidden_size, 1)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.lstm.reset_parameters()
        self.depression_prediction_head.reset_parameters()
        nn.init.xavier_uniform_(self.tweet_history_weight)
        nn.init.xavier_uniform_(self.tweet_history_bias)
        nn.init.xavier_uniform_(self.attenion_weights)

    def forward(self, text_data) -> Tensor:
        logits = []
        for user in text_data:
            b = user['user_embeddings']
            #The BiLSTM transforms historical post encodings [b1, b2, b3, ...] into contextual representations [h1, h2, h3]
            h, (_, _) = self.lstm(b)
            lin_proj = h @ self.tweet_history_weight + self.tweet_history_bias
            u_i = torch.tanh(lin_proj)

            # Compute the attention weights
            attention_weights = u_i @ self.attenion_weights
            
            attention_weights = F.softmax(attention_weights, dim = 0)
            
            out = torch.sum(h * attention_weights, dim = 0)
            logits.append(out)
        logits = torch.stack(logits)
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


class UsmanBiLSTMPlus(UsmanBiLSTM):
    def __init__(self, args, data_info):
        super(UsmanBiLSTMPlus, self).__init__(args, data_info)

    def forward(self, text_data) -> Tensor:
        logits = []
        for user in text_data:
            b = torch.cat([user['user_embeddings'], user['friend_embeddings']])
            #The BiLSTM transforms historical post encodings [b1, b2, b3, ...] into contextual representations [h1, h2, h3]
            h, (_, _) = self.lstm(b)
            lin_proj = h @ self.tweet_history_weight + self.tweet_history_bias
            u_i = torch.tanh(lin_proj)

            # Compute the attention weights
            attention_weights = u_i @ self.attenion_weights
            
            attention_weights = F.softmax(attention_weights, dim = 0)
            out = torch.sum(h * attention_weights, dim = 0)
            logits.append(out)
        logits = torch.stack(logits)
        cls_logits = self.depression_prediction_head(logits)
        prediction_scores = torch.sigmoid(cls_logits)

        return ModelOutput(
            logits = logits,
            cls_logits = cls_logits,
            prediction_scores = prediction_scores,
        )