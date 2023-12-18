import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Any, Union, Callable

class SimpleAttentionLayer(nn.Module):
    def __init__(self, input_dim, activation = F.relu):
        super(SimpleAttentionLayer, self).__init__()
        self.input_dim = input_dim
        self.lin_proj = nn.Linear(input_dim, input_dim)
        self.attenion_weights = nn.Parameter(torch.Tensor(input_dim, 1))
        self.act = activation
        self.softmax = nn.Softmax(dim = 1)
    
    def reset_parameters(self):
        self.lin_proj.reset_parameters()
        nn.init.xavier_uniform_(self.attenion_weights)

    def forward(self, x):
        inputs = x
        x = self.lin_proj(x)
        x = self.act(x)
        
        # Compute the attention weights
        attention_weights = x @ self.attenion_weights
        attention_weights = F.softmax(attention_weights, dim = 1)
        
        weighted_sum = torch.sum(inputs * attention_weights, dim = 1)
        return weighted_sum
    
class TemporalAttentionLayer(nn.Module):
    def __init__(self, d_in: int,
                 dim_feedforward: int,
                 nhead: int,
                 dropout: float = 0.1,
                 activation: Union[str, Callable[[Tensor], Tensor]] = F.relu) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_in, nhead, dropout = dropout, batch_first = True)
        # Implementation of Feedforward model
        self.linear1 = nn.Linear(d_in, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_in)
        self.activation = activation

    def reset_parameters(self):
        self.self_attn._reset_parameters()
        self.linear1.reset_parameters()
        self.linear2.reset_parameters()

    def forward(self, x, attn_mask = None, key_padding_mask = None):
        x = self._sa_block(x, attn_mask = attn_mask, key_padding_mask = key_padding_mask)
        x = self._ff_block(x)
        return x

    # self-attention block
    def _sa_block(self, x: Tensor, attn_mask: Optional[Tensor] = None, key_padding_mask: Optional[Tensor] = None) -> Tensor:
        return self.self_attn(x, x, x, attn_mask=attn_mask, key_padding_mask=key_padding_mask, need_weights=False)[0]

    # feed forward block
    def _ff_block(self, x: Tensor) -> Tensor:
        return self.linear2(self.dropout(self.activation(self.linear1(x))))
    