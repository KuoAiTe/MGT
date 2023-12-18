import torch
import torch.nn as nn
import torch.nn.functional as F

import math
BertLayerNorm = torch.nn.LayerNorm

class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(config) for _ in range(config.num_layers)
            ]
        )
    
    def reset_parameters(self):
        for layer in self.layers:
            layer.reset_parameters()
            
    def forward(self, hidden_states, attn_mask = None, src_key_padding_mask = None):
        attention_probs = []
        for i, layer_module in enumerate(self.layers):
            hidden_states, attention_probs_layer = layer_module(hidden_states, attn_mask, src_key_padding_mask)
            attention_probs.append(attention_probs_layer)

        return hidden_states, attention_probs
        
class Feedforward(nn.Module):
    def __init__(self, config):
        super(Feedforward, self).__init__()
        self.lin_1 = nn.Linear(config.hidden_size, config.hidden_size)
        self.lin_2 = nn.Linear(config.hidden_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
    def reset_parameters(self):
        self.lin_1.reset_parameters()
        self.lin_2.reset_parameters()
    def forward(self, x):
        x = self.lin_1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.lin_2(x)
        return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, config):
        super(TransformerEncoderLayer, self).__init__()
        self.multihead_attention = nn.MultiheadAttention(config.hidden_size, config.num_attention_heads, batch_first = True)
        #self.query = nn.Linear(config.hidden_size, config.hidden_size)
        #self.key = nn.Linear(config.hidden_size, config.hidden_size)
        #self.value = nn.Linear(config.hidden_size, config.hidden_size)
        #self.multihead_attention = MultiheadSelfAttention(config)
        self.feedforward = Feedforward(config)
        self.layernorm1 = nn.LayerNorm(config.hidden_size, eps=1e-12)
        self.layernorm2 = nn.LayerNorm(config.hidden_size, eps=1e-12)
    def reset_parameters(self):
        self.multihead_attention.reset_parameters()
        self.feedforward.reset_parameters()
        self.layernorm1.reset_parameters()
        self.layernorm2.reset_parameters()
        #self.query.reset_parameters()
        #self.key.reset_parameters()
        #self.value.reset_parameters()
    def forward(self, hidden_states, attn_mask = None, key_padding_mask = None):
        attention_outputs, attention_probs = self.multihead_attention(hidden_states, hidden_states, hidden_states, attn_mask = attn_mask, key_padding_mask = key_padding_mask)
        #attention_outputs, attention_probs = self.multihead_attention(self.query(hidden_states), self.key(hidden_states), self.value(hidden_states), attn_mask = attn_mask, key_padding_mask = key_padding_mask)
        #print("attention_probs", attention_probs)
        # Add & Norm (Residual)
        norm_output = self.layernorm1(hidden_states + attention_outputs)
        
        #feedfoward
        feedfoward_output = self.feedforward(norm_output)
        
        # Add & Norm (Residual)
        block_output = self.layernorm2(norm_output + feedfoward_output)

        return block_output, attention_probs

class MultiheadSelfAttention(nn.Module):
    def __init__(self, config):
        super(MultiheadSelfAttention, self).__init__()
        self.n_heads = config.num_attention_heads
        self.hidden_size = config.hidden_size
        self.attention_reign_per_head = self.hidden_size // self.n_heads

        self.query = nn.Linear(self.hidden_size, self.hidden_size)
        self.key = nn.Linear(self.hidden_size, self.hidden_size)
        self.value = nn.Linear(self.hidden_size, self.hidden_size)
        # ff
        # dropout
        self.attn_dp = nn.Dropout(config.attention_probs_dropout_prob)
        self.layernorm2 = nn.LayerNorm(self.hidden_size, eps=1e-12)
        self.reset_parameters()
    def reset_parameters(self):
        self.query.reset_parameters()
        self.key.reset_parameters()
        self.value.reset_parameters()
        self.layernorm2.reset_parameters()

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.n_heads, self.attention_reign_per_head)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
        
    def forward(self, query, key, value, attn_mask = None, key_padding_mask = None):
        mixed_query_layer = self.query(query)

        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        key_layer = self.transpose_for_scores(self.key(key))
        value_layer = self.transpose_for_scores(self.value(value))
        query_layer = self.transpose_for_scores(mixed_query_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_reign_per_head)
        #print(key_padding_mask.shape, key_padding_mask)


        if key_padding_mask is not None:
            l = query.shape[1]
            #attn_mask = (torch.tril(torch.ones(l, l, device = query.device) == 1)).transpose(0, 1)
            attn_mask = ~key_padding_mask
            attn_mask = attn_mask.float().masked_fill(attn_mask == 0, float('-inf')).masked_fill(attn_mask == 1, float(0.0))
            attn_mask = attn_mask[:, None, None, :]
            #print(attention_scores.shape, attn_mask.shape, attn_mask)
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attn_mask
        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim = -1)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.attn_dp(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.hidden_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        return context_layer, attention_probs.squeeze()


