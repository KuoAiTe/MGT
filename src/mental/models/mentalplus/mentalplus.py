
from multiprocessing import pool
import torch
import numpy as np
import math
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from torch_geometric.nn.models.mlp import MLP
from .. import BaseModel, BaseConModel, ModelOutput
from ...utils.dataprocessing import prepare_dynamic_hyper_graphs, prepare_batch_dynamic_hyper_graphs
from .layers import TransformerEncoder
from .heterognn import HomoGNN, HeteroGNN, HeteroDynamicGNN
from .attnconv import TorchAttentionBlock
from torch_geometric.utils import scatter, to_dense_batch




class ContentAttention(nn.Module):
    def __init__(self, input_size, hidden_channel, temporal_args, dropout = 0.1):
        super(ContentAttention, self).__init__()
        factor = 1
        output_size = hidden_channel * factor
        self.downsample = nn.Linear(input_size, output_size)
        self.linear_projection = nn.Linear(output_size, output_size)
        self.attention_weights = nn.Parameter(torch.Tensor(output_size, 1))
        self.layer_norm = nn.LayerNorm(output_size, eps=1e-12)
        self.dropout = nn.Dropout(p=dropout)
        self.activation = nn.ReLU()

        self.transformer_encoder = TransformerEncoder(temporal_args)
        

        self.pooling_token = nn.Parameter(torch.Tensor(1, 1, output_size))
        self.multihead_attention = nn.MultiheadAttention(output_size, 1, batch_first = True)
        self.ffn = nn.Sequential(
            nn.Linear(output_size, output_size),
            nn.BatchNorm1d(output_size, eps=1e-12),
            nn.ReLU(inplace=True),
            nn.Linear(output_size, output_size),
            nn.BatchNorm1d(output_size, eps=1e-12),
        )
        self.reset_parameters()

    def reset_parameters(self):
        self.downsample.reset_parameters()
        self.linear_projection.reset_parameters()
        self.layer_norm.reset_parameters()
        nn.init.xavier_uniform_(self.attention_weights)
        nn.init.xavier_uniform_(self.pooling_token)
        for module in self.ffn:
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()


    def forward(self, inputs, augment = False, key_padding_mask = None):
        inputs = self.downsample(inputs)
        inputs = self.layer_norm(inputs)
        inputs = self.dropout(inputs)
        """
        batch_size = inputs.shape[0]
        cls_tokens = self.pooling_token.repeat(batch_size, 1, 1)
        output = torch.cat([cls_tokens, inputs], dim = 1)
        key_padding_mask = torch.sum(output, dim = -1) == 0

        l = output.shape[1]
        output, attention_probs = self.transformer_encoder(output, src_key_padding_mask = key_padding_mask)
        output = output[:, 0]

        """
        lin_proj = self.linear_projection(inputs)
        lin_proj = self.activation(lin_proj)
        attention_scores = lin_proj @ self.attention_weights
        if key_padding_mask is not None:
            attention_scores[key_padding_mask] = attention_scores[key_padding_mask] - 10000
        attention_probs = nn.functional.softmax(attention_scores, dim=1)
        #if augment:
        #    #pass
        #    attention_probs = self.dropout(attention_probs)
        output = torch.einsum('bij, bik -> bj', inputs, attention_probs)
        output = self.ffn(output)
        #output = torch.sum(inputs * attention_probs, dim = 1)
        return output, attention_probs#attention_weights
    

class ContrastEgo_Base(BaseModel):
    def __init__(self, args, dataset_info):
        super(ContrastEgo_Base, self).__init__()
        self.args = args
        self.config = args.mental_net
        config = self.config
        self.gnn_out_dim = config.heterognn.gnn_hidden_channel
        self.num_time_steps = 8

        self.content_attention = ContentAttention(768, config.heterognn.gnn_hidden_channel, args.temporal_attention_config)
        self.gnn = HeteroDynamicGNN(config.heterognn)
        self.position_embeddings = nn.Embedding(self.num_time_steps, self.gnn_out_dim)
        #self.heteroGNN = HeteroGNN(config.heterognn)
        #self.gnn = HomoGNN(config.heterognn)

        self.depression_prediction_head =  MLP([self.gnn_out_dim, self.gnn_out_dim, 1])
        
        #nn.Linear(self.mentalnet_out_dim, 1)
        self.allowed_inputs = ['hyper_graph']

        ########################################

    def reset_parameters(self):
        self.gnn.reset_parameters()
        self.position_embeddings.reset_parameters()
        self.depression_prediction_head.reset_parameters()
        self.content_attention.reset_parameters()
        
    def forward(self, graph):
        x_dict, edge_index_dict, edge_weight_dict, group, label = graph.x_dict, graph.edge_index_dict, graph.edge_weight_dict, graph.group, graph.label
        temporal_attention_weights = None
    
        f, tweet_attention_weights = self.content_attention(x_dict['user'])
        f, graph_attention_weights = self.gnn(f, edge_index_dict, edge_weight_dict, graph)
        user_nodes = (label != -100)
        node_over_time_readout = scatter(f[user_nodes], group[user_nodes], dim = 0, reduce = 'mean')
        logits = node_over_time_readout
        supcon_logits = f
        cls_logits = self.depression_prediction_head(logits)
        prediction_scores = torch.sigmoid(cls_logits)
        return ModelOutput(
            logits = logits,
            cls_logits = cls_logits,
            supcon_logits = supcon_logits,
            prediction_scores = prediction_scores,
            tweet_attention_weights = tweet_attention_weights,
            graph_attention_weights = graph_attention_weights,
            temporal_attention_weights = temporal_attention_weights,
        )
        
    def prepare_inputs(self, inputs):
        return prepare_dynamic_hyper_graphs(inputs)
    
    def prepare_batch_inputs(self, inputs):
        return prepare_batch_dynamic_hyper_graphs(inputs, self.training)



import time



class MentalPlus_Base(BaseModel):
    def __init__(self, args, data_info):
        super(MentalPlus_Base, self).__init__()
        self.args = args
        self.config = args.mental_net
        config = self.config
        self.gnn_out_dim = config.heterognn.gnn_hidden_channel
        self.num_time_steps = 8
        self.position_embeddings = nn.Embedding(self.num_time_steps + 2, self.gnn_out_dim)

        #encoder_layer = nn.TransformerEncoderLayer(d_model = config.heterognn.gnn_hidden_channel, nhead = 2, dropout = 0.2, dim_feedforward = config.heterognn.gnn_hidden_channel, batch_first = True)
        #self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers = 1, enable_nested_tensor = True, mask_check = True)
        self.transformer_encoder = TransformerEncoder(args.temporal_attention_config)
        self.layer_norm = nn.LayerNorm(self.gnn_out_dim, eps=1e-12)
        self.batch_norm = nn.BatchNorm1d(self.gnn_out_dim, eps=1e-12)
        self.gnn = HeteroGNN(config.heterognn)
        self.content_attention = ContentAttention(768, config.heterognn.gnn_hidden_channel, args.temporal_attention_config)

        self.mask_token = nn.Parameter(torch.rand(1, 1, self.gnn_out_dim))
        self.cls_token = nn.Parameter(torch.rand(1, 1, self.gnn_out_dim))
        self.unk_token = nn.Parameter(torch.rand(1, 768))
        self.graph_pooling_token = nn.Parameter(torch.rand(1, self.gnn_out_dim))
        self.projection = nn.Linear(self.gnn_out_dim, self.gnn_out_dim, bias = False)
        self.register_buffer('position_ids', torch.arange(self.num_time_steps + 2))

        self.maxpool = nn.AdaptiveMaxPool1d(output_size = 1)
        self.depression_prediction_head = MLP([self.gnn_out_dim, self.gnn_out_dim, 1])#nn.Linear(self.gnn_out_dim, 1)
        self.use_layernorm = True
        self.use_gnn = True
        self.use_transformer = True
        self.use_content_attention = True
        self.interaction_cutout = True
        self.timeframe_cutout = True
        

    def reset_parameters(self):
        self.gnn.reset_parameters()
        self.position_embeddings.reset_parameters()
        self.layer_norm.reset_parameters()
        self.batch_norm.reset_parameters()
        self.projection.reset_parameters()
        self.content_attention.reset_parameters()
        nn.init.xavier_uniform_(self.mask_token)
        nn.init.xavier_uniform_(self.cls_token)
        nn.init.xavier_uniform_(self.unk_token)
        nn.init.xavier_uniform_(self.graph_pooling_token)
        
        for p in self.transformer_encoder.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        self.depression_prediction_head.reset_parameters()
        print("self.use_gnn", self.use_gnn, '\n', 'self.use_transformer ', self.use_transformer, 'self.use_content_attention', self.use_content_attention, 'self.interaction_cutout', self.interaction_cutout)

    @torch.no_grad()
    def __get_attention_scores__(self, **kwargs):
        return self.forward(**kwargs)
    
    def forward(self, graph):
        x_dict, edge_index_dict, edge_weight_dict, group, label = graph.x_dict, graph.edge_index_dict, graph.edge_weight_dict, graph.group, graph.label
        is_contrastive_learning = self.training and hasattr(self, 'contrastive')
        groups = group.unique()
        user_nodes = (label != -100)
        batch_size = len(groups)
        f = x_dict['user']
        tweet_attention_weights = graph_attention_weights = temporal_attention_weights = None
        if self.use_content_attention:
            key_padding_mask = None
            if is_contrastive_learning:
               new_inputs = f.clone()
               probability_matrix = torch.full(f.shape[:2], 0.2, device = f.device)
               key_padding_mask = torch.bernoulli(probability_matrix).bool()
               #at least one is being attended
               new_inputs[key_padding_mask] = 0
               f = new_inputs

            f, tweet_attention_weights = self.content_attention(f, is_contrastive_learning, key_padding_mask)
        else:
            attention_mask = torch.full(f.shape[:2], True, device = f.device)
            """
            if True and self.training and hasattr(self, 'contrastive'):
                
                new_inputs = f.clone()
                probability_matrix = torch.full(f.shape[:2], 0.1, device = f.device)
                key_padding_mask = torch.bernoulli(probability_matrix).bool()
                #at least one is being attended
                new_inputs[key_padding_mask] = self.unk_token
                f = new_inputs

                attention_mask = ~key_padding_mask
            """
            input_mask_expanded = attention_mask.unsqueeze(-1).expand(f.size()).float()
            sum_mask = torch.clamp(input_mask_expanded.sum(1), min = 1)
            f = torch.sum(f * input_mask_expanded, 1) / sum_mask
            #f = torch.mean(f, dim = 1)
            f = self.content_attention.downsample(f)

        if self.use_gnn:
            f, graph_attention_weights = self.gnn(
                f,
                edge_index_dict,
                edge_weight_dict, 
                graph = graph, 
                interaction_cutout = is_contrastive_learning and self.interaction_cutout
            )
            
        if not self.use_layernorm:
            position_embeddings = self.position_embeddings(graph.period_id)
            f = f + position_embeddings

            f = self.batch_norm(f)
        #f = f + position_embeddings
        #f = self.projection(f)
        #f = self.mlp_for_trans(f)
        
        
        output = []
        for i, group_id in enumerate(groups):
            index = (group == group_id) & user_nodes
            if is_contrastive_learning and self.timeframe_cutout:
                probability_matrix = torch.full(index.shape, 0.2, device = f.device)
                masked_indices = torch.bernoulli(probability_matrix).bool()
                index = index & ~masked_indices
            # Augmentation
            user_embeddings = f[index]
            output.append(user_embeddings)
        
        output = torch.nn.utils.rnn.pad_sequence(output, batch_first = True, padding_value = 0.0)
        if self.use_transformer:
            attn_mask = None
            if self.use_layernorm:
                cls_tokens = self.cls_token.repeat(batch_size, 1, 1)
                siam_tokens = self.graph_pooling_token.repeat(batch_size, 1, 1)
                output = torch.cat([cls_tokens, siam_tokens, output], dim = 1)
                src_key_padding_mask = torch.sum(output, dim = -1) == 0
                

                position_embeddings = self.position_embeddings(self.position_ids[:output.shape[1]])
                output = output + position_embeddings
                output = self.layer_norm(output)
                #l = output.shape[1]
                l = output.shape[1]

                #attn_mask = (torch.tril(torch.ones(l, l, device = output.device) == 1)).transpose(0, 1)
                #attn_mask = attn_mask.float().masked_fill(attn_mask == 0, float('-inf')).masked_fill(attn_mask == 1, float(0.0))
                #attn_mask[0, :2] = 0.0
                #attn_mask[:, :1] = 0.0
                x, temporal_attention_weights = self.transformer_encoder(output, attn_mask = attn_mask, src_key_padding_mask = src_key_padding_mask)
                #print(temporal_attention_weights)
                user_node_logits = x[:, 0]
            else:
                cls_tokens = self.cls_token.repeat(batch_size, 1, 1)
                siam_tokens = self.graph_pooling_token.repeat(batch_size, 1, 1)
                output = torch.cat([cls_tokens, siam_tokens, output], dim = 1)
                src_key_padding_mask = torch.sum(output, dim = -1) == 0
                l = output.shape[1]
                attn_mask = (torch.tril(torch.ones(l, l, device = output.device) == 1)).transpose(0, 1)
                attn_mask = attn_mask.float().masked_fill(attn_mask == 0, float('-inf')).masked_fill(attn_mask == 1, float(0.0))
                attn_mask[0, :2] = 0.0
                attn_mask[:, :1] = 0.0
                x, temporal_attention_weights = self.transformer_encoder(output, attn_mask = attn_mask, src_key_padding_mask = src_key_padding_mask)
                user_node_logits = x[:, 0]
            #print("temporal_attention_weights", temporal_attention_weights)
            #x = self.transformer_encoder(output, src_key_padding_mask = src_key_padding_mask)
            #user_node_logits = x[:, 0]

            if self.use_layernorm:
                #supcon_logits = x.reshape(-1, x.shape[-1])
                input_mask_expanded = (~src_key_padding_mask).unsqueeze(-1).expand(x.size()).float()
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min = 1)
                supcon_logits = torch.sum(x * input_mask_expanded, 1) / sum_mask
            else:
                supcon_logits = x[:, 1]
        else:
            #src_key_padding_mask = torch.sum(output, dim = -1) == 0
            #input_mask_expanded = (~src_key_padding_mask).unsqueeze(-1).expand(output.size()).float()
            #sum_mask = torch.clamp(input_mask_expanded.sum(1), min = 1)
            user_node_logits = torch.max(output, dim = 1)[0]
            supcon_logits = user_node_logits
            # ignore cls_token, mean pooling

        cls_logits = self.depression_prediction_head(user_node_logits)
        #if self.use_transformer:
        #    supcon_logits =  x[:, 1]
        #else:


        prediction_scores = torch.sigmoid(cls_logits)

        return ModelOutput(
            logits = user_node_logits,
            cls_logits = cls_logits,
            supcon_logits = supcon_logits,
            prediction_scores = prediction_scores,
            tweet_attention_weights = tweet_attention_weights,
            graph_attention_weights = graph_attention_weights,
            temporal_attention_weights = temporal_attention_weights,
        )
        

    def prepare_inputs(self, inputs):
        return prepare_dynamic_hyper_graphs(inputs)

    def prepare_batch_inputs(self, inputs):
        return prepare_batch_dynamic_hyper_graphs(inputs, self.training)

class SimSiam_Base(nn.Module):
    def __init__(self, encoder, out_dim):
        super().__init__()
        self.encoder = encoder
        #self.encoder.depression_prediction_head = MLP([out_dim, out_dim, 1], norm = None)
        self.projector = nn.Sequential(
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim, out_dim),
            nn.BatchNorm1d(out_dim)
        )
        self.predictor = nn.Sequential(
            nn.Linear(out_dim, out_dim // 2),
            nn.BatchNorm1d(out_dim // 2),
            nn.ReLU(inplace=True),
            nn.Linear(out_dim // 2, out_dim)
        )
        self.criterion = nn.CosineSimilarity(dim = 1)
        self.encoder.contrastive = True

    def forward(self, **kwargs):
        return self.encoder(**kwargs)
    
    def reset_parameters(self):
        self.encoder.reset_parameters()
        for module in self.projector:
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
        for module in self.predictor:
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()
    
    def compute_loss(self, **kwargs):
        # twin backbones
        output = self.encoder.compute_loss(**kwargs)
        output_2 = self.encoder.compute_loss(**kwargs)
        siamese_loss = 0
        if self.training:
            if output.supcon_logits == None:
                z1 = self.projector(output.logits)
                p1 = self.predictor(z1)

                z2 = self.projector(output_2.logits)
                p2 = self.predictor(z2)
                siamese_loss = -(self.criterion(p1, z2.detach()).mean() + self.criterion(p2, z1.detach()).mean()) / 2

            else:
                z1 = self.projector(output.supcon_logits)
                p1 = self.predictor(z1)
                z2 = self.projector(output_2.supcon_logits)
                p2 = self.predictor(z2)

                siamese_loss = -(self.criterion(p1, z2.detach()).mean() + self.criterion(p2, z1.detach()).mean()) / 2

        # Don't waste output_2
        #print(f'{self.training} loss1 : {output.loss}, loss2: {output_2.loss}, siamese_loss: {siamese_loss}')
        output.loss = (output.loss + output_2.loss) / 2 + 1.0 * siamese_loss
        return output
    
    @torch.no_grad()
    def __get_attention_scores__(self, **kwargs):
        return self.encoder.__get_attention_scores__(**kwargs)
    
    @torch.no_grad()
    def predict(self, **kwargs):
        return self.compute_loss(**kwargs)
    
class ContrastEgo(BaseModel):
    def __init__(self, args, dataset_info):
        super(ContrastEgo, self).__init__()
        encoder = ContrastEgo_Base(args, dataset_info)
        self.model = SimSiam_Base(encoder, 64)

    def __get_attention_scores__(self, **kwargs):
        return self.model.__get_attention_scores__(**kwargs)
    
    def reset_parameters(self):
        self.model.reset_parameters()
    
    def compute_loss(self, **kwargs):
        return self.model.compute_loss(**kwargs)
    
    @torch.no_grad()
    def predict(self, **kwargs):
        return self.model.predict(**kwargs)
     
    def forward(self, **kwargs):
        return self.model(**kwargs)
        
    def prepare_inputs(self, inputs):
        return prepare_dynamic_hyper_graphs(inputs)
    
    def prepare_batch_inputs(self, inputs):
        return prepare_batch_dynamic_hyper_graphs(inputs, self.training)
    

class MentalPlus(ContrastEgo):
    def __init__(self, args, dataset_info):
        super(MentalPlus, self).__init__(args, dataset_info)
        self.backbone = MentalPlus_Base(args, dataset_info)
        self.model = SimSiam_Base(self.backbone, 64)

class MentalPlus_BatchNorm(ContrastEgo):
    def __init__(self, args, dataset_info):
        super(MentalPlus_BatchNorm, self).__init__(args, dataset_info)
        self.backbone = MentalPlus_Base(args, dataset_info)
        self.model = SimSiam_Base(self.backbone, 64)
        self.backbone.use_layernorm = False


class MentalPlus_SUP(BaseConModel):
    def __init__(self, args, dataset_info):
        backbone = MentalPlus_Base(args, dataset_info)
        super(MentalPlus_SUP, self).__init__(backbone)

class MentalPlus_NO_GNN(MentalPlus):
    def __init__(self, args, dataset_info):
        super(MentalPlus_NO_GNN, self).__init__(args, dataset_info)
        self.backbone.use_gnn = False



class MentalPlus_HOMO(MentalPlus):
    def __init__(self, args, dataset_info):
        super(MentalPlus_HOMO, self).__init__(args, dataset_info)
        self.backbone.gnn = HomoGNN(args.mental_net.heterognn)
        
class MentalPlus_NO_TIMEFRAME_CUTOUT(MentalPlus):
    def __init__(self, args, dataset_info):
        super(MentalPlus_NO_TIMEFRAME_CUTOUT, self).__init__(args, dataset_info)
        self.backbone.timeframe_cutout = False

class MentalPlus_NO_INTERACTION_CUTOUT(MentalPlus):
    def __init__(self, args, dataset_info):
        super(MentalPlus_NO_INTERACTION_CUTOUT, self).__init__(args, dataset_info)
        self.backbone.interaction_cutout = False

class MentalPlus_NO_CONTENT_ATTENTION(MentalPlus):
    def __init__(self, args, dataset_info):
        super(MentalPlus_NO_CONTENT_ATTENTION, self).__init__(args, dataset_info)
        self.backbone.use_content_attention = False
        
class MentalPlus_Without_Transformer(MentalPlus):
    def __init__(self, args, dataset_info):
        super(MentalPlus_Without_Transformer, self).__init__(args, dataset_info)
        self.backbone.use_transformer = False
        self.backbone.depression_prediction_head = MLP([self.backbone.gnn_out_dim , self.backbone.gnn_out_dim, 1], norm = None)
        