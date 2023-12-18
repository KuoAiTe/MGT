import numpy as np
import math

import torch
import torch.nn as nn
import torch.nn.functional as F 
from torch import Tensor

from torch_geometric.nn.inits import glorot
from torch_geometric.utils import softmax
from torch_scatter import scatter

import mental

##########################################################################
##########################################################################
##########################################################################

def get_common_neighbors(active_nodes, edge_encode_list, PPR, edge_dist_encode, feats, FLAGS, deterministic_neighbor_sampling=None):

    if deterministic_neighbor_sampling == None:
        deterministic_neighbor_sampling = FLAGS.deterministic_neighbor_sampling

    all_nodes, all_nodes_to_new_index, target_node_size_1, context_nodes_size_1 = sample_joint_neighbors(active_nodes, PPR, 
                                                                                                         deterministic_neighbor_sampling, 
                                                                                                         FLAGS.two_steam_model,
                                                                                                         FLAGS.max_neighbors)
    all_node_size = target_node_size_1 + context_nodes_size_1
    time_size = len(edge_encode_list)

    edge_encode_np_1 = np.zeros((all_node_size, all_node_size, time_size))

    for t in range(time_size):
        edge_encode_np_1[:, :, t] = 2*t
        edge_encode_tmp = edge_encode_list[t]
        edge_encode_tmp = edge_encode_tmp[all_nodes, :][:, all_nodes].tocoo()
        edge_encode_np_1[edge_encode_tmp.row, edge_encode_tmp.col, t] = 2*t+1

    edge_dist_encode_np_1 = np.ones((all_node_size, all_node_size)) * (FLAGS.max_dist+1)
    edge_dist_encode_tmp = edge_dist_encode[all_nodes, :][:, all_nodes].tocoo()
    edge_dist_encode_np_1[edge_dist_encode_tmp.row, edge_dist_encode_tmp.col] = edge_dist_encode_tmp.data

    node_feats_np_1 = feats[all_nodes].toarray()

    return node_feats_np_1, edge_encode_np_1, edge_dist_encode_np_1, target_node_size_1, context_nodes_size_1, all_nodes_to_new_index

def sample_joint_neighbors(active_nodes, PPR, deterministic=True, two_stream_structure=True, max_neighbors=-1):
    num_active_nodes = len(active_nodes)
    
    if max_neighbors >= 0:
        num_neighbors = min(num_neighbors, max_neighbors)
    
    if two_stream_structure:
        num_active_nodes = len(active_nodes)
        num_neighbors = PPR.shape[0] - len(active_nodes)
        
        # no bigger than num_neighbors
        if max_neighbors >= 0:
            num_neighbors = min(num_active_nodes, max_neighbors)
        else:
            num_neighbors = num_active_nodes

        
        sampling_prbs = np.array(np.sum(PPR[active_nodes, :], axis=0)).reshape(-1)
        sampling_prbs[active_nodes] = 0.
        sampling_prbs = sampling_prbs/np.sum(sampling_prbs)
        num_neighbors = min(num_neighbors, np.sum(sampling_prbs>0))
        if deterministic:
            shared_neighbors = np.argsort(sampling_prbs)[-num_neighbors:]
        else:
            shared_neighbors = np.random.choice(PPR.shape[0], p=sampling_prbs, size=num_neighbors, replace=False)
            
        num_shared_neighbors = len(shared_neighbors)
        if num_shared_neighbors < num_active_nodes:
            select = np.random.permutation(num_active_nodes)[:num_active_nodes-num_shared_neighbors]
            shared_neighbors = np.concatenate([shared_neighbors, active_nodes[select]])
        num_shared_neighbors = len(shared_neighbors)
        
        assert num_shared_neighbors == num_active_nodes
        
        all_nodes = np.concatenate([active_nodes, shared_neighbors])
        target_node_size = len(active_nodes)
        context_nodes_size = len(shared_neighbors)
    else:
        all_nodes = active_nodes
        target_node_size = len(active_nodes)
        context_nodes_size = 0
        
    new_index = np.arange(len(all_nodes))
    all_nodes_to_new_index = dict(zip(all_nodes, new_index))
    return all_nodes, all_nodes_to_new_index, target_node_size, context_nodes_size

def generate_temporal_edges(edge_encode_np, target_node_size, neg_sample_size):
    time_steps = edge_encode_np.shape[-1]

    # edge_encode_np [num_nodes, num_nodes, time_steps]
    edge_encode_np = edge_encode_np[:target_node_size, :target_node_size, :]
    sampled_temporal_edges = []
    for t in range(time_steps):
        # positive
        pos_row, pos_col = np.where(edge_encode_np[:, :, t]%2==1)
        pos_edges = np.stack([pos_row, pos_col])
        num_pos_samples = len(pos_row)

        # negative
        neg_row, neg_col = np.where(edge_encode_np[:, :, t]%2==0)
        num_neg_samples = len(neg_row)
        select = np.random.permutation(num_neg_samples)[:neg_sample_size*num_pos_samples]
        neg_row, neg_col = neg_row[select], neg_col[select]
        neg_edges = np.stack([neg_row, neg_col])

        sampled_temporal_edges.append([pos_edges, neg_edges])
    
    return sampled_temporal_edges

def pad_more_nodes(cur_ind, num_all_nodes, expect_size=800):

    if len(cur_ind) < expect_size:
        candidate = np.setdiff1d(np.arange(num_all_nodes), cur_ind)
        padding = np.random.permutation(candidate)[:expect_size - len(cur_ind)]
        cur_ind = np.concatenate([cur_ind, padding])
        
    return cur_ind
    
class FullyConnectedGT_UGformerV2(nn.Module):
    def __init__(
        self, args, dataset_info,
        num_features = 768,        ## original feature vector dim: D
        num_heads = 4,           ## number of attention heads for Transformer: H_s
        num_hids = 192,            ## number of hidden units for Transformer: also as output embedding for Transformer: F_s
        num_layers = 4,
        attn_drop = 0.1,           ## dropout % for attn layer
        feat_drop = 0.1,
        edge_encode_num = 2,
        edge_dist_encode_num = 5 + 2,
        window_size = 5, 
        use_unsupervised_loss = True,
        neighbor_sampling_size = 0.5,
    ):
        super(FullyConnectedGT_UGformerV2, self).__init__()
        self.num_features   = num_features
        self.num_heads      = num_heads
        self.num_hids       = num_hids
        self.num_layers     = num_layers
        
        self.attn_drop      = attn_drop
        self.feat_drop      = feat_drop
        
        self.edge_encode_num      = edge_encode_num
        self.edge_dist_encode_num = edge_dist_encode_num
        self.window_size          = window_size
        
        self.use_unsupervised_loss = use_unsupervised_loss
        self.neighbor_sampling_size = neighbor_sampling_size
                
        self.edge_embedding      = nn.Embedding(self.edge_encode_num*self.window_size, self.num_hids, max_norm=True)
        self.edge_dist_embedding = nn.Embedding(self.edge_dist_encode_num, self.num_hids, max_norm=True)
        
        self.temporal_edge_encode_weights = torch.nn.Parameter(torch.ones(window_size))
        
        self.transformer_layer_list   = nn.ModuleList()
        for ell in range(self.num_layers):
            block = EncoderLayer(self.num_hids, self.feat_drop, self.attn_drop, self.num_heads)
            self.transformer_layer_list.append(block)
            
        self.node_feats_fc           = nn.Linear(self.num_features, self.num_hids)
        self.edge_encode_fc          = nn.Linear(self.num_hids, self.num_heads)
        self.edge_dist_encode_num_fc = nn.Linear(self.num_hids, self.num_heads)
        
        if self.use_unsupervised_loss:
            self.decoder_heads = nn.ModuleList()
            for t in range(self.window_size):
                self.decoder_heads.append(nn.Linear(self.num_hids, self.num_hids))

            self.BYOL_decoder = nn.Linear(self.num_hids, self.num_hids)
        else:
            pass
            # self.decoder = nn.Linear(self.num_hids, self.num_hids)
                
    def prepare_data(self, data):
        return mental.utils.utilities.prepare_dynamic_homo_graph(data)
    def get_encodings(self, edge_encodes, edge_dist_encodes, target_node_size, context_node_size,
                      tgt2cxt_sparse_row, tgt2cxt_sparse_col,
                      cxt2tgt_sparse_row, cxt2tgt_sparse_col):
        
        ### attn_bias_tgt2cxt
        if context_node_size > 0:
            context_edge_encodes = edge_encodes[target_node_size:, :target_node_size, :] # [context_size, target_size, :]
            attn_bias_1_context = self.edge_embedding(context_edge_encodes[tgt2cxt_sparse_row, tgt2cxt_sparse_col, :])

            temporal_weights = F.softmax(self.temporal_edge_encode_weights[:attn_bias_1_context.size(-2)], dim=0).view(1, -1, 1)
            attn_bias_1_context = torch.sum(attn_bias_1_context*temporal_weights, axis=-2)
            attn_bias_1_context = self.edge_encode_fc(attn_bias_1_context)

            context_edge_dist_encodes = edge_dist_encodes[target_node_size:, :target_node_size] # [context_size, target_size,, :]
            attn_bias_2_context = self.edge_dist_embedding(context_edge_dist_encodes[tgt2cxt_sparse_row, tgt2cxt_sparse_col])
            attn_bias_2_context = self.edge_dist_encode_num_fc(attn_bias_2_context)

            attn_bias_tgt2cxt = attn_bias_1_context + attn_bias_2_context

            ### attn_bias_ctx2tgt
            target_edge_encodes = edge_encodes[:target_node_size, target_node_size:, :] # [target_size, context_size, :]
            
            attn_bias_1_target = self.edge_embedding(target_edge_encodes[cxt2tgt_sparse_row, cxt2tgt_sparse_col, :])
            temporal_weights = F.softmax(self.temporal_edge_encode_weights[:attn_bias_1_target.size(-2)], dim=0).view(1, -1, 1)
            attn_bias_1_target = torch.sum(attn_bias_1_target*temporal_weights, axis=-2)
            attn_bias_1_target = self.edge_encode_fc(attn_bias_1_target)

            target_edge_dist_encodes = edge_dist_encodes[:target_node_size:, target_node_size:]
            attn_bias_2_target = self.edge_dist_embedding(target_edge_dist_encodes[cxt2tgt_sparse_row, cxt2tgt_sparse_col])
            attn_bias_2_target = self.edge_dist_encode_num_fc(attn_bias_2_target)
            attn_bias_ctx2tgt = attn_bias_1_target + attn_bias_2_target
            
        else:
            attn_bias_tgt2cxt = None

            target_edge_encodes = edge_encodes[:target_node_size, :target_node_size, :] # [target_size, target_size, :]
            
            
            attn_bias_1_target = self.edge_embedding(target_edge_encodes[cxt2tgt_sparse_row, cxt2tgt_sparse_col, :])
            temporal_weights = F.softmax(self.temporal_edge_encode_weights[:attn_bias_1_target.size(-2)], dim=0).view(1, -1, 1)
            attn_bias_1_target = torch.sum(attn_bias_1_target*temporal_weights, axis=-2)
            
            attn_bias_1_target = self.edge_encode_fc(attn_bias_1_target)

            target_edge_dist_encodes = edge_dist_encodes[:target_node_size:, :target_node_size]
            attn_bias_2_target = self.edge_dist_embedding(target_edge_dist_encodes[cxt2tgt_sparse_row, cxt2tgt_sparse_col])
            attn_bias_2_target = self.edge_dist_encode_num_fc(attn_bias_2_target)

            attn_bias_ctx2tgt = attn_bias_1_target + attn_bias_2_target
                    
        return attn_bias_tgt2cxt, attn_bias_ctx2tgt
    
    @torch.no_grad()
    def get_encodings_using_cached_memory(self, edge_encodes, edge_dist_encodes, target_node_size, context_node_size,
                      tgt2cxt_sparse_row, tgt2cxt_sparse_col,
                      cxt2tgt_sparse_row, cxt2tgt_sparse_col, device):
        
        #### step 1.0: attn_bias_tgt2cxt.
        if context_node_size > 0:
            num_attn_compute        = tgt2cxt_sparse_row.size(0)
            attn_compute_ind        = torch.arange(num_attn_compute).to(device)
            attn_compute_ind_splits = torch.split(attn_compute_ind, 4096)
            
            context_edge_encodes = edge_encodes[target_node_size:, :target_node_size, :]
            context_edge_dist_encodes = edge_dist_encodes[target_node_size:, :target_node_size]
            
            attn_bias_1_context_list, attn_bias_2_context_list = [], []
            for cur_attn_compute_ind in attn_compute_ind_splits:
                cur_tgt2cxt_sparse_row = tgt2cxt_sparse_row[cur_attn_compute_ind]
                cur_tgt2cxt_sparse_col = tgt2cxt_sparse_col[cur_attn_compute_ind]
                
                attn_bias_1_context = self.edge_embedding(context_edge_encodes[cur_tgt2cxt_sparse_row, cur_tgt2cxt_sparse_col, :])
                temporal_weights = F.softmax(self.temporal_edge_encode_weights[:attn_bias_1_context.size(-2)], dim=0).view(1, -1, 1)
                attn_bias_1_context = torch.sum(attn_bias_1_context*temporal_weights, axis=-2)
                attn_bias_1_context_list.append(self.edge_encode_fc(attn_bias_1_context))
                
                attn_bias_2_context = self.edge_dist_embedding(context_edge_dist_encodes[cur_tgt2cxt_sparse_row, cur_tgt2cxt_sparse_col])
                attn_bias_2_context_list.append(self.edge_dist_encode_num_fc(attn_bias_2_context))
                
            attn_bias_1_context = torch.cat(attn_bias_1_context_list, dim=0)
            attn_bias_2_context = torch.cat(attn_bias_2_context_list, dim=0)
            del attn_bias_1_context_list, attn_bias_2_context_list
            
            attn_bias_tgt2cxt = attn_bias_1_context + attn_bias_2_context
                    
            #### step 2.0: attn_bias_ctx2tgt
            num_attn_compute        = cxt2tgt_sparse_row.size(0)
            attn_compute_ind        = torch.arange(num_attn_compute).to(device)
            attn_compute_ind_splits = torch.split(attn_compute_ind, 4096)
            
            target_edge_encodes = edge_encodes[:target_node_size, target_node_size:, :]
            target_edge_dist_encodes = edge_dist_encodes[:target_node_size:, target_node_size:]
            
            attn_bias_1_target_list, attn_bias_2_target_list = [], []
            for cur_attn_compute_ind in attn_compute_ind_splits:
                cur_cxt2tgt_sparse_row = cxt2tgt_sparse_row[cur_attn_compute_ind]
                cur_cxt2tgt_sparse_col = cxt2tgt_sparse_col[cur_attn_compute_ind]
                
                attn_bias_1_target = self.edge_embedding(target_edge_encodes[cur_cxt2tgt_sparse_row, cur_cxt2tgt_sparse_col, :])
                temporal_weights = F.softmax(self.temporal_edge_encode_weights[:attn_bias_1_target.size(-2)], dim=0).view(1, -1, 1)
                attn_bias_1_target = torch.sum(attn_bias_1_target*temporal_weights, axis=-2)
                attn_bias_1_target_list.append(self.edge_encode_fc(attn_bias_1_target))
                
                attn_bias_2_target = self.edge_dist_embedding(target_edge_dist_encodes[cur_cxt2tgt_sparse_row, cur_cxt2tgt_sparse_col])
                attn_bias_2_target_list.append(self.edge_dist_encode_num_fc(attn_bias_2_target))
                
            attn_bias_1_target = torch.cat(attn_bias_1_target_list, dim=0)
            attn_bias_2_target = torch.cat(attn_bias_2_target_list, dim=0)
            del attn_bias_1_target_list, attn_bias_2_target_list
            attn_bias_ctx2tgt = attn_bias_1_target + attn_bias_2_target
            
        else:
            num_attn_compute        = cxt2tgt_sparse_row.size(0)
            attn_compute_ind        = torch.arange(num_attn_compute).to(device)
            attn_compute_ind_splits = torch.split(attn_compute_ind, 4096)
            
            target_edge_encodes = edge_encodes[:target_node_size, :target_node_size, :]
            target_edge_dist_encodes = edge_dist_encodes[:target_node_size:, :target_node_size]
            
            attn_bias_1_target_list, attn_bias_2_target_list = [], []
            for cur_attn_compute_ind in attn_compute_ind_splits:
                cur_cxt2tgt_sparse_row = cxt2tgt_sparse_row[cur_attn_compute_ind]
                cur_cxt2tgt_sparse_col = cxt2tgt_sparse_col[cur_attn_compute_ind]
                
                attn_bias_1_target = self.edge_embedding(target_edge_encodes[cur_cxt2tgt_sparse_row, cur_cxt2tgt_sparse_col, :])
                temporal_weights = F.softmax(self.temporal_edge_encode_weights[:attn_bias_1_target.size(-2)], dim=0).view(1, -1, 1)
                attn_bias_1_target = torch.sum(attn_bias_1_target*temporal_weights, axis=-2)
                attn_bias_1_target_list.append(self.edge_encode_fc(attn_bias_1_target))
                
                attn_bias_2_target = self.edge_dist_embedding(target_edge_dist_encodes[cur_cxt2tgt_sparse_row, cur_cxt2tgt_sparse_col])
                attn_bias_2_target_list.append(self.edge_dist_encode_num_fc(attn_bias_2_target))
                
            attn_bias_1_target = torch.cat(attn_bias_1_target_list, dim=0)
            attn_bias_2_target = torch.cat(attn_bias_2_target_list, dim=0)
            del attn_bias_1_target_list, attn_bias_2_target_list
            attn_bias_ctx2tgt = attn_bias_1_target + attn_bias_2_target
                    
            attn_bias_tgt2cxt = None
        
        return attn_bias_tgt2cxt, attn_bias_ctx2tgt
        
    def compute_loss(self, feed_dict):
        #pad_more_nodes
        node_feats_np, edge_encode_np, edge_dist_encode_np, target_node_size, context_nodes_size, _  = get_common_neighbors(eval_nodes_with_pad, eval_edge_encode, eval_PPR, 
                                                                                                                                eval_edge_dist_encode,
                                                                                                                                feats_train[-1], FLAGS, 
                                                                                                                                deterministic_neighbor_sampling=True)
    def forward(self, x, edge_encodes, edge_dist_encodes, 
                target_node_size, context_node_size, device):
        
        if context_node_size > 0:
            # for each node, random sample some neighbors
            cxt2tgt_sparse_row, cxt2tgt_sparse_col = [], []
            tgt2cxt_sparse_row, tgt2cxt_sparse_col = [], []
            
            if self.training:
                context_sample_size = math.ceil(context_node_size*self.neighbor_sampling_size)
                target_sample_size  = math.ceil(target_node_size *self.neighbor_sampling_size)
            
                for i in range(target_node_size):
                    col_sample = torch.arange(context_node_size)
                    select = torch.randperm(len(col_sample))[:context_sample_size]
                    cxt2tgt_sparse_row.append(torch.ones(len(select))*i)
                    cxt2tgt_sparse_col.append(col_sample[select])


                for i in range(context_node_size):
                    col_sample = torch.arange(target_node_size)
                    select = torch.randperm(len(col_sample))[:target_sample_size]
                    tgt2cxt_sparse_row.append(torch.ones(len(select))*i)
                    tgt2cxt_sparse_col.append(col_sample[select])
            else:
                for i in range(target_node_size):
                    col_sample = torch.arange(context_node_size)
                    cxt2tgt_sparse_row.append(torch.ones(len(col_sample))*i)
                    cxt2tgt_sparse_col.append(col_sample)


                for i in range(context_node_size):
                    col_sample = torch.arange(target_node_size)
                    tgt2cxt_sparse_row.append(torch.ones(len(col_sample))*i)
                    tgt2cxt_sparse_col.append(col_sample)
                
            cxt2tgt_sparse_row = torch.cat(cxt2tgt_sparse_row).long().to(device)
            cxt2tgt_sparse_col = torch.cat(cxt2tgt_sparse_col).long().to(device)        
            tgt2cxt_sparse_row = torch.cat(tgt2cxt_sparse_row).long().to(device)
            tgt2cxt_sparse_col = torch.cat(tgt2cxt_sparse_col).long().to(device)
        else:
            cxt2tgt_sparse_row, cxt2tgt_sparse_col = [], []
            tgt2cxt_sparse_row, tgt2cxt_sparse_col = None, None

            if self.training:
                context_sample_size = 0
                target_sample_size  = math.ceil(target_node_size * self.neighbor_sampling_size)

                for i in range(target_node_size):
                    col_sample = torch.arange(target_sample_size)
                    select = torch.randperm(len(col_sample))[:target_sample_size]
                    cxt2tgt_sparse_row.append(torch.ones(len(select))*i)
                    cxt2tgt_sparse_col.append(col_sample[select])

            else:
                for i in range(target_node_size):
                    col_sample = torch.arange(target_node_size)
                    cxt2tgt_sparse_row.append(torch.ones(len(col_sample))*i)
                    cxt2tgt_sparse_col.append(col_sample)
                
            cxt2tgt_sparse_row = torch.cat(cxt2tgt_sparse_row).long().to(device)
            cxt2tgt_sparse_col = torch.cat(cxt2tgt_sparse_col).long().to(device)        

        #######################################################
        #######################################################
        #######################################################
        x = self.node_feats_fc(x) 
        
        if self.training: # if context_node_size = 0, we only use attn_bias_ctx2tgt
            attn_bias_tgt2cxt, attn_bias_ctx2tgt = self.get_encodings(edge_encodes, edge_dist_encodes, target_node_size, context_node_size,
                                                                     tgt2cxt_sparse_row, tgt2cxt_sparse_col,
                                                                     cxt2tgt_sparse_row, cxt2tgt_sparse_col)
        else:
            attn_bias_tgt2cxt, attn_bias_ctx2tgt = self.get_encodings_using_cached_memory(
                                                                     edge_encodes, edge_dist_encodes, target_node_size, context_node_size,
                                                                     tgt2cxt_sparse_row, tgt2cxt_sparse_col,
                                                                     cxt2tgt_sparse_row, cxt2tgt_sparse_col, device)
            
        for ell in range(self.num_layers):
            x = self.transformer_layer_list[ell](x, target_node_size, context_node_size, 
                                                 attn_bias_ctx2tgt, attn_bias_tgt2cxt, 
                                                 tgt2cxt_sparse_row, tgt2cxt_sparse_col, 
                                                 cxt2tgt_sparse_row, cxt2tgt_sparse_col, device)
        x = x.squeeze(0)

        if self.use_unsupervised_loss and self.training:
            temporal_output = []
            for decoder_head in self.decoder_heads:
                temporal_output.append(decoder_head(x))
            x_unsupervised = torch.stack(temporal_output)
            return x, x_unsupervised
        else:
            # x = self.decoder(x)
            return x, None

##########################################################################
##########################################################################
##########################################################################

class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, dropout_rate, attention_dropout_rate, head_size):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size)
        self.self_attention = MultiHeadAttention(hidden_size, attention_dropout_rate, head_size)
        self.self_attention_dropout = nn.Dropout(dropout_rate)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, hidden_size, dropout_rate)
        self.ffn_dropout = nn.Dropout(dropout_rate)

    def forward(self, x_all, target_node_size, context_node_size, 
                attn_bias_ctx2tgt, attn_bias_tgt2cxt,
                tgt2cxt_sparse_row, tgt2cxt_sparse_col,
                cxt2tgt_sparse_row, cxt2tgt_sparse_col, device):
        
        y_all = self.self_attention_norm(x_all)
        
        if context_node_size > 0:
            
            y_target, y_context = torch.split(y_all, [target_node_size, context_node_size], dim=0)

            if self.training:
                y_context_ = self.self_attention(q=y_target, k=y_context, v=y_context, 
                                                   attn_bias=attn_bias_ctx2tgt, 
                                                   sparse_row = cxt2tgt_sparse_row, 
                                                   sparse_col = cxt2tgt_sparse_col, 
                                                   attn_shape = (target_node_size, context_node_size))  # attn_bias_ctx2tgt  = [target_size,  context_size]

                y_target_  = self.self_attention(q=y_context, k=y_target,  v=y_target,  
                                                   attn_bias=attn_bias_tgt2cxt, 
                                                   sparse_row = tgt2cxt_sparse_row, 
                                                   sparse_col = tgt2cxt_sparse_col,
                                                   attn_shape = (context_node_size, target_node_size)) # attn_bias_tgt2cxt = [context_size, context_size]
            else:
                y_context_ = self.self_attention.forward_using_cached_memory(
                                                   q=y_target, k=y_context, v=y_context, 
                                                   attn_bias  = attn_bias_ctx2tgt, 
                                                   sparse_row = cxt2tgt_sparse_row, 
                                                   sparse_col = cxt2tgt_sparse_col, 
                                                   attn_shape = (target_node_size, context_node_size), device = device)  # attn_bias_ctx2tgt  = [target_size,  context_size]

                y_target_  = self.self_attention.forward_using_cached_memory(
                                                   q=y_context, k=y_target,  v=y_target,  
                                                   attn_bias  = attn_bias_tgt2cxt, 
                                                   sparse_row = tgt2cxt_sparse_row, 
                                                   sparse_col = tgt2cxt_sparse_col,
                                                   attn_shape = (context_node_size, target_node_size), device = device) # attn_bias_tgt2cxt = [context_size, context_size]

            y_all = torch.cat([y_target_, y_context_], dim=0)
            
        else:
            # print('No context node')
            # print(y_all.shape, attn_bias_ctx2tgt.shape, target_node_size)
            if self.training:
                # attn_bias_ctx2tgt  = [target_size,  context_size]
                y_all = self.self_attention(
                               q=y_all, k=y_all, v=y_all, 
                               attn_bias=attn_bias_tgt2cxt, 
                               sparse_row = cxt2tgt_sparse_row, 
                               sparse_col = cxt2tgt_sparse_col, 
                               attn_shape = (target_node_size, target_node_size))  

        
            else:
                # attn_bias_ctx2tgt  = [target_size,  context_size]
                y_all = self.self_attention.forward_using_cached_memory(
                               q=y_all, k=y_all, v=y_all, 
                               attn_bias  = attn_bias_ctx2tgt, 
                               sparse_row = cxt2tgt_sparse_row, 
                               sparse_col = cxt2tgt_sparse_col, 
                               attn_shape = (target_node_size, target_node_size), device = device) 
        
        # print(y_all.shape, x_all.shape)
        y_all = self.self_attention_dropout(y_all)
        x_all = x_all + y_all

        y_all = self.ffn_norm(x_all)
        y_all = self.ffn(y_all)
        y_all = self.ffn_dropout(y_all)
        x_all = x_all + y_all
        return x_all

##########################################################################
##########################################################################
##########################################################################

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout_rate, head_size):
        super(MultiHeadAttention, self).__init__()

        self.head_size = head_size

        self.att_size = att_size = hidden_size // head_size
        self.scale = att_size ** -0.5

        self.linear_q = nn.Linear(hidden_size, head_size * att_size)
        self.linear_k = nn.Linear(hidden_size, head_size * att_size)
        self.linear_v = nn.Linear(hidden_size, head_size * att_size)
        self.att_dropout = nn.Dropout(attention_dropout_rate)

        self.output_layer = nn.Linear(head_size * att_size, hidden_size)

    def forward(self, q, k, v, attn_bias, sparse_row, sparse_col, attn_shape):
        
        ##########################

        d_k = self.att_size
        d_v = self.att_size
        
        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(-1, self.head_size, d_k)
        k = self.linear_k(k).view(-1, self.head_size, d_k)
        v = self.linear_v(v).view(-1, self.head_size, d_v)
                
        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        x = torch.sum(q[sparse_row, :, :] * k[sparse_col, :, :], dim=-1) * self.scale  # [b, h, q_len, k_len]
        
      
        if attn_bias is not None:
            x = x + attn_bias

        x = softmax(values=x, indices=sparse_row)
        x = self.att_dropout(x)
        
        x = x.view(-1, self.head_size, 1)
        x = scatter(x * v[sparse_col, :, :], sparse_row, reduce='sum', dim=0)
        x = x.reshape(-1, self.head_size * d_v)
        
        x = self.output_layer(x)
        
        return x
    
    # used for large-scale inference
    @torch.no_grad()
    def forward_using_cached_memory(self, q, k, v, attn_bias, sparse_row, sparse_col, attn_shape, device):
        ##########################
        d_k = self.att_size
        d_v = self.att_size
        ##########################
        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(-1, self.head_size, d_k)
        k = self.linear_k(k).view(-1, self.head_size, d_k)
        v = self.linear_v(v).view(-1, self.head_size, d_v)
        
        ##########################
        
        num_attn_compute        = sparse_row.size(0)
        
        attn_compute_ind        = torch.arange(num_attn_compute).to(device)
        attn_compute_ind_splits = torch.split(attn_compute_ind, 1024)
        
        x = []
        for cur_attn_compute_ind in attn_compute_ind_splits:
            # Scaled Dot-Product Attention.
            # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
            cur_sparse_row = sparse_row[cur_attn_compute_ind]
            cur_sparse_col = sparse_col[cur_attn_compute_ind]
            x_ = torch.sum(q[cur_sparse_row, :, :] * k[cur_sparse_col, :, :], dim=-1) * self.scale + attn_bias[cur_attn_compute_ind, :]
            x.append(x_)
        x = torch.cat(x, dim=0)
        
        x = softmax(values=x, indices=sparse_row)
        
        # borrow implementation from DySAT, seems like it waste more memory and takes more time
        x = x.view(-1, self.head_size)
        edge_inds = torch.stack([sparse_row, sparse_col])
        
        x_head_outputs = []
        for head in range(self.head_size):
            x_sparse = torch.sparse.FloatTensor(edge_inds, x[:, head], attn_shape)     
            x_ = torch.sparse.mm(x_sparse, v[:, head, :]) 
            x_head_outputs.append(x_)
        x = torch.cat(x_head_outputs, dim=-1)
        
        x = self.output_layer(x)
        
        return x
        
##########################################################################
##########################################################################
##########################################################################

def softmax(values, indices):

    src_max = scatter(values, indices, reduce='max', dim=0)
    out = (values-src_max[indices, :]).exp()
    out_sum = scatter(out, indices, reduce='sum', dim=0) 

    return out / (out_sum[indices] + 1e-16)

def average(values, indices):
    values = values.flatten()
    indices = indices.flatten()
    values_sum = scatter(values, indices, reduce='sum') 
    return values / (values_sum[indices] + 1e-16)

def prune_weak_attn(values, indices, thresh): 
    # the smaller thresh, the more nodes left
    values = values.flatten()
    indices = indices.flatten()
    
    src_mean = scatter(values, indices, reduce='mean')
    src_mean = src_mean[indices]
    
    src_mask = torch.zeros_like(values).bool()
    src_mask[values > thresh * src_mean] = True
    
    return average(values[src_mask], indices[src_mask]), src_mask

##########################################################################
##########################################################################
##########################################################################

class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size, dropout_rate):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)
        
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        # x = self.dropout(x)
        x = self.layer2(x)
        # x = self.dropout(x)
        return x
    
