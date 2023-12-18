import os
from datetime import datetime
from .base_pipeline import AbstractPipeline
from ..utils.dataprocessing.model_loader import get_model
from pathlib import Path
from ..utils.dataprocessing import get_dataloader
from ..utils.training.callback import get_ckpt_save_path
from ..models.model import LightingModel
import torch_geometric
from torch_geometric.utils.subgraph import subgraph
from torch_geometric.utils import add_self_loops, contains_isolated_nodes
import torch
import pprint
import pandas as pd
import glob
import numpy as np
import json
import itertools
import networkx as nx
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
    
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        print(obj)
        return json.JSONEncoder.default(self, obj)
    
class ModelVisualization(AbstractPipeline):
    def check_graph(self, data):

        isolated_count = 0
        for row in data:
            for graph in row['graphs']:
                isolated_count += len(list(nx.isolates(graph)))
        return isolated_count == 0
    def check_graph2(self, graph):
        groups = graph.group.unique()
        batch_size = graph.group.shape[0]
        node_indices = torch.arange(batch_size)
        period_indices = graph.period_id.unique().detach().cpu().numpy().astype(np.int32).tolist()
        flag = False
        for i, group_id in enumerate(groups):
            for relation in ['quote', 'reply', 'mention']:
                edge_index = graph.edge_index_dict[relation]
                period_group = (graph.group == group_id) & (graph.period_id == graph.period_id)
                new_edge_index, edge_attr, _ = subgraph(period_group, edge_index, relabel_nodes = True, return_edge_mask = True)
                isolated = contains_isolated_nodes(new_edge_index)
    def run(self, context):
        from transformers import AutoModelForSequenceClassification
        from transformers import AutoTokenizer, AutoConfig
        data = context.data
        results = []
        ckpt_dir = Path("../checkpoints/")
        cv_gorup = int(datetime.timestamp(datetime.now()))

        counter = 0
        dfs = []
        MODEL = f"cardiffnlp/twitter-roberta-base-sentiment-latest"
        tokenizer = AutoTokenizer.from_pretrained(MODEL)
        config = AutoConfig.from_pretrained(MODEL)
    
        # PT
        nlpmodel = AutoModelForSequenceClassification.from_pretrained(MODEL)


        user_records = []
        data_counter = 0
        for cv_data in context.cv_data:
            model, args, model_args = get_model(context.model_class, context.dataset_info)
            model.reset_parameters()
            original_data = data#[cv_data.test_indices]
            test_data = model.prepare_inputs(original_data)
            test_dataloader = get_dataloader(test_data, batch_size = 999, shuffle = False)
            ckpt_name = get_ckpt_save_path(model.__class__.__name__, context.dataset_info, cv_data.fold)
            ckpt_path = ckpt_dir / f"{ckpt_name}.ckpt"
            if not os.path.exists(ckpt_path):
                raise ValueError(f"FileNotExists: {ckpt_path}.")
            
            checkpoint = torch.load(ckpt_path)
            lighting_model = LightingModel.load_from_checkpoint(ckpt_path, model = model).cpu()
            lighting_model.eval()
            softmax = torch.nn.Softmax(dim=-1)
            for batch in test_dataloader:
                # train step
                if not self.check_graph(original_data):
                    raise ValueError("Graph error")
                inputs = model.prepare_batch_inputs(batch)
                #if not self.check_graph2(inputs['graph']):
                #    raise ValueError("Wrong ")
                batch_output, labels = model.get_attention_scores(**inputs)
                prediction_scores = batch_output.prediction_scores.detach().cpu().numpy()
                labels = labels.cpu().detach().numpy()
                graph = inputs['graph']
                x_dict, edge_index_dict, edge_weight_dict, group, label = graph.x_dict, graph.edge_index_dict, graph.edge_weight_dict, graph.group, graph.label
                
                user_nodes = (label != -100)
                groups = group.unique()
                batch_size = group.shape[0]
                node_indices = torch.arange(batch_size)
                period_indices = graph.period_id.unique().detach().cpu().numpy().astype(np.int32).tolist()
                for i, group_id in enumerate(groups):
                    if original_data[i]['user_original_id'] != '123456789': continue
                
                    if labels[i] == 0: continue
                    user_id = original_data[i]['user_id']
                    user_index = (group == group_id) & user_nodes
                    friend_indices = (group == group_id) & ~user_nodes
                    user_graphs = original_data[i]['graphs']
                    user_tweets = []
                    user_tweets_indices = []

                    friend_tweets = []
                    friend_tweets_indices = []
                    for user_graph in user_graphs:
                        for n_id, node_data in user_graph.nodes(data=True):
                            for k in range(len(node_data['tweets_content'])):
                                node_data['tweets_content'][k] = node_data['tweets_content'][k].replace('fucking', '***').replace('fuck', '***').replace('cunt', '***')
                        
                            if node_data['label'] != -100:
                                user_tweets.append(node_data['tweets_content'])
                                user_tweets_indices.append(node_data['tweets_id'])
                            else:
                                friend_tweets.append(node_data['tweets_content'])
                                friend_tweets_indices.append(node_data['tweets_id'])


                    gat_data = {period_id: {} for period_id in period_indices}
                    for period_id in period_indices:
                        for relation, layer_data in batch_output.graph_attention_weights.items():
                            for layer, (edge_index, gat_weights) in enumerate(layer_data):
                                if relation not in gat_data[period_id]:
                                    gat_data[period_id][relation] = {0: {}, 1: {}, 2:{}, 3: {}}
                                period_group = (group == group_id) & (graph.period_id == period_id)
                                new_edge_index, new_attn_gat_weights, _ = subgraph(period_group, edge_index, edge_attr = gat_weights, relabel_nodes = True, return_edge_mask = True)
                                
                                adj = torch_geometric.utils.to_scipy_sparse_matrix(new_edge_index, new_attn_gat_weights[:, 0]).todense().T.tolist()
                                new_edge_index = new_edge_index.detach().cpu().numpy()
                                new_attn_gat_weights = new_attn_gat_weights.detach().cpu().numpy()

                                nodes = {}
                                #print(period_id, relation, new_edge_index, new_attn_gat_weights)
                                for j in range(new_edge_index.shape[1]):
                                    src, dst = int(new_edge_index[0, j]), int(new_edge_index[1, j])
                                    #print(j, '||', src, dst, new_attn_gat_weights[j][0])
                                    #rint(period_id, relation, "Enter", src, dst)
                                    if dst not in nodes:
                                        nodes[dst] = {}
                                    if src not in nodes[dst]:
                                        nodes[dst][src] = 0
                                    nodes[dst][src] = new_attn_gat_weights[j][0]
                                    #print(f'nodes[{src}][{dst}] = {new_attn_gat_weights[j][0]}')
                            
                                for key, value in nodes.items():
                                    node_name = ''
                                    if key == 0:
                                        node_name = 'User'
                                    else:
                                        node_name = f'Friend_{key}'
                                    gat_data[period_id][relation][layer][key] = {
                                        'name': node_name,
                                        'attn_data': value,
                                        'adj': adj,
                                    }
                    user_tweets_sentiments = []
                    friend_tweets_sentiments = []

                    """
                    for tweets, container in [(user_tweets, user_tweets_sentiments), (friend_tweets, friend_tweets_sentiments)]:
                        encoded_input = tokenizer(list(itertools.chain(*tweets)), return_tensors = 'pt', max_length = 128,padding = 'max_length', truncation = True)
                        output = nlpmodel(**encoded_input)
                        scores = softmax(output.logits).detach().cpu().numpy()
                        sentiment_scores = []
                        for x in range(scores.shape[0]):
                            ranking = np.argsort(scores[x])
                            ranking = ranking[::-1]
                            sentiment = {}
                            for y in range(scores[x].shape[0]):
                                l = config.id2label[ranking[y]]
                                s = scores[x][ranking[y]]
                                sentiment[l] = np.round(float(s), 4)
                            sentiment_scores.append(sentiment)
                        container.extend(sentiment_scores)
                    """
                    #pprint.pprint(gat_data)
                    user_record = {
                        'user_id': f'{data_counter}_{cv_data.fold}_+{user_id}',
                        'labels': labels[i],
                        'original_user_id': original_data[i]['user_original_id'],
                        'prediction_scores': prediction_scores[i],
                        'prediction': (prediction_scores > 0.5)[0],
                        'user_tweets': user_tweets,
                        'user_tweets_sentiments': user_tweets_sentiments,
                        'user_tweet_indices': user_tweets_indices,
                        'user_tweet_period': graph.period_id[user_index].cpu().detach().numpy(),
                        'user_tweet_attention_weights': batch_output.tweet_attention_weights[user_index].cpu().detach().numpy(),
                        'friend_tweets': friend_tweets,
                        'friend_tweets_sentiments': friend_tweets_sentiments,
                        'friend_tweets_indices': friend_tweets_indices,
                        'friend_tweet_period': graph.period_id[friend_indices].cpu().detach().numpy(),
                        'friend_tweet_attention_weights': batch_output.tweet_attention_weights[friend_indices].cpu().detach().numpy(),
                        'gat_data': gat_data,
                        **({f'user_temporal_attention_layer_{j}': batch_output.temporal_attention_weights[j][i].cpu().detach().numpy() for j in range(len(batch_output.temporal_attention_weights))})
                    }
                    data_counter += 1
                    print("Processing User #", i)
                    user_records.append(user_record)
            # Write the dictionary to a JSON file
            user_records = sorted(user_records, key = lambda x: x['prediction_scores'][0], reverse=True)
            with open("data.json", "w") as json_file:
                json.dump(user_records, json_file, cls=NumpyEncoder)
            with open("data.json.js", "w") as json_file:
                text_to_append = "var DATA = "
                json_file.write(text_to_append)
                json.dump(user_records, json_file, cls=NumpyEncoder)
        exit()
        context.merge_update({
            'results': results,
        })
