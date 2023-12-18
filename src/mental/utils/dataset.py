import torch
import networkx as nx
import re
from torch.utils.data import Dataset

from .dataprocessing.merge import (
    merge_ego_graphs_from_users,
    merge_hyper_graphs_from_users,
    merge_hetero_graphs_from_users,
    to_geometric_data,
)
from .dataprocessing.transform import (
    construct_hetero_data,
    to_hetero_graph,
    to_hetero_graphs,
)

from torch_geometric.utils import subgraph
class MedDataset(Dataset):
    def __init__(self, raw_data):
        super(MedDataset, self).__init__()
        allowed_keys = ['index', 'label', 'user_id', 'user_node_id', 'period_id', 'graph', 'graphs', 'static_graph', 'hyper_graph', 'hyper_graph_augmentation', 'hyper_graphs', 'hetero_graph', 'hetero_graphs', 'hetero_graph_by_user']
        self.users = []
        for row in raw_data:
            inputs = {}
            keys = list(row.keys())
            for key in keys:
                if key in allowed_keys:
                    inputs[key] = row[key]
            
            self.users.append(inputs)

    def __len__(self):
        return len(self.users)

    def __getitem__(self, index):
        return self.users[index]

    @staticmethod
    def collate_fn(samples):
        batch_dict = {}
        for key in ['index', 'user_id', 'label', 'text_data', 'graph', 'graphs', 'hyper_graph', 'static_hetero_graph', 'dynamic_graph', 'hyper_graphs', 'hyper_graph_augmentation', 'hetero_graphs', 'hetero_graph_by_user', 'static_graph', 'mention_graph', 'quote_graph', 'reply_graph', 'hetero_graph', 'period_id', 'timestamp']:
            data_list = []
            for sample in samples:
                if key in sample:
                    data_list.append(sample[key])
            if len(data_list) > 0:
                assert(len(data_list) == len(samples))
                batch_dict[key] = data_list
        # merge static_graph
        if 'label' in batch_dict:
            batch_dict['label'] = torch.Tensor(batch_dict['label']).unsqueeze(-1)
        if 'static_graph' in batch_dict:
            index = set(batch_dict['index'])
            graph = batch_dict['static_graph'][0]
            retain_nodes = []

            for user_id, data in graph.nodes(data=True):
                if int(data['group']) in index:
                    retain_nodes.append(user_id)
            batch_graph = graph.subgraph(retain_nodes)
            batch_graph = to_geometric_data(batch_graph, undirected = True)

            batch_dict['static_graph'] = batch_graph

        if 'static_hetero_graph' in batch_dict:
            index = set(batch_dict['index'])
            graph = batch_dict['static_hetero_graph'][0]
            
            retain_nodes = set()
            G = nx.MultiGraph()
            new_group_mapping = {}
            counter = 0
            for user_id, data in graph.nodes(data=True):
                result = re.match(r"(user|friend)?_(\d+)_(\d+)", user_id)
                old_group_id = int(result.groups()[2])
                if data['group'] in index:
                    if old_group_id not in new_group_mapping:
                        new_group_mapping[old_group_id] = counter
                        counter += 1
                    data['group'] = new_group_mapping[old_group_id]
                    G.add_node(user_id, **data)
                    retain_nodes.add(user_id)
            for src, dst, data in graph.edges(data=True):
                if src in retain_nodes and dst in retain_nodes:
                    G.add_edge(src, dst, **data)
            batch_graph = to_geometric_data(G, undirected = False)
            batch_dict['static_hetero_graph'] = batch_graph
        if 'hyper_graph' in batch_dict:
            # All are the same, use the first one.
            index = set(batch_dict['index'])
            graph = batch_dict['hyper_graph'][0]
            retain_nodes = set()
            G = nx.MultiGraph()
            new_group_mapping = {}
            counter = 0
            for user_id, data in graph.nodes(data=True):
                old_group_id = int(user_id.split('_')[0])
                if old_group_id in index:
                    if old_group_id not in new_group_mapping:
                        new_group_mapping[old_group_id] = counter
                        counter += 1
                    data['group'] = new_group_mapping[old_group_id]
                    G.add_node(user_id, **data)
                    retain_nodes.add(user_id)
            for src, dst, data in graph.edges(data=True):
                if src in retain_nodes and dst in retain_nodes:
                    G.add_edge(src, dst, **data)
            
            batch_graph = G
            #batch_graph = graph.subgraph(retain_nodes)
            batch_graph = to_geometric_data(batch_graph, undirected = False)
            batch_dict['hyper_graph'] = batch_graph

        if 'dynamic_graph' in batch_dict:
            # All are the same, use the first one.
            index = batch_dict['index']
            dynamic_graphs = []
            graphs = batch_dict['dynamic_graph'][0]

            labels = []
            dynamic_graphs = []
            for i in index:
                dynamic_graphs.append(graphs[i])
                labels.append(batch_dict['label'])
            batch_dict['dynamic_graph'] = dynamic_graphs
            

        if 'hetero_graph' in batch_dict:
            batch_dict['hetero_graph'] = merge_hetero_graphs_from_users(batch_dict['hetero_graph'[0]])
        if 'hetero_graphs' in batch_dict:
            hetero_graphs = []
            for i in range(len(batch_dict['hetero_graphs'])):
                user_graphs = [to_hetero_graph(graph, i) for graph in batch_dict['hetero_graphs'][i]]
                hetero_graphs.append(user_graphs)
            
            batch_dict['hetero_graphs'] = hetero_graphs
        if 'hetero_graph_by_user' in batch_dict:
            batch_dict['hetero_graph_by_user'] = [to_hetero_graph(batch_dict['hetero_graph_by_user'][i], i) for i in range(len(batch_dict['hetero_graph_by_user']))]

        return batch_dict
