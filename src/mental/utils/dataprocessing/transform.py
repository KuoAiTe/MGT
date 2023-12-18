import networkx as nx
import numpy as np
import torch
import torch_geometric
from torch.utils.data import Dataset
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import coalesce, contains_isolated_nodes

def normalized_graph_weight(graph):
    weights = {}
    for src, dst, data in graph.edges(data=True):
        edge_label, weight = data['label'], data['weight']
        if edge_label not in weights:
            weights[edge_label] = 0
        weights[edge_label] += weight

    for src, dst, data in graph.edges(data=True):
        edge_label = data['label']
        if weights[edge_label] > 0:
            data['weight'] = data['weight'] / weights[edge_label]
    return graph

def nx_graphs_to_tensor(graphs, device):
    return [torch_geometric.utils.from_networkx(graph).to(device) for graph in graphs]

def construct_hetero_data(homo_data):
    hetero_data = HeteroData()
    hetero_data['user'].x = homo_data.features
    edge_index = homo_data.edge_index
    edge_weight = homo_data.weight
    hetero_data['period_id'] = homo_data.period_id
    if 'group' in homo_data:
        hetero_data['group'] = homo_data.group
    hetero_data['label'] = homo_data.label
    if 'last_snapshot' in homo_data:
        hetero_data['last_snapshot'] = homo_data.last_snapshot
    hetero_data['edge_time'] = (hetero_data['period_id'][edge_index[0, :]] - hetero_data['period_id'][edge_index[1, :]]).abs()
    edge_relation = np.array(homo_data.edge_label)

    relations = np.unique(edge_relation)
    original_edge_index = []
    original_edge_weight = []
    original_edge_relation = []
    for r_id, relation in enumerate(relations):
        indices = np.where(edge_relation == relation, True, False)
        #print(edge_index)
        #print(indices)
        #print(edge_relation)
        #print('----------------------')
        hetero_data[relation].edge_index = edge_index[:, indices]
        hetero_data[relation].edge_weight = edge_weight[indices].float()
        hetero_data[relation].edge_time = hetero_data['edge_time'][indices]

        original_edge_index.append(hetero_data[relation].edge_index)
        original_edge_weight.append(hetero_data[relation].edge_weight)
        original_edge_relation.append(torch.ones_like(edge_weight).long() * r_id)
    original_edge_index = torch.cat(original_edge_index, dim = -1)
    original_edge_weight = torch.cat(original_edge_weight)
    original_edge_relation = torch.cat(original_edge_relation)
    original_edge_index, original_edge_relation = coalesce(original_edge_index, original_edge_relation)
    hetero_data['edge_index'] = original_edge_index
    hetero_data['edge_weight'] = original_edge_weight
    hetero_data['edge_relation'] = original_edge_relation
    return hetero_data

def to_hyper_graphs(graphs, user_counter):
    hyper_graph = nx.MultiDiGraph()
    same_node_mapping = {}
    counter = 0
    new_node_mapping = {}
    user_embeddings = []
    size = len(graphs)

    for i, graph in enumerate(graphs):
        mapping = {}
        period_id = graph.graph['period_id']
        for old_node_id, data in graph.nodes(data = True):
            if old_node_id not in new_node_mapping:
                new_node_mapping[old_node_id] = counter
                counter += 1
            if data['label'] != -100:
                user_embeddings.append(data['features'])
            node_id = new_node_mapping[old_node_id]
            new_node_id = f'{user_counter}_{node_id}_{period_id}'
            mapping[old_node_id] = new_node_id
            if node_id not in same_node_mapping:
                same_node_mapping[node_id] = []
            same_node_mapping[node_id].append(new_node_id)
            nx.set_node_attributes(graph, {old_node_id: {'node_id': node_id, 'period_id': period_id}})
            if i == size -1 and data['label'] != -100:
                nx.set_node_attributes(graph, {old_node_id: {'last_snapshot': True}})
            else:
                nx.set_node_attributes(graph, {old_node_id: {'last_snapshot': False}})
            
        graph = nx.relabel_nodes(graph, mapping)
        hyper_graph.add_nodes_from(graph.nodes(data = True))
        
        weights = {}
        for src, dst, data in graph.edges(data=True):
            edge_label, weight = data['label'], data['weight']
            if edge_label not in weights:
                weights[edge_label] = 0
            weights[edge_label] += weight

        for src, dst, data in graph.edges(data=True):
            edge_label = data['label']
            if weights[edge_label] > 0:
                data['weight'] = data['weight'] / weights[edge_label]
            data['weight'] = np.float32(data['weight'])
        hyper_graph.add_edges_from(graph.edges(data = True))

    nx.set_node_attributes(hyper_graph, {f'{user_counter}': {'features': user_embeddings}})
    """
    labels = ['mention', 'reply', 'quote']
    for nodes in same_node_mapping.values():
        size = len(nodes)
        for i in range(size - 1):
            src = nodes[i]
            for j in range(i + 1, size):
                dst = nodes[j]
                #for label in labels:
                hyper_graph.add_edge(src, dst, weight = 1, label = 'is')
                #hyper_graph.add_edge(dst, src, weight = 1, label = 'is')
    
    for node_id in hyper_graph.nodes():
        hyper_graph.add_edge(node_id, node_id, weight = 1, label = 'is')
    """
    return hyper_graph


def to_hetero_graphs(graphs, group, device):
    return [to_hetero_graph(graph, group).to(device) for graph in graphs]

def to_hetero_graph(graph, group):
    hetero_data = HeteroData()
    hetero_data['user'].x = graph['features']
    hetero_data['group'] = torch.full(graph.period_id.shape, group)
    hetero_data['period_id'] = graph.period_id
    edge_index = graph.edge_index
    if edge_index.shape[1] > 0:
        edge_weight = graph.weight
        edge_relation = np.array(graph.edge_label)
        relations = np.unique(edge_relation)
        for relation in relations:
            indices = np.where(edge_relation == relation, True, False)
            hetero_data[relation].edge_index = edge_index[:, indices]
            hetero_data[relation].edge_weight = edge_weight[indices].float()
    return hetero_data
