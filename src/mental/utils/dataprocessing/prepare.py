import random
from .merge import merge_ego_graphs_for_user_over_time, merge_ego_graphs_from_users, merge_hyper_graphs_from_users, merge_text_from_user, to_geometric_data
from .transform import nx_graphs_to_tensor, normalized_graph_weight, to_hetero_graph, to_hyper_graphs, construct_hetero_data
import torch

def prepare_static_graph(data):
    labels = []
    graphs = []
    for i, row in enumerate(data):
        labels.append(row['label'])
        graphs.append(row['graphs'])
    user_graph_list = [merge_ego_graphs_for_user_over_time(user_graphs, hetero = False) for user_graphs in graphs]
    merged_graph = merge_ego_graphs_from_users(user_graph_list)
    return [{'index': i, 'label': labels[i], 'static_graph': merged_graph} for i in range(len(labels))]


def prepare_batch_static_graph(batch_data):
    graph = batch_data['static_graph']
    labels =  graph.label[graph.label != -100].unsqueeze(-1)
    return {'labels': labels, 'graph':  batch_data['static_graph']}

def prepare_dynamic_homo_graphs(data):
    labels = []
    graphs = []
    for i, row in enumerate(data):
        labels.append(row['label'])
        graphs.append(row['graphs'])
        
    graphs = [user_graphs for user_graphs in graphs]
    return [{'index': i, 'label': labels[i], 'dynamic_graph': graphs} for i in range(len(labels))]

def prepare_batch_dynamic_homo_graphs(batch_data):
    device = batch_data['label'].device
    batch_graphs = batch_data['dynamic_graph']
    graphs = [nx_graphs_to_tensor(user_graphs, device) for user_graphs in batch_graphs]
    return {'labels': batch_data['label'], 'graphs': graphs}

def prepare_dynamic_hetero_graphs(data):
    labels = []
    graphs = []
    for i, row in enumerate(data):
        labels.append(row['label'])
        graphs.append(row['graphs'])
        
    graphs = [user_graphs for user_graphs in graphs]
    return [{'index': i, 'label': labels[i], 'dynamic_graph': graphs} for i in range(len(labels))]

def prepare_batch_dynamic_hetero_graphs(batch_data):
    device = batch_data['label'].device
    batch_graphs = batch_data['dynamic_graph']
    graphs = [nx_graphs_to_tensor(user_graphs, device) for user_graphs in batch_graphs]
    hetero_data = [[construct_hetero_data(user_graph) for user_graph in user_graphs] for user_graphs in graphs]

    return {'labels': batch_data['label'], 'graphs': hetero_data}

def prepare_dynamic_hyper_graphs(data):
    user_indices = []
    labels = []
    period_indices = []
    graphs = []
    user_indices = []
    for i, row in enumerate(data):
        labels.append(row['label'])
        period_indices.append(row['period_id'])
        user_indices.append(row['user_id'])
        graphs.append(row['graphs'])
        user_indices.append(row['user_original_id'])
    
    user_hyper_graphs = []
    for user_index, user_graphs in enumerate(graphs):
        for i, G in enumerate(user_graphs):
            G.graph['period_id'] = i
            G.graph['id'] = user_index
        user_hyper_graphs.append(to_hyper_graphs(user_graphs, user_index))
    merged_graph = merge_hyper_graphs_from_users(user_hyper_graphs)
    return [{'index': i, 'label': labels[i], 'user_original_id': user_indices[i], 'hyper_graph': merged_graph} for i in range(len(labels))]

def prepare_batch_dynamic_hyper_graphs(batch_data, training = False):
    device = batch_data['label'].device
    graph = construct_hetero_data(batch_data['hyper_graph']).to(device)
    labels = []
    for i, group_id in enumerate(graph.group.unique()):
        index = (graph.group == group_id) & ((graph.label != -100))
        #Just in case the order is different.
        labels.append(graph.label[index][0])
    labels = torch.tensor(labels, device = device).unsqueeze(-1)

    return {'labels': labels, 'graph': graph}

def prepare_static_hetero_graph(data):
    labels = []
    graphs = []
    for i, row in enumerate(data):
        labels.append(row['label'])
        graphs.append(row['graphs'])
    user_graph_list = [merge_ego_graphs_for_user_over_time(user_graphs) for user_graphs in graphs]
    merged_graph = merge_ego_graphs_from_users(user_graph_list)
    return [{'index': i, 'label': labels[i], 'static_hetero_graph': merged_graph} for i in range(len(labels))]


def prepare_batch_static_hetero_graph(batch_data):
    device = batch_data['label'].device
    graph = construct_hetero_data(batch_data['static_hetero_graph']).to(device)
    labels =  graph.label[graph.label != -100].unsqueeze(-1)
    return {'labels': labels, 'graph':  graph}


def prepare_static_hetero_graph_by_user(batch_data):
    device = batch_data['label'].device
    batch_graphs = batch_data['graphs']
    graphs = [merge_ego_graphs_for_user_over_time(user_graphs) for user_graphs in batch_graphs]
    graphs = nx_graphs_to_tensor(graphs, device)
    batch_hetero_graphs = [to_hetero_graph(user_graph, group_index).to(device) for group_index, user_graph in enumerate(graphs)]
    return {'labels': batch_data['label'], 'graphs': batch_hetero_graphs}

def prepare_text_inputs(data):
    labels = []
    graphs = []
    for i, row in enumerate(data):
        labels.append(row['label'])
        graphs.append(row['graphs'])
    
    text_data = [merge_text_from_user(user_graphs) for user_graphs in graphs]
    return [{'index': i, 'label': labels[i], 'text_data': text_data} for i in range(len(labels))]

def prepare_batch_text_inputs(batch_data):
    device = batch_data['label'].device
    indices = batch_data['index']
    text_data = []
    for i in range(len(indices)):
        text_data.append(batch_data['text_data'][0][indices[i]].to(device))
    
    return {
        'labels': batch_data['label'],
        'text_data': text_data,
    }
