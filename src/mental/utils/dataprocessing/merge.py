import networkx as nx
import numpy as np
import torch
import torch_geometric
from torch_geometric.data import Data, HeteroData
from torch_geometric.utils import coalesce

from collections import defaultdict
from typing import List

def merge_edge_data(data, existing_data=None):
    # Define how to merge the edge data. Here, we define a new set of attributes for the merged edge
    # that include the 'type' attribute and the maximum value of the 'weight' attribute across all edges.
    if existing_data:
        merged_weight = data.get('weight', 0) + existing_data.get('weight', 0)
    else:
        merged_weight = data.get('weight', 0)
    merged_data = {
        'weight': merged_weight
    }
    return merged_data

def merge_edges(graph):
    merged_edges = {}
    for u, v, keys, data in graph.edges(keys=True, data=True):
        # Determine how to merge the edge data
        merged_data = merge_edge_data(data)
        # Create a new key that represents the merged edge
        merged_key = (u, v)
        # Add the merged edge to the dictionary
        if merged_key not in merged_edges:
            merged_edges[merged_key] = merged_data
        else:
            merged_edges[merged_key] = merge_edge_data(merged_edges[merged_key], merged_data)
        # Remove the original edge
    graph.remove_edges_from(list(graph.edges))
    # Add the merged edges to the graph
    for (u, v), data in merged_edges.items():
        graph.add_edge(u, v, **data)
    return graph
    
def merge_ego_graphs_for_user_over_time(graphs, hetero = True):
    # Dictionary to store node embeddings for each node id
    node_embeddings = defaultdict(list)
    weights = {}
    for graph_index, graph in enumerate(graphs):
        for node_id, data in graph.nodes(data = True):
            node_embeddings[node_id].append(data['features'])
            
        for src, dst, data in graph.edges(data = True):
            edge_label, weight = data['label'], data['weight']
            if (dst, edge_label) not in weights:
                weights[(dst, edge_label)] = 0
            weights[(dst, edge_label)] += weight
            data['time'] = graph_index
    
    # Normalize the edge weights
    for graph in graphs:
        for src, dst, data in graph.edges(data = True):
            edge_label = data['label']
            if weights[(dst, edge_label)] > 0:
                data['weight'] = data['weight'] / weights[(dst, edge_label)]
            data['weight'] = np.float32(data['weight'])
    
   
    # Compute the mean embedding for each node in node_embeddings
    for node_id in node_embeddings.keys():
        node_embeddings[node_id] = np.concatenate(node_embeddings[node_id])

    G = nx.compose_all(graphs)
    
    if not hetero:
        G = merge_edges(G)
    # Set the 'features' attribute of each node in the merged graph
    nx.set_node_attributes(G, {node_id: {'features': embeddings} for node_id, embeddings in node_embeddings.items()})
    
    return G

def merge_text_from_user(ego_graphs: List[nx.Graph]) -> Data:
    """
    Combines text embeddings from multiple ego-centric graphs from a user.

    Returns:
        A single list that combines all text embeddings from all ego-centric graphs.
    """
    # Dictionary to store node embeddings for each node id
    #user_ids = []
    user_embeddings = []
    #friend_ids = []
    friend_embeddings = []
    user_tweets_timestamps = []
    friend_tweets_timestamps = []
    
    
    # List to store all ego-centric graphs
    # Loop over each ego-centric graph in the list
    for graph_index, graph in enumerate(ego_graphs):
        # Loop over each node in the graph
        for node_id, data in graph.nodes(data=True):
            # Add the node features to the node_embeddings dictionary
            if data['label'] != -100:
                #user_ids.append(node_id)
                user_embeddings.append(data['features'])
                if 'tweets_created_at' in data:
                    user_tweets_timestamps.append(data['tweets_created_at'])
            else:
                #friend_ids.append(node_id)
                friend_embeddings.append(data['features'])
                if 'tweets_created_at' in data:
                    friend_tweets_timestamps.append(data['tweets_created_at'])
            
    # Compute the mean embedding for each node in node_embeddings
    user_embeddings = np.concatenate(user_embeddings, axis = 0)
    user_embeddings = torch.Tensor(user_embeddings)
    if len(friend_embeddings) > 0:
        friend_embeddings = np.concatenate(friend_embeddings, axis = 0)
    friend_embeddings = torch.Tensor(friend_embeddings)
    if len(user_tweets_timestamps) > 0:
        user_tweets_timestamps = np.concatenate(user_tweets_timestamps, axis = 0)
    user_tweets_timestamps = torch.Tensor(user_tweets_timestamps.astype(np.float32))
    if len(friend_tweets_timestamps) > 0:
        friend_tweets_timestamps = np.concatenate(friend_tweets_timestamps, axis = 0)
    else:
        friend_tweets_timestamps = np.array([])
    friend_tweets_timestamps = torch.Tensor(friend_tweets_timestamps.astype(np.float32))

    # Sort if time is provided
    if len(user_tweets_timestamps) > 0:
        sort_indices = np.argsort(user_tweets_timestamps)
        user_embeddings = user_embeddings[sort_indices]
        user_tweets_timestamps = user_tweets_timestamps[sort_indices]

    if len(friend_tweets_timestamps) > 0:
        sort_indices = np.argsort(friend_tweets_timestamps)
        friend_embeddings = friend_embeddings[sort_indices]
        friend_tweets_timestamps = friend_tweets_timestamps[sort_indices]
    # Compose all ego-centric graphs into a single graph
    data = Data(user_embeddings = user_embeddings, user_tweets_timestamps = user_tweets_timestamps, friend_embeddings = friend_embeddings, friend_tweets_timestamps = friend_tweets_timestamps)
    return data

def merge_ego_graphs_from_users(ego_graphs: List[nx.Graph], undirected: bool = True) -> nx.Graph:
    """
    Combines multiple ego-centric graphs into a single graph.

    Args:
        ego_graphs: A list of ego-centric graphs, where each graph represents the connections
            of a single user to their neighbors.

    Returns:
        A single graph that combines all ego-centric graphs into a big whole graph.
    """
    # Dictionary to store node embeddings for each node id
    node_embeddings = defaultdict(list)
    
    # List to store all ego-centric graphs
    graphs = []

    # Loop over each ego-centric graph in the list
    for graph_index, graph in enumerate(ego_graphs):
        # Loop over each node in the graph
        for node_id, data in graph.nodes(data=True):
            # Add the node features to the node_embeddings dictionary
            node_embeddings[node_id].append(data['features'])
            # Set the 'group' attribute of the node to the index of the graph in ego_graphs
            graph.nodes[node_id]['group'] = graph_index
        # Add the graph to the list of all ego-centric graphs
        graphs.append(graph)
    
    # Compute the mean embedding for each node in node_embeddings
    for node_id in node_embeddings.keys():
        node_embeddings[node_id] = np.concatenate(node_embeddings[node_id], axis = 1)
    # Compose all ego-centric graphs into a single graph
    G = nx.compose_all(graphs)
    
    # Set the 'features' attribute of each node in the merged graph
    nx.set_node_attributes(G, {node_id: {'features': np.mean(embeddings, axis = 0)} for node_id, embeddings in node_embeddings.items()})

    # Convert the merged graph to a PyTorch Geometric object and return it
    #G = to_geometric_data(G, undirected = undirected)
    return G

def to_geometric_data(nx_graph, undirected = False):
    G = torch_geometric.utils.from_networkx(nx_graph)
    # Convert the graph to undirected graph. This step also includes the coalesce function that removes duplicate edges.
    #
    if undirected:
        G.edge_index, G.weight = torch_geometric.utils.to_undirected(G.edge_index, G.weight)
    return G

def merge_hyper_graphs_from_users(hyper_graphs):
    for graph_index, graph in enumerate(hyper_graphs):
        for node_id, data in graph.nodes(data = True):
            data['group'] = graph_index
    G = nx.compose_all(hyper_graphs)
    #G = to_geometric_data(G, undirected = False)
    return G

def merge_hetero_graphs_from_users(graphs):
    for graph_index, graph in enumerate(graphs):
        for node_id, data in graph.nodes(data = True):
            data['group'] = graph_index
    G = nx.compose_all(graphs)
    G = to_geometric_data(G)
    hetero_data = HeteroData()
    hetero_data['user'].x = G['features']
    hetero_data['period_id'] = G.period_id
    hetero_data['group'] = G.group
    hetero_data['label'] = G.label
    hetero_data['edge_index'] = G.edge_index
    edge_index = G.edge_index
    edge_weight = G.weight
    edge_relation = np.array(G.edge_label)
    relations = np.unique(edge_relation)
    for relation in relations:
        indices = np.where(edge_relation == relation, True, False)
        relation_edge_index = edge_index[:, indices]
        hetero_data[relation].edge_index = relation_edge_index
        hetero_data[relation].edge_weight = edge_weight[indices].float()
    return hetero_data
