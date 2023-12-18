import re
import glob
import networkx as nx
import numpy as np
from itertools import product
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from ..dataset import MedDataset

def get_train_test_data(data, test_size, random_state = 44):
    stratify = [_['label'] for _ in data]
    train_data, test_data = train_test_split(data, test_size = test_size, stratify = stratify, random_state = random_state)
    return train_data, test_data

def get_dataloader(dataset, **kwargs):
    return DataLoader(
        dataset = dataset,
        collate_fn = MedDataset.collate_fn,
        **kwargs
    )
    
def get_db_location(data_info):
    base = f'{data_info.dataset_location}/data/{data_info.dataset_name}/ut_{data_info.num_tweets_per_period}_mnf_{data_info.max_num_friends}_p_{data_info.periods_in_months}_l_{data_info.period_length}'
    return f'{base}/*/*/*.pickle'

def load_files(data_info):
    files = []
    user_id_set = []
    file_location = get_db_location(data_info)
    pattern = re.compile('(control_group|depressed_group)\/(\w+)\/(\w{1,2})\.pickle')
    file_paths = sorted(glob.glob(file_location, recursive = True))
    for file_path in file_paths:
        m = pattern.findall(file_path)
        if len(m) != 0:
            group, user_id, period_id = m[0]
            label = 0 if group == 'control_group' else 1
            if user_id not in user_id_set:
                user_id_set.append(user_id)
            files.append({'file': file_path, 'label': label, 'user_original_id': user_id, 'user_id': user_id_set.index(user_id), 'period_id': int(period_id)})
    return files



def process_graph(G, user_id, period_id):
    """Process the graph and update node and edge attributes."""
    for node_id, data in G.nodes(data = True):
        if data['label'] == 'friend':
            nx.set_node_attributes(G, {node_id: {'period_id': period_id, 'label': -100}})
        elif data['label'] == 'depressed_group' or data['label'] == 'depression_group':
            nx.set_node_attributes(G, {node_id: {'period_id': period_id, 'label': 1}})
        elif data['label'] == 'control_group':
            nx.set_node_attributes(G, {node_id: {'period_id': period_id, 'label': 0}})
    
    friend_to_user_edges = {'mention': [], 'reply': [], 'quote': []}
    remove_edges = []
    for src, dst, key, data in G.edges(data=True, keys=True):
        if data['weight'] == 0:
            remove_edges.append((src, dst, key))
        else:
            friend_to_user_edges[data['label']].append((dst, src, key))

    G.remove_edges_from(remove_edges)
    for label, items in friend_to_user_edges.items():
        G.add_edges_from(items, weight=1, label=label)

    G = nx.relabel_nodes(G, lambda x: f'{x}_{user_id}')
    
    return G

def load_data(data_info):
    from platform import python_version
    from packaging.version import Version
    if Version(python_version()) < Version('3.8.0'):
        import pickle5 as pickle
    else:
        import pickle
    
    files = load_files(data_info)
    graphs_by_user = {}
    count = 0
    for row in files:
        count += 1
        file, label, user_id, user_original_id, period_id = row['file'], row['label'], row['user_id'], row['user_original_id'], row['period_id']
        if user_id not in graphs_by_user:
            graphs_by_user[user_id] = {'user_id': user_id, 'user_original_id': user_original_id, 'label': label, 'period_id': [], 'graphs': []}

        with open(file, 'rb') as f:
            G = pickle.load(f)
        G = process_graph(G, user_id, period_id)
        graphs_by_user[user_id]['period_id'].append(period_id)
        graphs_by_user[user_id]['graphs'].append(G)
    
    # Read in reverse order so the latest graph will be the first one.
    # That would be easier to read.
    for value in graphs_by_user.values():
        value['period_id'] = list(value['period_id'])
        value['graphs'] = list(value['graphs'])
    data = np.array(list(graphs_by_user.values()))
    labels = np.array([row['label'] for row in data])
    return data, labels

def get_settings(
        dataset_location,
        dataset_names,
        default_num_tweets,
        num_tweets_per_period_list,
        default_num_friends,
        max_num_friends_list,
        default_interval,
        periods_in_months_list,
        random_state,
        all_permutations = False
    ):
    from ..dataclass import DatasetInfo
    dataset_list = []
    if all_permutations:
        for dataset_name in dataset_names:
            dataset_list.extend(list(
                map(
                    lambda x:
                    DatasetInfo(
                        num_tweets_per_period = str(x[0]),
                        max_num_friends = str(x[1]),
                        periods_in_months = str(x[2]),
                        period_length = str(10),
                        dataset_location = dataset_location,
                        dataset_name = dataset_name,
                        random_state = random_state),
                        product(num_tweets_per_period_list, max_num_friends_list, periods_in_months_list)
                    )
                )
            )
    else:
        for dataset_name in dataset_names:
            dataset_list.extend(list(
                map(
                    lambda x:
                    DatasetInfo(
                        num_tweets_per_period = str(x[0]),
                        max_num_friends = str(x[1]),
                        periods_in_months = str(x[2]),
                        period_length = str(10),
                        dataset_location = dataset_location,
                        dataset_name = dataset_name,
                        random_state = random_state),
                        product([default_num_tweets], max_num_friends_list, [default_interval])
                    )
                )
            )
            periods_in_months_list.remove(default_interval)
            dataset_list.extend(list(
                map(
                    lambda x:
                    DatasetInfo(
                        num_tweets_per_period = str(x[0]),
                        max_num_friends = str(x[1]),
                        periods_in_months = str(x[2]),
                        period_length = str(10),
                        dataset_location = dataset_location,
                        dataset_name = dataset_name,
                        random_state = random_state),
                        product([default_num_tweets], [default_num_friends], periods_in_months_list)
                    )
                )
            )
            num_tweets_per_period_list.remove(default_num_tweets)
            dataset_list.extend(list(
                map(
                    lambda x:
                    DatasetInfo(
                        num_tweets_per_period = str(x[0]),
                        max_num_friends = str(x[1]),
                        periods_in_months = str(x[2]),
                        period_length = str(10),
                        dataset_location = dataset_location,
                        dataset_name = dataset_name,
                        random_state = random_state),
                        product(num_tweets_per_period_list, [default_num_friends], [default_interval])
                    )
                )
            )
            print('length:', len(dataset_list))
    return dataset_list