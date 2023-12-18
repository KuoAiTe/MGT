
import math
import torch
import re
import os
import glob
from tkinter import E
import pandas as pd
import numpy as np
import networkx as nx
import pickle
import json
from datetime import datetime
from collections import defaultdict
from pathlib import Path
from dateutil.relativedelta import relativedelta
from ast import literal_eval
from utils.util import clean_tweet, NumpyEncoder, sanitize_filename, read_embeddings_cache, write_embeddings_cache
import sqlite3

from transformers import AutoTokenizer
from transformers import AutoModel

def connect_db(db):

    # Connect to an existing database or create a new one (e.g., "mydatabase.db")
    conn = sqlite3.connect(db)

    return conn
def create_db_table(conn):

    cursor = conn.cursor()
    # Example: Create a "tweet" table
    cursor.execute('''CREATE TABLE IF NOT EXISTS tweet (
                        tweet_id INTEGER PRIMARY KEY,
                        user_id TEXT,
                        group_id TEXT,
                        tweet TEXT,
                        tweet_embeddings TEXT,
                        created_at TIMESTAMP,
                        last_update TIMESTAMP
                    )''')

    # Commit the changes to save the table structure
    conn.commit()
def find_tweet_id(conn, tweet_id):
    cursor = conn.cursor()
    # Retrieve data from the "users" table
    cursor.execute("SELECT tweet_id FROM tweet WHERE tweet_id = ?", (str(tweet_id), ))
    result = cursor.fetchone()
    return result
def find_embeddings(conn, tweet_id):
    cursor = conn.cursor()
    # Retrieve data from the "users" table
    cursor.execute("SELECT tweet_embeddings FROM tweet WHERE tweet_id = ?", (str(tweet_id), ))
    result = cursor.fetchone()
    if result is not None:
        result = np.frombuffer(result[0], dtype=np.float32)
    
    #print(f"Retrieve embeddings {tweet_id}: {result.shape}")
    return result
def update_embeddings(conn, user_id, group, tweet_id, tweet, tweet_embeddings, created_at):
    cursor = conn.cursor()
    current_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    # Insert data into the "users" table
    cursor.execute("INSERT OR REPLACE INTO tweet (user_id, group_id, tweet_id, tweet, tweet_embeddings, created_at, last_update) VALUES (?, ?, ?, ?, ?, ?, ?)"
                   , (user_id, group, tweet_id, tweet, tweet_embeddings, created_at, current_timestamp))
    # Commit the changes
    conn.commit()
NLP_MODEL = "cardiffnlp/twitter-roberta-base-2022-154m"
CACHE_BASE_DIR = Path(f"./cache")
CACHE_BASE_DIR.mkdir(parents=True, exist_ok=True)
SAVE_BASE_DIR = Path(f"./graph_data")
NUM_TWEETS_PER_PERIOD = [6, 4, 2]
default_num_tweets = 4
NUM_FRIENDS_PER_PERIOD = [6, 4, 2]
default_num_friends = 4
PERIOD_LENGTH = [8, 4, 2]
default_period_length = 4
NUM_SNAPSHOTS_AHEAD = [-4, -2, -1]


NUM_TWEETS_PER_PERIOD = [4]
NUM_FRIENDS_PER_PERIOD = [4]
PERIOD_LENGTH = [4]
NUM_SNAPSHOTS_AHEAD = []

default_num_tweets = 4
default_num_friends = 4
default_period_length = 4


MAX_PERIOD_LENGTH = max(PERIOD_LENGTH)
BATCH_SIZE = 128
TOTAL_UPDATES = 0

TOKENIZER = AutoTokenizer.from_pretrained(NLP_MODEL)
MODEL = AutoModel.from_pretrained(NLP_MODEL).to('cuda:0')
MODEL.eval()
print(MODEL.device)
NLP_MODEL = sanitize_filename(NLP_MODEL)

conn = connect_db(CACHE_BASE_DIR /"mydatabase.db")
create_db_table(conn)

#tweets_cache = read_embeddings_cache(CACHE_BASE_DIR, NLP_MODEL)
#tweets_cache = {}
group_types = ["depressed_group", "control_group"]
columns = ['user_id', 'group', 'period_id', 'tweet_id', 'created_at', 'mention_count', 'reply_count', 'quote_count', 'total_count', 'embeddings']

friend_selected = {}

INTERVALS_IN_MONTHS = [3]
def save_graphs(graphs, user_id, group_type, num_tweets_per_period, num_friends_per_period, interval_in_months, period_length):
    base_dir = SAVE_BASE_DIR / NLP_MODEL /f'ut_{num_tweets_per_period}_mnf_{num_friends_per_period}_p_{interval_in_months}_l_{period_length}' / group_type / str(user_id)
    base_dir.mkdir(parents=True, exist_ok=True)
    if period_length > 0:
        graphs = graphs[:period_length]
    else:
        graphs = graphs[-period_length:]
        graphs = graphs[:default_period_length]
    for period_id, graph in enumerate(graphs):
        filename = f'{period_id}.pickle'
        filepath = base_dir / filename
        nodes = list(graph.nodes())
        user_node = nodes[0:1]
        friend_nodes = nodes[1:]
        friend_nodes = friend_nodes[-num_friends_per_period:]
        retain_nodes = user_node + friend_nodes
        G = nx.MultiGraph()
        for user_id, data in graph.nodes(data=True):
            if user_id in retain_nodes:
                G.add_node(user_id, **data)
        for src, dst, data in graph.edges(data=True):
            if src in retain_nodes and dst in retain_nodes:
                G.add_edge(src, dst, **data)
        

        for node, data in G.nodes(data=True):
            #replace embeddings placeholder
            data['features'] = retrieve_embedding_from_tweet_cache(data['tweets_id'])
            for key in ['features', 'tweets_id', 'tweets_content', 'tweets_created_at']:
                data[key] = data[key][:num_tweets_per_period]
            print(f'P#{period_id}', node)
            for i, tweet in enumerate(data['tweets_content']):
                print(f'P#{period_id} #{i}: {tweet}')
            print('\n\n\n')

        with open(filepath, 'wb') as f:
            pickle.dump(G, f, protocol=pickle.DEFAULT_PROTOCOL)
        
        with open(filepath, 'rb') as f:
            G = pickle.load(f)
            #print(list(G.nodes()))
            assert(list(G.nodes())[0].startswith('user'))
    print(f'{group_type} | {len(graphs)} graphs ({num_tweets_per_period}, {num_friends_per_period}, {interval_in_months}, {period_length}), {user_id}')

def process_file(file_path, interval_in_months):
    m = re.search(r'([0-9]+)\.csv', str(file_path))
    if m is None:
        raise ValueError("Filename does not match expected pattern.")

    user_id = int(m.group(1))
    user_node_indices = {user_id: 0}  # Assuming this should be reset for each file
    user_node_id = f'user_{user_node_indices[user_id]}'

    df = pd.read_csv(file_path, header=0, index_col=0, engine='python')
    df = df.dropna()
    df['cleantweet'] = df['tweet'].apply(clean_tweet)
    df['word_count'] = df['cleantweet'].str.count(' ') + 1
    df = df[df['word_count'] > 5]
    df = df[df['word_count'] < 30]
    
    df['created_at'] = pd.to_datetime(df['created_at'])
    df['year'] = df['created_at'].dt.year
    df['month'] = df['created_at'].dt.month
    max_created_at = max(df['created_at'])
    df['period_id'] = df['created_at'].apply(lambda x: max_created_at - x).dt.days // (30 * interval_in_months)
    #df['period_id'] = ((df['year'] - min(df['year'])) * 12 + df['month']) // interval_in_months
    """
    period_size = int(math.ceil(len(df.index) // interval_in_months))
    tdf = df[df['user_id'].astype(str) == df['group'].astype(str)]
    period_size = len(tdf) // interval_in_months
    index_ = tdf.iloc[[i * period_size for i in range(interval_in_months)]].index.values
    def find_in(x):
        for i in range(len(index_)):
            if index_[i] > x:
                return i - 1
        return len(index_) - 1
    df['index'] = df.index
    df['period_id'] = df['index'].apply(find_in)
    """
    return user_id, user_node_id, df


def generate_embedding_from_tweets(df):
    tdf = df
    exists_cache = []
    for i, row in enumerate(tdf.itertuples()):
        result = find_tweet_id(conn, row.tweet_id)
        if result is not None:
            exists_cache.append(result[0])
        else:
            pass
            #print(f"Search#{i}: {row.user_id} - {row.tweet_id}: {row.tweet}")
    tdf = df[~df['tweet_id'].isin(exists_cache)]
    tweets_embeddings_list = []
    tweets_remaining = tdf['cleantweet'].values.tolist()
    while len(tweets_remaining) > 0:
        tweets_to_be_encoded = tweets_remaining[:BATCH_SIZE]
        inputs = TOKENIZER(tweets_to_be_encoded, return_tensors='pt', padding = True, truncation = True, max_length = 256)
        for key in inputs.keys():
            inputs[key] = inputs[key].to('cuda:0')
        with torch.no_grad():
            encoded_outputs = MODEL(**inputs).last_hidden_state
        input_mask_expanded = inputs['attention_mask'].unsqueeze(-1).expand(encoded_outputs.size()).float()
        sum_embeddings = torch.sum(encoded_outputs * input_mask_expanded, 1)
        sum_mask = input_mask_expanded.sum(1)
        sum_mask = torch.clamp(sum_mask, min = 1e-9)
        tweets_embeddings = sum_embeddings / sum_mask
        tweets_embeddings_list.append(tweets_embeddings)
        tweets_remaining = tweets_remaining[BATCH_SIZE:]
    if len(tweets_embeddings_list) > 0:
        tweets_embeddings_list = torch.cat(tweets_embeddings_list).detach().cpu().numpy()
    #tdf['tweet_embeddings'] = tweets_embeddings_list

    for i, row in enumerate(tdf.itertuples()):
        print(f"Update#{i}: {row.user_id} - {row.tweet_id}: {row.tweet} {tweets_embeddings_list[i].shape}")
        update_embeddings(
            conn = conn,
            user_id = row.user_id,
            group = row.group,
            tweet_id = row.tweet_id,
            tweet = row.tweet,
            tweet_embeddings = tweets_embeddings_list[i].astype(np.float32).tobytes(),
            created_at = row.created_at.to_pydatetime()
        )


def create_graph(user_id, user_node_id, group_type, period_id, df, used_tweets, friend_used_tweets, user_tweet_choice_dict, friend_used, friend_choice_dict, friend_node_indices, num_tweets_per_period, num_friends_per_period, offset):
    period_df = df[df['period_id'] == period_id]
    user_df = period_df[period_df['user_id'] == user_id]
    #random shuffle
    G = nx.MultiDiGraph()
    if (user_id, period_id) not in user_tweet_choice_dict:
        #user_df = user_df.sample(frac=1)
        user_df = user_df.sort_values(by="word_count", ascending = False)
        choice = user_df['tweet_id'].values
        user_tweet_choice_dict[(user_id, period_id)].extend(choice)
    tweet_choice = user_tweet_choice_dict[(user_id, period_id)][num_tweets_per_period * offset:]
    user_df = df[(df['tweet_id'].isin(tweet_choice)) & (~df['tweet_id'].isin(used_tweets))]

    if len(user_df) != num_tweets_per_period:
        previous_period_df = df[(df['period_id'] > period_id) & (~df['tweet_id'].isin(used_tweets))].sort_values(by="period_id", ascending = True).sort_values(by="period_id", ascending = True)
        previous_user_df = previous_period_df[(previous_period_df['user_id'] == user_id) & ~(previous_period_df['tweet_id'].isin(tweet_choice))]
        previous_user_df = previous_user_df[num_tweets_per_period * offset:]
        new_tweet_choice = previous_user_df['tweet_id'].values.tolist()
        user_df = pd.concat([user_df, previous_user_df])
        tweet_choice.extend(new_tweet_choice)
        user_df = df[(df['tweet_id'].isin(tweet_choice))]
        
    else:
        user_df = df[(df['tweet_id'].isin(tweet_choice))]
        #continue
    user_df = user_df[:num_tweets_per_period]
    difference = num_tweets_per_period - len(user_df)
    if len(user_df) < num_tweets_per_period:
        return None

    assert(len(user_df) == num_tweets_per_period)
    #user_df = period_df[is_user][:num_tweets_per_period]
    period_df = df[df['period_id'] >= period_id & ~(df['user_id'] == user_id)]
    period_df = period_df[period_df['total_count'] > 0]
    period_df = period_df.sort_values(by="period_id", ascending = True)
    f_choice = period_df['user_id'].unique().tolist()

    friend_dfs = []
    for f_period_id in sorted(period_df['period_id'].unique()):
        if f_period_id < period_id: continue
        f_df = period_df[period_df['period_id'] == f_period_id]
        f_df = f_df.sort_values(by="total_count", ascending = False)
        friend_dfs.append(f_df)
    if len(friend_dfs) == 0: return None
    friend_df = pd.concat(friend_dfs)

    user_embeddings = None
    user_tweets_id = list(user_df['tweet_id'].values)
    user_tweets_content = list(user_df['cleantweet'].values)
    user_tweets_created_at = list(user_df['created_at'].values)
    
    G.add_node(user_node_id, user_id = user_id, features = user_embeddings, tweets_id = user_tweets_id, tweets_content = user_tweets_content, tweets_created_at = user_tweets_created_at, label = group_type)
    
    if (user_id, period_id) not in friend_choice_dict:
        choice = list(friend_df['user_id'].unique())
        friend_choice_dict[(user_id, period_id)].extend(choice)
    #print(choice, len(friend_choice_dict))
    #print(user_df, friend_df)
    friend_choice = friend_choice_dict[(user_id, period_id)]
    
    #print('friend_rank_df', graph_index, friend_df['user_id'].unique())
    #print('?', friend_choice)
    #print('')
    friend_df = friend_df[(friend_df['user_id'].isin(friend_choice)) & ~(friend_df['user_id'].isin(friend_used[period_id]))]
    temp_friend_used = []
    friend_counter = 0
    friend_df = friend_df[~(friend_df['tweet_id'].isin(friend_used_tweets))]
    for friend_id in f_choice:
        group_df = friend_df[friend_df['user_id'] == friend_id]
        if len(group_df) == 0: continue
        new_friend_df = group_df.sort_values(by="period_id", ascending = True)
        
        #new_friend_df = group_df.sample(frac=1).sort_values(by="period_id", ascending = True)
        new_friend_df = new_friend_df[~(new_friend_df['tweet_id'].isin(friend_used_tweets))]
        new_friend_df = new_friend_df[:num_tweets_per_period]
        friend_node_id = f'friend_{friend_node_indices.setdefault(friend_id, len(friend_node_indices))}'
        friend_embeddings = None
        difference = num_tweets_per_period - len(new_friend_df)
        if difference != 0:
            continue
        assert(len(new_friend_df) == num_tweets_per_period)
        
        friend_tweets_id = list(new_friend_df['tweet_id'].values)
        friend_tweets_content = list(new_friend_df['cleantweet'].values)
        friend_tweets_created_at = list(new_friend_df['created_at'].values)
        friend_used_tweets.extend(friend_tweets_id)
        G.add_node(friend_node_id, user_id = friend_id, features = friend_embeddings, tweets_id = friend_tweets_id, tweets_content = friend_tweets_content, tweets_created_at = friend_tweets_created_at, label = 'friend')
        for relation in ['mention', 'reply', 'quote']:
            mean = new_friend_df[f'{relation}_count'].mean()
            if mean != 0:
                G.add_edge(user_node_id, friend_node_id, label = relation, weight = mean)
                #G.add_edge(friend_node_id, user_node_id, label = relation, weight = mean)
        friend_counter += 1
        temp_friend_used.append(friend_id)
        if friend_counter >= num_friends_per_period:
            break
    if friend_counter < num_friends_per_period:
        return None
    if G.number_of_edges() == 0:
        return None
    used_tweets.extend(user_tweets_id)
    used_tweets.extend(friend_used_tweets)
    friend_used[period_id].extend(temp_friend_used)

    return G


def build_graphs(df, user_id, user_node_id, group_type, user_node_indices, user_tweet_choice_dict, friend_node_indices, friend_choice_dict, num_tweets_per_period, num_friends_per_period):
    total_graphs = []
    period_indices = sorted(df['period_id'].unique())
    count = 0
    used_tweets = []
    friend_used_tweets = []
    friend_used = {i: [] for i in range(100)}
    for graph_index, period_id in enumerate(period_indices):
        G = create_graph(user_id, user_node_id, group_type, period_id, df, used_tweets, friend_used_tweets, user_tweet_choice_dict, friend_used, friend_choice_dict, friend_node_indices, num_tweets_per_period, num_friends_per_period, offset = 0)
        if G is not None:
            total_graphs.append((period_indices[graph_index], G))

    offset = {i:0 for i in range(100)}
    if len(total_graphs) > 0 and len(total_graphs) != MAX_PERIOD_LENGTH:
        while True and len(total_graphs) < MAX_PERIOD_LENGTH:
            insert_happens = False
            for i in reversed(range(len(total_graphs))):
                period_id = total_graphs[i][0]
                offset[period_id] += 1
                graph = create_graph(user_id, user_node_id, group_type, period_id, df, used_tweets, friend_used_tweets, user_tweet_choice_dict, friend_used, friend_choice_dict, friend_node_indices, num_tweets_per_period, num_friends_per_period, offset = offset[period_id])
                if graph is not None:
                    total_graphs.insert(i, (period_id, graph))
                    insert_happens = True
            if not insert_happens:
                break
    total_graphs = [_[1] for _ in total_graphs]
    difference = MAX_PERIOD_LENGTH - len(total_graphs)
    print(len(period_indices), '/ flattend_', len(total_graphs), 'diff', difference)
    return total_graphs

def retrieve_embedding_from_tweet_cache(tweet_ids):
    outputs = []
    for tweet_id in tweet_ids:
        embedding = find_embeddings(conn, tweet_id)
        assert(embedding is not None)
        outputs.append(embedding)
    features_mean = np.array(outputs)
    return np.float32(features_mean)

def process_users_for_group(interval_in_months, group_types):
    for group_type in group_types:
        base_path = f'temporal_df/ut_20_p_1_mnf10_version3/{group_type}/*.csv'
        counter = 0
        file_list = sorted(Path().glob(base_path))
        for counter2, file_path in enumerate(file_list):
            print(file_path)
            user_id, user_node_id, df = process_file(file_path, interval_in_months)
            user_node_indices = {}
            user_tweet_choice_dict = defaultdict(list)
            friend_node_indices = {}
            friend_choice_dict = defaultdict(list)
            total_graphs = []
            flag = False
            min_num_graphs = 1000
            graphs = build_graphs(df, user_id, user_node_id, group_type, user_node_indices, user_tweet_choice_dict, friend_node_indices, friend_choice_dict, max(NUM_TWEETS_PER_PERIOD), max(NUM_FRIENDS_PER_PERIOD))
            graphs = graphs[:MAX_PERIOD_LENGTH]
            if len(graphs) < MAX_PERIOD_LENGTH:
                continue
            print(f'{counter} {counter2} /{len(file_list)} Processed user {user_id} for group {group_type}')
            
            tids = []
            for i, graph in enumerate(graphs):
                for id, data in graph.nodes(data=True):
                    for tid in data['tweets_id']:
                        tids.append(tid)
                        #print(i, datetime.fromtimestamp(data['tweets_id'][0] // 1000000000))
            tdf = df[df['tweet_id'].isin(tids)]
    
            generate_embedding_from_tweets(tdf)
            
            for num_tweets_per_period in NUM_TWEETS_PER_PERIOD:
                save_graphs(graphs, user_id, group_type, num_tweets_per_period, default_num_friends, interval_in_months, default_period_length)
            for num_friends_per_period in NUM_FRIENDS_PER_PERIOD:
                save_graphs(graphs, user_id, group_type, default_num_tweets, num_friends_per_period, interval_in_months, default_period_length)
            for period_length in PERIOD_LENGTH:
                save_graphs(graphs, user_id, group_type, default_num_tweets, default_num_friends, interval_in_months, period_length)
            for period_length in NUM_SNAPSHOTS_AHEAD:
                save_graphs(graphs, user_id, group_type, default_num_tweets, default_num_friends, interval_in_months, period_length)

            counter += 1
        

count_dict = {'num_users': 0, 'reply_count': 0, 'quote_count': 0, 'mention_count': 0, 'total_count': 0}

if __name__ == "__main__":
    import shutil
    group_types = ["depressed_group"]#, "control_group"]
    for interval in INTERVALS_IN_MONTHS:
        process_users_for_group(interval, group_types)
    
    """
    user_count = {}
    for group_type in group_types:
        for interval_in_months in INTERVALS_IN_MONTHS:
            for num_tweets_per_period in NUM_TWEETS_PER_PERIOD:
                for num_friends_per_period in NUM_FRIENDS_PER_PERIOD:
                    base_dir = SAVE_BASE_DIR / NLP_MODEL /f'ut_{num_tweets_per_period}_mnf_{num_friends_per_period}_p_{interval_in_months}_l_{PERIOD_LENGTH}'

                    base_path = str(base_dir / group_type / '**')
                    counter = 0
                    file_list = sorted(Path().glob(base_path))
                    for file_path in file_list:
                        m = re.search(r'(control_group|depressed_group)\/([0-9]+)', str(file_path))
                        if m is None:
                            continue
                        user_id = m.groups()[1]
                        if user_id not in user_count:
                            user_count[user_id] = 0
                        user_count[user_id] += 1
    for user_id, count in user_count.items():
        if count != 27:
            base_dir = SAVE_BASE_DIR / NLP_MODEL 
            glob_path = f'{base_dir}/**/{user_id}'
            for file in glob.glob(glob_path, recursive=True):
                #pass
                #print(file)
                shutil.rmtree(file)
    """
