import torch
import pandas as pd
import numpy as np
import networkx as nx
import pickle
import json
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from utils.util import clean_tweet
from datetime import datetime
NLP_MODEL = "cardiffnlp/twitter-roberta-base-2022-154m"

#TWEET_INDEX = 
INDEX_COLUMN = 'tweet_id'
USER_ID_COLUMN = 'author_id'
CREATED_AT_COLUMN = 'created_at'
WORD_COUNT_COLUMN = 'word_count'
LABLE_COLUMN = 'labels'
TEXT_COLUMN = 'text'
MONTH_INDEX = 'month_id'
PERIOD_INDEX = 'period_id'
MINIMUM_WORDS_IN_POST = 5
INTERVAL_IN_MONTHS = 3
MINIMUM_POSTS_PER_PERIOD = 6
MINIMUM_NUM_PERIODS = 8
CACHE_NEEDS_UPDATE = False
NUM_TWEETS_PER_PERIOD = [2, 4, 6]
MAXIMUM_POSTS_PER_PERIOID = max(NUM_TWEETS_PER_PERIOD)
MAX_PERIOD_LENGTH = 8
from transformers import AutoTokenizer
from transformers import AutoModel
device = 'cuda'
TOKENIZER = AutoTokenizer.from_pretrained(NLP_MODEL)
MODEL = AutoModel.from_pretrained(NLP_MODEL)
MODEL.eval()

import sqlite3

def connect_db(db):

    # Connect to an existing database or create a new one (e.g., "mydatabase.db")
    conn = sqlite3.connect(db)

    return conn
def create_db_table(conn):

    cursor = conn.cursor()
    # Example: Create a "tweet" table
    cursor.execute('''CREATE TABLE IF NOT EXISTS tweet (
                        tweet_id TEXT PRIMARY KEY,
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

def retrieve_embedding_from_tweet_cache(tweet_ids):
    outputs = []
    for tweet_id in tweet_ids:
        embedding = find_embeddings(conn, tweet_id)
        assert(embedding is not None)
        outputs.append(embedding)
    features_mean = np.array(outputs)
    return np.float32(features_mean)


def generate_embedding_from_tweets(df):
    tdf = df
    exists_cache = []
    for i, row in enumerate(tdf.itertuples()):
        result = find_tweet_id(conn, getattr(row, INDEX_COLUMN))
        if result is not None:
            exists_cache.append(result[0])
        else:
            pass
            #print(f"Search#{i}: {row.user_id} - {row.tweet_id}: {row.tweet}")
    tdf = df[~df[INDEX_COLUMN].isin(exists_cache)]
    tweets_embeddings_list = []
    tweets_remaining = tdf[TEXT_COLUMN].values.tolist()
    BATCH_SIZE = 64
    while len(tweets_remaining) > 0:
        tweets_to_be_encoded = tweets_remaining[:BATCH_SIZE]
        inputs = TOKENIZER(tweets_to_be_encoded, return_tensors='pt', padding = True, truncation = True, max_length = 256)
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

    id_to_index = {}
    id_counter = 0
    for i, row in enumerate(tdf.itertuples()):
        user_id = getattr(row, USER_ID_COLUMN)
        tweet_id = getattr(row, INDEX_COLUMN)
        tweet = getattr(row, TEXT_COLUMN)
        if user_id not in id_to_index:
            id_to_index[user_id] = id_counter
            id_counter += 1
        group = id_to_index[user_id]

        print(f"Update#{i}: {user_id} - {tweet_id}: {tweet} {tweets_embeddings_list[i].shape}")
        update_embeddings(
            conn = conn,
            user_id = user_id,
            group = group,
            tweet_id = tweet_id,
            tweet = tweet,
            tweet_embeddings = tweets_embeddings_list[i].astype(np.float32).tobytes(),
            created_at = row.created_at.to_pydatetime()
        )

data_base_dir = Path(f"./data")
cache_base_dir = Path(f"./cache")
cache_base_dir.mkdir(parents=True, exist_ok=True)

def sanitize_filename(filename):
    # Replace characters that are not allowed in filenames with underscores
    return ''.join(char if char.isalnum() or char in ('-', '_', '.') else '_' for char in filename)
                   
cache_json_file = cache_base_dir / f'{sanitize_filename(NLP_MODEL)}.json'

conn = connect_db(cache_base_dir /"mydatabase.db")
create_db_table(conn)

def build_graphs(user_id, group, dfs, num_tweets_per_period):
    graphs = []
    for df in dfs:
        G = nx.MultiDiGraph()
        df = df[:num_tweets_per_period]
        tweet_indices = list(df[INDEX_COLUMN])
        user_embeddings = retrieve_embedding_from_tweet_cache(tweet_indices)
        tweets_created_at = list(df[CREATED_AT_COLUMN].values)
        G.add_node(user_id, tweets_id = tweet_indices, features = user_embeddings, label = group, tweets_created_at = tweets_created_at)
        G.add_edge(user_id, user_id, label = 'mention', weight = 1)
        G.add_edge(user_id, user_id, label = 'reply', weight = 1)
        G.add_edge(user_id, user_id, label = 'quote', weight = 1)
        graphs.append(G)
    return graphs
def save_graphs(graphs, user_id, group, num_tweets_per_period, period_length):
    base_dir = data_base_dir / sanitize_filename(NLP_MODEL) / f'ut_{num_tweets_per_period}_mnf_{4}_p_{INTERVAL_IN_MONTHS}_l_{period_length}' / group / user_id
    save_dir = Path(base_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    for period_id, graph in enumerate(graphs):
        filename = f'{period_id}.pickle'
        filepath = save_dir / filename
        with open(filepath, 'wb') as f:
            pickle.dump(graph, f, protocol=pickle.DEFAULT_PROTOCOL)
    print(f'{group} | {len(graphs)} graphs, {user_id}')

for group in ['depressed_group', 'control_group']:
    df = pd.read_csv(data_base_dir / f'{group}_tweets.csv')
    df = df.dropna()
    df[WORD_COUNT_COLUMN] = df[TEXT_COLUMN].str.split().apply(len)
    df = df[df[WORD_COUNT_COLUMN] > MINIMUM_WORDS_IN_POST - 1]
    df[INDEX_COLUMN] = df.index
    df[INDEX_COLUMN] = df.apply(lambda x: f'{group}_{x[USER_ID_COLUMN]}_{x[INDEX_COLUMN]}', axis = 1)
    df[CREATED_AT_COLUMN] = pd.to_datetime(df[CREATED_AT_COLUMN], utc=True)
    counter = 0
    total = 0
    for (user_id), user_df in df.groupby(USER_ID_COLUMN):
        total += 1
        max_created_at = max(user_df[CREATED_AT_COLUMN])
        user_df[PERIOD_INDEX] = user_df[CREATED_AT_COLUMN].apply(lambda x: max_created_at - x).dt.days // (30 * INTERVAL_IN_MONTHS)
        user_df = user_df.sort_values(PERIOD_INDEX)
        qualified_dfs = {i:[] for i in user_df[PERIOD_INDEX].unique()}
        backup_dfs = {i:[] for i in user_df[PERIOD_INDEX].unique()}
        later_dfs = []
        for (pid), peroid_df in user_df.groupby(PERIOD_INDEX, sort=True):
            offset = 0
            qualified_dfs[pid].append(peroid_df[:MAXIMUM_POSTS_PER_PERIOID])
            backup_dfs[pid].append(peroid_df[MAXIMUM_POSTS_PER_PERIOID:])
            if pid >= 4:
                later_dfs.append(peroid_df[:MAXIMUM_POSTS_PER_PERIOID])
        
        new_dfs = []
        for i in range(4):
            flag = False
            if i in qualified_dfs and len(qualified_dfs[i]) > 0:
                difference = MAXIMUM_POSTS_PER_PERIOID - len(qualified_dfs[i][0])
                if difference == 0:
                    new_dfs.append(qualified_dfs[i][0])
                else:
                    flag = True
            else:
                flag = True
            if flag:
                if i in qualified_dfs and i < MAX_PERIOD_LENGTH - 1 and (i + 1) in backup_dfs and len(backup_dfs[i + 1]) > difference:
                    new_dfs.append(pd.concat([qualified_dfs[i][0], backup_dfs[i + 1].sample(frac=1)[:difference]]))
                else:
                    break
        print(counter, total, user_id, len(new_dfs), '/', MAX_PERIOD_LENGTH, user_df[PERIOD_INDEX].unique())
        if len(new_dfs) < 4 or len(later_dfs) == 0:
            continue
        new_dfs.extend(later_dfs)
        counter += 1

        qualified_dfs = new_dfs
        qualified_user_df = pd.concat(new_dfs)



        #print('qualified_user_df[INDEX_COLUMN]', qualified_user_df[INDEX_COLUMN])
        qualified_user_df[TEXT_COLUMN] = qualified_user_df[TEXT_COLUMN].apply(clean_tweet)
        generate_embedding_from_tweets(qualified_user_df)
        total_graphs = []
        min_num_graphs = 10000
        for num_tweets_per_period in NUM_TWEETS_PER_PERIOD:
            graphs = build_graphs(user_id, group, qualified_dfs, num_tweets_per_period)

            min_num_graphs = min(min_num_graphs, len(graphs))
            total_graphs.append((user_id, graphs, group, num_tweets_per_period))

            
        for user_id, graphs, group, num_tweets_per_period in total_graphs:
            tids = []
            #Use the closet result
            #selected_graphs = graphs[-min(MAX_PERIOD_LENGTH, min_num_graphs):]
            selected_graphs = graphs[:min(MAX_PERIOD_LENGTH, min_num_graphs) + 4]
            print("graphs", len(selected_graphs), "/", len(graphs))
            for graph in selected_graphs:
                for id, data in graph.nodes(data=True):
                    for tid in data['tweets_id']:
                        tids.append(tid)
            tdf = df[df[INDEX_COLUMN].isin(tids)]
            #generate_embedding_from_tweets(tdf)
            for period_length in range(2, 5):
                save_graphs(selected_graphs[:period_length], user_id, group, num_tweets_per_period, period_length)
            for period_length in [-1, -2, -3, -4]:
                save_graphs(selected_graphs[-period_length:-period_length + 4], user_id, group, num_tweets_per_period, period_length)
        save_dir = data_base_dir / group / user_id
        save_dir.mkdir(parents=True, exist_ok=True)
        file_path = save_dir / 'data.csv'
        qualified_user_df.to_csv(file_path)
    #print(df)
    #print(df[CREATED_AT_COLUMN])
"""
if CACHE_NEEDS_UPDATE:
    with open(cache_json_file, 'w') as cache_file:
        dumped = json.dumps(tweets_cache, cls=NumpyEncoder)
        json.dump(dumped, cache_file)
"""