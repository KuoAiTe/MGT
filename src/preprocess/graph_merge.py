
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

NLP_MODEL = "cardiffnlp/twitter-roberta-base-2022-154m"
CACHE_BASE_DIR = Path(f"./cache")
CACHE_BASE_DIR.mkdir(parents=True, exist_ok=True)
SAVE_BASE_DIR = Path(f"./graph_data")
NUM_TWEETS_PER_PERIOD = [8, 4, 2]
NUM_FRIENDS_PER_PERIOD = [8, 4, 2]
MIN_PERIOD_LENGTH = 2
MAX_PERIOD_LENGTH = 8
BATCH_SIZE = 64
TOTAL_UPDATES = 0

NLP_MODEL = sanitize_filename(NLP_MODEL)

#tweets_cache = read_embeddings_cache(CACHE_BASE_DIR, NLP_MODEL)
#tweets_cache = {}
group_types = ["depressed_group", "control_group"]
columns = ['user_id', 'group', 'period_id', 'tweet_id', 'created_at', 'mention_count', 'reply_count', 'quote_count', 'total_count', 'embeddings']

friend_selected = {}

PERIOD_LENGTH = 10
MAX_NUM_SNAPSHOTS = [4, 6, 8, -1, -2, -4]

def process_users_for_group(max_num_snapshots, group_types):
    for group_type in group_types:
        
        base_path = f'graph_data/nov14/*/{group_type}/*'
        file_list = sorted(Path().glob(base_path))
        for base_path in file_list:

            graphs = []
            for file_path in sorted(Path().glob(f'{base_path}/*.pickle')):
                result = re.search(r'ut_(\d+)_mnf_(\d+)_p_(\d+)_l_10\/(control_group|depressed_group)\/([0-9]+)\/(\d+).pickle', str(file_path))
                if result == None: continue
                result = result.groups()
                num_tweets_per_period = result[0]
                num_friends_per_period = result[1]
                interval_in_months = result[2]
                group = result[3]
                user_id = result[4]
                period_id = int(result[5])
    
                base_dir = SAVE_BASE_DIR / NLP_MODEL /f'ut_{num_tweets_per_period}_mnf_{num_friends_per_period}_p_{max_num_snapshots}_l_{PERIOD_LENGTH}'
                destination_file_path = None
                if max_num_snapshots > 0 and period_id < max_num_snapshots:
                    destination_file_path = base_dir / group / user_id / f'{period_id}.pickle'
                    # Copy source to destination
                    
                elif max_num_snapshots < 0:
                    new_period_id = period_id + max_num_snapshots
                    if new_period_id >= 0:
                        destination_file_path = base_dir / group / user_id / f'{new_period_id}.pickle'
                if destination_file_path is not None:
                    print(file_path, destination_file_path)
                    destination_file_path.mkdir(parents=True, exist_ok=True)
                    shutil.copy(file_path, 'destination_file_path')
                    

if __name__ == "__main__":
    import shutil
    group_types = ["depressed_group", "control_group"]
    for max_num_snapshots in MAX_NUM_SNAPSHOTS:
        process_users_for_group(max_num_snapshots, group_types)
    
    #write_embeddings_cache(tweets_cache, CACHE_BASE_DIR, NLP_MODEL)
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
