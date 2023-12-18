import re
import os
import glob
from tkinter import E
import pandas as pd
import numpy as np
import networkx as nx
import pickle
from datetime import datetime
from pathlib import Path
from dateutil.relativedelta import relativedelta
from collections import defaultdict
from ast import literal_eval
group_types = ["depressed_group", "control_group", ]

columns = ['user_id', 'group', 'period_id', 'tweet_id', 'created_at', 'mention_count', 'reply_count', 'quote_count', 'total_count', 'embeddings']
NLP_MODEL = 'embeddings_twitter-roberta-base-jun2022'
LENGTH_MAX_INTERVALS = 2
INTERVALS_IN_MONTHS = [3, 6, 12]
MAX_FRIENDS = 10
NUM_PERIODS_BEFORE_DIAGNOSIS = 0
INTERPOLATION = False

DATASET_LIST = [
    '/media/aite/easystore/db/df_data/embeddings_nlp_model/20230109/ut_10_p_1_mnf10_version3'
]
def select_active_friends_per_period(file, period_id, friend_df):
    filtered_df = friend_df[friend_df['interaction_count'] > 0]
    friend_selected = {}
    key = f'{file}-{period_id}'
    if key not in friend_selected:
        friend_indices = []
        for friend_id, group_df in filtered_df.groupby('user_id'):
            if len(group_df) >= num_tweets_per_period:
                friend_indices.append(friend_id)
        friend_selected[key] = friend_indices
    friend_indices = friend_selected[key]
    return friend_df[friend_df['user_id'].isin(friend_indices)]
def select_df_by_period(df, period_id, interval_in_months):
    return df[df[f'period_id_{interval_in_months}'] == period_id]
def filter_friend_without_enough_tweets(friend_df):
    friend_rank_df = friend_df.sort_values(by="interaction_count", ascending = False)
    friend_data = []
    for friend_id, group_df in friend_rank_df.groupby('user_id'):
        if len(group_df) < num_tweets_per_period:
            continue
        #print(friend_id, group_df)
        #first_row = group_df.head(1)
        #friend_embeddings = np.float32(np.mean(group_df['embeddings'].sample(n = NUM_TWEETS_PER_PERIOD).values))
        #mention_count = first_row['mention_count'].values[0]
        #reply_count = first_row['reply_count'].values[0]
        #quote_count = first_row['quote_count'].values[0]
        group_df = group_df.sort_values(by="tweet", key=lambda x: -x.str.len())[:num_tweets_per_period]
        friend_data.append(group_df)
    return friend_data
def get_period_id(date_str, interval_in_months):
    if date_str == None: return -1
    date_obj = datetime.strptime(date_str, '%Y-%m-%d %H:%M:%S')
    start_date = datetime(year = 2007, month = 1, day = 1, hour = 0, minute = 0, second = 0)
    month_diff =  12 * (date_obj.year - start_date.year) + (date_obj.month - start_date.month)
    period_id = month_diff // interval_in_months
    return period_id
count_dict = {'num_users': 0, 'reply_count': 0, 'quote_count': 0, 'mention_count': 0, 'total_count': 0}
for refer_base_dir in DATASET_LIST: 
    m = re.search(r'ut_([0-9]+)_p_([0-9]+)_mnf([0-9]+)', refer_base_dir)
    num_tweets_per_period = 5
    for group_type in group_types:
        base_path = f'{refer_base_dir}/{group_type}/*csv'
        user_node_indices = {}
        counter = 0
        size = len(sorted(glob.glob(base_path, recursive = True)))
        t = False
        for file_counter, file in enumerate(sorted(glob.glob(base_path, recursive = True))):
            m = re.search(r'([0-9]+)\.csv', file)
            assert m != None, "m == None"
            user_id = int(m.groups()[0])
            user_node_indices.setdefault(user_id, len(user_node_indices))
            
            user_node_id = f'user_{user_node_indices[user_id]}'
            df = pd.read_csv(file, header = 0, index_col = 0, engine = 'python')
            for interval_in_months in INTERVALS_IN_MONTHS:
                df[f'period_id_{interval_in_months}'] = df['created_at'].apply(get_period_id, interval_in_months = interval_in_months)
            df['interaction_count'] = df['reply_count'] + df['mention_count'] + df['quote_count']
            max_period_index_key = f'period_id_{max(INTERVALS_IN_MONTHS)}'
            #
            period_indices_by_max = df[max_period_index_key].unique()
            #Do not consider less than one year.
            if len(period_indices_by_max) < LENGTH_MAX_INTERVALS:
                continue
            # Check if it has consecutive data
            qualified = True
            for period_id in range(max(period_indices_by_max), max(period_indices_by_max) - LENGTH_MAX_INTERVALS, -1):
                if period_id not in period_indices_by_max:
                    qualified = False
                    break
            if not qualified:
                continue
            allowed_period_indices = sorted(period_indices_by_max)[-LENGTH_MAX_INTERVALS:]
            df = df[df[max_period_index_key].isin(allowed_period_indices)]
            
            
            for friend_id in df['user_id'].unique():
                temp_df = df[df['user_id'] == friend_id].head(1)
                for key in count_dict.keys():
                    if key != 'num_users':
                        count_dict[key] += temp_df[key].values[0]
                count_dict['num_users'] += 1
            
            
            data = defaultdict(list)
            for interval_in_months in INTERVALS_IN_MONTHS:
                qualified_user_data = []
                qualified_friend_data = []
                period_indices = sorted(df[f'period_id_{interval_in_months}'].unique())
                for period_id in period_indices:
                    period_df = select_df_by_period(df, period_id, interval_in_months)
                    is_user = period_df['user_id'] == user_id
                    friend_df = period_df[~is_user]
                    # Skip if no friend in that period
                    if len(friend_df) == 0: continue

                    user_df = period_df[is_user]
                    # Skip if user has insufficient number of tweets in the period
                    if len(user_df) < num_tweets_per_period:
                        continue
                    
                    
                    user_df = user_df.sort_values(by="tweet", key=lambda x: -x.str.len())[:num_tweets_per_period]
                    friend_df = select_active_friends_per_period(file, period_id, friend_df)
                    friend_data = filter_friend_without_enough_tweets(friend_df)
                        #print(mention_count, reply_count, quote_count, friend_id)
                    if len(friend_data) < MAX_FRIENDS:
                        continue
                    
                    qualified_user_data.append(user_df)
                    qualified_friend_data.extend(friend_data[:MAX_FRIENDS])
                    #print(period_id, user_df, friend_df)
                    #user_df = 
                    #print(df)
                if len(qualified_friend_data) == 0: continue
                final_user_df = pd.concat(qualified_user_data)
                final_friend_df = pd.concat(qualified_friend_data)
                final_df = pd.concat([final_user_df, final_friend_df])
                num_period_qualified = final_df[f'period_id_{interval_in_months}'].unique()
                if len(num_period_qualified) == 1:
                    continue
                print(f"{user_id} m:{interval_in_months} p:{period_id} f:{len(final_df['user_id'].unique()) - 1}")
                
                data[interval_in_months] = final_df
            if len(data) < len(INTERVALS_IN_MONTHS):
                continue
            #unique_graphs = set()
            #for interval_in_months, merged_df in data.items():
            #    unique_graphs.add(len(merged_df[f'period_id_{interval_in_months}']))
            #if len(unique_graphs) != len(INTERVALS_IN_MONTHS):
            # At least shorter intervals should have more graphs
            #    continue

            for interval_in_months, merged_df in data.items():
                output_file = Path(f"./temporal_df/months_{interval_in_months}/{group_type}/{user_id}.csv")
                output_file.parent.mkdir(exist_ok=True, parents=True)
                print(f"{group_type} (ut: {num_tweets_per_period:02},  p: {interval_in_months:02})  [{len(merged_df[f'period_id_{interval_in_months}'].unique())} graphs] | #{counter:03}, total: {file_counter}/{size}  - {user_id}")
                merged_df['period_id'] = merged_df[f'period_id_{interval_in_months}']
                
                drop_columns = [f'period_id_{interval_in_months}' for interval_in_months in INTERVALS_IN_MONTHS]
                merged_df.drop(columns=drop_columns)
                merged_df.to_csv(output_file)
            counter += 1
