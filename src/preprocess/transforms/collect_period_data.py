import torch
from re import S
import pandas as pd
from pathlib import Path
    
from dateutil.relativedelta import relativedelta
from utils.util import clean_tweet
from utils.util import connect


def generate_embedding_from_tweets(model, group, tweets, helper, nlp_model, tokenizer):
    tweets_embeddings = torch.tensor([])
    tweets_to_be_encoded = []
    for i in range(len(tweets)):
        tweet = tweets[i]
        embeddings = helper.find_tweet_embeddings(nlp_model, group, tweet['id'])
        #embeddings = None
        if embeddings == None:
            tweets_to_be_encoded.append((tweet['id'], clean_tweet(tweet['tweet'])))
        else:
            tweets_embeddings = torch.cat((tweets_embeddings, torch.tensor(embeddings).unsqueeze(0)))

    if len(tweets_to_be_encoded) > 0:
        inputs = tokenizer(list(map(lambda x: x[1], tweets_to_be_encoded)), return_tensors='pt', padding = True, truncation = True, max_length = 512)

        with torch.no_grad():
            encoded_outputs = model(**inputs).last_hidden_state
            input_mask_expanded = inputs['attention_mask'].unsqueeze(-1).expand(encoded_outputs.size()).float()
            sum_embeddings = torch.sum(encoded_outputs * input_mask_expanded, 1)
            sum_mask = input_mask_expanded.sum(1)
            sum_mask = torch.clamp(sum_mask, min = 1e-9)
            pool_embedding = sum_embeddings / sum_mask
        for i in range(len(tweets_to_be_encoded)):
            tweet_id, embedding = tweets_to_be_encoded[i][0], pool_embedding[i].detach().cpu().tolist()
            helper.update_tweet_embeddings(nlp_model, group, tweet_id, embedding)
            #print(f'helper.update_tweet_embeddings({nlp_model}, {group}, {tweet_id}, {embedding[:4]}')

        tweets_embeddings = torch.cat((tweets_embeddings, pool_embedding)) 
    features_mean = torch.mean(tweets_embeddings, axis=0).detach().cpu().numpy()
    return features_mean


def get_all_periods(start_date, end_date, periods_in_months):
    period_start_date, period_end_date = start_date, start_date + relativedelta(months = periods_in_months, seconds = 0)
    periods = {}
    period_id = 0
    period_iterator = []
    while period_start_date < end_date:
        period_end_date = period_start_date + relativedelta(months = periods_in_months)
        period_key = f'{period_start_date}-{period_end_date}'
        periods[period_key] = period_id
        period_iterator.append((period_id, period_start_date, period_end_date, period_key))
        period_id += 1
        period_start_date = period_end_date
    return period_iterator
    
def find_marked_user_tweets(helper, user_id, period_start_date, period_end_date, num_tweets_per_period, strict):
    marked_user_tweets_in_period = helper.find_user_tweets_between_dates_marked(user_id, period_start_date, period_end_date, min_num_tweets = num_tweets_per_period)
    if len(marked_user_tweets_in_period) != num_tweets_per_period:
        difference = num_tweets_per_period - len(marked_user_tweets_in_period)
        user_tweets_in_period = helper.find_user_tweets_between_dates(user_id, period_start_date, period_end_date, min_num_tweets = difference)
        if (strict and len(user_tweets_in_period) != difference) or (not strict and len(user_tweets_in_period) == 0):
            success = False
            return []
        else:
            for user_tweet in user_tweets_in_period:
                helper.mark_user_tweet(user_tweet['id'])
            user_tweets_in_period = marked_user_tweets_in_period + user_tweets_in_period
            return user_tweets_in_period
    else:
        return marked_user_tweets_in_period

def find_marked_friend_tweets(helper, friend_id, period_start_date, period_end_date, num_tweets_per_period, strict):
    marked_friend_tweets_in_period = helper.find_friend_tweets_between_dates_marked(friend_id, period_start_date, period_end_date, min_num_tweets = num_tweets_per_period)
    if len(marked_friend_tweets_in_period) != num_tweets_per_period:
        difference = num_tweets_per_period - len(marked_friend_tweets_in_period)
        friend_tweets_in_period = helper.find_friend_tweets_between_dates(friend_id, period_start_date, period_end_date, min_num_tweets = difference)
        if (strict and len(friend_tweets_in_period) != difference) or (not strict and len(friend_tweets_in_period) == 0):
            success = False
            return []
        else:
            for friend_tweet in friend_tweets_in_period:
                helper.mark_friend_tweet(friend_tweet['id'])
            friend_tweets_in_period = marked_friend_tweets_in_period + friend_tweets_in_period
            return friend_tweets_in_period
    else:
        return marked_friend_tweets_in_period
def process_tweet(helper, user_or_friend, user_id, period_id, group_type, tweets_in_period, mention_count, reply_count, quote_count, total_count):
    result = []
    for tweet in tweets_in_period:
        result.append({
            'user_id': tweet['user_id'],
            'group': user_id,
            'period_id': period_id,
            'label': group_type,
            'tweet_id' : tweet['id'],
            'tweet': tweet['tweet'],
            'created_at': tweet['created_at'],
            'mention_count': mention_count,
            'reply_count': reply_count,
            'quote_count': quote_count,
            'total_count': total_count,
        })
    return result

def count_stats(helper, user_id, username, friend_id, friend_username, period_start_date, period_end_date):
    doc = helper.find_period_interaction_cache(user_id, friend_id, period_start_date, period_end_date)
    if doc != None: return doc['mention_count'], doc['reply_count'], doc['quote_count'], doc['total_count']
    user_interaction_tweets_in_period = helper.find_user_interact_friend_between_dates(user_id, friend_id, friend_username, period_start_date, period_end_date, limit = 10000)
    friend_interaction_tweets_in_period = helper.find_friend_interact_user_between_dates(friend_id, user_id, username, period_start_date, period_end_date, limit = 10000)
    mention_count = 0
    reply_count = 0
    quote_count = 0
    total_count = 0
    if len(user_interaction_tweets_in_period) + len(friend_interaction_tweets_in_period) != 0:

        for tweet in user_interaction_tweets_in_period:
            if 'mentions' in tweet:
                if friend_id in tweet['mentions']:
                    mention_count += 1
            if 'reply_to' in tweet:
                if friend_id in tweet['reply_to']:
                    reply_count += 1
            if 'quote_tweet_username' in tweet:
                if friend_username in tweet['quote_tweet_username']:
                    quote_count += 1

        for tweet in friend_interaction_tweets_in_period:
            if 'mentions' in tweet:
                if user_id in tweet['mentions']:
                    mention_count += 1
            if 'reply_to' in tweet:
                if user_id in tweet['reply_to']:
                    reply_count += 1
            if 'quote_tweet_username' in tweet:
                if username in tweet['quote_tweet_username']:
                    quote_count += 1
                    
        total_count = mention_count + reply_count + quote_count
    helper.update_period_interaction_cache(user_id, friend_id, period_start_date, period_end_date, mention_count, reply_count, quote_count, total_count)
    return mention_count, reply_count, quote_count, total_count

def collect_period_data_all(
    client,
    config,
    group_type,
    start_date,
    end_date,
    num_tweets_per_period,
    num_friend_in_period,
    periods_in_months,
    ):  
    #MODEL = "roberta-base"
    STRICT_TWEET_NUM = False
    STRICT_FRIEND_NUM = False
    MIN_PERIOD_LENGTH = 3
    helper, csv_dir, csv_location = connect(client, config, group_type, 'friends')

    embedding_column_name = helper.get_embedding_column_name('nlp_model')
    #start_date, end_date = helper.find_start_and_end_dates()
    period_start_date, period_end_date = start_date, start_date + relativedelta(months = periods_in_months, seconds = 0)

    period_iterator = get_all_periods(start_date, end_date, periods_in_months)
    
    cursors = helper.find_users_if_interaction_exists()
    friends_allowed = {}
    interaction_count = {}
    interaction_types = ['mentions', 'mentioned', 'replies', 'replied', 'quotes', 'quoted', 'total']
    for row in cursors:
        user_id = row["user_id"]
        friend_id = row["friend_id"]
        key = f'{user_id}-{friend_id}'
        if user_id not in interaction_count:
            count_dict = {}
            for interaction_type in interaction_types:
                count_dict[interaction_type] = 0
            interaction_count[user_id] = count_dict

        update_dict = interaction_count[user_id]
        for interaction_type in interaction_types:
            if interaction_type in row:
                update_dict[interaction_type] += len(row[interaction_type])
                update_dict['total'] += len(row[interaction_type])
        result = update_dict['total'] > 0
        if not result:
            continue
        else:
            friends_allowed[key] = row
            
    users = helper.find_users_if_friends_exists(limit = 100000)
    user_node_id = -1
    user_size = len(users)
    count = 0
    user_counter = 0
    for user_node_id in range(user_size):
        user = users[user_node_id]
        user_id = user['user_id']
        #is_file_exists = len(glob.glob(f'df_data/{embedding_column_name}/ut_{num_tweets_per_period}_p_{periods_in_months}/{group_type}/{user_id}/*.csv')) > 0
        #if is_file_exists: continue
        diagnosis_time = helper.find_user_diagnosis_time(user_id)
        username = helper.find_username_by_id(user_id)
        if diagnosis_time == None: continue
        if username == None: continue
        friends = [friends_allowed[f'{user_id}-{friend_id}'] for friend_id in user['friends'] if f'{user_id}-{friend_id}' in friends_allowed]
        friends.sort(key = lambda x: -x['total_interactions'])
        
        period_start_date, period_end_date = start_date, start_date + relativedelta(months = periods_in_months, seconds = 0)
        friend_size = len(friends)
        data = []
        print(f'USER[{user_counter:03} | {user_node_id:03} / {user_size:05}], FRIEND[#{friend_size:03} / #{len(user["friends"]):03}] Diagnosis_time: {diagnosis_time}')
        for (period_id, period_start_date, period_end_date, period_key) in period_iterator:
            if period_start_date >= diagnosis_time:
                break
            if period_end_date >= diagnosis_time:
                period_end_date = diagnosis_time
            
            #print((period_id, period_start_date, period_end_date, period_key))
            # User must have min_num_tweets during the specified period
            user_tweets_in_period = find_marked_user_tweets(helper, user_id, period_start_date, period_end_date, num_tweets_per_period, strict = STRICT_TWEET_NUM)
            if len(user_tweets_in_period) == 0:
                continue
            period_user_data = process_tweet(helper, 'user', user_id, period_id, group_type, user_tweets_in_period, 0, 0, 0, 0)
            friend_counter = 0
            period_friend_data = []
            num_friend_counter = 0
            i = 0
            for friend in friends:
                i += 1
                friend_id = friend['friend_id']
                friend_username = helper.find_friend_username_by_id(friend_id)
                friend_tweets_in_period = find_marked_friend_tweets(helper, friend_id, period_start_date, period_end_date, num_tweets_per_period, strict = STRICT_TWEET_NUM)
                if len(friend_tweets_in_period) == 0: continue
                mention_count, reply_count, quote_count, total_count = count_stats(helper, user_id, username, friend_id, friend_username, period_start_date, period_end_date)
                if total_count == 0: continue
                num_friend_counter += 1
                period_friend_data.extend(process_tweet(helper, 'friend', user_id, period_id, group_type, friend_tweets_in_period, mention_count, reply_count, quote_count, total_count))
                print(f'USER[{user_counter:03} | {user_node_id:03} / {user_size:05}], FRIEND[#{num_friend_counter:03} | {i:03}/ {friend_size:03}], PERIOD: {period_id:02}| {num_tweets_per_period:02} tweets: in {periods_in_months:02} months with {num_friend_in_period:02} friends')
                if num_friend_counter >= num_friend_in_period:
                    break
            if (STRICT_FRIEND_NUM and num_friend_counter == num_friend_in_period) or (not STRICT_FRIEND_NUM and num_friend_counter > 0):
                friend_counter += 1
                # at least one friend
                data.extend(period_user_data)
                data.extend(period_friend_data)

        columns = ['user_id', 'group', 'period_id', 'tweet_id', 'tweet', 'created_at', 'mention_count', 'reply_count', 'quote_count', 'total_count']
        data_in_dict = {column:[] for column in columns}
        for row in data:
            for column in columns:
                data_in_dict[column].append(row[column])
        df = pd.DataFrame.from_dict(data_in_dict)
        period_id = df['period_id'].unique()
        if len(period_id) < MIN_PERIOD_LENGTH:
            continue
        
        filepath = Path(f'/media/aite/easystore/db/df_data/{embedding_column_name}/20230109/ut_{num_tweets_per_period}_p_{periods_in_months}_mnf{num_friend_in_period}_version3/{group_type}/{user_id}.csv')
        filepath.parent.mkdir(parents=True, exist_ok=True)
        print(df)
        print(df['user_id'].nunique(), period_id)
        df.to_csv(filepath)
        user_counter += 1

