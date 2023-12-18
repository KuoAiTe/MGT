import pandas as pd
import re
import numpy as np
import os
import pprint
import glob
from pathlib import Path

from mental.utils.utilities import compute_metrics_from_results
from mental.utils.logger import Logger
file_path = str(Path(os.path.abspath(__file__)).parent / 'results' / 'nov30' / '**.csv')
files = glob.glob(file_path)
d = []
for f in files:
    print(f)
    df = pd.read_csv(f)
    df = df[[
        'fold',
        'num_tweets_per_period',
        'max_num_friends',
        'period_length',
        'num_features',
        'periods_in_months',
        'test_class_0_aucroc',
        'test_class_1_f1',
        'test_accuracy',
        'test_class_0_f1',
        'test_class_0_precision',
        'test_class_0_recall',
        'test_class_1_aucroc',
        'test_class_1_precision',
        'test_class_1_recall',
        'model',
        'max_period_length',
        'random_state',
        'test_loss',
        'test_size',
        'cv_gorup',
        'tweet_processing_model_name',
        'dataset_location',
        'dataset_name',
        'date',
        'executed_at',
    ]]
    d.append(df)
df = pd.concat(d, ignore_index=True)
NUM_TWEETS_COLUMN = 'num_tweets_per_period'
NUM_FRIENDS_COLUMN = 'max_num_friends'
PERIOD_COLUMN = 'periods_in_months'
NUM_SNAPSHOT_COLUMN = 'period_length'
df = df.sort_values(by=[
    'model',
    'random_state',
    NUM_TWEETS_COLUMN,
    NUM_FRIENDS_COLUMN,
    PERIOD_COLUMN,
    NUM_SNAPSHOT_COLUMN,
    'fold'
    ])
df.reset_index(drop=True, inplace = True)
"""
for model_name in df['model'].unique():
    temp_df = df[df['model'] == model_name]
    temp_df.to_csv(f'./results/nov23/{model_name}.csv', index=False, float_format='%.4f')
"""
#df.to_csv(file_path, header = None)
metrics_for_depressed = [ 'test_class_1_precision', 'test_class_1_recall', 'test_class_1_f1', 'test_class_1_aucroc']
metrics_for_control = ['test_class_0_precision', 'test_class_0_recall', 'test_class_0_f1']
metrics = metrics_for_control + metrics_for_depressed
column_mapping = {
    'test_class_1_f1': 'F$_{1}$',
    'test_class_1_recall': 'Recall',
    'test_class_1_precision': 'Precision',
    'test_class_1_aucroc': 'AUROC',
    'test_class_0_f1': 'F$_{1}$',
    'test_class_0_recall': 'Recall',
    'test_class_0_precision': 'Precision',
    'test_class_0_aucroc': 'AUROC'
}

header_column_mapping = {
    'test_class_1_f1': 'F$_{1}$',
    'test_class_1_recall': 'R',
    'test_class_1_precision': 'P',
    'test_class_1_aucroc': 'AUROC',
    'test_class_0_f1': 'F$_{1}$',
    'test_class_0_recall': 'R',
    'test_class_0_precision': 'P',
    'test_class_0_aucroc': 'AUROC'
}

NAME_MAPPINGS = {
    'GATWrapper': 'GAT',
    'GCNWrapper': 'GCN',
    'GraphSAGEWrapper': 'GraphSAGE',
    'MentalNet_Original': 'MentalNet',
    'Sawhney_EMNLP_20': 'STATENet',
    'Sawhney_NAACL_21': 'Hyper-SOS',
    'CNNWithMaxPlus': 'CNNWithMax+',
    'UsmanBiLSTM': 'BiLSTMAttn',
    'UsmanBiLSTMPlus': 'BiLSTMAttn+',
    'MentalPlus': 'MGT',
    'MentalPlus_Base': 'MGT (w/o S)',
    'MentalPlus_HOMO': 'MGT-H',
    'MentalPlus_NO_GNN': 'MGT (w/o G)',
    'MentalPlus_NO_CONTENT_ATTENTION': 'MGT-CA',
    'MentalPlus_Without_Transformer': 'MGT (w/o T)',
}
NAME_MAPPINGS = {key.lower(): value for key, value in NAME_MAPPINGS.items()}

ORDER = [
    'UsmanBiLSTM',
    'UsmanBiLSTMPlus',
    'Sawhney_EMNLP_20',
    'GCNWrapper',
    'GATWrapper',
    'MentalNet',
    'Sawhney_NAACL_21',
    'DySAT',
    'DyHAN',
    'GraphSAGEWrapper',
    'CNNWithMax',
    'CNNWithMaxPlus',
    'MentalNet_Original',
    'MentalPlus',
    'MentalPlus_SUP',
    'ContrastEgo',
    'MentalPlus_Base',
    'MentalPlus_BatchNorm',
    'MentalPlus_HOMO',
    'MentalPlus_CLS_POOLING',
    'MentalPlus_MEAN_POOLING',
    'MentalPlus_NO_POSITION',
    'MentalPlus_NO_POSITION_CLS_POOLING',
    'MentalPlus_NO_HGNN',
    'MentalPlus_NO_GNN',
    'MentalPlus_USE_GRAPH',
    'MentalPlus_USE_NODE',
    'MentalPlus_NO_SUPERVISED',
    'MentalPlus_NO_GRAPH_AGGREGATION',
    'MentalPlus_Without_Transformer',
    'MentalPlus_NO_INTERACTION_CUTOUT',
    'MentalPlus_NO_TIMEFRAME_CUTOUT',
    'MentalPlus_NO_CONTENT_ATTENTION',
]

NUM_TWEETS_PER_PERIOD_LIST = [2, 4, 6]
NUM_FRIEND_LIST = [2, 4, 6]
PERIOD_IN_MONTHS_LIST = [3, 6, 12]
PERIOD_LENGTH = [2, 4, 8]
PERIOD_LENGTH_AHEAD = [-4, -2, -1]

# Custom
ALLOWED_RANDOM_STATE = 42
ALLOWED_TWEETS = [2, 4, 6]
ALLOWED_FRIENDS = [2, 4, 6]
ALLOWED_PERIOD_LENGTH = [2, 4, 8]
ALLOWED_PERIOD_LENGTH_AHEAD = [4, -1, -2, -4]

ABALATION_STUDY = True
DISALLOWED_VARIANTS = [
    'GraphSAGEWrapper',
    'MentalPlus_BatchNorm',
    'MentalPlus_NO_INTERACTION_CUTOUT',
    'MentalPlus_NO_TIMEFRAME_CUTOUT',
    'MentalPlus_NO_CONTENT_ATTENTION',
    'MentalPlus_SUP',
    'MentalPlus_HOMO',
    #'UsmanBiLSTMPlus',
    #'DyHAN',
    'GCNWrapper',
    #'GATWrapper',
]
if ABALATION_STUDY:
    DISALLOWED_VARIANTS = DISALLOWED_VARIANTS + [_ for _ in df['model'].unique() if not _.startswith('MentalPlus')]
    DISALLOWED_VARIANTS = DISALLOWED_VARIANTS + [_ for _ in df['model'].unique() if 'USE' in _]
    
else:
    #pass
    DISALLOWED_VARIANTS = DISALLOWED_VARIANTS + [_ for _ in df['model'].unique() if _.startswith('MentalPlus_')]
df = df[~df['model'].isin(DISALLOWED_VARIANTS)]
print(df['model'].unique)
def get_performance(df):
    
    performance = {}
    #'time', 
    for (model, random_state), group_df in df.groupby(['model', 'random_state']):
        
        if random_state != ALLOWED_RANDOM_STATE: continue
        qualified_df = []
        for (cv_group, round_df) in group_df.groupby('cv_gorup'):
            if len(round_df['fold']) % 5 != 0:
                continue
            qualified_df.append(round_df[metrics].mean().to_frame().T)
            #
        qualified_df = pd.concat(qualified_df)
        metric_result = (qualified_df[metrics].mean().to_dict(), qualified_df[metrics].std().to_dict())
        #print(metric_result)
        performance[(model, random_state)] = metric_result
        #pprint.pprint((model, random_state))
    
    return performance 

def print_line(model, model_data, columns):
    segments = []
    for data, best_data in model_data:
        segment = []

        for column in columns:
            if type(data) == dict:
                segment.append(data[column])
                continue
            else:
                mean_data, std_data = data[0], data[1]
                if type(mean_data[column]) == np.float64 or type(mean_data[column]) == float:

                    if np.round(mean_data[column], 2) == np.round(best_data[column], 2):
                        segment.append('\\textbf{%.2f} \\textbf{$\pm$ %.2f}' % (mean_data[column], std_data[column]))
                    else:
                        segment.append('%.2f $\pm$ %.2f' % (mean_data[column], std_data[column]))
                
        segment = '&'.join(segment)
        segments.append(segment)
    line = model.replace("_", "\_") + '&' + '&'.join(segments)
    return line


def get_header(experimental_variable_list, experimental_name, caption, label = ''):
    column_size = len(experimental_variable_list)
    first_line = ['']
    for i in range(column_size):
        first_line.append("\\multicolumn{7}{c}{%d %s}" % (experimental_variable_list[i], experimental_name))
    first_line = '&'.join(first_line)


    second_line = ['']
    for i in range(column_size):
        second_line.append("\\multicolumn{3}{c}{Healthy (0)} &  \\multicolumn{1}{c}{} & \\multicolumn{3}{c}{Depessed (1)}")
    second_line = '&'.join(second_line)

    third_line = ['']
    for i in range(column_size):
        third_line.append("&".join([header_column_mapping[metric] for metric in metrics]))
        
            
    third_line = '&'.join(third_line)
    caption = caption.replace("_", '\_')
    header = "\\begin{table*}[]\n\\renewcommand{\\arraystretch}{1.11}\n\\caption{" + caption + "\label{table:" + label + "}}\n\\bgroup\n\\resizebox{\\linewidth}{!}{%\n  \\begin{tabular}{"
    header += 'l'
    for _ in range(column_size):
        header += 'lllllll|'
    header += "}\n"
    lines = [first_line, second_line, third_line]
    for line in lines:
        header += f'    {line}\\\\\n'
    return header

def get_footer():
    return '\n \\end{tabular}\n}\n\\egroup\n\\end{table*}\n\n\n'
    
def generate_table(exp_data, settings, experimental_variable_list, experimental_name = '',  caption = '', label = ''):
    columns = metrics_for_control + metrics_for_depressed
    experimental_variable_list = sorted(experimental_variable_list)
    lines = []
    best_performance = {}
    performance = {}

    SETTING_SIZE = len(experimental_variable_list)
    num_models = 0
    model_mapping = {}
    model_names = set()
    for v_, visiting_exp_data in exp_data.items():
        for (model_name, random_state) in visiting_exp_data.keys():
            model_names.add(model_name)
    model_names = sorted(list(model_names), key=lambda item: ORDER.index(item))
    for num_models, model_name in enumerate(model_names):
        model_mapping[model_name] = num_models
    print(model_mapping)
    WIDTH = num_models + 1 + 3 + 1
    USED_METRICS = ['Precision', 'Recall', 'F$_{1}$', 'AUROC']
    METRIC_SIZE = len(USED_METRICS)
    SPACEA_BETWEEN_EXP = len(metrics)
    HEIGHT = SPACEA_BETWEEN_EXP * SETTING_SIZE + 1
    SPACE_BETWEEN_1 = 3
    SPACE_BETWEEN_2 = 3
    data = np.full((HEIGHT, WIDTH) , None)
    for i in range(HEIGHT):
        for j in range(WIDTH):
            data[i, j] = ""
    data[:, WIDTH - 1] = "\\\\"
    empty_count = 0
    print(experimental_variable_list, '-' * 100)
    for i in range(len(experimental_variable_list)):
        base_index = SPACEA_BETWEEN_EXP * i + 1
        exp_key = experimental_variable_list[i]
        visiting_exp_data = exp_data[exp_key]
        if len(visiting_exp_data) == 0: empty_count += 1
        data[base_index, 0] = "\\multirow{6}{*}{\\rotatebox[origin=c]{90}{%s %s}}" % (int(experimental_variable_list[i]), experimental_name)
        visiting_exp_data = dict(sorted(visiting_exp_data.items(), key=lambda item: ORDER.index(item[0][0])))
        for (model_name, random_state), values in visiting_exp_data.items():
            model_column_pos = model_mapping[model_name]
            data[0, 3 + model_column_pos] = NAME_MAPPINGS[model_name.lower()] if model_name.lower() in NAME_MAPPINGS else model_name.replace('_', '')

            for k, group in enumerate(['Healthy', 'Depressed']):
                if group == 'Healthy':
                    data[base_index, 1] = "\\multirow{3}{*}{\\rotatebox[origin=c]{90}{%s}}" % (group)
                else:
                    data[base_index + len(metrics_for_control), 1] = "\\multirow{4}{*}{\\rotatebox[origin=c]{90}{%s}}" % (group)
                for l, metric in enumerate(metrics):
                    data[base_index + l, 2] = column_mapping[metric]
                    data[base_index + l, 3 + model_column_pos] = (values[0][metric], values[1][metric])
                    data[base_index + l, -1] = ' \\\\'
                data[base_index + len(metrics_for_control) - 1, -1] = '\\bigstrut[b] \\\\\\cdashline{2-%d}' % (WIDTH -1)
                data[base_index + len(metrics_for_control), -1] = '\\bigstrut[t] \\\\'
                data[base_index, -1] = '\\bigstrut[t] \\\\'
                data[base_index + l - 1, -1] = '\\bigstrut[t] \\\\'
                data[base_index + l, -1] = '\\bigstrut[b] \\\\\\hline'
        for l, metric in enumerate(metrics):
            max_value = 0
            second_value = 999999999999
            for z in range(len(model_mapping)):
                value = data[base_index + l, 3 + z]
                if type(value) == tuple:
                    value = value[0]
                else:
                    value = 0
                if value > max_value:
                    second_value = max_value
                    max_value = value
            
            for z in range(len(model_mapping)):
                if type(data[base_index + l, 3 + z]) == tuple and data[base_index + l, 3 + z][0] == max_value:
                    data[base_index + l, 3 + z] = '\\textbf{' + '{:.3f} $\pm$ {:.3f}'.format(data[base_index + l, 3 + z][0], data[base_index + l, 3 + z][1]) + '}'
                elif type(data[base_index + l, 3 + z]) == tuple and data[base_index + l, 3 + z][0] == second_value:
                    data[base_index + l, 3 + z] = '\\underline{' + '{:.3f} $\pm$ {:.3f}'.format(data[base_index + l, 3 + z][0], data[base_index + l, 3 + z][1]) + '}'
                elif  type(data[base_index + l, 3 + z]) == tuple:
                    data[base_index + l, 3 + z] = '{:.3f} $\pm$ {:.3f}'.format(data[base_index + l, 3 + z][0], data[base_index + l, 3 + z][1])

    #row, col = np.unravel_index(np.argsort(x.ravel()),x.shape)
    if empty_count != len(experimental_variable_list) :
        lines = []
        print(caption)
        catpion = ''
        header = "\\begin{table*}[]\n\\caption{" + caption + "\label{table:" + label + "}}\n\\bgroup\n\\resizebox{\\linewidth}{!}{%\n  \\begin{tabular}{"
        header += 'ccr' + 'c' * (WIDTH - 3)
        header += "}\n"
        #  \multicolumn{3}{c}{}  & \multicolumn{3}{c}{Deep learning-based} & \multicolumn{2}{c}{Static graph-based}&\multicolumn{3}{c}{ Dynamic graph-based} & &
 
        lines.append(header)
        for i in range(HEIGHT):
            line = ' & '.join(data[i, :-1]) + data[i, -1]
            lines.append(line)
        footer = "\n \\end{tabular}\n}\n\\egroup\n\\end{table*}\n\n\n"
        lines.append(footer)
        lines = '\n'.join(lines)
    else:
        lines = ''
    return lines

def get_settings(df):
    settings = {}
    for i in df[['model', 'random_state']].values.tolist():
        key = f'{i[0]},{i[1]}'
        settings[key] = {
            'model': i[0],
            'random_state': i[1],
        }
    settings = dict(sorted(settings.items(), key=lambda x: ORDER.index(x[1]['model'])))

    return settings
def plot_by_num_friend(
    df,
    num_tweet_per_period,
    num_snapshots,
):
    df = df[(df[NUM_TWEETS_COLUMN] == num_tweet_per_period) & (df[NUM_SNAPSHOT_COLUMN] == num_snapshots)]
    num_friends_list = df[NUM_FRIENDS_COLUMN].unique()
    print('num_friends_list', num_friends_list)
    num_friends_list = [_ for _ in num_friends_list if _ in ALLOWED_FRIENDS]
    data = {}
    for num_friend in num_friends_list:
        _ = df[df[NUM_FRIENDS_COLUMN] == num_friend]
        data[num_friend] = get_performance(_)

    caption = f'Impact of number of friends in each snapshot.'
    print("plot_by_num_friend", num_tweet_per_period, num_snapshots, caption)
    #df['setting'] = df[['time', 'model', 'random_state']]
    label = f'vary_friend_t_{num_tweet_per_period}_p_{num_snapshots}'
    table = generate_table(data, get_settings(df), num_friends_list, experimental_name = 'friend', caption = caption, label = label)
    return table

def plot_by_num_tweet(
    df,
    num_friend,
    num_snapshots,
):
    df = df[(df[NUM_FRIENDS_COLUMN] == num_friend) & (df[NUM_SNAPSHOT_COLUMN] == num_snapshots)]
    num_tweet_per_period_list = df[NUM_TWEETS_COLUMN].unique()
    print('num_tweet_per_period_list', num_tweet_per_period_list)
    num_tweet_per_period_list = [_ for _ in num_tweet_per_period_list if _ in ALLOWED_TWEETS]

    data = {}
    for num_tweet_per_period in num_tweet_per_period_list:
        _ = df[df[NUM_TWEETS_COLUMN] == num_tweet_per_period]
        data[num_tweet_per_period] = get_performance(_)
    caption = f'mpact of number of tweets per user/friend per snapshot.'
    label = f'vary_tweet_f_{num_friend}_p_{num_snapshots}'
    table = generate_table(data, get_settings(df), num_tweet_per_period_list, experimental_name = 'tweet', caption = caption, label = label)

    return table


def plot_by_period(
    df,
    num_tweet_per_period,
    num_friend,
):
    df = df[(df[NUM_TWEETS_COLUMN] == num_tweet_per_period) & (df[NUM_FRIENDS_COLUMN] == num_friend)]
    period_in_months_list = df[PERIOD_COLUMN].unique()
    print('period_in_months_list', period_in_months_list)
    period_in_months_list = [_ for _ in period_in_months_list if _ in ALLOWED_PERIOD_LENGTH]
    data = {}
    for period_in_months in period_in_months_list:
        _ = df[df[PERIOD_COLUMN] == period_in_months]
        data[period_in_months] = get_performance(_)
    caption = f'Effect of Changing Number of Snapshots with  {num_tweet_per_period} tweets and {num_friend} friends per snapshot.'
    label = f'vary_period_t_{num_tweet_per_period}_f_{num_friend}'
    table = generate_table(data, get_settings(df), period_in_months_list, experimental_name = 'months', caption = caption, label = label)
    return table

def plot_by_snapshots(
    df,
    num_tweet_per_period,
    num_friend,
):
    df = df[(df[NUM_TWEETS_COLUMN] == num_tweet_per_period) & (df[NUM_FRIENDS_COLUMN] == num_friend)]
    snapshot_list = df[NUM_SNAPSHOT_COLUMN].unique()
    print('period_in_months_list', snapshot_list)
    period_in_months_list = [_ for _ in snapshot_list if _ in ALLOWED_PERIOD_LENGTH]
    data = {}
    for num_snapshot in snapshot_list:
        _ = df[df[NUM_SNAPSHOT_COLUMN] == num_snapshot]
        data[num_snapshot] = get_performance(_)
    caption = f'Impact of the number of snapshots.'
    label = f'vary_period_t_{num_tweet_per_period}_f_{num_friend}'
    table = generate_table(data, get_settings(df), period_in_months_list, experimental_name = 'snapshots', caption = caption, label = label)
    
    return table


def plot_by_snapshots_ahead(
    df,
    num_tweet_per_period,
    num_friend,
):
    df = df[(df[NUM_TWEETS_COLUMN] == num_tweet_per_period) & (df[NUM_FRIENDS_COLUMN] == num_friend)]
    snapshot_list = df[NUM_SNAPSHOT_COLUMN].unique()
    print('snapshot_list', snapshot_list)
    period_in_months_list = [_ for _ in snapshot_list if _ in ALLOWED_PERIOD_LENGTH_AHEAD]
    data = {}
    for num_snapshot in snapshot_list:
        _ = df[df[NUM_SNAPSHOT_COLUMN] == num_snapshot]
        data[num_snapshot] = get_performance(_)
    caption = f'Impact of how early the prediction was made before self-diagnosis.'
    label = f'vary_timeline_t_{num_tweet_per_period}_f_{num_friend}'
    table = generate_table(data, get_settings(df), period_in_months_list, experimental_name = 'months', caption = caption, label = label)
    return table

def plot_all_by_num_friend():
    # Vary by num_friend
    tables = []
    for num_tweet_per_period in NUM_TWEETS_PER_PERIOD_LIST:
        for num_snapshots in PERIOD_LENGTH:
            table = plot_by_num_friend(df, num_tweet_per_period, num_snapshots)
            tables.append(table)
    return tables
def plot_all_by_num_tweet():
    tables = []
    for num_friend in NUM_FRIEND_LIST:
        for num_snapshots in PERIOD_LENGTH:
            table = plot_by_num_tweet(df, num_friend, num_snapshots)
            tables.append(table)
    return tables

def plot_all_by_period():
    # Vary by period
    tables = []
    for num_tweet_per_period in NUM_TWEETS_PER_PERIOD_LIST:
        for num_friend in NUM_FRIEND_LIST:
            table = plot_by_period(df, num_tweet_per_period, num_friend)
            tables.append(table)
    return tables

def plot_all_by_snapshots():

    # Vary by period
    tables = []
    for num_tweet_per_period in NUM_TWEETS_PER_PERIOD_LIST:
        for num_friend in NUM_FRIEND_LIST:
            table = plot_by_snapshots(df, num_tweet_per_period, num_friend)
            tables.append(table)
    return tables

def plot_all_by_snapshots_ahead():

    # Vary by period
    tables = []
    for num_tweet_per_period in NUM_TWEETS_PER_PERIOD_LIST:
        for num_friend in NUM_FRIEND_LIST:
            table = plot_by_snapshots_ahead(df, num_tweet_per_period, num_friend)
            tables.append(table)
    return tables

def write_tables(filename, tables):
    filepath = Path(f"tables/{filename}")
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with filepath.open("w", encoding ="utf-8") as f:
        tables = [_ for _ in tables if len(_) > 0]
        for i, table in enumerate(tables):
            f.write(f"% =======Table{i + 1}====================\n\n\n")
            f.write(table)

table_friend = plot_all_by_num_friend()
table_tweet = plot_all_by_num_tweet()
table_snapshots = plot_all_by_snapshots()
table_snapshots_ahead = plot_all_by_snapshots_ahead()
write_tables('num_friend.txt', table_friend)
write_tables('num_tweet.txt', table_tweet)
write_tables('snapshots.txt', table_snapshots)
tables = table_snapshots_ahead + table_snapshots + table_friend + table_tweet
#tables = table_snapshots
write_tables('total.txt', tables)

exit()
for key, group_df in df.groupby([NUM_TWEETS_COLUMN, NUM_FRIENDS_COLUMN, PERIOD_COLUMN]):
    num_tweets_per_period, max_num_friends, periods_in_months = key
    models = group_df['model'].unique()
    sep = " " * 4
    print('num_tweets_per_period:', num_tweets_per_period)
    print('max_num_friends', max_num_friends)
    print('periods_in_months', periods_in_months)
    for model in models:
        model_df = group_df[group_df['model'] == model][metrics].max(axis = 0)
        part1 = []
        part2 = []

        for metric in metrics_for_depressed:
            title = f'{column_mapping[metric]}'
            score = f'{model_df[metric]:.3f}'.rjust(len(title))
            part1.append(title)
            part2.append(score)
        part1 = sep.join(part1)
        part2 = sep.join(part2)

        part3 = []
        part4 = []
        for metric in metrics_for_control:
            title = f'{column_mapping[metric]}'
            score = f'{model_df[metric]:.3f}'.rjust(len(title))
            part3.append(title)
            part4.append(score)

        part3 = sep.join(part3)
        part4 = sep.join(part4)

        group_sep = " " * 4
        line1 = f'{part1}{group_sep}{part3}'
        line2 = f'{part2}{group_sep}{part4}'
        model = model.center(len(line1), '-')
        depressed_headline = 'Depressed'.center(len(part1), '-')
        healthy_headline = 'Healthy'.center(len(part3), '-')

        print(f'{model}\n{depressed_headline}{group_sep}{healthy_headline}\n{line1}\n{line2}\n')
