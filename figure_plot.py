import csv
import glob
import pandas as pd
import re
from collections import defaultdict
from pathlib import Path
"""
# Specify the directory containing the CSV files
directory_path = './data/'

# Use glob to find all CSV files in the directory
csv_files = glob.glob(directory_path + '**/*.csv', recursive=True)

# Initialize an empty list to store DataFrames
dataframes = []
# Iterate through the list of CSV files and read each one into a DataFrame
for csv_file in csv_files:
    m = re.search(r'(\d+)', csv_file)
    user_id = m.groups()[0]
    try:
        df = pd.read_csv(csv_file)
    except:
        continue
    dataframes.append(df)
dataframes = pd.concat(dataframes)
dataframes.to_csv('vocat.csv')

"""
df = pd.read_csv('vocat.csv')


import pathlib
import numpy as np
from src.mental.utils.dataprocessing import load_data, get_settings
from src.mental.utils.dataprocessing.model_loader import get_model
from src.mental.utils.dataclass import BaselineModel
from src.mental.utils.training.trainer import CrossValidator

from src.mental.utils.logger import Logger

from wordcloud import WordCloud
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

dataset_location = pathlib.Path(__file__).parent.resolve()
logger = Logger('./results')

baselines = [
    BaselineModel.MentalPlus,
    #BaselineModel.MentalNet,
    #BaselineModel.DySAT,
    #BaselineModel.DyHAN,
    #BaselineModel.UsmanBiLSTM,
    #BaselineModel.Sawhney_NAACL_21,
    #BaselineModel.Sawhney_EMNLP_20,
    #BaselineModel.GCN,
    #BaselineModel.GAT,
    #BaselineModel.GraphSAGE,
]

#dataset_info -> current location

#dataset_name = 
dataset_names = ['twitter-roberta-base-2022-154m']#'twitter-roberta-base-emotion']#, ]
#'twitter-roberta-base-2022-154m',
default_num_tweets = 5
num_tweets_per_period_list = [5]

default_num_friends = 4
max_num_friends_list = [4]

default_interval = 3
periods_in_months_list = [3]
all_permutations = False
random_state = 5


dataset_list = get_settings(
    dataset_location = dataset_location,
    dataset_names = dataset_names,
    default_num_tweets = default_num_tweets,
    num_tweets_per_period_list = num_tweets_per_period_list,
    default_num_friends = default_num_friends,
    max_num_friends_list = max_num_friends_list,
    default_interval = default_interval,
    periods_in_months_list = periods_in_months_list,
    random_state = random_state,
    all_permutations = all_permutations
)

class TSNEVisualizer:
    def __init__(self, output_file='tsne_plot.png'):
        super().__init__()
        self.output_file = output_file
        self.results = []
        self.target_fold = 4
        self.new_name = {
            'MentalPlus': 'MGT',
            'Sawhney_NAACL_21': 'Hyper-SOS',
            'Sawhney_EMNLP_20': 'STATENet',
            'UsmanBiLSTM': 'BiLSTMAttn',
            'GCNWrapper': 'GCN',
            'GATWrapper': 'GAT',
            'GraphSAGEWrapper': 'GraphSAGE',
            'DySAT': 'DySAT',
            'DyHAN': 'DyHAN',
            'MentalNet': 'MentalNet',
        }
    def add_result(self, results):
        self.results.append(results)
    def visualize(self):
        # Step 1: PCA for dimensionality reduction
        #pca = PCA(n_components = 20, random_state = 30)  # Reduce to 10 principal components
        #data_pca = pca.fit_transform(logits)

        fig, ax = plt.subplots(nrows = 2, ncols = 5, figsize=(20 / 1.3, 5 / 1.3))
        #fig.figure(figsize=(5, 2.5))
        markers = ['+', '*']
        for i, test_result in enumerate(self.results):
            y = i % 5
            x = i // 5
            for fold in test_result.keys():
                
                model_name = test_result[fold]['model']
                logits = test_result[fold]['logits']
                labels = test_result[fold]['labels']
                tsne = TSNE(n_components = 2, perplexity = 50)
                embeddings_2d = tsne.fit_transform(logits)

                for label in np.unique(labels):
                    mask = labels == label
                    label_name = 'Depression' if label == 1 else 'Control'
                    c = '#bf0603' if label == 1 else '#003049'
                    ax[x, y].scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], label = label_name, c = c, marker = markers[label])

                ax[x, y].set_title(self.new_name[model_name])
                # Disable x-axis and y-axis ticks
                ax[x, y].set_xticks([])
                ax[x, y].set_yticks([])
        handles, labels = ax[0, 0].get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        print(by_label)
        fig.legend(by_label.values(), by_label.keys(), frameon = False, loc='lower center', bbox_to_anchor=(0.5, -0.07),fancybox=False, ncols = 2)
        fig.tight_layout()  # Adjust layout for subplots
        fig.savefig('aaa.pdf', dpi=300, bbox_inches='tight')
        fig.clf()
        plt.show()
    
if __name__ == '__main__':

    def get_tweet(tweet_id):
        return df[df['tweet_id'] == tweet_id]['tweet'].values[0]
    def wordcloud_to_file(wc, save_path):
        plt.figure( figsize=(30,20), facecolor='k')
        plt.imshow(wc, interpolation='bilinear')
        plt.axis("off")
        plt.savefig(save_path,  bbox_inches='tight')
        plt.show()
    save_dir = Path(f'./wordcloud/')
    stopwords = ['meta']
    for dataset_info in dataset_list:
        vizer = TSNEVisualizer()
        for baseline in baselines:
            data = load_data(dataset_info)
            model, args, model_args = get_model(baseline, dataset_info)
            print('-' * 10)
            print(dataset_info)
            print('-' * 10)
            cv = CrossValidator(
                data,
                n_folds = 5,
                max_epochs = 100,
                batch_size = 64,
                use_stratified = True,
            )
            data, train_index, test_index = cv.get_data()
            total_text = []
            for user_data in data:
                user_id, label = user_data['user_id'], user_data['label']
                if label == 0: continue
                user_tweets = []
                friends_tweets = []
                for user_snapshot_data in user_data['graphs']:
                    for node_id, node_data in user_snapshot_data.nodes(data=True):
                        target_container = user_tweets if node_data['label'] != -100 else friends_tweets
                        for i in range(len(node_data['tweets_id'])): 
                            target_container.append({
                                'user_id': node_id,
                                'tweet_id': node_data['tweets_id'][i],
                                'tweet': get_tweet(node_data['tweets_id'][i]),
                                'tweets_created_at': node_data['tweets_created_at'][i],
                                'period_id': node_data['period_id']
                            })
                word_cloud_user_tweets = ' '.join([row['tweet'] for row in user_tweets])
                word_cloud_friend_tweets = ' '.join([row['tweet'] for row in friends_tweets])
                total_text.append(word_cloud_user_tweets)
                print(user_id)
                dir = Path(f'./wordcloud/{label}/{user_id}')
                dir.mkdir(parents=True, exist_ok=True)
                wc = WordCloud(background_color='white', stopwords = ['meta'], max_words=2000, contour_width=3, scale = 4, contour_color='steelblue').generate(word_cloud_user_tweets)
                wc = WordCloud(background_color='white', stopwords = ['meta'], width = 800, height = 500).generate(word_cloud_friend_tweets).to_file(dir / 'friend.pdf')
            total_text = ' '.join(total_text)
            WordCloud(background_color='white', stopwords = ['meta'], width = 800, height = 500, scale=4).generate(total_text).to_file(save_dir / 'user.pdf')
            exit()
            test_results = cv.get_embeddings(
                model,
                dataset_info,
            )
            vizer.add_result(test_results)
            break
        vizer.visualize()
        exit()