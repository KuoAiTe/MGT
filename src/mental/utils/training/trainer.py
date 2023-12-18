import pytorch_lightning as pl
import copy
import torch
import numpy as np
import torch.nn as nn
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit
from sklearn.decomposition import PCA
from datetime import datetime
from .callback import custom_callbacks, get_ckpt_save_path
from ..dataprocessing import get_dataloader
from ...models.model import LightingModel
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

class TSNEVisualizer:
    def __init__(self, output_file='tsne_plot.png'):
        super().__init__()
        self.output_file = output_file

    def visualize(self, model_name, logits, labels):
        # Step 1: PCA for dimensionality reduction
        #pca = PCA(n_components = 20, random_state = 30)  # Reduce to 10 principal components
        #data_pca = pca.fit_transform(logits)
        tsne = TSNE(n_components = 2, perplexity = 50, random_state = 30)
        embeddings_2d = tsne.fit_transform(logits)

        plt.figure(figsize=(4, 2))
        cmap = plt.cm.get_cmap('tab10')

        for label in np.unique(labels):
            mask = labels == label
            label_name = 'Depression' if label == 1 else 'Control'
            c = '#bf0603' if label == 1 else '#003049'
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], label = label_name, c = c)

        plt.title(model_name)
        # Disable x-axis and y-axis ticks
        plt.xticks([])
        plt.yticks([])
        plt.legend(frameon=False)
        #plt.show()

        plt.savefig(self.output_file, dpi=300, bbox_inches='tight')
        plt.clf()

class CrossValidator:
    def __init__(self, data, n_folds = 5, max_epochs = 50, batch_size = 64, test_size = 0.2, use_stratified = True, random_state = 42):
        self.data = data
        self.labels = np.array([row['label'] for row in data])
        self.n_folds = n_folds
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.random_state = random_state
        self.kf = StratifiedKFold(n_splits = n_folds, random_state = random_state, shuffle = True) if use_stratified else KFold(n_splits = n_folds, random_state = random_state, shuffle = True)
        self.sss = StratifiedShuffleSplit(n_splits = 1, test_size = test_size, random_state = random_state)
    def get_data(self):
        train_index, test_index = list(self.sss.split(self.data, y = self.labels))[0]
        return self.data, train_index, test_index
    def get_embeddings(self, input_model, dataset_info):
        train_index, test_index = list(self.sss.split(self.data, y = self.labels))[0]
        data = self.data[train_index]
        labels = self.labels[train_index]
        test_dataloader = get_dataloader(self.data[test_index], batch_size = self.batch_size, shuffle = False)
        round = int(datetime.timestamp(datetime.now()))
        test_results = {}
        for current_fold, (train_index, val_index) in enumerate(self.kf.split(data, y = labels)):
            #if current_fold != 1: continue
            # easiest way to reset models, double init to make sure we don't miss something that are not being reset in the next round.
            model = copy.deepcopy(input_model)
            model_name = model.__class__.__name__
            model.eval()
            ckpt_path = get_ckpt_save_path(model_name, dataset_info, current_fold)
            ckpt_path = 'twitter-roberta-base-2022-154m_MentalPlus_0_ntpp5_mnf4_pim_3'
            try:
                print(f'Loading {model_name}')
                model = LightingModel.load_from_checkpoint(f"./checkpoints/{ckpt_path}.ckpt", model = model).cpu()
            except:
                raise Exception("Checkpoint cannot be found.")
            
            
            logits = []
            labels = []
            for batch in test_dataloader:
                # train step
                outputs = model.predict(**batch)
                logits.append(outputs.logits)
                labels.append(outputs.labels)
            test_results[current_fold] = {
                'model': model_name,
                'logits': torch.cat(logits, dim = 0).detach().cpu().numpy(),
                'labels': torch.cat(labels, dim = 0).long().squeeze().detach().cpu().numpy(),
            }
            # Save the array to a file
            # Assuming 'logits' is your array

            #plt.show()
        return test_results
    def train(self, input_model, dataset_info):
        
        train_index, test_index = list(self.sss.split(self.data, y = self.labels))[0]
        data = self.data[train_index]
        labels = self.labels[train_index]
        test_dataloader = get_dataloader(self.data[test_index], batch_size = self.batch_size, shuffle = False)
        round = int(datetime.timestamp(datetime.now()))
        test_results = []

        for current_fold, (train_index, val_index) in enumerate(self.kf.split(data, y = labels)):
            # easiest way to reset models, double init to make sure we don't miss something that are not being reset in the next round.
            model = copy.deepcopy(input_model)
            model.reset_parameters()
            
            train_dataloader = get_dataloader(self.data[train_index], batch_size = self.batch_size, shuffle = True)
            val_dataloader = get_dataloader(self.data[val_index], batch_size = self.batch_size, shuffle = False)
            executed_at = int(datetime.timestamp(datetime.now()))
            trainer = pl.Trainer(
                max_epochs = self.max_epochs,
                callbacks = custom_callbacks(
                    model_name = model.__class__.__name__,
                    dataset_info = dataset_info,
                    current_fold = current_fold,
                )
            )
            lighting_model = LightingModel(model)
            trainer.fit(lighting_model, train_dataloader, val_dataloaders = val_dataloader)
            result = trainer.test(lighting_model, test_dataloader, ckpt_path = 'best')[0]
            test_result = {
                'round': round,
                'executed_at': executed_at,
                'fold': current_fold,
                **result
            }
            test_results.append(test_result)
        return test_results