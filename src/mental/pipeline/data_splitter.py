from cgi import test
import numpy as np
from box import Box
from datetime import datetime
from .base_pipeline import AbstractPipeline
from ..utils.dataprocessing import load_data
from sklearn.model_selection import KFold, StratifiedKFold, StratifiedShuffleSplit

class DataSplitter(AbstractPipeline):
    def run(self, context):
        config = context.config.exp.splitter
        kf = StratifiedKFold(n_splits = config.n_splits, random_state = config.random_state, shuffle = True)
        if False:
            sss = StratifiedShuffleSplit(n_splits = 1, test_size = config.test_size, random_state = config.random_state)
            _train_indices, test_indices = list(sss.split(context.labels, y = context.labels))[0]
            cv_data = []
            for fold, (train_indices, val_indices) in enumerate(kf.split(context.labels[_train_indices], y = context.labels[_train_indices])):
                #np.random.shuffle(train_indices)
                #np.random.shuffle(val_indices)
                #np.random.shuffle(test_indices)
                cv_data.append(
                    Box({
                    'fold': fold,
                    'train_indices': train_indices,
                    'val_indices': val_indices,
                    'test_indices': test_indices,
                    })
                )
        else:
            cv_data = []
            test_size = 1 / config.n_splits
            new_test_size = test_size / (1 - test_size)

            for fold, (_train_indices, test_indices) in enumerate(kf.split(context.labels, y = context.labels)):
                sss = StratifiedShuffleSplit(n_splits = 1, test_size = new_test_size, random_state = config.random_state)
                train_indices, val_indices = list(sss.split(context.labels[_train_indices], y = context.labels[_train_indices]))[0]
                #np.random.shuffle(train_indices)
                cv_data.append(
                    Box({
                    'fold': fold,
                    'train_indices': train_indices,
                    'val_indices': val_indices,
                    'test_indices': test_indices,
                    })
                )
        context.merge_update({
            'cv_data': cv_data,
        })