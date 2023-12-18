import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

class Logger:
    def __init__(self, file_dir = './results'):
        self.file_dir = Path(file_dir)
    def log(self, model_name, dataset_info, args, result):
        result = {
            'dataset_location': f'{dataset_info.dataset_location}',
            'dataset_name': f'{dataset_info.dataset_name}',
            'random_state': f'{dataset_info.random_state}',
            'ntpp': dataset_info.num_tweets_per_period,
            'mnf': dataset_info.max_num_friends,
            'pim': dataset_info.periods_in_months,
            'period_length': dataset_info.period_length,
            'model': model_name,
            'lr': f'{args.learning_rate}',
            'batch_size': f'{args.train_batch_size}',
            #'labels': np.array2string(r.labels, separator = ''),
            #'predictions': np.array2string(r.predictions, separator = ''),
            **result
        }
        file_dir = self.file_dir / dataset_info.dataset_name 
        Path(file_dir).mkdir(parents=True, exist_ok=True)
        file_path = file_dir/ f'{model_name}.log'
        Path(file_path).touch(exist_ok = True)
        with open(file_path, 'a+') as writer:
            writer.write(f'{result}\n')