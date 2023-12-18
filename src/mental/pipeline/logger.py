import pytorch_lightning as pl
import pandas as pd
import os
import csv
import pprint
from box import Box
from datetime import datetime
from .base_pipeline import AbstractPipeline
from pathlib import Path
from ..utils.utilities import sanitize_filename


class PrepareLogger(AbstractPipeline):
    def run(self, context):
        project = 'afib'
        logger = [pl.loggers.CSVLogger("logs", name=project)]
        #if context.config.exp.use_wandb:
        #    logger.append(pl.loggers.WandbLogger(project = project))
        context['logger'] = logger
        return context
    

class Logger(AbstractPipeline):
    def run(self, context):
        model_name = sanitize_filename(context.model_class.__name__)
        dataset_name = sanitize_filename(context.dataset_info.dataset_name)
        log_dir = Path(f"./results/") / dataset_name
        log_dir.mkdir(parents = True,  exist_ok = True)
        log_path =  log_dir / f'{model_name}.csv'
        for result in context.results:
            result = {
                'date': datetime.now(),
                'model': model_name,
                **({key:getattr(context.dataset_info, key) for key in context.dataset_info.__annotations__.keys()}),
                **({key:f'{value:.4f}' for key, value in result.items()})
            }
            result = dict(sorted(result.items()))
            pprint.pprint(result)
            # If file doesn't exist, write headers.
            if not os.path.exists(log_path):
                with open(log_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames = result.keys())
                    writer.writeheader()

            # Append result to CSV without reading the entire file.
            with open(log_path, 'a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames = result.keys())
                writer.writerow(result)
            
        return context
    