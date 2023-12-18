import pytorch_lightning as pl
import numpy as np
import torch.nn as nn
from datetime import datetime
from .base_pipeline import AbstractPipeline
from ..utils.dataprocessing.model_loader import get_model
from ..utils.dataprocessing import get_dataloader
from ..utils.training.callback import custom_callbacks
from ..models.model import LightingModel

class ModelTraining(AbstractPipeline):
    def run(self, context):
        self._train_deep_learning_model(context)

    def _train_deep_learning_model(self, context):
        config = context.config.exp.training
        data = context.data
        results = []

        cv_gorup = int(datetime.timestamp(datetime.now()))
        for cv_data in context.cv_data:
            model, args, model_args = get_model(context.model_class, context.dataset_info)
            model.reset_parameters()
            train_data = model.prepare_inputs(data[cv_data.train_indices])
            val_data = model.prepare_inputs(data[cv_data.val_indices])
            test_data = model.prepare_inputs(data[cv_data.test_indices])
            train_dataloader = get_dataloader(train_data, batch_size = config.batch_size, shuffle = True)
            val_dataloader = get_dataloader(val_data, batch_size = config.batch_size, shuffle = False)
            test_dataloader = get_dataloader(test_data, batch_size = config.batch_size, shuffle = False)
            executed_at = int(datetime.timestamp(datetime.now()))

            trainer = pl.Trainer(
                accelerator=context.accelerator,
                max_epochs = config.max_epochs,
                callbacks = custom_callbacks(
                    model_name = model.__class__.__name__,
                    dataset_info = context.dataset_info,
                    current_fold = cv_data.fold,
                )
            )
            lighting_model = LightingModel(model)
            trainer.fit(lighting_model, train_dataloader, val_dataloaders = val_dataloader)
            result = trainer.test(lighting_model, test_dataloader, ckpt_path = 'best')[0]

            result = {
                'cv_gorup': cv_gorup,
                'executed_at': executed_at,
                'fold': cv_data.fold,
                **result
            }
            results.append(result)
        
        context.merge_update({
            'results': results,
        })
