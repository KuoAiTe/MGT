import numpy as np
from ..utils import preprocess_data
from .base_pipeline import AbstractPipeline
from src.models import TraditionalModel
from src.utils import load_data, get_dataloaders, custom_callbacks as callbacks

class ModelEvaluation(AbstractPipeline):
    def run(self, context):
        if issubclass(context.model_class, TraditionalModel):
            features = np.array([np.concatenate([row['categorical_features'], row['numerical_features']]) for row in context.test_data])
            labels = np.array([row['label'] for row in context.test_data])
            context.evaluation = context.trainer.test(features, labels)
        else:
            context.evaluation = context.trainer.test(dataloaders = context.test_dataloader, ckpt_path = 'best', verbose = False)