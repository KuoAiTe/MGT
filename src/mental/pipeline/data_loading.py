import numpy as np
from box import Box
from .base_pipeline import AbstractPipeline
from ..utils.dataprocessing import load_data

class DataLoading(AbstractPipeline):
    def run(self, context):
        data, labels = load_data(context.dataset_info)
        context.merge_update({
            'data': data,
            'labels': labels,
        })