import os
import datetime
import torch
from box import Box
from dataclasses import dataclass
from .data_loading import DataLoading
from .data_splitter import DataSplitter
from .logger import PrepareLogger, Logger
from .model_training import ModelTraining
from .model_visualization import ModelVisualization
from typing import Any, Dict, List


@dataclass
class Pipeline:
    name: int = -1
    function: callable = None

class PipelineExecutor:
    def __init__(self, config, dataset_info, verbose = False):
        self.components = {
            100: Pipeline(name = "DataLoading", function = DataLoading()),
            200: Pipeline(name = "DataSplit", function = DataSplitter()),
            300: Pipeline(name = "PrepareLogger", function =  PrepareLogger()),
            400: Pipeline(name = "ModelTraining", function = ModelTraining()),
            600: Pipeline(name = "Logger", function = Logger()),
        }
        self.context = Box({
            'config': config,
            'dataset_info': dataset_info,
            'verbose': verbose,
            'trainer': None,
            'model_class': None,
            'evaluation': None,
        })
    def change_accelerator(self, accelerator):
        self.context['accelerator'] = accelerator
    def change_order(self, pipeline_name, order):
        pass
    def get_context(self):
        return self.context
    def insert_context(self, key, value):
        self.context[key] = value
    def delete_context(self, key):
        if key in self.context:
            del self.context[key]

    def register_component(self, component_name, instance):
        next_available_priority = max(self.components.keys()) + 1
        found = None
        for key, pipeline in self.components.items():
            if pipeline.name == component_name:
                found = key
                break
        if found is not None:
            self.components[found] = Pipeline(name = "Logger", function = instance)
        else:
            self.components[next_available_priority] = Pipeline(name = "Logger", function = instance)
    def unregister_component(self, component_name):
        deleted_key = None
        for key, component in self.components.items():
            if component.name == component_name:
                deleted_key = key
                break
        if deleted_key is not None:
            del self.components[deleted_key]

    def register_model_class(self, model_class):
        self.insert_context("model_class", model_class)

    def run(self):
        torch.set_float32_matmul_precision('medium')
        pipelines = dict(sorted(self.components.items())).items()
        print('=' * 100)
        print('Pipeline flow:\n->', '\n-> '.join([f'{i}: {pipeline.name}' for i, (_, pipeline) in enumerate(pipelines)]), '\n')
        for i, (priority, pipeline) in enumerate(pipelines):
            pipeline.function.run(self.context)
            if self.context.verbose:
                print(f"#{i} priority {priority}: Finished {pipeline.name}")
        print('=' * 100, '\n\n')


class DepressionVisualizer(PipelineExecutor):
    def __init__(self, config, dataset_info, verbose = False):
        super().__init__(config, dataset_info, verbose)
        self.components = {
            100: Pipeline(name = "DataLoading", function = DataLoading()),
            200: Pipeline(name = "DataSplit", function = DataSplitter()),
            300: Pipeline(name = "ModelVisualization", function = ModelVisualization()),
        }