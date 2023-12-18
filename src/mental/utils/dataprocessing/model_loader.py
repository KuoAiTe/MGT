
from ...utils.dataclass import BaselineModel, TrainingArguments, ModelArguments, EvaluationResult

def get_model(baseline, data_info):
    model = None
    args = TrainingArguments()
    model_args = ModelArguments()
    model = baseline(model_args, data_info)
    return model, args, model_args