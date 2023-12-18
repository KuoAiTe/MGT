import os
import pprint
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.nn import BCELoss, BCEWithLogitsLoss
from datetime import datetime
from pytorch_lightning import LightningModule
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from dataclasses import dataclass
from typing import Optional
from ..utils.losses.supcon import SupConLoss
def weighted_binary_cross_entropy(sigmoid_x, targets, pos_weight, weight=None, size_average=True, reduce=True):
    """
    Args:
        sigmoid_x: predicted probability of size [N,C], N sample and C Class. Eg. Must be in range of [0,1], i.e. Output from Sigmoid.
        targets: true value, one-hot-like vector of size [N,C]
        pos_weight: Weight for postive sample
    """
    if not (targets.size() == sigmoid_x.size()):
        raise ValueError("Target size ({}) must be the same as input size ({})".format(targets.size(), sigmoid_x.size()))

    loss = -pos_weight* targets * sigmoid_x.log() - (1-targets)*(1-sigmoid_x).log()

    if weight is not None:
        loss = loss * weight

    if not reduce:
        return loss
    elif size_average:
        return loss.mean()
    else:
        return loss.sum()

class WeightedBCELoss:
    def __init__(self, pos_weight=None, weight=None, PosWeightIsDynamic= True, WeightIsDynamic= False, size_average=True, reduce=True):
        """
        Args:
            pos_weight = Weight for postive samples. Size [1,C]
            weight = Weight for Each class. Size [1,C]
            PosWeightIsDynamic: If True, the pos_weight is computed on each batch. If pos_weight is None, then it remains None.
            WeightIsDynamic: If True, the weight is computed on each batch. If weight is None, then it remains None.
        """
        super().__init__()

        #self.register_buffer('weight', weight)
        #self.register_buffer('pos_weight', pos_weight)
        self.weight = weight
        self.pos_weight = pos_weight
        self.size_average = size_average
        self.reduce = reduce
        self.PosWeightIsDynamic = PosWeightIsDynamic

    def forward(self, input, target):
        # pos_weight = Variable(self.pos_weight) if not isinstance(self.pos_weight, Variable) else self.pos_weight
        if self.PosWeightIsDynamic:
            positive_counts = target.sum(dim=0)
            nBatch = len(target)
            self.pos_weight = (nBatch - positive_counts)/(positive_counts +1e-5)

        if self.weight is not None:
            # weight = Variable(self.weight) if not isinstance(self.weight, Variable) else self.weight
            return weighted_binary_cross_entropy(input, target,
                                                 self.pos_weight,
                                                 weight=self.weight.to(input.device),
                                                 size_average=self.size_average,
                                                 reduce=self.reduce)
        else:
            return weighted_binary_cross_entropy(input, target,
                                                 self.pos_weight,
                                                 weight=None,
                                                 size_average=self.size_average,
                                                 reduce=self.reduce)

class FocalLoss(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        alpha: float = 0.25,
        gamma: float = 2,
        reduction: str = "sum",
    ) -> torch.Tensor:
        # Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py
        p = torch.sigmoid(inputs)
        ce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** gamma)

        if alpha >= 0:
            alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
            loss = alpha_t * loss

        # Check reduction option and return loss accordingly
        if reduction == "none":
            pass
        elif reduction == "mean":
            loss = loss.mean()
        elif reduction == "sum":
            loss = loss.sum()
        else:
            raise ValueError(
                f"Invalid Value for arg 'reduction': '{reduction} \n Supported reduction modes: 'none', 'mean', 'sum'"
            )
        return loss
@dataclass
class ModelOutput:
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    supcon_logits: Optional[torch.Tensor] = None
    cls_logits: Optional[torch.Tensor] = None
    prediction_scores: Optional[torch.Tensor] = None
    attention_mask: Optional[torch.Tensor] = None
    labels: Optional[torch.Tensor] = None
    tweet_attention_weights: Optional[torch.Tensor] = None
    graph_attention_weights: Optional[torch.Tensor] = None
    temporal_attention_weights: Optional[torch.Tensor] = None

class BaseModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.allowed_inputs = ['label']
        num_positive = 180
        num_negative = 303
        weight0 = num_positive / (num_negative + num_positive)
        self.loss_fct = WeightedBCELoss(weight=torch.FloatTensor([weight0, 1-weight0]))
        #self.loss_fct = 
        #pos_weight=pos_weight

    def remove_unused_inputs(self, inputs):
        deleted_keys = [key for key in inputs.keys() if key not in self.allowed_inputs]
        for key in deleted_keys:
            del inputs[key]
        return inputs
    
    def prepare_inputs(self, kwargs):
        raise NotImplementedError

    def prepare_batch_inputs(self, kwargs):
        raise NotImplementedError
    def reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
                
    def compute_loss(self, **kwargs):
        kwargs = self.prepare_batch_inputs(kwargs)
        labels = kwargs.pop('labels', None)
        if labels is None:
            raise AssertionError("No labels for computing metrics.")
        output = self.forward(**kwargs)
        output.loss = self.loss_fct.forward(output.prediction_scores, labels).unsqueeze(0)
        output.labels = labels
        return output

    @torch.no_grad()
    def predict(self, **kwargs):
        kwargs = self.prepare_batch_inputs(kwargs)
        labels = kwargs.pop('labels', None)
        if labels is None:
            raise AssertionError("No labels for computing metrics.")
        output = self.forward(**kwargs)
        output.loss = self.loss_fct.forward(output.prediction_scores, labels).unsqueeze(0)
        output.labels = labels
        return output


    @torch.no_grad()
    def get_attention_scores(self, **kwargs):
        labels = kwargs.pop('labels', None)
        output = self.__get_attention_scores__(**kwargs)
        return output, labels




class BaseConModel(BaseModel):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.supcon_loss = SupConLoss()
        self.criterion = nn.CosineSimilarity(dim=1)
        self.head = nn.Sequential(
                nn.Linear(64, 64),
                nn.ReLU(inplace=True),
                nn.Linear(64, 64)
        )
    def reset_parameters(self):
        self.backbone.reset_parameters()
        for module in self.head:
            if hasattr(module, 'reset_parameters'):
                module.reset_parameters()

        
    def prepare_inputs(self, inputs):
        return self.backbone.prepare_inputs(inputs)
    
    def prepare_batch_inputs(self, inputs):
        return self.backbone.prepare_batch_inputs(inputs)
    
    def compute_loss(self, **kwargs):
        kwargs = self.prepare_batch_inputs(kwargs)
        labels = kwargs.pop('labels', None)
        if labels is None:
            raise AssertionError("No labels for computing metrics.")
        output = self.backbone.forward(**kwargs)
        features = self.head(output.logits)
        cls_logits = self.backbone.depression_prediction_head(output.logits)
        prediction_scores = torch.sigmoid(cls_logits)
        output.loss = self.loss_fct.forward(prediction_scores, labels).unsqueeze(0)
        output.loss = output.loss + self.supcon_loss(features.unsqueeze(1), labels)
        output.labels = labels
        return output

    @torch.no_grad()
    def predict(self, **kwargs):
        kwargs = self.prepare_batch_inputs(kwargs)
        labels = kwargs.pop('labels', None)
        if labels is None:
            raise AssertionError("No labels for computing metrics.")
        output = self.backbone.forward(**kwargs)
        features = self.head(output.logits)
        cls_logits = self.backbone.depression_prediction_head(output.logits)
        prediction_scores = torch.sigmoid(cls_logits)
        output.loss = self.loss_fct.forward(prediction_scores, labels).unsqueeze(0)
        output.loss = output.loss + self.supcon_loss(features.unsqueeze(1), labels)
        output.labels = labels
        return output


class LightingModel(LightningModule):
    def __init__(self, model, *args, **kwargs):
        super().__init__()
        self.save_hyperparameters(
            ignore = ["model", "loss_fct", "model.loss_fct", "model.loss_fct.pos_weight", "model.loss_fct.weight"]
        )
        self.model = model
        self.learning_rate = 2e-4
        self.validation_step_outputs = []
        self.test_step_outputs = []
        self.log_path = 'raw.csv'

    def training_step(self, batch, batch_idx):
        output = self.model.compute_loss(**batch) 
        return output.loss

    def validation_step(self, batch, batch_idx):
        outputs = self._shared_eval(batch)
        self.validation_step_outputs.append(outputs)
        return outputs

    def test_step(self, batch, batch_idx):
        outputs = self._shared_eval(batch)
        self.test_step_outputs.append(outputs)
        return outputs
        
    def predict(self, **kwargs):
        return self.model.predict(**kwargs)

    def on_validation_epoch_end(self):
        self._shared_eval_on_epoch_end(self.validation_step_outputs, prefix = 'val')
        self.validation_step_outputs.clear()

    def on_test_epoch_end(self):
        result = self._shared_eval_on_epoch_end(self.test_step_outputs, prefix = 'test')

        result = {key:f'{value:.3f}' for key, value in result.items()}
        result['date'] = datetime.now()
        result['model'] = self.model.__class__.__name__
        pprint.pprint(result, width=1)
        new_df = pd.DataFrame([result])

        if os.path.exists(self.log_path):
            df = pd.read_csv(self.log_path)
            df = pd.concat([df, new_df])
        else:
            df = new_df
        df.to_csv(self.log_path, index = False)
        self.test_step_outputs.clear()
        
    def detach(self, cls_logits, prediction_scores):
        return (prediction_scores > 0.5).long(), prediction_scores.detach().cpu().numpy()

    def _shared_eval_on_epoch_end(self, outputs, prefix):
        y_true = torch.cat([output.labels for output in outputs], dim = 0).detach().cpu().numpy()
        prediction_scores = torch.cat([output.prediction_scores for output in outputs], dim = 0).detach().cpu().numpy()
        losses = torch.cat([output.loss for output in outputs])
        loss = torch.mean(losses).item()

        metrics = {
            f"{prefix}_loss": loss,
            F"{prefix}_size": y_true.shape[0],
        }
        y_pred = (prediction_scores > 0.5).astype(np.int32)
        precision = precision_score(y_true, y_pred, average = None, zero_division = 0).astype(np.float32)
        recall = recall_score(y_true, y_pred, average = None).astype(np.float32)
        f1 = f1_score(y_true, y_pred, average = None).astype(np.float32)
        aucroc = roc_auc_score(y_true = y_true, y_score = prediction_scores, average = None).astype(np.float32)
        class_labels = np.unique(y_true)
        for i in range(len(class_labels)):
            class_label = int(class_labels[i])
            metrics[f"{prefix}_accuracy"] = accuracy_score(y_true, y_pred).astype(np.float32)
            metrics[f"{prefix}_class_{class_label}_precision"] = precision[i]
            metrics[f"{prefix}_class_{class_label}_recall"] = recall[i]
            metrics[f"{prefix}_class_{class_label}_f1"] = f1[i]
            metrics[f"{prefix}_class_{class_label}_aucroc"] = aucroc
        if True or prefix == 'test':
            pprint.pprint(metrics, width=1)
        self.log_dict(metrics)
        return metrics

    def _shared_eval(self, batch):
        return self.predict(**batch)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr = self.learning_rate, weight_decay = 1e-6)
        return optimizer
