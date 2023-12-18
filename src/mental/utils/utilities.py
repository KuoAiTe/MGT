import numpy as np
import networkx as nx
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

from ..utils.dataclass import BaselineModel, TrainingArguments, ModelArguments, EvaluationResult

from .dataset import MedDataset


def sanitize_filename(filename):
    # Replace characters that are not allowed in filenames with underscores
    return ''.join(char if char.isalnum() or char in ('-', '_', '.') else '_' for char in filename)

def split_graph(graphs, train_user_node_ids, test_user_node_ids):
    train_graphs = []
    test_graphs = []
    for graph in graphs:
        for node, data in graph.nodes(data = True):
            if node in train_user_node_ids:
                train_graphs.append(graph)
                break
            elif node in test_user_node_ids:
                test_graphs.append(graph)
                break
            else:
                print("something wrong")
                exit()
    return train_graphs, test_graphs


def compute_metrics_from_results(y_true, y_pred):
    depressed_indices = (y_true == 1)
    control_indices = ~depressed_indices
    precision = precision_score(y_true, y_pred, pos_label = 1, average = None)
    recall = recall_score(y_true, y_pred, pos_label = 1, average = None)
    f1 = f1_score(y_true, y_pred, pos_label = 1, average = None)
    auc_roc_macro = roc_auc_score(y_true, y_pred, average = 'macro')
    auc_roc_micro = roc_auc_score(y_true, y_pred, average = 'micro')
    acc_depressed = accuracy_score(y_true[depressed_indices], y_pred[depressed_indices])
    acc_control = accuracy_score(y_true[control_indices], y_pred[control_indices])
    num_depressed = np.count_nonzero(depressed_indices)
    num_control = np.count_nonzero(control_indices)
    result = EvaluationResult(
        labels = y_true,
        predictions = y_pred,
        num_depressed = num_depressed,
        num_control = num_control,
        precision_depressed = precision[1],
        recall_depressed = recall[1],
        f1_depressed = f1[1],
        acc_depressed = acc_depressed,
        precision_control = precision[0],
        recall_control = recall[0],
        f1_control = f1[0],
        acc_control = acc_control,
        auc_roc_macro = auc_roc_macro,
        auc_roc_micro = auc_roc_micro
    )
    return result

