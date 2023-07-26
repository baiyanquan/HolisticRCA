import numpy as np
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score


def fault_type_classification(y_pred, y_true):
    evaluation_metric_dict = dict()

    evaluation_metric_dict['confusion_matrix'] = multilabel_confusion_matrix(y_true, y_pred)
    for i in ['micro', 'macro']:
        for j in ['precision_score', 'recall_score', 'f1_score']:
            evaluation_metric_dict[f'{i}_{j}'] = eval(f'{j}(y_true, y_pred, average="{i}", zero_division=0)')

    for j in ['precision_score', 'recall_score', 'f1_score']:
        evaluation_metric_dict[f'{j}'] = eval(f'{j}(y_true, y_pred, average=None, zero_division=0)')

    return evaluation_metric_dict
