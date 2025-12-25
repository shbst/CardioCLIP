from torchmetrics.functional import (accuracy, precision_recall,
                                     auroc, average_precision, specificity,
                                     f1_score)
import torch

def get_multilabel_metrics(y_hat, y, targets, threshold=0.5):
    """
    Args)
      ground truth and output from the model
      Note: y_hat.shape = (32, 2)
    Returns)
      dict of metrics for each class in y
    """
    d = {}
    for i in range(y.shape[1]):
      ret = get_metrics(y_hat[:,i], y[:,i])
      ret = {k+f'_{targets[i]}':v for k,v in ret.items()}
      d.update(ret)
    return d

def get_metrics(y_hat, y, threshold=0.5):
    """
    only for bionary classification!!
    """
    y_hat = torch.sigmoid(y_hat)
    y = y.to(torch.int16)
    acc = accuracy(
        y_hat,
        y,
        threshold=threshold
    )
    precision, recall = precision_recall(y_hat, y, threshold=threshold)
    _specificity = specificity(y_hat, y, threshold=threshold)
    _f1_score = f1_score(y_hat, y, threshold=threshold)
    auc = auroc(y_hat, y)
    avg_precision = average_precision(y_hat, y)

    return {
        'acc': acc,
        'precision': precision,
        'recall': recall,
        'auc': auc,
        'avg_precision': avg_precision,
        'specificity': _specificity,
        'f1_score': _f1_score,
    }
