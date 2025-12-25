from sklearn.metrics import r2_score, mean_absolute_percentage_error, mean_squared_error
from sklearn.metrics import precision_score, recall_score, accuracy_score, roc_auc_score, precision_recall_curve, auc
from sklearn.metrics import confusion_matrix
import numpy as np

def get_metrics(y_hat, y, prefix=''):
  """
    This functions calculates 
      * mse
      * rmse
      * mape
      * r2-score
    Args:
      y_hat (numpy array): predicted result
      y (numpy array): ground truth
      prefix: prefix of a label
    Returns:
      result (dict): dict of the calculated items.
      Example) 
        {'mse': 0.1, 'rmse':0.09, 'mape':..., 'r2-score':... }
  """
  result = {}
  result[prefix + 'mse'] = mean_squared_error(y, y_hat)
  result[prefix + 'rmse'] = np.sqrt(result[prefix + 'mse'])
  result[prefix + 'mape'] = mean_absolute_percentage_error(y, y_hat)
  result[prefix + 'r2-score'] = r2_score(y, y_hat)

  return result
   
def specificity_score(y_true, y_pred):
  """
    This function calculates specificity.
    This is copyed from https://qiita.com/player_ppp/items/547afe4b61bee266ea43
  """
  tn, fp, fn, tp = confusion_matrix(y_true, y_pred).flatten()
  return tn / (tn + fp)

def class_metrics(y_hat, y, prefix=''):
  """
    This functions calculates 
      * acc
      * precision
      * recall
      * auc_pr
      * specifisity
      * auc_roc
    Args:
      y_hat (numpy array): predicted result
      y (numpy array): ground truth
      prefix: prefix of a label
    Returns:
      result (dict): dict of the calculated items.
      Example) 
        {'acc': 0.1, 'precision':0.09, 'recall':..., }
  """
  result = {}
  result[prefix + 'acc'] = accuracy_score(y, np.round(y_hat))
  result[prefix + 'precision'] = precision_score(y, np.round(y_hat))
  result[prefix + 'recall'] = recall_score(y, np.round(y_hat))
  result[prefix + 'specificity'] = specificity_score(y, np.round(y_hat))
  result[prefix + 'auc_roc'] = roc_auc_score(y, y_hat)

  precisions, recalls, _ = precision_recall_curve(y, y_hat)
  result[prefix + 'auc_pr'] = auc(recalls, precisions)

  return result
  

def multiclass_metrics(y_hat, y, prefix=''):
  """
    This functions calculates 
      * micro_acc
      * weighted_acc
      * weighted_precision
      * weighted_recall
      * weighted_auc_roc
      * weighted_auc_pr
      * weighted_specifisity
    Equations for above scores are found in https://medium.com/@ramit.singh.pahwa/micro-macro-precision-recall-and-f-score-44439de1a044 
    In 'micro', accuracy, precision, and recall take same value.

    Args:
      y_hat (numpy array): predicted result, ohe-hot vector like [[0.1,0.1,0.8],...]
      y (numpy array): ground truth, one-hot vector like [[1,0,0],...]
      prefix: prefix of a label

    Returns:
      result (dict): dict of the calculated items.
      Example) 
        {'micro_acc': 0.1, 'micro_precision':0.09, 'micro_recall':..., }
  """
  result = {}
  #Initialize result
  titles = ['acc', 'precision', 'recall', 'auc_roc', 'auc_pr', 'specificity']
  for title in titles:
    result[prefix + 'weighted_' + title] = 0
  
  #Calculate weighted scores
  weights = np.bincount(np.argmax(y, axis=1)) / y.shape[0]
  for n in range(y_hat.shape[1]):
    tmp_result = class_metrics(y_hat[:,n],y[:,n]) 
    for title in titles:
      result[prefix + 'weighted_' + title] += weights[n] * tmp_result[title]

  #Calculate micro scores
  result[prefix + 'micro_acc'] = accuracy_score(np.argmax(y,axis=1), np.argmax(y_hat,axis=1))

  return result

if __name__=='__main__':
  y_hat = np.array([[0.1,0.9,0],[0,1,0],[0.8,0.1,0.1],[0,0,1],[1,0,0]])
  y_true = np.array([[1,0,0],[0,1,0],[1,0,0],[0,0,1],[0,1,0]])
  print(multiclass_metrics(y_hat,y_true))


