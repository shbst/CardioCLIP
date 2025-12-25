import torch
from torch import nn
from dataclasses import dataclass


class MAELoss(nn.Module):
  def __init__(self):
    super(MAELoss, self).__init__()

  def forward(self, outputs, targets):
    
    loss = torch.mean(torch.abs(outputs - targets))
    return loss

class MEstimatorLoss(nn.Module):
  def __init__(self, k=1):
    super(MEstimatorLoss, self).__init__()
    self.k = k

  def forward(self, outputs, targets):
    e = outputs - targets
    loss = torch.mean(e**2 / (self.k + e**2))
    return loss

class FocalRLoss(nn.Module):
  """
  Focal-R loss is regression loss inspired by Focal loss.
  This loss function was proposed in https://arxiv.org/pdf/2102.09554.pdf.
  """
  def __init__(self, beta=1, gamma=2):
    super(FocalRLoss, self).__init__()
    self.beta = beta
    self.gamma = gamma

  def forward(self, outputs, targets):
    e = torch.abs(outputs - targets)
    loss = torch.mean(torch.sigmoid(self.beta * e)**self.gamma * e)
    return loss

@dataclass
class Losses:
  losses = {
    'mae': MAELoss(),
    'mestimator': MEstimatorLoss(),
    'mse': nn.MSELoss(),
    'focalr': FocalRLoss(),
  }


class Focal_MultiLabel_Loss_withLogits(nn.Module):
  """
  This code is based on https://take-tech-engineer.com/pytorch-focal-loss/
  Original paper of focal loss is https://arxiv.org/abs/1708.02002
  """
  def __init__(self, pos_weight=None, gamma=2.0, reduction="mean"):
    super().__init__()
    self.gamma = gamma
    self.softmax = nn.Softmax(dim=1)
    self.reduction = reduction
    self.pos_weight = pos_weight

  def forward(self, outputs, targets):
    y_hat = self.softmax(outputs)
    bce = -targets * torch.log(y_hat) - (1-targets) * torch.log(1 - y_hat)
    pt = y_hat * targets + (1 - y_hat) * (1 - targets)
    loss = (1 - pt)**self.gamma * bce

    loss = self._dot_weight(loss)

    if self.reduction=="mean":
      loss = loss.mean()
    elif self.reduction=="sum":
      loss = loss.sum()
    else:
      raise RuntimeError("reduction type {} is not available".format(self.reduction))
    return loss.mean()

  def _dot_weight(self, loss):
    if self.pos_weight is None:
      self.pos_weight = torch.ones(loss.shape[-1])
    self.pos_weight = self.pos_weight.to(loss.device)
    weighted_loss = torch.mm(self.pos_weight.unsqueeze(0), torch.permute(loss, (1,0)))
    return weighted_loss.squeeze()  

@dataclass
class ClsLosses:
  losses = {
    'bce': nn.BCEWithLogitsLoss,
    'focal': Focal_MultiLabel_Loss_withLogits,
  }
