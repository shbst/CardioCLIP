import datetime
from dataclasses import dataclass
import torch.nn as nn

def timestamp():
  ct = datetime.datetime.now()
  return ct.strftime("%Y-%m%d %H:%M:%S")

@dataclass
class Activations:
  activations = {
    'relu': nn.ReLU(),
    'gelu': nn.GELU(),
    'leakyrelu': nn.LeakyReLU(),
    'selu': nn.SELU(),
  }
