import torch.nn as nn
import torch


class SharedNet(nn.Module):
  def __init__(self, out_features):
    """
    同じ長さの入力を受け取って同じ長さの出力を返すシンプルなネットワーク
    """
    super().__init__()
    hidden_size = out_features // 2
    self.fc1 = nn.Linear(out_features, hidden_size)
    self.bn1 = nn.BatchNorm1d(hidden_size)
    self.activation = nn.GELU()
    self.fc2 = nn.Linear(hidden_size, out_features)
    self.bn2 = nn.BatchNorm1d(out_features) #最後にバッチノルムに通して重心を一致させる


  def forward(self, x):
    x = self.fc1(x)
    x = self.bn1(x)
    x = self.activation(x)
    x = self.fc2(x)
    y = self.bn2(x)

    return y 


