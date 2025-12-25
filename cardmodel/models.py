"""
      This code is implemented based on https://qiita.com/tchih11/items/377cbf9162e78a639958
"""

import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import pytorch_lightning as pl

class BaseBlock(nn.Module):
  def __init__(self, first_conv_in_channels, first_conv_out_channels, identity_conv=None, stride=1, activation=nn.ReLU()):
    """
      Args:
        first_conv_in_channnels: the number of input channnels of first conv
        first_conv_out_channnels: the nuber of output channels of first conv
        identity_conv: this layer should ajust the numbe of channels
        stride: stride number of 5-conv layer. The size is reduced to half when it is set to 2.
    """
    super(BaseBlock, self).__init__()

    #first conv layer(kernel size is 1)
    self.conv1 = nn.Conv1d(
      first_conv_in_channels, first_conv_out_channels, kernel_size=1, stride=1, padding=0)
    self.bn1 = nn.BatchNorm1d(first_conv_out_channels)

    #second conv layer(kernel size is 5)
    self.conv2 = nn.Conv1d(
      first_conv_out_channels, first_conv_out_channels, kernel_size=3, stride=stride, padding=1)
    self.bn2 = nn.BatchNorm1d(first_conv_out_channels)

    #third conv layer(kernel size is 1)
    self.conv3 = nn.Conv1d(
      first_conv_out_channels, first_conv_out_channels*4, kernel_size=1, stride=1, padding=0)
    self.bn3 = nn.BatchNorm1d(first_conv_out_channels*4)
    self.activation = activation

    # identity layer adjusts the number of channels of the skip connection.
    self.identity_conv = identity_conv

  def forward(self, x):
    identity = x.clone()

    x = self.conv1(x)
    x = self.bn1(x)
    x = self.activation(x)

    x = self.conv2(x)
    x = self.bn2(x)
    x = self.activation(x)

    x = self.conv3(x)
    x = self.bn3(x)

    if self.identity_conv is not None:
      identity = self.identity_conv(identity)
    x += identity

    x = self.activation(x)

    return x

class OneDResNet50(nn.Module):
  def __init__(self, in_channels=12, activation=nn.ReLU()):
    super(OneDResNet50, self).__init__()

    #first conv layer
    self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3)
    self.bn1 = nn.BatchNorm1d(64)
    self.activation = activation
    self.maxpool = nn.MaxPool1d(kernel_size=5, stride=2, padding=2)
    
    #conv2_x (stride is 1)
    self.conv2_x = self._make_layer(BaseBlock, 3, res_block_in_channels=64, first_conv_out_channels=64, stride=1)

    #conv[3|4|5]_x (stride is 2)
    self.conv3_x = self._make_layer(BaseBlock, 4, res_block_in_channels=256, first_conv_out_channels=128, stride=2)
    self.conv4_x = self._make_layer(BaseBlock, 6, res_block_in_channels=512, first_conv_out_channels=256, stride=2)
    self.conv5_x = self._make_layer(BaseBlock, 3, res_block_in_channels=1024, first_conv_out_channels=512, stride=2)

    self.avgpool = nn.AdaptiveAvgPool1d(1)
    self.fc = nn.Linear(512*4, 1)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.activation(x)
    x = self.maxpool(x)

    x = self.conv2_x(x)
    x = self.conv3_x(x)
    x = self.conv4_x(x)
    x = self.conv5_x(x)
    x = self.avgpool(x)
    x = x.view(x.size(0),-1)
    x = self.fc(x)
    return x

  def _make_layer(self, block, num_res_blocks, res_block_in_channels, first_conv_out_channels, stride):
    layers = []

    identity_conv = nn.Conv1d(res_block_in_channels, first_conv_out_channels*4, kernel_size=1, stride=stride)
    layers.append(block(res_block_in_channels, first_conv_out_channels, identity_conv, stride))

    in_channels = first_conv_out_channels*4

    for i in range(num_res_blocks - 1):
      layers.append(block(in_channels, first_conv_out_channels, identity_conv=None, stride=1))

    return nn.Sequential(*layers)


class OneDResNet50_v2(nn.Module):
  """
    This module is built based on OneDResNet50. 
    The number of last fully connected layers is doubled.
  """
  def __init__(self, in_channels=12, out_features=1, activation=nn.ReLU()):
    super(OneDResNet50_v2, self).__init__()

    #first conv layer
    self.conv1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3)
    self.bn1 = nn.BatchNorm1d(64)
    self.activation = activation
    self.maxpool = nn.MaxPool1d(kernel_size=5, stride=2, padding=2)
    
    #conv2_x (stride is 1)
    self.conv2_x = self._make_layer(BaseBlock, 3, res_block_in_channels=64, first_conv_out_channels=64, stride=1)

    #conv[3|4|5]_x (stride is 2)
    self.conv3_x = self._make_layer(BaseBlock, 4, res_block_in_channels=256, first_conv_out_channels=128, stride=2)
    self.conv4_x = self._make_layer(BaseBlock, 6, res_block_in_channels=512, first_conv_out_channels=256, stride=2)
    self.conv5_x = self._make_layer(BaseBlock, 3, res_block_in_channels=1024, first_conv_out_channels=512, stride=2)

    self.avgpool = nn.AdaptiveAvgPool1d(1)
    self.fc1 = nn.Linear(512*4, 512)
    self.dropout1 = nn.Dropout(p=0.2)
    self.bn2 = nn.BatchNorm1d(512)
    self.fc2 = nn.Linear(512, out_features)

  def forward(self, x):
    x = self.conv1(x)
    x = self.bn1(x)
    x = self.activation(x)
    x = self.maxpool(x)

    x = self.conv2_x(x)
    x = self.conv3_x(x)
    x = self.conv4_x(x)
    x = self.conv5_x(x)
    x = self.avgpool(x)
    x = x.view(x.size(0),-1)
    x = self.fc1(x)
    x = self.activation(x)
    x = self.bn2(x)
    x = self.dropout1(x)
    x = self.fc2(x)

    return x

  def _make_layer(self, block, num_res_blocks, res_block_in_channels, first_conv_out_channels, stride):
    layers = []

    identity_conv = nn.Conv1d(res_block_in_channels, first_conv_out_channels*4, kernel_size=1, stride=stride)
    layers.append(block(res_block_in_channels, first_conv_out_channels, identity_conv, stride))

    in_channels = first_conv_out_channels*4

    for i in range(num_res_blocks - 1):
      layers.append(block(in_channels, first_conv_out_channels, identity_conv=None, stride=1))

    return nn.Sequential(*layers)

class OneDResNet50_2IN(nn.Module):
  def __init__(self, in_channels=12, activation=nn.ReLU()):
    super(OneDResNet50_2IN, self).__init__()

    #first conv layer1
    self.conv1_1 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3)
    self.bn1_1 = nn.BatchNorm1d(64)

    #first conv layer2
    self.conv1_2 = nn.Conv1d(in_channels, 64, kernel_size=7, stride=2, padding=3)
    self.bn1_2 = nn.BatchNorm1d(64)

    self.activation = activation
    self.maxpool = nn.MaxPool1d(kernel_size=5, stride=2, padding=2)
    
    #conv2_x (stride is 1)
    self.conv2_x = self._make_layer(BaseBlock, 3, res_block_in_channels=64, first_conv_out_channels=64, stride=1)

    #conv[3|4|5]_x (stride is 2)
    self.conv3_x = self._make_layer(BaseBlock, 4, res_block_in_channels=256, first_conv_out_channels=128, stride=2)
    self.conv4_x = self._make_layer(BaseBlock, 6, res_block_in_channels=512, first_conv_out_channels=256, stride=2)
    self.conv5_x = self._make_layer(BaseBlock, 3, res_block_in_channels=1024, first_conv_out_channels=512, stride=2)

    self.avgpool = nn.AdaptiveAvgPool1d(1)
    self.fc = nn.Linear(512*4, 1)

  def forward(self, x):
    x1 = x[:,0,:,:]
    x2 = x[:,1,:,:]

    x1 = self.conv1_1(x1)
    x1 = self.bn1_1(x1)
    x1 = self.activation(x1)
    x1 = self.maxpool(x1)

    x2 = self.conv1_1(x2)
    x2 = self.bn1_1(x2)
    x2 = self.activation(x2)
    x2 = self.maxpool(x2)

    x = torch.cat((x1,x2), dim=-1)

    x = self.conv2_x(x)
    x = self.conv3_x(x)
    x = self.conv4_x(x)
    x = self.conv5_x(x)
    x = self.avgpool(x)
    x = x.view(x.size(0),-1)
    x = self.fc(x)
    return x

  def _make_layer(self, block, num_res_blocks, res_block_in_channels, first_conv_out_channels, stride):
    layers = []

    identity_conv = nn.Conv1d(res_block_in_channels, first_conv_out_channels*4, kernel_size=1, stride=stride)
    layers.append(block(res_block_in_channels, first_conv_out_channels, identity_conv, stride))

    in_channels = first_conv_out_channels*4

    for i in range(num_res_blocks - 1):
      layers.append(block(in_channels, first_conv_out_channels, identity_conv=None, stride=1))

    return nn.Sequential(*layers)
