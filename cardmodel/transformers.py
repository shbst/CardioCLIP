import torch
import torch.nn as nn
from torch.optim.lr_scheduler import CosineAnnealingLR
import pytorch_lightning as pl
from .models import OneDResNet50_v2
#from .metrics import get_metrics
from .loss import Losses
from .activations import Activations
import math
from .util import get_multilabel_metrics
import numpy as np

class PositionalEncoding(nn.Module):
    """普通のpositional encoder
    """

    def __init__(self, d_model, vocab_size=10, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(vocab_size, d_model)
        position = torch.arange(0, vocab_size, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float()
            * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)

class TransformerModel(nn.Module):
  """
    TransformerModel.

    Note)
      Time length of input cardiograms varys. 
         example) A cardiogram with 2 secs has 1000 points.(Sampling rate is 0.002 sec/point)
      The input to this model is expected to be the shape of (batch_size, vocab_size, time-length). 
         example) input_shape = (batch_size, 12, 1000)

    Args)
      embedding_model: This projects the input cardiogram into the feature space with dimensions defined by d_model.
      d_model: The number of expected features in the input.
      vocab_size: Number of types of input cardiograms.
                  Basically this should be set to 12.
      nhead: Number of MultiHeadAttention modules. (default 8)
      dim_feedforward: The dimension of the feedforward network model. 
                        (Number of hidden states in feedforward model.)
      num_layers: The number of the attention layers.

    Returns)
      This model returns a scholar value of log(bnp).  
  """
  def __init__(
    self,
    embedding_model,
    d_model=1000,
    vocab_size=10,
    nhead=8,
    dim_feedforward=2048,
    num_layers=12,
    dropout=0.1,
    activation=nn.ReLU(),
    out_features=768,
    ):
    super().__init__()
    self.vocab_size = vocab_size

    assert d_model % nhead == 0, "nheads must be divided evenly into d_model"

    self.emb = embedding_model

    self.layernorm = nn.LayerNorm(d_model)

    self.cls_token = nn.Parameter(torch.randn(1,1,d_model))

    self.pos_encoder = PositionalEncoding(
      d_model=d_model,
      dropout=dropout,
      vocab_size=vocab_size + 1, #add class token
    )

    encoder_layer = nn.TransformerEncoderLayer(
      d_model=d_model,
      nhead=nhead,
      dim_feedforward=dim_feedforward,
      dropout=dropout,
      batch_first=True,
      activation=activation,
    )

    self.transformer_encoder = nn.TransformerEncoder(
      encoder_layer,
      num_layers=num_layers,
    )
    self.predictor = nn.Linear(d_model, out_features)
    self.d_model = d_model


  def forward(self, x):
    """
      x.shape: (batch_size, choosenum, ch, time-length)
    """
    bsize, n, ch,_ = x.shape
    assert n == 1, "choosenum must be 1 in this code."
    x = x.reshape(bsize*ch,1,-1)
    x = self.emb(x)
    x = x.reshape(bsize,ch,-1)
    x = self.layernorm(x)

    cls_tokens = self.cls_token.repeat(bsize,1,1) 
    x = torch.cat((cls_tokens, x), dim=1)

    x = self.pos_encoder(x)
    x = self.transformer_encoder(x)
    x = x[:,0,:]
    x = self.predictor(x)
    return x

class CardTransformer(nn.Module):
  """
    This model predicts feature from a input cardiogram.
  """
  def __init__(self, 
    resemb_features=256, #this must be a multiple of 8(nheads).
    activation='relu',
    out_features=768,
  ):
    super().__init__()

    self.activation = Activations.activations[activation]
    resemb_model = nn.Sequential(
      OneDResNet50_v2(in_channels=1, out_features=resemb_features, activation=self.activation), 
      nn.ReLU(),
      )
    
    self.transformer = TransformerModel(
      d_model=resemb_features,
      embedding_model=resemb_model,
      vocab_size=12,
      activation=self.activation,
      out_features=out_features,
    )

  def forward(self, x):
    x = self.transformer(x)
    return x

class CardTransformerPredictor(pl.LightningModule):
  def __init__(self,
    lr,
    max_epoch,
    cardmodel,
    targets=['BNP'],
    optimizer='adam',
    out_feature_from_transformer=768,
    echo_fetch=None,
    thresholds={'BNP':100},
    ):
    super().__init__()
    self.save_hyperparameters()
    self.transformer = cardmodel
    self.thresholds = thresholds
    self.echo_fetch = echo_fetch
    self.targets = targets
    self.optimizer = optimizer
    self.lr = lr
    self.max_epoch = max_epoch

    self.model = nn.Sequential(
      self.transformer,
      nn.Linear(out_feature_from_transformer, len(targets)),
      )

    self.criterion = nn.BCEWithLogitsLoss()

    self.training_step_outputs = []
    self.validation_step_outputs = []


  def forward(self, x):
    x = self.model(x)
    return x

  def common_step(self, batch):
    _, card, bnp, patientid = batch

    y = []
    mask = [True] * card.shape[0]
    for target in self.targets:
      if target=='BNP':
        fetched_data = bnp
        fetched_data = fetched_data.squeeze()
      else:
        fetched_data = np.array(self.echo_fetch(patientid))
        mask *= np.logical_not(np.isnan(fetched_data))
      y.append(torch.Tensor(fetched_data > self.thresholds[target]).to(card.device))
    y = torch.stack(y)
    y = torch.permute(y, (1, 0)) #y.shape = (2, 32) -> (32, 2)

    card = card[mask]
    y = y[mask]

    output = self.model(card)
    y = y.to(output.dtype)
    loss = self.criterion(output, y)

    return loss, output, y

  def training_step(self, batch, batch_idx):
    loss, output, target = self.common_step(batch)

    pred = {
      'loss': loss,
      'y_hat': output,
      'y': target,
      'batch_loss': loss.item() * target.shape[0],
    }
    self.training_step_outputs.append(pred)
    return pred

  def on_train_epoch_end(self):
    train_step_outputs = self.training_step_outputs
    y_hat = torch.cat([val['y_hat'] for val in train_step_outputs], dim=0)
    y = torch.cat([val['y'] for val in train_step_outputs], dim=0)

    epoch_loss = sum(
        [val['batch_loss'] for val in train_step_outputs]
        ) / y_hat.size(0)

    self.log_dict(
      {
        'loss': epoch_loss,
        **get_multilabel_metrics(y_hat, y, self.targets),
      },
      on_epoch=True
      )
    print("Train [{}/{}] loss:{:.8f}".format(self.current_epoch+1,self.max_epoch,epoch_loss))
    self.training_step_outputs.clear()

  def validation_step(self, batch, batch_idx):
    loss, output, target = self.common_step(batch)

    pred = {
      'loss': loss,
      'y_hat': output,
      'y': target,
      'batch_loss': loss.item() * target.shape[0],
    }
    self.validation_step_outputs.append(pred)
    return pred

  def on_validation_epoch_end(self):
    val_step_outputs = self.validation_step_outputs
    y_hat = torch.cat([val['y_hat'] for val in val_step_outputs], dim=0)
    y = torch.cat([val['y'] for val in val_step_outputs], dim=0)
    epoch_loss = sum(
        [val['batch_loss'] for val in val_step_outputs]
        ) / y_hat.size(0)

    metrics_dict = {f'val_{k}': v for k, v in get_multilabel_metrics(y_hat, y, self.targets).items()}
    self.log_dict(
        {
          'val_loss': epoch_loss,
          **metrics_dict,
        },
        on_epoch=True
        )
    print("Valid [{}/{}] val_loss:{:.8f}".format(self.current_epoch+1,self.max_epoch,epoch_loss))
    self.validation_step_outputs.clear()

  def configure_optimizers(self):
    if self.optimizer == 'adam':
      opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
    elif self.optimizer == 'adamw':
      opt = torch.optim.AdamW(self.model.parameters(), lr=self.lr)
    scheduler = CosineAnnealingLR(opt, T_max=10)
    return [opt], [scheduler]

