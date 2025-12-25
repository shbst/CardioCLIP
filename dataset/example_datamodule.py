from typing import Optional
import pytorch_lightning as pl
from torch.utils.data import DataLoader, Dataset
import torch


class _DummyDataset(Dataset):
    """
    公開用の最小stub。
    実際にはユーザーが自分のDatasetに差し替えてください。

    __getitem__ は wrapper.py の期待に合わせて
    (image, card, dummy, patientid, examdate) を返します。
    """
    def __init__(self, n=128, image_shape=(1, 224, 224), card_shape=(1000, 12)):
        self.n = n
        self.image_shape = image_shape
        self.card_shape = card_shape

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        image = torch.randn(*self.image_shape)          # CXR tensor
        card  = torch.randn(*self.card_shape)           # ECG tensor
        dummy = 0
        patientid = idx                                 # int or str
        examdate  = "20000101"                          # YYYYMMDD string
        return image, card, dummy, patientid, examdate


class ExampleDataModule(pl.LightningDataModule):
    def __init__(
        self,
        xpconfig,
        cardconfig,
        train_transform=None,
        valid_transform=None,
        batch_size: int = 32,
        num_workers: int = 4,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transform = train_transform
        self.valid_transform = valid_transform

    def setup(self, stage: Optional[str] = None):
        self.train_ds = _DummyDataset(n=256)
        self.valid_ds = _DummyDataset(n=64)

    def train_dataloader(self):
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.valid_ds, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

