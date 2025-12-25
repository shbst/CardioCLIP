# config.py
from dataclasses import dataclass


@dataclass
class CLIPConfig:
    batch_size: int = 32
    num_workers: int = 4

