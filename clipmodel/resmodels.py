import torch
import torch.nn as nn
import torch.nn.functional as F

class FCResBlock(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int = None, p_drop: float = 0.0):
        super().__init__()
        hidden_dim = hidden_dim or max(in_dim, out_dim)

        self.fc1 = nn.Linear(in_dim, hidden_dim)
        self.bn1 = nn.BatchNorm1d(hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)
        self.bn2 = nn.BatchNorm1d(out_dim)
        self.act = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_drop) if p_drop > 0 else nn.Identity()

        # 投影ショートカット（次元が違うときに使用）
        if in_dim != out_dim:
            self.shortcut = nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.dropout(out)
        out = self.fc2(out)
        out = self.bn2(out)
        out = out + identity
        out = self.act(out)
        return out


class MultiTaskModel(nn.Module):
    def __init__(self, in_dim: int, n_tasks: int, p_drop: float = 0.0):
        super().__init__()
        self.input_layer = nn.Linear(in_dim, 512)
        self.bn0 = nn.BatchNorm1d(512)
        self.act = nn.ReLU(inplace=True)

        # 512→512→512→256→256 のように、途中で次元を下げる
        self.res_blocks = nn.Sequential(
            FCResBlock(512, 512, hidden_dim=512, p_drop=p_drop),
            FCResBlock(512, 512, hidden_dim=512, p_drop=p_drop),
            FCResBlock(512, 256, hidden_dim=512, p_drop=p_drop),  # ← 次元変更（投影あり）
            FCResBlock(256, 256, hidden_dim=256, p_drop=p_drop),
        )

        self.output_layer = nn.Linear(256, n_tasks)

    def forward(self, x):
        h = self.input_layer(x)   # [B, 512]
        h = self.bn0(h)
        h = self.act(h)
        h = self.res_blocks(h)    # 最終は [B, 256]
        logits = self.output_layer(h)  # [B, n_tasks]
        return logits  # BCEWithLogitsLossに渡すロジット

