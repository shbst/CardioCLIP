# clipmodel/wrapper.py
# Public-ready wrapper (wandb-free, NO CSV/compat-mode label generation)
# - Contrastive CLIP-style loss (symmetric)
# - Optional "switch" loss (swap modality embeddings for k samples per batch)
# - Optional multitask BCE loss where labels (y, mask) are provided by the DataModule batch
#
# Expected batch formats:
#   Dict:
#     {"image": Tensor[B,C,H,W], "card": Tensor[B,D,K], "y": Tensor[B,T], "mask": Tensor[B,T] (optional)}
#   Tuple:
#     (image, card) or (image, card, y) or (image, card, y, mask)
#
# IMPORTANT:
# - This file no longer uses patientid/examdate and does NOT generate labels.
# - If you enable multitask loss, your DataModule must provide y (and optional mask).
#
from __future__ import annotations

import os
import os.path as osp
from dataclasses import asdict
from typing import Any, Dict, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from .clip import CLIP
from .resmodels import MultiTaskModel

try:
    # Optional scheduler (keep import guarded for public release)
    from cosine_annealing_warmup import CosineAnnealingWarmupRestarts  # type: ignore
except Exception:
    CosineAnnealingWarmupRestarts = None


BatchType = Union[
    Dict[str, torch.Tensor],
    Tuple[torch.Tensor, ...],
]


def _ensure_dir(path: str) -> None:
    if path is None:
        return
    os.makedirs(path, exist_ok=True)


class CLIPWrapper(pl.LightningModule):
    """
    Base CLIP wrapper:
    - Builds CLIP model from encoders
    - Computes symmetric contrastive loss between image and card embeddings
    """

    def __init__(
        self,
        image_encoder: nn.Module,
        card_encoder: nn.Module,
        minibatch_size: int,
        config_path: str,
        model_name: str,
        learning_rate: float = 5e-4,
        avg_word_embs: bool = True,
        tmp_save_dir: Optional[str] = None,
        image_card_switch_ratio: float = 0.0,
        shared_net: Optional[nn.Module] = None,
        temperature_init: float = 1 / 0.07,
        use_scheduler: bool = False,
        scheduler_t0: int = 10,
        scheduler_tmult: int = 2,
        scheduler_warmup: int = 2,
        scheduler_gamma: float = 0.5,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["image_encoder", "card_encoder", "shared_net"])

        self.learning_rate = float(learning_rate)
        self.minibatch_size = int(minibatch_size)
        self.model_name = str(model_name)
        self.config_path = str(config_path)

        # CLIP model (your local implementation)
        self.model = CLIP(
            image_encoder=image_encoder,
            card_encoder=card_encoder,
            shared_net=shared_net,
            config_path=config_path,
            model_name=model_name,
            avg_word_embs=avg_word_embs,
            temperature_init=temperature_init,
        )

        self.tmp_save_dir = tmp_save_dir
        self.image_card_switch_ratio = float(image_card_switch_ratio)

        # Manual optimization (keeps your original "optimizer.step() inside training_step" style)
        self.automatic_optimization = False

        # Scheduler toggles (optional)
        self.use_scheduler = bool(use_scheduler)
        self.scheduler_t0 = int(scheduler_t0)
        self.scheduler_tmult = int(scheduler_tmult)
        self.scheduler_warmup = int(scheduler_warmup)
        self.scheduler_gamma = float(scheduler_gamma)

    # -------------------------
    # Batch parsing helpers
    # -------------------------
    def _parse_inputs(self, batch: BatchType) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(batch, dict):
            # preferred keys for CLIP stage
            if "image" in batch and "card" in batch:
                return batch["image"], batch["card"]
            # allow alternative naming
            if "cxr" in batch and "ecg" in batch:
                return batch["cxr"], batch["ecg"]
            raise KeyError("Batch dict must contain ('image','card') or ('cxr','ecg').")
        # tuple
        if len(batch) < 2:
            raise ValueError("Tuple batch must be at least (image, card).")
        return batch[0], batch[1]

    # -------------------------
    # Core loss functions
    # -------------------------
    def _contrastive_logits(
        self, z_a: torch.Tensor, z_b: torch.Tensor
    ) -> torch.Tensor:
        # z_a, z_b: [B, d] normalized
        logit_scale = self.model.logit_scale.exp()
        return logit_scale * (z_a @ z_b.t())

    def _symmetric_clip_loss(
        self, z_img: torch.Tensor, z_card: torch.Tensor
    ) -> torch.Tensor:
        logits = self._contrastive_logits(z_img, z_card)
        gt = torch.arange(logits.size(0), device=logits.device)
        loss_i2c = F.cross_entropy(logits, gt)
        loss_c2i = F.cross_entropy(logits.t(), gt)
        return 0.5 * (loss_i2c + loss_c2i)

    def _switch_symmetric_loss(
        self, z_img: torch.Tensor, z_card: torch.Tensor, k: int
    ) -> torch.Tensor:
        """
        Switch loss:
        randomly select k indices and swap modality roles for those indices
        within the symmetric softmax objective.

        This is a simplified version to stay public-friendly.
        """
        B = z_img.size(0)
        if k <= 0 or k >= B:
            return self._symmetric_clip_loss(z_img, z_card)

        # Pick indices
        perm = torch.randperm(B, device=z_img.device)
        I = perm[:k]       # swapped
        J = perm[k:]       # normal

        # Construct swapped view
        # For indices in I, treat z_img[i] as if it were from card and z_card[i] as if from img
        z_img2 = z_img.clone()
        z_card2 = z_card.clone()
        z_img2[I] = z_card[I]
        z_card2[I] = z_img[I]

        return self._symmetric_clip_loss(z_img2, z_card2)

    # -------------------------
    # Lightning hooks
    # -------------------------
    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.learning_rate)

        if self.use_scheduler and CosineAnnealingWarmupRestarts is not None:
            sch = CosineAnnealingWarmupRestarts(
                opt,
                first_cycle_steps=self.scheduler_t0,
                cycle_mult=self.scheduler_tmult,
                max_lr=self.learning_rate,
                min_lr=self.learning_rate * 0.01,
                warmup_steps=self.scheduler_warmup,
                gamma=self.scheduler_gamma,
            )
            return {"optimizer": opt, "lr_scheduler": {"scheduler": sch, "interval": "epoch"}}

        return opt

    def on_train_epoch_start(self):
        # Make sure tmp dir exists if used
        if self.tmp_save_dir is not None:
            _ensure_dir(self.tmp_save_dir)

    def training_step(self, batch: BatchType, batch_idx: int):
        opt = self.optimizers()

        image, card = self._parse_inputs(batch)

        # Encode
        z_img = self.model.encode_image(image)
        z_card = self.model.encode_card(card)

        # Shared net (if present inside CLIP) already applied in CLIP implementation;
        # if your CLIP returns unshared features, keep it there.
        z_img = F.normalize(z_img, dim=1)
        z_card = F.normalize(z_card, dim=1)

        # CLIP loss (optionally switch)
        if self.image_card_switch_ratio > 0:
            k = int(round(self.image_card_switch_ratio * z_img.size(0)))
            clip_loss = self._switch_symmetric_loss(z_img, z_card, k=k)
        else:
            clip_loss = self._symmetric_clip_loss(z_img, z_card)

        loss = clip_loss

        self.manual_backward(loss)
        opt.step()
        opt.zero_grad()

        # scheduler step if present
        sch = self.lr_schedulers()
        if sch is not None:
            try:
                sch.step()
            except Exception:
                pass

        self.log("loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("clip_loss", clip_loss, prog_bar=False, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: BatchType, batch_idx: int):
        image, card = self._parse_inputs(batch)

        z_img = F.normalize(self.model.encode_image(image), dim=1)
        z_card = F.normalize(self.model.encode_card(card), dim=1)

        if self.image_card_switch_ratio > 0:
            k = int(round(self.image_card_switch_ratio * z_img.size(0)))
            clip_loss = self._switch_symmetric_loss(z_img, z_card, k=k)
        else:
            clip_loss = self._symmetric_clip_loss(z_img, z_card)

        self.log("val_loss", clip_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_clip_loss", clip_loss, prog_bar=False, on_step=False, on_epoch=True)
        return clip_loss


class CustomCLIPWrapper_multitask(CLIPWrapper):
    """
    CLIP wrapper + multitask BCE loss.

    Labels must be provided by the DataModule batch:
      - y: Tensor[B, T] (float 0/1)
      - mask: Tensor[B, T] (optional; 1 valid, 0 missing)
    """

    def __init__(
        self,
        image_encoder: nn.Module,
        card_encoder: nn.Module,
        minibatch_size: int,
        config_path: str,
        model_name: str,
        learning_rate: float = 5e-4,
        avg_word_embs: bool = True,
        tmp_save_dir: Optional[str] = None,
        image_card_switch_ratio: float = 0.0,
        shared_net: Optional[nn.Module] = None,
        hidden_nodes: int = 768,
        num_tasks: Optional[int] = None,  # REQUIRED if you want multitask loss
        multitask_weight: float = 1.0,
        use_scheduler: bool = False,
        scheduler_t0: int = 10,
        scheduler_tmult: int = 2,
        scheduler_warmup: int = 2,
        scheduler_gamma: float = 0.5,
    ):
        super().__init__(
            image_encoder=image_encoder,
            card_encoder=card_encoder,
            minibatch_size=minibatch_size,
            config_path=config_path,
            model_name=model_name,
            learning_rate=learning_rate,
            avg_word_embs=avg_word_embs,
            tmp_save_dir=tmp_save_dir,
            image_card_switch_ratio=image_card_switch_ratio,
            shared_net=shared_net,
            use_scheduler=use_scheduler,
            scheduler_t0=scheduler_t0,
            scheduler_tmult=scheduler_tmult,
            scheduler_warmup=scheduler_warmup,
            scheduler_gamma=scheduler_gamma,
        )

        self.hidden_nodes = int(hidden_nodes)
        self.num_tasks = num_tasks  # if None, multitask is disabled unless user sets later
        self.multitask_weight = float(multitask_weight)

        # Multi-task head
        self.multitask_model: Optional[nn.Module] = None
        if self.num_tasks is not None:
            self.multitask_model = MultiTaskModel(in_dim=self.hidden_nodes, n_tasks=int(self.num_tasks))

        self.multitask_criterion = nn.BCEWithLogitsLoss(reduction="none")

    def _parse_multitask(self, batch: BatchType) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        y = None
        mask = None
        if isinstance(batch, dict):
            y = batch.get("y", None)
            mask = batch.get("mask", None)
        else:
            # tuple
            if len(batch) >= 3:
                y = batch[2]
            if len(batch) >= 4:
                mask = batch[3]

        if y is None:
            return None, None
        if mask is None:
            mask = torch.ones_like(y)
        return y, mask

    def _masked_bce_with_logits(self, logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # logits,y,mask: [B,T]
        loss = self.multitask_criterion(logits, y)
        loss = loss * mask
        denom = torch.clamp(mask.sum(), min=1.0)
        return loss.sum() / denom

    def training_step(self, batch: BatchType, batch_idx: int):
        opt = self.optimizers()

        image, card = self._parse_inputs(batch)
        y, mask = self._parse_multitask(batch)

        # Encode (CLIP embeddings)
        z_img = F.normalize(self.model.encode_image(image), dim=1)
        z_card = F.normalize(self.model.encode_card(card), dim=1)

        # CLIP loss (optionally switch)
        if self.image_card_switch_ratio > 0:
            k = int(round(self.image_card_switch_ratio * z_img.size(0)))
            clip_loss = self._switch_symmetric_loss(z_img, z_card, k=k)
        else:
            clip_loss = self._symmetric_clip_loss(z_img, z_card)

        multitask_loss = torch.zeros((), device=z_img.device)

        # Multitask loss if labels provided
        if y is not None:
            if self.multitask_model is None:
                raise ValueError(
                    "Multitask labels (y) were provided by the DataModule, but num_tasks was not set. "
                    "Initialize CustomCLIPWrapper_multitask with num_tasks=T."
                )
            # Predict from both modality embeddings (as in your original design)
            logits_img = self.multitask_model(z_img)
            logits_card = self.multitask_model(z_card)

            multitask_loss = (
                self._masked_bce_with_logits(logits_img, y, mask) +
                self._masked_bce_with_logits(logits_card, y, mask)
            )

        loss = clip_loss + self.multitask_weight * multitask_loss

        self.manual_backward(loss)
        opt.step()
        opt.zero_grad()

        sch = self.lr_schedulers()
        if sch is not None:
            try:
                sch.step()
            except Exception:
                pass

        self.log("loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log("clip_loss", clip_loss, prog_bar=False, on_step=True, on_epoch=True)
        self.log("multitask_loss", multitask_loss, prog_bar=False, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch: BatchType, batch_idx: int):
        image, card = self._parse_inputs(batch)
        y, mask = self._parse_multitask(batch)

        z_img = F.normalize(self.model.encode_image(image), dim=1)
        z_card = F.normalize(self.model.encode_card(card), dim=1)

        if self.image_card_switch_ratio > 0:
            k = int(round(self.image_card_switch_ratio * z_img.size(0)))
            clip_loss = self._switch_symmetric_loss(z_img, z_card, k=k)
        else:
            clip_loss = self._symmetric_clip_loss(z_img, z_card)

        multitask_loss = torch.zeros((), device=z_img.device)
        if y is not None:
            if self.multitask_model is None:
                raise ValueError(
                    "Multitask labels (y) were provided by the DataModule, but num_tasks was not set. "
                    "Initialize CustomCLIPWrapper_multitask with num_tasks=T."
                )
            logits_img = self.multitask_model(z_img)
            logits_card = self.multitask_model(z_card)
            multitask_loss = (
                self._masked_bce_with_logits(logits_img, y, mask) +
                self._masked_bce_with_logits(logits_card, y, mask)
            )

        val_loss = clip_loss + self.multitask_weight * multitask_loss

        self.log("val_loss", val_loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log("val_clip_loss", clip_loss, prog_bar=False, on_step=False, on_epoch=True)
        self.log("val_multitask_loss", multitask_loss, prog_bar=False, on_step=False, on_epoch=True)
        return val_loss

