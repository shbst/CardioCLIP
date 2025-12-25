# train_sequential.py
import argparse
import importlib
import os
import os.path as osp
import yaml

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger

from config import CLIPConfig
from utils import get_encoders


def _import_obj(path: str):
    # 'package.module:ClassName'
    if ":" not in path:
        raise ValueError("datamodule_path must be 'package.module:ClassName'")
    mod_name, obj_name = path.split(":")
    mod = importlib.import_module(mod_name)
    return getattr(mod, obj_name)


class SequentialPredictor(pl.LightningModule):
    def __init__(
        self,
        ecg_encoder,
        cxr_encoder,
        shared_net,
        num_labels: int,
        hidden_size: int = 256,
        lr: float = 1e-4,
        order: str = "ecg_cxr",  # ecg_cxr / cxr_ecg / both
        use_shared_net: bool = True,
        freeze_encoders: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["ecg_encoder", "cxr_encoder", "shared_net"])

        self.ecg_encoder = ecg_encoder
        self.cxr_encoder = cxr_encoder
        self.shared_net = shared_net
        self.use_shared_net = use_shared_net

        if freeze_encoders:
            for p in self.ecg_encoder.parameters():
                p.requires_grad = False
            for p in self.cxr_encoder.parameters():
                p.requires_grad = False
            for p in self.shared_net.parameters():
                p.requires_grad = False

        # Infer embedding dim by a dummy forward is risky; assume encoders output 768 (your code path).
        emb_dim = 768

        self.gru = torch.nn.GRU(
            input_size=emb_dim,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
        )
        self.head = torch.nn.Linear(hidden_size, num_labels)
        self.loss_fn = torch.nn.BCEWithLogitsLoss(reduction="none")

    def _parse_batch(self, batch):
        if isinstance(batch, dict):
            ecg = batch["ecg"]
            cxr = batch["cxr"]
            y = batch["y"]
            mask = batch.get("mask", None)
        else:
            if len(batch) == 3:
                ecg, cxr, y = batch
                mask = None
            elif len(batch) == 4:
                ecg, cxr, y, mask = batch
            else:
                raise ValueError("Batch must be dict or tuple of (ecg, cxr, y[, mask])")
        if mask is None:
            mask = torch.ones_like(y)
        return ecg, cxr, y, mask

    @torch.no_grad()
    def _encode(self, ecg, cxr):
        # encoders expect specific shapes; adapt here if needed in your encoders
        z_ecg = self.ecg_encoder(ecg)
        z_cxr = self.cxr_encoder(cxr)
        if self.use_shared_net and self.shared_net is not None:
            z_ecg = self.shared_net(z_ecg)
            z_cxr = self.shared_net(z_cxr)
        return z_ecg, z_cxr

    def _forward_order(self, z_first, z_second):
        # sequence length 2: [first, second]
        seq = torch.stack([z_first, z_second], dim=1)  # [B, 2, d]
        h, _ = self.gru(seq)  # [B, 2, hidden]
        logits_step1 = self.head(h[:, 0, :])  # after first exam
        logits_step2 = self.head(h[:, 1, :])  # after second exam
        return logits_step1, logits_step2

    def _masked_bce(self, logits, y, mask):
        per_elem = self.loss_fn(logits, y)  # [B, T]
        per_elem = per_elem * mask
        denom = torch.clamp(mask.sum(), min=1.0)
        return per_elem.sum() / denom

    def training_step(self, batch, batch_idx):
        ecg, cxr, y, mask = self._parse_batch(batch)

        with torch.no_grad():
            z_ecg, z_cxr = self._encode(ecg, cxr)

        losses = []

        if self.hparams.order in ("ecg_cxr", "both"):
            l1, l2 = self._forward_order(z_ecg, z_cxr)
            loss_ecg = self._masked_bce(l1, y, mask)
            loss_ecg_cxr = self._masked_bce(l2, y, mask)
            losses.append(0.5 * (loss_ecg + loss_ecg_cxr))

        if self.hparams.order in ("cxr_ecg", "both"):
            l1, l2 = self._forward_order(z_cxr, z_ecg)
            loss_cxr = self._masked_bce(l1, y, mask)
            loss_cxr_ecg = self._masked_bce(l2, y, mask)
            losses.append(0.5 * (loss_cxr + loss_cxr_ecg))

        loss = torch.stack(losses).mean()

        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        ecg, cxr, y, mask = self._parse_batch(batch)

        with torch.no_grad():
            z_ecg, z_cxr = self._encode(ecg, cxr)

        losses = []

        if self.hparams.order in ("ecg_cxr", "both"):
            l1, l2 = self._forward_order(z_ecg, z_cxr)
            losses.append(0.5 * (self._masked_bce(l1, y, mask) + self._masked_bce(l2, y, mask)))

        if self.hparams.order in ("cxr_ecg", "both"):
            l1, l2 = self._forward_order(z_cxr, z_ecg)
            losses.append(0.5 * (self._masked_bce(l1, y, mask) + self._masked_bce(l2, y, mask)))

        val_loss = torch.stack(losses).mean()
        self.log("val_loss", val_loss, prog_bar=True)
        return val_loss

    def configure_optimizers(self):
        # train predictor only (encoders frozen by default)
        opt = torch.optim.AdamW(self.parameters(), lr=self.hparams.lr)
        return opt


def parse_args():
    p = argparse.ArgumentParser("Sequential Prediction Step (no compat mode, no wandb)")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--run_name", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)

    p.add_argument("--datamodule_path", type=str, required=True,
                   help="package.module:ClassName")
    p.add_argument("--datamodule_kwargs", type=str, default=None,
                   help="YAML file with kwargs for DataModule")

    p.add_argument("--clip_checkpoint", type=str, required=True)
    p.add_argument("--num_labels", type=int, required=True)

    p.add_argument("--order", type=str, default="both", choices=["ecg_cxr", "cxr_ecg", "both"])

    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--hidden_size", type=int, default=256)
    p.add_argument("--freeze_encoders", action="store_true")
    p.add_argument("--use_shared_net", action="store_true")

    # trainer args
    p.add_argument("--accelerator", type=str, default="gpu", choices=["cpu", "gpu", "mps"])
    p.add_argument("--devices", type=str, default="0", help='e.g. "0" or "0,1" or "auto"')
    p.add_argument("--precision", type=str, default="16-mixed")
    p.add_argument("--max_epochs", type=int, default=50)
    p.add_argument("--min_epochs", type=int, default=1)
    p.add_argument("--earlystopping_patience", type=int, default=10)

    return p.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)

    out_root = osp.join(args.output_dir, args.run_name)
    ckpt_dir = osp.join(out_root, "checkpoints")
    log_dir = osp.join(out_root, "logs")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # save hparams
    with open(osp.join(out_root, "hparams.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump(vars(args), f, sort_keys=False, allow_unicode=True)

    # --------------------------
    # Load Step-1 checkpoint & restore encoders
    # --------------------------
    checkpoint = torch.load(args.clip_checkpoint, map_location="cpu")
    model_name = checkpoint["hyper_parameters"]["model_name"]

    cxr_encoder, ecg_encoder, shared_net = get_encoders(model_name, pretrained=False)

    # restore weights (same style as your original script)
    ecg_w = {k.replace("model.cardtransformer.", ""): v
             for k, v in checkpoint["state_dict"].items()
             if k.startswith("model.cardtransformer")}
    ecg_encoder.load_state_dict(ecg_w)
    ecg_encoder.eval()

    cxr_w = {k.replace("model.visual.", ""): v
             for k, v in checkpoint["state_dict"].items()
             if k.startswith("model.visual")}
    cxr_encoder.load_state_dict(cxr_w)
    cxr_encoder.eval()

    shared_w = {k.replace("model.shared_net.", ""): v
                for k, v in checkpoint["state_dict"].items()
                if k.startswith("model.shared_net")}
    if len(shared_w) > 0:
        shared_net.load_state_dict(shared_w)
        shared_net.eval()

    # --------------------------
    # Build user DataModule
    # --------------------------
    DMClass = _import_obj(args.datamodule_path)
    dm_kwargs = {}
    if args.datamodule_kwargs:
        with open(args.datamodule_kwargs, "r", encoding="utf-8") as f:
            dm_kwargs = yaml.safe_load(f) or {}
    dm = DMClass(**dm_kwargs)

    # --------------------------
    # Model
    # --------------------------
    model = SequentialPredictor(
        ecg_encoder=ecg_encoder,
        cxr_encoder=cxr_encoder,
        shared_net=shared_net,
        num_labels=args.num_labels,
        hidden_size=args.hidden_size,
        lr=args.lr,
        order=args.order,
        use_shared_net=args.use_shared_net,
        freeze_encoders=args.freeze_encoders,
    )

    # --------------------------
    # Trainer
    # --------------------------
    callbacks = [
        EarlyStopping(monitor="val_loss", mode="min", patience=args.earlystopping_patience, check_finite=True),
        ModelCheckpoint(monitor="val_loss", dirpath=ckpt_dir, filename="{epoch:03d}-{val_loss:.4f}",
                        save_top_k=1, mode="min", save_last=True),
    ]
    logger = CSVLogger(save_dir=log_dir, name="", version="")

    devices = "auto" if args.devices == "auto" else [int(x) for x in args.devices.split(",") if x.strip() != ""]
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=devices,
        precision=args.precision,
        max_epochs=args.max_epochs,
        min_epochs=args.min_epochs,
        callbacks=callbacks,
        logger=logger,
        default_root_dir=out_root,
        log_every_n_steps=10,
        num_sanity_val_steps=0,
    )

    trainer.fit(model, dm)


if __name__ == "__main__":
    main()

