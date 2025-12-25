# train_clip.py
import argparse
import importlib
import os
import os.path as osp
import yaml

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from pytorch_lightning.loggers import CSVLogger

from config import CLIPConfig
from transforms import XpCardTransforms
from utils import get_encoders
from clipmodel.wrapper import CustomCLIPWrapper_multitask


def _import_obj(path: str):
    # 'package.module:ClassName'
    if ":" not in path:
        raise ValueError("datamodule_path must be 'package.module:ClassName'")
    mod_name, obj_name = path.split(":")
    mod = importlib.import_module(mod_name)
    return getattr(mod, obj_name)


def parse_args():
    p = argparse.ArgumentParser("CardioCLIP Step-1 training (wandb-free, no CSV label generators)")
    p.add_argument("--config", type=str, default="configs/default.yaml")

    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--run_name", type=str, required=True)
    p.add_argument("--seed", type=int, default=0)

    # user DataModule injection
    p.add_argument("--datamodule_path", type=str, required=True)
    p.add_argument("--datamodule_kwargs", type=str, default=None)

    # trainer
    p.add_argument("--accelerator", type=str, default="gpu", choices=["cpu", "gpu", "mps"])
    p.add_argument("--devices", type=str, default="0")  # "0" or "0,1" or "auto"
    p.add_argument("--precision", type=str, default="16-mixed")

    return p.parse_args()


def main():
    args = parse_args()
    pl.seed_everything(args.seed, workers=True)

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    clip_cfg = CLIPConfig(**cfg.get("clip", {}))
    pm = cfg.get("params", {})

    out_root = osp.join(args.output_dir, args.run_name)
    ckpt_dir = osp.join(out_root, "checkpoints")
    log_dir = osp.join(out_root, "logs")
    tmp_dir = osp.join(out_root, "tmp_save")
    os.makedirs(ckpt_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(tmp_dir, exist_ok=True)

    # save merged run config
    with open(osp.join(out_root, "hparams.yaml"), "w", encoding="utf-8") as f:
        yaml.safe_dump({"config": cfg, "overrides": vars(args)}, f, sort_keys=False, allow_unicode=True)

    # encoders
    img_encoder, card_encoder, shared_net = get_encoders(
        pm["clip_model"],
        pretrained=pm.get("pretrained", True),
        freeze_parameters=pm.get("freeze_parameters", False),
        use_shared_net=pm.get("use_shared_net", True),
    )

    # transforms (optional; if your DataModule handles transforms, you can remove this entirely)
    xpcardtransform = XpCardTransforms(
        clip_cfg,
        noise=(-pm.get("noise", 0.0), pm.get("noise", 0.0)),
        randomcropscale=pm.get("randomcropscale", 0.0),
        randomrotation=pm.get("randomrotation", 0.0),
        randomerase=pm.get("randomerase", 0.0),
    )

    # user DataModule
    DMClass = _import_obj(args.datamodule_path)
    dm_kwargs = {}
    if args.datamodule_kwargs:
        with open(args.datamodule_kwargs, "r", encoding="utf-8") as f:
            dm_kwargs = yaml.safe_load(f) or {}

    # optional: pass transforms if DataModule accepts them
    dm_kwargs.setdefault("train_transform", xpcardtransform.get_transform(mode="train"))
    dm_kwargs.setdefault("valid_transform", xpcardtransform.get_transform(mode="valid"))
    dm_kwargs.setdefault("batch_size", clip_cfg.batch_size)
    dm_kwargs.setdefault("num_workers", clip_cfg.num_workers)

    dm = DMClass(**dm_kwargs)

    # model (IMPORTANT: wrapper must read y/mask from batch; no CSV target funcs)
    config_path = "./clipmodel/configs/RN.yaml" if "RN" in pm["clip_model"] else "./clipmodel/configs/ViT.yaml"

    model = CustomCLIPWrapper_multitask(
        img_encoder=img_encoder,
        card_encoder=card_encoder,
        minibatch_size=clip_cfg.batch_size,
        config_path=config_path,
        model_name=pm["clip_model"],
        learning_rate=pm.get("lr", 5e-4),
        avg_word_embs=True,
        tmp_save_dir=tmp_dir,
        image_card_switch_ratio=pm.get("image_card_switch_ratio", 0.0),
        shared_net=shared_net,
        # ↓↓↓ ここが変更点：CSV由来のターゲット定義を渡さない
        target_funcs=None,
        thresholds=None,
        lowerthebetters=None,
    )

    callbacks = []
    if pm.get("earlystopping", True):
        callbacks.append(
            EarlyStopping(
                monitor="val_loss",
                mode="min",
                patience=int(pm.get("earlystopping_patience", 10)),
                check_finite=True,
            )
        )

    callbacks.append(
        ModelCheckpoint(
            monitor="val_loss",
            dirpath=ckpt_dir,
            filename="{epoch:03d}-{val_loss:.4f}",
            save_top_k=1,
            mode="min",
            every_n_epochs=1,
            save_last=True,
        )
    )

    logger = CSVLogger(save_dir=log_dir, name="", version="")

    devices = "auto" if args.devices == "auto" else [int(x) for x in args.devices.split(",") if x.strip() != ""]
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        devices=devices,
        precision=args.precision,
        max_epochs=int(pm.get("max_epochs", 50)),
        min_epochs=int(pm.get("min_epochs", 1)),
        check_val_every_n_epoch=1,
        log_every_n_steps=10,
        num_sanity_val_steps=0,
        logger=logger,
        callbacks=callbacks,
        default_root_dir=out_root,
    )

    trainer.fit(model, dm)


if __name__ == "__main__":
    main()

