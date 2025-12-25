# CardioCLIP (Public Release)

This repository provides a **public, dataset-agnostic implementation** of **CardioCLIP**,
including:

- **Step 1: CLIP-style contrastive learning** between ECG and chest X-ray (CXR)
- **Step 2: Sequential Prediction Step** using a GRU-based predictor

All dataset-specific logic (CSV-based label generation, hospital databases, patient identifiers)
has been **completely removed**.  
Instead, **labels are provided directly by user-defined PyTorch Lightning DataModules**.

---

## Key Design Principles

- ❌ No CSV-based label generation (no `LabDataGenerator`, `EchoDataGenerator`)
- ❌ No `wandb` dependency
- ✅ Dataset-agnostic: users inject their own `LightningDataModule`
- ✅ Safe for public release (no patient identifiers, no private schemas)

---

## Installation

```bash
pip install -r requirements.txt
```

Minimum dependencies:
- torch
- torchvision
- pytorch-lightning
- numpy
- pyyaml

---

## Expected DataModule Interface (IMPORTANT)

Both **Step 1 (CLIP)** and **Step 2 (Sequential)** expect the DataLoader to return one of the following.

### Recommended (dict-based batch)

```python
batch = {
    "image": Tensor[B, C, H, W],   # chest X-ray
    "card":  Tensor[B, D, K],      # ECG (time × leads)
    "y":     Tensor[B, T],         # multitask binary labels (0/1 float)
    "mask":  Tensor[B, T],         # optional (1=valid, 0=missing)
}
```

### Tuple-based batch (also supported)

```python
(image, card, y, mask)   # mask optional
```

- `T` is the number of tasks / labels.
- If `mask` is omitted, all labels are treated as valid.

---

## Step 1: CLIP Training (Contrastive Learning)

### Command

```bash
python train_clip.py \
  --config configs/default.yaml \
  --output_dir outputs \
  --run_name clip_run \
  --datamodule_path your_pkg.your_dm:YourDataModule \
  --datamodule_kwargs configs/datamodule_kwargs.yaml
```

### Outputs

```
outputs/clip_run/
├── checkpoints/
├── logs/
└── hparams.yaml
```

---

## Step 2: Sequential Prediction Step

The sequential predictor learns to update disease probability estimates as modalities
are observed sequentially (e.g., ECG → CXR).

### Command

```bash
python train_sequential.py \
  --output_dir outputs \
  --run_name seq_run \
  --clip_checkpoint outputs/clip_run/checkpoints/last.ckpt \
  --datamodule_path your_pkg.your_dm:YourDataModule \
  --datamodule_kwargs configs/datamodule_kwargs.yaml \
  --num_labels 3 \
  --order both
```

### Sequential Orders

- `ecg_cxr` : ECG → CXR
- `cxr_ecg` : CXR → ECG
- `both`    : both orders (loss averaged)

---

## Reproducibility

- All runtime arguments are saved to `hparams.yaml`
- Set random seeds using `--seed`
- Encoders can be frozen or fine-tuned depending on configuration

---

## Repository Structure (Core Files)

```
CardioCLIP/
├── train_clip.py
├── train_sequential.py
├── config.py
├── utils.py
├── clipmodel/
│   ├── wrapper.py
│   ├── clip.py
│   └── resmodels.py
└── README.md
```
