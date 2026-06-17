# train_coty_classifier_v4.py

## Overview

Training script (v4) for the binary **Canola Cotyledon (cn_coty) vs Non-Coty** classifier. The predecessor to v5, this version introduced AdamW, label smoothing, LR warmup + cosine annealing, mixed precision, strong augmentation, MixUp, and EMA.

## Improvements Over v3

- AdamW optimizer + weight decay
- Label smoothing (default 0.1)
- LR warmup → CosineAnnealingLR schedule
- Mixed precision training (AMP)
- Stronger augmentation: `RandAugment`, `RandomErasing`, `RandomAffine`, stronger `ColorJitter`
- MixUp (optional, on by default at α=0.2, prob=0.5)
- EMA of model weights; evaluation on EMA model
- Class-weighted cross-entropy loss for imbalanced classes
- Best checkpoint selected by **val F1-macro** (not raw accuracy)
- Val transform aligned with production inference (`Resize((H,W))` force-stretch, no center-crop)
- `drop_last=True` on train loader; seeded generator for reproducibility

## Usage

```bash
python train_coty_classifier_v4.py \
    --model efficientnet_b0 \
    --data  /home/rameen/final_dataset_v7_1to1_final \
    --name  coty_effnetb0_v4 \
    --epochs 60 --batch 64
```

## Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--model` | `efficientnet_b0` | Architecture (see supported models below) |
| `--data` | required | Dataset root with `train/<class>/` and `val/<class>/` |
| `--epochs` | `60` | Maximum training epochs |
| `--batch` | `64` | Batch size |
| `--lr0` | `1e-3` | Initial learning rate |
| `--weight_decay` | `1e-4` | AdamW weight decay |
| `--label_smoothing` | `0.1` | Cross-entropy label smoothing |
| `--warmup_epochs` | `3` | Linear warmup epochs before cosine annealing |
| `--class_weights` | `balanced` | `balanced` (inverse-freq) or `none` |
| `--mixup_alpha` | `0.2` | MixUp beta parameter (0 = disabled) |
| `--mixup_prob` | `0.5` | Probability of applying MixUp per batch |
| `--amp` | `True` | Mixed precision training |
| `--ema` | `True` | EMA of model weights; evaluate on EMA |
| `--ema_decay` | `0.9998` | EMA decay factor |
| `--patience` | `25` | Early stopping patience |
| `--device` | `"0"` | GPU index or `"cpu"` |
| `--workers` | `8` | DataLoader worker processes |
| `--seed` | `42` | Random seed |

## Supported Models

```
efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3,
efficientnet_v2_s, resnet18, resnet50, mobilenet_v3_small, mobilenet_v3_large
```

## Architecture

- Pretrained on ImageNet (`IMAGENET1K_V1`)
- Head replacement:
  - EfficientNet: `classifier[-1]` → `Linear(in_features, num_classes)`
  - ResNet: `fc` → `Linear(in_features, num_classes)`

## Training Augmentation

```python
RandomResizedCrop(imgsz, scale=(0.55,1.0), ratio=(0.8,1.25))
RandomHorizontalFlip, RandomVerticalFlip
RandomAffine(degrees=20, translate=(0.20,0.20), scale=(0.85,1.15))
ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.05)
RandAugment(num_ops=2, magnitude=9)
ToTensor + Normalize(ImageNet)
RandomErasing(p=0.25, scale=(0.02,0.20))
```

## Key Components

| Component | Description |
|---|---|
| `ModelEMA` | Exponential moving average (decay=0.9998) |
| `mixup_batch()` | Standard MixUp — mixes two random batch items |
| `validate()` | Computes loss, accuracy, F1, AUROC, AP |
| `build_torchvision_model()` | Loads pretrained backbone + replaces head |

## Outputs (saved to `runs/classify/<name>/`)

| File | Description |
|---|---|
| `weights/best.pt` | Best val F1-macro checkpoint (EMA if enabled) |
| `weights/last.pt` | Last epoch model (for resuming) |
| `train_log.csv` | Per-epoch: loss, acc, F1, AUROC, AP, LR, time |

## Example Commands (from file)

```bash
# EfficientNet-B0, fast run
python3 train_coty_classifier_v4.py \
    --model efficientnet_b0 \
    --data /home/rameen/final_dataset_v7_1to1_split \
    --name coty_effnetb0_v8 --epochs 100 --batch 64

# EfficientNet-B2, tuned for v12
python3 train_coty_classifier_v4.py \
    --model efficientnet_b2 \
    --data /home/rameen/final_dataset_v7_1to1_final \
    --imgsz 224 --batch 32 --epochs 100 --patience 28 \
    --lr0 0.0005 --warmup_epochs 5 --ema_decay 0.9999 \
    --label_smoothing 0.05 --mixup_alpha 0 --weight_decay 2e-4 \
    --name coty_effnetb2_v12 --device 0
```

## Difference from v5

| Feature | v4 | v5 |
|---|---|---|
| Staged backbone unfreezing | No | Yes |
| CutMix | No | Yes |
| Focal loss | No | Yes |
| Gradient accumulation | No | Yes |
| TTA evaluation | No | Yes |
| Per-class epoch metrics | No | Yes |
| Threshold optimization | No | Yes |
| Augmentation strength control | Fixed | `light`/`medium`/`strong` |

## Dependencies

`torch`, `torchvision`, `numpy`, `sklearn`, `argparse`, `copy`, `time`
