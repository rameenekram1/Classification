# train_coty_classifier_v5.py

## Overview

Training script (v5) for the binary **Canola Cotyledon (cn_coty) vs Non-Coty** classifier using torchvision EfficientNet / ResNet / MobileNet backbones. Successor to v4 with significant improvements for drone-captured field imagery.

## Improvements Over v4

| Feature | v4 | v5 |
|---|---|---|
| Spatial augmentation | Strong | Toned-down (preserves small cotyledons) |
| ColorJitter | Stronger | Reduced (preserves green-hue signal) |
| RandAugment magnitude | 9 | 7 |
| MixUp | On by default | Off by default (harmful when non-coty = green vegetation) |
| CutMix | No | Added (preserves local regions) |
| Backbone freezing | No | Staged unfreezing (N epochs head-only, then full) |
| Head dropout | Default | 0.4 (reduces overfitting on ~12k images) |
| Gradient accumulation | No | Yes |
| TTA at evaluation | No | Yes (hflip + vflip) |
| Focal loss | No | Optional |
| Per-class epoch metrics | No | Yes |

## Usage

```bash
python train_coty_classifier_v5.py \
    --model efficientnet_b2 \
    --data  /home/rameen/final_dataset_v7_1to1_split \
    --name  coty_effnetb2_v5 \
    --epochs 80 --batch 32 --patience 25
```

## Key Arguments

| Argument | Default | Description |
|---|---|---|
| `--model` | `efficientnet_b2` | Architecture (see supported models below) |
| `--data` | required | Dataset root with `train/<class>/` and `val/<class>/` |
| `--epochs` | `80` | Maximum training epochs |
| `--batch` | `32` | Batch size per GPU |
| `--lr0` | `1e-3` | Initial learning rate |
| `--aug_strength` | `medium` | `light` / `medium` / `strong` |
| `--head_dropout` | `0.4` | Dropout before classifier head |
| `--freeze_epochs` | `5` | Epochs to freeze backbone (0 = train from scratch) |
| `--backbone_lr_mult` | `0.1` | Backbone LR = `lr0 × this` after unfreezing |
| `--cutmix_alpha` | `0.0` | CutMix beta param (0 = disabled; try 1.0) |
| `--focal_gamma` | `0.0` | Focal loss gamma (0 = standard CE; try 1.5–2.0) |
| `--grad_accum` | `1` | Gradient accumulation steps (effective batch = batch × steps) |
| `--patience` | `25` | Early stopping patience (epochs without F1 improvement) |
| `--tta` | `True` | TTA evaluation (hflip + vflip) at end |
| `--amp` | `True` | Mixed precision training |
| `--ema` | `True` | EMA of model weights, evaluate on EMA |
| `--class_weights` | `balanced` | Inverse-frequency class weighting |
| `--label_smoothing` | `0.08` | Cross-entropy label smoothing |

## Supported Models

```
efficientnet_b0, efficientnet_b1, efficientnet_b2, efficientnet_b3,
efficientnet_v2_s, resnet18, resnet50, mobilenet_v3_small, mobilenet_v3_large
```

## Architecture

- Pretrained on ImageNet (`IMAGENET1K_V1`)
- Classifier head replaced: `Dropout(p=head_dropout) → Linear(in_features, num_classes)`
- LR schedule: linear warmup → cosine annealing
- Optimizer: AdamW with weight decay
- Metric for best checkpoint: **val F1-macro**

## Key Components

| Component | Description |
|---|---|
| `ModelEMA` | Exponential moving average of weights (decay=0.9995) |
| `cutmix_batch()` | CutMix augmentation — preferred over MixUp for this task |
| `FocalCrossEntropyLoss` | Hard-example mining via focal weighting |
| `validate()` | Computes loss, accuracy, F1, AUROC, AP; supports TTA |
| `find_optimal_threshold()` | Sweeps 200 thresholds on softmax prob to maximize F1-macro |

## Outputs (saved to `runs/classify/<name>/`)

| File | Description |
|---|---|
| `weights/best.pt` | Best val F1-macro checkpoint (EMA weights) |
| `weights/last.pt` | Last epoch training model (for resuming) |
| `train_log.csv` | Per-epoch metrics |
| `optimal_threshold.txt` | Best softmax threshold for deployment |

## Recommended Runs

```bash
# Run 1: Baseline
python3 train_coty_classifier_v5.py \
    --model efficientnet_b2 --data /home/rameen/final_dataset_v7_1to1_split \
    --imgsz 224 --batch 32 --grad_accum 2 --epochs 80 --patience 25 \
    --freeze_epochs 5 --backbone_lr_mult 0.1 --head_dropout 0.4 \
    --aug_strength medium --label_smoothing 0.08 --name coty_effnetb2_v5_run1

# Run 2: + CutMix (if overfitting)
#   add: --cutmix_alpha 1.0 --cutmix_prob 0.3

# Run 3: + Focal Loss (if one class dominates errors)
#   add: --focal_gamma 1.5 --label_smoothing 0.05

# Run 4: Strong augmentation (if train_acc >> val_acc by 5%+)
#   --aug_strength strong --head_dropout 0.45 --epochs 100
```

## Dependencies

`torch`, `torchvision`, `numpy`, `sklearn`, `argparse`, `copy`, `time`
