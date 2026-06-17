# split_dataset_v2.py

## Overview

Group-aware, leakage-free dataset splitter for the canola cotyledon (coty) classifier. Prevents train/val contamination by assigning whole source-image groups and flight sessions to a single split.

## Problem It Solves

Random per-image splits cause two forms of data leakage:
1. Multiple `bboxNNN` crops from the same source image end up in both train and val.
2. Images from the same drone flight (same date + hour + field) look near-identical — splitting within a flight inflates val accuracy.

## Key Features

- Groups images by `source_id` (everything before `_bboxNNN`) so no source ever crosses splits.
- Groups sources by `flight_id` (`date_hour_field_sensor`) and assigns whole flights to train or val.
- Supports **field-stratified splitting** (default): each field's flights are split independently so every field with ≥2 flights appears in both splits.
- Supports **hard holdout** of specific dates (`--val_dates`) or field codes (`--val_fields`) for unseen-field generalization testing.
- Optional class balancing by subsampling the majority class in train and/or val.
- Asserts zero source-level leakage after splitting.

## Usage

```bash
# 1) Pure stratified group split, 80/20, balance train classes
python split_dataset_v2.py \
    --source /home/rameen/final_dataset_v7_1to1 \
    --output /home/rameen/final_dataset_v7_1to1_split \
    --val_split 0.20 \
    --balance train

# 2) Hold out specific field codes for val (generalization test)
python split_dataset_v2.py \
    --source /home/rameen/final_dataset_v7_1to1 \
    --output /home/rameen/final_dataset_v7_1to1_split \
    --val_fields FIELD7020LA,FIELD7008LA,FIELD7069LA \
    --val_split 0.20 \
    --balance train

# 3) Hold out specific dates for val
python split_dataset_v2.py \
    --source /home/rameen/final_dataset_v7_1to1 \
    --output /home/rameen/final_dataset_v7_1to1_split \
    --val_dates 20260511,20260512,20260518 \
    --balance train
```

## Arguments

| Argument | Default | Description |
|---|---|---|
| `--source` | required | Root directory with `<class>/` subdirs |
| `--output` | required | Destination for `train/val/<class>/` |
| `--val_split` | `0.20` | Fraction of images for validation |
| `--val_dates` | `""` | Comma-separated `YYYYMMDD` dates forced to val |
| `--val_fields` | `""` | Comma-separated field codes forced to val |
| `--balance` | `train` | Subsample majority class: `none`, `train`, or `both` |
| `--stratify_by_field` | `True` | Split flights per field (disable with `--no-stratify`) |
| `--seed` | `42` | Random seed for reproducibility |

## Output Structure

```
output/
├── train/
│   ├── cn_coty/
│   └── non_coty/
└── val/
    ├── cn_coty/
    └── non_coty/
```

## Filename Convention Parsed

```
DJI_<timestamp14>_<seq>_<sensor>_<rest>_bbox<NNN>.jpg
```

- `timestamp[:8]` → date, `[8:10]` → hour
- `FIELD\d+[A-Z]*` token in `<rest>` → field code (falls back to `"LA"`)
- `flight_id = "{date}_{hour}_{field}_{sensor}"`

## Next Step

```bash
python train_coty_classifier_v4.py \
    --model efficientnet_b0 \
    --data <output> \
    --name coty_effnetb0_v4 \
    --epochs 60 --batch 64
```

## Dependencies

`pathlib`, `random`, `re`, `shutil`, `collections` (stdlib only — no external deps)
