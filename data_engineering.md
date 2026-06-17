# data_engineering.py

## Overview

Diagnoses and cleans the coty classifier dataset by scoring each image for plant coverage using an HSV-based heuristic. Images are categorized as **BAD** (drop), **SMALL** (center-crop zoom), or **GOOD** (keep as-is). Produces a cleaned dataset with a per-image action report CSV.

## Motivation

Low-coverage images (mostly bare soil, straw, or out-of-focus backgrounds) degrade training:
- **cn_coty**: needs clear cotyledon features — strict filtering applied.
- **non_coty**: bare soil and low-plant images ARE valid negatives — conservative filtering applied.

## Plant Coverage Score

```python
plant_score(img_path) → float  # fraction of pixels classified as green/yellow-green plant
```

HSV criteria: hue 40–175°, saturation > 0.12, value > 40. Covers yellow-green through cyan-green tissue.

## Image Categories

| Category | Threshold | Action |
|---|---|---|
| BAD | score < `bad_thresh` | Dropped — teaches nothing about plant features |
| SMALL | `bad_thresh` ≤ score < `small_thresh` | Center-crop zoom → resize to original size |
| GOOD | score ≥ `small_thresh` | Copied as-is |

## Center-Crop Zoom

```python
center_crop_zoom(img, crop_frac=0.65) → Image
```

Crops the central `crop_frac` fraction and resizes back to original dimensions. At `crop_frac=0.65` this applies a ~1.54× zoom, making small plants fill more of the frame without padding or distortion.

## Default Per-Class Thresholds

| Class | Bad threshold | Small threshold |
|---|---|---|
| `cn_coty` | 3% | 10% |
| `non_coty` | 1% | 4% |

## Usage

```bash
# 1) Dry-run: analysis only (no files written)
python data_engineering.py \
    --src /home/rameen/final_dataset_v7_1to1_split \
    --analyze_only

# 2) Build cleaned dataset
python data_engineering.py \
    --src /home/rameen/final_dataset_v7_1to1_split \
    --dst /home/rameen/final_dataset_v8_cleaned \
    --cn_coty_bad 0.03  --cn_coty_small 0.10 \
    --non_coty_bad 0.01 --non_coty_small 0.04 \
    --zoom_crop 0.65

# 3) With visualization grid
python data_engineering.py --src ... --dst ... --visualize
```

## Arguments

| Argument | Default | Description |
|---|---|---|
| `--src` | required | Source dataset root (flat or train/val split) |
| `--dst` | `None` | Output root (required unless `--analyze_only`) |
| `--zoom_crop` | `0.65` | Center-crop fraction for SMALL images |
| `--analyze_only` | `False` | Print stats only, write nothing |
| `--cn_coty_bad` | `0.03` | Drop cn_coty below this coverage |
| `--cn_coty_small` | `0.10` | Zoom cn_coty below this coverage |
| `--non_coty_bad` | `0.01` | Drop non_coty below this coverage |
| `--non_coty_small` | `0.04` | Zoom non_coty below this coverage |
| `--visualize` | `False` | Save BAD/SMALL/GOOD grid images per class |

## Supported Layout

Automatically detects flat vs. split layout:

- **Flat**: `src/cn_coty/` and `src/non_coty/`
- **Split**: `src/train/cn_coty/`, `src/val/cn_coty/`, etc.

## Outputs

| File | Description |
|---|---|
| `dst/<split>/<class>/` | Cleaned images (kept + zoomed) |
| `dst/data_engineering_report.csv` | Per-image: `file`, `plant_score`, `action` |
| `vis_<split>_<class>.png` | (if `--visualize`) Sample grid: BAD / SMALL before→after / GOOD |

## Console Output (Analysis Mode)

```
  Plant-coverage percentiles:
     5th :  1.2%
    25th :  4.8%
    50th : 12.3%
    ...
  BAD   (<3% plant) :  412 / 6200  (6.6%) — will be DROPPED
  SMALL (3–10%)     :  890 / 6200  (14.4%) — will be ZOOMED
  GOOD  (>10% plant): 4898 / 6200  (79.0%) — kept as-is
```

## Recommended Next Steps After Cleaning

```bash
python split_dataset_v2.py \
    --source /home/rameen/final_dataset_v8_cleaned \
    --output /home/rameen/final_dataset_v8_split \
    --val_split 0.20 --balance train

python train_coty_classifier_v5.py \
    --model efficientnet_b2 \
    --data /home/rameen/final_dataset_v8_split \
    --name coty_effnetb2_v13 --epochs 100 --batch 32
```

## Dependencies

`numpy`, `PIL`, `argparse`, `csv`, `shutil`, `pathlib`, `matplotlib` (optional, for `--visualize`)
