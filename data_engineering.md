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

Thresholds are derived from this dataset's own `plant_score` distribution: every image in a class is scored, scores are sorted, and the value sitting at the requested percentile rank becomes the cutoff. This adapts to shifts in image characteristics (e.g. camera, altitude, lighting) that would make an absolute fraction too strict or too loose.

| Class | Bad percentile rank | Small percentile rank |
|---|---|---|
| `cn_coty` | 5th | 25th |
| `non_coty` | 5th | 25th |

E.g. `--cn_coty_bad_pct 5` means: whatever `plant_score` value sits at the 5th percentile of all `cn_coty` images *in this dataset* becomes the BAD cutoff — so ~5% of `cn_coty` images will always be dropped, regardless of what absolute coverage fraction that corresponds to.

## Usage

```bash
# 1) Dry-run: analysis only (no files written)
python data_engineering.py \
    --src /home/rameen/final_dataset_v7_1to1_split \
    --analyze_only \
    --cn_coty_bad_pct 5 --cn_coty_small_pct 25 \
    --non_coty_bad_pct 5 --non_coty_small_pct 25

# 2) Build cleaned dataset
python data_engineering.py \
    --src /home/rameen/final_dataset_v7_1to1_split \
    --dst /home/rameen/final_dataset_v8_cleaned \
    --cn_coty_bad_pct 5 --cn_coty_small_pct 25 \
    --non_coty_bad_pct 5 --non_coty_small_pct 25 \
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
| `--cn_coty_bad_pct` | `5` | cn_coty: percentile rank used as the drop cutoff |
| `--cn_coty_small_pct` | `25` | cn_coty: percentile rank used as the zoom cutoff |
| `--non_coty_bad_pct` | `5` | non_coty: percentile rank used as the drop cutoff |
| `--non_coty_small_pct` | `25` | non_coty: percentile rank used as the zoom cutoff |
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

Thresholds are derived first, then the resulting BAD/SMALL/GOOD split is printed:

```
[percentile mode] Deriving thresholds from this dataset's own distribution...
  cn_coty: 5th percentile = 2.10% (drop cutoff), 25th percentile = 8.40% (zoom cutoff)  [3200 images pooled across all splits]
  non_coty: 5th percentile = 0.80% (drop cutoff), 25th percentile = 3.60% (zoom cutoff)  [3000 images pooled across all splits]

  Plant-coverage percentiles:
     5th :  1.2%
    25th :  4.8%
    50th : 12.3%
    ...
  BAD   (<2.1% plant) :  412 / 6200  (6.6%) — will be DROPPED
  SMALL (2.1–8.4%)    :  890 / 6200  (14.4%) — will be ZOOMED
  GOOD  (>8.4% plant) : 4898 / 6200  (79.0%) — kept as-is
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
