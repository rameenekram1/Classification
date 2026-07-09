# Canola Cotyledon Classifier

A complete binary classification pipeline for identifying cotyledon-stage canola plants in drone imagery, built on PyTorch EfficientNet models.

---

## Pipeline Overview

```
DJI drone images + LabelMe/AnyLabeling JSONs
         │
         ▼
 clipping-bboxes.py          ← Extract 224×224 crops from bbox detections
         │
         ▼
 split_dataset_v2.py         ← Group-aware train/val split (no leakage)
         │
         ▼
 train_coty_classifier_v4.py ← Train with EMA + augmentation
         │
         ├──► classify_clips_effinet.py    ← Inference on flat folder
         └──► qc_classification_effinet.py ← Evaluation against labeled ground truth
```

**Classes:** `cn_coty` (cotyledon) · `non_coty` (non-cotyledon)

---

## Scripts

### Data Preparation

#### `clipping-bboxes.py`
Extracts fixed 224×224 crops centered on bounding box detections from LabelMe/AnyLabeling JSON annotations. Uses center-window cropping (shifts to stay in bounds, no black padding). Outputs cropped images and adjusted JSON files.

| Config | Description |
|--------|-------------|
| `INPUT_DIR` | Folder with JSON annotation files |
| `IMAGE_DIR` | Folder containing the source images |
| `OUT_DIR` | Destination for cropped clips |
| `MAX_CLIPS` | Stop early after N clips (`None` = all) |
| `MIN_CONF` | Only clip shapes with confidence ≥ this value |

---

#### `split_dataset_v2.py`
Group-aware, leakage-free dataset splitter. Groups images by source and DJI flight session to prevent same-source crops from appearing in both train and val. Parses DJI filename convention: `DJI_<timestamp>_<seq>_<sensor>_<rest>_bbox<NNN>.jpg`.

```bash
python split_dataset_v2.py \
    --source /path/to/dataset \
    --output /path/to/dataset_split \
    --val_split 0.20 \
    --balance train
```

| Argument | Default | Description |
|--------|---------|-------------|
| `--source` | required | Root directory with `<class>/` subdirs |
| `--output` | required | Destination for `train/val/<class>/` |
| `--val_split` | `0.20` | Fraction of images for validation |
| `--val_dates` | `""` | Comma-separated `YYYYMMDD` dates forced to val |
| `--val_fields` | `""` | Comma-separated field codes forced to val |
| `--balance` | `train` | Subsample majority class: `none`, `train`, or `both` |
| `--stratify_by_field` | `True` | Split flights per field (disable with `--no-stratify`) |

---

### Training

#### `train_coty_classifier_v4.py`
Training script with strong augmentation, tuned for imbalanced drone imagery.

**Key features:**
- AdamW + CosineAnnealingLR with linear warmup
- Mixed precision (AMP)
- RandAugment, RandomErasing, RandomAffine, MixUp (α=0.2)
- EMA (decay=0.9998)
- Class-weighted cross-entropy
- Best checkpoint selected by val F1-macro
- Val transform aligned with production inference (force-stretch resize, no center-crop)

```bash
python train_coty_classifier_v4.py \
    --model efficientnet_b0 \
    --data  /path/to/dataset_split \
    --name  coty_effnetb0_v4 \
    --epochs 60 --batch 64
```

| Argument | Default | Description |
|--------|---------|-------------|
| `--model` | `efficientnet_b0` | Architecture (see supported list below) |
| `--data` | required | Dataset root with `train/` and `val/` subfolders |
| `--epochs` | `60` | Number of training epochs |
| `--batch` | `64` | Batch size |
| `--lr0` | `1e-3` | Peak learning rate |
| `--class_weights` | `balanced` | `balanced` (inverse-freq) or `none` |
| `--mixup_alpha` | `0.2` | MixUp beta parameter (0 disables) |
| `--ema_decay` | `0.9998` | EMA decay factor |
| `--patience` | `25` | Early stopping patience |
| `--name` | — | Run name (outputs saved to `runs/classify/<name>/`) |

**Supported architectures:**

| Family | Options |
|--------|---------|
| EfficientNet | `efficientnet_b0` · `b1` · `b2` · `b3` · `efficientnet_v2_s` |
| ResNet | `resnet18` · `resnet50` |
| MobileNet | `mobilenet_v3_small` · `mobilenet_v3_large` |

All use ImageNet pretrained weights; the classification head is replaced for binary output.

**Outputs** (saved to `runs/classify/<name>/`): `weights/best.pt`, `weights/last.pt`, `train_log.csv`.

---

### Inference

#### `classify_clips_effinet.py`
Classifies a flat folder of clipped images. Sorts outputs into per-class subfolders and copies companion JSON files. Generates a per-image CSV and summary report.

```bash
python classify_clips_effinet.py
```

| Config | Description |
|--------|-------------|
| `MODEL_PATH` | Path to trained `.pt` checkpoint |
| `MODEL_NAME` | Architecture name (must match training) |
| `INPUT_FOLDER` | Flat folder of images to classify |
| `OUTPUT_FOLDER` | Results destination |
| `CONFIDENCE_THRESHOLD` | Minimum confidence to assign a class label (0 = keep all) |
| `BATCH_SIZE` | Inference batch size |

**Output structure:**
```
OUTPUT_FOLDER/
├── cn_coty/          ← classified images + JSONs
├── non_coty/
├── low_confidence/    ← only if CONFIDENCE_THRESHOLD > 0
├── classification_results.csv
└── summary_report.txt
```

---

### Evaluation

#### `qc_classification_effinet.py`
Evaluates a trained model against labeled validation data (ground truth = folder name). Computes per-class TP/FP/FN, precision, recall, and F1.

```bash
python qc_classification_effinet.py
```

| Config | Description |
|--------|-------------|
| `MODEL_PATH` | Path to trained checkpoint |
| `MODEL_NAME` | Architecture name (must match training) |
| `DATA_PATH` | Labeled dataset (subfolders = class names) |
| `OUTPUT_PATH` | Results destination |
| `BATCH_SIZE` | Inference batch size |

**Outputs:**

| File | Description |
|------|-------------|
| `classification_results.csv` | Prediction + confidence per image |
| `metrics_summary.csv` | Per-class precision / recall / F1 |
| `confusion_matrix.png` | Heatmap |
| `metrics_dashboard.png` | 2×2 panel: TP/FP/FN, P/R/F1, confidence histogram, summary |
| `misclassified_<class>/` | Annotated misclassified images, sorted by confidence |

---

### Utility

#### `rename-jpgs.py`
Appends a class-index suffix to filenames: `cn_coty` → `(0)`, `non_coty` → `(1)`. Operates recursively on the target folder. One-shot utility — do not run twice on the same folder.

---

## JSON Format (LabelMe / AnyLabeling)

```json
{
  "shapes": [
    {
      "label": "cn_coty",
      "points": [[x1, y1], [x2, y2]],
      "shape_type": "rectangle"
    }
  ],
  "imagePath": "image.jpg",
  "imageHeight": 1080,
  "imageWidth": 1920
}
```

---

## Dependencies

```bash
pip install torch torchvision pillow opencv-python numpy scikit-learn matplotlib
```

---

## Documentation

Each script has a corresponding `.md` file with full function signatures, parameter tables, and usage examples.
