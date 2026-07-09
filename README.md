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
 data_engineering.py         ← HSV-based cleaning (drop bad, zoom small)
         │
         ▼
 train_coty_classifier_v4/v5.py  ← Train with EMA + augmentation
         │
         ├──► classify_clips_effinet.py   ← Inference on flat folder
         ├──► inference_fenet.py          ← Inference across multi-field pipeline
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
| `INPUT_FOLDER` | Folder with images + JSON annotation files |
| `OUTPUT_FOLDER` | Destination for cropped clips |
| `CROP_SIZE` | Fixed crop size (default 224×224) |

---

#### `split_dataset_v2.py`
Group-aware, leakage-free dataset splitter. Groups images by source and DJI flight session to prevent same-source crops from appearing in both train and val. Parses DJI filename convention: `DJI_<timestamp>_<seq>_<sensor>_<rest>_bbox<NNN>.jpg`.

```bash
python split_dataset_v2.py
```

| Config | Description |
|--------|-------------|
| `SOURCE_FOLDER` | Flat folder of clipped images |
| `OUTPUT_FOLDER` | Output root for train/val folders |
| `TRAIN_RATIO` | Fraction for training (e.g. 0.8) |
| `HELD_OUT_FIELDS` | Fields to reserve for held-out test set |
| `BALANCE_CLASSES` | Undersample majority class |

---

#### `data_engineering.py`
Diagnoses and cleans the dataset using HSV-based plant coverage scoring. Categorizes each image as:
- **BAD** — drop (too little plant coverage)
- **SMALL** — center-crop zoom (partially cropped plant)
- **GOOD** — keep as-is

Uses per-class thresholds (cn_coty is stricter). Outputs a CSV report and visualization grids.

```bash
python data_engineering.py
```

| Config | Description |
|--------|-------------|
| `INPUT_FOLDER` | Dataset root (flat or split layout) |
| `OUTPUT_FOLDER` | Cleaned dataset destination |

---

### Training

#### `train_coty_classifier_v4.py`
General-purpose training script with strong augmentation.

**Key features:**
- AdamW + CosineAnnealingLR with linear warmup
- Mixed precision (AMP)
- RandAugment, RandomErasing, RandomAffine, MixUp (α=0.2)
- EMA (decay=0.9998)
- Class-weighted cross-entropy
- Best checkpoint selected by val F1-macro

```bash
python train_coty_classifier_v4.py
```

| Config | Description |
|--------|-------------|
| `DATA_PATH` | Dataset root with `train/` and `val/` subfolders |
| `OUTPUT_PATH` | Where to save checkpoints and logs |
| `ARCH` | Model architecture (see supported list below) |
| `EPOCHS` | Number of training epochs |
| `BATCH_SIZE` | Batch size |
| `LR` | Peak learning rate |
| `MIXUP_PROB` | MixUp probability (0 to disable) |

---

#### `train_coty_classifier_v5.py`
Improved version tuned for drone imagery. Use this by default.

**Improvements over v4:**
- Conservative spatial augmentation (preserves small cotyledons)
- Reduced ColorJitter (preserves green-hue signal)
- MixUp off by default; CutMix added as alternative
- Staged backbone unfreezing
- Higher classifier dropout (0.4)
- Gradient accumulation support
- TTA (hflip/vflip) at evaluation
- Focal loss option
- Per-class metrics logged every epoch
- Threshold optimization for deployment

```bash
python train_coty_classifier_v5.py
```

Same configuration keys as v4, plus:

| Config | Description |
|--------|-------------|
| `FOCAL_LOSS` | Use focal loss instead of cross-entropy |
| `GRAD_ACCUM_STEPS` | Gradient accumulation steps |
| `UNFREEZE_EPOCH` | Epoch to unfreeze backbone layers |
| `TTA` | Enable test-time augmentation |

**Supported architectures (both v4 and v5):**

| Family | Options |
|--------|---------|
| EfficientNet | `efficientnet_b0` · `b1` · `b2` · `b3` · `efficientnet_v2_s` · `v2_m` |
| ResNet | `resnet18` · `resnet50` |
| MobileNet | `mobilenet_v3_small` · `mobilenet_v3_large` |

All use ImageNet pretrained weights; the classification head is replaced for binary output.

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
| `INPUT_FOLDER` | Flat folder of images to classify |
| `OUTPUT_FOLDER` | Results destination |
| `CONF_THRESHOLD` | Minimum confidence to assign a class label |
| `BATCH_SIZE` | Inference batch size |

**Output structure:**
```
output/
├── cn_coty/          ← classified images + JSONs
├── non_coty/
├── results.csv
└── summary.txt
```

---

#### `inference_fenet.py`
Production inference across a multi-field pipeline directory. Iterates over field subdirectories, loads the model once, and processes each field. Resume-safe (skips already processed fields).

```bash
python inference_fenet.py
```

| Config | Description |
|--------|-------------|
| `MODEL_PATH` | Path to trained checkpoint |
| `PIPELINE_ROOT` | Root directory containing per-field folders |
| `OUTPUT_ROOT` | Output root (mirrors input structure) |
| `BATCH_SIZE` | Inference batch size |

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
| `INPUT_FOLDER` | Labeled dataset (subfolders = class names) |
| `OUTPUT_FOLDER` | Results destination |
| `BATCH_SIZE` | Inference batch size |

**Outputs:**

| File | Description |
|------|-------------|
| `per_image_results.csv` | Prediction + confidence per image |
| `metrics_summary.csv` | Per-class precision / recall / F1 |
| `confusion_matrix.png` | Heatmap |
| `metrics_dashboard.png` | 4-panel: TP/FP/FN, P/R/F1, confidence histogram, summary |
| `misclassified/` | Annotated misclassified images, sorted by confidence |

---

### Utility

#### `rename-jpgs.py`
Appends a class-index suffix to filenames: `cn_coty` → `(0)`, `non_coty` → `(1)`. Operates recursively on the target folder.

#### `data_engineering.py` (also utility)
Can be run standalone to audit dataset quality before training.

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
