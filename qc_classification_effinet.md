# qc_classification_effinet.py

## Overview

QC (Quality Control) pipeline for EfficientNet classification models. Runs batch inference on a labeled validation dataset, computes per-class and overall metrics, exports CSVs, plots confusion matrices and dashboards, and saves annotated misclassified images for manual review.

## Use Case

Post-training evaluation: given a trained `best.pt` checkpoint and a labeled dataset (ground truth = folder name), measure real-world accuracy and identify failure modes.

## Configuration (top of file)

```python
MODEL_PATH   = "runs/classify/coty_effnetb2_v12/weights/best.pt"
MODEL_NAME   = "efficientnet_b2"    # must match training
DATA_PATH    = "/path/to/val"       # folder with cn_coty/ and non_coty/ subfolders
OUTPUT_PATH  = "/path/to/output"
CLASS_FOLDERS = ["cn_coty", "non_coty"]

BATCH_SIZE            = 32
PREDICTION_IMAGE_SIZE = 224
CONFIDENCE_THRESHOLD  = 0.0    # 0.0 = keep all predictions

MAX_PLOT_IMAGES = None   # None = export ALL misclassified images
SCALE_PERCENT   = 50     # resize misclassified images before saving
```

## Input Folder Structure

```
DATA_PATH/
├── cn_coty/
│   ├── image1.jpg
│   └── ...
└── non_coty/
    ├── image2.jpg
    └── ...
```

Ground truth label is determined by the subfolder name.

## Supported Model Architectures

```
efficientnet_b0, b1, b2, b3, b4, efficientnet_v2_s, efficientnet_v2_m
```

## Class: `QCClassificationEffiNet`

### Constructor Parameters

| Parameter | Description |
|---|---|
| `model_path` | Path to `best.pt` state dict |
| `model_name` | Architecture name (must match training) |
| `data_path` | Root of labeled image folders |
| `output_path` | Where all outputs are saved |
| `class_folders` | List of class names (alphabetical, matches training) |
| `batch_size` | Inference batch size |
| `image_size` | Resize target (default 224) |
| `confidence_threshold` | Minimum confidence to record a prediction (0 = all) |

### Key Methods

| Method | Description |
|---|---|
| `load_model()` | Builds architecture, loads state dict, sets eval mode |
| `run_inference()` | Batch inference over all class folders; populates `self.results` |
| `calculate_metrics()` | Computes TP, FP, FN, precision, recall, F1, accuracy per class |
| `export_csv()` | Saves per-image predictions to `classification_results.csv` |
| `export_metrics_csv()` | Saves per-class summary to `metrics_summary.csv` |
| `plot_confusion_matrix()` | Saves `confusion_matrix.png` |
| `plot_metrics_dashboard()` | Saves 2×2 `metrics_dashboard.png` (TP/FP/FN, P/R/F1, confidence dist, summary text) |
| `plot_misclassified_images()` | Saves annotated misclassified images per class with GT/Pred header + confidence bar |
| `print_summary()` | Logs per-class TP/FP/FN/P/R/F1 to console and log file |
| `run()` | Full pipeline: load → infer → metrics → CSV → plots |

## Usage

```bash
python qc_classification_effinet.py
```

Edit `MODEL_PATH`, `DATA_PATH`, `OUTPUT_PATH` at the top of the file before running.

## Outputs

| File | Description |
|---|---|
| `classification_results.csv` | Per-image: image name, ground truth, predicted, confidence, correct, error type |
| `metrics_summary.csv` | Per-class: TP, FP, FN, precision, recall, F1, support |
| `confusion_matrix.png` | Heatmap confusion matrix |
| `metrics_dashboard.png` | 2×2 dashboard: TP/FP/FN bars, P/R/F1 bars, confidence histogram, text summary |
| `misclassified_<class>/` | Annotated images sorted by confidence descending |
| `qc_effinet_<timestamp>.log` | Full run log |

## Dependencies

`torch`, `torchvision`, `numpy`, `cv2`, `PIL`, `matplotlib`, `csv`, `logging`
