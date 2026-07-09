# classify_clips_effinet.py

## Overview

Classifies a flat folder of clipped bounding-box images using a trained EfficientNet (or other torchvision) model. Sorts images into per-class subfolders, copies companion JSON annotation files alongside them, and generates a CSV report and summary text file.

## Configuration (top of file)

```python
MODEL_PATH   = "runs/classify/coty_effnetb2_v12/weights/best.pt"
MODEL_NAME   = "efficientnet_b2"          # must match training
INPUT_FOLDER = "/path/to/Tile_images"     # flat folder of images to classify
OUTPUT_FOLDER= "/path/to/output_v12"

BATCH_SIZE            = 32
PREDICTION_IMAGE_SIZE = 224
CONFIDENCE_THRESHOLD  = 0.0    # 0.0 = keep all; e.g. 0.7 = high-confidence only
CLASS_NAMES = ["cn_coty", "non_coty"]     # must match training folder order (alphabetical)
```

## Functions

### `load_model(model_path, model_name, num_classes) → (model, device)`

Rebuilds the torchvision architecture from a weights map, replaces the classifier head to match `num_classes`, loads state dict from `model_path`. Supports a wide range of architectures:

```
efficientnet_b0–b4, efficientnet_v2_s/m
resnet18, resnet34, resnet50, resnet101
mobilenet_v3_small/large
convnext_tiny, convnext_small
```

### `classify_and_sort(input_folder, output_folder)`

Full pipeline:
1. Loads model.
2. Creates `output_folder/<class>/` subfolders. If `CONFIDENCE_THRESHOLD > 0`, also creates `output_folder/low_confidence/`.
3. Collects and sorts all image files from `input_folder`.
4. Runs batched inference (PIL → tensor → softmax → top-1 class + confidence).
5. Copies each image to its predicted class subfolder (or `low_confidence/` if below threshold).
6. Copies companion `.json` file alongside each image if present.
7. Writes `classification_results.csv` and `summary_report.txt`.

## Usage

```bash
python classify_clips_effinet.py
```

Edit `MODEL_PATH`, `MODEL_NAME`, `INPUT_FOLDER`, and `OUTPUT_FOLDER` at the top before running.

## Output Structure

```
OUTPUT_FOLDER/
├── cn_coty/
│   ├── image_bbox000.jpg
│   ├── image_bbox000.json
│   └── ...
├── non_coty/
│   └── ...
├── low_confidence/           # only created if CONFIDENCE_THRESHOLD > 0
│   └── ...
├── classification_results.csv
├── summary_report.txt
└── classify_<timestamp>.log
```

## Output Files

| File | Description |
|---|---|
| `classification_results.csv` | Per-image: `image`, `predicted_label`, `confidence` |
| `summary_report.txt` | Total count and per-class percentage |
| `classify_<timestamp>.log` | Full run log (also printed to console) |

## Val Transform

Matches the production inference transform used during training:
```python
Resize((224, 224)) → ToTensor() → Normalize(ImageNet mean/std)
```

## Dependencies

`torch`, `torchvision`, `PIL`, `cv2`, `numpy`, `csv`, `shutil`, `logging`, `pathlib`, `traceback`
