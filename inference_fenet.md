# inference_fenet.py

## Overview

Production inference script that classifies a flat folder of cropped bounding-box images using a trained EfficientNet model. Sorts output images into per-class subfolders and writes a summary CSV. Designed to run across multiple field directories in a pipeline.

## Configuration (top of file)

```python
MODEL_PATH      = "runs/classify/coty_effnetb2_v12/weights/best.pt"
MODEL_NAME      = "efficientnet_b2"     # must match training
PIPELINE_OUTPUT = "/path/to/Prescout_pipeline_output/pipeline_output"
OUTPUT_FOLDER   = "output_v12"          # created inside each field's Tile_images/

CLASS_NAMES = ["cn_coty", "non_coty"]   # must match training order (alphabetical)

BATCH_SIZE            = 32
PREDICTION_IMAGE_SIZE = 224
COPY_MODE             = True    # True = copy files, False = move files
```

## Functions

### `load_model() → (model, device)`

Builds the EfficientNet architecture with head replaced to match `len(CLASS_NAMES)`, loads the state dict from `MODEL_PATH`, moves to GPU if available, sets eval mode.

### `predict_batch(model, device, pil_images) → list[(label, confidence)]`

Transforms a list of PIL images, runs batched inference, returns `(class_name, confidence)` tuples for each image.

### `run(data_path, output_path, model=None, device=None)`

Classifies all images in `data_path` (flat folder), copies/moves each to `output_path/<class>/`, and saves `classification_results.csv`.

- Creates one subfolder per class inside `output_path`.
- Loads model on demand if not passed in (allows reuse across fields).
- Logs per-image predictions and a final per-class count summary.

## Pipeline Entry Point (`__main__`)

When run directly, iterates over all field directories under `PIPELINE_OUTPUT`:

```
PIPELINE_OUTPUT/
├── <field_dir>/
│   └── object_detection/
│       └── Tile_images/        ← input images
│           └── output_v12/     ← created here
│               ├── cn_coty/
│               ├── non_coty/
│               └── classification_results.csv
```

**Skip logic:** If `Tile_images/` does not exist → skip. If `classification_results.csv` already exists → skip (resume-safe).

Model is loaded **once** and reused across all fields for efficiency.

## Usage

```bash
python inference_fenet.py
```

Edit `MODEL_PATH`, `MODEL_NAME`, `PIPELINE_OUTPUT`, and `OUTPUT_FOLDER` at the top before running.

## Output Per Field

| File/Folder | Description |
|---|---|
| `output_v12/cn_coty/` | Images classified as cotyledon |
| `output_v12/non_coty/` | Images classified as non-cotyledon |
| `output_v12/classification_results.csv` | `image`, `predicted_label`, `confidence` |

## Console Output

- Per-image: `<filename> -> <class> (<confidence>)`
- Per-field summary: count per class
- Grand total across all fields at end

## Dependencies

`torch`, `torchvision`, `PIL`, `csv`, `shutil`, `logging`, `pathlib`
