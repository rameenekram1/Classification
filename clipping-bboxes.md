# clipping-bboxes.py

## Overview

Extracts fixed-size (224×224) image crops centered on bounding box detections from LabelMe/AnyLabeling JSON annotation files. Produces clipped image crops and adjusted JSON annotation files for each detection, ready for classifier input.

## Key Design: Center-Window Cropping

Unlike a simple bbox crop (which yields variable-size patches), this script extracts a **fixed 224×224 window centered on the bbox midpoint**. If the window would exceed image boundaries, it is shifted (not padded) to stay within the image.

This produces uniform-size crops for direct input to the EfficientNet classifier without any resizing distortion.

## Core Functions

### `clip_bbox(img, points, target=224) → (patch, (wx1, wy1, wx2, wy2))`

Computes the center of `points` (`[[x1,y1],[x2,y2]]`), places a `target×target` window around it, and shifts the window if needed to stay within image bounds. Returns the PIL crop and the actual window coordinates.

### `build_clip_json(original, shape, clip_img, clip_filename, x1, y1) → dict`

Rebuilds a LabelMe JSON for the cropped image: adjusts the shape's points by subtracting the window origin `(x1, y1)` so coordinates remain valid within the new image dimensions.

### `find_image(json_path, image_dir=None) → Path | None`

Searches for an image file matching the JSON stem. Tries extensions: `.jpg`, `.jpeg`, `.png`, `.tif`, `.tiff`, `.webp`, `.JPG`. Searches in `image_dir` if provided, otherwise in the JSON's parent directory.

### `process_pair(json_path, img_path, out_dir)`

Processes one JSON+image pair: reads all rectangle shapes, clips each one, saves the cropped image as JPEG (quality=100) and the adjusted JSON to `out_dir`.

## Usage

Edit the configuration block at the bottom of the script:

```python
INPUT_DIR = Path("/path/to/object_detection")   # folder containing .json files
IMAGE_DIR = Path("/path/to/field_images")        # folder containing the source images
OUT_DIR   = Path("/path/to/Tile_images")         # where crops are saved
MAX_CLIPS  = None    # int to stop early, or None for all
MIN_CONF   = 0.0     # only clip shapes with conf >= this value
```

```bash
python clipping-bboxes.py
```

## Output Per Detection

For each rectangle shape in each JSON, the script produces:

| File | Description |
|---|---|
| `<stem>_bbox<NNN>.jpg` | 224×224 crop centered on detection |
| `<stem>_bbox<NNN>.json` | LabelMe JSON with adjusted coordinates |

## Console Output

```
Processing: DJI_20260511133924_0023_V_...json
  [0] DJI_..._bbox000.jpg  (224x224)  conf=0.812
  [1] DJI_..._bbox001.jpg  (224x224)  conf=0.734
Done: 2 clips saved to 'Tile_images'

Finished. Total clips saved: 47
```

## Notes

- Only `shape_type == "rectangle"` shapes are processed; others are skipped.
- JPEG quality is 100 (lossless-equivalent) to preserve pixel fidelity.
- The commented-out code at the top is an earlier version that cropped the exact bbox (variable size) rather than a fixed-size centered window.
- The commented-out `process_field()` function shows an alternative multi-field batch mode.

## Dependencies

`PIL` (Pillow), `json`, `pathlib`, `copy`
