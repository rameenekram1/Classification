# import json
# from copy import deepcopy
# from pathlib import Path

# from PIL import Image


# def clip_bbox(img, points):
#     (x1, y1), (x2, y2) = points
#     x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
#     x1 = max(0, min(x1, img.width))
#     y1 = max(0, min(y1, img.height))
#     x2 = max(0, min(x2, img.width))
#     y2 = max(0, min(y2, img.height))
#     return img.crop((x1, y1, x2, y2)), (x1, y1, x2, y2)


# def build_clip_json(original, shape, clip_img, clip_filename, x1, y1):
#     w, h = clip_img.size
#     new_points = [
#         [shape["points"][0][0] - x1, shape["points"][0][1] - y1],
#         [shape["points"][1][0] - x1, shape["points"][1][1] - y1],
#     ]
#     new_shape = deepcopy(shape)
#     new_shape["points"] = new_points
#     return {
#         "version": original.get("version", "0.4.29"),
#         "flags": original.get("flags", {}),
#         "shapes": [new_shape],
#         "imagePath": clip_filename,
#         "imageData": None,
#         "imageHeight": h,
#         "imageWidth": w,
#     }


# def find_image(json_path):
#     for ext in (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".JPG"):
#         candidate = json_path.with_suffix(ext)
#         if candidate.exists():
#             return candidate
#     return None


# def process_pair(json_path, img_path, out_dir):
#     with open(json_path, "r") as f:
#         data = json.load(f)

#     img = Image.open(img_path).convert("RGB")
#     stem = json_path.stem
#     out_dir.mkdir(parents=True, exist_ok=True)

#     for i, shape in enumerate(data.get("shapes", [])):
#         if shape.get("shape_type") != "rectangle":
#             print(f"  Skipping non-rectangle shape {i}")
#             continue

#         clip_img, (x1, y1, x2, y2) = clip_bbox(img, shape["points"])

#         clip_stem = f"{stem}_bbox{i:03d}"
#         clip_img_name = f"{clip_stem}.jpg"
#         clip_json_name = f"{clip_stem}.json"

#         clip_img.save(out_dir / clip_img_name, "JPEG", quality=95)

#         clip_json = build_clip_json(data, shape, clip_img, clip_img_name, x1, y1)
#         with open(out_dir / clip_json_name, "w") as f:
#             json.dump(clip_json, f, indent=4)

#         print(f"  [{i}] {clip_img_name}  ({clip_img.width}x{clip_img.height})  conf={shape.get('conf', 'N/A'):.3f}")

#     print(f"Done: {len(data['shapes'])} clips saved to '{out_dir}'")



import json
from copy import deepcopy
from pathlib import Path

from PIL import Image



TARGET = 224  # output size in pixels


def clip_bbox(img, points, target=TARGET):
    """Crop a TARGET x TARGET window centered on the bbox, shifting if near image edge (no padding)."""
    (x1, y1), (x2, y2) = points
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    half = target / 2

    wx1 = int(round(cx - half))
    wy1 = int(round(cy - half))
    wx2 = wx1 + target
    wy2 = wy1 + target

    # Shift window to stay within image bounds (no padding)
    if wx1 < 0:
        wx2 -= wx1; wx1 = 0
    if wy1 < 0:
        wy2 -= wy1; wy1 = 0
    if wx2 > img.width:
        wx1 -= (wx2 - img.width); wx2 = img.width
    if wy2 > img.height:
        wy1 -= (wy2 - img.height); wy2 = img.height

    patch = img.crop((wx1, wy1, wx2, wy2))
    return patch, (wx1, wy1, wx2, wy2)


def build_clip_json(original, shape, clip_img, clip_filename, x1, y1):
    w, h = clip_img.size
    new_points = [
        [shape["points"][0][0] - x1, shape["points"][0][1] - y1],
        [shape["points"][1][0] - x1, shape["points"][1][1] - y1],
    ]
    new_shape = deepcopy(shape)
    new_shape["points"] = new_points
    return {
        "version": original.get("version", "0.4.29"),
        "flags": original.get("flags", {}),
        "shapes": [new_shape],
        "imagePath": clip_filename,
        "imageData": None,
        "imageHeight": h,
        "imageWidth": w,
    }


def find_image(json_path, image_dir=None):
    base_dir = Path(image_dir) if image_dir else json_path.parent
    stem = json_path.stem
    for ext in (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".JPG"):
        candidate = base_dir / (stem + ext)
        if candidate.exists():
            return candidate
    return None


def process_pair(json_path, img_path, out_dir):
    with open(json_path, "r") as f:
        data = json.load(f)

    img = Image.open(img_path).convert("RGB")
    stem = json_path.stem
    out_dir.mkdir(parents=True, exist_ok=True)

    for i, shape in enumerate(data.get("shapes", [])):
        if shape.get("shape_type") != "rectangle":
            print(f"  Skipping non-rectangle shape {i}")
            continue

        clip_img, (x1, y1, x2, y2) = clip_bbox(img, shape["points"])

        clip_stem = f"{stem}_bbox{i:03d}"
        clip_img_name = f"{clip_stem}.jpg"
        clip_json_name = f"{clip_stem}.json"

        clip_img.save(out_dir / clip_img_name, "JPEG", quality=100)

        clip_json = build_clip_json(data, shape, clip_img, clip_img_name, x1, y1)
        with open(out_dir / clip_json_name, "w") as f:
            json.dump(clip_json, f, indent=4)

        conf = shape.get('conf')
        conf_str = f"{conf:.3f}" if conf is not None else "N/A"
        print(f"  [{i}] {clip_img_name}  ({clip_img.width}x{clip_img.height})  conf={conf_str}")


    print(f"Done: {len(data['shapes'])} clips saved to '{out_dir}'")



# def process_field(field_dir: Path, max_clips=None, min_conf=0.0):
#     """
#     Process one field folder:
#       - JSONs read from {field_dir}/object_detection/
#       - Images read from {field_dir}/
#       - Clips saved to  {field_dir}/object_detection/Tile_images/  (created if missing)
#     """
#     od_dir  = field_dir / "object_detection"
#     out_dir = od_dir / "Tile_images"
# 
#     if not od_dir.exists():
#         print(f"  [skip] no object_detection folder: {field_dir.name}")
#         return 0
# 
#     json_files = sorted(od_dir.glob("*.json"))
#     if not json_files:
#         print(f"  [skip] no JSON files in {od_dir}")
#         return 0
# 
#     # Skip if Tile_images already exists and contains clips
#     if out_dir.exists() and any(out_dir.glob("*.jpg")):
#         existing = sum(1 for _ in out_dir.glob("*.jpg"))
#         print(f"  [skip] Tile_images already has {existing} clips — skipping")
#         return existing
# 
#     out_dir.mkdir(parents=True, exist_ok=True)
#     total = 0
# 
#     for json_path in json_files:
#         if max_clips and total >= max_clips:
#             print(f"  Reached MAX_CLIPS={max_clips}, stopping.")
#             break
# 
#         img_path = find_image(json_path, field_dir)
#         if not img_path:
#             print(f"  No matching image for {json_path.name}, skipping.")
#             continue
# 
#         with open(json_path) as f:
#             data = json.load(f)
# 
#         img  = Image.open(img_path).convert("RGB")
#         stem = json_path.stem
# 
#         for i, shape in enumerate(data.get("shapes", [])):
#             if max_clips and total >= max_clips:
#                 break
#             if shape.get("shape_type") != "rectangle":
#                 continue
#             if shape.get("conf", 0) < min_conf:
#                 continue
# 
#             clip_img, (x1, y1, x2, y2) = clip_bbox(img, shape["points"])
#             clip_stem     = f"{stem}_bbox{i:03d}"
#             clip_img_name = f"{clip_stem}.jpg"
# 
#             clip_img.save(out_dir / clip_img_name, "JPEG", quality=100)
# 
#             clip_json = build_clip_json(data, shape, clip_img, clip_img_name, x1, y1)
#             with open(out_dir / f"{clip_stem}.json", "w") as f:
#                 json.dump(clip_json, f, indent=4)
# 
#             total += 1
#             conf     = shape.get("conf")
#             conf_str = f"{conf:.3f}" if conf is not None else "N/A"
#             print(f"    [{total}] {clip_img_name}  ({clip_img.width}x{clip_img.height})  conf={conf_str}")
# 
#     return total


# if __name__ == "__main__":
#     # ── Configure ──────────────────────────────────────────────────────────────
#     PIPELINE_OUTPUT = Path(r"/home/rameen/Desktop/Prescout_pipeline_output/pipeline_output")
#     MAX_CLIPS       = None  # int to cap total clips per field, or None for all
#     MIN_CONF        = 0.0   # minimum confidence to include a detection
#     # ───────────────────────────────────────────────────────────────────────────
# 
#     field_dirs = sorted(d for d in PIPELINE_OUTPUT.iterdir() if d.is_dir())
#     grand_total = 0
# 
#     print(f"Found {len(field_dirs)} field folder(s) under {PIPELINE_OUTPUT}\n")
# 
#     for field_dir in field_dirs:
#         print(f"\n{'='*60}")
#         print(f"Field: {field_dir.name}")
#         print(f"  JSON  : {field_dir}/object_detection/*.json")
#         print(f"  Output: {field_dir}/object_detection/Tile_images/")
# 
#         n = process_field(field_dir, max_clips=MAX_CLIPS, min_conf=MIN_CONF)
#         grand_total += n
#         print(f"  → {n} clips saved")
# 
#     print(f"\n{'='*60}")
#     print(f"All done. Grand total clips saved: {grand_total}")
if __name__ == "__main__":
    # --- Configure these ---
    INPUT_DIR = Path("/home/rameen/Desktop/Prescout_pipeline_output/pipeline_output/2026_Pre-Scout_5300_-_SE_25-15-16/object_detection")
    IMAGE_DIR = Path("/home/rameen/Desktop/Prescout_pipeline_output/pipeline_output/2026_Pre-Scout_5300_-_SE_25-15-16")
    OUT_DIR   = Path("/home/rameen/Desktop/Prescout_pipeline_output/pipeline_output/2026_Pre-Scout_5300_-_SE_25-15-16/object_detection/Tile_images")
    MAX_CLIPS  = None  # set to an int e.g. 50 to stop after that many clips, or None for all
    MIN_CONF   = 0.0   # only clip shapes with conf >= this value, set to 0.0 for all
    # -----------------------

    out_dir = OUT_DIR or INPUT_DIR / "clips"
    json_files = sorted(INPUT_DIR.glob("*.json"))
    total = 0

    if not json_files:
        print("No JSON files found.")
    else:
        for json_path in json_files:
            if MAX_CLIPS and total >= MAX_CLIPS:
                break
            img_path = find_image(json_path, IMAGE_DIR)
            if not img_path:
                print(f"No matching image for {json_path.name}, skipping.")
                continue
            print(f"Processing: {json_path.name}")
            with open(json_path) as f:
                data = json.load(f)
            img = Image.open(img_path).convert("RGB")
            stem = json_path.stem
            out_dir.mkdir(parents=True, exist_ok=True)
            for i, shape in enumerate(data.get("shapes", [])):
                if MAX_CLIPS and total >= MAX_CLIPS:
                    print(f"Reached MAX_CLIPS={MAX_CLIPS}, stopping.")
                    break
                if shape.get("shape_type") != "rectangle":
                    continue
                if shape.get("conf", 0) < MIN_CONF:
                    print(f"  Skipping shape {i} (conf={shape.get('conf', 0):.3f} < {MIN_CONF})")
                    continue
                clip_img, (x1, y1, x2, y2) = clip_bbox(img, shape["points"])
                clip_stem = f"{stem}_bbox{i:03d}"
                clip_img_name = f"{clip_stem}.jpg"
                clip_img.save(out_dir / clip_img_name, "JPEG", quality=100)
                clip_json = build_clip_json(data, shape, clip_img, clip_img_name, x1, y1)
                with open(out_dir / f"{clip_stem}.json", "w") as f:
                    json.dump(clip_json, f, indent=4)
                total += 1
                print(f"  [{total}] {clip_img_name}  ({clip_img.width}x{clip_img.height})  conf={shape.get('conf', 'N/A'):.3f}")

    print(f"\nFinished. Total clips saved: {total}")

