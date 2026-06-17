# rename-jpgs.py

## Overview

Renames image files in the `cn_coty` and `non_coty` validation subfolders by appending a class suffix to each filename stem.

## What It Does

For each class folder, appends a numeric suffix in parentheses to every file's stem:

| Class folder | Suffix appended | Example |
|---|---|---|
| `cn_coty` | `(0)` | `image_bbox000.jpg` → `image_bbox000(0).jpg` |
| `non_coty` | `(1)` | `image_bbox001.jpg` → `image_bbox001(1).jpg` |

The extension is preserved unchanged.

## Configuration (hardcoded)

```python
base_dir = Path("/home/farmevo/Downloads/test-effi/val")
```

Edit `base_dir` directly in the script before running.

## Usage

```bash
python rename-jpgs.py
```

## Code

```python
for folder, suffix in [("cn_coty", "0"), ("non_coty", "1")]:
    for file in (base_dir / folder).rglob("*"):
        if file.is_file():
            new_name = file.stem + f"({suffix})" + file.suffix
            file.rename(file.parent / new_name)
```

- Uses `rglob("*")` — recurses into all subdirectories.
- Renames **in place** (no copy).

## Notes

- This is a one-shot utility — running it twice will double-append the suffix (e.g., `image(0)(0).jpg`). Ensure files haven't already been renamed.
- There is a minor bug: `print(f"{folder} done")` is outside the outer `for` loop, so it only prints once after both folders are processed.

## Dependencies

`os`, `pathlib` (stdlib only)
