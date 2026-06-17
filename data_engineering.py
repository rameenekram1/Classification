"""
Data Engineering for Coty Classifier
=====================================
Diagnoses and fixes small-plant images in the dataset.

Steps:
  1. Scan all images, score plant coverage via HSV hue range (green/yellow-green).
  2. Report distribution per class.
  3. Build a cleaned dataset:
     - BAD  (<bad_thresh)          : skip (these teach nothing about plant features).
     - SMALL (bad_thresh–small_thresh): center-crop zoom → resize so plant fills more of frame.
     - GOOD  (>small_thresh)       : copy as-is.

Per-class thresholds rationale:
  cn_coty:  apply full filtering — we need clear cotyledon features to learn from.
  non_coty: apply conservative filtering only — soil-only and low-plant images ARE
            valid non-coty examples (bare ground, dense canopy where individual plant
            is hard to isolate). Only drop truly empty frames.

USAGE:
    # 1) Dry-run analysis only (no files written):
    python data_engineering.py \\
        --src /home/rameen/final_dataset_v7_1to1_split \\
        --analyze_only

    # 2) Build cleaned dataset (recommended settings):
    python data_engineering.py \\
        --src /home/rameen/final_dataset_v7_1to1_split \\
        --dst /home/rameen/final_dataset_v8_cleaned \\
        --cn_coty_bad 0.03  --cn_coty_small 0.10  \\
        --non_coty_bad 0.01 --non_coty_small 0.04  \\
        --zoom_crop 0.65

    # 3) Then re-split and train:
    python split_dataset_v2.py \\
        --source /home/rameen/final_dataset_v8_cleaned \\
        --output /home/rameen/final_dataset_v8_split \\
        --val_split 0.20 --balance train
    python train_coty_classifier_v5.py \\
        --model efficientnet_b2 \\
        --data /home/rameen/final_dataset_v8_split \\
        --name coty_effnetb2_v13 --epochs 100 --batch 32
"""

import argparse
import csv
import os
import shutil
from pathlib import Path

import numpy as np
from PIL import Image

# ─── Plant coverage score ─────────────────────────────────────────────────────

def plant_score(img_path: str) -> float:
    """
    Fraction of pixels that look like green/yellow-green plant tissue.
    Uses HSV hue 40-170° (covers yellow-green → green → cyan-green),
    saturation > 0.12, value > 40.
    """
    img = np.array(Image.open(img_path).convert("RGB"), dtype=np.float32)
    r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    maxc = np.maximum(np.maximum(r, g), b)
    minc = np.minimum(np.minimum(r, g), b)
    delta = maxc - minc + 1e-6

    sat = np.where(maxc > 0, delta / maxc, 0.0)

    # Compute hue only for green-channel-dominant pixels (hue 60–180)
    hue = np.full_like(r, -1.0)
    m = (maxc == g) & (delta > 0)
    hue[m] = 60.0 * ((b[m] - r[m]) / delta[m]) + 120.0

    plant_mask = (hue >= 40) & (hue <= 175) & (sat > 0.12) & (maxc > 40)
    return float(plant_mask.mean())


# ─── Center-crop zoom ─────────────────────────────────────────────────────────

def center_crop_zoom(img: Image.Image, crop_frac: float) -> Image.Image:
    """
    Crop the central `crop_frac` fraction of the image and resize back to
    the original dimensions. E.g. crop_frac=0.65 on a 224×224 image crops
    the central 145×145 pixels and stretches it back to 224×224.
    """
    w, h = img.size
    cw, ch = int(w * crop_frac), int(h * crop_frac)
    left  = (w - cw) // 2
    top   = (h - ch) // 2
    return img.crop((left, top, left + cw, top + ch)).resize((w, h), Image.LANCZOS)


# ─── Analyse one folder ───────────────────────────────────────────────────────

def analyse_folder(folder: Path, bad_thresh: float, small_thresh: float):
    exts = {".jpg", ".jpeg", ".png"}
    files = [f for f in folder.iterdir() if f.suffix.lower() in exts]
    scores = []
    print(f"  Scoring {len(files)} images in {folder.name} …", flush=True)
    for f in files:
        try:
            scores.append((plant_score(str(f)), f))
        except Exception:
            pass
    scores.sort()
    if not scores:
        return scores

    vals = [s for s, _ in scores]
    print(f"  Plant-coverage percentiles:")
    for pct in [5, 10, 25, 50, 75, 90]:
        idx = int(pct / 100 * len(vals))
        print(f"    {pct:2d}th : {vals[idx]*100:5.1f}%")

    bad   = sum(1 for v in vals if v < bad_thresh)
    small = sum(1 for v in vals if bad_thresh <= v < small_thresh)
    good  = sum(1 for v in vals if v >= small_thresh)
    n = len(vals)
    print(f"  BAD   (<{bad_thresh*100:.0f}% plant) : {bad:4d} / {n}  ({bad/n*100:.1f}%) — will be DROPPED")
    print(f"  SMALL ({bad_thresh*100:.0f}–{small_thresh*100:.0f}%)    : {small:4d} / {n}  ({small/n*100:.1f}%) — will be ZOOMED")
    print(f"  GOOD  (>{small_thresh*100:.0f}% plant) : {good:4d} / {n}  ({good/n*100:.1f}%) — kept as-is")
    return scores


def visualize_samples(folder: Path, bad_thresh: float, small_thresh: float,
                      zoom_crop: float, out_path: Path, n_per_tier: int = 4):
    """Save a grid: BAD | SMALL (before→after zoom) | GOOD — to show data quality."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  [visualize] matplotlib not installed — skipping.")
        return

    exts = {".jpg", ".jpeg", ".png"}
    files = [f for f in folder.iterdir() if f.suffix.lower() in exts]
    scored = []
    for f in files[:2000]:
        try:
            scored.append((plant_score(str(f)), f))
        except Exception:
            pass
    scored.sort()

    bad_files   = [f for s, f in scored if s < bad_thresh][:n_per_tier]
    small_files = [f for s, f in scored if bad_thresh <= s < small_thresh][:n_per_tier]
    good_files  = [f for s, f in scored if s >= small_thresh][-n_per_tier:]

    n_cols = n_per_tier * 3  # bad | small_before | small_after... | good
    fig, axes = plt.subplots(2, n_per_tier * 2, figsize=(n_per_tier * 5, 8))
    fig.suptitle(f"{folder.name} — BAD (dropped)  |  SMALL before→after zoom  |  GOOD (kept)",
                 fontsize=11)

    def show(ax, path, title, zoom=False):
        img = Image.open(path)
        if zoom:
            img = center_crop_zoom(img, zoom_crop)
        ax.imshow(img)
        ax.set_title(title, fontsize=7, pad=2)
        ax.axis("off")

    for i in range(n_per_tier):
        if i < len(bad_files):
            show(axes[0, i], bad_files[i], f"BAD ({plant_score(str(bad_files[i]))*100:.1f}%)")
        else:
            axes[0, i].axis("off")
        if i < len(small_files):
            s = plant_score(str(small_files[i]))
            show(axes[0, n_per_tier + i], small_files[i], f"SMALL before ({s*100:.1f}%)")
            show(axes[1, n_per_tier + i], small_files[i], f"SMALL after zoom", zoom=True)
        else:
            axes[0, n_per_tier + i].axis("off")
            axes[1, n_per_tier + i].axis("off")
        if i < len(good_files):
            show(axes[1, i], good_files[i], f"GOOD ({plant_score(str(good_files[i]))*100:.1f}%)")
        else:
            axes[1, i].axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  [visualize] Saved → {out_path}")


# ─── Build cleaned dataset ────────────────────────────────────────────────────

def process_split(
    src_split: Path,
    dst_split: Path,
    class_thresholds: dict,
    default_bad: float,
    default_small: float,
    zoom_crop: float,
    report_rows: list,
):
    """Process one split (train/val) of the dataset."""
    dst_split.mkdir(parents=True, exist_ok=True)
    kept = dropped = zoomed = 0

    for cls_dir in sorted(src_split.iterdir()):
        if not cls_dir.is_dir():
            continue
        out_cls = dst_split / cls_dir.name
        out_cls.mkdir(exist_ok=True)
        exts = {".jpg", ".jpeg", ".png"}
        files = [f for f in cls_dir.iterdir() if f.suffix.lower() in exts]

        bad_t, small_t = class_thresholds.get(cls_dir.name, (default_bad, default_small))
        print(f"    [{src_split.name}/{cls_dir.name}]  {len(files)} images  "
              f"(bad<{bad_t*100:.0f}%  zoom<{small_t*100:.0f}%)", flush=True)

        for f in files:
            try:
                score = plant_score(str(f))
            except Exception:
                continue

            if score < bad_t:
                action = "dropped"
                dropped += 1
                report_rows.append([str(f), f"{score:.4f}", action])
                continue

            img = Image.open(f)
            if score < small_t:
                img = center_crop_zoom(img, zoom_crop)
                action = "zoomed"
                zoomed += 1
            else:
                action = "kept"
                kept += 1

            img.save(out_cls / f.name, quality=95)
            report_rows.append([str(f), f"{score:.4f}", action])

    print(f"    → kept={kept}  zoomed={zoomed}  dropped={dropped}")
    return kept, zoomed, dropped


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src",          required=True,  help="Source dataset root")
    ap.add_argument("--dst",          default=None,   help="Output root (required unless --analyze_only)")
    ap.add_argument("--zoom_crop",    type=float, default=0.65,
                    help="Center-crop fraction for SMALL images (0.65 → 1.54× zoom in)")
    ap.add_argument("--analyze_only", action="store_true",
                    help="Print stats and exit without writing any files")
    # Per-class thresholds — cn_coty needs stricter filtering (must learn clear features)
    # non_coty is kept lenient because bare soil / other backgrounds ARE valid negatives
    ap.add_argument("--cn_coty_bad",   type=float, default=0.03,
                    help="cn_coty:  drop images with plant coverage below this (default 3%%)")
    ap.add_argument("--cn_coty_small", type=float, default=0.10,
                    help="cn_coty:  center-zoom images with coverage below this (default 10%%)")
    ap.add_argument("--non_coty_bad",  type=float, default=0.01,
                    help="non_coty: drop images with plant coverage below this (default 1%%)")
    ap.add_argument("--non_coty_small",type=float, default=0.04,
                    help="non_coty: center-zoom images with coverage below this (default 4%%)")
    ap.add_argument("--visualize", action="store_true",
                    help="Save a sample grid image per class showing BAD/SMALL/GOOD examples")
    args = ap.parse_args()

    src = Path(args.src)

    class_thresholds = {
        "cn_coty":  (args.cn_coty_bad,  args.cn_coty_small),
        "non_coty": (args.non_coty_bad, args.non_coty_small),
    }

    # ── Detect layout: flat (cn_coty/non_coty) or split (train/val/cn_coty) ──
    sub = [d for d in src.iterdir() if d.is_dir()]
    has_splits = any(d.name in {"train", "val", "test"} for d in sub)

    print(f"\n{'='*60}")
    print(f"Source : {src}")
    print(f"Layout : {'train/val splits' if has_splits else 'flat class dirs'}")
    print(f"cn_coty  thresholds: bad<{args.cn_coty_bad*100:.0f}%  zoom<{args.cn_coty_small*100:.0f}%")
    print(f"non_coty thresholds: bad<{args.non_coty_bad*100:.0f}%  zoom<{args.non_coty_small*100:.0f}%")
    print(f"zoom_crop: {args.zoom_crop}  ({1/args.zoom_crop:.2f}× zoom in)")
    print(f"{'='*60}\n")

    # ── Analysis pass ─────────────────────────────────────────────────────────
    def _get_thresh(cls_name):
        return class_thresholds.get(cls_name, (args.cn_coty_bad, args.cn_coty_small))

    vis_base = src  # save visualizations alongside source
    if has_splits:
        for split_dir in sorted(sub):
            if not split_dir.is_dir():
                continue
            print(f"── {split_dir.name} ──")
            for cls_dir in sorted(split_dir.iterdir()):
                if cls_dir.is_dir():
                    bad_t, small_t = _get_thresh(cls_dir.name)
                    print(f"  Class: {cls_dir.name}  (bad<{bad_t*100:.0f}%  zoom<{small_t*100:.0f}%)")
                    analyse_folder(cls_dir, bad_t, small_t)
                    if args.visualize:
                        vp = vis_base / f"vis_{split_dir.name}_{cls_dir.name}.png"
                        visualize_samples(cls_dir, bad_t, small_t, args.zoom_crop, vp)
    else:
        for cls_dir in sorted(sub):
            if cls_dir.is_dir():
                bad_t, small_t = _get_thresh(cls_dir.name)
                print(f"── Class: {cls_dir.name}  (bad<{bad_t*100:.0f}%  zoom<{small_t*100:.0f}%) ──")
                analyse_folder(cls_dir, bad_t, small_t)
                if args.visualize:
                    vp = vis_base / f"vis_{cls_dir.name}.png"
                    visualize_samples(cls_dir, bad_t, small_t, args.zoom_crop, vp)

    if args.analyze_only:
        print("\n[analyze_only] Done. No files written.")
        return

    # ── Build output dataset ───────────────────────────────────────────────────
    if not args.dst:
        print("\nERROR: --dst is required when not using --analyze_only")
        return

    dst = Path(args.dst)
    dst.mkdir(parents=True, exist_ok=True)

    report_rows = []
    total_kept = total_zoomed = total_dropped = 0

    print(f"\nBuilding cleaned dataset → {dst}\n")
    if has_splits:
        for split_dir in sorted(sub):
            if not split_dir.is_dir():
                continue
            k, z, d = process_split(
                split_dir, dst / split_dir.name,
                class_thresholds, args.cn_coty_bad, args.cn_coty_small,
                args.zoom_crop, report_rows
            )
            total_kept += k; total_zoomed += z; total_dropped += d
    else:
        k, z, d = process_split(
            src, dst,
            class_thresholds, args.cn_coty_bad, args.cn_coty_small,
            args.zoom_crop, report_rows
        )
        total_kept += k; total_zoomed += z; total_dropped += d

    # Save CSV report
    report_path = dst / "data_engineering_report.csv"
    with open(report_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["file", "plant_score", "action"])
        w.writerows(report_rows)

    total = total_kept + total_zoomed + total_dropped
    print(f"\n{'='*60}")
    print(f"DONE")
    print(f"  Total processed : {total}")
    print(f"  Kept as-is      : {total_kept}  ({total_kept/max(total,1)*100:.1f}%)")
    print(f"  Center-zoomed   : {total_zoomed}  ({total_zoomed/max(total,1)*100:.1f}%)")
    print(f"  Dropped (bad)   : {total_dropped}  ({total_dropped/max(total,1)*100:.1f}%)")
    print(f"  Report          : {report_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
