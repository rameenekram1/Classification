"""
Group-aware, leakage-free dataset splitter for the coty classifier.
==================================================================

Why this exists
---------------
Random per-image splits leak across train/val in two ways:
  1. Multiple bboxes (`bboxNNN`) are crops of the SAME source image.
     If one goes to train and another to val, val accuracy is fake.
  2. Images from the same flight session (same date+hour+field) look
     near-identical because lighting, soil, camera angle are all locked
     in for that flight. Splitting within a flight leaks too.

This splitter groups by source image AND by flight session, then assigns
whole groups to train or val. Same flight, same source -> same split.

It also supports pinning specific dates or fields to val (the "hard" set):
  --val_dates  20260511,20260512,20260518
  --val_fields FIELD7020LA,FIELD7008LA
This is how you get an unseen-field generalization estimate.

Optionally re-balances class counts in train (and/or val) by subsampling
the majority class. With class_weights="balanced" in train_coty_classifier_v4.py
this is optional — but cleaner training metrics if you do it.

USAGE
-----
# 1) Pure stratified group split, 80/20, balance train classes
python split_dataset_v2.py \
    --source /home/rameen/final_dataset_v7_1to1 \
    --output /home/rameen/final_dataset_v7_1to1_split \
    --val_split 0.20 \
    --balance train

# 2) Hold out the 2026 field codes for val (recommended generalization test)
python split_dataset_v2.py \
    --source /home/rameen/final_dataset_v7_1to1 \
    --output /home/rameen/final_dataset_v7_1to1_split \
    --val_fields FIELD7020LA,FIELD7008LA,FIELD7069LA \
    --val_split 0.20 \
    --balance train

# 3) Hold out specific dates for val
python split_dataset_v2.py \
    --source /home/rameen/final_dataset_v7_1to1 \
    --output /home/rameen/final_dataset_v7_1to1_split \
    --val_dates 20260511,20260512,20260518 \
    --balance train
"""

from __future__ import annotations
import argparse
import os
import random
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# Lenient DJI parser: captures timestamp, seq, sensor, then anything before _bbox.
# Examples it handles:
#   DJI_20250101123926_0002_D_LA_bbox000.jpg
#   DJI_20250520153021_0058_D_LA_tile_0_0_bbox000.jpg
#   DJI_20260511133924_0023_V_FIELD7020LA_20260512_123536_bbox05.jpg
#   DJI_20260512125405_0008_V_20260512_132602_LA_bbox000.jpg
RE_FILE = re.compile(
    r'^DJI_(?P<ts>\d{14})_(?P<seq>\d+)_(?P<sensor>[DV])_(?P<rest>.+?)_bbox\d+',
    re.IGNORECASE,
)
RE_BBOX_STRIP = re.compile(r'_bbox\d+.*$', re.IGNORECASE)
RE_FIELD_TOKEN = re.compile(r'FIELD\d+[A-Z]*', re.IGNORECASE)


def parse_image(path: Path) -> dict:
    """Return a dict with date, hour, sensor, field, source_id, flight_id."""
    name = path.name
    # source_id = everything before _bboxNNN — same source image = same crop origin
    source_id = RE_BBOX_STRIP.sub("", name)

    m = RE_FILE.match(name)
    if m:
        ts     = m.group("ts")
        sensor = m.group("sensor").upper()
        rest   = m.group("rest")
        date   = ts[:8]
        hour   = ts[8:10]
        # Pick a specific FIELD code if present, else fall back to generic "LA"
        fm = RE_FIELD_TOKEN.search(rest)
        field = fm.group(0).upper() if fm else "LA"
    else:
        # Unparseable: bucket on its own so it doesn't pollute groups
        date, hour, sensor, field = "UNK", "UNK", "U", "UNK"

    # Flight session = same date + hour + field. Tight enough that all crops
    # from the same flight stay together.
    flight_id = f"{date}_{hour}_{field}_{sensor}"
    return {
        "date":      date,
        "hour":      hour,
        "sensor":    sensor,
        "field":     field,
        "source_id": source_id,
        "flight_id": flight_id,
    }


def collect(class_dir: Path) -> list[dict]:
    out = []
    for f in class_dir.iterdir():
        if f.suffix.lower() not in IMG_EXTS:
            continue
        meta = parse_image(f)
        meta["path"] = f
        out.append(meta)
    return out


def assign_groups(items: list[dict],
                  val_split: float,
                  val_dates: set[str],
                  val_fields: set[str],
                  stratify_by_field: bool,
                  seed: int) -> tuple[list[dict], list[dict]]:
    """
    Group items by source_id (no source ever splits). For each field, bin its
    sources by flight_id and shuffle whole flights into train/val to hit
    val_split — done PER FIELD so every field appears in both splits when it
    has >=2 flights. Items matching val_dates / val_fields go to val
    unconditionally (overrides stratification).
    """
    rng = random.Random(seed)

    # source_id -> list of items
    sources: dict[str, list[dict]] = defaultdict(list)
    for it in items:
        sources[it["source_id"]].append(it)

    # Forced-val: any source whose date or field is explicitly held out
    forced_val_sources: set[str] = set()
    for sid, group in sources.items():
        meta = group[0]
        if meta["date"] in val_dates or meta["field"] in val_fields:
            forced_val_sources.add(sid)

    val_sources: set[str] = set(forced_val_sources)

    if stratify_by_field:
        # Per-field stratified flight assignment — each field's flights are
        # split val_split-ways independently, so every field with enough
        # flights appears in train AND val.
        by_field: dict[str, list[str]] = defaultdict(list)
        for sid in sources:
            if sid in forced_val_sources:
                continue
            by_field[sources[sid][0]["field"]].append(sid)

        for field, field_sids in by_field.items():
            # Group this field's sources by flight
            flights: dict[str, list[str]] = defaultdict(list)
            for sid in field_sids:
                flights[sources[sid][0]["flight_id"]].append(sid)
            flight_ids = list(flights.keys())
            rng.shuffle(flight_ids)

            # Single-flight field: can't split without source leakage. Keep
            # in TRAIN so the model gets to learn the field.
            if len(flight_ids) < 2:
                continue

            total_field_n  = sum(len(sources[s]) for s in field_sids)
            target_val_n   = int(round(val_split * total_field_n))
            count = 0
            # Always reserve at least one flight for train (first in shuffle).
            for fid in flight_ids[1:]:
                if count >= target_val_n:
                    break
                for sid in flights[fid]:
                    val_sources.add(sid)
                    count += len(sources[sid])
    else:
        # Global flight shuffle (no per-field stratification) — for held-out
        # generalization tests via --val_fields / --val_dates.
        remaining = [s for s in sources if s not in forced_val_sources]
        flights: dict[str, list[str]] = defaultdict(list)
        for sid in remaining:
            flights[sources[sid][0]["flight_id"]].append(sid)
        total_items  = sum(len(g) for g in sources.values())
        forced_val_n = sum(len(sources[s]) for s in forced_val_sources)
        need_val_n   = max(0, int(round(val_split * total_items)) - forced_val_n)
        flight_ids   = list(flights.keys())
        rng.shuffle(flight_ids)
        count = forced_val_n
        for fid in flight_ids:
            if count >= need_val_n:
                break
            for sid in flights[fid]:
                val_sources.add(sid)
                count += len(sources[sid])

    train_items, val_items = [], []
    for sid, group in sources.items():
        (val_items if sid in val_sources else train_items).extend(group)
    return train_items, val_items


def balance(items: list[dict], by_field="path", seed=42) -> list[dict]:
    """Subsample so the (implicit) majority is trimmed — here we just
    return items as-is; balancing across classes is done at the caller."""
    return items  # placeholder; class-level balancing happens elsewhere


def copy_items(items: list[dict], dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    for it in items:
        shutil.copy2(it["path"], dest / it["path"].name)


def summarise(items: list[dict]) -> dict:
    return {
        "n":          len(items),
        "n_sources":  len({it["source_id"] for it in items}),
        "n_flights":  len({it["flight_id"] for it in items}),
        "dates":      sorted({it["date"]   for it in items}),
        "fields":     sorted({it["field"]  for it in items}),
        "sensors":    Counter(it["sensor"] for it in items),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source",   required=True, help="Root with <class>/ subdirs")
    ap.add_argument("--output",   required=True, help="Where train/val/<class>/ go")
    ap.add_argument("--val_split", type=float, default=0.20)
    ap.add_argument("--val_dates",  type=str, default="",
                    help="Comma-separated YYYYMMDD dates forced to val")
    ap.add_argument("--val_fields", type=str, default="",
                    help="Comma-separated field codes forced to val (e.g. FIELD7020LA,FIELD7008LA)")
    ap.add_argument("--balance", choices=["none", "train", "both"], default="train",
                    help="Subsample majority class in train (and/or val) to match minority")
    ap.add_argument("--stratify_by_field", action="store_true", default=True,
                    help="Default. Split flights PER FIELD so every field appears in train AND val.")
    ap.add_argument("--no-stratify", dest="stratify_by_field", action="store_false",
                    help="Disable field stratification — global flight shuffle. Use for "
                         "held-out-field tests (with --val_fields).")
    ap.add_argument("--seed",     type=int, default=42)
    args = ap.parse_args()

    src    = Path(args.source)
    out    = Path(args.output)
    val_dates  = {d.strip() for d in args.val_dates.split(",")  if d.strip()}
    val_fields = {f.strip().upper() for f in args.val_fields.split(",") if f.strip()}

    classes = sorted([d.name for d in src.iterdir() if d.is_dir()])
    if not classes:
        raise SystemExit(f"No class subdirs in {src}")

    print(f"Classes:           {classes}")
    print(f"val_split:         {args.val_split}")
    print(f"val_dates:         {sorted(val_dates) or '—'}")
    print(f"val_fields:        {sorted(val_fields) or '—'}")
    print(f"stratify_by_field: {args.stratify_by_field}")
    print(f"balance:           {args.balance}")
    print(f"seed:              {args.seed}")
    print()

    splits = {"train": {}, "val": {}}
    for cls in classes:
        items = collect(src / cls)
        tr, va = assign_groups(items, args.val_split, val_dates, val_fields,
                               args.stratify_by_field, args.seed)
        splits["train"][cls] = tr
        splits["val"][cls]   = va

        s_tr, s_va = summarise(tr), summarise(va)
        n_total = s_tr["n"] + s_va["n"]
        print(f"[{cls}] total={n_total}")
        print(f"  TRAIN n={s_tr['n']:>5}  sources={s_tr['n_sources']:>4}  flights={s_tr['n_flights']:>3}  "
              f"dates={len(s_tr['dates'])}  fields={s_tr['fields']}")
        print(f"  VAL   n={s_va['n']:>5}  sources={s_va['n_sources']:>4}  flights={s_va['n_flights']:>3}  "
              f"dates={len(s_va['dates'])}  fields={s_va['fields']}")
        # Leakage assertion: no source_id appears in both splits
        overlap = ({it["source_id"] for it in tr} & {it["source_id"] for it in va})
        assert not overlap, f"Leakage! {len(overlap)} sources in both train and val for {cls}"

    # Class balancing AFTER no-leakage split — subsample inside each class's pool
    # to equalize across classes per split.
    rng = random.Random(args.seed)
    if args.balance in ("train", "both"):
        target = min(len(splits["train"][c]) for c in classes)
        for cls in classes:
            if len(splits["train"][cls]) > target:
                splits["train"][cls] = rng.sample(splits["train"][cls], target)
        print(f"\nBalanced TRAIN to {target} per class.")
    if args.balance == "both":
        target = min(len(splits["val"][c]) for c in classes)
        for cls in classes:
            if len(splits["val"][cls]) > target:
                splits["val"][cls] = rng.sample(splits["val"][cls], target)
        print(f"Balanced VAL to {target} per class.")

    # Write out
    print(f"\nWriting to {out} ...")
    for split, by_class in splits.items():
        for cls, items in by_class.items():
            copy_items(items, out / split / cls)
            print(f"  {split}/{cls}: {len(items)}")

    print("\nDone.")
    print(f"\nNext:\n  python train_coty_classifier_v4.py --model efficientnet_b0 "
          f"--data {out} --name coty_effnetb0_v4 --epochs 60 --batch 64")


if __name__ == "__main__":
    main()
