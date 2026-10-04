#!/usr/bin/env python3
"""
download_jewellery.py - Download jewellery images from Open Images V7 via FiftyOne
and integrate them into the YOLO fashion dataset.

Verified Open Images V7 boxable classes:
    - 'Necklace' (ID /m/01llwg): EXISTS
    - 'Earrings' (ID /m/01r546): EXISTS
    - 'Ring' / 'Rings': DOES NOT EXIST in Open Images V7 boxable detection classes
    - 'Bracelet': DOES NOT EXIST in Open Images V7 boxable detection classes
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path
from PIL import Image
from tqdm import tqdm

# Ensure cache dirs are on D: to prevent filling C:
PROJECT_DIR = Path(__file__).resolve().parent
CACHE_TMP = PROJECT_DIR / ".cache_tmp"
FIFTYONE_DIR = PROJECT_DIR / ".fiftyone_cache"

CACHE_TMP.mkdir(parents=True, exist_ok=True)
FIFTYONE_DIR.mkdir(parents=True, exist_ok=True)

os.environ["TEMP"] = str(CACHE_TMP)
os.environ["TMP"] = str(CACHE_TMP)
os.environ["FIFTYONE_DEFAULT_DATASET_DIR"] = str(FIFTYONE_DIR)
os.environ["FIFTYONE_DATASET_ZOO_DIR"] = str(FIFTYONE_DIR)

import pandas as pd
import fiftyone as fo
import fiftyone.zoo as foz
import fiftyone.utils.openimages as fouo

fo.config.default_dataset_dir = str(FIFTYONE_DIR)
fo.config.dataset_zoo_dir = str(FIFTYONE_DIR)

TARGET_JEWELLERY_CLASSES = ["Necklace", "Earrings"]
CANDIDATE_CLASSES = ["Necklace", "Earrings", "Ring", "Rings", "Bracelet"]

# Open Images class IDs for Necklace (/m/01llwg) and Earrings (/m/01r546)
TARGET_JEWELLERY_OIDS = {"/m/01llwg", "/m/01r546"}

_orig_parse_csv = fouo._parse_csv

def _chunked_parse_csv(filename, dataframe=False, index_col=None):
    if dataframe and os.path.exists(filename) and os.path.getsize(filename) > 200_000_000:
        print(f"[stream] Streaming large CSV in chunks ({filename})...")
        chunks = []
        for chunk in pd.read_csv(filename, chunksize=250000, low_memory=True):
            sub = chunk[chunk["LabelName"].isin(TARGET_JEWELLERY_OIDS)]
            if len(sub) > 0:
                chunks.append(sub)
        df = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
        print(f"[stream] Filtered down to {len(df)} annotations across {df['ImageID'].nunique()} images.")
        return df
    return _orig_parse_csv(filename, dataframe=dataframe, index_col=index_col)

fouo._parse_csv = _chunked_parse_csv


def check_available_classes() -> dict[str, bool]:
    """Check which of the candidate jewellery classes exist in Open Images V7."""
    print("[check] Querying Open Images V7 boxable detection classes...")
    all_classes = fouo.get_classes(version="v7")
    class_set = set(all_classes)
    
    results = {}
    for c in CANDIDATE_CLASSES:
        exists = c in class_set
        results[c] = exists
        print(f"  - {c:12s}: {'EXISTS' if exists else 'NOT FOUND'}")
    
    # Also check if any other jewellery-related terms exist
    extras = [c for c in all_classes if any(w in c.lower() for w in ["jewel", "ring", "pendant", "bracelet", "earring", "necklace"])]
    print(f"[check] All matching classes in Open Images V7: {extras}")
    return results


def download_and_export_jewellery(
    data_dir: Path,
    n_train: int = 1000,
    n_val: int = 100,
    jewellery_class_id: int = 3,
    max_side: int = 1024,
    jpeg_quality: int = 95,
) -> dict:
    """Download jewellery images via FiftyOne and save in YOLO format."""
    print(f"\n[download] Starting download of Open Images V7 jewellery dataset...")
    print(f"  Target classes: {TARGET_JEWELLERY_CLASSES}")
    print(f"  Requested: {n_train} train images, {n_val} val images")
    print(f"  Storage cache: {FIFTYONE_DIR}")

    train_img_dir = data_dir / "images" / "train"
    train_lbl_dir = data_dir / "labels" / "train"
    val_img_dir = data_dir / "images" / "val"
    val_lbl_dir = data_dir / "labels" / "val"

    for d in [train_img_dir, train_lbl_dir, val_img_dir, val_lbl_dir]:
        d.mkdir(parents=True, exist_ok=True)

    stats = {"train_images": 0, "train_boxes": 0, "val_images": 0, "val_boxes": 0}

    # 1. Download Validation Split
    if n_val > 0:
        print("\n[download] Downloading validation samples...")
        val_ds = foz.load_zoo_dataset(
            "open-images-v7",
            split="validation",
            label_types=["detections"],
            classes=TARGET_JEWELLERY_CLASSES,
            max_samples=n_val,
            shuffle=True,
            seed=42,
        )
        print(f"[download] Loaded {len(val_ds)} validation samples from FiftyOne.")
        v_imgs, v_boxes = convert_and_save_samples(
            val_ds, val_img_dir, val_lbl_dir, jewellery_class_id, "oi_val", max_side, jpeg_quality
        )
        stats["val_images"] = v_imgs
        stats["val_boxes"] = v_boxes
        print(f"[data] Saved {v_imgs} val images with {v_boxes} jewellery boxes.")

    # 2. Download Train Split
    if n_train > 0:
        print("\n[download] Downloading training samples (~1k requested)...")
        train_ds = foz.load_zoo_dataset(
            "open-images-v7",
            split="train",
            label_types=["detections"],
            classes=TARGET_JEWELLERY_CLASSES,
            max_samples=n_train,
            shuffle=True,
            seed=42,
        )
        print(f"[download] Loaded {len(train_ds)} train samples from FiftyOne.")
        t_imgs, t_boxes = convert_and_save_samples(
            train_ds, train_img_dir, train_lbl_dir, jewellery_class_id, "oi_train", max_side, jpeg_quality
        )
        stats["train_images"] = t_imgs
        stats["train_boxes"] = t_boxes
        print(f"[data] Saved {t_imgs} train images with {t_boxes} jewellery boxes.")

    return stats


def convert_and_save_samples(
    dataset,
    img_out_dir: Path,
    lbl_out_dir: Path,
    class_id: int,
    prefix: str,
    max_side: int,
    jpeg_quality: int,
) -> tuple[int, int]:
    """Extract jewellery bounding boxes, normalize, resize if needed, and save YOLO files."""
    num_images = 0
    total_boxes = 0

    for sample in tqdm(dataset, desc=f"Converting {prefix}"):
        src_path = Path(sample.filepath)
        if not src_path.exists():
            continue

        try:
            with Image.open(src_path) as im:
                img = im.convert("RGB")
                W, H = img.size
        except Exception as e:
            print(f"[WARN] Failed opening {src_path}: {e}")
            continue

        # In FiftyOne detections, bounding_box = [x, y, w, h] (normalized top-left + width/height)
        detections = getattr(sample, "ground_truth", None)
        if detections is None or not detections.detections:
            continue

        lines = []
        for det in detections.detections:
            if det.label not in TARGET_JEWELLERY_CLASSES:
                continue

            x, y, bw, bh = det.bounding_box
            # Clamp to [0, 1]
            x1 = max(0.0, min(1.0, float(x)))
            y1 = max(0.0, min(1.0, float(y)))
            w = max(0.0, min(1.0 - x1, float(bw)))
            h = max(0.0, min(1.0 - y1, float(bh)))

            if w < 0.004 or h < 0.004:
                continue

            cx = x1 + w / 2.0
            cy = y1 + h / 2.0
            lines.append(f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}")
            total_boxes += 1

        if not lines:
            continue

        # Downscale if required
        if max(W, H) > max_side:
            scale = max_side / max(W, H)
            img = img.resize((round(W * scale), round(H * scale)), Image.Resampling.LANCZOS)

        stem = f"{prefix}_{src_path.stem}"
        img.save(img_out_dir / f"{stem}.jpg", quality=jpeg_quality)
        (lbl_out_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n")
        num_images += 1

    return num_images, total_boxes


def update_dataset_yaml_and_stats(data_dir: Path, jewellery_stats: dict):
    """Update dataset.yaml and stats.json to reflect the 4th class (jewellery)."""
    yaml_path = data_dir / "dataset.yaml"
    class_names = ["shoes", "accessories", "clothing", "jewellery"]
    yaml_content = (
        f"path: {data_dir.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(class_names))
    )
    yaml_path.write_text(yaml_content)
    print(f"[config] Updated {yaml_path} with 4 classes: {class_names}")

    stats_path = data_dir / "stats.json"
    if stats_path.exists():
        try:
            stats = json.loads(stats_path.read_text())
        except Exception:
            stats = {}
    else:
        stats = {}

    if "train" in stats:
        stats["train"]["images"] = stats["train"].get("images", 0) + jewellery_stats["train_images"]
        stats["train"]["jewellery"] = jewellery_stats["train_boxes"]
    if "val" in stats:
        stats["val"]["images"] = stats["val"].get("images", 0) + jewellery_stats["val_images"]
        stats["val"]["jewellery"] = jewellery_stats["val_boxes"]

    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"[config] Updated {stats_path}")

    # Remove any existing .cache files
    for c in data_dir.rglob("*.cache"):
        c.unlink()
        print(f"[config] Removed stale cache: {c}")


def main():
    parser = argparse.ArgumentParser(description="Download Open Images V7 Jewellery dataset via FiftyOne")
    parser.add_argument("--data-dir", default="fashion_yolo", help="Destination YOLO dataset directory")
    parser.add_argument("--n-train", type=int, default=1000, help="Number of training images (default: 1000)")
    parser.add_argument("--n-val", type=int, default=100, help="Number of validation images (default: 100)")
    parser.add_argument("--max-side", type=int, default=1024, help="Maximum image side dimension")
    parser.add_argument("--jpeg-quality", type=int, default=95)
    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    
    # 1. Check classes
    check_available_classes()

    # 2. Download & convert
    stats = download_and_export_jewellery(
        data_dir=data_dir,
        n_train=args.n_train,
        n_val=args.n_val,
        jewellery_class_id=3,
        max_side=args.max_side,
        jpeg_quality=args.jpeg_quality,
    )

    # 3. Update dataset configuration
    update_dataset_yaml_and_stats(data_dir, stats)
    print("\n[done] Jewellery dataset download and integration complete!")


if __name__ == "__main__":
    main()
