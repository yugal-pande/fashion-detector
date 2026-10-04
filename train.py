#!/usr/bin/env python3
"""
train.py - Fashion Item Detector (YOLO11s fine-tuned on Fashionpedia)

Detects four classes, one box per visible item:
    0 shoes        - shoes, sandals, heels, other footwear
    1 accessories  - bags, belts, sunglasses/glasses, hats, watches, gloves, ties, scarves, ...
    2 clothing     - every garment (tops, trousers, dresses, sarees, lehengas, dupattas, ...)
                     under ONE label, no clothing sub-categories
    3 jewellery    - necklaces, earrings, and other jewellery items (via Open Images V7)

Pipeline
    1. Download Fashionpedia from the Hugging Face Hub (detection-datasets/fashionpedia)
    2. Remap its 46 categories to the 3 classes above (garment-part categories such as
       sleeve / collar / zipper / sequin are dropped), subsample to ~10k images
    3. Write a YOLO-format dataset + dataset.yaml
    4. Fine-tune YOLO11s from COCO weights (optional 10-layer freeze for the first N epochs)
    5. Log per-class mAP50 on val every epoch (CSV + TensorBoard and/or W&B), keep best.pt

Install:
    pip install -U ultralytics datasets pillow tqdm tensorboard wandb

Usage:
    python train.py                                   # defaults (10 epochs, batch 16, TensorBoard)
    python train.py --logger wandb --epochs 80 --batch 32
    python train.py --freeze-epochs 0                 # no layer freezing
    python train.py --resume runs/fashion/yolo11s_fashion/weights/last.pt
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from tqdm import tqdm

# --------------------------------------------------------------------------------------
# Class definitions / remapping
# --------------------------------------------------------------------------------------
CLASS_NAMES = ["shoes", "accessories", "clothing", "jewellery"]

# Fashionpedia's 46 categories, in dataset order (used as a fallback if the HF
# ClassLabel names cannot be read from the dataset features).
FASHIONPEDIA_CATEGORIES = [
    "shirt, blouse", "top, t-shirt, sweatshirt", "sweater", "cardigan", "jacket", "vest",
    "pants", "shorts", "skirt", "coat", "dress", "jumpsuit", "cape", "glasses", "hat",
    "headband, head covering, hair accessory", "tie", "glove", "watch", "belt", "leg warmer",
    "tights, stockings", "sock", "shoe", "bag, wallet", "scarf", "umbrella", "hood", "collar",
    "lapel", "epaulette", "sleeve", "pocket", "neckline", "buckle", "zipper", "applique",
    "bead", "bow", "flower", "fringe", "ribbon", "rivet", "ruffle", "sequin", "tassel",
]

# Edit this table to move a category between classes.
# Anything NOT listed here is dropped: hood, collar, lapel, epaulette, sleeve, pocket,
# neckline, buckle, zipper, applique, bead, bow, flower, fringe, ribbon, rivet, ruffle,
# sequin, tassel (garment parts / embellishments, not standalone items).
FASHIONPEDIA_TO_TARGET = {
    # 0: shoes
    "shoe": 0,
    # 1: accessories
    "glasses": 1,
    "hat": 1,
    "headband, head covering, hair accessory": 1,
    "tie": 1,
    "glove": 1,
    "watch": 1,
    "belt": 1,
    "bag, wallet": 1,
    "scarf": 1,
    "umbrella": 1,
    # 2: clothing (all garments share one label)
    "shirt, blouse": 2,
    "top, t-shirt, sweatshirt": 2,
    "sweater": 2,
    "cardigan": 2,
    "jacket": 2,
    "vest": 2,
    "pants": 2,
    "shorts": 2,
    "skirt": 2,
    "coat": 2,
    "dress": 2,
    "jumpsuit": 2,
    "cape": 2,
    "leg warmer": 2,
    "tights, stockings": 2,
    "sock": 2,
}

MIN_BOX_FRAC = 0.004  # drop boxes thinner than 0.4% of the image side (~4 px at 1024)


# --------------------------------------------------------------------------------------
# Dataset preparation
# --------------------------------------------------------------------------------------
def get_category_names(ds) -> list[str] | None:
    """Read the ClassLabel names of objects.category from a HF dataset, if available."""
    try:
        feat = ds.features["objects"]
        cat = feat["category"] if isinstance(feat, dict) else feat.feature["category"]
        cat = getattr(cat, "feature", cat)
        return list(cat.names)
    except Exception:
        return None


def build_id_map(ds) -> dict[int, int]:
    names = get_category_names(ds) or FASHIONPEDIA_CATEGORIES
    missing = set(FASHIONPEDIA_TO_TARGET) - set(names)
    if missing:
        raise ValueError(f"Category names not found in dataset: {sorted(missing)}")
    return {i: FASHIONPEDIA_TO_TARGET[n] for i, n in enumerate(names) if n in FASHIONPEDIA_TO_TARGET}


def to_xyxy(box, fmt: str):
    if fmt == "xyxy":
        return [float(v) for v in box]
    x, y, w, h = (float(v) for v in box)
    return [x, y, x + w, y + h]


def check_bbox_format(ds, indices, fmt: str, n: int = 300) -> None:
    """Cheap sanity check that the assumed bbox format matches the data."""
    try:
        meta = ds.select_columns(["width", "height", "objects"])
    except Exception:
        return
    bad = total = 0
    for i in indices[:n]:
        row = meta[int(i)]
        W, H = row["width"], row["height"]
        for box in row["objects"]["bbox"]:
            x1, y1, x2, y2 = to_xyxy(box, fmt)
            total += 1
            if x2 <= x1 or y2 <= y1 or x2 > W * 1.01 or y2 > H * 1.01:
                bad += 1
    if total and bad / total > 0.05:
        print(f"[WARN] {bad}/{total} boxes look invalid under bbox format '{fmt}'. "
              f"Try --bbox-format {'xywh' if fmt == 'xyxy' else 'xyxy'}.")
    else:
        print(f"[OK] bbox format '{fmt}' looks consistent ({bad}/{total} suspicious boxes).")


def convert_one(ds, i, split, img_dir, lbl_dir, id_map, fmt, max_side, jpeg_quality):
    """Convert one HF row to a YOLO image + label file. Returns per-class counts or None."""
    row = ds[int(i)]
    img = row["image"].convert("RGB")
    W, H = img.size
    objs = row["objects"]

    lines, counts = [], [0] * len(CLASS_NAMES)
    for cat, box in zip(objs["category"], objs["bbox"]):
        cls = id_map.get(int(cat))
        if cls is None:
            continue
        x1, y1, x2, y2 = to_xyxy(box, fmt)
        x1, x2 = max(0.0, min(x1, W)), max(0.0, min(x2, W))
        y1, y2 = max(0.0, min(y1, H)), max(0.0, min(y2, H))
        bw, bh = x2 - x1, y2 - y1
        if bw / W < MIN_BOX_FRAC or bh / H < MIN_BOX_FRAC:
            continue
        lines.append(f"{cls} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} "
                     f"{bw / W:.6f} {bh / H:.6f}")
        counts[cls] += 1
    if not lines:
        return None

    if max(W, H) > max_side:  # normalized labels are unaffected by resizing
        s = max_side / max(W, H)
        img = img.resize((round(W * s), round(H * s)))
    stem = f"{split}_{row.get('image_id', i)}"
    img.save(img_dir / f"{stem}.jpg", quality=jpeg_quality)
    (lbl_dir / f"{stem}.txt").write_text("\n".join(lines) + "\n")
    return counts


def prepare_dataset(args) -> Path:
    out = Path(args.data_dir).resolve()
    yaml_path = out / "dataset.yaml"
    if yaml_path.exists() and not args.force_prepare:
        print(f"[data] Reusing prepared dataset at {out} (use --force-prepare to rebuild)")
        return yaml_path

    import os

    hf_cache = str(args.hf_cache or Path(__file__).resolve().parent / ".hf_cache")
    os.environ.setdefault("HF_HOME", hf_cache)

    from datasets import load_dataset

    print(f"[data] Loading detection-datasets/fashionpedia from the Hugging Face Hub (cache: {hf_cache}) ...")
    raw = load_dataset("detection-datasets/fashionpedia", cache_dir=hf_cache)
    split_src = {
        "train": "train",
        "val": next(s for s in ("val", "validation", "test") if s in raw),
    }
    id_map = build_id_map(raw["train"])
    print(f"[data] Category remap: {len(id_map)} of {len(get_category_names(raw['train']) or FASHIONPEDIA_CATEGORIES)} "
          f"source categories kept -> {CLASS_NAMES}")

    if out.exists():
        shutil.rmtree(out)
    rng = random.Random(args.seed)
    totals = {}

    for split, src in split_src.items():
        ds = raw[src]
        n_req = args.n_train if split == "train" else args.n_val
        n = len(ds) if n_req <= 0 else min(n_req, len(ds))
        indices = sorted(rng.sample(range(len(ds)), n))

        if split == "train":
            check_bbox_format(ds, indices, args.bbox_format)

        img_dir, lbl_dir = out / "images" / split, out / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        class_counts, kept = np.zeros(len(CLASS_NAMES), dtype=int), 0
        prep_workers = max(1, args.workers) if args.workers > 0 else 4
        with ThreadPoolExecutor(max_workers=prep_workers) as pool:
            futures = pool.map(
                lambda i: convert_one(ds, i, split, img_dir, lbl_dir, id_map,
                                      args.bbox_format, args.max_side, args.jpeg_quality),
                indices,
            )
            for res in tqdm(futures, total=len(indices), desc=f"[data] {split}"):
                if res is not None:
                    kept += 1
                    class_counts += np.array(res)
        totals[split] = {"images": kept, **{CLASS_NAMES[c]: int(class_counts[c]) for c in range(len(CLASS_NAMES))}}
        print(f"[data] {split}: {kept} images, boxes per class = "
              f"{dict(zip(CLASS_NAMES, class_counts.tolist()))}")

    if getattr(args, "jewellery_train", 0) > 0:
        try:
            from download_jewellery import download_and_export_jewellery
            j_stats = download_and_export_jewellery(
                data_dir=out,
                n_train=args.jewellery_train,
                n_val=args.jewellery_val,
                jewellery_class_id=3,
                max_side=args.max_side,
                jpeg_quality=args.jpeg_quality,
            )
            for split_name, key_prefix in [("train", "train"), ("val", "val")]:
                if split_name in totals:
                    totals[split_name]["images"] += j_stats[f"{key_prefix}_images"]
                    totals[split_name]["jewellery"] = j_stats[f"{key_prefix}_boxes"]
        except Exception as e:
            print(f"[WARN] Failed integrating Open Images V7 jewellery: {e}")

    yaml_path.write_text(
        f"path: {out}\ntrain: images/train\nval: images/val\nnames:\n"
        + "".join(f"  {i}: {n}\n" for i, n in enumerate(CLASS_NAMES))
    )
    (out / "stats.json").write_text(json.dumps(totals, indent=2))
    for cache in out.rglob("*.cache"):
        cache.unlink()
    return yaml_path


# --------------------------------------------------------------------------------------
# Windows / CUDA workarounds
# --------------------------------------------------------------------------------------
def configure_workers(n: int) -> int:
    """Resolve DataLoader worker count.

    On Windows, PyTorch DataLoader workers share tensors via memory-mapped files
    (CreateFileMappingW).  When the system page file is not large enough the
    mapping fails with error 1455 (ERROR_COMMITMENT_LIMIT) and the workers
    crash ('Couldn't open shared file mapping … error code: <1455>').

    n = -1 (auto) -> 0 on Windows (single-process loading), 4 elsewhere.
    """
    import sys

    if n < 0:  # auto
        n = 0 if sys.platform.startswith("win") else 4
    print(f"[env] DataLoader workers: {n} (--workers {n})")
    return n


def configure_pin_memory(mode: str) -> bool:
    """Control DataLoader pin_memory (page-locked host memory).

    The DataLoader pin thread calls cuMemHostAlloc; on Windows boxes that are low on
    lockable RAM the driver fails with 'Failed to allocate physical memory' /
    CUDA_ERROR_ALREADY_MAPPED ('resource already mapped') and kills the run.

    mode = auto -> off on Windows, on elsewhere. 'off' is enforced by wrapping
    torch.utils.data.DataLoader.__init__, which covers every loader Ultralytics builds.
    """
    import sys

    import torch.utils.data

    if mode == "auto":
        mode = "off" if sys.platform.startswith("win") else "on"
    enabled = mode == "on"
    print(f"[env] DataLoader pin_memory: {'on' if enabled else 'off'} (--pin-memory {mode})")
    if enabled:
        return False

    orig_init = torch.utils.data.DataLoader.__init__

    def _init(self, *args, **kwargs):
        if len(args) > 7:  # pin_memory passed positionally (7th arg after dataset)
            args = (*args[:7], False, *args[8:])
        else:
            kwargs["pin_memory"] = False
        orig_init(self, *args, **kwargs)

    torch.utils.data.DataLoader.__init__ = _init
    return False


# --------------------------------------------------------------------------------------
# Callbacks / training tweaks
# --------------------------------------------------------------------------------------
def patch_fitness_to_map50() -> None:
    """Ultralytics picks best.pt by 0.1*mAP50 + 0.9*mAP50-95. Re-weight it towards mAP50
    (IoU >= 0.5 matching) so the best checkpoint and early stopping follow mAP50."""
    from ultralytics.utils import metrics as M

    def fitness(self):
        w = np.array([0.0, 0.0, 0.9, 0.1])  # [P, R, mAP50, mAP50-95]
        return float((np.nan_to_num(np.array(self.mean_results())) * w).sum())

    M.Metric.fitness = fitness


def _core_model(trainer):
    m = trainer.model
    return m.module if hasattr(m, "module") else m


def _set_backbone_frozen(trainer, n_layers: int, frozen: bool) -> int:
    prefixes = tuple(f"model.{i}." for i in range(n_layers))
    n = 0
    for name, p in _core_model(trainer).named_parameters():
        if name.startswith(prefixes):
            p.requires_grad = not frozen
            n += 1
    return n


def make_freeze_callbacks(n_layers: int, n_epochs: int):
    """Freeze the first `n_layers` layers for the first `n_epochs` epochs, then unfreeze."""

    def on_train_start(trainer):
        if getattr(trainer, "start_epoch", 0) < n_epochs:
            n = _set_backbone_frozen(trainer, n_layers, True)
            print(f"[freeze] Froze {n} param tensors in layers 0-{n_layers - 1} "
                  f"for the first {n_epochs} epochs")

    def on_train_epoch_start(trainer):
        if trainer.epoch == n_epochs:
            _set_backbone_frozen(trainer, n_layers, False)
            print(f"[freeze] Epoch {trainer.epoch + 1}: unfroze layers 0-{n_layers - 1}")

    return on_train_start, on_train_epoch_start


def make_per_class_logger(use_tb: bool, use_wandb: bool):
    """Log per-class mAP50 (IoU=0.5) on val after every epoch: CSV + TensorBoard + W&B."""
    state = {"tb": None}

    def on_fit_epoch_end(trainer):
        box = getattr(getattr(trainer.validator, "metrics", None), "box", None)
        if box is None or len(box.ap_class_index) == 0:
            return
        epoch = trainer.epoch + 1
        per_class = {n: float("nan") for n in CLASS_NAMES}
        for idx, ap50 in zip(box.ap_class_index, box.ap50):
            per_class[CLASS_NAMES[int(idx)]] = float(ap50)
        per_class["all"] = float(box.map50)

        print(f"[val] epoch {epoch}  mAP50: " + "  ".join(f"{k}={v:.4f}" for k, v in per_class.items()))

        csv_path = Path(trainer.save_dir) / "per_class_map50.csv"
        write_header = not csv_path.exists()
        with open(csv_path, "a", newline="") as f:
            w = csv.writer(f)
            if write_header:
                w.writerow(["epoch", *per_class.keys()])
            w.writerow([epoch, *[f"{v:.6f}" for v in per_class.values()]])

        if use_tb:
            try:
                if state["tb"] is None:
                    from torch.utils.tensorboard import SummaryWriter
                    state["tb"] = SummaryWriter(str(Path(trainer.save_dir) / "per_class"))
                for k, v in per_class.items():
                    state["tb"].add_scalar(f"val_per_class/mAP50_{k}", v, epoch)
            except Exception as e:  # tensorboard not installed, etc.
                print(f"[WARN] TensorBoard per-class logging disabled: {e}")
                state["tb"] = False
                
        if use_wandb:
            try:
                import wandb
                if wandb.run is not None:
                    wandb.log({f"val_per_class/mAP50_{k}": v for k, v in per_class.items()}, step=epoch)
            except Exception as e:
                print(f"[WARN] W&B per-class logging failed: {e}")

    def on_train_end(trainer):
        if state["tb"]:
            state["tb"].close()

    return on_fit_epoch_end, on_train_end


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser(description="Train YOLO11s fashion detector (shoes / accessories / clothing)")
    # data
    p.add_argument("--data-dir", default="fashion_yolo", help="where the YOLO-format dataset is written")
    p.add_argument("--hf-cache", default=str(Path(__file__).resolve().parent / ".hf_cache"),
                   help="HF datasets cache dir (default: .hf_cache inside the project folder)")
    p.add_argument("--n-train", type=int, default=10000, help="train images to sample (<=0: all)")
    p.add_argument("--n-val", type=int, default=0, help="val images to sample (<=0: full val split, ~1.1k)")
    p.add_argument("--bbox-format", choices=["xyxy", "xywh"], default="xyxy",
                   help="box format of the HF dataset (detection-datasets/fashionpedia uses xyxy)")
    p.add_argument("--max-side", type=int, default=1024, help="downscale saved images to this longest side")
    p.add_argument("--jpeg-quality", type=int, default=95)
    p.add_argument("--force-prepare", action="store_true")
    p.add_argument("--prepare-only", action="store_true")
    p.add_argument("--jewellery-train", type=int, default=1000,
                   help="Open Images V7 jewellery training samples to download if preparing dataset (default: 1000)")
    p.add_argument("--jewellery-val", type=int, default=100,
                   help="Open Images V7 jewellery val samples to download if preparing dataset (default: 100)")
    p.add_argument("--model",
                   default="fashion_best.pt" if Path("fashion_best.pt").exists() else "yolo11s.pt",
                   help="Base model weights (default: fashion_best.pt for fine-tuning if available, else yolo11s.pt)")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--epochs", type=int, default=10, help="10-80 recommended")
    p.add_argument("--batch", type=int, default=16, help="16-32 (use -1 for AutoBatch)")
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--lr0", type=float, default=0.01)
    p.add_argument("--lrf", type=float, default=0.01, help="final LR = lr0 * lrf (cosine schedule)")
    p.add_argument("--optimizer", default="SGD")
    p.add_argument("--close-mosaic", type=int, default=10, help="disable mosaic for the last N epochs")
    p.add_argument("--freeze-layers", type=int, default=10)
    p.add_argument("--freeze-epochs", type=int, default=3,
                   help="epochs to keep the first --freeze-layers layers frozen (0 = never freeze)")
    p.add_argument("--nms-iou", type=float, default=0.7, help="NMS IoU threshold used in validation")
    p.add_argument("--select-by", choices=["map50", "default"], default="map50",
                   help="best.pt criterion: mAP50-weighted fitness, or Ultralytics default (mAP50-95-weighted)")
    p.add_argument("--workers", type=int, default=-1,
                   help="dataloader workers; -1 = auto (0 on Windows to avoid shared-memory"
                        " crashes, 4 elsewhere). Keep low on 16 GB RAM boxes")
    p.add_argument("--pin-memory", choices=["auto", "on", "off"], default="auto",
                   help="page-locked host memory for the train loader "
                        "(auto = off on Windows, where cuMemHostAlloc can fail with "
                        "'resource already mapped' / CUDA_ERROR_ALREADY_MAPPED)")
    p.add_argument("--device", default=None, help="e.g. 0, 0,1, cpu (default: GPU 0 if available)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cache", default="False", help="False | disk | ram")
    # logging / output
    p.add_argument("--logger", choices=["tensorboard", "wandb", "both", "none"], default="tensorboard")
    p.add_argument("--project", default="runs/fashion")
    p.add_argument("--name", default="yolo11s_fashion")
    p.add_argument("--export-best", default="fashion_best.pt", help="copy of the best checkpoint")
    p.add_argument("--resume", default=None, help="path to last.pt to resume an interrupted run")
    return p.parse_args()


def main():
    args = parse_args()

    yaml_path = prepare_dataset(args)
    if args.prepare_only:
        print(f"[data] Done. dataset.yaml -> {yaml_path}")
        return

    import torch
    from ultralytics import YOLO, settings

    workers = configure_workers(args.workers)
    configure_pin_memory(args.pin_memory)

    use_tb = args.logger in ("tensorboard", "both")
    use_wandb = args.logger in ("wandb", "both")
    if use_wandb:
        try:
            import wandb  # noqa: F401
        except ImportError:
            print("[WARN] wandb not installed (pip install wandb); continuing without it")
            use_wandb = False
    settings.update({"tensorboard": use_tb, "wandb": use_wandb})  # built-in loss/metric logging

    if args.select_by == "map50":
        patch_fitness_to_map50()

    device = args.device if args.device is not None else ("0" if torch.cuda.is_available() else "cpu")
    cache = {"false": False, "true": "ram"}.get(str(args.cache).lower(), args.cache)

    model = YOLO(args.resume if args.resume else args.model)

    # Optional temporary backbone freeze + per-class mAP50 logging
    if args.freeze_epochs > 0:
        on_start, on_epoch_start = make_freeze_callbacks(args.freeze_layers, args.freeze_epochs)
        model.add_callback("on_train_start", on_start)
        model.add_callback("on_train_epoch_start", on_epoch_start)
    per_class_cb, train_end_cb = make_per_class_logger(use_tb, use_wandb)
    model.add_callback("on_fit_epoch_end", per_class_cb)
    model.add_callback("on_train_end", train_end_cb)

    if args.resume:
        model.train(resume=True)
    else:
        model.train(
            data=str(yaml_path),
            imgsz=args.imgsz,
            epochs=args.epochs,
            batch=args.batch,
            patience=args.patience,
            device=device,
            workers=workers,
            seed=args.seed,
            cache=cache,
            pretrained=True,
            amp=True,
            project=args.project,
            name=args.name,
            exist_ok=True,
            # optimisation: cosine LR decay
            optimizer=args.optimizer,
            lr0=args.lr0,
            lrf=args.lrf,
            cos_lr=True,
            warmup_epochs=3,
            # validation: mAP@0.5 is the tracked metric; NMS IoU is separate (see --nms-iou)
            val=True,
            conf=0.001,
            iou=args.nms_iou,
            plots=True,
            save=True,
            # augmentation
            mosaic=1.0,
            close_mosaic=args.close_mosaic,  # mosaic off for the last N epochs
            hsv_h=0.015, hsv_s=0.7, hsv_v=0.4,
            scale=0.5,
            translate=0.1,
            fliplr=0.5,
            flipud=0.0,                      # no vertical flips
            degrees=0.0, shear=0.0, perspective=0.0,
            mixup=0.0, copy_paste=0.0,
        )

    # ---- Final evaluation of the best checkpoint --------------------------------------
    save_dir = Path(model.trainer.save_dir)
    best = Path(getattr(model.trainer, "best", save_dir / "weights" / "best.pt"))
    if not best.exists():
        best = save_dir / "weights" / "last.pt"
    shutil.copy(best, args.export_best)
    print(f"\n[done] Best checkpoint: {best}  (copied to {args.export_best})")

    metrics = YOLO(str(best)).val(
        data=str(yaml_path), imgsz=args.imgsz, split="val", conf=0.001, iou=args.nms_iou,
        batch=args.batch if args.batch > 0 else 16, device=device, workers=workers,
        plots=True, project=args.project, name=f"{args.name}_final_val", exist_ok=True,
    )
    final = {CLASS_NAMES[int(c)]: round(float(a), 4)
             for c, a in zip(metrics.box.ap_class_index, metrics.box.ap50)}
    final.update({"mAP50 (all)": round(float(metrics.box.map50), 4),
                  "mAP50-95 (all)": round(float(metrics.box.map), 4)})
    (save_dir / "final_val_metrics.json").write_text(json.dumps(final, indent=2))
    print("[done] Final val metrics (best.pt):")
    for k, v in final.items():
        print(f"   {k:>16}: {v}")


if __name__ == "__main__":
    main()
