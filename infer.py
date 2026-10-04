#!/usr/bin/env python3
"""
infer.py — Fashion-detector inference script.

For every input image:
  1. Runs YOLO detection (conf=0.30, class-aware NMS @ IoU 0.55, agnostic_nms off).
  2. Draws annotated bounding boxes (fixed colour per class) → outputs/annotated/<image>.jpg
  3. Saves padded crops                                      → outputs/crops/<image>/<idx>_<class>_<conf>.jpg
  4. Writes per-image JSON                                   → outputs/json/<image>.json

Coordinate convention (in the JSON):
    Absolute pixels, [x_min, y_min, x_max, y_max], origin = top-left.

Usage:
    python infer.py image1.jpg image2.png ...
    python infer.py path/to/folder/
    python infer.py --model runs/fashion/yolo11s_fashion/weights/best.pt --conf 0.30 image.jpg

Install:
    pip install ultralytics opencv-python-headless
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

# -------------------------------------------------------------------------------------
# Constants
# -------------------------------------------------------------------------------------
CLASS_NAMES = ["shoes", "accessories", "clothing", "jewellery"]

# Fixed BGR colours per class (vivid, easily distinguishable on most backgrounds).
#   shoes       → coral-orange
#   accessories → teal
#   clothing    → purple
#   jewellery   → gold
CLASS_COLOURS_BGR: dict[int, tuple[int, int, int]] = {
    0: (71, 99, 255),    # shoes       – #FF6347 (tomato)
    1: (208, 224, 64),   # accessories – #40E0D0 (turquoise)
    2: (219, 112, 147),  # clothing    – #9370DB (medium purple)
    3: (0, 215, 255),    # jewellery   – #FFD700 (gold)
}

# Fallback for any unseen class id
DEFAULT_COLOUR_BGR = (200, 200, 200)

# Crop padding fraction (applied to each side).  5-10 % → we use 7 % as a sweet spot.
CROP_PAD_FRAC = 0.07

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif", ".webp"}

# -------------------------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------------------------

def gather_images(paths: list[str]) -> list[Path]:
    """Expand a mix of files and directories into a flat, sorted list of image paths."""
    images: list[Path] = []
    for p in paths:
        pp = Path(p)
        if pp.is_dir():
            images.extend(
                sorted(f for f in pp.iterdir() if f.suffix.lower() in IMAGE_EXTENSIONS)
            )
        elif pp.is_file() and pp.suffix.lower() in IMAGE_EXTENSIONS:
            images.append(pp)
        else:
            print(f"[WARN] Skipping '{p}' (not an image or directory)")
    return images


def clamp(val: float, lo: float, hi: float) -> float:
    return max(lo, min(val, hi))


def padded_crop(
    img: np.ndarray, x1: float, y1: float, x2: float, y2: float, pad_frac: float
) -> np.ndarray:
    """Extract a crop with `pad_frac` padding on every side, clamped to image bounds."""
    h, w = img.shape[:2]
    bw, bh = x2 - x1, y2 - y1
    px, py = bw * pad_frac, bh * pad_frac
    cx1 = int(clamp(x1 - px, 0, w))
    cy1 = int(clamp(y1 - py, 0, h))
    cx2 = int(clamp(x2 + px, 0, w))
    cy2 = int(clamp(y2 + py, 0, h))
    return img[cy1:cy2, cx1:cx2]


def draw_boxes(
    img: np.ndarray,
    boxes: np.ndarray,
    class_ids: np.ndarray,
    confs: np.ndarray,
    class_names: list[str],
    thickness: int = 2,
    font_scale: float = 0.55,
) -> np.ndarray:
    """Draw bounding boxes with 'class: score' labels, one fixed colour per class."""
    canvas = img.copy()
    for box, cls_id, conf in zip(boxes, class_ids, confs):
        x1, y1, x2, y2 = map(int, box)
        colour = CLASS_COLOURS_BGR.get(int(cls_id), DEFAULT_COLOUR_BGR)
        label = f"{class_names[int(cls_id)]}: {conf:.2f}"

        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, thickness)

        # Label background
        (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)
        lbl_y1 = max(y1 - th - baseline - 4, 0)
        lbl_y2 = max(y1, th + baseline + 4)
        cv2.rectangle(canvas, (x1, lbl_y1), (x1 + tw + 4, lbl_y2), colour, cv2.FILLED)
        cv2.putText(
            canvas, label, (x1 + 2, lbl_y2 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA,
        )
    return canvas


# -------------------------------------------------------------------------------------
# Core pipeline
# -------------------------------------------------------------------------------------

def process_image(
    model: YOLO,
    img_path: Path,
    output_root: Path,
    conf: float,
    iou: float,
    imgsz: int,
    class_names: list[str],
    crop_pad: float,
    save_quality: int,
) -> dict:
    """Run detection on one image and produce annotated image, crops, and JSON dict."""

    # --- Predict ---------------------------------------------------------------
    results = model.predict(
        source=str(img_path),
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        agnostic_nms=False,   # class-aware NMS: clothing items legitimately overlap
        max_det=100,
        verbose=False,
    )
    result = results[0]
    img_bgr = cv2.imread(str(img_path))
    if img_bgr is None:
        print(f"[ERROR] Cannot read image: {img_path}")
        return {}
    img_h, img_w = img_bgr.shape[:2]
    stem = img_path.stem

    # Unpack detections
    boxes_xyxy = result.boxes.xyxy.cpu().numpy()       # (N, 4) float32
    confs      = result.boxes.conf.cpu().numpy()        # (N,)
    cls_ids    = result.boxes.cls.cpu().numpy().astype(int)  # (N,)
    n_det      = len(boxes_xyxy)

    # --- Annotated image -------------------------------------------------------
    ann_dir = output_root / "annotated"
    ann_dir.mkdir(parents=True, exist_ok=True)
    annotated = draw_boxes(img_bgr, boxes_xyxy, cls_ids, confs, class_names)
    ann_path = ann_dir / f"{stem}.jpg"
    cv2.imwrite(str(ann_path), annotated, [cv2.IMWRITE_JPEG_QUALITY, save_quality])

    # --- Crops ------------------------------------------------------------------
    crop_dir = output_root / "crops" / stem
    crop_dir.mkdir(parents=True, exist_ok=True)
    detections_list: list[dict] = []

    for idx in range(n_det):
        x1, y1, x2, y2 = boxes_xyxy[idx]
        cls_id = int(cls_ids[idx])
        score  = float(confs[idx])
        cat    = class_names[cls_id]

        crop = padded_crop(img_bgr, x1, y1, x2, y2, crop_pad)
        crop_name = f"{idx}_{cat}_{score:.2f}.jpg"
        crop_path_rel = f"crops/{stem}/{crop_name}"
        cv2.imwrite(
            str(crop_dir / crop_name), crop,
            [cv2.IMWRITE_JPEG_QUALITY, save_quality],
        )

        detections_list.append({
            "id": idx,
            "category": cat,
            "confidence": round(score, 4),
            "bbox_xyxy": [round(float(v), 1) for v in (x1, y1, x2, y2)],
            "crop_path": crop_path_rel,
        })

    # --- JSON -------------------------------------------------------------------
    json_dir = output_root / "json"
    json_dir.mkdir(parents=True, exist_ok=True)

    doc = {
        "image": img_path.name,
        "width": img_w,
        "height": img_h,
        "coordinate_convention": {
            "format": "absolute pixels, [x_min, y_min, x_max, y_max]",
            "origin": "top-left",
        },
        "detections": detections_list,
    }
    json_path = json_dir / f"{stem}.json"
    json_path.write_text(json.dumps(doc, indent=2), encoding="utf-8")

    return doc


# -------------------------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run fashion-detector inference on images and save annotated "
                    "images, crops, and structured JSON.",
    )
    p.add_argument(
        "inputs", nargs="+",
        help="Image files and/or directories to process.",
    )
    p.add_argument(
        "--model", default="fashion_best.pt",
        help="Path to the YOLO .pt weights (default: fashion_best.pt).",
    )
    p.add_argument(
        "--output", "-o", default="outputs",
        help="Root directory for annotated/, crops/, and json/ (default: outputs).",
    )
    p.add_argument(
        "--conf", type=float, default=0.30,
        help="Confidence threshold (default: 0.30; tune on val).",
    )
    p.add_argument(
        "--iou", type=float, default=0.55,
        help="NMS IoU threshold (default: 0.55; recommended range 0.50-0.60).",
    )
    p.add_argument(
        "--imgsz", type=int, default=640,
        help="Inference image size (default: 640).",
    )
    p.add_argument(
        "--crop-pad", type=float, default=CROP_PAD_FRAC,
        help=f"Fractional padding added to each crop side (default: {CROP_PAD_FRAC}).",
    )
    p.add_argument(
        "--jpeg-quality", type=int, default=95,
        help="JPEG save quality for annotated images and crops (default: 95).",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # Resolve images
    images = gather_images(args.inputs)
    if not images:
        print("[ERROR] No valid images found. Pass image files or directories.")
        sys.exit(1)
    print(f"[infer] {len(images)} image(s) to process")

    # Load model once
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[ERROR] Model not found: {model_path}")
        sys.exit(1)
    model = YOLO(str(model_path))

    # Resolve class names from the model, fall back to built-in list
    model_names = getattr(model, "names", None)
    if model_names and isinstance(model_names, dict):
        class_names = [model_names[i] for i in sorted(model_names.keys())]
    else:
        class_names = CLASS_NAMES
    print(f"[infer] Classes: {class_names}")
    print(f"[infer] conf={args.conf}, iou={args.iou}, agnostic_nms=False")

    output_root = Path(args.output)

    for img_path in images:
        doc = process_image(
            model=model,
            img_path=img_path,
            output_root=output_root,
            conf=args.conf,
            iou=args.iou,
            imgsz=args.imgsz,
            class_names=class_names,
            crop_pad=args.crop_pad,
            save_quality=args.jpeg_quality,
        )
        n = len(doc.get("detections", []))
        print(f"  ✓ {img_path.name}: {n} detection(s)")

    print(f"\n[done] Results saved to {output_root.resolve()}/")


if __name__ == "__main__":
    main()
