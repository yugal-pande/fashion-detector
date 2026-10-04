#!/usr/bin/env python3
"""
annotate_ground_truth.py — Generate initial ground-truth labels for the test
dataset using the trained model, then allow manual correction.

Workflow:
  1. Runs the model on every test image at a LOW confidence threshold (0.15)
     to maximise recall — we'd rather have false positives to delete than miss
     true objects.
  2. Writes YOLO-format .txt labels into test_dataset/labels/.
  3. Generates a visual review sheet (annotated images) in test_dataset/review/
     so you can visually verify each label.
    4. Launch ground_truth_editor.py to visually correct boxes and class IDs.

Classes:
    0 = shoes
    1 = accessories
    2 = clothing
    3 = jewellery

YOLO label format per line:
    <class_id> <x_center> <y_center> <width> <height>
    (all values normalised to [0, 1] relative to image dimensions)

Usage:
    python test_dataset/annotate_ground_truth.py
    python test_dataset/annotate_ground_truth.py --model fashion_best.pt --conf 0.15
    python test_dataset/ground_truth_editor.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

CLASS_NAMES = ["shoes", "accessories", "clothing", "jewellery"]

# Vivid BGR colours for each class (same palette as infer.py)
COLOURS_BGR = {
    0: (71, 99, 255),    # shoes       – coral-orange
    1: (208, 224, 64),   # accessories – teal
    2: (219, 112, 147),  # clothing    – purple
    3: (0, 215, 255),    # jewellery   – gold
}

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def draw_gt_boxes(img: np.ndarray, labels: list[str], w: int, h: int) -> np.ndarray:
    """Draw YOLO-format labels on the image for visual review."""
    canvas = img.copy()
    for line in labels:
        parts = line.strip().split()
        if len(parts) != 5:
            continue
        cls_id = int(parts[0])
        xc, yc, bw, bh = [float(v) for v in parts[1:]]
        x1 = int((xc - bw / 2) * w)
        y1 = int((yc - bh / 2) * h)
        x2 = int((xc + bw / 2) * w)
        y2 = int((yc + bh / 2) * h)
        colour = COLOURS_BGR.get(cls_id, (200, 200, 200))
        label = f"{CLASS_NAMES[cls_id]} ({cls_id})"
        cv2.rectangle(canvas, (x1, y1), (x2, y2), colour, 2)
        (tw, th_), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(canvas, (x1, max(y1 - th_ - 6, 0)), (x1 + tw + 4, y1), colour, -1)
        cv2.putText(canvas, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def main():
    parser = argparse.ArgumentParser(description="Generate initial GT labels for the test set")
    parser.add_argument("--model", default="fashion_best.pt", help="Model weights")
    parser.add_argument("--conf", type=float, default=0.15,
                        help="Low confidence threshold for high recall")
    parser.add_argument("--iou", type=float, default=0.55, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--test-dir", default="test_dataset",
                        help="Root directory of the test dataset")
    parser.add_argument("--force", action="store_true",
                        help="Overwrite existing labels")
    args = parser.parse_args()

    test_root = Path(args.test_dir).resolve()
    img_dir   = test_root / "images"
    lbl_dir   = test_root / "labels"
    review_dir = test_root / "review"
    lbl_dir.mkdir(parents=True, exist_ok=True)
    review_dir.mkdir(parents=True, exist_ok=True)

    images = sorted(f for f in img_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS)
    if not images:
        print(f"[ERROR] No images found in {img_dir}")
        return

    print(f"Loading model: {args.model}")
    model = YOLO(args.model)
    print(f"Generating labels for {len(images)} images (conf={args.conf}, iou={args.iou})")

    total_boxes = {c: 0 for c in CLASS_NAMES}
    summary = []

    for img_path in images:
        lbl_path = lbl_dir / f"{img_path.stem}.txt"

        # Skip if label already exists (unless --force)
        if lbl_path.exists() and not args.force:
            print(f"  [skip] {img_path.name} — label exists (use --force to overwrite)")
            lines = lbl_path.read_text().strip().split("\n")
            lines = [l for l in lines if l.strip()]
        else:
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                print(f"  [FAIL] Cannot read {img_path.name}; skipping")
                continue

            # Run model
            results = model.predict(
                source=str(img_path),
                conf=args.conf,
                iou=args.iou,
                imgsz=args.imgsz,
                agnostic_nms=False,
                max_det=100,
                verbose=False,
            )
            if not results:
                print(f"  [FAIL] Model returned no result for {img_path.name}; skipping")
                continue
            result = results[0]

            h, w = img_bgr.shape[:2]

            boxes = result.boxes.xyxy.cpu().numpy()
            confs = result.boxes.conf.cpu().numpy()
            cls_ids = result.boxes.cls.cpu().numpy().astype(int)

            lines = []
            for box, cls_id, conf in zip(boxes, cls_ids, confs):
                x1, y1, x2, y2 = box
                xc = (x1 + x2) / 2 / w
                yc = (y1 + y2) / 2 / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                lines.append(f"{cls_id} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

            lbl_path.write_text("\n".join(lines) + "\n" if lines else "")
            print(f"  [ok]   {img_path.name}: {len(lines)} boxes")

        # Count classes
        img_classes = {}
        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 1:
                cls_name = CLASS_NAMES[int(parts[0])]
                total_boxes[cls_name] = total_boxes.get(cls_name, 0) + 1
                img_classes[cls_name] = img_classes.get(cls_name, 0) + 1
        summary.append({"image": img_path.name, "boxes": img_classes})

        # Draw review image
        img_bgr = cv2.imread(str(img_path))
        if img_bgr is not None:
            h, w = img_bgr.shape[:2]
            annotated = draw_gt_boxes(img_bgr, lines, w, h)
            cv2.imwrite(str(review_dir / img_path.name), annotated,
                        [cv2.IMWRITE_JPEG_QUALITY, 90])

    # Summary
    print(f"\n{'='*60}")
    print("Ground-truth box counts per class:")
    for cls, cnt in total_boxes.items():
        status = "✓" if cnt >= 10 else "⚠ (< 10 instances)"
        print(f"  {cls:>12}: {cnt:>4}  {status}")
    print(f"{'='*60}")
    print(f"\nLabels written to: {lbl_dir}")
    print(f"Review images to:  {review_dir}")
    print("\n[NEXT STEP] Run: python test_dataset/ground_truth_editor.py")
    print("Visually correct boxes and classes before running evaluate.py.")

    # Write summary JSON
    (test_root / "annotation_summary.json").write_text(
        json.dumps({"per_image": summary, "total_boxes": total_boxes}, indent=2)
    )


if __name__ == "__main__":
    main()
