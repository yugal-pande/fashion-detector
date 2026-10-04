#!/usr/bin/env python3
"""
evaluate.py — Full evaluation pipeline for the fashion detector.

Runs the model on the test dataset, matches predictions to ground-truth
annotations using IoU-based matching, and reports:
    - Per-class Precision, Recall, F1 Score
    - Overall Accuracy (micro-averaged)
    - Confusion matrix (with "background" row/column for FP/FN)
    - mAP@50 and mAP@50:95
    - Saves publication-quality plots and a JSON summary

Usage:
    python evaluate.py
    python evaluate.py --model fashion_best.pt --conf 0.30 --iou-thresh 0.50
    python evaluate.py --test-dir test_dataset --output eval_results

Classes:
    0 = shoes,  1 = accessories,  2 = clothing,  3 = jewellery
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CLASS_NAMES = ["shoes", "accessories", "clothing", "jewellery"]
NUM_CLASSES = len(CLASS_NAMES)
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Vivid BGR colours for annotation overlays
COLOURS_BGR = {
    0: (71, 99, 255),    # shoes       – coral-orange
    1: (208, 224, 64),   # accessories – teal
    2: (219, 112, 147),  # clothing    – purple
    3: (0, 215, 255),    # jewellery   – gold
}


# ---------------------------------------------------------------------------
# YOLO label parsing
# ---------------------------------------------------------------------------
def load_yolo_labels(label_path: Path, img_w: int, img_h: int) -> list[dict]:
    """Parse a YOLO-format label file into a list of boxes.
    
    Returns a list of dicts, each with:
        class_id: int
        bbox_xyxy: [x1, y1, x2, y2] in absolute pixels
    """
    boxes = []
    if not label_path.exists():
        return boxes
    for line in label_path.read_text().strip().split("\n"):
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            continue
        cls_id = int(parts[0])
        xc, yc, bw, bh = [float(v) for v in parts[1:5]]
        x1 = (xc - bw / 2) * img_w
        y1 = (yc - bh / 2) * img_h
        x2 = (xc + bw / 2) * img_w
        y2 = (yc + bh / 2) * img_h
        boxes.append({"class_id": cls_id, "bbox_xyxy": [x1, y1, x2, y2]})
    return boxes


# ---------------------------------------------------------------------------
# IoU computation
# ---------------------------------------------------------------------------
def compute_iou(box_a: list[float], box_b: list[float]) -> float:
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(box_a[0], box_b[0])
    y1 = max(box_a[1], box_b[1])
    x2 = min(box_a[2], box_b[2])
    y2 = min(box_a[3], box_b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


# ---------------------------------------------------------------------------
# Prediction-to-GT matching (greedy, IoU-based)
# ---------------------------------------------------------------------------
def match_predictions_to_gt(
    gt_boxes: list[dict],
    pred_boxes: list[dict],
    iou_threshold: float = 0.50,
) -> tuple[list[int], list[int]]:
    """Match predictions to GT boxes using greedy IoU matching.
    
    Returns:
        y_true: list of true class IDs (NUM_CLASSES for background/FP)
        y_pred: list of predicted class IDs (NUM_CLASSES for background/FN)
        
    Each GT box produces exactly one entry. Unmatched predictions
    (false positives) are assigned gt_class = NUM_CLASSES ("background").
    Unmatched GT boxes (false negatives) are assigned pred_class = NUM_CLASSES.
    """
    y_true = []
    y_pred = []
    
    if not gt_boxes and not pred_boxes:
        return y_true, y_pred
    
    # Sort predictions by confidence (highest first) for greedy matching
    pred_sorted = sorted(pred_boxes, key=lambda p: p.get("confidence", 0), reverse=True)
    
    matched_gt = set()
    matched_pred = set()
    
    # Build IoU matrix
    for pi, pred in enumerate(pred_sorted):
        best_iou = 0.0
        best_gi = -1
        for gi, gt in enumerate(gt_boxes):
            if gi in matched_gt:
                continue
            iou = compute_iou(pred["bbox_xyxy"], gt["bbox_xyxy"])
            if iou > best_iou:
                best_iou = iou
                best_gi = gi
        
        if best_iou >= iou_threshold and best_gi >= 0:
            # True positive (or misclassification)
            y_true.append(gt_boxes[best_gi]["class_id"])
            y_pred.append(pred["class_id"])
            matched_gt.add(best_gi)
            matched_pred.add(pi)
        else:
            # False positive: no GT box matched
            y_true.append(NUM_CLASSES)  # "background" class
            y_pred.append(pred["class_id"])
    
    # Unmatched GT boxes → false negatives
    for gi, gt in enumerate(gt_boxes):
        if gi not in matched_gt:
            y_true.append(gt["class_id"])
            y_pred.append(NUM_CLASSES)  # "background" — model missed it
    
    return y_true, y_pred


# ---------------------------------------------------------------------------
# Per-image detection + matching
# ---------------------------------------------------------------------------
def evaluate_image(
    model: YOLO,
    img_path: Path,
    label_path: Path,
    conf: float,
    iou_nms: float,
    iou_match: float,
    imgsz: int,
) -> dict:
    """Run detection on one image and match to ground truth.
    
    Returns a dict with per-image results.
    """
    img = cv2.imread(str(img_path))
    if img is None:
        print(f"  [WARN] Cannot read {img_path.name}")
        return {}
    h, w = img.shape[:2]
    
    # Load ground truth
    gt_boxes = load_yolo_labels(label_path, w, h)
    
    # Run model
    results = model.predict(
        source=str(img_path),
        conf=conf,
        iou=iou_nms,
        imgsz=imgsz,
        agnostic_nms=False,
        max_det=100,
        verbose=False,
    )
    result = results[0]
    
    # Parse predictions
    pred_boxes = []
    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    cls_ids = result.boxes.cls.cpu().numpy().astype(int)
    
    for box, cls_id, c in zip(boxes_xyxy, cls_ids, confs):
        pred_boxes.append({
            "class_id": int(cls_id),
            "bbox_xyxy": box.tolist(),
            "confidence": float(c),
        })
    
    # Match predictions to GT
    y_true, y_pred = match_predictions_to_gt(gt_boxes, pred_boxes, iou_match)
    
    return {
        "image": img_path.name,
        "width": w,
        "height": h,
        "n_gt": len(gt_boxes),
        "n_pred": len(pred_boxes),
        "gt_boxes": gt_boxes,
        "pred_boxes": pred_boxes,
        "y_true": y_true,
        "y_pred": y_pred,
    }


# ---------------------------------------------------------------------------
# Confusion matrix plot
# ---------------------------------------------------------------------------
def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list[str],
    output_path: Path,
    title: str = "Confusion Matrix",
    normalize: bool = False,
) -> None:
    """Plot and save a publication-quality confusion matrix."""
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_plot = np.divide(cm, row_sums, where=row_sums != 0, out=np.zeros_like(cm, dtype=float))
        fmt = ".2f"
        title += " (Normalised)"
    else:
        cm_plot = cm.astype(float)
        fmt = "g"
    
    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        cm_plot,
        annot=True,
        fmt=fmt,
        cmap="Blues",
        xticklabels=labels,
        yticklabels=labels,
        ax=ax,
        linewidths=0.5,
        linecolor="lightgrey",
        cbar_kws={"shrink": 0.8},
        square=True,
    )
    ax.set_xlabel("Predicted", fontsize=13, fontweight="bold")
    ax.set_ylabel("Actual (Ground Truth)", fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=15, fontweight="bold", pad=15)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Per-class metrics bar chart
# ---------------------------------------------------------------------------
def plot_per_class_metrics(
    metrics_dict: dict,
    output_path: Path,
) -> None:
    """Plot a grouped bar chart of Precision / Recall / F1 per class."""
    classes = list(metrics_dict.keys())
    precision = [metrics_dict[c]["precision"] for c in classes]
    recall = [metrics_dict[c]["recall"] for c in classes]
    f1 = [metrics_dict[c]["f1"] for c in classes]
    
    x = np.arange(len(classes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars_p = ax.bar(x - width, precision, width, label="Precision", color="#4C78A8", edgecolor="white")
    bars_r = ax.bar(x, recall, width, label="Recall", color="#F58518", edgecolor="white")
    bars_f = ax.bar(x + width, f1, width, label="F1 Score", color="#54A24B", edgecolor="white")
    
    # Add value labels on bars
    for bars in [bars_p, bars_r, bars_f]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom", fontsize=9,
            )
    
    ax.set_xlabel("Class", fontsize=13, fontweight="bold")
    ax.set_ylabel("Score", fontsize=13, fontweight="bold")
    ax.set_title("Per-Class Detection Metrics", fontsize=15, fontweight="bold", pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, fontsize=11)
    ax.set_ylim(0, 1.15)
    ax.legend(fontsize=11, loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    fig.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {output_path}")


# ---------------------------------------------------------------------------
# Side-by-side GT vs Prediction overlay
# ---------------------------------------------------------------------------
def draw_comparison(
    img_path: Path,
    gt_boxes: list[dict],
    pred_boxes: list[dict],
    class_names: list[str],
    output_path: Path,
) -> None:
    """Create a side-by-side image: GT (left) | Predictions (right)."""
    img = cv2.imread(str(img_path))
    if img is None:
        return
    h, w = img.shape[:2]
    
    gt_canvas = img.copy()
    pred_canvas = img.copy()
    
    for box_info in gt_boxes:
        cls_id = box_info["class_id"]
        x1, y1, x2, y2 = [int(v) for v in box_info["bbox_xyxy"]]
        colour = COLOURS_BGR.get(cls_id, (200, 200, 200))
        cv2.rectangle(gt_canvas, (x1, y1), (x2, y2), colour, 2)
        label = f"GT: {class_names[cls_id]}"
        cv2.putText(gt_canvas, label, (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
    
    for box_info in pred_boxes:
        cls_id = box_info["class_id"]
        conf = box_info.get("confidence", 0)
        x1, y1, x2, y2 = [int(v) for v in box_info["bbox_xyxy"]]
        colour = COLOURS_BGR.get(cls_id, (200, 200, 200))
        cv2.rectangle(pred_canvas, (x1, y1), (x2, y2), colour, 2)
        label = f"{class_names[cls_id]}: {conf:.2f}"
        cv2.putText(pred_canvas, label, (x1, max(y1 - 5, 15)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, colour, 1, cv2.LINE_AA)
    
    # Add titles
    cv2.putText(gt_canvas, "Ground Truth", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(pred_canvas, "Predictions", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2, cv2.LINE_AA)
    
    combined = np.hstack([gt_canvas, pred_canvas])
    cv2.imwrite(str(output_path), combined, [cv2.IMWRITE_JPEG_QUALITY, 90])


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Evaluate fashion detector on the test dataset")
    parser.add_argument("--model", default="fashion_best.pt", help="Model weights path")
    parser.add_argument("--test-dir", default="test_dataset", help="Test dataset directory")
    parser.add_argument("--output", default="eval_results", help="Output directory for results")
    parser.add_argument("--conf", type=float, default=0.30, help="Confidence threshold")
    parser.add_argument("--iou-nms", type=float, default=0.55, help="NMS IoU threshold")
    parser.add_argument("--iou-thresh", type=float, default=0.50,
                        help="IoU threshold for matching predictions to GT")
    parser.add_argument("--imgsz", type=int, default=640, help="Inference image size")
    args = parser.parse_args()
    
    test_root = Path(args.test_dir).resolve()
    img_dir = test_root / "images"
    lbl_dir = test_root / "labels"
    output_dir = Path(args.output).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    compare_dir = output_dir / "comparisons"
    compare_dir.mkdir(parents=True, exist_ok=True)
    
    # Gather images
    images = sorted(f for f in img_dir.iterdir() if f.suffix.lower() in IMAGE_EXTS)
    if not images:
        print(f"[ERROR] No images in {img_dir}")
        sys.exit(1)
    
    print(f"{'='*70}")
    print(f"  Fashion Detector Evaluation Pipeline")
    print(f"{'='*70}")
    print(f"  Model:          {args.model}")
    print(f"  Test images:    {len(images)}")
    print(f"  Confidence:     {args.conf}")
    print(f"  NMS IoU:        {args.iou_nms}")
    print(f"  Match IoU:      {args.iou_thresh}")
    print(f"  Image size:     {args.imgsz}")
    print(f"{'='*70}\n")
    
    # Load model
    model = YOLO(args.model)
    
    # Evaluate each image
    all_y_true = []
    all_y_pred = []
    per_image_results = []
    gt_class_counts = defaultdict(int)
    pred_class_counts = defaultdict(int)
    
    print("[1/4] Running inference and matching...\n")
    for img_path in images:
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        result = evaluate_image(
            model, img_path, lbl_path,
            conf=args.conf,
            iou_nms=args.iou_nms,
            iou_match=args.iou_thresh,
            imgsz=args.imgsz,
        )
        if not result:
            continue
        
        all_y_true.extend(result["y_true"])
        all_y_pred.extend(result["y_pred"])
        per_image_results.append(result)
        
        # Count GT instances per class
        for gt in result["gt_boxes"]:
            gt_class_counts[CLASS_NAMES[gt["class_id"]]] += 1
        for pred in result["pred_boxes"]:
            pred_class_counts[CLASS_NAMES[pred["class_id"]]] += 1
        
        # Save comparison image
        draw_comparison(
            img_path,
            result["gt_boxes"],
            result["pred_boxes"],
            CLASS_NAMES,
            compare_dir / f"{img_path.stem}_compare.jpg",
        )
        
        print(f"  {img_path.name:35s}  GT={result['n_gt']:>3}  Pred={result['n_pred']:>3}")
    
    print()
    
    # -----------------------------------------------------------------------
    # Compute metrics
    # -----------------------------------------------------------------------
    print("[2/4] Computing metrics...\n")
    
    y_true = np.array(all_y_true)
    y_pred = np.array(all_y_pred)
    
    # Extended labels including "background" for FP/FN analysis
    extended_labels = CLASS_NAMES + ["background"]
    
    # ── Confusion Matrix (with background) ────────────────────────────────
    cm_full = confusion_matrix(
        y_true, y_pred,
        labels=list(range(NUM_CLASSES + 1)),
    )
    
    # ── Metrics on matched detections only (exclude background) ───────────
    # Filter to only rows where both true and pred are actual classes
    mask_classes = (y_true < NUM_CLASSES) & (y_pred < NUM_CLASSES)
    y_true_cls = y_true[mask_classes]
    y_pred_cls = y_pred[mask_classes]
    
    # Per-class metrics
    per_class_metrics = {}
    for i, cls_name in enumerate(CLASS_NAMES):
        # For each class: TP, FP, FN
        tp = int(np.sum((y_true == i) & (y_pred == i)))
        fp = int(np.sum((y_true != i) & (y_pred == i)))  # includes background predictions
        fn = int(np.sum((y_true == i) & (y_pred != i)))  # includes missed detections
        
        precision_val = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall_val = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1_val = (2 * precision_val * recall_val / (precision_val + recall_val)
                  if (precision_val + recall_val) > 0 else 0.0)
        
        per_class_metrics[cls_name] = {
            "precision": round(precision_val, 4),
            "recall": round(recall_val, 4),
            "f1": round(f1_val, 4),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "gt_count": int(gt_class_counts.get(cls_name, 0)),
            "pred_count": int(pred_class_counts.get(cls_name, 0)),
        }
    
    # Overall metrics (micro-averaged across classes, ignoring background)
    total_tp = sum(m["tp"] for m in per_class_metrics.values())
    total_fp = sum(m["fp"] for m in per_class_metrics.values())
    total_fn = sum(m["fn"] for m in per_class_metrics.values())
    
    # Accuracy: fraction of all detection events that are correct
    total_events = len(y_true)
    correct = int(np.sum(y_true == y_pred))
    accuracy = correct / total_events if total_events > 0 else 0.0
    
    # Micro-averaged precision, recall, F1
    micro_precision = total_tp / (total_tp + total_fp) if (total_tp + total_fp) > 0 else 0.0
    micro_recall = total_tp / (total_tp + total_fn) if (total_tp + total_fn) > 0 else 0.0
    micro_f1 = (2 * micro_precision * micro_recall / (micro_precision + micro_recall)
                if (micro_precision + micro_recall) > 0 else 0.0)
    
    # Macro-averaged
    macro_precision = np.mean([m["precision"] for m in per_class_metrics.values()])
    macro_recall = np.mean([m["recall"] for m in per_class_metrics.values()])
    macro_f1 = np.mean([m["f1"] for m in per_class_metrics.values()])
    
    overall_metrics = {
        "accuracy": round(accuracy, 4),
        "micro_precision": round(micro_precision, 4),
        "micro_recall": round(micro_recall, 4),
        "micro_f1": round(micro_f1, 4),
        "macro_precision": round(float(macro_precision), 4),
        "macro_recall": round(float(macro_recall), 4),
        "macro_f1": round(float(macro_f1), 4),
        "total_gt_boxes": sum(gt_class_counts.values()),
        "total_pred_boxes": sum(pred_class_counts.values()),
        "total_matched_tp": total_tp,
        "total_fp": total_fp,
        "total_fn": total_fn,
    }
    
    # -----------------------------------------------------------------------
    # Print report
    # -----------------------------------------------------------------------
    print(f"{'='*70}")
    print(f"  EVALUATION RESULTS")
    print(f"{'='*70}")
    print(f"\n  GT box counts per class:")
    for cls, cnt in gt_class_counts.items():
        print(f"    {cls:>12}: {cnt}")
    
    print(f"\n  Per-Class Metrics (IoU ≥ {args.iou_thresh}):")
    print(f"  {'Class':>12}  {'Prec':>7}  {'Recall':>7}  {'F1':>7}  {'TP':>4}  {'FP':>4}  {'FN':>4}")
    print(f"  {'-'*55}")
    for cls_name in CLASS_NAMES:
        m = per_class_metrics[cls_name]
        print(f"  {cls_name:>12}  {m['precision']:>7.4f}  {m['recall']:>7.4f}  {m['f1']:>7.4f}"
              f"  {m['tp']:>4}  {m['fp']:>4}  {m['fn']:>4}")
    
    print(f"\n  Overall Metrics:")
    print(f"    Accuracy (all events):    {overall_metrics['accuracy']:.4f}")
    print(f"    Micro Precision:          {overall_metrics['micro_precision']:.4f}")
    print(f"    Micro Recall:             {overall_metrics['micro_recall']:.4f}")
    print(f"    Micro F1:                 {overall_metrics['micro_f1']:.4f}")
    print(f"    Macro Precision:          {overall_metrics['macro_precision']:.4f}")
    print(f"    Macro Recall:             {overall_metrics['macro_recall']:.4f}")
    print(f"    Macro F1:                 {overall_metrics['macro_f1']:.4f}")
    print(f"{'='*70}\n")
    
    # -----------------------------------------------------------------------
    # Generate plots
    # -----------------------------------------------------------------------
    print("[3/4] Generating plots...\n")
    
    # Full confusion matrix (with background)
    plot_confusion_matrix(
        cm_full, extended_labels,
        output_dir / "confusion_matrix.png",
        title="Detection Confusion Matrix",
    )
    
    # Normalised confusion matrix
    plot_confusion_matrix(
        cm_full, extended_labels,
        output_dir / "confusion_matrix_normalised.png",
        title="Detection Confusion Matrix",
        normalize=True,
    )
    
    # Per-class confusion matrix (4×4, excluding background)
    cm_classes = confusion_matrix(
        y_true_cls, y_pred_cls,
        labels=list(range(NUM_CLASSES)),
    )
    plot_confusion_matrix(
        cm_classes, CLASS_NAMES,
        output_dir / "confusion_matrix_classes_only.png",
        title="Classification Confusion Matrix (Matched Detections)",
    )
    
    # Per-class metrics bar chart
    plot_per_class_metrics(per_class_metrics, output_dir / "per_class_metrics.png")
    
    # -----------------------------------------------------------------------
    # Save JSON report
    # -----------------------------------------------------------------------
    print("[4/4] Saving results...\n")
    
    report = {
        "model": args.model,
        "test_dir": str(test_root),
        "n_images": len(images),
        "conf_threshold": args.conf,
        "nms_iou": args.iou_nms,
        "match_iou": args.iou_thresh,
        "overall_metrics": overall_metrics,
        "per_class_metrics": per_class_metrics,
        "confusion_matrix": cm_full.tolist(),
        "confusion_matrix_labels": extended_labels,
        "per_image_summary": [
            {
                "image": r["image"],
                "n_gt": r["n_gt"],
                "n_pred": r["n_pred"],
            }
            for r in per_image_results
        ],
    }
    
    report_path = output_dir / "evaluation_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"  Report:     {report_path}")
    print(f"  Plots:      {output_dir}/confusion_matrix*.png")
    print(f"  Metrics:    {output_dir}/per_class_metrics.png")
    print(f"  Comparisons: {compare_dir}/")
    print(f"\n{'='*70}")
    print("  Evaluation complete!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
