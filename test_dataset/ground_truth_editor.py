#!/usr/bin/env python3
"""Visual editor for correcting YOLO ground-truth annotations.

Usage:
    python test_dataset/ground_truth_editor.py
    python test_dataset/ground_truth_editor.py --test-dir test_dataset

The editor writes the same normalized YOLO files consumed by evaluate.py.
"""
from __future__ import annotations

import argparse
import json
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import cv2
from PIL import Image, ImageTk

CLASS_NAMES = ["shoes", "accessories", "clothing", "jewellery"]
COLOURS = ["#ff6347", "#40e0d0", "#9370db", "#ffd700"]
COLOURS_BGR = {
    0: (71, 99, 255),
    1: (208, 224, 64),
    2: (219, 112, 147),
    3: (0, 215, 255),
}
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
MIN_BOX_SIZE = 4


class GroundTruthEditor:
    def __init__(self, root: tk.Tk, test_dir: Path) -> None:
        self.root = root
        self.test_dir = test_dir
        self.image_dir = test_dir / "images"
        self.label_dir = test_dir / "labels"
        self.images = sorted(
            path for path in self.image_dir.iterdir()
            if path.suffix.lower() in IMAGE_EXTS
        )
        self.index = 0
        self.boxes: list[dict[str, float | int]] = []
        self.selected: int | None = None
        self.drag_kind: str | None = None
        self.drag_start: tuple[float, float] | None = None
        self.drag_origin_box: list[float] | None = None
        self.drag_box: list[float] | None = None
        self.dirty = False
        self.image_bgr = None
        self.photo = None
        self.scale = 1.0
        self.offset = (0, 0)
        self.add_mode = False

        self.root.title("Ground Truth Editor")
        self.root.geometry("1200x800")
        self.root.minsize(800, 600)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.build_ui()
        self.root.bind("<Left>", lambda _event: self.change_image(-1))
        self.root.bind("<Right>", lambda _event: self.change_image(1))
        self.root.bind("<Delete>", lambda _event: self.delete_selected())
        self.root.bind("<Escape>", lambda _event: self.cancel_add_mode())
        self.root.bind("a", lambda _event: self.toggle_add_mode())
        self.root.bind("<Control-s>", lambda _event: self.save())
        self.load_image()

    def build_ui(self) -> None:
        toolbar = ttk.Frame(self.root, padding=(10, 8))
        toolbar.pack(fill="x")
        ttk.Button(toolbar, text="Previous", command=lambda: self.change_image(-1)).pack(side="left")
        ttk.Button(toolbar, text="Next", command=lambda: self.change_image(1)).pack(side="left", padx=(5, 14))
        self.image_label = ttk.Label(toolbar, width=28)
        self.image_label.pack(side="left")
        ttk.Button(toolbar, text="Save", command=self.save).pack(side="right")
        self.add_button = ttk.Button(toolbar, text="Add box", command=self.toggle_add_mode)
        self.add_button.pack(side="right", padx=5)

        body = ttk.Frame(self.root)
        body.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self.canvas = tk.Canvas(body, background="#202124", highlightthickness=0,
                                cursor="crosshair")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.bind("<Configure>", lambda _event: self.render())
        self.canvas.bind("<Button-1>", self.mouse_down)
        self.canvas.bind("<B1-Motion>", self.mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.mouse_up)

        side = ttk.Frame(body, width=190, padding=(12, 0, 0, 0))
        side.pack(side="right", fill="y")
        ttk.Label(side, text="Selected box", font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
        self.class_var = tk.StringVar(value=CLASS_NAMES[0])
        self.class_box = ttk.Combobox(side, textvariable=self.class_var,
                                      values=CLASS_NAMES, state="readonly", width=16)
        self.class_box.pack(anchor="w", pady=(6, 10))
        self.class_box.bind("<<ComboboxSelected>>", self.change_class)
        ttk.Button(side, text="Delete selected", command=self.delete_selected).pack(anchor="w")
        ttk.Separator(side).pack(fill="x", pady=16)
        ttk.Label(side, text="How to edit", font=("TkDefaultFont", 10, "bold")).pack(anchor="w")
        ttk.Label(side, text="Click and hold inside a box\nto move it.\n\nClick and hold any vertex\nto resize it.\n\nUse Add box, then drag\non the image to create one.\n\nArrow keys: navigate\nDelete: remove box\nCtrl+S: save", justify="left").pack(anchor="w", pady=(6, 0))
        self.status = ttk.Label(self.root, padding=(10, 0, 10, 8))
        self.status.pack(fill="x")

    def load_image(self) -> None:
        if not self.images:
            messagebox.showerror("No images", f"No images found in {self.image_dir}")
            self.root.destroy()
            return
        self.image_bgr = cv2.imread(str(self.images[self.index]))
        if self.image_bgr is None:
            messagebox.showerror("Read error", f"Cannot read {self.images[self.index]}")
            return
        self.boxes = self.read_labels(self.images[self.index])
        self.selected = None
        self.dirty = False
        self.image_label.configure(text=f"{self.images[self.index].name}  ({self.index + 1}/{len(self.images)})")
        self.status.configure(text=f"{len(self.boxes)} boxes | unsaved changes appear here")
        self.render()

    def read_labels(self, image_path: Path) -> list[dict[str, float | int]]:
        image = self.image_bgr
        height, width = image.shape[:2]
        label_path = self.label_dir / f"{image_path.stem}.txt"
        boxes = []
        if not label_path.exists():
            return boxes
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                class_id = int(parts[0])
                xc, yc, box_width, box_height = (float(value) for value in parts[1:5])
            except ValueError:
                continue
            if class_id not in range(len(CLASS_NAMES)):
                continue
            boxes.append({
                "class_id": class_id,
                "x1": (xc - box_width / 2) * width,
                "y1": (yc - box_height / 2) * height,
                "x2": (xc + box_width / 2) * width,
                "y2": (yc + box_height / 2) * height,
            })
        return boxes

    def render(self) -> None:
        if self.image_bgr is None or not self.canvas.winfo_exists():
            return
        height, width = self.image_bgr.shape[:2]
        canvas_width = max(self.canvas.winfo_width(), 1)
        canvas_height = max(self.canvas.winfo_height(), 1)
        self.scale = min(canvas_width / width, canvas_height / height)
        display_width = max(1, int(width * self.scale))
        display_height = max(1, int(height * self.scale))
        self.offset = ((canvas_width - display_width) // 2, (canvas_height - display_height) // 2)
        rgb = cv2.cvtColor(self.image_bgr, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb).resize((display_width, display_height), Image.Resampling.LANCZOS)
        self.photo = ImageTk.PhotoImage(image)
        self.canvas.delete("all")
        self.canvas.create_image(*self.offset, image=self.photo, anchor="nw")
        for box_index, box in enumerate(self.boxes):
            x1, y1 = self.to_canvas(box["x1"], box["y1"])
            x2, y2 = self.to_canvas(box["x2"], box["y2"])
            class_id = int(box["class_id"])
            colour = COLOURS[class_id]
            line_width = 3 if box_index == self.selected else 2
            self.canvas.create_rectangle(x1, y1, x2, y2, outline=colour, width=line_width)
            self.canvas.create_text(x1 + 4, y1 + 4, text=CLASS_NAMES[class_id], anchor="nw",
                                    fill=colour, font=("TkDefaultFont", 10, "bold"))
            if box_index == self.selected:
                for handle_x, handle_y in ((x1, y1), (x2, y1), (x1, y2), (x2, y2)):
                    self.canvas.create_rectangle(handle_x - 4, handle_y - 4, handle_x + 4,
                                                 handle_y + 4, fill=colour, outline="white")

    def to_canvas(self, x: float, y: float) -> tuple[float, float]:
        return self.offset[0] + x * self.scale, self.offset[1] + y * self.scale

    def to_image(self, x: float, y: float) -> tuple[float, float]:
        height, width = self.image_bgr.shape[:2]
        return (max(0, min(width, (x - self.offset[0]) / self.scale)),
                max(0, min(height, (y - self.offset[1]) / self.scale)))

    def box_at(self, x: float, y: float) -> int | None:
        for box_index in range(len(self.boxes) - 1, -1, -1):
            box = self.boxes[box_index]
            if box["x1"] <= x <= box["x2"] and box["y1"] <= y <= box["y2"]:
                return box_index
        return None

    def handle_at(self, box: dict, x: float, y: float) -> str | None:
        tolerance = max(10 / self.scale, MIN_BOX_SIZE)
        for name, hx, hy in (("nw", box["x1"], box["y1"]), ("ne", box["x2"], box["y1"]),
                             ("sw", box["x1"], box["y2"]), ("se", box["x2"], box["y2"])):
            if abs(x - hx) <= tolerance and abs(y - hy) <= tolerance:
                return name
        return None

    def box_or_handle_at(self, x: float, y: float) -> tuple[int | None, str | None]:
        for box_index in range(len(self.boxes) - 1, -1, -1):
            box = self.boxes[box_index]
            handle = self.handle_at(box, x, y)
            if handle is not None:
                return box_index, handle
            if box["x1"] <= x <= box["x2"] and box["y1"] <= y <= box["y2"]:
                return box_index, "move"
        return None, None

    def mouse_down(self, event: tk.Event) -> None:
        x, y = self.to_image(event.x, event.y)
        if self.add_mode:
            self.drag_kind = "add"
            self.drag_start = (x, y)
            self.drag_origin_box = None
            self.drag_box = [x, y, x, y]
            return
        self.selected, self.drag_kind = self.box_or_handle_at(x, y)
        if self.selected is not None:
            self.drag_start = (x, y)
            box = self.boxes[self.selected]
            self.drag_origin_box = [box["x1"], box["y1"], box["x2"], box["y2"]]
            self.drag_box = self.drag_origin_box.copy()
            self.class_var.set(CLASS_NAMES[int(box["class_id"])])
        self.render()

    def mouse_drag(self, event: tk.Event) -> None:
        if self.drag_kind is None or self.drag_start is None or self.drag_box is None:
            return
        x, y = self.to_image(event.x, event.y)
        start_x, start_y = self.drag_start
        x1, y1, x2, y2 = self.drag_box
        if self.drag_kind == "add":
            self.drag_box = [start_x, start_y, x, y]
        elif self.drag_kind == "move":
            dx, dy = x - start_x, y - start_y
            origin_x1, origin_y1, origin_x2, origin_y2 = self.drag_origin_box
            height, width = self.image_bgr.shape[:2]
            dx = max(-origin_x1, min(width - origin_x2, dx))
            dy = max(-origin_y1, min(height - origin_y2, dy))
            self.drag_box = [origin_x1 + dx, origin_y1 + dy,
                             origin_x2 + dx, origin_y2 + dy]
        else:
            if "w" in self.drag_kind:
                x1 = x
            else:
                x2 = x
            if "n" in self.drag_kind:
                y1 = y
            else:
                y2 = y
            self.drag_box = [x1, y1, x2, y2]
        self.draw_drag_box()

    def draw_drag_box(self) -> None:
        self.render()
        if self.drag_box is not None:
            x1, y1, x2, y2 = self.drag_box
            self.canvas.create_rectangle(*self.to_canvas(x1, y1), *self.to_canvas(x2, y2),
                                         outline="white", width=2, dash=(5, 3))

    def mouse_up(self, _event: tk.Event) -> None:
        if self.drag_kind is None or self.drag_box is None:
            return
        x1, y1, x2, y2 = self.drag_box
        x1, x2 = sorted((x1, x2))
        y1, y2 = sorted((y1, y2))
        if x2 - x1 >= MIN_BOX_SIZE and y2 - y1 >= MIN_BOX_SIZE:
            if self.drag_kind == "add":
                self.boxes.append({"class_id": CLASS_NAMES.index(self.class_var.get()),
                                   "x1": x1, "y1": y1, "x2": x2, "y2": y2})
                self.selected = len(self.boxes) - 1
                self.cancel_add_mode()
            elif self.selected is not None:
                self.boxes[self.selected].update(x1=x1, y1=y1, x2=x2, y2=y2)
            self.dirty = True
        self.drag_kind = None
        self.drag_start = None
        self.drag_origin_box = None
        self.drag_box = None
        self.status.configure(text=f"{len(self.boxes)} boxes | {'unsaved changes' if self.dirty else 'saved'}")
        self.render()

    def toggle_add_mode(self) -> None:
        self.add_mode = not self.add_mode
        self.add_button.configure(text="Cancel add" if self.add_mode else "Add box")
        self.canvas.configure(cursor="crosshair" if self.add_mode else "arrow")

    def cancel_add_mode(self) -> None:
        self.add_mode = False
        self.add_button.configure(text="Add box")
        self.canvas.configure(cursor="arrow")

    def change_class(self, _event=None) -> None:
        if self.selected is not None:
            self.boxes[self.selected]["class_id"] = CLASS_NAMES.index(self.class_var.get())
            self.dirty = True
            self.render()

    def delete_selected(self) -> None:
        if self.selected is not None:
            del self.boxes[self.selected]
            self.selected = None
            self.dirty = True
            self.status.configure(text=f"{len(self.boxes)} boxes | unsaved changes")
            self.render()

    def save(self) -> None:
        if self.image_bgr is None:
            return
        height, width = self.image_bgr.shape[:2]
        lines = []
        for box in self.boxes:
            x1 = max(0.0, min(width, float(box["x1"])))
            y1 = max(0.0, min(height, float(box["y1"])))
            x2 = max(0.0, min(width, float(box["x2"])))
            y2 = max(0.0, min(height, float(box["y2"])))
            lines.append(f"{int(box['class_id'])} {(x1 + x2) / 2 / width:.6f} "
                         f"{(y1 + y2) / 2 / height:.6f} {(x2 - x1) / width:.6f} "
                         f"{(y2 - y1) / height:.6f}")
        self.label_dir.mkdir(parents=True, exist_ok=True)
        label_path = self.label_dir / f"{self.images[self.index].stem}.txt"
        label_path.write_text("\n".join(lines) + ("\n" if lines else ""))
        self.update_review_image(label_path, lines, width, height)
        self.update_summary()
        self.dirty = False
        self.status.configure(text=f"{len(self.boxes)} boxes | labels, review image, and summary saved")

    def update_review_image(self, label_path: Path, lines: list[str], width: int, height: int) -> None:
        review_dir = self.test_dir / "review"
        review_dir.mkdir(parents=True, exist_ok=True)
        annotated = self.image_bgr.copy()
        for line in lines:
            parts = line.split()
            class_id = int(parts[0])
            xc, yc, box_width, box_height = (float(value) for value in parts[1:])
            x1 = int((xc - box_width / 2) * width)
            y1 = int((yc - box_height / 2) * height)
            x2 = int((xc + box_width / 2) * width)
            y2 = int((yc + box_height / 2) * height)
            colour = COLOURS_BGR[class_id]
            label = f"{CLASS_NAMES[class_id]} ({class_id})"
            cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            label_y = max(y1 - text_height - 6, 0)
            cv2.rectangle(annotated, (x1, label_y), (x1 + text_width + 4, y1), colour, -1)
            cv2.putText(annotated, label, (x1 + 2, max(y1 - 4, text_height)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(review_dir / self.images[self.index].name), annotated,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])

    def update_summary(self) -> None:
        per_image = []
        total_boxes = {class_name: 0 for class_name in CLASS_NAMES}
        for image_path in self.images:
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            boxes = self.read_labels_for_image(image_path, image.shape[1], image.shape[0])
            image_classes = {}
            for box in boxes:
                class_name = CLASS_NAMES[int(box["class_id"])]
                total_boxes[class_name] += 1
                image_classes[class_name] = image_classes.get(class_name, 0) + 1
            per_image.append({"image": image_path.name, "boxes": image_classes})
        (self.test_dir / "annotation_summary.json").write_text(
            json.dumps({"per_image": per_image, "total_boxes": total_boxes}, indent=2)
        )

    def read_labels_for_image(self, image_path: Path, width: int, height: int) -> list[dict[str, float | int]]:
        label_path = self.label_dir / f"{image_path.stem}.txt"
        boxes = []
        if not label_path.exists():
            return boxes
        for line in label_path.read_text().splitlines():
            parts = line.split()
            if len(parts) < 5:
                continue
            try:
                class_id = int(parts[0])
                xc, yc, box_width, box_height = (float(value) for value in parts[1:5])
            except ValueError:
                continue
            if class_id not in range(len(CLASS_NAMES)):
                continue
            boxes.append({
                "class_id": class_id,
                "x1": (xc - box_width / 2) * width,
                "y1": (yc - box_height / 2) * height,
                "x2": (xc + box_width / 2) * width,
                "y2": (yc + box_height / 2) * height,
            })
        return boxes

    def change_image(self, step: int) -> None:
        if self.dirty:
            answer = messagebox.askyesnocancel("Unsaved changes", "Save changes before changing image?")
            if answer is None:
                return
            if answer:
                self.save()
        self.index = max(0, min(len(self.images) - 1, self.index + step))
        self.load_image()

    def close(self) -> None:
        if self.dirty:
            answer = messagebox.askyesnocancel("Unsaved changes", "Save changes before closing?")
            if answer is None:
                return
            if answer:
                self.save()
        self.root.destroy()


def main() -> None:
    parser = argparse.ArgumentParser(description="Visually correct YOLO ground-truth labels")
    parser.add_argument("--test-dir", default="test_dataset", help="Root directory of the test dataset")
    args = parser.parse_args()
    root = tk.Tk()
    GroundTruthEditor(root, Path(args.test_dir).resolve())
    root.mainloop()


if __name__ == "__main__":
    main()