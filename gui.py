#!/usr/bin/env python3
"""Desktop GUI for running fashion-detector inference."""
from __future__ import annotations

import json
import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from PIL import Image, ImageTk
from ultralytics import YOLO

from infer import CLASS_NAMES, CROP_PAD_FRAC, IMAGE_EXTENSIONS, process_image


class InferenceApp:
    """Small Tkinter front end around the existing inference pipeline."""

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Fashion Detector")
        self.root.geometry("1180x760")
        self.root.minsize(900, 620)

        self.input_var = tk.StringVar()
        self.model_var = tk.StringVar(value="fashion_best.pt")
        self.output_var = tk.StringVar(value="outputs")
        self.conf_var = tk.StringVar(value="0.30")
        self.iou_var = tk.StringVar(value="0.55")
        self.status_var = tk.StringVar(value="Choose an image to begin.")
        self.result_queue: queue.Queue[tuple[str, object]] = queue.Queue()
        self.preview_image: ImageTk.PhotoImage | None = None
        self.crop_images: list[ImageTk.PhotoImage] = []
        self.run_button: ttk.Button

        self._build_style()
        self._build_layout()

    def _build_style(self) -> None:
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Title.TLabel", font=("Segoe UI", 20, "bold"), foreground="#17324d")
        style.configure("Subtitle.TLabel", font=("Segoe UI", 10), foreground="#5d7183")
        style.configure("Section.TLabel", font=("Segoe UI", 11, "bold"), foreground="#17324d")
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), padding=(14, 8))

    def _build_layout(self) -> None:
        root_frame = ttk.Frame(self.root, padding=18)
        root_frame.pack(fill="both", expand=True)
        root_frame.columnconfigure(1, weight=1)
        root_frame.rowconfigure(2, weight=1)

        ttk.Label(root_frame, text="Fashion Detector", style="Title.TLabel").grid(
            row=0, column=0, columnspan=2, sticky="w"
        )
        ttk.Label(
            root_frame,
            text="Run YOLO inference and inspect the annotated image, crops, and detection data.",
            style="Subtitle.TLabel",
        ).grid(row=1, column=0, columnspan=2, sticky="w", pady=(2, 14))

        settings = ttk.LabelFrame(root_frame, text="Inference settings", padding=12)
        settings.grid(row=2, column=0, sticky="ns", padx=(0, 14))
        settings.columnconfigure(1, weight=1)

        self._path_row(settings, 0, "Input image", self.input_var, self._choose_input)
        self._path_row(settings, 1, "Model weights", self.model_var, self._choose_model)
        self._path_row(settings, 2, "Output folder", self.output_var, self._choose_output)
        self._entry_row(settings, 3, "Confidence", self.conf_var)
        self._entry_row(settings, 4, "IoU threshold", self.iou_var)

        self.run_button = ttk.Button(
            settings, text="Run inference", style="Accent.TButton", command=self._start_inference
        )
        self.run_button.grid(row=5, column=0, columnspan=3, sticky="ew", pady=(18, 8))
        ttk.Label(settings, textvariable=self.status_var, wraplength=270).grid(
            row=6, column=0, columnspan=3, sticky="w"
        )

        results = ttk.Notebook(root_frame)
        results.grid(row=2, column=1, sticky="nsew")

        image_tab = ttk.Frame(results, padding=10)
        crops_tab = ttk.Frame(results, padding=10)
        data_tab = ttk.Frame(results, padding=10)
        results.add(image_tab, text="Annotated image")
        results.add(crops_tab, text="Detected crops")
        results.add(data_tab, text="Detection data")

        self.image_canvas = tk.Canvas(image_tab, background="#edf2f5", highlightthickness=0)
        self.image_canvas.pack(fill="both", expand=True)

        crops_tab.columnconfigure(0, weight=1)
        crops_tab.rowconfigure(0, weight=1)
        self.crops_canvas = tk.Canvas(crops_tab, background="#edf2f5", highlightthickness=0)
        crops_scroll = ttk.Scrollbar(crops_tab, orient="vertical", command=self.crops_canvas.yview)
        self.crops_canvas.configure(yscrollcommand=crops_scroll.set)
        self.crops_canvas.grid(row=0, column=0, sticky="nsew")
        crops_scroll.grid(row=0, column=1, sticky="ns")
        self.crops_frame = ttk.Frame(self.crops_canvas)
        self.crops_window = self.crops_canvas.create_window((0, 0), window=self.crops_frame, anchor="nw")
        self.crops_frame.bind("<Configure>", self._update_crop_scroll)
        self.crops_canvas.bind("<Configure>", self._resize_crop_frame)

        data_tab.columnconfigure(0, weight=1)
        data_tab.rowconfigure(0, weight=1)
        self.data_text = tk.Text(
            data_tab, wrap="none", state="disabled", font=("Consolas", 10),
            background="#f7f9fa", foreground="#17324d", padx=10, pady=10,
        )
        data_scroll = ttk.Scrollbar(data_tab, orient="vertical", command=self.data_text.yview)
        self.data_text.configure(yscrollcommand=data_scroll.set)
        self.data_text.grid(row=0, column=0, sticky="nsew")
        data_scroll.grid(row=0, column=1, sticky="ns")

    def _path_row(
        self, parent: ttk.Frame, row: int, label: str, variable: tk.StringVar, command: object
    ) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=5)
        ttk.Entry(parent, textvariable=variable, width=27).grid(row=row, column=1, sticky="ew", padx=8)
        ttk.Button(parent, text="Browse", command=command).grid(row=row, column=2)

    def _entry_row(self, parent: ttk.Frame, row: int, label: str, variable: tk.StringVar) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=5)
        ttk.Entry(parent, textvariable=variable, width=10).grid(row=row, column=1, sticky="w", padx=8)

    def _choose_input(self) -> None:
        path = filedialog.askopenfilename(
            title="Select input image",
            filetypes=[("Images", " ".join(f"*{ext}" for ext in sorted(IMAGE_EXTENSIONS))), ("All files", "*.*")],
        )
        if path:
            self.input_var.set(path)

    def _choose_model(self) -> None:
        path = filedialog.askopenfilename(
            title="Select model weights", filetypes=[("YOLO weights", "*.pt"), ("All files", "*.*")]
        )
        if path:
            self.model_var.set(path)

    def _choose_output(self) -> None:
        path = filedialog.askdirectory(title="Select output folder")
        if path:
            self.output_var.set(path)

    def _start_inference(self) -> None:
        input_path = Path(self.input_var.get().strip())
        model_path = Path(self.model_var.get().strip())
        if not input_path.is_file() or input_path.suffix.lower() not in IMAGE_EXTENSIONS:
            messagebox.showerror("Invalid input", "Select a valid image file first.")
            return
        if not model_path.is_file():
            messagebox.showerror("Model not found", f"Could not find model weights:\n{model_path}")
            return
        try:
            conf = float(self.conf_var.get())
            iou = float(self.iou_var.get())
            if not 0 <= conf <= 1 or not 0 <= iou <= 1:
                raise ValueError
        except ValueError:
            messagebox.showerror("Invalid parameters", "Confidence and IoU must be numbers from 0 to 1.")
            return

        self.run_button.configure(state="disabled")
        self.status_var.set("Loading model and running inference...")
        worker = threading.Thread(
            target=self._run_inference,
            args=(input_path, model_path, Path(self.output_var.get().strip() or "outputs"), conf, iou),
            daemon=True,
        )
        worker.start()
        self.root.after(100, self._poll_result)

    def _run_inference(
        self, input_path: Path, model_path: Path, output_root: Path, conf: float, iou: float
    ) -> None:
        try:
            model = YOLO(str(model_path))
            model_names = getattr(model, "names", None)
            class_names = (
                [model_names[i] for i in sorted(model_names)]
                if isinstance(model_names, dict) and model_names
                else CLASS_NAMES
            )
            doc = process_image(
                model, input_path, output_root, conf, iou, 640, class_names, CROP_PAD_FRAC, 95
            )
            self.result_queue.put(("success", (doc, output_root)))
        except Exception as error:  # Keep model/runtime errors inside the GUI.
            self.result_queue.put(("error", str(error)))

    def _poll_result(self) -> None:
        try:
            kind, payload = self.result_queue.get_nowait()
        except queue.Empty:
            self.root.after(100, self._poll_result)
            return
        self.run_button.configure(state="normal")
        if kind == "error":
            self.status_var.set("Inference failed.")
            messagebox.showerror("Inference failed", str(payload))
            return
        doc, output_root = payload
        self._show_results(doc, output_root)
        self.status_var.set(f"Finished: {len(doc['detections'])} detection(s)")

    def _show_results(self, doc: dict, output_root: Path) -> None:
        annotated_path = output_root / "annotated" / f"{Path(doc['image']).stem}.jpg"
        self._show_image(annotated_path)
        self._show_crops(doc, output_root)
        self.data_text.configure(state="normal")
        self.data_text.delete("1.0", tk.END)
        self.data_text.insert("1.0", json.dumps(doc, indent=2))
        self.data_text.configure(state="disabled")

    def _show_image(self, path: Path) -> None:
        image = Image.open(path).convert("RGB")
        image.thumbnail((850, 620), Image.Resampling.LANCZOS)
        self.preview_image = ImageTk.PhotoImage(image)
        self.image_canvas.delete("all")
        self.image_canvas.create_image(10, 10, image=self.preview_image, anchor="nw")

    def _show_crops(self, doc: dict, output_root: Path) -> None:
        for child in self.crops_frame.winfo_children():
            child.destroy()
        self.crop_images.clear()
        stem = Path(doc["image"]).stem
        if not doc["detections"]:
            ttk.Label(self.crops_frame, text="No detections at this confidence threshold.").grid(padx=16, pady=16)
            return
        for index, detection in enumerate(doc["detections"]):
            crop_path = output_root / detection["crop_path"]
            image = Image.open(crop_path).convert("RGB")
            image.thumbnail((180, 150), Image.Resampling.LANCZOS)
            photo = ImageTk.PhotoImage(image)
            self.crop_images.append(photo)
            card = ttk.Frame(self.crops_frame, padding=8, relief="ridge")
            card.grid(row=index // 3, column=index % 3, padx=8, pady=8, sticky="n")
            ttk.Label(card, image=photo).pack()
            ttk.Label(
                card,
                text=f"{detection['category']}  {detection['confidence']:.2f}\n"
                     f"bbox: {detection['bbox_xyxy']}",
                justify="center",
            ).pack(pady=(6, 0))

    def _update_crop_scroll(self, _event: tk.Event) -> None:
        self.crops_canvas.configure(scrollregion=self.crops_canvas.bbox("all"))

    def _resize_crop_frame(self, event: tk.Event) -> None:
        self.crops_canvas.itemconfigure(self.crops_window, width=event.width)


def main() -> None:
    root = tk.Tk()
    InferenceApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()