"""
YOLO detection wrapper (refactored from model.py).
Handles training, inference, and result visualisation.
"""

import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO


class YOLODetector:

    def __init__(self, weights: str = "yolo11n.pt"):
        self.weights = weights
        self.model = YOLO(weights)
        self.device = "0" if torch.cuda.is_available() else "cpu"

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self, data_yaml: str, epochs: int = 50, imgsz: int = 640,
              batch: int = 16, project: str = "runs/detect", name: str = "train") -> Path:
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            device=self.device,
            project=project,
            name=name,
            exist_ok=True,
        )
        best = Path(project) / name / "weights" / "best.pt"
        print(f"Training complete. Best weights → {best}")
        return best

    # ------------------------------------------------------------------
    # Inference on a single image (PIL or path)
    # ------------------------------------------------------------------

    def predict_image(self, image, conf: float = 0.25):
        """
        Returns list of dicts: {box, conf, cls_id, cls_name}
        image: str path OR PIL.Image OR np.ndarray
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")

        results = self.model(image, conf=conf, verbose=False)[0]
        detections = []
        for box in results.boxes:
            detections.append({
                "box": box.xyxy[0].cpu().numpy().tolist(),   # [x1,y1,x2,y2]
                "conf": float(box.conf[0]),
                "cls_id": int(box.cls[0]),
                "cls_name": results.names[int(box.cls[0])],
            })
        return detections

    def render(self, image, conf: float = 0.25) -> np.ndarray:
        """Return BGR numpy array with bounding boxes drawn."""
        if isinstance(image, str):
            img_np = np.array(Image.open(image).convert("RGB"))
        elif isinstance(image, Image.Image):
            img_np = np.array(image.convert("RGB"))
        else:
            img_np = image.copy()

        detections = self.predict_image(img_np, conf)
        H, W = img_np.shape[:2]

        for d in detections:
            x1, y1, x2, y2 = [int(v) for v in d["box"]]
            label = f"{d['cls_name']} {d['conf']:.2f}"
            color = (0, 200, 100)
            cv2.rectangle(img_np, (x1, y1), (x2, y2), color, 2)
            cv2.putText(img_np, label, (x1, max(y1 - 6, 0)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

        return img_np

    # ------------------------------------------------------------------
    # Batch inference on a directory
    # ------------------------------------------------------------------

    def run_on_directory(self, image_dir: str, conf: float = 0.25,
                         n_show: int = 15, rows: int = 3):
        from glob import glob
        images = sorted(glob(f"{image_dir}/*"))[:n_show]
        cols = n_show // rows
        plt.figure(figsize=(cols * 5, rows * 4))

        for idx, img_path in enumerate(images, 1):
            plt.subplot(rows, cols, idx)
            rendered = self.render(img_path, conf)
            plt.imshow(rendered)
            plt.axis("off")
            plt.title(f"#{idx}", fontsize=8)

        plt.suptitle("YOLO Detections — Test Set", fontsize=14)
        plt.tight_layout()
        plt.show()

    # ------------------------------------------------------------------
    # Speed benchmark
    # ------------------------------------------------------------------

    def benchmark_fps(self, image_dir: str, conf: float = 0.25, max_images: int = 100) -> float:
        from glob import glob
        paths = sorted(glob(f"{image_dir}/*"))[:max_images]
        start = time.perf_counter()
        for p in paths:
            self.predict_image(p, conf)
        elapsed = time.perf_counter() - start
        fps = len(paths) / elapsed
        print(f"YOLO FPS: {fps:.1f} ({len(paths)} images in {elapsed:.2f}s)")
        return fps
