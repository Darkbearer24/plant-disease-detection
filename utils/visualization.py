"""
Dataset visualization utilities (refactored from model.py).
"""

import os
import random
from glob import glob
from pathlib import Path

import cv2
import yaml
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image


def load_class_names(data_yaml: str) -> dict[int, str]:
    with open(data_yaml) as f:
        names = yaml.safe_load(f)["names"]
    return {i: n for i, n in enumerate(names)}


class DatasetVisualizer:

    def __init__(self, dataset_root: str, data_yaml: str, splits=("train", "valid", "test")):
        self.root = dataset_root
        self.splits = splits
        self.class_dict = load_class_names(data_yaml)
        self._load()

    def _load(self):
        self.im_paths = {}
        self.vis_datas = {}
        self.analysis_datas = {}

        for split in self.splits:
            im_dir = Path(self.root) / split / "images"
            lbl_dir = Path(self.root) / split / "labels"
            im_files = sorted(im_dir.glob("*"))

            bboxes_per_image = []
            cls_counts: dict[str, int] = {}
            valid_paths = []

            for img_path in im_files:
                lbl_path = lbl_dir / (img_path.stem + ".txt")
                if not lbl_path.exists():
                    continue

                bboxes = []
                for line in lbl_path.read_text().strip().splitlines():
                    parts = line.strip().split()
                    if len(parts) < 5:
                        continue
                    cls_name = self.class_dict[int(parts[0])]
                    bboxes.append([cls_name] + [float(x) for x in parts[1:5]])
                    cls_counts[cls_name] = cls_counts.get(cls_name, 0) + 1

                bboxes_per_image.append(bboxes)
                valid_paths.append(str(img_path))

            self.vis_datas[split] = bboxes_per_image
            self.analysis_datas[split] = cls_counts
            self.im_paths[split] = valid_paths

    def show_samples(self, split: str = "train", n: int = 20, rows: int = 5):
        data = self.vis_datas[split]
        paths = self.im_paths[split]
        if not data:
            print(f"No images found for split '{split}'.")
            return

        cols = n // rows
        plt.figure(figsize=(cols * 5, rows * 4))
        indices = random.sample(range(len(data)), min(n, len(data)))

        for pos, idx in enumerate(indices, 1):
            plt.subplot(rows, cols, pos)
            img = np.array(Image.open(paths[idx]).convert("RGB"))
            H, W = img.shape[:2]

            for bbox in data[idx]:
                cls_name, xc, yc, bw, bh = bbox
                x1 = int((xc - bw / 2) * W)
                y1 = int((yc - bh / 2) * H)
                x2 = int((xc + bw / 2) * W)
                y2 = int((yc + bh / 2) * H)
                color = tuple(random.randint(50, 255) for _ in range(3))
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 3)
                cv2.putText(img, cls_name[:12], (x1, max(y1 - 5, 0)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            plt.imshow(img)
            plt.axis("off")
            plt.title(f"{len(data[idx])} object(s)", fontsize=8)

        plt.suptitle(f"{split.upper()} — sample images", fontsize=14)
        plt.tight_layout()
        plt.show()

    def show_class_distribution(self, split: str = "train"):
        counts = self.analysis_datas[split]
        cls_names = list(counts.keys())
        values = list(counts.values())

        _, ax = plt.subplots(figsize=(20, 6))
        bars = ax.bar(range(len(values)), values, color="steelblue", width=0.7)
        ax.set_xticks(range(len(cls_names)))
        ax.set_xticklabels(cls_names, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Annotation count")
        ax.set_title(f"{split.upper()} — class distribution")

        for bar, v in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.5,
                    str(v), ha="center", fontsize=7, color="navy")

        plt.tight_layout()
        plt.show()

    def show_all(self, n: int = 20, rows: int = 5):
        for split in self.splits:
            self.show_class_distribution(split)
            self.show_samples(split, n, rows)
