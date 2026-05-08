"""
Utilities for PlantDoc dataset.

Primary jobs:
  1. Extract cropped leaf images from YOLO-format labels → used to train the ViT classifier.
  2. Provide a PyTorch Dataset class for those crops.
"""

import os
import yaml
import random
from glob import glob
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


# ---------------------------------------------------------------------------
# Class name helpers
# ---------------------------------------------------------------------------

def load_class_names(data_yaml: str) -> list[str]:
    with open(data_yaml) as f:
        return yaml.safe_load(f)["names"]


# ---------------------------------------------------------------------------
# Crop extraction  (YOLO labels → cropped images per class)
# ---------------------------------------------------------------------------

def extract_crops(
    dataset_root: str,
    data_yaml: str,
    crops_dir: str,
    splits: list[str] = ("train", "valid"),
    padding: float = 0.05,
) -> None:
    """
    Walk YOLO image/label pairs and save each annotated bounding box as a
    separate cropped JPEG under:
        crops_dir/<class_name>/<split>_<img_stem>_<idx>.jpg

    padding: fraction of box dimension added on each side.
    """
    class_names = load_class_names(data_yaml)
    crops_root = Path(crops_dir)

    # Create per-class directories
    for name in class_names:
        (crops_root / name).mkdir(parents=True, exist_ok=True)

    total = 0
    for split in splits:
        img_dir = Path(dataset_root) / split / "images"
        lbl_dir = Path(dataset_root) / split / "labels"

        for img_path in sorted(img_dir.glob("*")):
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if not lbl_path.exists():
                continue

            try:
                img = Image.open(img_path).convert("RGB")
            except Exception:
                continue

            W, H = img.size
            lines = lbl_path.read_text().strip().splitlines()

            for idx, line in enumerate(lines):
                parts = line.strip().split()
                if len(parts) < 5:
                    continue

                cls_id = int(parts[0])
                xc, yc, bw, bh = map(float, parts[1:5])

                # Add padding
                bw_p = bw * (1 + 2 * padding)
                bh_p = bh * (1 + 2 * padding)

                x1 = max(0, int((xc - bw_p / 2) * W))
                y1 = max(0, int((yc - bh_p / 2) * H))
                x2 = min(W, int((xc + bw_p / 2) * W))
                y2 = min(H, int((yc + bh_p / 2) * H))

                if x2 <= x1 or y2 <= y1:
                    continue

                crop = img.crop((x1, y1, x2, y2))
                cls_name = class_names[cls_id]
                save_path = crops_root / cls_name / f"{split}_{img_path.stem}_{idx}.jpg"
                crop.save(save_path, quality=95)
                total += 1

    print(f"Extracted {total} crops → {crops_dir}")


# ---------------------------------------------------------------------------
# PyTorch Dataset for ViT training
# ---------------------------------------------------------------------------

def make_transforms(imgsz: int = 224, augment: bool = True):
    if augment:
        return transforms.Compose([
            transforms.RandomResizedCrop(imgsz, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
            transforms.RandomRotation(20),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])
    return transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])


class PlantCropDataset(Dataset):
    """
    Flat directory of per-class crops:
        crops_dir/<class_name>/*.jpg

    Class index is sorted alphabetically to stay deterministic.
    """

    def __init__(self, crops_dir: str, imgsz: int = 224, augment: bool = True):
        self.transform = make_transforms(imgsz, augment)

        class_dirs = sorted(
            [d for d in Path(crops_dir).iterdir() if d.is_dir()]
        )
        self.class_names = [d.name for d in class_dirs]
        self.class_to_idx = {name: i for i, name in enumerate(self.class_names)}

        self.samples: list[tuple[Path, int]] = []
        for cls_dir in class_dirs:
            cls_idx = self.class_to_idx[cls_dir.name]
            for img_path in sorted(cls_dir.glob("*.jpg")):
                self.samples.append((img_path, cls_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        return self.transform(img), label


def split_dataset(dataset: PlantCropDataset, val_fraction: float = 0.15):
    """Return train/val subsets (no data leakage — indices only)."""
    from torch.utils.data import Subset

    n = len(dataset)
    indices = list(range(n))
    random.shuffle(indices)
    split = int(n * val_fraction)
    return Subset(dataset, indices[split:]), Subset(dataset, indices[:split])
