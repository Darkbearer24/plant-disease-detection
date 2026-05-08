"""
Fine-tune the Vision Transformer on PlantDoc crops.

Step 1 (one-time): extract crops from YOLO labels
    python train_vit.py --extract-crops

Step 2: train
    python train_vit.py
    python train_vit.py --epochs 20 --batch 32
"""

import argparse
import yaml
import json
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader, Subset

from utils.dataset_utils import (
    extract_crops, PlantCropDataset, split_dataset
)
from models.vit_classifier import build_vit, train_vit


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune ViT on PlantDoc crops")
    parser.add_argument("--config",         default="config/config.yaml")
    parser.add_argument("--extract-crops",  action="store_true",
                        help="Extract crops from YOLO labels before training")
    parser.add_argument("--epochs",         type=int, default=None)
    parser.add_argument("--batch",          type=int, default=None)
    parser.add_argument("--lr",             type=float, default=None)
    return parser.parse_args()


def plot_history(history: dict, save_dir: str):
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.plot(epochs, history["train_loss"], label="train")
    ax1.plot(epochs, history["val_loss"],   label="val")
    ax1.set_title("Loss")
    ax1.set_xlabel("Epoch")
    ax1.legend()

    ax2.plot(epochs, history["train_acc"], label="train")
    ax2.plot(epochs, history["val_acc"],   label="val")
    ax2.set_title("Accuracy")
    ax2.set_xlabel("Epoch")
    ax2.legend()

    plt.tight_layout()
    fig.savefig(f"{save_dir}/vit_training_curves.png", dpi=150)
    print(f"Training curves saved → {save_dir}/vit_training_curves.png")


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    dc  = cfg["dataset"]
    vc  = cfg["vit"]

    crops_dir = dc["crops_dir"]
    save_dir  = str(Path(vc["checkpoint"]).parent)

    # ---- (Optional) Extract crops ----
    if args.extract_crops:
        print("Extracting crops from YOLO labels...")
        extract_crops(
            dataset_root=dc["root"],
            data_yaml=dc["data_yaml"],
            crops_dir=crops_dir,
            splits=["train", "valid"],
        )

    if not Path(crops_dir).exists():
        print(f"Crops directory not found: {crops_dir}")
        print("Run with --extract-crops first.")
        return

    # ---- Dataset ----
    epochs     = args.epochs or vc["epochs"]
    batch_size = args.batch  or vc["batch_size"]
    lr         = args.lr     or vc["lr"]
    imgsz      = vc["imgsz"]
    workers    = vc["num_workers"]
    patience   = vc["patience"]
    backbone   = vc["backbone"]
    checkpoint = vc["checkpoint"]

    full_dataset = PlantCropDataset(crops_dir, imgsz=imgsz, augment=True)
    # Build a no-augment mirror with the same deterministic sample order
    val_dataset  = PlantCropDataset(crops_dir, imgsz=imgsz, augment=False)

    train_set, val_set = split_dataset(full_dataset, val_fraction=0.15)
    # Use the same val indices on the no-augment dataset so validation is clean
    val_set_noaug = Subset(val_dataset, val_set.indices)

    print(f"Train: {len(train_set)} samples | Val: {len(val_set_noaug)} samples")
    print(f"Classes: {len(full_dataset.class_names)}")

    train_loader = DataLoader(train_set,     batch_size=batch_size,
                              shuffle=True,  num_workers=workers, pin_memory=True)
    val_loader   = DataLoader(val_set_noaug, batch_size=batch_size,
                              shuffle=False, num_workers=workers, pin_memory=True)

    # Save class names alongside checkpoint (needed for inference)
    Path(checkpoint).parent.mkdir(parents=True, exist_ok=True)
    class_names_path = str(Path(checkpoint).parent / "class_names.json")
    with open(class_names_path, "w") as f:
        json.dump(full_dataset.class_names, f, indent=2)
    print(f"Class names saved → {class_names_path}")

    # ---- Model (use local cache if available, else download) ----
    local_cache = "runs/vit/backbone_cache"
    use_cache = Path(local_cache).exists()
    print(f"Loading backbone: {backbone}" + (" (from local cache)" if use_cache else " (downloading)"))
    model = build_vit(
        num_classes=len(full_dataset.class_names),
        backbone=backbone,
        cache_dir=local_cache if use_cache else None,
    )

    # ---- Train ----
    history = train_vit(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        weight_decay=vc["weight_decay"],
        patience=patience,
        save_path=checkpoint,
    )

    plot_history(history, save_dir)


if __name__ == "__main__":
    main()
