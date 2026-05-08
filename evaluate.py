"""
Evaluate both models on the test split and produce a comparison report.

Usage:
    python evaluate.py
    python evaluate.py --config config/config.yaml
"""

import argparse
import json
from pathlib import Path

import yaml
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from utils.dataset_utils import PlantCropDataset
from utils.metrics import evaluate_classifier, plot_confusion_matrix, plot_comparison
from models.vit_classifier import ViTInference, build_vit, inference_transform


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/config.yaml")
    return parser.parse_args()


def load_class_names(checkpoint_dir: str) -> list[str]:
    path = Path(checkpoint_dir) / "class_names.json"
    with open(path) as f:
        return json.load(f)


def evaluate_vit(cfg: dict):
    vc = cfg["vit"]
    dc = cfg["dataset"]
    ckpt_dir = str(Path(vc["checkpoint"]).parent)

    class_names = load_class_names(ckpt_dir)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    infer = ViTInference(
        checkpoint=vc["checkpoint"],
        class_names=class_names,
        backbone=vc["backbone"],
        device=device,
    )

    test_dataset = PlantCropDataset(dc["crops_dir"], imgsz=vc["imgsz"], augment=False)
    test_loader  = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    metrics = evaluate_classifier(infer.model, test_loader, class_names, device)
    fps = infer.benchmark_fps(f"{dc['root']}/test/images")
    metrics["fps"] = fps
    return metrics, class_names


def evaluate_yolo_as_classifier(cfg: dict):
    """
    Use YOLO's top-1 classification prediction on test crops to produce
    comparable classification metrics.
    """
    import time
    import numpy as np
    from glob import glob
    from PIL import Image
    from models.yolo_detector import YOLODetector

    dc  = cfg["dataset"]
    yc  = cfg["yolo"]
    vc  = cfg["vit"]

    ckpt_dir     = str(Path(vc["checkpoint"]).parent)
    class_names  = load_class_names(ckpt_dir)
    cls_to_idx   = {n: i for i, n in enumerate(class_names)}
    crops_dir    = Path(dc["crops_dir"])
    weights      = yc["trained_weights"]

    detector = YOLODetector(weights=weights)

    all_preds, all_labels = [], []
    start = time.perf_counter()

    for cls_dir in sorted(crops_dir.iterdir()):
        if not cls_dir.is_dir() or cls_dir.name not in cls_to_idx:
            continue
        true_idx = cls_to_idx[cls_dir.name]
        for img_path in cls_dir.glob("*.jpg"):
            dets = detector.predict_image(str(img_path), conf=0.05)
            if dets:
                pred_name = dets[0]["cls_name"]
                pred_idx  = cls_to_idx.get(pred_name, -1)
            else:
                pred_idx = -1

            all_preds.append(pred_idx)
            all_labels.append(true_idx)

    elapsed = time.perf_counter() - start
    fps = len(all_preds) / elapsed

    from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)

    valid = all_preds >= 0
    acc = accuracy_score(all_labels[valid], all_preds[valid])
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None,
        labels=list(range(len(class_names))), zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))

    return {
        "accuracy": acc,
        "fps": fps,
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "support": support.tolist(),
        "confusion_matrix": cm,
        "class_names": class_names,
    }


def main():
    args = parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    out_dir = Path("runs/evaluation")
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- ViT ----
    print("=" * 60)
    print("Evaluating ViT...")
    vit_metrics, class_names = evaluate_vit(cfg)
    print(vit_metrics["report"])

    # ---- YOLO ----
    print("=" * 60)
    print("Evaluating YOLO (as classifier on crops)...")
    yolo_metrics = evaluate_yolo_as_classifier(cfg)

    # ---- Confusion matrices ----
    fig = plot_confusion_matrix(vit_metrics["confusion_matrix"], class_names, "ViT Confusion Matrix")
    fig.savefig(out_dir / "vit_confusion_matrix.png", dpi=120, bbox_inches="tight")

    fig = plot_confusion_matrix(yolo_metrics["confusion_matrix"], class_names, "YOLO Confusion Matrix")
    fig.savefig(out_dir / "yolo_confusion_matrix.png", dpi=120, bbox_inches="tight")

    # ---- Side-by-side comparison ----
    fig = plot_comparison(yolo_metrics, vit_metrics, str(out_dir / "comparison.png"))

    # ---- Summary ----
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    import numpy as np
    for name, m in [("YOLO", yolo_metrics), ("ViT", vit_metrics)]:
        print(f"\n{name}:")
        print(f"  Accuracy         : {m['accuracy']:.4f}")
        print(f"  Precision (macro): {np.mean(m['precision']):.4f}")
        print(f"  Recall (macro)   : {np.mean(m['recall']):.4f}")
        print(f"  F1 (macro)       : {np.mean(m['f1']):.4f}")
        print(f"  FPS              : {m['fps']:.1f}")

    print(f"\nOutputs saved to {out_dir}/")


if __name__ == "__main__":
    main()
