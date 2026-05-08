"""
Evaluation metrics for YOLO vs ViT comparison.
"""

import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import torch
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_recall_fscore_support,
)


def evaluate_classifier(model, dataloader, class_names: list[str], device: str = "cuda"):
    """
    Run model on dataloader, return dict with accuracy, per-class metrics,
    confusion matrix, and inference FPS.
    """
    model.eval()
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(dev)

    all_preds, all_labels = [], []
    start = time.perf_counter()

    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs = imgs.to(dev)
            out = model(imgs)
            logits = out.logits if hasattr(out, "logits") else out
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())

    elapsed = time.perf_counter() - start
    n_samples = len(all_labels)
    fps = n_samples / elapsed

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    acc = accuracy_score(all_labels, all_preds)
    precision, recall, f1, support = precision_recall_fscore_support(
        all_labels, all_preds, average=None, labels=list(range(len(class_names))), zero_division=0
    )
    cm = confusion_matrix(all_labels, all_preds, labels=list(range(len(class_names))))
    report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)

    return {
        "accuracy": acc,
        "fps": fps,
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "support": support.tolist(),
        "confusion_matrix": cm,
        "report": report,
        "class_names": class_names,
    }


def plot_confusion_matrix(cm: np.ndarray, class_names: list[str], title: str = "Confusion Matrix"):
    fig, ax = plt.subplots(figsize=(16, 14))
    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.colorbar(im, ax=ax)

    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels(class_names, fontsize=7)

    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]),
                    ha="center", va="center", fontsize=6,
                    color="white" if cm[i, j] > thresh else "black")

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(title)
    plt.tight_layout()
    return fig


def plot_comparison(yolo_metrics: dict, vit_metrics: dict, save_path: str = None):
    """Bar chart comparing YOLO vs ViT macro-averaged metrics + speed."""
    labels = ["Accuracy", "Precision (macro)", "Recall (macro)", "F1 (macro)"]

    def macro(m):
        return [
            m["accuracy"],
            float(np.mean(m["precision"])),
            float(np.mean(m["recall"])),
            float(np.mean(m["f1"])),
        ]

    y_vals = macro(yolo_metrics)
    v_vals = macro(vit_metrics)

    x = np.arange(len(labels))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    bars1 = ax1.bar(x - width / 2, y_vals, width, label="YOLO", color="#ef5350")
    bars2 = ax1.bar(x + width / 2, v_vals, width, label="ViT",  color="#42a5f5")

    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=15, ha="right")
    ax1.set_ylim(0, 1.1)
    ax1.set_ylabel("Score")
    ax1.set_title("Classification Metrics — YOLO vs ViT")
    ax1.legend()

    for bar in bars1:
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{bar.get_height():.3f}", ha="center", fontsize=8)
    for bar in bars2:
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                 f"{bar.get_height():.3f}", ha="center", fontsize=8)

    # Speed comparison
    speeds = [yolo_metrics.get("fps", 0), vit_metrics.get("fps", 0)]
    ax2.bar(["YOLO", "ViT"], speeds, color=["#ef5350", "#42a5f5"], width=0.4)
    ax2.set_ylabel("Images / second")
    ax2.set_title("Inference Speed")
    for i, v in enumerate(speeds):
        ax2.text(i, v + 0.5, f"{v:.1f} FPS", ha="center", fontsize=10)

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
