"""Generate all graphs for the project report."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from pathlib import Path

Path("report_assets").mkdir(exist_ok=True)

# ── 1. YOLO training curves ───────────────────────────────────────────────────
csv = pd.read_csv("runs/detect/runs/detect/train/results.csv")
csv.columns = csv.columns.str.strip()

fig, axes = plt.subplots(2, 3, figsize=(16, 9))
fig.suptitle("YOLOv11 Training Curves (50 Epochs)", fontsize=15, fontweight='bold')

plots = [
    ("train/box_loss",       "Box Loss",       "tab:blue"),
    ("train/cls_loss",       "Class Loss",      "tab:orange"),
    ("train/dfl_loss",       "DFL Loss",        "tab:green"),
    ("metrics/mAP50(B)",     "mAP@50",          "tab:red"),
    ("metrics/mAP50-95(B)",  "mAP@50-95",       "tab:purple"),
    ("metrics/precision(B)", "Precision",       "tab:brown"),
]
for ax, (col, title, color) in zip(axes.flat, plots):
    ax.plot(csv["epoch"], csv[col], color=color, linewidth=2)
    ax.set_title(title, fontweight='bold')
    ax.set_xlabel("Epoch")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, 50)

plt.tight_layout()
plt.savefig("report_assets/yolo_training_curves.png", dpi=150, bbox_inches='tight')
plt.close()
print("1/8 yolo_training_curves.png")

# ── 2. ViT training curves ────────────────────────────────────────────────────
epochs     = list(range(1, 21))
train_loss = [2.5099,1.4980,1.0880,0.8600,0.7127,0.6143,0.5224,0.4541,0.3971,0.3415,
              0.3116,0.2720,0.2471,0.2300,0.2151,0.2013,0.1858,0.1853,0.1821,0.1760]
val_loss   = [1.9040,1.3903,1.1199,0.9868,0.8899,0.8459,0.8012,0.7681,0.7473,0.7486,
              0.7309,0.7251,0.7178,0.7063,0.7069,0.6941,0.6931,0.6874,0.6892,0.6908]
train_acc  = [0.3354,0.5960,0.6958,0.7552,0.8045,0.8290,0.8526,0.8735,0.8903,0.9082,
              0.9204,0.9304,0.9352,0.9406,0.9455,0.9535,0.9576,0.9592,0.9594,0.9631]
val_acc    = [0.5304,0.6470,0.7045,0.7236,0.7460,0.7524,0.7636,0.7700,0.7819,0.7827,
              0.7804,0.7859,0.7899,0.7923,0.7891,0.7915,0.7955,0.7923,0.7899,0.7907]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle("ViT-Base Fine-Tuning Curves (20 Epochs)", fontsize=14, fontweight='bold')

ax1.plot(epochs, train_loss, 'b-o', markersize=4, label='Train Loss')
ax1.plot(epochs, val_loss,   'r-o', markersize=4, label='Val Loss')
ax1.axvline(17, color='green', linestyle='--', alpha=0.8, label='Best checkpoint (ep.17)')
ax1.set_title("Cross-Entropy Loss")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(epochs, [a*100 for a in train_acc], 'b-o', markersize=4, label='Train Acc')
ax2.plot(epochs, [a*100 for a in val_acc],   'r-o', markersize=4, label='Val Acc')
ax2.axvline(17, color='green', linestyle='--', alpha=0.8, label='Best checkpoint (ep.17)')
ax2.axhline(90, color='orange', linestyle=':', alpha=0.7, label='90% reference')
ax2.set_title("Accuracy")
ax2.set_xlabel("Epoch")
ax2.set_ylabel("Accuracy (%)")
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim(30, 100)

plt.tight_layout()
plt.savefig("report_assets/vit_training_curves.png", dpi=150, bbox_inches='tight')
plt.close()
print("2/8 vit_training_curves.png")

# ── 3. ViT per-class F1 ───────────────────────────────────────────────────────
classes = [
    'Apple leaf','Apple rust leaf','Apple Scab','Bell pepper leaf',
    'Bell pepper spot','Blueberry leaf','Cherry leaf','Corn Gray spot',
    'Corn leaf blight','Corn rust leaf','Grape leaf','Grape black rot',
    'Peach leaf','Potato leaf','Potato early blight','Potato late blight',
    'Raspberry leaf','Soyabean leaf','Soybean leaf','Squash mildew',
    'Strawberry leaf','Tomato Early blight','Tomato leaf','Tomato bacterial spot',
    'Tomato late blight','Tomato mosaic virus','Tomato yellow virus',
    'Tomato mold leaf','Tomato Septoria','Spider mites'
]
f1 = [0.92,0.94,0.88,0.87,0.93,0.96,0.90,0.74,0.93,0.95,0.96,0.98,
      0.92,0.79,0.31,0.77,0.98,0.96,0.89,0.98,0.99,0.75,0.90,0.87,
      0.74,0.82,0.75,0.82,0.88,0.00]

colors = ['#e74c3c' if v < 0.75 else '#f39c12' if v < 0.85 else '#27ae60' for v in f1]
fig, ax = plt.subplots(figsize=(14, 9))
bars = ax.barh(classes, f1, color=colors, edgecolor='white', linewidth=0.5)
ax.axvline(0.836, color='navy', linestyle='--', linewidth=1.5, label='Macro F1 = 0.836')
ax.set_xlabel('F1 Score', fontsize=12)
ax.set_title('ViT Per-Class F1 Score (30 Classes)', fontsize=14, fontweight='bold')
ax.set_xlim(0, 1.1)
ax.legend(fontsize=11)
ax.grid(True, axis='x', alpha=0.3)
for bar, val in zip(bars, f1):
    ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
            f'{val:.2f}', va='center', fontsize=8.5)
red_p    = mpatches.Patch(color='#e74c3c', label='F1 < 0.75 (underperforming)')
orange_p = mpatches.Patch(color='#f39c12', label='F1 0.75–0.85 (moderate)')
green_p  = mpatches.Patch(color='#27ae60', label='F1 > 0.85 (strong)')
ax.legend(handles=[green_p, orange_p, red_p], loc='lower right', fontsize=9)
plt.tight_layout()
plt.savefig("report_assets/vit_per_class_f1.png", dpi=150, bbox_inches='tight')
plt.close()
print("3/8 vit_per_class_f1.png")

# ── 4. YOLO per-class mAP50 ──────────────────────────────────────────────────
yolo_cls = [
    'Apple Scab','Apple leaf','Apple rust','Bell pepper spot','Bell pepper leaf',
    'Blueberry leaf','Cherry leaf','Corn Gray spot','Corn blight','Corn rust',
    'Peach leaf','Potato late blight','Potato leaf','Raspberry leaf',
    'Soyabean leaf','Squash mildew','Strawberry leaf','Tomato Early blight',
    'Tomato Septoria','Tomato bacterial spot','Tomato late blight',
    'Tomato mosaic virus','Tomato yellow virus','Tomato leaf','Tomato mold',
    'Grape black rot','Grape leaf'
]
yolo_map50 = [0.454,0.837,0.518,0.300,0.234,0.808,0.542,0.473,0.659,0.995,
              0.753,0.154,0.438,0.707,0.713,0.731,0.842,0.325,0.666,0.690,
              0.212,0.414,0.476,0.479,0.174,0.960,0.378]

colors_y = ['#e74c3c' if v < 0.4 else '#f39c12' if v < 0.65 else '#27ae60' for v in yolo_map50]
fig, ax = plt.subplots(figsize=(14, 8))
bars = ax.barh(yolo_cls, yolo_map50, color=colors_y, edgecolor='white', linewidth=0.5)
ax.axvline(0.553, color='navy', linestyle='--', linewidth=1.5, label='Overall mAP50 = 0.553')
ax.set_xlabel('mAP@50', fontsize=12)
ax.set_title('YOLO Per-Class mAP@50', fontsize=14, fontweight='bold')
ax.set_xlim(0, 1.1)
ax.legend(fontsize=11)
ax.grid(True, axis='x', alpha=0.3)
for bar, val in zip(bars, yolo_map50):
    ax.text(val + 0.01, bar.get_y() + bar.get_height()/2,
            f'{val:.3f}', va='center', fontsize=8.5)
plt.tight_layout()
plt.savefig("report_assets/yolo_per_class_map.png", dpi=150, bbox_inches='tight')
plt.close()
print("4/8 yolo_per_class_map.png")

# ── 5. Pipeline accuracy comparison ──────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 6))
configs = ['YOLO only\n(as classifier)', 'ViT on full image\n(no TTA)',
           'ViT on YOLO crop\n+ TTA (final)', 'ViT on clean crops\n(val benchmark)']
accs    = [60, 71, 75, 90]
bar_colors = ['#e74c3c', '#f39c12', '#2ecc71', '#27ae60']
bars = ax.bar(configs, accs, color=bar_colors, edgecolor='white', linewidth=0.5, width=0.5)
for bar, val in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.8,
            f'{val}%', ha='center', va='bottom', fontweight='bold', fontsize=13)
ax.set_ylabel('Accuracy (%)', fontsize=12)
ax.set_title('Pipeline Accuracy Comparison (100 Real Test Images)', fontsize=13, fontweight='bold')
ax.set_ylim(0, 100)
ax.grid(True, axis='y', alpha=0.3)
ax.axhline(75, color='navy', linestyle='--', alpha=0.4, label='Final system (75%)')
ax.legend(fontsize=10)
plt.tight_layout()
plt.savefig("report_assets/pipeline_comparison.png", dpi=150, bbox_inches='tight')
plt.close()
print("5/8 pipeline_comparison.png")

# ── 6. Model summary metrics ──────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Model Performance Summary', fontsize=14, fontweight='bold')

yolo_metrics = ['Precision', 'Recall', 'mAP@50', 'mAP@50-95']
yolo_vals    = [0.534, 0.561, 0.553, 0.392]
axes[0].barh(yolo_metrics, yolo_vals, color='#3498db', edgecolor='white')
axes[0].set_xlim(0, 0.75)
axes[0].set_title('YOLOv11-nano Detection Metrics', fontweight='bold')
for i, v in enumerate(yolo_vals):
    axes[0].text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=11)
axes[0].grid(True, axis='x', alpha=0.3)

vit_metrics = ['Accuracy', 'Precision (macro)', 'Recall (macro)', 'F1 (macro)']
vit_vals    = [0.900, 0.871, 0.826, 0.836]
axes[1].barh(vit_metrics, vit_vals, color='#e67e22', edgecolor='white')
axes[1].set_xlim(0, 1.05)
axes[1].set_title('ViT-Base Classification Metrics', fontweight='bold')
for i, v in enumerate(vit_vals):
    axes[1].text(v + 0.005, i, f'{v:.3f}', va='center', fontsize=11)
axes[1].grid(True, axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig("report_assets/model_summary.png", dpi=150, bbox_inches='tight')
plt.close()
print("6/8 model_summary.png")

# ── 7. Dataset class distribution ────────────────────────────────────────────
crop_counts = {
    'Blueberry leaf': 816, 'Tomato leaf': 759, 'Peach leaf': 579,
    'Raspberry leaf': 539, 'Strawberry leaf': 443, 'Tomato Septoria': 402,
    'Tomato bacterial spot': 373, 'Corn leaf blight': 356, 'Bell pepper spot': 312,
    'Potato late blight': 301, 'Tomato mold leaf': 279, 'Squash mildew': 248,
    'Bell pepper leaf': 248, 'Soyabean leaf': 246, 'Apple leaf': 237,
    'Potato leaf': 235, 'Tomato yellow virus': 225, 'Cherry leaf': 220,
    'Grape black rot': 205, 'Tomato Early blight': 193, 'Apple rust leaf': 167,
    'Apple Scab': 158, 'Grape leaf': 125, 'Corn rust leaf': 117,
    'Tomato mosaic virus': 204, 'Corn Gray spot': 72,
    'Soybean leaf': 15, 'Potato early blight': 11, 'Spider mites': 2,
    'Tomato late blight': 266
}
sorted_items = sorted(crop_counts.items(), key=lambda x: x[1], reverse=True)
names, counts = zip(*sorted_items)
colors_d = ['#e74c3c' if c < 50 else '#f39c12' if c < 150 else '#27ae60' for c in counts]

fig, ax = plt.subplots(figsize=(14, 9))
ax.barh(names, counts, color=colors_d, edgecolor='white')
ax.set_xlabel('Number of Crops', fontsize=12)
ax.set_title('Dataset Class Distribution — 8,353 ViT Training Crops', fontsize=14, fontweight='bold')
mean_val = np.mean(counts)
ax.axvline(mean_val, color='navy', linestyle='--', label=f'Mean = {mean_val:.0f} crops')
for i, (name, count) in enumerate(zip(names, counts)):
    ax.text(count + 5, i, str(count), va='center', fontsize=8.5)
ax.legend(fontsize=10)
ax.grid(True, axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig("report_assets/dataset_distribution.png", dpi=150, bbox_inches='tight')
plt.close()
print("7/8 dataset_distribution.png")

# ── 8. Improvement waterfall ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(10, 5))
steps  = ['Baseline\nYOLO only', 'Switch to ViT\nfull image', 'Fix pipeline\n(YOLO crop)', 'Add TTA\n(final)']
values = [60, 71, 73, 75]
deltas = [60, 11, 2, 2]
bottoms = [0, 60, 71, 73]
step_colors = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71']

bars = ax.bar(steps, deltas, bottom=bottoms, color=step_colors, edgecolor='white', width=0.5)
for bar, val, bot in zip(bars, deltas, bottoms):
    ax.text(bar.get_x() + bar.get_width()/2, bot + val/2,
            f'+{val}%' if val < 60 else f'{val}%',
            ha='center', va='center', fontweight='bold', fontsize=12, color='white')
ax.set_ylabel('Cumulative Accuracy (%)', fontsize=12)
ax.set_title('Accuracy Improvement — Step by Step', fontsize=13, fontweight='bold')
ax.set_ylim(0, 85)
ax.grid(True, axis='y', alpha=0.3)
for x, v in zip(range(len(steps)), values):
    ax.text(x, v + 1, f'{v}%', ha='center', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig("report_assets/improvement_waterfall.png", dpi=150, bbox_inches='tight')
plt.close()
print("8/8 improvement_waterfall.png")

print("\nAll graphs saved to report_assets/")
