# Plant Disease Detection — YOLOv11 + Vision Transformer

A two-model pipeline for detecting and classifying plant diseases in leaf images.
Built on the PlantDoc dataset (2,569 images, 30 classes across 13 plant species).

**Pipeline:**
1. **YOLOv11** — detects diseased regions and draws bounding boxes
2. **Vision Transformer (ViT)** — classifies the disease from each detected crop
3. **FastAPI dashboard** — web UI for live demo and image upload

---

## Requirements

- Python 3.10 or higher
- NVIDIA GPU recommended (CUDA 11.8+); CPU fallback works but is slow
- ~4 GB disk space for dataset + model weights

```bash
pip install -r requirements.txt
```

---

## Quick Start

### One-command option

```bash
python run_pipeline.py
```

Runs every step automatically in order. Already-completed steps are skipped on
re-runs (so it's safe to stop and restart). Options:

```bash
python run_pipeline.py --yolo-epochs 30 --vit-epochs 10   # faster training
python run_pipeline.py --skip-download                     # backbone already cached
python run_pipeline.py --skip-yolo --skip-vit              # skip training, just eval + demo
python run_pipeline.py --launch-web                        # open dashboard when done
```

---

### Step-by-step option

### Step 0 — Pre-download ViT backbone (do this once, before going offline)

```bash
python download_weights.py
```

This caches the `google/vit-base-patch16-224` backbone (~330 MB) locally at
`runs/vit/backbone_cache/` so training works without internet access.

---

### Step 1 — Train YOLO

```bash
python train_yolo.py
# Or with custom parameters:
python train_yolo.py --epochs 100 --imgsz 640 --batch 16
```

Trained weights are saved to `runs/detect/train/weights/best.pt`.

---

### Step 2 — Extract crops (one-time, before ViT training)

```bash
python train_vit.py --extract-crops
```

This reads YOLO bounding-box labels from `dataset/train/labels/` and
`dataset/valid/labels/` and saves cropped leaf patches to `dataset/crops/`.

---

### Step 3 — Train ViT

```bash
python train_vit.py
# Or with custom parameters:
python train_vit.py --epochs 20 --batch 32
```

Checkpoint saved to `runs/vit/best.pt`. Class names saved to
`runs/vit/class_names.json`.

---

### Step 4 — Evaluate both models

```bash
python evaluate.py
```

Prints accuracy / F1 / confusion matrix for ViT and YOLO detection metrics.
Plots saved to `runs/evaluation/`.

---

### Step 5 — Single-image prediction (CLI)

```bash
python predict.py path/to/leaf.jpg
python predict.py path/to/leaf.jpg --show           # display result window
python predict.py path/to/leaf.jpg --show --attention  # also show ViT heatmap
```

---

### Step 6 — Pre-cache demo gallery (optional, for web dashboard)

```bash
python prepare_demo.py
```

Randomly selects 20 test images, runs both models, and saves results to
`demo_cache/` so the web dashboard can show them instantly.

---

### Step 7 — Launch the web dashboard

```bash
python web/app.py
```

Open `http://localhost:8000` in a browser.

Features:
- Upload any leaf image and run YOLO, ViT, or both
- Attention heatmap overlay showing which regions influenced the ViT prediction
- Demo gallery of pre-cached results (if `prepare_demo.py` was run)
- Model status badges (shows green/red depending on whether weights exist)

---

## Project Structure

```
MAJOR2/
├── config/
│   └── config.yaml          # All tunable parameters (epochs, paths, thresholds)
├── dataset/
│   ├── data.yaml            # Class names and split paths (30 classes)
│   ├── train/               # Training images + YOLO labels
│   ├── valid/               # Validation split
│   └── test/                # Test split
├── models/
│   ├── yolo_detector.py     # YOLODetector: train, predict, render
│   └── vit_classifier.py    # ViT: build, train, inference, attention maps
├── utils/
│   ├── dataset_utils.py     # Crop extraction, PlantCropDataset
│   ├── metrics.py           # Confusion matrix, comparison plots
│   └── visualization.py     # Dataset visualization (bboxes, class distribution)
├── web/
│   ├── app.py               # FastAPI server (lazy model loading)
│   ├── templates/index.html # Dashboard UI
│   └── static/              # CSS and JS assets
├── train_yolo.py            # Entry point: train YOLO
├── train_vit.py             # Entry point: train ViT
├── evaluate.py              # Entry point: evaluate both models
├── predict.py               # Entry point: CLI single-image prediction
├── prepare_demo.py          # Pre-cache demo gallery for web UI
├── download_weights.py      # Download ViT backbone before going offline
├── yolo11n.pt               # Pre-trained YOLOv11-nano base weights
├── requirements.txt         # Python dependencies
└── _archive/                # Old notebooks and reference docs (not needed to run)
```

---

## Configuration

All parameters are in `config/config.yaml`. Key fields:

| Key | Default | Description |
|-----|---------|-------------|
| `yolo.epochs` | 50 | Training epochs for YOLO |
| `yolo.imgsz` | 640 | Input image size |
| `yolo.batch` | 16 | Batch size |
| `yolo.trained_weights` | `runs/detect/train/weights/best.pt` | Path to trained YOLO checkpoint |
| `vit.epochs` | 20 | Training epochs for ViT |
| `vit.batch_size` | 32 | Batch size |
| `vit.lr` | 2e-5 | Learning rate |
| `vit.patience` | 5 | Early stopping patience |
| `vit.checkpoint` | `runs/vit/best.pt` | Path to trained ViT checkpoint |
| `web.port` | 8000 | Dashboard port |
| `web.confidence_threshold` | 0.25 | YOLO detection confidence cutoff |

---

## Dataset

**PlantDoc** — YOLO format, 30 classes:
- 8 healthy leaf types
- 22 disease types across Apple, Tomato, Potato, Corn, Grape, and 8 other species

Label format per line: `class_id x_center y_center width height` (normalized 0–1).

---

## Key Design Decisions

- **ViT trained on crops, not full images** — the ViT sees only the bounding-box
  crop from YOLO labels, which forces it to learn disease-specific features
  rather than background context.
- **Lazy model loading** — `web/app.py` loads models on first request. The
  server starts instantly even if weights are not yet trained.
- **Attention heatmap** — hooks the last ViT attention layer, averages over
  heads, and overlays the `[CLS]→patch` attention scores as a heatmap. Useful
  for explaining which leaf region triggered the prediction.
- **All paths relative** — `config.yaml` uses relative paths; every script
  resolves paths from its own `__file__` location. Works after unzipping on
  any machine.
