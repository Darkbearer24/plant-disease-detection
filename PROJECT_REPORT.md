# Plant Disease Detection Using YOLOv11 + Vision Transformer
### End-Semester Major Project Report — May 2026

**University:** University of Petroleum & Energy Studies, Dehradun
**Guide:** Dr. Rohitesh Kumar — AI Cluster, School of Computer Science

---

## Table of Contents

1. [Abstract](#1-abstract)
0. [Project Evolution: Midterm → Final](#0-project-evolution-midterm--final)
2. [Introduction & Motivation](#2-introduction--motivation)
3. [Research Papers Implemented](#3-research-papers-implemented)
4. [Dataset](#4-dataset)
5. [System Architecture](#5-system-architecture)
6. [Model 1 — YOLOv11 Object Detector](#6-model-1--yolov11-object-detector)
7. [Model 2 — Vision Transformer (ViT) Classifier](#7-model-2--vision-transformer-vit-classifier)
8. [Two-Model Pipeline Design](#8-two-model-pipeline-design)
9. [Training Pipeline](#9-training-pipeline)
10. [Web Dashboard](#10-web-dashboard)
11. [Evaluation & Metrics](#11-evaluation--metrics)
12. [Key Design Decisions](#12-key-design-decisions)
13. [Project Status & Completeness](#13-project-status--completeness)
14. [Results Summary](#14-results-summary)
15. [Presentation Talking Points](#15-presentation-talking-points)
16. [References](#16-references)

---

## 0. Project Evolution: Midterm → Final

Your **Mid-Semester Report (February 2026)** established:
- Implemented YOLOv11 object detector on PlantDoc
- Achieved bounding-box detection of diseased leaf regions
- Evaluation using mAP, Precision, Recall

Your midterm report explicitly stated in its **Conclusion / Future Work**:
> *"Future work will explore the integration of Vision Transformer (ViT) based architectures for improved feature extraction and object detection."*

**The final project delivers exactly that promised future work:**

| Midterm | Final (End-Sem) |
|---------|----------------|
| YOLOv11 detection only | YOLOv11 + ViT two-stage pipeline |
| Classification from YOLO head | Dedicated ViT-Base classifier (30 classes) |
| No interpretability | ViT attention heatmap visualization |
| CLI evaluation only | Full FastAPI web dashboard |
| No demo infrastructure | Pre-cached demo gallery, drag-drop UI |

**In your report and presentation, frame this as:** *"We proposed ViT integration as future work in our midterm submission. This final report presents the complete implementation, evaluation, and deployment of that vision."*

---

## 1. Abstract

We present a **two-stage plant disease detection system** that combines YOLOv11 object detection with a fine-tuned Vision Transformer (ViT) classifier. Given an image of a plant leaf, the system first localizes diseased regions using YOLO bounding boxes, then classifies the cropped region into one of **30 disease/healthy classes** across 13 plant species using ViT. The pipeline is deployed as a **FastAPI web dashboard** for live demonstration. The entire system runs locally — no cloud services, no paid APIs.

---

## 2. Introduction & Motivation

### Problem Statement

Plant diseases cause **20–40% of global crop yield losses** annually (FAO, 2021). Early, accurate identification is critical but requires expert agronomists — a scarce resource in developing countries. Automated visual detection from smartphone photos can democratize diagnosis.

### Why Two Models?

| Task | Best Tool | Why |
|------|-----------|-----|
| **Where** is the disease? (localization) | YOLO | Real-time bounding box regression, trained end-to-end on labeled boxes |
| **What** is the disease? (fine-grained classification) | Vision Transformer | Global self-attention captures subtle texture patterns across the entire patch; pre-trained representations transfer well |

A single model doing both often trades off classification accuracy for speed. Our pipeline **separates concerns**: YOLO is fast and spatially precise; ViT is accurate and interpretable via attention maps.

---

## 3. Research Papers Implemented

### Paper 1 — Vision Transformer (ViT)

> **"An Image is Worth 16×16 Words: Transformers for Image Recognition at Scale"**
> Dosovitskiy et al., Google Brain — ICLR 2021
> [https://arxiv.org/abs/2010.11929](https://arxiv.org/abs/2010.11929)

**What we implemented from this paper:**

| Concept | Where in our code |
|---------|-------------------|
| Patch tokenization (16×16 patches → 196 tokens) | `google/vit-base-patch16-224` backbone via HuggingFace `transformers` |
| [CLS] token for classification | Built into `ViTForImageClassification` |
| Pre-train → fine-tune transfer learning strategy | `build_vit()` in `models/vit_classifier.py` — loads ImageNet-21k weights, replaces head |
| Self-attention heatmap (CLS → patch attention row) | `get_attention_map()` in `models/vit_classifier.py` |
| AdamW optimizer + cosine LR schedule | `train_vit()` in `models/vit_classifier.py` |
| Fine-tuning on domain-specific data (PlantDoc crops) | `train_vit.py` with 20 epochs, lr=2e-5 |

**The Core ViT Idea (for your presentation):**
1. Take a 224×224 image
2. Cut into 16×16 pixel patches → 14×14 = **196 patches**
3. Flatten each patch: 16×16×3 = **768 numbers**
4. Project each to dimension D with a linear layer → 196 "tokens"
5. Prepend a learnable `[CLS]` token → sequence of **197 tokens**
6. Add **learnable 1D position embeddings** (so model knows spatial order)
7. Feed into **12 Transformer encoder layers** (ViT-Base)
8. Take the `[CLS]` token output → **MLP classification head** → 30 classes

**Why ViT over CNN?**
- CNNs have locality + translation equivariance hardcoded in. ViT has **global self-attention from layer 1** — every patch can attend to every other patch.
- With sufficient pre-training data (ImageNet-21k has 14M images), ViT learns better, more transferable features.
- ViT attention maps provide **visual interpretability** — you can see what the model is "looking at".

---

### Paper 2 — YOLOv11 (You Only Look Once, v11)

> **"Ultralytics YOLOv11"**
> Jocher et al., Ultralytics, 2024
> Official repo: [https://github.com/ultralytics/ultralytics](https://github.com/ultralytics/ultralytics)

**What we implemented:**

| Concept | Where in our code |
|---------|-------------------|
| Single-pass detection (anchor-free, multi-scale) | `YOLODetector` wrapping `ultralytics.YOLO` |
| Transfer learning from pretrained `yolo11n.pt` | `YOLODetector.__init__()` |
| YOLO training on custom dataset | `YOLODetector.train()` — 50 epochs, 640px input |
| Bounding box prediction + NMS | `predict_image()` returns `{box, conf, cls_id, cls_name}` |
| Confidence threshold filtering | `conf=0.25` default, tunable in `config.yaml` |

**YOLO's Key Idea:** Instead of proposing regions then classifying (two-stage like Faster R-CNN), YOLO divides the image into a grid and **predicts boxes and class probabilities simultaneously in one forward pass** — hence real-time inference speed.

---

## 4. Dataset

### PlantDoc

| Property | Value |
|----------|-------|
| Source | Roboflow Universe (CC BY 4.0) |
| Total images | ~2,569 |
| Format | YOLO (normalized bounding boxes in `.txt` files) |
| Classes | **30** (8 healthy + 22 diseased) |
| Species | Apple, Tomato, Potato, Corn, Grape, Pepper, Strawberry, Peach, Cherry, Blueberry, Squash, Soybean, Raspberry |
| Splits | `train / valid / test` |

### The 30 Classes

8 **healthy** leaf types and 22 **disease** types, including:
- Apple Scab, Apple Black Rot, Cedar Apple Rust
- Tomato Yellow Leaf Curl Virus, Tomato Mosaic Virus, Tomato Bacterial Spot
- Potato Early Blight, Potato Late Blight
- Corn Northern Leaf Blight, Corn Gray Leaf Spot, Corn Common Rust
- Grape Black Rot, Grape Esca, Grape Leaf Blight
- (and more...)

### Annotation Format (YOLO)

Each `.txt` label file contains one line per object:
```
class_id  x_center  y_center  width  height
```
All values normalized to `[0, 1]` relative to image dimensions.

---

## 5. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     INPUT: Leaf Image                        │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              STAGE 1: YOLOv11 Object Detector                │
│  • Input: full image (640×640)                              │
│  • Output: bounding boxes + class labels + confidence        │
│  • Trained: 50 epochs on PlantDoc train/valid splits         │
└───────────────────────────┬─────────────────────────────────┘
                            │  crop each detection box
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              STAGE 2: Vision Transformer (ViT)               │
│  • Input: cropped leaf region (224×224)                      │
│  • Backbone: google/vit-base-patch16-224 (ImageNet-21k)      │
│  • Fine-tuned: 20 epochs on PlantDoc crops                   │
│  • Output: top-K class probabilities + attention heatmap     │
└───────────────────────────┬─────────────────────────────────┘
                            │
                            ▼
┌─────────────────────────────────────────────────────────────┐
│              STAGE 3: FastAPI Web Dashboard                  │
│  • Displays annotated image with bounding boxes              │
│  • Shows ViT top-5 predictions with confidence bars          │
│  • Renders attention heatmap overlay                         │
│  • Demo gallery with pre-cached results (offline fallback)   │
└─────────────────────────────────────────────────────────────┘
```

### File Structure

```
MAJOR2/
├── config/config.yaml           ← All hyperparameters (one place, no hardcoding)
├── dataset/                     ← PlantDoc (train/valid/test + YOLO labels)
│   └── data.yaml                ← 30 class names, split paths
├── models/
│   ├── yolo_detector.py         ← YOLODetector: train(), predict_image(), render()
│   └── vit_classifier.py        ← build_vit(), train_vit(), ViTInference, get_attention_map()
├── utils/
│   ├── dataset_utils.py         ← extract_crops(), PlantCropDataset, split_dataset()
│   ├── metrics.py               ← evaluate_classifier(), plot_confusion_matrix(), plot_comparison()
│   └── visualization.py         ← DatasetVisualizer
├── train_yolo.py                ← YOLO training entry point
├── train_vit.py                 ← ViT fine-tuning entry point
├── evaluate.py                  ← YOLO vs ViT side-by-side comparison
├── predict.py                   ← CLI single-image prediction
├── prepare_demo.py              ← Pre-cache demo results for presentation day
└── web/
    ├── app.py                   ← FastAPI server (lazy model loading)
    ├── templates/index.html     ← Dark-theme dashboard UI
    └── static/css|js/           ← Styles and frontend JS
```

---

## 6. Model 1 — YOLOv11 Object Detector

### Architecture Summary

YOLOv11n (nano variant) — smallest and fastest in the YOLOv11 family.

| Component | Detail |
|-----------|--------|
| Backbone | CSPNet-style feature extractor |
| Neck | Path Aggregation Network (PAN) for multi-scale features |
| Head | Anchor-free, decoupled detection head |
| Input size | 640×640 |
| Parameters | ~2.6M (nano) |

### Training Configuration

```yaml
weights:  yolo11n.pt       # pretrained on COCO
epochs:   50
imgsz:    640
batch:    16
device:   GPU (auto-fallback to CPU)
```

### What YOLO Outputs

For each detected leaf region:
```python
{
    "box": [x1, y1, x2, y2],   # pixel coordinates
    "conf": 0.87,               # detection confidence
    "cls_id": 3,                # class index
    "cls_name": "Tomato Bacterial Spot"
}
```

### Code Implementation

**`models/yolo_detector.py`** — `YOLODetector` class:
- `train()` — wraps `ultralytics.YOLO.train()` with config-driven parameters
- `predict_image()` — returns list of detection dicts
- `render()` — draws bounding boxes on image using OpenCV
- `run_on_directory()` — batch visualization on test set
- `benchmark_fps()` — measures inference speed

---

## 7. Model 2 — Vision Transformer (ViT) Classifier

### Architecture Summary

`google/vit-base-patch16-224` — the ViT-Base variant from the original paper, pre-trained on ImageNet-21k (14M images).

| Component | Detail |
|-----------|--------|
| Input | 224×224 RGB image |
| Patch size | 16×16 pixels |
| Number of patches | 14×14 = 196 |
| Token dimension (D) | 768 |
| Transformer layers | 12 |
| Attention heads | 12 |
| MLP hidden dim | 3072 |
| Parameters | 86M |
| Pre-training data | ImageNet-21k (14M images, 21k classes) |
| Fine-tuning | PlantDoc crops (30 classes) |

### Transfer Learning Strategy

```
ImageNet-21k pre-trained ViT-Base
        │
        │  Replace classification head
        │  (21k classes → 30 classes)
        ▼
Fine-tune on PlantDoc crops
  - lr = 2e-5 (small, don't destroy pretrained features)
  - AdamW optimizer + weight_decay = 1e-4
  - CosineAnnealingLR scheduler
  - 20 epochs, early stopping (patience=5)
  - Data augmentation: RandomCrop, HFlip, VFlip, ColorJitter, Rotation
```

### Attention Heatmap Generation

Implemented from the ViT paper's interpretability analysis:

```python
# In get_attention_map() — models/vit_classifier.py

# 1. Register a forward hook on the LAST attention layer
last_attn = model.vit.encoder.layer[-1].attention.attention
handle = last_attn.register_forward_hook(_hook)

# 2. Run a forward pass
model(img_tensor)

# 3. Extract attention weights: shape (batch, heads, seq_len, seq_len)
attn = attentions[0]

# 4. Average over heads, take [CLS] → patch tokens row
attn = attn[0].mean(0)        # (seq+1, seq+1)
cls_attn = attn[0, 1:]        # (196,) — CLS attending to each patch

# 5. Reshape to 14×14 spatial grid, normalize to [0,1]
attn_map = cls_attn.reshape(14, 14)

# 6. Upsample to image size and overlay as JET colormap heatmap
```

This shows **where in the image the ViT is looking** when making its prediction — a key visual for your presentation.

### Data Augmentation (Training)

```python
transforms.RandomResizedCrop(224, scale=(0.7, 1.0))
transforms.RandomHorizontalFlip()
transforms.RandomVerticalFlip()
transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05)
transforms.RandomRotation(20)
transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
```

---

## 8. Two-Model Pipeline Design

### The Key Innovation: Crop-Based ViT Training

Rather than training ViT on full images (which contain background, multiple leaves, noise), we:

1. **Extract crops** using YOLO's ground-truth bounding box annotations
2. Train ViT **only on the cropped leaf region** — the exact same type of crop it will see at inference time
3. This creates **perfect train/inference alignment** — the ViT never sees background clutter

```python
# utils/dataset_utils.py — extract_crops()

for each image in train/valid:
    for each YOLO annotation box:
        crop = image.crop(box + 5% padding)
        save to: crops_dir/<class_name>/<split>_<stem>_<idx>.jpg
```

### Inference Flow

```
Input Image
     │
     ├──→ YOLODetector.predict_image(img)
     │         Returns: [{box:[x1,y1,x2,y2], conf, cls_name}, ...]
     │
     └──→ For each detected box:
               crop = img[y1:y2, x1:x2]
               ViTInference.predict(crop, top_k=5)
               Returns: [{class_name, confidence}, ...]  ← sorted desc
               
               ViTInference.attention_heatmap(crop)
               Returns: (overlay_PIL_image, attn_14x14_array)
```

### Why This Works Well

- YOLO handles the spatial generalization problem (where is the leaf?)
- ViT handles the fine-grained semantic problem (what disease pattern?)
- The two are trained **independently** so each can be optimized for its specific task
- **Crop-training** means ViT only ever sees what it needs to classify — no spatial confusion

---

## 9. Training Pipeline

### Step-by-Step Order

```
Step 1: python train_yolo.py
        └── Trains YOLOv11n on dataset/train + valid splits
        └── Saves best checkpoint → runs/detect/train/weights/best.pt
        └── ~50 epochs, ~30-90 min on GPU

Step 2: python train_vit.py --extract-crops
        └── Reads every YOLO .txt label file
        └── Crops each bounding box from the corresponding image
        └── Saves ~N crops organized in crops_dir/<class_name>/
        └── (One-time operation)

Step 3: python train_vit.py
        └── Loads google/vit-base-patch16-224 (downloads ~330MB once)
        └── Replaces 21k-class head with 30-class head
        └── Fine-tunes on PlantCropDataset with augmentation
        └── Saves best checkpoint → runs/vit/best.pt
        └── Saves class names → runs/vit/class_names.json

Step 4: python evaluate.py
        └── Evaluates ViT on test crops
        └── Evaluates YOLO (as classifier) on test crops
        └── Generates confusion matrices (PNG)
        └── Generates side-by-side comparison bar chart

Step 5: python prepare_demo.py
        └── Picks 20 diverse test images
        └── Pre-computes YOLO + ViT + attention heatmap for each
        └── Saves to demo_cache/ (instant loading on presentation day)

Step 6: python web/app.py
        └── Launches FastAPI server at http://localhost:8000
        └── Ready for live demo
```

### Configuration (config/config.yaml)

All hyperparameters in one file — no hardcoded values anywhere in the codebase:

```yaml
dataset:
  root:       "dataset"
  data_yaml:  "dataset/data.yaml"
  num_classes: 30
  crops_dir:  "dataset/crops"

yolo:
  weights:         "yolo11n.pt"      # pretrain
  trained_weights: "runs/detect/train/weights/best.pt"
  epochs: 50    imgsz: 640    batch: 16

vit:
  backbone:    "google/vit-base-patch16-224"
  checkpoint:  "runs/vit/best.pt"
  epochs: 20    batch_size: 32    lr: 2.0e-5
  weight_decay: 1.0e-4    patience: 5

web:
  confidence_threshold: 0.25
  host: "0.0.0.0"    port: 8000
```

---

## 10. Web Dashboard

### Technology Stack

| Layer | Technology |
|-------|-----------|
| Server | FastAPI + Uvicorn |
| Templating | Jinja2 |
| Frontend | Vanilla HTML/CSS/JS |
| Design | Dark theme, glassmorphism cards |
| Data transfer | Base64-encoded images in JSON |

### API Endpoints

| Route | Method | Description |
|-------|--------|-------------|
| `/` | GET | Main dashboard (HTML) |
| `/predict/yolo` | POST | YOLO detection only |
| `/predict/vit` | POST | ViT classification + attention heatmap |
| `/predict/both` | POST | Both models, side-by-side comparison |
| `/demo/gallery` | GET | Pre-cached demo manifest |
| `/demo/result/{id}` | GET | Load pre-cached result instantly |
| `/status` | GET | Model readiness check |

### Key Features

1. **Drag-and-drop upload** — user drops any leaf image
2. **Model status badges** — shows YOLO/ViT ready/not-trained at a glance
3. **Three prediction modes** — YOLO only, ViT only, or Compare Both
4. **Attention heatmap overlay** — visual explanation of ViT's decision
5. **Top-5 classification bar** — confidence percentages for top 5 classes
6. **Pre-cached demo gallery** — 20 pre-computed results load instantly even if GPU is unavailable
7. **Lazy model loading** — server starts instantly even if weights aren't trained yet

### Lazy Loading (Smart Design)

```python
# web/app.py

_yolo = None
_vit  = None

def get_yolo():
    global _yolo
    if _yolo is None:          # load only on first request
        _yolo = YOLODetector(weights=YOLO_WEIGHTS)
    return _yolo
```

This means the server starts in <1 second even without weights, and models load once on first use.

---

## 11. Evaluation & Metrics

### Metrics Computed (`utils/metrics.py`)

For both YOLO and ViT:

| Metric | Definition |
|--------|-----------|
| **Accuracy** | Correct predictions / total predictions |
| **Precision (per class)** | TP / (TP + FP) — of predicted positives, how many were correct |
| **Recall (per class)** | TP / (TP + FN) — of actual positives, how many were found |
| **F1 Score (per class)** | 2 × (P × R) / (P + R) — harmonic mean |
| **Macro-averaged F1** | Simple average of per-class F1 (treats all classes equally) |
| **Confusion Matrix** | 30×30 grid of actual vs predicted |
| **FPS** | Images processed per second (inference speed) |

### Outputs Generated by `evaluate.py`

```
runs/evaluation/
├── vit_confusion_matrix.png    ← 30×30 heatmap
├── yolo_confusion_matrix.png   ← 30×30 heatmap
└── comparison.png              ← Side-by-side bar chart: accuracy, precision, recall, F1, FPS
```

### Training Curves

Generated by `train_vit.py`:
```
runs/vit/
├── vit_training_curves.png    ← Train/val loss + accuracy over epochs
├── best.pt                    ← Best checkpoint (selected by val_acc)
└── class_names.json           ← 30 class name list
```

---

## 12. Key Design Decisions

### 1. Crop-Based ViT Training (Most Important)

**Decision:** Train ViT only on bounding-box crops, not full images.

**Why:** At inference time, YOLO gives us a crop. If ViT was trained on full images, there's a train/inference distribution mismatch — ViT would be confused by the sudden absence of background context. Training on crops means ViT sees **exactly what it will classify at inference**.

**Where:** `extract_crops()` in `utils/dataset_utils.py`, called by `train_vit.py --extract-crops`.

---

### 2. Freeze Nothing — Fine-Tune the Whole ViT

**Decision:** Fine-tune all 86M parameters (not just the head).

**Why:** The learning rate is very small (2e-5). With such a low LR and cosine decay, the backbone updates gently without forgetting its ImageNet features. Full fine-tuning consistently beats head-only on small datasets.

---

### 3. Early Stopping with Patience=5

**Decision:** Stop training when validation accuracy doesn't improve for 5 consecutive epochs.

**Why:** Prevents overfitting on the relatively small PlantDoc dataset (~2,500 images → ~N crops). Saves training time. Best checkpoint is always saved.

---

### 4. Independent Models (Not End-to-End)

**Decision:** Train YOLO and ViT separately. Do not back-propagate through both.

**Why:** Simpler, more modular. Each model can be independently updated, replaced, or swapped without retraining the other. YOLO can be retrained with more data without touching the classifier.

---

### 5. Demo Cache (Presentation Safety Net)

**Decision:** Pre-compute all predictions for 20 demo images before the presentation.

**Why:** On presentation day, GPU may be unavailable or slow. Pre-cached results load in milliseconds from JSON files. The live inference path still works if hardware cooperates.

---

## 13. Project Status & Completeness

### What Is Complete

| Component | Status |
|-----------|--------|
| `models/yolo_detector.py` | Complete — train, infer, render, benchmark |
| `models/vit_classifier.py` | Complete — build, train, infer, attention maps |
| `utils/dataset_utils.py` | Complete — crop extraction, Dataset class, augmentation |
| `utils/metrics.py` | Complete — all metrics, confusion matrix, comparison plots |
| `train_yolo.py` | Complete — config-driven, CLI args |
| `train_vit.py` | Complete — extract-crops flag, history plots |
| `evaluate.py` | Complete — YOLO vs ViT comparison |
| `predict.py` | Complete — CLI tool with `--show` and `--attention` flags |
| `prepare_demo.py` | Complete — pre-caches 20 diverse demo results |
| `web/app.py` | Complete — FastAPI server, all endpoints, lazy loading |
| `web/templates/index.html` | Complete — dark theme UI, drag-drop, gallery |
| `config/config.yaml` | Complete — all parameters centralized |
| `requirements.txt` | Complete |
| `download_weights.py` | Complete — pre-downloads ViT backbone offline |
| `colab_train.ipynb` | Complete — Colab training notebook |

### What Needs to Happen Before Presentation

```bash
# On the lab GPU (or Colab), in order:

python train_yolo.py                     # ~30-90 min on GPU
python train_vit.py --extract-crops      # one-time crop extraction
python train_vit.py                      # ~20-60 min on GPU
python evaluate.py                       # generates comparison plots
python prepare_demo.py                   # pre-cache 20 demo images
python web/app.py                        # start dashboard
```

The **code is 100% complete**. The trained weight files (`best.pt`) are what need to be generated by running training.

---

## 14. Results Summary

### Expected Performance Ranges (PlantDoc, 30 classes)

| Model | Top-1 Accuracy | Macro F1 | Speed |
|-------|---------------|----------|-------|
| YOLOv11n (detection) | ~65–75% mAP50 | — | 100+ FPS (GPU) |
| ViT-Base fine-tuned | ~75–85% | ~0.70–0.80 | 30–60 FPS (GPU) |

> Note: Actual numbers depend on your training run. Fill in real numbers after running `evaluate.py`.

### YOLO vs ViT — What to Expect

- **YOLO** is faster but slightly less accurate at classification (it's primarily a detector)
- **ViT** is more accurate at classification due to global attention over the crop
- **Combined pipeline**: spatial precision of YOLO + classification accuracy of ViT

---

## 15. Presentation Talking Points

### Opening Hook (30 seconds)

> "Farmers lose 20–40% of crops to disease every year. Expert agronomists can diagnose from a photo — but they're not always available. We built a system that does it automatically: take a photo of a leaf, and in under a second, it tells you exactly what disease it is."

### Architecture Slide (60 seconds)

> "We use a **two-model pipeline**. The first model — YOLOv11 — is a real-time object detector. It finds the diseased region in the image and draws a box around it. The second model — a Vision Transformer fine-tuned from Google's ViT-Base — takes that crop and classifies it into one of 30 categories. The key insight is: **YOLO answers WHERE, ViT answers WHAT.**"

### ViT Explanation (90 seconds)

> "The Vision Transformer comes from a 2021 Google Brain paper called *'An Image is Worth 16×16 Words.'* The idea is beautiful in its simplicity: cut the image into 16×16 patches, treat each patch like a word, feed the sequence into BERT — the same Transformer architecture used in language models. The model has never seen a convolution. No locality assumptions, no translation equivariance. It learns all of that from data. With 14 million ImageNet images as pre-training, it transfers remarkably well to plant disease classification."

### Attention Heatmap Demo (45 seconds)

> "Here's the coolest part. The Transformer has 12 layers of self-attention. We hook into the last attention layer and extract the attention weight from the classification token to each of the 196 patches. This tells us exactly which 16×16 tiles the model focused on when making its decision. On a diseased leaf — [show demo] — you can see it's looking at the lesion pattern, not the background."

### Results (30 seconds)

> "After training on the PlantDoc dataset — 2,569 images across 30 classes — YOLO achieves [X mAP50] detection accuracy, and the ViT classifier achieves [X%] top-1 accuracy on held-out test crops. The full pipeline runs at [Y] FPS on GPU."

### Live Demo Script

1. Open `http://localhost:8000`
2. Drag a pre-saved demo image (from `demo_images/`) onto the dashboard
3. Click "Compare Both"
4. Show: annotated image with bounding boxes (YOLO), top-5 predictions (ViT), attention heatmap
5. If live inference fails, click any pre-cached result from the gallery — identical output, instant load

### Likely Questions & Answers

**Q: Why not just use YOLO for classification too?**
> YOLO is optimized for fast spatial detection. For fine-grained classification between 30 visually similar disease categories, ViT's global attention and deep pre-training significantly outperform YOLO's classification head.

**Q: What is the attention mechanism?**
> Self-attention computes a score between every pair of patches: `softmax(Q·Kᵀ / √d) × V`. The [CLS] token aggregates information from all 196 patches through 12 layers. By epoch 12, it has built a rich representation of the whole image to classify from.

**Q: Why use a pre-trained ViT instead of training from scratch?**
> PlantDoc has ~2,500 images. ViT-Base has 86M parameters. Training from scratch would severely overfit. Transfer learning from ImageNet-21k (14M images) gives the model generalizable visual features (edges, textures, shapes) before we specialize it for plant diseases. This is called the **pre-train → fine-tune paradigm** introduced in the original ViT paper.

**Q: Why YOLOv11 specifically?**
> YOLOv11 is the current state-of-the-art in the YOLO family (2024), offering better accuracy/speed tradeoffs than YOLOv8. The nano variant (yolo11n) keeps inference fast while being trainable on modest hardware.

**Q: What's the limitation of your approach?**
> (a) Dataset size: 2,569 images across 30 classes is relatively small. More data would improve both models. (b) The two-model pipeline means YOLO must detect the leaf first — if YOLO misses the diseased region (false negative), ViT never sees it. (c) ViT's O(N²) attention doesn't scale to very high resolution images without modification.

---

## 16. References

1. **Dosovitskiy, A., et al.** (2021). *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.* ICLR 2021. [arXiv:2010.11929](https://arxiv.org/abs/2010.11929)

2. **Jocher, G., et al.** (2024). *Ultralytics YOLOv11.* [GitHub](https://github.com/ultralytics/ultralytics)

3. **Singh, D., et al.** (2020). *PlantDoc: A Dataset for Visual Plant Disease Detection.* CODS-COMAD 2020. *(Original dataset paper)*

4. **Vaswani, A., et al.** (2017). *Attention Is All You Need.* NeurIPS 2017. *(Foundation of Transformers)*

5. **Redmon, J., et al.** (2016). *You Only Look Once: Unified, Real-Time Object Detection.* CVPR 2016. *(YOLO original paper)*

6. **He, K., et al.** (2020). *Deep Residual Learning for Image Recognition.* *(Context: what ViT outperformed)*

7. **HuggingFace Transformers** — `google/vit-base-patch16-224` model card. [HuggingFace Hub](https://huggingface.co/google/vit-base-patch16-224)

---

*Prepared for the End-Semester Major Project Presentation — May 11, 2026*
*Dataset: PlantDoc (CC BY 4.0) · Models: YOLOv11 + ViT-Base · Framework: PyTorch + FastAPI*
