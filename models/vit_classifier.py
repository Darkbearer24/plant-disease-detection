"""
Vision Transformer (ViT) classifier for PlantDoc disease classification.

Uses google/vit-base-patch16-224 from HuggingFace (free).
Fine-tunes on the 30-class PlantDoc crop dataset.

Two-stage pipeline at inference time:
  1. YOLO detects and crops the leaf region from the input image.
  2. ViT classifies the crop into one of 30 disease categories.
"""

import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
from PIL import Image
from torchvision import transforms
from transformers import ViTForImageClassification, ViTImageProcessor


BACKBONE = "google/vit-base-patch16-224"
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_vit(num_classes: int, backbone: str = BACKBONE, cache_dir: str = None):
    """Return a ViT model with a freshly initialized classification head.

    cache_dir: path to a local HuggingFace cache directory (created by
    download_weights.py).  When provided and the directory exists, the model
    is loaded from that cache with local_files_only=True so no internet is
    needed (important for campus / presentation environments).
    """
    kwargs = {}
    if cache_dir and Path(cache_dir).exists():
        kwargs["cache_dir"] = cache_dir
        kwargs["local_files_only"] = True

    model = ViTForImageClassification.from_pretrained(
        backbone,
        num_labels=num_classes,
        ignore_mismatched_sizes=True,   # replaces the pretrained head
        **kwargs,
    )
    return model


# ---------------------------------------------------------------------------
# Inference transform (same as ViTImageProcessor, but pure torchvision)
# ---------------------------------------------------------------------------

def inference_transform(imgsz: int = 224):
    return transforms.Compose([
        transforms.Resize((imgsz, imgsz)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


# ---------------------------------------------------------------------------
# Attention map extraction
# ---------------------------------------------------------------------------

def get_attention_map(model, img_tensor: torch.Tensor) -> np.ndarray:
    """
    Extract a spatial attention map from the last ViT attention layer.

    img_tensor: (1, 3, 224, 224) on the same device as model.
    Returns: (H_patches, W_patches) float32 numpy array, values in [0, 1].
    """
    model.eval()
    attentions = []

    def _hook(module, input, output):
        # output is (batch, heads, seq_len, seq_len)
        attentions.append(output.detach())

    # Register hook on the last attention layer
    last_attn = model.vit.encoder.layer[-1].attention.attention
    handle = last_attn.register_forward_hook(_hook)

    with torch.no_grad():
        model(img_tensor)

    handle.remove()

    if not attentions:
        return np.zeros((14, 14), dtype=np.float32)

    attn = attentions[0]               # (1, heads, seq+1, seq+1)
    # Average over heads, take [CLS] → patch tokens
    attn = attn[0].mean(0)             # (seq+1, seq+1)
    cls_attn = attn[0, 1:]             # (num_patches,)  ignore [CLS]→[CLS]
    n = int(cls_attn.shape[0] ** 0.5)
    attn_map = cls_attn.reshape(n, n).cpu().numpy()
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
    return attn_map


def overlay_attention(pil_image: Image.Image, attn_map: np.ndarray) -> Image.Image:
    """Overlay a heatmap on the PIL image and return a new PIL image."""
    import cv2
    img_np = np.array(pil_image.convert("RGB"))
    H, W = img_np.shape[:2]

    # Resize attention map to image size
    attn_resized = cv2.resize(attn_map, (W, H), interpolation=cv2.INTER_LINEAR)
    heatmap = cv2.applyColorMap((attn_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    blended = (0.55 * img_np + 0.45 * heatmap).astype(np.uint8)
    return Image.fromarray(blended)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_vit(
    model,
    train_loader,
    val_loader,
    epochs: int = 20,
    lr: float = 2e-5,
    weight_decay: float = 1e-4,
    patience: int = 5,
    save_path: str = "runs/vit/best.pt",
    device: str = "cuda",
):
    dev = torch.device(device if torch.cuda.is_available() else "cpu")
    model.to(dev)

    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-7)
    criterion = nn.CrossEntropyLoss()

    Path(save_path).parent.mkdir(parents=True, exist_ok=True)

    best_val_acc = 0.0
    no_improve = 0
    history = {"train_loss": [], "val_loss": [], "train_acc": [], "val_acc": []}

    for epoch in range(1, epochs + 1):
        # ---- Train ----
        model.train()
        train_loss, train_correct, train_total = 0.0, 0, 0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(dev), labels.to(dev)
            optimizer.zero_grad()
            out = model(imgs)
            logits = out.logits if hasattr(out, "logits") else out
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)
            train_correct += (logits.argmax(1) == labels).sum().item()
            train_total += imgs.size(0)

        scheduler.step()

        # ---- Validate ----
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs, labels = imgs.to(dev), labels.to(dev)
                out = model(imgs)
                logits = out.logits if hasattr(out, "logits") else out
                loss = criterion(logits, labels)
                val_loss += loss.item() * imgs.size(0)
                val_correct += (logits.argmax(1) == labels).sum().item()
                val_total += imgs.size(0)

        t_loss = train_loss / train_total
        v_loss = val_loss / val_total
        t_acc  = train_correct / train_total
        v_acc  = val_correct / val_total

        history["train_loss"].append(t_loss)
        history["val_loss"].append(v_loss)
        history["train_acc"].append(t_acc)
        history["val_acc"].append(v_acc)

        print(f"Epoch {epoch:02d}/{epochs}  "
              f"train_loss={t_loss:.4f}  train_acc={t_acc:.4f}  "
              f"val_loss={v_loss:.4f}  val_acc={v_acc:.4f}")

        if v_acc > best_val_acc:
            best_val_acc = v_acc
            torch.save(model.state_dict(), save_path)
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping at epoch {epoch}. Best val_acc={best_val_acc:.4f}")
                break

    print(f"Best checkpoint saved → {save_path}")
    return history


# ---------------------------------------------------------------------------
# Inference
# ---------------------------------------------------------------------------

class ViTInference:
    """Load a saved ViT checkpoint and run inference."""

    def __init__(self, checkpoint: str, class_names: list[str],
                 backbone: str = BACKBONE, device: str = "cuda"):
        self.class_names = class_names
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.transform = inference_transform()

        # Prefer the local backbone cache next to the checkpoint so the model
        # loads offline (critical for campus / presentation environments).
        local_cache = str(Path(checkpoint).parent / "backbone_cache")
        self.model = build_vit(
            len(class_names),
            backbone,
            cache_dir=local_cache if Path(local_cache).exists() else None,
        )
        state = torch.load(checkpoint, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, image, top_k: int = 3):
        """
        image: str path, PIL.Image, or np.ndarray.
        Returns list of {class_name, confidence} sorted by confidence desc.
        """
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")
        else:
            image = image.convert("RGB")

        tensor = self.transform(image).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.model(tensor)
            logits = out.logits if hasattr(out, "logits") else out
            probs = torch.softmax(logits, dim=1)[0]

        topk = probs.topk(top_k)
        return [
            {"class_name": self.class_names[idx], "confidence": float(conf)}
            for conf, idx in zip(topk.values, topk.indices)
        ]

    def attention_heatmap(self, image) -> tuple[Image.Image, np.ndarray]:
        """Returns (overlaid PIL image, raw attn_map array)."""
        if isinstance(image, str):
            image = Image.open(image).convert("RGB")
        elif isinstance(image, np.ndarray):
            image = Image.fromarray(image).convert("RGB")

        tensor = self.transform(image).unsqueeze(0).to(self.device)
        attn_map = get_attention_map(self.model, tensor)
        overlay = overlay_attention(image, attn_map)
        return overlay, attn_map

    def benchmark_fps(self, image_dir: str, max_images: int = 100) -> float:
        from glob import glob
        paths = sorted(glob(f"{image_dir}/*"))[:max_images]
        start = time.perf_counter()
        for p in paths:
            self.predict(p)
        elapsed = time.perf_counter() - start
        fps = len(paths) / elapsed
        print(f"ViT FPS: {fps:.1f} ({len(paths)} images in {elapsed:.2f}s)")
        return fps
