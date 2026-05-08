"""
Pre-download the ViT backbone from HuggingFace and cache it locally.
Run this ONCE on any machine with internet access before going to the
presentation venue (where internet may be slow or blocked).

Usage:
    python download_weights.py
"""

import yaml
from pathlib import Path
from transformers import ViTForImageClassification, ViTImageProcessor

CFG_PATH = "config/config.yaml"
with open(CFG_PATH) as f:
    cfg = yaml.safe_load(f)

backbone   = cfg["vit"]["backbone"]          # google/vit-base-patch16-224
cache_dir  = "runs/vit/backbone_cache"

Path(cache_dir).mkdir(parents=True, exist_ok=True)

print(f"Downloading {backbone} → {cache_dir}")
print("(This is ~330 MB, one-time only)\n")

ViTForImageClassification.from_pretrained(backbone, cache_dir=cache_dir)
ViTImageProcessor.from_pretrained(backbone, cache_dir=cache_dir)

print(f"\nDone. Weights cached to: {cache_dir}")
print("The training scripts will use this cache automatically.")
print("You can now work offline during the presentation.")
