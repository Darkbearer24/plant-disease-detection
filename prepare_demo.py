"""
Pre-compute and cache predictions for a curated set of demo images.

Run this AFTER training both models (but before the presentation):
    python prepare_demo.py

Creates demo_cache/ with pre-rendered images + prediction JSONs.
The web dashboard will serve these instantly — no GPU needed on demo day
if something goes wrong with live inference.

The script also copies the curated images to demo_images/ so you can
drag-and-drop them into the dashboard during the presentation.
"""

import io
import json
import base64
import shutil
import random
from pathlib import Path
from glob import glob

import yaml
import numpy as np
from PIL import Image


CACHE_DIR       = Path("demo_cache")
DEMO_IMAGES_DIR = Path("demo_images")
N_IMAGES        = 20   # how many curated demo images to cache
CFG_PATH        = "config/config.yaml"

with open(CFG_PATH) as f:
    CFG = yaml.safe_load(f)

YC = CFG["yolo"]
VC = CFG["vit"]


def pil_to_b64(img: Image.Image) -> str:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return base64.b64encode(buf.getvalue()).decode()


def ndarray_to_b64(arr: np.ndarray) -> str:
    return pil_to_b64(Image.fromarray(arr.astype(np.uint8)))


def pick_diverse_images(test_dir: str, n: int) -> list[Path]:
    """
    Pick n test images spread across as many classes as possible,
    choosing images that have at least one annotation (more visual).
    """
    lbl_dir = Path(test_dir).parent.parent / "test" / "labels"
    candidates = []
    for img_path in Path(test_dir).glob("*"):
        lbl = lbl_dir / (img_path.stem + ".txt")
        if lbl.exists() and lbl.stat().st_size > 0:
            n_boxes = len(lbl.read_text().strip().splitlines())
            candidates.append((img_path, n_boxes))

    # Sort by n_boxes desc (richer images first), then shuffle within tiers
    candidates.sort(key=lambda x: -x[1])
    selected = [p for p, _ in candidates[:n * 3]]
    random.shuffle(selected)
    return selected[:n]


def main():
    CACHE_DIR.mkdir(exist_ok=True)
    DEMO_IMAGES_DIR.mkdir(exist_ok=True)

    # ---- Load models ----
    yolo_ok = Path(YC["trained_weights"]).exists()
    vit_ok  = Path(VC["checkpoint"]).exists()

    if not yolo_ok and not vit_ok:
        print("Neither model is trained yet. Train both models first.")
        return

    yolo_det = vit_inf = None

    if yolo_ok:
        from models.yolo_detector import YOLODetector
        print(f"Loading YOLO from {YC['trained_weights']}...")
        yolo_det = YOLODetector(weights=YC["trained_weights"])

    if vit_ok:
        ckpt_dir = Path(VC["checkpoint"]).parent
        with open(ckpt_dir / "class_names.json") as f:
            class_names = json.load(f)
        from models.vit_classifier import ViTInference
        print(f"Loading ViT from {VC['checkpoint']}...")
        vit_inf = ViTInference(
            checkpoint=VC["checkpoint"],
            class_names=class_names,
            backbone=VC["backbone"],
        )

    # ---- Pick images ----
    test_images_dir = f"{CFG['dataset']['root']}/test/images"
    selected = pick_diverse_images(test_images_dir, N_IMAGES)
    print(f"\nSelected {len(selected)} demo images.")

    manifest = []

    for i, img_path in enumerate(selected, 1):
        img = Image.open(img_path).convert("RGB")
        stem = img_path.stem
        out = {"id": stem, "filename": img_path.name}

        print(f"  [{i:02d}/{len(selected)}] {img_path.name}", end="")

        # Copy to demo_images/
        shutil.copy(img_path, DEMO_IMAGES_DIR / img_path.name)

        # YOLO
        if yolo_det:
            dets     = yolo_det.predict_image(img, conf=CFG["web"]["confidence_threshold"])
            rendered = yolo_det.render(img, conf=CFG["web"]["confidence_threshold"])
            out["yolo"] = {
                "detections":      dets,
                "annotated_image": ndarray_to_b64(rendered),
            }
            print(f" | YOLO: {len(dets)} det(s)", end="")

        # ViT
        if vit_inf:
            preds      = vit_inf.predict(img, top_k=5)
            overlay, _ = vit_inf.attention_heatmap(img)
            out["vit"] = {
                "predictions":    preds,
                "attention_image": pil_to_b64(overlay),
            }
            top1 = preds[0]["class_name"] if preds else "?"
            print(f" | ViT: {top1}", end="")

        out["original_image"] = pil_to_b64(img)
        print()

        # Save per-image cache
        cache_file = CACHE_DIR / f"{stem}.json"
        with open(cache_file, "w") as f:
            json.dump(out, f)

        manifest.append({"id": stem, "filename": img_path.name})

    # Save manifest
    with open(CACHE_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nCached {len(selected)} demo results → {CACHE_DIR}/")
    print(f"Demo images copied → {DEMO_IMAGES_DIR}/")
    print("\nOn presentation day, drag any image from demo_images/ into the dashboard.")


if __name__ == "__main__":
    main()
