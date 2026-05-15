"""
CLI prediction tool — run YOLO + ViT on an image and print results.

Usage:
    python predict.py path/to/leaf.jpg
    python predict.py path/to/leaf.jpg --show
    python predict.py path/to/leaf.jpg --attention
"""

import argparse
import json
from pathlib import Path

import yaml
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def parse_args():
    parser = argparse.ArgumentParser(description="Predict plant disease from a leaf image")
    parser.add_argument("image", help="Path to leaf image")
    parser.add_argument("--config",    default="config/config.yaml")
    parser.add_argument("--show",      action="store_true", help="Show annotated images")
    parser.add_argument("--attention", action="store_true", help="Show ViT attention heatmap")
    parser.add_argument("--conf",      type=float, default=0.25, help="YOLO confidence threshold")
    parser.add_argument("--topk",      type=int,   default=3,    help="ViT top-K predictions")
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    vc = cfg["vit"]
    yc = cfg["yolo"]
    ckpt_dir    = str(Path(vc["checkpoint"]).parent)
    class_names_path = Path(ckpt_dir) / "class_names.json"

    image_path = args.image
    img = Image.open(image_path).convert("RGB")

    # ---- YOLO ----
    print("\n[YOLO Detection]")
    from models.yolo_detector import YOLODetector
    detector = YOLODetector(weights=yc["trained_weights"])
    detections = detector.predict_image(image_path, conf=args.conf)

    if detections:
        for d in detections:
            print(f"  {d['cls_name']:<40s}  conf={d['conf']:.3f}  box={[round(v) for v in d['box']]}")
    else:
        print("  No detections above confidence threshold.")

    # ---- ViT (optional — only if checkpoint exists) ----
    vit_available = class_names_path.exists() and Path(vc["checkpoint"]).exists()
    infer = None

    if vit_available:
        print(f"\n[ViT Classification — top {args.topk}]")
        with open(class_names_path) as f:
            class_names = json.load(f)

        from models.vit_classifier import ViTInference
        infer = ViTInference(
            checkpoint=vc["checkpoint"],
            class_names=class_names,
            backbone=vc["backbone"],
        )

        # Use best YOLO crop — ViT was trained on crops, not full images
        vit_input = img
        if detections:
            best = max(detections, key=lambda d: d["conf"])
            x1, y1, x2, y2 = [int(v) for v in best["box"]]
            W, H = img.size
            pad_x = int((x2 - x1) * 0.05)
            pad_y = int((y2 - y1) * 0.05)
            vit_input = img.crop((max(0, x1 - pad_x), max(0, y1 - pad_y),
                                   min(W, x2 + pad_x), min(H, y2 + pad_y)))
            print(f"  (classifying best YOLO crop: {best['cls_name']} conf={best['conf']:.2f})")
        else:
            print("  (no YOLO detections — classifying full image)")

        predictions = infer.predict_tta(vit_input, top_k=args.topk)
        for p in predictions:
            bar = "█" * int(p["confidence"] * 30)
            print(f"  {p['class_name']:<40s}  {bar}  {p['confidence']:.1%}")
    else:
        print("\n[ViT] Checkpoint not found — train ViT first with: python train_vit.py")

    # ---- Visualisation (--show always draws YOLO; --attention adds ViT heatmap) ----
    if args.show or args.attention:
        show_attn = args.attention and infer is not None
        n_panels = 2 if show_attn else 1
        fig, axes = plt.subplots(1, n_panels, figsize=(7 * n_panels, 5))
        if n_panels == 1:
            axes = [axes]

        rendered = detector.render(image_path, conf=args.conf)
        axes[0].imshow(rendered)
        axes[0].set_title("YOLO Detections")
        axes[0].axis("off")

        if show_attn:
            overlay, _ = infer.attention_heatmap(vit_input)
            axes[1].imshow(overlay)
            axes[1].set_title("ViT Attention Heatmap (on crop)" if detections else "ViT Attention Heatmap")
            axes[1].axis("off")
        elif args.attention and not vit_available:
            print("[attention] Skipped — ViT checkpoint not found.")

        plt.suptitle(f"Plant Disease Detection — {Path(image_path).name}")
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
