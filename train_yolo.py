"""
Train the YOLOv11 detector on PlantDoc.

Usage:
    python train_yolo.py
    python train_yolo.py --epochs 100 --imgsz 640 --batch 16
"""

import argparse
import yaml
from pathlib import Path
from models.yolo_detector import YOLODetector


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv11 on PlantDoc")
    parser.add_argument("--config",  default="config/config.yaml")
    parser.add_argument("--epochs",  type=int,   default=None)
    parser.add_argument("--imgsz",   type=int,   default=None)
    parser.add_argument("--batch",   type=int,   default=None)
    parser.add_argument("--weights", type=str,   default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    weights = args.weights or cfg["yolo"]["weights"]
    epochs  = args.epochs  or cfg["yolo"]["epochs"]
    imgsz   = args.imgsz   or cfg["yolo"]["imgsz"]
    batch   = args.batch   or cfg["yolo"]["batch"]
    data_yaml = cfg["dataset"]["data_yaml"]

    print(f"Training YOLO — weights={weights}  epochs={epochs}  imgsz={imgsz}  batch={batch}")

    detector = YOLODetector(weights=weights)
    best_ckpt = detector.train(
        data_yaml=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
    )
    print(f"\nDone. Best checkpoint: {best_ckpt}")


if __name__ == "__main__":
    main()
