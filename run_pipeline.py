"""
Full pipeline runner — trains both models, evaluates, and prepares the demo.

Usage:
    python run_pipeline.py                  # run all steps
    python run_pipeline.py --skip-download  # skip backbone download (already cached)
    python run_pipeline.py --skip-yolo      # skip YOLO training (weights exist)
    python run_pipeline.py --skip-vit       # skip ViT training (checkpoint exists)
    python run_pipeline.py --skip-eval      # skip evaluation
    python run_pipeline.py --skip-demo      # skip demo cache preparation
    python run_pipeline.py --launch-web     # open web dashboard after pipeline finishes

    # Custom training parameters
    python run_pipeline.py --yolo-epochs 30 --vit-epochs 10
"""

import argparse
import subprocess
import sys
import time
import yaml
from pathlib import Path


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

BOLD  = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED   = "\033[91m"
RESET = "\033[0m"


def banner(title: str):
    line = "=" * 60
    print(f"\n{BOLD}{line}")
    print(f"  {title}")
    print(f"{line}{RESET}\n")


def success(msg: str):
    print(f"{GREEN}  [OK] {msg}{RESET}")


def warn(msg: str):
    print(f"{YELLOW}  [SKIP] {msg}{RESET}")


def error(msg: str):
    print(f"{RED}  [ERROR] {msg}{RESET}")


def run(cmd: list[str], label: str) -> bool:
    """Run a subprocess, stream its output, return True if it succeeded."""
    print(f"  Running: {' '.join(cmd)}\n")
    result = subprocess.run(cmd, executable=sys.executable)
    if result.returncode != 0:
        error(f"{label} failed (exit code {result.returncode})")
        return False
    return True


def hms(seconds: float) -> str:
    h, rem = divmod(int(seconds), 3600)
    m, s   = divmod(rem, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="Plant disease detection — full pipeline")
    p.add_argument("--config",         default="config/config.yaml")
    p.add_argument("--skip-download",  action="store_true", help="Skip ViT backbone download")
    p.add_argument("--skip-yolo",      action="store_true", help="Skip YOLO training")
    p.add_argument("--skip-vit",       action="store_true", help="Skip ViT training")
    p.add_argument("--skip-eval",      action="store_true", help="Skip evaluation")
    p.add_argument("--skip-demo",      action="store_true", help="Skip demo cache preparation")
    p.add_argument("--launch-web",     action="store_true", help="Launch web dashboard when done")
    p.add_argument("--yolo-epochs",    type=int, default=None, help="Override YOLO epoch count")
    p.add_argument("--vit-epochs",     type=int, default=None, help="Override ViT epoch count")
    p.add_argument("--yolo-batch",     type=int, default=None, help="Override YOLO batch size")
    p.add_argument("--vit-batch",      type=int, default=None, help="Override ViT batch size")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Steps
# ---------------------------------------------------------------------------

def step_download(cfg: dict, args) -> bool:
    banner("Step 0 — Download ViT backbone weights")
    cache_dir = Path("runs/vit/backbone_cache")
    if cache_dir.exists() and any(cache_dir.iterdir()):
        warn("Backbone cache already exists — skipping download.")
        return True
    if args.skip_download:
        warn("--skip-download set — skipping.")
        return True
    return run([sys.executable, "download_weights.py"], "download_weights")


def step_train_yolo(cfg: dict, args) -> bool:
    banner("Step 1 — Train YOLO detector")
    weights_path = Path(cfg["yolo"]["trained_weights"])
    if args.skip_yolo:
        warn("--skip-yolo set — skipping YOLO training.")
        return True
    if weights_path.exists():
        warn(f"Trained weights found at {weights_path} — skipping YOLO training.")
        warn("Delete that file or pass --yolo-epochs to retrain.")
        return True

    cmd = [sys.executable, "train_yolo.py", "--config", args.config]
    if args.yolo_epochs:
        cmd += ["--epochs", str(args.yolo_epochs)]
    if args.yolo_batch:
        cmd += ["--batch", str(args.yolo_batch)]
    return run(cmd, "train_yolo")


def step_extract_crops(cfg: dict, args) -> bool:
    banner("Step 2 — Extract crops from YOLO labels")
    crops_dir = Path(cfg["dataset"]["crops_dir"])
    if crops_dir.exists() and any(crops_dir.iterdir()):
        warn(f"Crops directory already populated at {crops_dir} — skipping extraction.")
        return True
    return run(
        [sys.executable, "train_vit.py", "--extract-crops", "--config", args.config],
        "extract-crops"
    )


def step_train_vit(cfg: dict, args) -> bool:
    banner("Step 3 — Train Vision Transformer classifier")
    ckpt_path = Path(cfg["vit"]["checkpoint"])
    if args.skip_vit:
        warn("--skip-vit set — skipping ViT training.")
        return True
    if ckpt_path.exists():
        warn(f"ViT checkpoint found at {ckpt_path} — skipping ViT training.")
        warn("Delete that file or use --vit-epochs to retrain.")
        return True

    cmd = [sys.executable, "train_vit.py", "--config", args.config]
    if args.vit_epochs:
        cmd += ["--epochs", str(args.vit_epochs)]
    if args.vit_batch:
        cmd += ["--batch", str(args.vit_batch)]
    return run(cmd, "train_vit")


def step_evaluate(cfg: dict, args) -> bool:
    banner("Step 4 — Evaluate both models")
    if args.skip_eval:
        warn("--skip-eval set — skipping evaluation.")
        return True
    return run([sys.executable, "evaluate.py", "--config", args.config], "evaluate")


def step_prepare_demo(cfg: dict, args) -> bool:
    banner("Step 5 — Prepare demo gallery cache")
    if args.skip_demo:
        warn("--skip-demo set — skipping demo preparation.")
        return True
    manifest = Path("demo_cache/manifest.json")
    if manifest.exists():
        warn("Demo cache already exists — skipping.")
        return True
    return run([sys.executable, "prepare_demo.py"], "prepare_demo")


def step_launch_web(cfg: dict):
    banner("Step 6 — Launching web dashboard")
    host = cfg["web"]["host"]
    port = cfg["web"]["port"]
    print(f"  Open your browser at: http://localhost:{port}\n")
    print("  Press Ctrl+C to stop the server.\n")
    subprocess.run([sys.executable, "web/app.py"])


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    print(f"\n{BOLD}Plant Disease Detection — Full Pipeline{RESET}")
    print(f"Config: {args.config}")
    print(f"Python: {sys.executable}\n")

    pipeline_start = time.perf_counter()
    step_times = {}
    failed_at  = None

    steps = [
        ("download",   lambda: step_download(cfg, args)),
        ("train_yolo", lambda: step_train_yolo(cfg, args)),
        ("crops",      lambda: step_extract_crops(cfg, args)),
        ("train_vit",  lambda: step_train_vit(cfg, args)),
        ("evaluate",   lambda: step_evaluate(cfg, args)),
        ("demo",       lambda: step_prepare_demo(cfg, args)),
    ]

    for name, fn in steps:
        t0 = time.perf_counter()
        ok = fn()
        step_times[name] = time.perf_counter() - t0
        if ok:
            success(f"Step '{name}' done in {hms(step_times[name])}")
        else:
            failed_at = name
            break

    # ---- Summary ----
    total = time.perf_counter() - pipeline_start
    banner("Pipeline Summary")
    labels = {
        "download":   "0. Download weights",
        "train_yolo": "1. Train YOLO",
        "crops":      "2. Extract crops",
        "train_vit":  "3. Train ViT",
        "evaluate":   "4. Evaluate",
        "demo":       "5. Prepare demo",
    }
    for name, t in step_times.items():
        status = "FAILED" if name == failed_at else "done"
        color  = RED if name == failed_at else GREEN
        print(f"  {color}{labels[name]:<25} {hms(t):>10}  [{status}]{RESET}")

    print(f"\n  {BOLD}Total time: {hms(total)}{RESET}")

    if failed_at:
        error(f"\nPipeline stopped at step '{failed_at}'. Fix the error above and re-run.")
        error("Already-completed steps will be skipped automatically on the next run.")
        sys.exit(1)
    else:
        success("\nAll steps complete!")
        print(f"\n  Web dashboard: python web/app.py  →  http://localhost:{cfg['web']['port']}")
        print( "  Single image:  python predict.py <image.jpg> --show --attention\n")

    if args.launch_web and not failed_at:
        step_launch_web(cfg)


if __name__ == "__main__":
    main()
