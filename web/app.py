"""
FastAPI web server for the plant disease detection dashboard.

Run:
    python web/app.py
    uvicorn web.app:app --host 0.0.0.0 --port 8000 --reload

Then open: http://localhost:8000
"""

import io
import json
import base64
import sys
from pathlib import Path

import yaml
import numpy as np
from PIL import Image

# Ensure project root is on path when running as web/app.py directly
sys.path.insert(0, str(Path(__file__).parent.parent))

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CONFIG_PATH = Path(__file__).parent.parent / "config" / "config.yaml"
with open(CONFIG_PATH) as f:
    CFG = yaml.safe_load(f)

YC = CFG["yolo"]
VC = CFG["vit"]

YOLO_WEIGHTS = str(Path(__file__).parent.parent / YC["trained_weights"])
VIT_CKPT     = str(Path(__file__).parent.parent / VC["checkpoint"])
VIT_NAMES    = str(Path(VIT_CKPT).parent / "class_names.json")
CONF         = CFG["web"]["confidence_threshold"]

# ---------------------------------------------------------------------------
# Lazy-load models (avoid import-time crash if weights not yet trained)
# ---------------------------------------------------------------------------

_yolo = None
_vit  = None


def get_yolo():
    global _yolo
    if _yolo is None:
        from models.yolo_detector import YOLODetector
        _yolo = YOLODetector(weights=YOLO_WEIGHTS)
    return _yolo


def get_vit():
    global _vit
    if _vit is None:
        if not Path(VIT_CKPT).exists():
            return None
        with open(VIT_NAMES) as f:
            class_names = json.load(f)
        from models.vit_classifier import ViTInference
        _vit = ViTInference(
            checkpoint=VIT_CKPT,
            class_names=class_names,
            backbone=VC["backbone"],
        )
    return _vit


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(title="PlantDoc Disease Detector", version="1.0")

STATIC_DIR    = Path(__file__).parent / "static"
TEMPLATES_DIR = Path(__file__).parent / "templates"

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def pil_to_b64(img: Image.Image, fmt: str = "JPEG") -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode()


def ndarray_to_b64(arr: np.ndarray) -> str:
    return pil_to_b64(Image.fromarray(arr.astype(np.uint8)))


# ---------------------------------------------------------------------------
# Demo cache helpers
# ---------------------------------------------------------------------------

DEMO_CACHE_DIR = Path(__file__).parent.parent / "demo_cache"


def get_demo_manifest() -> list[dict]:
    manifest_path = DEMO_CACHE_DIR / "manifest.json"
    if not manifest_path.exists():
        return []
    with open(manifest_path) as f:
        return json.load(f)


def load_demo_result(image_id: str) -> dict | None:
    cache_file = DEMO_CACHE_DIR / f"{image_id}.json"
    if not cache_file.exists():
        return None
    with open(cache_file) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    yolo_ready   = Path(YOLO_WEIGHTS).exists()
    vit_ready    = Path(VIT_CKPT).exists()
    demo_ready   = (DEMO_CACHE_DIR / "manifest.json").exists()
    demo_gallery = get_demo_manifest() if demo_ready else []
    return templates.TemplateResponse("index.html", {
        "request":      request,
        "yolo_ready":   yolo_ready,
        "vit_ready":    vit_ready,
        "demo_ready":   demo_ready,
        "demo_gallery": demo_gallery,
    })


@app.get("/demo/gallery")
async def demo_gallery():
    """Return the list of pre-cached demo images."""
    return JSONResponse(get_demo_manifest())


@app.get("/demo/result/{image_id}")
async def demo_result(image_id: str):
    """Return a pre-cached prediction result by image ID."""
    result = load_demo_result(image_id)
    if result is None:
        raise HTTPException(404, f"No cached result for image '{image_id}'")
    return JSONResponse(result)


@app.post("/predict/yolo")
async def predict_yolo(file: UploadFile = File(...)):
    if not Path(YOLO_WEIGHTS).exists():
        raise HTTPException(503, "YOLO weights not found. Train the model first.")

    data = await file.read()
    img = Image.open(io.BytesIO(data)).convert("RGB")

    detector = get_yolo()
    detections = detector.predict_image(img, conf=CONF)
    rendered   = detector.render(img, conf=CONF)            # RGB ndarray

    return JSONResponse({
        "detections": detections,
        "annotated_image": ndarray_to_b64(rendered),
    })


@app.post("/predict/vit")
async def predict_vit(file: UploadFile = File(...)):
    infer = get_vit()
    if infer is None:
        raise HTTPException(503, "ViT checkpoint not found. Train the model first.")

    data = await file.read()
    img  = Image.open(io.BytesIO(data)).convert("RGB")

    predictions = infer.predict(img, top_k=5)
    overlay, _  = infer.attention_heatmap(img)

    return JSONResponse({
        "predictions":    predictions,
        "attention_image": pil_to_b64(overlay),
        "original_image":  pil_to_b64(img),
    })


@app.post("/predict/both")
async def predict_both(file: UploadFile = File(...)):
    data = await file.read()
    img  = Image.open(io.BytesIO(data)).convert("RGB")

    result: dict = {}

    # YOLO
    if Path(YOLO_WEIGHTS).exists():
        detector   = get_yolo()
        detections = detector.predict_image(img, conf=CONF)
        rendered   = detector.render(img, conf=CONF)
        result["yolo"] = {
            "detections":     detections,
            "annotated_image": ndarray_to_b64(rendered),
        }

    # ViT
    infer = get_vit()
    if infer:
        predictions = infer.predict(img, top_k=5)
        overlay, _  = infer.attention_heatmap(img)
        result["vit"] = {
            "predictions":    predictions,
            "attention_image": pil_to_b64(overlay),
        }

    result["original_image"] = pil_to_b64(img)

    if not result.get("yolo") and not result.get("vit"):
        raise HTTPException(503, "No trained models found. Train YOLO and/or ViT first.")

    return JSONResponse(result)


@app.get("/status")
async def status():
    return {
        "yolo_ready": Path(YOLO_WEIGHTS).exists(),
        "vit_ready":  Path(VIT_CKPT).exists(),
        "yolo_weights": YOLO_WEIGHTS,
        "vit_checkpoint": VIT_CKPT,
    }


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import uvicorn
    host = CFG["web"]["host"]
    port = CFG["web"]["port"]
    print(f"Starting server at http://{host}:{port}")
    uvicorn.run("web.app:app", host=host, port=port, reload=False)
