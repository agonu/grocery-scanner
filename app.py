"""Flask web frontend for the Grocery Scanner.

Loads all models at startup, serves a single-page UI, and provides API
endpoints for scanning shelf images and resolving ambiguous detections.

Usage:
    python app.py
    python app.py --port 8080
    python app.py --no-ocr          # skip OCR for faster startup
"""

import argparse
import base64
import importlib.util
import io
import sys
import time
from pathlib import Path

from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageDraw

# ── Project imports ────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import (
    TOP_K, DETECTOR_CONF_THRESHOLD,
    OCR_TEXTS_PATH, OCR_ALPHA, OCR_RERANK_DEPTH,
    DEFAULT_SCORE_THRESHOLD, DEFAULT_MIN_MARGIN,
)
from src.detector import Detector, Detection
from src.embedder import Embedder
from src.index import load_sku_index, load_sku_labels, search, compute_confidence
from src.ocr import extract_text_pil, text_rerank, load_ocr_texts

# Import ScanResult and helpers from scripts/scan.py via importlib
_scan_spec = importlib.util.spec_from_file_location(
    "scan_module", str(Path(__file__).resolve().parent / "scripts" / "scan.py")
)
_scan_mod = importlib.util.module_from_spec(_scan_spec)
_scan_spec.loader.exec_module(_scan_mod)

ScanResult = _scan_mod.ScanResult
classify_tier = _scan_mod.classify_tier
recognize_detection = _scan_mod.recognize_detection

# ── Globals (loaded once at startup) ──────────────────────
detector: Detector = None
embedder: Embedder = None
sku_index = None
sku_labels: list[dict] = None
ocr_reader = None
sku_ocr_texts: dict[str, str] = None

app = Flask(__name__)
CORS(app)


# ── Model loading ─────────────────────────────────────────
def load_models(use_ocr: bool = True) -> None:
    global detector, embedder, sku_index, sku_labels, ocr_reader, sku_ocr_texts

    print("Loading detector ...", end=" ", flush=True)
    t0 = time.time()
    detector = Detector()
    detector.load()
    print(f"({time.time() - t0:.1f}s)")

    print("Loading CLIP embedder ...", end=" ", flush=True)
    t0 = time.time()
    embedder = Embedder()
    embedder.load()
    print(f"({time.time() - t0:.1f}s)")

    print("Loading SKU index ...", end=" ", flush=True)
    t0 = time.time()
    sku_index = load_sku_index()
    sku_labels = load_sku_labels()
    print(f"({time.time() - t0:.1f}s, {sku_index.ntotal} SKUs)")

    if use_ocr and OCR_TEXTS_PATH.exists():
        print("Loading OCR text index ...", end=" ", flush=True)
        t0 = time.time()
        sku_ocr_texts = load_ocr_texts(OCR_TEXTS_PATH)
        print(f"({time.time() - t0:.1f}s, {len(sku_ocr_texts)} SKUs)")

        print("Initializing EasyOCR ...", end=" ", flush=True)
        t0 = time.time()
        import easyocr
        ocr_reader = easyocr.Reader(["en"], gpu=True, verbose=False)
        print(f"({time.time() - t0:.1f}s)")
    else:
        print("  (OCR disabled or index not found)")


# ── Annotation drawing ────────────────────────────────────
TIER_COLORS = {"CONFIDENT": "green", "AMBIGUOUS": "yellow", "UNKNOWN": "red"}


def draw_annotations(
    image: Image.Image,
    detections: list[Detection],
    results: list[ScanResult],
) -> Image.Image:
    """Draw colored bounding boxes + labels, return annotated copy."""
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    for det, res in zip(detections, results):
        color = TIER_COLORS.get(res.tier, "white")
        x1, y1, x2, y2 = det.box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        pid = res.product_id or (res.ranked_pids[0] if res.ranked_pids else "?")
        if res.tier == "UNKNOWN":
            label = "???"
        else:
            score_pct = res.ranked_scores[0] * 100 if res.ranked_scores else 0
            label = f"SKU {pid} ({score_pct:.0f}%)"

        draw.text((x1, max(0, y1 - 14)), label, fill=color)

    return annotated


def image_to_base64(img: Image.Image, fmt: str = "JPEG", quality: int = 85) -> str:
    """Encode a PIL image to a base64 string."""
    buf = io.BytesIO()
    img.save(buf, format=fmt, quality=quality)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ── Routes ────────────────────────────────────────────────
@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/scan", methods=["POST"])
def api_scan():
    """Accept an image upload, run detection + recognition, return JSON results."""
    if "image" not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files["image"]
    try:
        image = Image.open(file.stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": f"Invalid image: {e}"}), 400

    img_w, img_h = image.size

    # Phase 1: Detect
    t0 = time.time()
    detections = detector.detect(image, conf_threshold=DETECTOR_CONF_THRESHOLD)
    detect_ms = int((time.time() - t0) * 1000)

    if len(detections) == 0:
        # Whole-image fallback: treat entire image as one product
        detections = [Detection(
            box=(0, 0, img_w, img_h),
            score=1.0,
            class_id=-1,
            class_name="product",
            crop=image.copy(),
        )]

    # Phase 2: Recognize each detection
    t0 = time.time()
    results: list[ScanResult] = []
    for i, det in enumerate(detections):
        result = recognize_detection(
            det, i, embedder, sku_index, sku_labels,
            TOP_K, ocr_reader, sku_ocr_texts,
        )
        results.append(result)
    recognize_ms = int((time.time() - t0) * 1000)

    # Phase 3: Draw annotations
    annotated = draw_annotations(image, detections, results)
    annotated_b64 = image_to_base64(annotated)

    # Build response
    response = {
        "num_detections": len(detections),
        "image_size": [img_w, img_h],
        "detections": [
            {
                "index": res.detection_index,
                "box": list(det.box),
                "detector_class": det.class_name,
                "detector_score": round(det.score, 3),
                "tier": res.tier,
                "product_id": res.product_id,
                "top_k": [
                    {"product_id": pid, "score": round(sc, 4)}
                    for pid, sc in zip(res.ranked_pids, res.ranked_scores)
                ],
                "confidence": {
                    "top1_score": round(res.confidence["top1_score"], 4),
                    "margin": round(res.confidence["margin"], 4),
                    "is_confident": res.confidence["is_confident"],
                },
                "ocr_text": res.ocr_text,
            }
            for det, res in zip(detections, results)
        ],
        "annotated_image": annotated_b64,
        "timing": {
            "detect_ms": detect_ms,
            "recognize_ms": recognize_ms,
            "total_ms": detect_ms + recognize_ms,
        },
    }

    return jsonify(response)


@app.route("/api/resolve", methods=["POST"])
def api_resolve():
    """Accept user overrides and return the final shopping list."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "No JSON body"}), 400

    detections_data = data.get("detections", [])
    overrides = data.get("overrides", {})

    items = []
    for det in detections_data:
        idx = det["index"]
        idx_str = str(idx)

        # Check for user override
        if idx_str in overrides:
            pid = overrides[idx_str]
            score = det["top_k"][0]["score"] if det["top_k"] else 0
            items.append({
                "index": idx,
                "product_id": pid,
                "score": score,
                "method": "user",
            })
        elif det["tier"] == "CONFIDENT" and det["product_id"]:
            items.append({
                "index": idx,
                "product_id": det["product_id"],
                "score": det["top_k"][0]["score"] if det["top_k"] else 0,
                "method": "auto",
            })
        # Skip AMBIGUOUS without override and UNKNOWN without override

    return jsonify({"items": items})


# ── Entry point ───────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grocery Scanner web frontend.")
    parser.add_argument("--port", type=int, default=5000, help="Port (default: 5000).")
    parser.add_argument("--no-ocr", action="store_true", help="Skip OCR for faster startup.")
    args = parser.parse_args()

    load_models(use_ocr=not args.no_ocr)
    print(f"\nStarting server on http://0.0.0.0:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=False)
