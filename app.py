"""Flask web frontend for the Grocery Scanner.

Loads all models at startup, serves a single-page UI, and provides API
endpoints for scanning shelf images, resolving ambiguous detections,
and registering new products on the fly.

Usage:
    python app.py
    python app.py --port 8080
    python app.py --no-ocr          # skip OCR for faster startup
    python app.py --adapter catalog/adapter.pt
"""

import argparse
import base64
import importlib.util
import io
import json
import sys
import time
from pathlib import Path

import faiss
import numpy as np
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from PIL import Image, ImageDraw

# ── Project imports ────────────────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

from src.config import (
    TOP_K, DETECTOR_CONF_THRESHOLD,
    OCR_TEXTS_PATH, OCR_ALPHA, OCR_RERANK_DEPTH,
    DEFAULT_SCORE_THRESHOLD, DEFAULT_MIN_MARGIN,
    SKU_INDEX_PATH, SKU_LABELS_PATH, EMBEDDING_DIM,
    ADAPTER_PATH,
)
from src.detector import Detector, Detection
from src.embedder import Embedder
from src.index import (
    load_sku_index, load_sku_labels, save_sku_index, save_sku_labels,
    search_sku, compute_confidence,
)
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
def load_models(use_ocr: bool = True, adapter_path: str | None = None) -> None:
    global detector, embedder, sku_index, sku_labels, ocr_reader, sku_ocr_texts

    print("Loading detector ...", end=" ", flush=True)
    t0 = time.time()
    detector = Detector()
    detector.load()
    print(f"({time.time() - t0:.1f}s)")

    print("Loading CLIP embedder ...", end=" ", flush=True)
    t0 = time.time()
    embedder = Embedder()
    embedder.load(adapter_path=adapter_path)
    print(f"({time.time() - t0:.1f}s)")

    print("Loading SKU index ...", end=" ", flush=True)
    t0 = time.time()
    sku_index = load_sku_index()
    sku_labels = load_sku_labels()
    n_skus = len(set(l["product_id"] for l in sku_labels))
    print(f"({time.time() - t0:.1f}s, {sku_index.ntotal} prototypes, {n_skus} SKUs)")

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

        if idx_str in overrides:
            pid = overrides[idx_str]
            score = det["top_k"][0]["score"] if det["top_k"] else 0
            items.append({"index": idx, "product_id": pid, "score": score, "method": "user"})
        elif det["tier"] == "CONFIDENT" and det["product_id"]:
            items.append({
                "index": idx,
                "product_id": det["product_id"],
                "score": det["top_k"][0]["score"] if det["top_k"] else 0,
                "method": "auto",
            })

    return jsonify({"items": items})


@app.route("/api/register", methods=["POST"])
def api_register():
    """Register a new product from one or more reference images.

    Accepts multipart/form-data with:
        product_id  — (string) the product / SKU identifier
        images      — one or more image files

    Embeds all uploaded images, averages them into a single prototype, and
    upserts it into the in-memory SKU index (and saves to disk).
    """
    global sku_index, sku_labels

    product_id = request.form.get("product_id", "").strip()
    if not product_id:
        return jsonify({"error": "product_id is required"}), 400

    files = request.files.getlist("images")
    if not files:
        return jsonify({"error": "At least one image is required"}), 400

    images = []
    for f in files:
        try:
            images.append(Image.open(f.stream).convert("RGB"))
        except Exception as e:
            return jsonify({"error": f"Invalid image '{f.filename}': {e}"}), 400

    # Embed and compute prototype
    embs = embedder.embed_batch(images)
    prototype = embs.mean(axis=0).astype(np.float32)
    norm = np.linalg.norm(prototype)
    if norm > 0:
        prototype /= norm

    # Remove existing entry for this product if present
    existing = [l for l in sku_labels if l["product_id"] == product_id]
    if existing:
        keep_rows = [i for i, l in enumerate(sku_labels) if l["product_id"] != product_id]
        all_vecs = np.zeros((sku_index.ntotal, EMBEDDING_DIM), dtype=np.float32)
        sku_index.reconstruct_n(0, sku_index.ntotal, all_vecs)
        kept_vecs = np.ascontiguousarray(all_vecs[keep_rows])
        new_index = faiss.IndexFlatIP(EMBEDDING_DIM)
        if len(kept_vecs):
            new_index.add(kept_vecs)
        sku_labels = [
            {"index": new_i, "product_id": sku_labels[old_i]["product_id"]}
            for new_i, old_i in enumerate(keep_rows)
        ]
        sku_index = new_index

    new_idx = sku_index.ntotal
    sku_index.add(prototype.reshape(1, -1))
    sku_labels.append({"index": new_idx, "product_id": product_id})

    # Persist to disk
    save_sku_index(sku_index)
    save_sku_labels(sku_labels)

    n_skus = len(set(l["product_id"] for l in sku_labels))
    return jsonify({
        "registered": product_id,
        "images_used": len(images),
        "catalog_skus": n_skus,
        "catalog_prototypes": sku_index.ntotal,
    })


# ── Entry point ───────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Grocery Scanner web frontend.")
    parser.add_argument("--port", type=int, default=5000)
    parser.add_argument("--no-ocr", action="store_true", help="Skip OCR for faster startup.")
    parser.add_argument("--adapter", type=str, default=None,
                        help="Path to CLIPAdapter checkpoint.")
    args = parser.parse_args()

    adapter_path = args.adapter or (str(ADAPTER_PATH) if ADAPTER_PATH.exists() else None)
    load_models(use_ocr=not args.no_ocr, adapter_path=adapter_path)
    print(f"\nStarting server on http://0.0.0.0:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=False)
