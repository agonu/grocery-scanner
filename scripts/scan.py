"""
Scan a shelf image: detect products, recognize each, and apply confidence gating.

Usage:
    python scripts/scan.py shelf_photo.jpg
    python scripts/scan.py shelf_photo.jpg --save-annotated annotated.jpg
    python scripts/scan.py shelf_photo.jpg --json results.json --non-interactive
    python scripts/scan.py shelf_photo.jpg --no-ocr --det-threshold 0.3
"""

import argparse
import dataclasses
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    TOP_K, DETECTOR_CONF_THRESHOLD,
    OCR_TEXTS_PATH, OCR_ALPHA, OCR_RERANK_DEPTH,
    DEFAULT_SCORE_THRESHOLD, DEFAULT_MIN_MARGIN,
)
from src.detector import Detector, Detection
from src.embedder import Embedder
from src.index import load_sku_index, load_sku_labels, search, compute_confidence
from src.ocr import extract_text_pil, text_rerank, load_ocr_texts


@dataclasses.dataclass
class ScanResult:
    """Recognition result for a single detection."""
    detection_index: int
    ranked_pids: list[str]
    ranked_scores: list[float]
    confidence: dict
    tier: str             # "CONFIDENT" | "AMBIGUOUS" | "UNKNOWN"
    ocr_text: str | None
    user_override: str | None = None

    @property
    def product_id(self) -> str | None:
        """Final product ID (user override or top-1 if confident)."""
        if self.user_override:
            return self.user_override
        if self.tier == "CONFIDENT" and self.ranked_pids:
            return self.ranked_pids[0]
        return None


def classify_tier(conf: dict) -> str:
    """Map confidence metadata to CONFIDENT / AMBIGUOUS / UNKNOWN."""
    if conf["top1_score"] < DEFAULT_SCORE_THRESHOLD:
        return "UNKNOWN"
    if conf["is_confident"]:
        return "CONFIDENT"
    return "AMBIGUOUS"


def recognize_detection(
    det: Detection,
    index: int,
    embedder: Embedder,
    sku_index,
    sku_labels: list[dict],
    top_k: int,
    ocr_reader=None,
    sku_ocr_texts: dict[str, str] | None = None,
) -> ScanResult:
    """Run full recognition pipeline on a single detection crop."""
    # Embed with multicrop (best mode)
    query_emb = embedder.embed_multicrop(det.crop)

    if ocr_reader and sku_ocr_texts:
        # Deep search for OCR reranking
        scores, indices = search(sku_index, query_emb, top_k=OCR_RERANK_DEPTH)
        candidate_pids = [sku_labels[idx]["product_id"] for idx in indices if idx >= 0]
        candidate_scores = [float(scores[j]) for j, idx in enumerate(indices) if idx >= 0]

        ocr_text = extract_text_pil(det.crop, ocr_reader)

        if ocr_text.strip():
            reranked = text_rerank(
                ocr_text, candidate_pids, candidate_scores,
                sku_ocr_texts, alpha=OCR_ALPHA,
            )
            final_pids = [pid for pid, _, _, _ in reranked[:top_k]]
            final_scores = [fs for _, fs, _, _ in reranked[:top_k]]
        else:
            final_pids = candidate_pids[:top_k]
            final_scores = candidate_scores[:top_k]
    else:
        # Visual-only search
        scores, indices = search(sku_index, query_emb, top_k=top_k)
        final_pids = [sku_labels[idx]["product_id"] for idx in indices if idx >= 0]
        final_scores = [float(scores[j]) for j, idx in enumerate(indices) if idx >= 0]
        ocr_text = None

    # Confidence gating
    score_arr = np.array(final_scores) if final_scores else np.array([0.0])
    conf = compute_confidence(score_arr)
    tier = classify_tier(conf)

    return ScanResult(
        detection_index=index,
        ranked_pids=final_pids,
        ranked_scores=final_scores,
        confidence=conf,
        tier=tier,
        ocr_text=ocr_text,
    )


def handle_interactive(detections: list[Detection], results: list[ScanResult]) -> None:
    """Prompt user for AMBIGUOUS and UNKNOWN detections."""
    ambiguous = [(d, r) for d, r in zip(detections, results) if r.tier == "AMBIGUOUS"]
    unknown = [(d, r) for d, r in zip(detections, results) if r.tier == "UNKNOWN"]

    if not ambiguous and not unknown:
        return

    print(f"\n{'=' * 60}")
    print(f"  {len(ambiguous)} ambiguous + {len(unknown)} unknown need input")
    print(f"{'=' * 60}")

    for det, res in ambiguous:
        print(f"\n  Detection #{res.detection_index + 1} [{det.class_name}] "
              f"at ({det.box[0]},{det.box[1]})-({det.box[2]},{det.box[3]})")
        print(f"  Top-3 candidates:")
        for rank, (pid, score) in enumerate(
            zip(res.ranked_pids[:3], res.ranked_scores[:3]), 1
        ):
            print(f"    {rank}) Product {pid}  ({score * 100:.1f}%)")
        choice = input("  Pick 1/2/3, or 's' to skip: ").strip()
        if choice in ("1", "2", "3"):
            idx = int(choice) - 1
            if idx < len(res.ranked_pids):
                res.user_override = res.ranked_pids[idx]

    for det, res in unknown:
        print(f"\n  Detection #{res.detection_index + 1} [{det.class_name}] "
              f"at ({det.box[0]},{det.box[1]})-({det.box[2]},{det.box[3]})")
        best = f"Product {res.ranked_pids[0]} at {res.ranked_scores[0] * 100:.1f}%" if res.ranked_pids else "none"
        print(f"  No confident match (best: {best})")
        pid = input("  Type product ID, or 's' to skip: ").strip()
        if pid and pid.lower() != "s":
            res.user_override = pid


def print_results(
    detections: list[Detection],
    results: list[ScanResult],
    image_name: str,
    detect_time: float,
    recog_time: float,
) -> None:
    """Print structured scan results to stdout."""
    n_conf = sum(1 for r in results if r.tier == "CONFIDENT")
    n_ambig = sum(1 for r in results if r.tier == "AMBIGUOUS")
    n_unk = sum(1 for r in results if r.tier == "UNKNOWN")

    print()
    print(f"{'=' * 70}")
    print(f"  Shelf Scan: {image_name}")
    print(f"  Detected: {len(detections)} items | "
          f"Confident: {n_conf} | Ambiguous: {n_ambig} | Unknown: {n_unk}")
    print(f"{'=' * 70}")
    print()

    hdr = f"  {'#':>2s}  {'Box':<24s} {'Tier':<11s} {'Product':>7s} {'Score':>6s}  {'Method'}"
    print(hdr)
    print(f"  {'--':>2s}  {'-' * 24} {'-' * 11} {'-' * 7} {'-' * 6}  {'-' * 8}")

    for det, res in zip(detections, results):
        box_str = f"({det.box[0]:>4d},{det.box[1]:>4d},{det.box[2]:>4d},{det.box[3]:>4d})"

        if res.user_override:
            pid_str = f"SKU {res.user_override:>3s}"
            method = "user"
        elif res.tier == "CONFIDENT" and res.ranked_pids:
            pid_str = f"SKU {res.ranked_pids[0]:>3s}"
            method = "auto"
        elif res.tier == "AMBIGUOUS" and res.ranked_pids:
            pid_str = f"SKU {res.ranked_pids[0]:>3s}"
            method = "ambig"
        else:
            pid_str = "    ---"
            method = "skip"

        score_str = f"{res.ranked_scores[0] * 100:5.1f}%" if res.ranked_scores else "  N/A"
        print(f"  {res.detection_index + 1:>2d}  {box_str:<24s} {res.tier:<11s} {pid_str} {score_str}  [{method}]")

    n_det = len(detections)
    per_item = recog_time / n_det * 1000 if n_det > 0 else 0
    print()
    print(f"  Timing: detect={detect_time * 1000:.0f}ms, "
          f"recognize={recog_time:.1f}s ({per_item:.0f}ms/item), "
          f"total={detect_time + recog_time:.1f}s")


def save_annotated_image(
    image: Image.Image,
    detections: list[Detection],
    results: list[ScanResult],
    output_path: str,
) -> None:
    """Draw bounding boxes + labels on the shelf image and save."""
    COLORS = {"CONFIDENT": "green", "AMBIGUOUS": "yellow", "UNKNOWN": "red"}
    annotated = image.copy()
    draw = ImageDraw.Draw(annotated)

    for det, res in zip(detections, results):
        color = COLORS.get(res.tier, "white")
        x1, y1, x2, y2 = det.box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        pid = res.product_id or (res.ranked_pids[0] if res.ranked_pids else "?")
        if res.tier == "UNKNOWN":
            label = "???"
        else:
            score_pct = res.ranked_scores[0] * 100 if res.ranked_scores else 0
            label = f"SKU {pid} ({score_pct:.0f}%)"

        draw.text((x1, max(0, y1 - 14)), label, fill=color)

    annotated.save(output_path)
    print(f"\n  Annotated image saved: {output_path}")


def save_json_results(
    detections: list[Detection],
    results: list[ScanResult],
    output_path: str,
) -> None:
    """Save structured scan results as JSON."""
    data = {
        "scan_time": datetime.now().isoformat(),
        "num_detections": len(detections),
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
                "user_override": res.user_override,
            }
            for det, res in zip(detections, results)
        ],
    }
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  JSON results saved: {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scan a shelf image: detect products, recognize each, confidence-gate."
    )
    parser.add_argument("image", help="Path to the shelf image (JPG/PNG).")
    parser.add_argument("--top-k", type=int, default=3, help="Results per detection (default: 3).")
    parser.add_argument("--det-threshold", type=float, default=DETECTOR_CONF_THRESHOLD,
                        help=f"Detector confidence threshold (default: {DETECTOR_CONF_THRESHOLD}).")
    parser.add_argument("--no-ocr", action="store_true", help="Skip OCR reranking (faster).")
    parser.add_argument("--save-annotated", type=str, default=None,
                        help="Save annotated image with boxes + labels to this path.")
    parser.add_argument("--json", type=str, default=None, help="Save structured results as JSON.")
    parser.add_argument("--non-interactive", action="store_true",
                        help="Skip interactive prompts for ambiguous/unknown items.")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"ERROR: Image not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    # ── Phase 1: Load models ──────────────────────────────
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

    ocr_reader = None
    sku_ocr_texts = None
    if not args.no_ocr:
        if OCR_TEXTS_PATH.exists():
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
            print(f"  (OCR index not found at {OCR_TEXTS_PATH}, skipping OCR)")

    # ── Phase 2: Detect ───────────────────────────────────
    image = Image.open(image_path).convert("RGB")
    print(f"\nDetecting products in {image_path.name} ({image.size[0]}x{image.size[1]}) ...", end=" ", flush=True)
    t0 = time.time()
    detections = detector.detect(image, conf_threshold=args.det_threshold)
    detect_time = time.time() - t0
    print(f"{len(detections)} found ({detect_time * 1000:.0f}ms)")

    if len(detections) == 0:
        print("\nNo objects detected. Using whole-image fallback...")
        img_w, img_h = image.size
        detections = [Detection(
            box=(0, 0, img_w, img_h),
            score=1.0,
            class_id=-1,
            class_name="product",
            crop=image.copy(),
        )]

    # ── Phase 3: Recognize each detection ─────────────────
    print(f"\nRecognizing {len(detections)} detections ...")
    t0 = time.time()
    results: list[ScanResult] = []
    for i, det in enumerate(detections):
        result = recognize_detection(
            det, i, embedder, sku_index, sku_labels,
            args.top_k, ocr_reader, sku_ocr_texts,
        )
        results.append(result)
        tier_char = {"CONFIDENT": "+", "AMBIGUOUS": "?", "UNKNOWN": "-"}[result.tier]
        pid = result.ranked_pids[0] if result.ranked_pids else "?"
        score = result.ranked_scores[0] * 100 if result.ranked_scores else 0
        print(f"  [{tier_char}] #{i + 1} {det.class_name:<12s} -> SKU {pid:>3s} ({score:.1f}%)")
    recog_time = time.time() - t0

    # ── Phase 4: Interactive prompts ──────────────────────
    if not args.non_interactive:
        handle_interactive(detections, results)

    # ── Phase 5: Output ───────────────────────────────────
    print_results(detections, results, image_path.name, detect_time, recog_time)

    if args.save_annotated:
        save_annotated_image(image, detections, results, args.save_annotated)

    if args.json:
        save_json_results(detections, results, args.json)


if __name__ == "__main__":
    main()
