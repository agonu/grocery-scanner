"""
Query the GroZi-120 catalog with an image.

Usage:
    python scripts/query.py path/to/query.jpg
    python scripts/query.py path/to/query.jpg --multicrop
    python scripts/query.py path/to/query.jpg --multicrop --query-expand
    python scripts/query.py path/to/query.jpg --multicrop --ocr-rerank
"""

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    TOP_K, EMBEDDINGS_PATH, SKU_EMBEDDINGS_PATH,
    DEFAULT_SCORE_THRESHOLD, DEFAULT_MIN_MARGIN,
    OCR_TEXTS_PATH, OCR_ALPHA, OCR_RERANK_DEPTH,
)
from src.embedder import Embedder
from src.index import (
    load_labels, load_sku_index, load_sku_labels, search,
    expand_query, compute_confidence,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query the GroZi-120 product catalog with an image."
    )
    parser.add_argument("image", help="Path to the query image (JPG/PNG).")
    parser.add_argument("--top-k", type=int, default=TOP_K, help="Number of results.")
    parser.add_argument("--multicrop", action="store_true", help="Letterbox + multi-crop query.")
    parser.add_argument("--two-stage", action="store_true", help="Two-stage: SKU coarse → image rerank.")
    parser.add_argument("--query-expand", action="store_true", help="Expand query by averaging with top-3 SKU matches.")
    parser.add_argument("--threshold", type=float, default=None, help="Min top-1 score for confident prediction.")
    parser.add_argument("--min-margin", type=float, default=None, help="Min margin between top-1 and top-2 scores.")
    parser.add_argument("--ocr-rerank", action="store_true", help="Rerank top matches using OCR text similarity.")
    parser.add_argument("--ocr-alpha", type=float, default=OCR_ALPHA, help=f"Visual weight in OCR fusion (default: {OCR_ALPHA}).")
    args = parser.parse_args()

    image_path = Path(args.image)
    if not image_path.exists():
        print(f"ERROR: Image not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    # Load model
    print("Loading model ...", end=" ", flush=True)
    t0 = time.time()
    embedder = Embedder()
    embedder.load()
    print(f"({time.time() - t0:.1f}s)")

    # Load SKU index
    print("Loading SKU index ...", end=" ", flush=True)
    t0 = time.time()
    sku_index = load_sku_index()
    sku_labels = load_sku_labels()
    print(f"({time.time() - t0:.1f}s, {sku_index.ntotal} SKUs)")

    # Load SKU embeddings for query expansion
    sku_embeddings = None
    if args.query_expand:
        sku_embeddings = np.load(SKU_EMBEDDINGS_PATH).astype(np.float32)

    # Load image embeddings for two-stage
    image_embeddings = None
    pid_to_img_indices = None
    if args.two_stage:
        print("Loading image embeddings ...", end=" ", flush=True)
        t0 = time.time()
        image_labels = load_labels()
        image_embeddings = np.load(EMBEDDINGS_PATH).astype(np.float32)
        faiss.normalize_L2(image_embeddings)
        pid_to_img_indices = defaultdict(list)
        for lbl in image_labels:
            pid_to_img_indices[lbl["product_id"]].append(lbl["index"])
        pid_to_img_indices = {k: np.array(v) for k, v in pid_to_img_indices.items()}
        print(f"({time.time() - t0:.1f}s)")

    # Load OCR data
    sku_ocr_texts = None
    ocr_reader = None
    if args.ocr_rerank:
        from src.ocr import load_ocr_texts, extract_text, text_rerank
        print("Loading OCR text index ...", end=" ", flush=True)
        t0 = time.time()
        sku_ocr_texts = load_ocr_texts(OCR_TEXTS_PATH)
        print(f"({time.time() - t0:.1f}s, {len(sku_ocr_texts)} SKUs)")

        print("Initializing EasyOCR ...", end=" ", flush=True)
        t0 = time.time()
        import easyocr
        ocr_reader = easyocr.Reader(["en"], gpu=True, verbose=False)
        print(f"({time.time() - t0:.1f}s)")

    # Embed query
    print(f"Embedding query: {image_path} ...", end=" ", flush=True)
    t0 = time.time()
    if args.multicrop:
        query_emb = embedder.embed_multicrop(str(image_path))
    else:
        query_emb = embedder.embed_file(str(image_path))
    embed_time = time.time() - t0
    print(f"({embed_time:.2f}s)")

    # Query expansion
    if args.query_expand and sku_embeddings is not None:
        query_emb = expand_query(query_emb, sku_index, sku_embeddings)

    # Search
    t0 = time.time()

    if args.ocr_rerank:
        # Visual search with deeper candidate pool
        scores, indices = search(sku_index, query_emb, top_k=OCR_RERANK_DEPTH)
        candidate_pids = [sku_labels[idx]["product_id"] for idx in indices if idx >= 0]
        candidate_scores = [float(scores[j]) for j, idx in enumerate(indices) if idx >= 0]

        # OCR the query
        query_text = extract_text(str(image_path), ocr_reader)

        if query_text.strip():
            reranked = text_rerank(
                query_text, candidate_pids, candidate_scores,
                sku_ocr_texts, alpha=args.ocr_alpha,
            )
            results = [(pid, final, vscore, tscore) for pid, final, vscore, tscore in reranked[:args.top_k]]
        else:
            query_text = ""
            results = [(pid, sc, sc, 0.0) for pid, sc in zip(candidate_pids[:args.top_k], candidate_scores[:args.top_k])]

    elif args.two_stage:
        # Stage 1: coarse SKU search
        sku_scores, sku_indices = search(sku_index, query_emb, top_k=10)
        candidate_pids = [sku_labels[idx]["product_id"] for idx in sku_indices if idx >= 0]

        # Stage 2: rerank by max image-level similarity
        query = query_emb.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(query)
        results = []
        for pid in candidate_pids:
            img_idx = pid_to_img_indices.get(pid, [])
            if len(img_idx) == 0:
                continue
            pid_embs = image_embeddings[img_idx]
            max_score = float((query @ pid_embs.T).max())
            results.append((pid, max_score))
        results.sort(key=lambda x: x[1], reverse=True)
        results = results[: args.top_k]
    else:
        # Direct SKU prototype search
        scores, indices = search(sku_index, query_emb, top_k=args.top_k)
        results = []
        for score, idx in zip(scores, indices):
            if idx < 0:
                continue
            results.append((sku_labels[idx]["product_id"], float(score)))

    search_time = time.time() - t0

    # Display
    mode_parts = []
    if args.multicrop:
        mode_parts.append("multicrop")
    if args.query_expand:
        mode_parts.append("query-expand")
    if args.ocr_rerank:
        mode_parts.append("OCR-rerank")
    if args.two_stage:
        mode_parts.append("two-stage")
    else:
        mode_parts.append("SKU-prototype")
    mode = " + ".join(mode_parts)

    print()
    print(f"{'=' * 60}")
    print(f"  Top-{args.top_k} SKU matches for: {image_path.name} ({mode})")
    print(f"{'=' * 60}")
    print()

    if args.ocr_rerank:
        for rank, (pid, final, vscore, tscore) in enumerate(results, 1):
            print(f"  #{rank}  Product {pid:>3s}  |  final: {final*100:5.1f}%  visual: {vscore*100:5.1f}%  text: {tscore*100:5.1f}%")
        if query_text.strip():
            print(f"\n  OCR text: {query_text[:100]!r}")
        else:
            print(f"\n  (No OCR text detected)")
    else:
        raw_scores = []
        for rank, item in enumerate(results, 1):
            pid, score = item[0], item[1]
            similarity_pct = max(0.0, score) * 100
            raw_scores.append(score)
            print(f"  #{rank}  Product {pid:>3s}  |  similarity: {similarity_pct:5.1f}%")

    # Confidence display
    if args.threshold is not None or args.min_margin is not None:
        st = args.threshold if args.threshold is not None else DEFAULT_SCORE_THRESHOLD
        mm = args.min_margin if args.min_margin is not None else DEFAULT_MIN_MARGIN
        if args.ocr_rerank:
            conf_scores = np.array([f for _, f, _, _ in results])
        else:
            conf_scores = np.array(raw_scores)
        if len(conf_scores) >= 2:
            conf = compute_confidence(conf_scores, st, mm)
            label = "CONFIDENT" if conf["is_confident"] else "LOW CONFIDENCE"
            print(f"\n  Confidence: {label}")
            print(f"    top-1 score: {conf['top1_score']:.3f}, margin: {conf['margin']:.3f}")
            if conf["rejection_reason"]:
                print(f"    reason: {conf['rejection_reason']}")

    print()
    print(f"  Timing: embed={embed_time * 1000:.0f}ms, search={search_time * 1000:.2f}ms")


if __name__ == "__main__":
    main()
