"""
Evaluate retrieval accuracy across GroZi-120 in-situ test images.

Supports three modes:
  - SKU prototype search (default): search 120-vector SKU index directly
  - Two-stage: coarse SKU search → rerank by max image-level similarity
  - Legacy image-level: search 2K-vector image index + vote (--legacy)

Usage:
    python scripts/eval_retrieval.py --multicrop
    python scripts/eval_retrieval.py --multicrop --two-stage
"""

import argparse
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    INSITU_DIR, IMAGE_EXTENSIONS, EMBEDDINGS_PATH, SKU_EMBEDDINGS_PATH,
    DEFAULT_SCORE_THRESHOLD, DEFAULT_MIN_MARGIN,
    OCR_TEXTS_PATH, OCR_ALPHA, OCR_RERANK_DEPTH,
)
from src.embedder import Embedder
from src.index import (
    load_index, load_labels, load_sku_index, load_sku_labels, search,
    expand_query, compute_confidence,
)


def collect_test_set(
    insitu_dir: Path, per_sku: int, seed: int
) -> list[tuple[str, str]]:
    """Sample up to `per_sku` in-situ images per product."""
    rng = random.Random(seed)
    test_set = []

    if not insitu_dir.exists():
        raise FileNotFoundError(
            f"In-situ directory not found: {insitu_dir}. Run download_dataset.py first."
        )

    for product_dir in sorted(
        insitu_dir.iterdir(),
        key=lambda p: int(p.name) if p.name.isdigit() else p.name,
    ):
        if not product_dir.is_dir():
            continue
        product_id = product_dir.name
        images = sorted(
            f
            for f in product_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not images:
            continue
        sampled = rng.sample(images, min(per_sku, len(images)))
        for img_path in sampled:
            test_set.append((product_id, str(img_path)))

    return test_set


def _find_rank(ranked_pids: list[str], gt_pid: str) -> int:
    """Find 1-based rank of gt_pid in ranked list. Returns 0 if not found."""
    for r, pid in enumerate(ranked_pids, 1):
        if pid == gt_pid:
            return r
    return 0


def evaluate_sku(test_set, embedder, sku_index, sku_labels, top_k, multicrop,
                 query_expand=False, sku_embeddings=None,
                 score_threshold=None, min_margin=None):
    """Evaluate using SKU prototype index directly."""
    results = []
    do_calibration = score_threshold is not None or min_margin is not None
    st = score_threshold if score_threshold is not None else DEFAULT_SCORE_THRESHOLD
    mm = min_margin if min_margin is not None else DEFAULT_MIN_MARGIN

    for i, (gt_pid, image_path) in enumerate(test_set):
        query_emb = embedder.embed_multicrop(image_path) if multicrop else embedder.embed_file(image_path)

        if query_expand and sku_embeddings is not None:
            query_emb = expand_query(query_emb, sku_index, sku_embeddings)

        scores, indices = search(sku_index, query_emb, top_k=top_k)
        ranked_pids = [sku_labels[idx]["product_id"] for idx in indices if idx >= 0]
        rank = _find_rank(ranked_pids, gt_pid)

        result = {
            "hit_at_1": rank == 1,
            "hit_at_5": 1 <= rank <= 5,
            "reciprocal_rank": 1.0 / rank if rank > 0 else 0.0,
        }

        if do_calibration:
            conf = compute_confidence(scores, st, mm)
            result["is_confident"] = conf["is_confident"]
            result["top1_score"] = conf["top1_score"]
            result["margin"] = conf["margin"]

        results.append(result)
        if (i + 1) % 100 == 0:
            print(f"  [{i + 1}/{len(test_set)}]")

    return _aggregate(results)


def evaluate_two_stage(
    test_set, embedder, sku_index, sku_labels,
    image_embeddings, pid_to_img_indices, top_k, multicrop, n_candidates=10,
):
    """Evaluate with two-stage: coarse SKU search → image-level rerank."""
    results = []
    for i, (gt_pid, image_path) in enumerate(test_set):
        query_emb = embedder.embed_multicrop(image_path) if multicrop else embedder.embed_file(image_path)

        # Stage 1: coarse SKU search
        sku_scores, sku_indices = search(sku_index, query_emb, top_k=n_candidates)
        candidate_pids = [sku_labels[idx]["product_id"] for idx in sku_indices if idx >= 0]

        # Stage 2: rerank by max image-level similarity
        query = query_emb.astype(np.float32).reshape(1, -1)
        faiss.normalize_L2(query)

        reranked = []
        for pid in candidate_pids:
            img_idx = pid_to_img_indices.get(pid)
            if img_idx is None or len(img_idx) == 0:
                continue
            pid_embs = image_embeddings[img_idx]
            max_score = float((query @ pid_embs.T).max())
            reranked.append((pid, max_score))

        reranked.sort(key=lambda x: x[1], reverse=True)
        ranked_pids = [pid for pid, _ in reranked[:top_k]]
        rank = _find_rank(ranked_pids, gt_pid)

        results.append({
            "hit_at_1": rank == 1,
            "hit_at_5": 1 <= rank <= 5,
            "reciprocal_rank": 1.0 / rank if rank > 0 else 0.0,
        })
        if (i + 1) % 100 == 0:
            print(f"  [{i + 1}/{len(test_set)}]")

    return _aggregate(results)


def evaluate_sku_ocr(
    test_set, embedder, sku_index, sku_labels, top_k, multicrop,
    query_expand, sku_embeddings,
    ocr_reader, sku_ocr_texts, alpha, rerank_depth,
    score_threshold=None, min_margin=None,
):
    """Evaluate with OCR-fused reranking on top of SKU prototype search."""
    from src.ocr import extract_text, text_rerank

    do_calibration = score_threshold is not None or min_margin is not None
    st = score_threshold if score_threshold is not None else DEFAULT_SCORE_THRESHOLD
    mm = min_margin if min_margin is not None else DEFAULT_MIN_MARGIN

    results = []
    ocr_hit_count = 0

    for i, (gt_pid, image_path) in enumerate(test_set):
        query_emb = embedder.embed_multicrop(image_path) if multicrop else embedder.embed_file(image_path)

        if query_expand and sku_embeddings is not None:
            query_emb = expand_query(query_emb, sku_index, sku_embeddings)

        scores, indices = search(sku_index, query_emb, top_k=rerank_depth)
        candidate_pids = [sku_labels[idx]["product_id"] for idx in indices if idx >= 0]
        candidate_scores = [float(scores[j]) for j, idx in enumerate(indices) if idx >= 0]

        query_text = extract_text(image_path, ocr_reader)

        if query_text.strip():
            reranked = text_rerank(
                query_text, candidate_pids, candidate_scores,
                sku_ocr_texts, alpha=alpha,
            )
            ranked_pids = [pid for pid, _, _, _ in reranked[:top_k]]
            final_scores = np.array([fs for _, fs, _, _ in reranked[:top_k]])

            if ranked_pids and candidate_pids and ranked_pids[0] != candidate_pids[0]:
                ocr_hit_count += 1
        else:
            ranked_pids = candidate_pids[:top_k]
            final_scores = np.array(candidate_scores[:top_k])

        rank = _find_rank(ranked_pids, gt_pid)

        result = {
            "hit_at_1": rank == 1,
            "hit_at_5": 1 <= rank <= 5,
            "reciprocal_rank": 1.0 / rank if rank > 0 else 0.0,
        }

        if do_calibration and len(final_scores) >= 2:
            conf = compute_confidence(final_scores, st, mm)
            result["is_confident"] = conf["is_confident"]
            result["top1_score"] = conf["top1_score"]
            result["margin"] = conf["margin"]

        results.append(result)
        if (i + 1) % 50 == 0:
            print(f"  [{i + 1}/{len(test_set)}]")

    metrics = _aggregate(results)
    metrics["ocr_reranks"] = ocr_hit_count
    return metrics


def _aggregate(results):
    n = len(results)
    metrics = {
        "top1": sum(r["hit_at_1"] for r in results) / n,
        "top5": sum(r["hit_at_5"] for r in results) / n,
        "mrr": sum(r["reciprocal_rank"] for r in results) / n,
        "n": n,
    }

    if results and "is_confident" in results[0]:
        confident = [r for r in results if r["is_confident"]]
        n_conf = len(confident)
        metrics["coverage"] = n_conf / n
        if n_conf > 0:
            metrics["conf_top1"] = sum(r["hit_at_1"] for r in confident) / n_conf
            metrics["conf_mrr"] = sum(r["reciprocal_rank"] for r in confident) / n_conf
        else:
            metrics["conf_top1"] = 0.0
            metrics["conf_mrr"] = 0.0
        metrics["avg_top1_score"] = np.mean([r["top1_score"] for r in results])
        metrics["avg_margin"] = np.mean([r["margin"] for r in results])

    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GroZi-120 retrieval.")
    parser.add_argument("--per-sku", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--multicrop", action="store_true", help="Letterbox + multi-crop query.")
    parser.add_argument("--two-stage", action="store_true", help="Two-stage: SKU coarse → image rerank.")
    parser.add_argument("--query-expand", action="store_true", help="Expand query by averaging with top-3 SKU matches.")
    parser.add_argument("--threshold", type=float, default=None, help="Min top-1 score for confident prediction.")
    parser.add_argument("--min-margin", type=float, default=None, help="Min margin between top-1 and top-2 scores.")
    parser.add_argument("--ocr-rerank", action="store_true", help="Rerank top matches using OCR text similarity.")
    parser.add_argument("--ocr-alpha", type=float, default=OCR_ALPHA, help=f"Visual weight in OCR fusion (default: {OCR_ALPHA}).")
    args = parser.parse_args()

    # 1. Collect test set
    print("Collecting test set ...")
    test_set = collect_test_set(INSITU_DIR, args.per_sku, args.seed)
    n_products = len(set(pid for pid, _ in test_set))
    print(f"  {len(test_set)} queries ({args.per_sku}/SKU x {n_products} SKUs)\n")

    # 2. Load model
    print("Loading model ...", end=" ", flush=True)
    t0 = time.time()
    embedder = Embedder()
    embedder.load()
    print(f"({time.time() - t0:.1f}s)")

    # 3. Load indexes
    print("Loading SKU index ...", end=" ", flush=True)
    t0 = time.time()
    sku_index = load_sku_index()
    sku_labels = load_sku_labels()
    print(f"({time.time() - t0:.1f}s, {sku_index.ntotal} SKUs)")

    sku_embeddings = None
    if args.query_expand:
        sku_embeddings = np.load(SKU_EMBEDDINGS_PATH).astype(np.float32)

    sku_ocr_texts = None
    ocr_reader = None
    if args.ocr_rerank:
        from src.ocr import load_ocr_texts
        print("Loading OCR text index ...", end=" ", flush=True)
        t0 = time.time()
        sku_ocr_texts = load_ocr_texts(OCR_TEXTS_PATH)
        print(f"({time.time() - t0:.1f}s, {len(sku_ocr_texts)} SKUs)")

        print("Initializing EasyOCR ...", end=" ", flush=True)
        t0 = time.time()
        import easyocr
        ocr_reader = easyocr.Reader(["en"], gpu=True, verbose=False)
        print(f"({time.time() - t0:.1f}s)")

    image_embeddings = None
    pid_to_img_indices = None
    if args.two_stage:
        print("Loading image index ...", end=" ", flush=True)
        t0 = time.time()
        image_labels = load_labels()
        image_embeddings = np.load(EMBEDDINGS_PATH).astype(np.float32)
        faiss.normalize_L2(image_embeddings)
        pid_to_img_indices = defaultdict(list)
        for lbl in image_labels:
            pid_to_img_indices[lbl["product_id"]].append(lbl["index"])
        # Convert to numpy arrays for fast slicing
        pid_to_img_indices = {k: np.array(v) for k, v in pid_to_img_indices.items()}
        print(f"({time.time() - t0:.1f}s, {len(image_labels)} images)")
    print()

    # 4. Evaluate
    mode_parts = []
    if args.multicrop:
        mode_parts.append("multicrop")
    if args.query_expand:
        mode_parts.append("query-expand")
    if args.ocr_rerank:
        mode_parts.append(f"OCR-rerank(a={args.ocr_alpha})")
    if args.two_stage:
        mode_parts.append("two-stage")
    else:
        mode_parts.append("SKU-prototype")
    mode = " + ".join(mode_parts) if mode_parts else "baseline"

    print(f"Running evaluation (top-{args.top_k}, {mode}) ...")
    t0 = time.time()

    if args.ocr_rerank:
        metrics = evaluate_sku_ocr(
            test_set, embedder, sku_index, sku_labels,
            args.top_k, args.multicrop,
            args.query_expand, sku_embeddings,
            ocr_reader, sku_ocr_texts, args.ocr_alpha, OCR_RERANK_DEPTH,
            score_threshold=args.threshold, min_margin=args.min_margin,
        )
    elif args.two_stage:
        metrics = evaluate_two_stage(
            test_set, embedder, sku_index, sku_labels,
            image_embeddings, pid_to_img_indices,
            args.top_k, args.multicrop,
        )
    else:
        metrics = evaluate_sku(
            test_set, embedder, sku_index, sku_labels,
            args.top_k, args.multicrop,
            query_expand=args.query_expand,
            sku_embeddings=sku_embeddings,
            score_threshold=args.threshold,
            min_margin=args.min_margin,
        )

    total_time = time.time() - t0

    # 5. Print results
    print()
    print("=" * 44)
    print("  Retrieval Evaluation Results")
    print("=" * 44)
    print()
    print(f"  Queries:      {metrics['n']} ({args.per_sku}/SKU x {n_products} SKUs)")
    print(f"  Mode:         {mode}")
    print(f"  Search depth: top-{args.top_k}")
    print()
    print(f"  {'Metric':<20s} {'Value':>8s}")
    print(f"  {'─' * 20} {'─' * 8}")
    print(f"  {'Top-1 Accuracy':<20s} {metrics['top1'] * 100:>7.1f}%")
    print(f"  {'Top-5 Accuracy':<20s} {metrics['top5'] * 100:>7.1f}%")
    print(f"  {'MRR':<20s} {metrics['mrr']:>8.3f}")

    if "coverage" in metrics:
        print()
        print(f"  {'--- Calibration ---':<20s}")
        print(f"  {'Coverage':<20s} {metrics['coverage'] * 100:>7.1f}%")
        print(f"  {'Conf. Top-1 Acc':<20s} {metrics['conf_top1'] * 100:>7.1f}%")
        print(f"  {'Conf. MRR':<20s} {metrics['conf_mrr']:>8.3f}")
        print(f"  {'Avg Top-1 Score':<20s} {metrics['avg_top1_score']:>8.3f}")
        print(f"  {'Avg Margin':<20s} {metrics['avg_margin']:>8.3f}")

    if "ocr_reranks" in metrics:
        print()
        print(f"  {'OCR Reranks':<20s} {metrics['ocr_reranks']:>8d}")

    print()
    print(f"  Timing: {total_time:.1f}s total, {total_time / metrics['n'] * 1000:.0f}ms/query")


if __name__ == "__main__":
    main()
