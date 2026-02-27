"""
Evaluate retrieval accuracy across GroZi-120 in-situ test images.

Modes:
  default   — SKU prototype search with multi-proto aggregation (search_sku)
  two-stage — coarse SKU search → image-level rerank (search_two_stage)

Usage:
    python scripts/eval_retrieval.py --multicrop
    python scripts/eval_retrieval.py --multicrop --two-stage
    python scripts/eval_retrieval.py --multicrop --ocr-rerank
    python scripts/eval_retrieval.py --multicrop --adapter catalog/adapter.pt
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
    load_index, load_labels, load_sku_index, load_sku_labels,
    search, search_sku, search_two_stage, expand_query, compute_confidence,
)


def collect_test_set(insitu_dir: Path, per_sku: int, seed: int) -> list[tuple[str, str]]:
    rng = random.Random(seed)
    test_set = []
    if not insitu_dir.exists():
        raise FileNotFoundError(f"In-situ directory not found: {insitu_dir}.")
    for product_dir in sorted(
        insitu_dir.iterdir(),
        key=lambda p: int(p.name) if p.name.isdigit() else p.name,
    ):
        if not product_dir.is_dir():
            continue
        product_id = product_dir.name
        images = sorted(
            f for f in product_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not images:
            continue
        sampled = rng.sample(images, min(per_sku, len(images)))
        for img_path in sampled:
            test_set.append((product_id, str(img_path)))
    return test_set


def _find_rank(ranked_pids: list[str], gt_pid: str) -> int:
    for r, pid in enumerate(ranked_pids, 1):
        if pid == gt_pid:
            return r
    return 0


def evaluate_sku(test_set, embedder, sku_index, sku_labels, top_k, multicrop,
                 score_threshold=None, min_margin=None):
    results = []
    st = score_threshold if score_threshold is not None else DEFAULT_SCORE_THRESHOLD
    mm = min_margin if min_margin is not None else DEFAULT_MIN_MARGIN
    do_cal = score_threshold is not None or min_margin is not None

    for i, (gt_pid, image_path) in enumerate(test_set):
        query_emb = embedder.embed_multicrop(image_path) if multicrop else embedder.embed_file(image_path)
        ranked_pids, scores_list = search_sku(sku_index, sku_labels, query_emb, top_k=top_k)
        rank = _find_rank(ranked_pids, gt_pid)
        result = {
            "hit_at_1": rank == 1,
            "hit_at_5": 1 <= rank <= 5,
            "reciprocal_rank": 1.0 / rank if rank > 0 else 0.0,
        }
        if do_cal and scores_list:
            conf = compute_confidence(np.array(scores_list), st, mm)
            result["is_confident"] = conf["is_confident"]
            result["top1_score"] = conf["top1_score"]
            result["margin"] = conf["margin"]
        results.append(result)
        if (i + 1) % 100 == 0:
            print(f"  [{i + 1}/{len(test_set)}]")
    return _aggregate(results)


def evaluate_two_stage(test_set, embedder, sku_index, sku_labels,
                       image_index, image_embeddings, pid_to_img_indices,
                       top_k, multicrop, n_coarse=20):
    results = []
    for i, (gt_pid, image_path) in enumerate(test_set):
        query_emb = embedder.embed_multicrop(image_path) if multicrop else embedder.embed_file(image_path)
        ranked_pids, _ = search_two_stage(
            sku_index, sku_labels, image_index, image_embeddings,
            pid_to_img_indices, query_emb, top_k=top_k, n_coarse=n_coarse,
        )
        rank = _find_rank(ranked_pids, gt_pid)
        results.append({
            "hit_at_1": rank == 1,
            "hit_at_5": 1 <= rank <= 5,
            "reciprocal_rank": 1.0 / rank if rank > 0 else 0.0,
        })
        if (i + 1) % 100 == 0:
            print(f"  [{i + 1}/{len(test_set)}]")
    return _aggregate(results)


def evaluate_sku_ocr(test_set, embedder, sku_index, sku_labels, top_k, multicrop,
                     ocr_reader, sku_ocr_texts, alpha, rerank_depth,
                     score_threshold=None, min_margin=None):
    from src.ocr import extract_text, text_rerank
    st = score_threshold if score_threshold is not None else DEFAULT_SCORE_THRESHOLD
    mm = min_margin if min_margin is not None else DEFAULT_MIN_MARGIN
    do_cal = score_threshold is not None or min_margin is not None
    results = []
    ocr_hit_count = 0

    for i, (gt_pid, image_path) in enumerate(test_set):
        query_emb = embedder.embed_multicrop(image_path) if multicrop else embedder.embed_file(image_path)
        candidate_pids, candidate_scores = search_sku(
            sku_index, sku_labels, query_emb, top_k=rerank_depth
        )
        query_text = extract_text(image_path, ocr_reader)

        if query_text.strip():
            reranked = text_rerank(query_text, candidate_pids, candidate_scores, sku_ocr_texts, alpha=alpha)
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
        if do_cal and len(final_scores) >= 2:
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
        metrics["conf_top1"] = sum(r["hit_at_1"] for r in confident) / n_conf if n_conf else 0.0
        metrics["conf_mrr"] = sum(r["reciprocal_rank"] for r in confident) / n_conf if n_conf else 0.0
        metrics["avg_top1_score"] = np.mean([r["top1_score"] for r in results])
        metrics["avg_margin"] = np.mean([r["margin"] for r in results])
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate GroZi-120 retrieval.")
    parser.add_argument("--per-sku", type=int, default=5)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--multicrop", action="store_true")
    parser.add_argument("--two-stage", action="store_true")
    parser.add_argument("--ocr-rerank", action="store_true")
    parser.add_argument("--ocr-alpha", type=float, default=OCR_ALPHA)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--min-margin", type=float, default=None)
    parser.add_argument("--adapter", type=str, default=None,
                        help="CLIPAdapter checkpoint (must match the index).")
    args = parser.parse_args()

    print("Collecting test set ...")
    test_set = collect_test_set(INSITU_DIR, args.per_sku, args.seed)
    n_products = len(set(pid for pid, _ in test_set))
    print(f"  {len(test_set)} queries ({args.per_sku}/SKU × {n_products} SKUs)\n")

    print("Loading embedder ...", end=" ", flush=True)
    t0 = time.time()
    embedder = Embedder()
    embedder.load(adapter_path=args.adapter)
    print(f"({time.time() - t0:.1f}s)")

    print("Loading SKU index ...", end=" ", flush=True)
    t0 = time.time()
    sku_index = load_sku_index()
    sku_labels = load_sku_labels()
    n_skus = len(set(l["product_id"] for l in sku_labels))
    print(f"({time.time() - t0:.1f}s, {sku_index.ntotal} prototypes, {n_skus} SKUs)")

    image_embeddings = pid_to_img_indices = image_index = None
    if args.two_stage:
        print("Loading image index ...", end=" ", flush=True)
        t0 = time.time()
        image_labels = load_labels()
        image_embeddings = np.load(EMBEDDINGS_PATH).astype(np.float32)
        faiss.normalize_L2(image_embeddings)
        image_index = load_index()
        pid_to_img_indices = defaultdict(list)
        for lbl in image_labels:
            pid_to_img_indices[lbl["product_id"]].append(lbl["index"])
        pid_to_img_indices = {k: np.array(v) for k, v in pid_to_img_indices.items()}
        print(f"({time.time() - t0:.1f}s, {len(image_labels)} images)")

    sku_ocr_texts = ocr_reader = None
    if args.ocr_rerank:
        from src.ocr import load_ocr_texts
        print("Loading OCR index ...", end=" ", flush=True)
        sku_ocr_texts = load_ocr_texts(OCR_TEXTS_PATH)
        print(f"({len(sku_ocr_texts)} SKUs)")
        print("Initializing EasyOCR ...", end=" ", flush=True)
        t0 = time.time()
        import easyocr
        ocr_reader = easyocr.Reader(["en"], gpu=True, verbose=False)
        print(f"({time.time() - t0:.1f}s)")

    print()

    mode_parts = []
    if args.multicrop: mode_parts.append("multicrop")
    if args.adapter:   mode_parts.append("adapter")
    if args.ocr_rerank: mode_parts.append(f"BM25-OCR(α={args.ocr_alpha})")
    if args.two_stage: mode_parts.append("two-stage")
    else:              mode_parts.append("SKU-prototype")
    mode = " + ".join(mode_parts) if mode_parts else "baseline"

    print(f"Running evaluation (top-{args.top_k}, {mode}) ...")
    t0 = time.time()

    if args.ocr_rerank:
        metrics = evaluate_sku_ocr(
            test_set, embedder, sku_index, sku_labels,
            args.top_k, args.multicrop, ocr_reader, sku_ocr_texts,
            args.ocr_alpha, OCR_RERANK_DEPTH,
            score_threshold=args.threshold, min_margin=args.min_margin,
        )
    elif args.two_stage:
        metrics = evaluate_two_stage(
            test_set, embedder, sku_index, sku_labels,
            image_index, image_embeddings, pid_to_img_indices,
            args.top_k, args.multicrop,
        )
    else:
        metrics = evaluate_sku(
            test_set, embedder, sku_index, sku_labels,
            args.top_k, args.multicrop,
            score_threshold=args.threshold, min_margin=args.min_margin,
        )

    total_time = time.time() - t0

    print()
    print("=" * 44)
    print("  Retrieval Evaluation Results")
    print("=" * 44)
    print()
    print(f"  Queries:      {metrics['n']} ({args.per_sku}/SKU × {n_products} SKUs)")
    print(f"  Mode:         {mode}")
    print(f"  Search depth: top-{args.top_k}")
    print()
    print(f"  {'Metric':<20s} {'Value':>8s}")
    print(f"  {'─'*20} {'─'*8}")
    print(f"  {'Top-1 Accuracy':<20s} {metrics['top1']*100:>7.1f}%")
    print(f"  {'Top-5 Accuracy':<20s} {metrics['top5']*100:>7.1f}%")
    print(f"  {'MRR':<20s} {metrics['mrr']:>8.3f}")

    if "coverage" in metrics:
        print()
        print(f"  {'--- Calibration ---':<20s}")
        print(f"  {'Coverage':<20s} {metrics['coverage']*100:>7.1f}%")
        print(f"  {'Conf. Top-1 Acc':<20s} {metrics['conf_top1']*100:>7.1f}%")
        print(f"  {'Avg Top-1 Score':<20s} {metrics['avg_top1_score']:>8.3f}")
        print(f"  {'Avg Margin':<20s} {metrics['avg_margin']:>8.3f}")

    if "ocr_reranks" in metrics:
        print(f"\n  {'BM25 Reranks':<20s} {metrics['ocr_reranks']:>8d}")

    print()
    print(f"  Timing: {total_time:.1f}s, {total_time/metrics['n']*1000:.0f}ms/query")


if __name__ == "__main__":
    main()
