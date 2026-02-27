"""Build FAISS indexes (image-level + SKU multi-prototype) from pre-computed embeddings.

SKU index uses FAISS k-means to store SKU_NUM_PROTOTYPES cluster centres per SKU
instead of a single mean vector, capturing intra-class appearance variation.
"""

import sys
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import EMBEDDINGS_PATH, SKU_EMBEDDINGS_PATH, SKU_NUM_PROTOTYPES
from src.index import (
    build_index, save_index, load_labels,
    save_sku_index, save_sku_labels,
)


def cluster_prototypes(embeddings: np.ndarray, k: int) -> np.ndarray:
    """Return up to k L2-normalised cluster centres for the given embeddings.

    Uses FAISS k-means (GPU-agnostic, fast even on CPU for small clusters).
    Falls back to returning all embeddings when n ≤ k.
    """
    n, d = embeddings.shape
    if n <= k:
        return embeddings  # already fewer images than desired prototypes

    embs = np.ascontiguousarray(embeddings, dtype=np.float32)
    kmeans = faiss.Kmeans(d, k, niter=20, verbose=False, gpu=False, seed=42)
    kmeans.train(embs)
    centres = kmeans.centroids.astype(np.float32)  # (k, d)

    # L2-normalise each centre
    norms = np.linalg.norm(centres, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0)
    centres = centres / norms
    return centres


def main() -> None:
    print("=== Build FAISS Indexes ===\n")

    if not EMBEDDINGS_PATH.exists():
        print(f"ERROR: Embeddings not found at {EMBEDDINGS_PATH}")
        print("Run build_embeddings.py first.")
        sys.exit(1)

    embeddings = np.load(EMBEDDINGS_PATH)
    labels = load_labels()
    print(f"Loaded embeddings: {embeddings.shape} ({embeddings.nbytes / 1024 / 1024:.1f} MB)")
    print(f"Loaded labels:     {len(labels)} entries\n")

    # ── Image-level index ────────────────────────────────
    print("[1/2] Building image-level IndexFlatIP ...")
    image_index = build_index(embeddings)
    save_index(image_index)
    print(f"  {image_index.ntotal} vectors → catalog/catalog.index\n")

    # ── SKU multi-prototype index ────────────────────────
    k = SKU_NUM_PROTOTYPES
    print(f"[2/2] Building SKU prototype index (k={k} per SKU) ...")
    product_embs = defaultdict(list)
    for i, label in enumerate(labels):
        product_embs[label["product_id"]].append(embeddings[i])

    sku_embeddings_list = []
    sku_labels_list = []
    faiss_idx = 0
    n_skus_single = 0

    for pid in sorted(product_embs, key=lambda x: int(x) if x.isdigit() else x):
        embs = np.stack(product_embs[pid]).astype(np.float32)
        prototypes = cluster_prototypes(embs, k)
        if len(prototypes) < k:
            n_skus_single += 1

        for proto in prototypes:
            sku_embeddings_list.append(proto)
            sku_labels_list.append({"index": faiss_idx, "product_id": pid})
            faiss_idx += 1

    sku_embeddings = np.stack(sku_embeddings_list).astype(np.float32)
    np.save(SKU_EMBEDDINGS_PATH, sku_embeddings)

    sku_index = build_index(sku_embeddings)
    save_sku_index(sku_index)
    save_sku_labels(sku_labels_list)

    n_skus = len(product_embs)
    avg_protos = sku_index.ntotal / n_skus
    print(f"  {n_skus} SKUs × ≤{k} prototypes = {sku_index.ntotal} vectors → catalog/sku.index")
    print(f"  Avg prototypes/SKU: {avg_protos:.2f}  ({n_skus_single} SKUs had fewer than {k} images)\n")

    # Sanity check: query the first SKU prototype, expect it to rank #1
    print("Sanity check: querying SKU prototype #0 ...")
    query = sku_embeddings[0:1].copy()
    faiss.normalize_L2(query)
    scores, indices = sku_index.search(query, 3)
    hit_pid = sku_labels_list[indices[0][0]]["product_id"]
    expected_pid = sku_labels_list[0]["product_id"]
    print(f"  Top-1: score={scores[0][0]:.4f}, product={hit_pid}")
    if hit_pid == expected_pid:
        print(f"  PASS: matches expected product ({expected_pid})")
    else:
        print(f"  WARN: Expected {expected_pid}, got {hit_pid}")

    print("\nDone.")


if __name__ == "__main__":
    main()
