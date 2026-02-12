"""Build FAISS indexes (image-level + SKU prototypes) from pre-computed embeddings."""

import sys
from collections import defaultdict
from pathlib import Path

import faiss
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import EMBEDDINGS_PATH, SKU_EMBEDDINGS_PATH
from src.index import (
    build_index, save_index, load_labels,
    save_sku_index, save_sku_labels,
)


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

    # ── SKU prototype index ──────────────────────────────
    print("[2/2] Building SKU prototype index ...")
    product_embs = defaultdict(list)
    for i, label in enumerate(labels):
        product_embs[label["product_id"]].append(embeddings[i])

    sku_embeddings_list = []
    sku_labels_list = []
    for idx, (pid, embs) in enumerate(
        sorted(product_embs.items(), key=lambda x: int(x[0]) if x[0].isdigit() else x[0])
    ):
        avg = np.mean(embs, axis=0).astype(np.float32)
        avg /= np.linalg.norm(avg)
        sku_embeddings_list.append(avg)
        sku_labels_list.append({"index": idx, "product_id": pid})

    sku_embeddings = np.stack(sku_embeddings_list)
    np.save(SKU_EMBEDDINGS_PATH, sku_embeddings)

    sku_index = build_index(sku_embeddings)
    save_sku_index(sku_index)
    save_sku_labels(sku_labels_list)
    print(f"  {sku_index.ntotal} SKU prototypes → catalog/sku.index")
    print(f"  Avg {np.mean([len(v) for v in product_embs.values()]):.1f} images/SKU\n")

    # Sanity check
    print("Sanity check: querying SKU prototype #0 ...")
    query = sku_embeddings[0:1].copy()
    faiss.normalize_L2(query)
    scores, indices = sku_index.search(query, 3)
    expected = sku_labels_list[0]["product_id"]
    top_pid = sku_labels_list[indices[0][0]]["product_id"]
    print(f"  Top-1: score={scores[0][0]:.4f}, product={top_pid}")
    if top_pid == expected:
        print(f"  PASS: matches expected product ({expected})")
    else:
        print(f"  WARN: Expected {expected}, got {top_pid}")

    print("\nDone.")


if __name__ == "__main__":
    main()
