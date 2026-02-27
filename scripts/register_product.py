"""Register new products into the catalog from reference images.

Embeds one or more reference images, computes a mean prototype, appends it
to the existing SKU FAISS index and sku_labels.json, and saves both.

No full rebuild is required for new products.  To update an existing product
re-run with the same product_id — the old entry is removed first.

Usage:
    python scripts/register_product.py SKU-ABC img1.jpg img2.jpg img3.jpg
    python scripts/register_product.py --adapter catalog/adapter.pt SKU-ABC img1.jpg
"""

import argparse
import json
import sys
from pathlib import Path

import faiss
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import SKU_INDEX_PATH, SKU_LABELS_PATH, EMBEDDING_DIM, ADAPTER_PATH
from src.embedder import Embedder
from src.index import save_sku_index, save_sku_labels


def _rebuild_without(
    old_index: faiss.IndexFlatIP,
    old_labels: list[dict],
    remove_pid: str,
) -> tuple[faiss.IndexFlatIP, list[dict]]:
    """Return a new index + labels with all rows for remove_pid stripped out."""
    keep_rows = [i for i, l in enumerate(old_labels) if l["product_id"] != remove_pid]
    if not keep_rows:
        new_index = faiss.IndexFlatIP(EMBEDDING_DIM)
        return new_index, []

    # Reconstruct kept vectors from the old index
    all_vecs = np.zeros((old_index.ntotal, EMBEDDING_DIM), dtype=np.float32)
    old_index.reconstruct_n(0, old_index.ntotal, all_vecs)
    kept_vecs = np.ascontiguousarray(all_vecs[keep_rows])

    new_index = faiss.IndexFlatIP(EMBEDDING_DIM)
    new_index.add(kept_vecs)

    new_labels = []
    for new_i, old_i in enumerate(keep_rows):
        new_labels.append({"index": new_i, "product_id": old_labels[old_i]["product_id"]})

    return new_index, new_labels


def main() -> None:
    parser = argparse.ArgumentParser(description="Register a new product into the catalog.")
    parser.add_argument("product_id", help="Product / SKU identifier string.")
    parser.add_argument("images", nargs="+", help="Reference image paths.")
    parser.add_argument(
        "--adapter", type=str, default=None,
        help="Optional adapter checkpoint (must match the one used to build the index).",
    )
    args = parser.parse_args()

    image_paths = [Path(p) for p in args.images]
    missing = [p for p in image_paths if not p.exists()]
    if missing:
        print(f"ERROR: Image(s) not found: {[str(p) for p in missing]}", file=sys.stderr)
        sys.exit(1)

    adapter_path = args.adapter or (str(ADAPTER_PATH) if ADAPTER_PATH.exists() else None)

    # ── Load model ────────────────────────────────────────
    print("Loading CLIP embedder ...", end=" ", flush=True)
    embedder = Embedder()
    embedder.load(adapter_path=adapter_path)
    print("done")

    # ── Embed reference images ────────────────────────────
    print(f"Embedding {len(image_paths)} reference image(s) ...")
    images = [Image.open(p).convert("RGB") for p in image_paths]
    embs = embedder.embed_batch(images)

    prototype = embs.mean(axis=0).astype(np.float32)
    norm = np.linalg.norm(prototype)
    if norm > 0:
        prototype /= norm

    # ── Load existing index ───────────────────────────────
    if SKU_INDEX_PATH.exists() and SKU_LABELS_PATH.exists():
        sku_index = faiss.read_index(str(SKU_INDEX_PATH))
        with open(SKU_LABELS_PATH) as f:
            sku_labels = json.load(f)
        print(f"Loaded existing index: {sku_index.ntotal} prototypes, {len(set(l['product_id'] for l in sku_labels))} SKUs")
    else:
        sku_index = faiss.IndexFlatIP(EMBEDDING_DIM)
        sku_labels = []
        print("No existing index found — creating new one.")

    # Remove old entry for this product if it exists
    existing = [l for l in sku_labels if l["product_id"] == args.product_id]
    if existing:
        print(f"  Removing {len(existing)} existing prototype(s) for '{args.product_id}' ...")
        sku_index, sku_labels = _rebuild_without(sku_index, sku_labels, args.product_id)

    # ── Append new prototype ──────────────────────────────
    new_idx = sku_index.ntotal
    sku_index.add(prototype.reshape(1, -1))
    sku_labels.append({"index": new_idx, "product_id": args.product_id})

    # ── Save ──────────────────────────────────────────────
    save_sku_index(sku_index)
    save_sku_labels(sku_labels)

    n_skus = len(set(l["product_id"] for l in sku_labels))
    print(f"Registered '{args.product_id}' — catalog now has {sku_index.ntotal} prototypes across {n_skus} SKUs.")


if __name__ == "__main__":
    main()
