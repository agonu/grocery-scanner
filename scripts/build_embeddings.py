"""Build catalog embeddings from in-vitro images."""

import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import INVITRO_DIR, CATALOG_DIR, EMBEDDINGS_PATH, IMAGE_EXTENSIONS
from src.embedder import Embedder
from src.index import save_labels


def collect_catalog_images(invitro_dir: Path) -> list[dict]:
    """Walk the in-vitro directory and collect image metadata."""
    if not invitro_dir.exists():
        raise FileNotFoundError(
            f"In-vitro directory not found: {invitro_dir}. Run download_dataset.py first."
        )

    entries = []
    for product_dir in sorted(
        invitro_dir.iterdir(),
        key=lambda p: int(p.name) if p.name.isdigit() else p.name,
    ):
        if not product_dir.is_dir():
            continue
        product_id = product_dir.name
        image_files = sorted(
            f for f in product_dir.rglob("*")
            if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
        )
        for img_path in image_files:
            entries.append({
                "product_id": product_id,
                "image_path": str(img_path),
            })
    return entries


def main() -> None:
    print("=== Build Catalog Embeddings ===\n")

    # 1. Collect images
    print("Scanning in-vitro images ...")
    entries = collect_catalog_images(INVITRO_DIR)
    products = set(e["product_id"] for e in entries)
    print(f"  Found {len(entries)} images across {len(products)} products\n")

    # 2. Load embedder
    print("Loading CLIP ViT-B/32 ...")
    embedder = Embedder()
    embedder.load()
    print("  Model ready.\n")

    # 3. Compute embeddings in batches
    print(f"Computing embeddings ({len(entries)} images) ...")
    start = time.time()

    batch_size = 64
    all_embeddings = []
    valid_entries = []

    for i in range(0, len(entries), batch_size):
        batch_entries = entries[i : i + batch_size]
        batch_images = []
        batch_valid = []

        for entry in batch_entries:
            try:
                img = Image.open(entry["image_path"])
                batch_images.append(img)
                batch_valid.append(entry)
            except Exception as e:
                print(f"  WARNING: Skipping {entry['image_path']}: {e}")

        if batch_images:
            embs = embedder.embed_batch(batch_images)
            all_embeddings.append(embs)
            valid_entries.extend(batch_valid)

        done = min(i + batch_size, len(entries))
        elapsed = time.time() - start
        rate = done / elapsed if elapsed > 0 else 0
        print(f"  [{done}/{len(entries)}] {rate:.1f} img/s")

    embeddings = np.concatenate(all_embeddings, axis=0).astype(np.float32)
    elapsed = time.time() - start
    print(f"  Done: {embeddings.shape} in {elapsed:.1f}s ({len(valid_entries)/elapsed:.1f} img/s)\n")

    # 4. Add index field to labels
    for i, entry in enumerate(valid_entries):
        entry["index"] = i

    # 5. Save
    CATALOG_DIR.mkdir(parents=True, exist_ok=True)
    np.save(EMBEDDINGS_PATH, embeddings)
    save_labels(valid_entries)
    print(f"Saved embeddings: {EMBEDDINGS_PATH} ({embeddings.nbytes / 1024 / 1024:.1f} MB)")
    print(f"Saved labels:     {len(valid_entries)} entries")
    print("\nDone. Next: run build_index.py")


if __name__ == "__main__":
    main()
