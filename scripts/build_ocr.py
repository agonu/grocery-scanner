"""Build OCR text index from catalog (in-vitro) images.

Usage:
    python scripts/build_ocr.py

Produces: catalog/ocr_texts.json
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import OCR_TEXTS_PATH
from src.index import load_labels
from src.ocr import build_sku_ocr_texts, save_ocr_texts


def main() -> None:
    print("=== Build OCR Text Index ===\n")

    # 1. Load catalog labels (to get image paths per product)
    labels = load_labels()
    n_products = len(set(l["product_id"] for l in labels))
    print(f"Loaded {len(labels)} catalog images across {n_products} SKUs\n")

    # 2. Initialize easyocr
    print("Initializing EasyOCR ...", end=" ", flush=True)
    t0 = time.time()
    import easyocr
    reader = easyocr.Reader(["en"], gpu=True, verbose=False)
    print(f"({time.time() - t0:.1f}s)")

    # 3. OCR all catalog images
    print(f"\nRunning OCR on {len(labels)} images ...")
    t0 = time.time()
    ocr_texts = build_sku_ocr_texts(labels, reader)
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s ({len(labels) / elapsed:.1f} img/s)\n")

    # 4. Save
    save_ocr_texts(ocr_texts, OCR_TEXTS_PATH)

    # 5. Summary
    n_with_text = sum(1 for t in ocr_texts.values() if t.strip())
    avg_len = sum(len(t) for t in ocr_texts.values()) / max(len(ocr_texts), 1)
    print(f"Saved: {OCR_TEXTS_PATH}")
    print(f"  {len(ocr_texts)} SKUs, {n_with_text} with non-empty OCR text")
    print(f"  Avg text length: {avg_len:.0f} chars")

    # Show some examples
    print("\nSample OCR texts:")
    for pid in sorted(ocr_texts.keys(), key=lambda x: int(x) if x.isdigit() else x)[:5]:
        text = ocr_texts[pid][:80]
        print(f"  SKU {pid:>3s}: {text!r}")

    print("\nDone.")


if __name__ == "__main__":
    main()
