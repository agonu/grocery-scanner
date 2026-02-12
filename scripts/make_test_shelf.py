"""
Compose a synthetic shelf image from catalog (inVitro) product images.

Pastes randomly selected catalog images side-by-side on a shelf-colored canvas.
Saves the composite image and a ground-truth JSON with bounding boxes.

Usage:
    python scripts/make_test_shelf.py --products 3,7,15,42,88 --output test_shelf.jpg
    python scripts/make_test_shelf.py --n-random 8 --output test_shelf.jpg
"""

import argparse
import json
import random
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import INVITRO_DIR, IMAGE_EXTENSIONS


def find_product_image(product_id: str) -> Path | None:
    """Find the first valid image for a product in the inVitro catalog."""
    product_dir = INVITRO_DIR / product_id
    if not product_dir.exists():
        return None
    for img_path in sorted(product_dir.rglob("*")):
        if img_path.suffix.lower() in IMAGE_EXTENSIONS and "mask" not in img_path.name.lower():
            return img_path
    return None


def get_all_product_ids() -> list[str]:
    """List all product IDs in the inVitro catalog."""
    pids = []
    for d in sorted(INVITRO_DIR.iterdir()):
        if d.is_dir() and d.name.isdigit():
            pids.append(d.name)
    return pids


def compose_shelf(
    product_ids: list[str],
    product_height: int = 200,
    gap: int = 12,
    margin: int = 30,
    shelf_color: tuple[int, int, int] = (210, 200, 185),
) -> tuple[Image.Image, list[dict]]:
    """Compose a synthetic shelf image.

    Returns:
        (shelf_image, ground_truth) where ground_truth is a list of
        {"product_id": str, "box": [x1, y1, x2, y2]} dicts.
    """
    # Load and resize product images
    items = []
    for pid in product_ids:
        img_path = find_product_image(pid)
        if img_path is None:
            print(f"  Warning: no image for product {pid}, skipping")
            continue
        img = Image.open(img_path).convert("RGB")
        # Resize to fixed height, preserve aspect ratio
        w, h = img.size
        new_h = product_height
        new_w = max(1, int(w * (new_h / h)))
        img = img.resize((new_w, new_h), Image.BICUBIC)
        items.append((pid, img))

    if not items:
        raise ValueError("No valid product images found")

    # Compute canvas size
    total_w = margin + sum(img.size[0] + gap for _, img in items) - gap + margin
    total_h = margin + product_height + margin

    # Create canvas
    canvas = Image.new("RGB", (total_w, total_h), shelf_color)

    # Paste products and record ground truth
    ground_truth = []
    x_cursor = margin
    for pid, img in items:
        y = margin
        w, h = img.size
        canvas.paste(img, (x_cursor, y))
        ground_truth.append({
            "product_id": pid,
            "box": [x_cursor, y, x_cursor + w, y + h],
        })
        x_cursor += w + gap

    return canvas, ground_truth


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compose a synthetic shelf image from catalog products."
    )
    parser.add_argument("--products", type=str, default=None,
                        help="Comma-separated product IDs (e.g. 3,7,15,42,88).")
    parser.add_argument("--n-random", type=int, default=None,
                        help="Number of random products to place (alternative to --products).")
    parser.add_argument("--output", type=str, default="test_shelf.jpg",
                        help="Output image path (default: test_shelf.jpg).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    random.seed(args.seed)

    if args.products:
        product_ids = [p.strip() for p in args.products.split(",")]
    elif args.n_random:
        all_pids = get_all_product_ids()
        product_ids = random.sample(all_pids, min(args.n_random, len(all_pids)))
    else:
        product_ids = random.sample(get_all_product_ids(), 6)

    print(f"Composing shelf with products: {', '.join(product_ids)}")

    canvas, ground_truth = compose_shelf(product_ids)
    print(f"Canvas size: {canvas.size[0]}x{canvas.size[1]}")

    # Save image
    output_path = Path(args.output)
    canvas.save(output_path, quality=95)
    print(f"Saved: {output_path}")

    # Save ground truth
    gt_path = output_path.with_suffix(".json")
    gt_data = {
        "image": str(output_path),
        "products": product_ids,
        "ground_truth": ground_truth,
    }
    with open(gt_path, "w") as f:
        json.dump(gt_data, f, indent=2)
    print(f"Ground truth: {gt_path}")


if __name__ == "__main__":
    main()
