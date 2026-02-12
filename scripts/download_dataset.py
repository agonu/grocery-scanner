"""Download and extract the GroZi-120 dataset."""

import sys
import urllib.request
import zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import DATA_DIR, INVITRO_DIR, INSITU_DIR, INVITRO_URL, INSITU_URL, IMAGE_EXTENSIONS


def _progress(block_num: int, block_size: int, total_size: int) -> None:
    downloaded = block_num * block_size
    if total_size > 0:
        pct = min(100.0, downloaded / total_size * 100)
        mb = downloaded / 1024 / 1024
        total_mb = total_size / 1024 / 1024
        print(f"\r  {mb:.1f}/{total_mb:.1f} MB ({pct:.0f}%)", end="", flush=True)


def download_and_extract(url: str, dest_dir: Path, name: str) -> None:
    dest_dir.mkdir(parents=True, exist_ok=True)
    zip_path = dest_dir / f"{name}.zip"

    if zip_path.exists():
        print(f"  {zip_path.name} already downloaded, skipping.")
    else:
        print(f"  Downloading {name} from {url} ...")
        urllib.request.urlretrieve(url, zip_path, reporthook=_progress)
        print()

    target = dest_dir / name
    if target.exists() and any(target.iterdir()):
        print(f"  {name}/ already extracted, skipping.")
    else:
        print(f"  Extracting {zip_path.name} ...")
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(dest_dir)
        print(f"  Extracted to {target}/")


def main() -> None:
    print("=== GroZi-120 Dataset Download ===\n")

    print("[1/2] In-vitro images (catalog):")
    download_and_extract(INVITRO_URL, DATA_DIR, "inVitro")
    print()

    print("[2/2] In-situ images (test queries):")
    download_and_extract(INSITU_URL, DATA_DIR, "inSitu")
    print()

    invitro_count = sum(
        1 for f in INVITRO_DIR.rglob("*")
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )
    insitu_count = sum(
        1 for f in INSITU_DIR.rglob("*")
        if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
    )
    product_dirs = sorted(
        [d.name for d in INVITRO_DIR.iterdir() if d.is_dir()],
        key=lambda x: int(x) if x.isdigit() else x,
    )

    print("Verification:")
    print(f"  In-vitro images: {invitro_count}")
    print(f"  In-situ images:  {insitu_count}")
    print(f"  Product classes: {len(product_dirs)}")
    print(f"  Product IDs:     {product_dirs[:5]} ... {product_dirs[-3:]}")
    print("\nDone.")


if __name__ == "__main__":
    main()
