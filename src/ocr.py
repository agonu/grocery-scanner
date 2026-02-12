"""OCR text extraction and text-based similarity for reranking."""

import json
import re
from collections import defaultdict
from pathlib import Path


def _clean_text(raw: str) -> str:
    """Lowercase, strip non-alphanumeric (except spaces), collapse whitespace."""
    text = raw.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _tokenize(text: str) -> set[str]:
    """Split cleaned text into a set of tokens, filtering short ones."""
    return {t for t in text.split() if len(t) >= 2}


def extract_text(image_path: str, reader) -> str:
    """Run OCR on a single image file and return cleaned text."""
    try:
        results = reader.readtext(str(image_path), detail=0)
        raw = " ".join(results)
        return _clean_text(raw)
    except Exception:
        return ""


def extract_text_pil(image, reader) -> str:
    """Run OCR on a PIL Image (in-memory) and return cleaned text."""
    import numpy as np
    try:
        img_array = np.array(image.convert("RGB"))
        results = reader.readtext(img_array, detail=0)
        raw = " ".join(results)
        return _clean_text(raw)
    except Exception:
        return ""


def build_sku_ocr_texts(
    catalog_labels: list[dict], reader, progress_interval: int = 50
) -> dict[str, str]:
    """Build a mapping from product_id to concatenated OCR text from all catalog images."""
    sku_texts = defaultdict(list)

    for i, entry in enumerate(catalog_labels):
        text = extract_text(entry["image_path"], reader)
        if text:
            sku_texts[entry["product_id"]].append(text)
        if (i + 1) % progress_interval == 0:
            print(f"  OCR [{i + 1}/{len(catalog_labels)}]")

    return {pid: " ".join(texts) for pid, texts in sku_texts.items()}


def save_ocr_texts(ocr_texts: dict[str, str], path: Path) -> None:
    """Save OCR text mapping to JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(ocr_texts, f, indent=2)


def load_ocr_texts(path: Path) -> dict[str, str]:
    """Load OCR text mapping from JSON."""
    if not path.exists():
        raise FileNotFoundError(
            f"No OCR texts at {path}. Run build_ocr.py first."
        )
    with open(path) as f:
        return json.load(f)


def jaccard_similarity(tokens_a: set[str], tokens_b: set[str]) -> float:
    """Compute Jaccard similarity between two token sets."""
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union)


def text_rerank(
    query_text: str,
    candidate_pids: list[str],
    candidate_visual_scores: list[float],
    sku_ocr_texts: dict[str, str],
    alpha: float = 0.85,
) -> list[tuple[str, float, float, float]]:
    """Rerank candidates by fusing visual scores with OCR text similarity.

    Returns:
        List of (pid, final_score, visual_score, text_score) sorted by final_score desc.
    """
    query_tokens = _tokenize(query_text)

    fused = []
    for pid, vscore in zip(candidate_pids, candidate_visual_scores):
        sku_text = sku_ocr_texts.get(pid, "")
        sku_tokens = _tokenize(sku_text)
        tscore = jaccard_similarity(query_tokens, sku_tokens)
        final = alpha * vscore + (1.0 - alpha) * tscore
        fused.append((pid, final, vscore, tscore))

    fused.sort(key=lambda x: x[1], reverse=True)
    return fused
