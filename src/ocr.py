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


def _tokenize_list(text: str) -> list[str]:
    """Like _tokenize but returns a list (order-preserved, with duplicates)."""
    return [t for t in text.split() if len(t) >= 2]


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


def text_rerank(
    query_text: str,
    candidate_pids: list[str],
    candidate_visual_scores: list[float],
    sku_ocr_texts: dict[str, str],
    alpha: float = 0.85,
) -> list[tuple[str, float, float, float]]:
    """Rerank candidates by fusing visual scores with BM25 text similarity.

    Builds a BM25 index over the candidate OCR texts, scores against the
    query, normalises to [0, 1], then blends:
        final = alpha * visual + (1 - alpha) * bm25_text

    Returns:
        List of (pid, final_score, visual_score, text_score) sorted desc.
    """
    from rank_bm25 import BM25Okapi

    query_tokens = _tokenize_list(_clean_text(query_text))
    corpus_tokens = [
        _tokenize_list(_clean_text(sku_ocr_texts.get(pid, "")))
        for pid in candidate_pids
    ]

    text_scores = [0.0] * len(candidate_pids)
    if query_tokens and any(corpus_tokens):
        bm25 = BM25Okapi(corpus_tokens if corpus_tokens else [[]])
        raw = bm25.get_scores(query_tokens)
        max_score = float(max(raw)) if max(raw) > 0 else 1.0
        text_scores = [float(s) / max_score for s in raw]

    fused = []
    for pid, vscore, tscore in zip(candidate_pids, candidate_visual_scores, text_scores):
        final = alpha * vscore + (1.0 - alpha) * tscore
        fused.append((pid, final, vscore, tscore))

    fused.sort(key=lambda x: x[1], reverse=True)
    return fused
