"""FAISS index management for catalog search."""

import json

import faiss
import numpy as np

from src.config import (
    EMBEDDING_DIM, INDEX_PATH, LABELS_PATH, SKU_INDEX_PATH, SKU_LABELS_PATH,
    EXPAND_TOP_K, EXPAND_QUERY_WEIGHT, DEFAULT_SCORE_THRESHOLD, DEFAULT_MIN_MARGIN,
)


def build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Build a FAISS inner-product index from L2-normalized embeddings."""
    assert embeddings.ndim == 2 and embeddings.shape[1] == EMBEDDING_DIM
    embeddings = np.ascontiguousarray(embeddings, dtype=np.float32)
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings)
    return index


def save_index(index: faiss.IndexFlatIP) -> None:
    """Write FAISS index to disk."""
    INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(INDEX_PATH))


def load_index() -> faiss.IndexFlatIP:
    """Load FAISS index from disk."""
    if not INDEX_PATH.exists():
        raise FileNotFoundError(f"No FAISS index at {INDEX_PATH}. Run build_index.py first.")
    return faiss.read_index(str(INDEX_PATH))


def save_labels(labels: list[dict]) -> None:
    """Save label metadata to JSON."""
    LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LABELS_PATH, "w") as f:
        json.dump(labels, f, indent=2)


def load_labels() -> list[dict]:
    """Load label metadata from JSON."""
    if not LABELS_PATH.exists():
        raise FileNotFoundError(f"No labels file at {LABELS_PATH}. Run build_embeddings.py first.")
    with open(LABELS_PATH) as f:
        return json.load(f)


def search(index: faiss.IndexFlatIP, query_embedding: np.ndarray, top_k: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Search the index. Returns (scores, indices), each shape (top_k,)."""
    query = np.ascontiguousarray(query_embedding.astype(np.float32).reshape(1, -1))
    faiss.normalize_L2(query)
    scores, indices = index.search(query, top_k)
    return scores[0], indices[0]


def expand_query(
    query_emb: np.ndarray,
    index: faiss.IndexFlatIP,
    index_embeddings: np.ndarray,
    top_k_expand: int = EXPAND_TOP_K,
    query_weight: float = EXPAND_QUERY_WEIGHT,
) -> np.ndarray:
    """Expand a query by averaging it with its top-K nearest prototype embeddings."""
    scores, indices = search(index, query_emb, top_k=top_k_expand)
    valid = indices[indices >= 0]
    if len(valid) == 0:
        return query_emb

    ref_embs = index_embeddings[valid].astype(np.float32)
    q = query_emb.astype(np.float32).reshape(1, -1)

    expanded = query_weight * q + ref_embs.sum(axis=0, keepdims=True)
    expanded = expanded / (query_weight + len(valid))

    norm = np.linalg.norm(expanded)
    if norm > 0:
        expanded = expanded / norm

    return expanded.flatten()


def compute_confidence(
    scores: np.ndarray,
    score_threshold: float = DEFAULT_SCORE_THRESHOLD,
    min_margin: float = DEFAULT_MIN_MARGIN,
) -> dict:
    """Compute confidence metadata from search scores."""
    top1_score = float(scores[0]) if len(scores) > 0 else 0.0
    top2_score = float(scores[1]) if len(scores) > 1 else 0.0
    margin = top1_score - top2_score

    reasons = []
    if top1_score < score_threshold:
        reasons.append(f"score {top1_score:.3f} < {score_threshold}")
    if margin < min_margin:
        reasons.append(f"margin {margin:.3f} < {min_margin}")

    return {
        "top1_score": top1_score,
        "margin": margin,
        "is_confident": len(reasons) == 0,
        "rejection_reason": "; ".join(reasons) if reasons else None,
    }


# ── SKU-level index ──────────────────────────────────────

def save_sku_index(index: faiss.IndexFlatIP) -> None:
    SKU_INDEX_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(SKU_INDEX_PATH))


def load_sku_index() -> faiss.IndexFlatIP:
    if not SKU_INDEX_PATH.exists():
        raise FileNotFoundError(f"No SKU index at {SKU_INDEX_PATH}. Run build_index.py first.")
    return faiss.read_index(str(SKU_INDEX_PATH))


def save_sku_labels(labels: list[dict]) -> None:
    SKU_LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SKU_LABELS_PATH, "w") as f:
        json.dump(labels, f, indent=2)


def load_sku_labels() -> list[dict]:
    if not SKU_LABELS_PATH.exists():
        raise FileNotFoundError(f"No SKU labels at {SKU_LABELS_PATH}. Run build_index.py first.")
    with open(SKU_LABELS_PATH) as f:
        return json.load(f)
