"""Shared configuration for the GroZi-120 retrieval system."""

from pathlib import Path

# ── Paths ────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "grozi120"
INVITRO_DIR = DATA_DIR / "inVitro"
INSITU_DIR = DATA_DIR / "inSitu"

CATALOG_DIR = PROJECT_ROOT / "catalog"
EMBEDDINGS_PATH = CATALOG_DIR / "embeddings.npy"
LABELS_PATH = CATALOG_DIR / "labels.json"
INDEX_PATH = CATALOG_DIR / "catalog.index"

SKU_EMBEDDINGS_PATH = CATALOG_DIR / "sku_embeddings.npy"
SKU_LABELS_PATH = CATALOG_DIR / "sku_labels.json"
SKU_INDEX_PATH = CATALOG_DIR / "sku.index"

# ── Download URLs ────────────────────────────────────────
INVITRO_URL = "https://grozi.calit2.net/GroZi-120/inVitro.zip"
INSITU_URL = "https://grozi.calit2.net/GroZi-120/inSitu.zip"

# ── Model (baseline) ─────────────────────────────────────
CLIP_MODEL = "ViT-L-14"
CLIP_PRETRAINED = "openai"
EMBEDDING_DIM = 768
TOP_K = 5

# ── Alternative backbones ────────────────────────────────
# To switch backbone, change the three vars above and rebuild embeddings + indexes.
# SigLIP-L:  ("ViT-SO400M-14-SigLIP-384", "webli",              1152)
# EVA-02-L:  ("EVA02-L-14-336",            "merged2b_s6b_b61k",  768)

# ── Multi-prototype indexing ─────────────────────────────
# k-means cluster centres stored per SKU. 1 = original mean-prototype.
SKU_NUM_PROTOTYPES = 1

# ── Adapter finetuning ───────────────────────────────────
ADAPTER_PATH = CATALOG_DIR / "adapter.pt"
ADAPTER_BOTTLENECK = 256  # bottleneck hidden dim

# ── Query Expansion ────────────────────────────────────
EXPAND_TOP_K = 3
EXPAND_QUERY_WEIGHT = 2.0

# ── Calibration / Thresholding ─────────────────────────
DEFAULT_SCORE_THRESHOLD = 0.25
DEFAULT_MIN_MARGIN = 0.02

# ── OCR Fusion ─────────────────────────────────────────
OCR_TEXTS_PATH = CATALOG_DIR / "ocr_texts.json"
OCR_ALPHA = 0.85
OCR_RERANK_DEPTH = 20

# ── Detection ──────────────────────────────────────────
DETECTOR_CLASSES = [
    "grocery product", "bottle", "can", "box", "package",
    "bag", "jar", "carton", "container",
]
DETECTOR_CONF_THRESHOLD = 0.3
DETECTOR_MIN_BOX_AREA = 400
DETECTOR_BOX_PADDING = 2

# ── Image Extensions ────────────────────────────────────
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".gif"}
