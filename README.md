# Grocery Scanner

AI-powered grocery shelf scanner that detects and identifies products from shelf images. Point a camera at a supermarket shelf, upload the photo, and get a structured list of recognized SKUs.

## How it works

```
Shelf image
    │
    ▼
┌─────────────────────┐
│  YOLOWorld detector │  Open-vocabulary detection (9 grocery classes)
│  (YOLOv8s-worldv2)  │
└─────────────────────┘
    │  bounding box crops
    ▼
┌─────────────────────┐    ┌──────────────────────┐
│  Barcode detector   │───▶│  Short-circuit result │  (if barcode found)
│  (pyzbar)           │    │  CONFIDENT, score=1.0 │
└─────────────────────┘    └──────────────────────┘
    │  (no barcode)
    ▼
┌─────────────────────┐
│  CLIP embedder      │  ViT-L/14, multi-crop query augmentation
│  (ViT-L/14 openai)  │  + optional CLIPAdapter residual MLP
└─────────────────────┘
    │  768-d query vector
    ▼
┌─────────────────────┐
│  FAISS search       │  Cosine similarity over SKU prototype index
│  (IndexFlatIP)      │  (multi-prototype aggregation per SKU)
└─────────────────────┘
    │  top-K candidates
    ▼
┌─────────────────────┐
│  BM25 OCR reranking │  EasyOCR + BM25 text fusion (optional)
│  (EasyOCR + BM25)   │  α=0.85 visual + 0.15 text
└─────────────────────┘
    │  confidence-gated result
    ▼
 CONFIDENT / AMBIGUOUS / UNKNOWN
```

**Recognition pipeline per crop:**
1. Check for barcodes — if found, return immediately as CONFIDENT (score 1.0)
2. Embed the crop with CLIP (3-variant multi-crop average for robustness); apply CLIPAdapter if loaded
3. Search the FAISS SKU prototype index for top-K visual matches (multi-prototype aggregation: max score per unique SKU)
4. If OCR is enabled, run EasyOCR on the crop and fuse BM25 text similarity with visual scores (α=0.85 visual + 0.15 text)
5. Gate on confidence: top-1 score threshold (0.25) and margin between top-1 and top-2 (0.02)

## Dataset: GroZi-120

The catalog is built from [GroZi-120](https://grozi.calit2.net/), a standard grocery product recognition benchmark with 120 product classes photographed both in-vitro (isolated, catalog-style) and in-situ (on real store shelves).

| Split    | Description                        | Images | Used for             |
|----------|------------------------------------|--------|----------------------|
| inVitro  | Studio photos of each product      | 2,028  | Building the catalog |
| inSitu   | Real shelf images with annotations | 11,194 | Evaluation           |

## Retrieval accuracy

Evaluated on 1,200 queries (10 random inSitu images per SKU × 120 SKUs), multi-crop query augmentation, top-20 search depth.

| Configuration                            | Top-1  | Top-5  |  MRR  | Notes |
|------------------------------------------|--------|--------|-------|-------|
| Baseline (CLIP ViT-L/14, mean prototype) | 59.5%  | 73.2%  | 0.660 | Out-of-the-box CLIP |
| k=3 multi-prototype (k-means)            | 49.3%  | 60.2%  | 0.546 | Worse — studio photos cluster tightly |
| Two-stage (proto → image-level rerank)   | 48.6%  | 65.6%  | 0.571 | Also hurts; mean prototype is optimal |
| **CLIP-Adapter (SupCon, 30 epochs)**     | **69.8%** | **83.2%** | **0.761** | **+10.3 pp top-1 gain** |

**Key finding:** For tightly-clustered studio catalog photos, the mean prototype vector is a better representative than k-means cluster centres or individual images. The adapter finetuning is the most effective improvement.

## Setup

### Prerequisites

- Python 3.10+
- GPU strongly recommended (CLIP ViT-L/14 is large; CPU inference is slow)
- ~4 GB disk space for the dataset and catalog

### Install dependencies

```bash
pip install -r requirements.txt
```

For barcode scanning, also install the system `libzbar` library:

```bash
# Debian / Ubuntu
sudo apt-get install -y libzbar0
```

### Build the catalog (one-time)

Run these scripts in order:

```bash
# 1. Download GroZi-120 (~400 MB)
python scripts/download_dataset.py

# 2. Embed all inVitro images with CLIP (~10-20 min on GPU)
python scripts/build_embeddings.py

# 3. Build FAISS indexes (image-level + SKU prototypes)
python scripts/build_index.py

# 4. (Optional) Pre-extract OCR text for all catalog images (~30-60 min on GPU)
#    Enables BM25 text-based reranking at query time
python scripts/build_ocr.py
```

After step 3 you have a working system. Step 4 is optional but improves accuracy for products with distinctive text (brand names, labels).

### (Optional) Finetune a CLIP-Adapter

Train a lightweight residual adapter on top of the frozen CLIP backbone using Supervised Contrastive Loss. This takes ~8 minutes on a single GPU and improves Top-1 accuracy by ~10 percentage points.

```bash
# Train adapter (~30 epochs, saves best checkpoint to catalog/adapter.pt)
python scripts/finetune_clip.py

# Rebuild embeddings and index using the trained adapter
python scripts/build_embeddings.py --adapter catalog/adapter.pt
python scripts/build_index.py

# Evaluate
python scripts/eval_retrieval.py --adapter catalog/adapter.pt --multicrop
```

The adapter is a small bottleneck MLP (~400K parameters) applied as a residual on top of the CLIP embedding: `adapted = normalize(x + sigmoid(α) × MLP(x))`. The backbone remains fully frozen.

**Training options:**
```
python scripts/finetune_clip.py --epochs 50 --lr 5e-4 --batch-size 128
python scripts/finetune_clip.py --temp 0.07 --bottleneck 512
```

## Running the web app

```bash
# Standard launch (auto-loads adapter if catalog/adapter.pt exists)
python app.py

# Explicit adapter path
python app.py --adapter catalog/adapter.pt

# Skip OCR for faster startup
python app.py --no-ocr

# Different port
python app.py --port 8080
```

Then open [http://localhost:5000](http://localhost:5000).

**Web UI workflow:**
1. Drag & drop (or click to browse) a shelf image
2. The server detects products, embeds each crop, and searches the catalog
3. Each detection is shown as a card colored by confidence tier:
   - **Green (CONFIDENT)** — auto-added to the shopping list
   - **Yellow (AMBIGUOUS)** — pick the correct product from the top-3 candidates
   - **Red (UNKNOWN)** — enter a product ID manually or skip
4. Click **Finalize Shopping List** to generate the final list
5. **Download CSV** to export the list

Clicking a bounding box on the annotated image scrolls to the corresponding card, and hovering a card highlights the box.

### Registering new products

New products can be added to the catalog on the fly without rebuilding from scratch.

**Via the API:**
```bash
curl -X POST http://localhost:5000/api/register \
  -F "product_id=SKU-NEW-001" \
  -F "images=@front.jpg" \
  -F "images=@back.jpg"
```

**Via CLI:**
```bash
python scripts/register_product.py SKU-NEW-001 front.jpg back.jpg side.jpg

# With adapter
python scripts/register_product.py --adapter catalog/adapter.pt SKU-NEW-001 img1.jpg
```

Multiple reference images are averaged into a single prototype and upserted into the live index. Re-running with the same `product_id` replaces the existing entry.

## CLI usage

Scan a single image from the command line:

```bash
python scripts/scan.py shelf_photo.jpg

# Save annotated image and JSON results
python scripts/scan.py shelf_photo.jpg \
    --save-annotated annotated.jpg \
    --json results.json

# Skip interactive prompts (batch mode)
python scripts/scan.py shelf_photo.jpg --non-interactive

# Faster run without OCR reranking
python scripts/scan.py shelf_photo.jpg --no-ocr

# Use finetuned adapter
python scripts/scan.py shelf_photo.jpg --adapter catalog/adapter.pt

# Tune detector sensitivity
python scripts/scan.py shelf_photo.jpg --det-threshold 0.25
```

**CLI output example:**
```
======================================================================
  Shelf Scan: shelf_photo.jpg
  Detected: 8 items | Confident: 5 | Ambiguous: 2 | Unknown: 1
======================================================================

   #  Box                      Tier        Product  Score  Method
  --  ------------------------ ----------- ------- ------  --------
   1  ( 120, 200, 340, 480)    CONFIDENT   SKU  42  87.3%  [auto]
   2  ( 350, 195, 580, 475)    CONFIDENT   SKU  17  84.1%  [auto]
   3  ( 590, 210, 800, 470)    AMBIGUOUS   SKU   5  71.2%  [ambig]
   ...
```

## Evaluate retrieval accuracy

```bash
# Basic evaluation (1 query/SKU)
python scripts/eval_retrieval.py

# Full evaluation with multi-crop and adapter
python scripts/eval_retrieval.py --adapter catalog/adapter.pt --multicrop --per-sku 10 --top-k 20
```

## Project structure

```
grocery-scanner/
├── app.py                  Flask web app (API + frontend)
├── requirements.txt        Python dependencies
├── src/
│   ├── adapter.py          CLIPAdapter: residual bottleneck MLP
│   ├── barcode.py          Barcode / QR code detection (pyzbar)
│   ├── config.py           All paths and hyperparameters
│   ├── detector.py         YOLOWorld open-vocabulary detector
│   ├── embedder.py         CLIP ViT-L/14 embedder (multi-crop + adapter)
│   ├── index.py            FAISS index build / search / confidence
│   └── ocr.py              EasyOCR extraction and BM25 text reranking
├── scripts/
│   ├── download_dataset.py Download GroZi-120
│   ├── build_embeddings.py Embed inVitro catalog images (+ adapter)
│   ├── build_index.py      Build FAISS image-level + SKU indexes
│   ├── build_ocr.py        Pre-extract OCR text for all catalog images
│   ├── finetune_clip.py    Train CLIPAdapter with SupCon loss
│   ├── register_product.py Register a new product from reference images
│   ├── scan.py             CLI scanner (also imported by app.py)
│   ├── query.py            Interactive single-image query tool
│   ├── eval_retrieval.py   Accuracy evaluation on inSitu images
│   └── make_test_shelf.py  Compose a synthetic test shelf image
├── templates/
│   └── index.html          Single-page web UI
└── catalog/                (generated, gitignored)
    ├── sku.index           FAISS SKU prototype index
    ├── sku_labels.json     Product ID metadata
    ├── adapter.pt          (optional) trained CLIPAdapter checkpoint
    └── ocr_texts.json      (optional) pre-extracted OCR texts
```

## Configuration

All tunable parameters live in [src/config.py](src/config.py):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `CLIP_MODEL` | `ViT-L-14` | CLIP backbone |
| `TOP_K` | `5` | Candidates returned per detection |
| `DEFAULT_SCORE_THRESHOLD` | `0.25` | Min cosine similarity for CONFIDENT |
| `DEFAULT_MIN_MARGIN` | `0.02` | Min score gap between top-1 and top-2 |
| `OCR_ALPHA` | `0.85` | Visual score weight in BM25 OCR fusion |
| `OCR_RERANK_DEPTH` | `20` | Candidates fetched before OCR reranking |
| `DETECTOR_CONF_THRESHOLD` | `0.3` | YOLOWorld confidence cutoff |
| `DETECTOR_MIN_BOX_AREA` | `400` | Minimum detection box area (px²) |
| `SKU_NUM_PROTOTYPES` | `1` | k-means prototypes per SKU in catalog |
| `ADAPTER_BOTTLENECK` | `256` | CLIPAdapter hidden dimension |

## License

The code in this repository is released under the MIT License.

The GroZi-120 dataset is subject to its own terms; see [https://grozi.calit2.net/](https://grozi.calit2.net/).
