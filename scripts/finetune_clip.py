"""Finetune a CLIP-Adapter using Supervised Contrastive Loss on GroZi-120.

The CLIP backbone is kept frozen.  Only the lightweight adapter MLP is trained
(~400K parameters for the default bottleneck=256 configuration).

Training strategy
-----------------
* Positives: all images of the same SKU form a positive group.
* SupCon loss (Khosla et al. 2020) is applied within each mini-batch.
* Heavy augmentation (crops, colour jitter, flips) increases effective
  positives without collecting new data.
* The adapter is initialised near-identity (log_alpha = -2) so training
  starts from the CLIP baseline and grows from there.

After training
--------------
  python scripts/build_embeddings.py --adapter catalog/adapter.pt
  python scripts/build_index.py
  python scripts/eval_retrieval.py --adapter catalog/adapter.pt --multicrop

Usage
-----
  python scripts/finetune_clip.py
  python scripts/finetune_clip.py --epochs 50 --lr 5e-4 --batch-size 128
  python scripts/finetune_clip.py --temp 0.07 --bottleneck 512
"""

import argparse
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (
    INVITRO_DIR, IMAGE_EXTENSIONS, EMBEDDINGS_PATH, LABELS_PATH,
    CATALOG_DIR, ADAPTER_PATH, ADAPTER_BOTTLENECK, EMBEDDING_DIM,
)
from src.adapter import CLIPAdapter


# ── Augmentation ─────────────────────────────────────────────────────────────

def _make_aug_transform() -> transforms.Compose:
    """Strong augmentation pipeline for product images."""
    return transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.5, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.3, hue=0.1),
        transforms.RandomGrayscale(p=0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711),
        ),
    ])


# ── Supervised Contrastive Loss ───────────────────────────────────────────────

def supcon_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float = 0.07) -> torch.Tensor:
    """Supervised Contrastive Loss (Khosla et al. 2020, https://arxiv.org/abs/2004.11362).

    Args:
        features: (N, D) L2-normalised embeddings.
        labels:   (N,)   integer class labels.
        temperature: softmax temperature (lower = sharper).

    Returns:
        Scalar loss.
    """
    device = features.device
    N = features.shape[0]

    sim = torch.mm(features, features.T) / temperature  # (N, N)

    # Mask out self-comparisons
    self_mask = torch.eye(N, dtype=torch.bool, device=device)
    sim = sim.masked_fill(self_mask, float("-inf"))

    # Positive pair mask: same class, excluding self
    labels = labels.view(-1, 1)
    pos_mask = (labels == labels.T) & ~self_mask  # (N, N)

    if pos_mask.sum() == 0:
        return features.sum() * 0.0  # no positives in batch — skip gracefully

    log_prob = F.log_softmax(sim, dim=1)

    # Use torch.where to avoid -inf * 0 = NaN at non-positive positions.
    # Positions outside pos_mask contribute 0 to the sum.
    per_sample_log = torch.where(pos_mask, log_prob, torch.zeros_like(log_prob))
    n_positives = pos_mask.float().sum(dim=1).clamp(min=1)
    loss = -per_sample_log.sum(dim=1) / n_positives
    return loss.mean()


# ── Dataset ───────────────────────────────────────────────────────────────────

class InVitroDataset(torch.utils.data.Dataset):
    """InVitro product images with augmentation.

    Returns two independently-augmented views of the same image (SimCLR-style)
    so each image contributes two positive samples per forward pass.
    """

    def __init__(self, entries: list[dict], aug: transforms.Compose, n_views: int = 2):
        self.entries = entries
        self.aug = aug
        self.n_views = n_views

        # Map product_id strings to integer labels for SupCon
        pids = sorted(set(e["product_id"] for e in entries))
        self.pid_to_label = {pid: i for i, pid in enumerate(pids)}

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, idx: int):
        entry = self.entries[idx]
        label = self.pid_to_label[entry["product_id"]]
        try:
            img = Image.open(entry["image_path"]).convert("RGB")
        except Exception:
            # Return a black image on load failure — rare
            img = Image.new("RGB", (224, 224))
        views = torch.stack([self.aug(img) for _ in range(self.n_views)])
        return views, label  # (n_views, C, H, W), scalar


def collate_views(batch):
    """Flatten multi-view batch: (B, n_views, C, H, W) → (B*n_views, C, H, W)."""
    views_list, labels_list = zip(*batch)
    views = torch.stack(views_list)   # (B, n_views, C, H, W)
    labels = torch.tensor(labels_list)
    B, V, C, H, W = views.shape
    views = views.view(B * V, C, H, W)
    labels = labels.unsqueeze(1).expand(B, V).reshape(B * V)
    return views, labels


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Finetune CLIP-Adapter with SupCon loss.")
    parser.add_argument("--epochs", type=int, default=30, help="Training epochs (default: 30).")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate (default: 1e-3).")
    parser.add_argument("--batch-size", type=int, default=64, help="Images per batch (default: 64).")
    parser.add_argument("--temp", type=float, default=0.07, help="SupCon temperature (default: 0.07).")
    parser.add_argument("--bottleneck", type=int, default=ADAPTER_BOTTLENECK,
                        help=f"Adapter bottleneck dim (default: {ADAPTER_BOTTLENECK}).")
    parser.add_argument("--n-views", type=int, default=2, help="Augmented views per image (default: 2).")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}\n")

    # ── Collect catalog entries ───────────────────────────
    if LABELS_PATH.exists():
        import json
        with open(LABELS_PATH) as f:
            entries = json.load(f)
        print(f"Loaded {len(entries)} catalog entries from labels.json")
    else:
        print(f"ERROR: {LABELS_PATH} not found. Run build_embeddings.py first.", file=sys.stderr)
        sys.exit(1)

    # ── Dataset & loader ─────────────────────────────────
    aug = _make_aug_transform()
    dataset = InVitroDataset(entries, aug, n_views=args.n_views)
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=(device == "cuda"),
        collate_fn=collate_views,
        drop_last=True,
    )
    n_classes = len(dataset.pid_to_label)
    print(f"Dataset: {len(dataset)} images, {n_classes} products, {args.n_views} views each")
    print(f"Loader:  {len(loader)} batches/epoch (batch_size={args.batch_size})\n")

    # ── Load frozen CLIP backbone ────────────────────────
    print("Loading CLIP backbone (frozen) ...", end=" ", flush=True)
    import open_clip
    clip_model, _, _ = open_clip.create_model_and_transforms(
        "ViT-L-14", pretrained="openai"
    )
    clip_model = clip_model.to(device).eval()
    for p in clip_model.parameters():
        p.requires_grad_(False)
    print("done")

    # ── Adapter ──────────────────────────────────────────
    adapter = CLIPAdapter(dim=EMBEDDING_DIM, bottleneck=args.bottleneck).to(device)
    n_params = sum(p.numel() for p in adapter.parameters() if p.requires_grad)
    print(f"Adapter: {n_params:,} trainable parameters  (alpha={adapter.alpha.item():.4f})\n")

    optimiser = torch.optim.AdamW(adapter.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimiser, T_max=args.epochs)

    # ── Training loop ────────────────────────────────────
    print(f"{'Epoch':>6s}  {'Loss':>8s}  {'LR':>9s}  {'Alpha':>7s}  {'Time':>6s}")
    print("─" * 50)

    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        adapter.train()
        epoch_loss = 0.0
        t0 = time.time()

        for views, labels in loader:
            views = views.to(device)
            labels = labels.to(device)

            # Extract frozen CLIP features
            with torch.no_grad():
                feats = clip_model.encode_image(views)
                feats = F.normalize(feats, dim=-1)

            # Apply adapter
            adapted = adapter(feats)

            loss = supcon_loss(adapted, labels, temperature=args.temp)

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()

            epoch_loss += loss.item()

        scheduler.step()

        avg_loss = epoch_loss / len(loader)
        lr = scheduler.get_last_lr()[0]
        alpha = adapter.alpha.item()
        elapsed = time.time() - t0
        print(f"{epoch:>6d}  {avg_loss:>8.4f}  {lr:>9.2e}  {alpha:>7.4f}  {elapsed:>5.1f}s")

        if avg_loss < best_loss:
            best_loss = avg_loss
            CATALOG_DIR.mkdir(parents=True, exist_ok=True)
            torch.save(adapter.state_dict(), ADAPTER_PATH)

    print(f"\nBest loss: {best_loss:.4f}")
    print(f"Adapter saved to: {ADAPTER_PATH}")
    print("\nNext steps:")
    print("  python scripts/build_embeddings.py --adapter catalog/adapter.pt")
    print("  python scripts/build_index.py")
    print("  python scripts/eval_retrieval.py --adapter catalog/adapter.pt --multicrop")


if __name__ == "__main__":
    main()
