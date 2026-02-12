"""CLIP image embedding extractor with multi-crop query support."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps, ImageStat
from torchvision import transforms

from src.config import CLIP_MODEL, CLIP_PRETRAINED

# CLIP normalization constants
_CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
_CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
_CLIP_MEAN_PIX = tuple(int(m * 255) for m in _CLIP_MEAN)  # (122, 116, 104)


def _edge_avg(img: Image.Image) -> tuple[int, int, int]:
    """Average color of the image border (1px edge), used as padding fill."""
    stat = ImageStat.Stat(img)
    return tuple(int(m) for m in stat.mean[:3])


class Embedder:
    """Extracts feature vectors using CLIP ViT with optional multi-crop."""

    def __init__(self):
        self._model = None
        self._preprocess = None
        self._letterbox_transform = None
        self._device = "cuda" if torch.cuda.is_available() else "cpu"

    def load(self) -> None:
        """Load CLIP model and preprocessing transforms."""
        import open_clip

        model, _, preprocess = open_clip.create_model_and_transforms(
            CLIP_MODEL, pretrained=CLIP_PRETRAINED
        )
        model = model.to(self._device).eval()
        self._model = model
        self._preprocess = preprocess
        self._letterbox_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=_CLIP_MEAN, std=_CLIP_STD),
        ])

    @property
    def model(self):
        if self._model is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._model

    @property
    def preprocess(self):
        if self._preprocess is None:
            raise RuntimeError("Model not loaded. Call load() first.")
        return self._preprocess

    # ── Standard embedding (catalog) ─────────────────────

    def embed_pil(self, img: Image.Image) -> np.ndarray:
        """Embed a single PIL Image using standard CLIP preprocessing."""
        img = img.convert("RGB")
        tensor = self.preprocess(img).unsqueeze(0).to(self._device)
        with torch.no_grad():
            emb = self.model.encode_image(tensor)
            emb = emb / emb.norm(dim=-1, keepdim=True)
        return emb.cpu().numpy().flatten()

    def embed_file(self, path: str) -> np.ndarray:
        """Embed an image file using standard CLIP preprocessing."""
        img = Image.open(path)
        return self.embed_pil(img)

    def embed_batch(self, images: list[Image.Image], batch_size: int = 64) -> np.ndarray:
        """Embed a list of PIL Images using standard CLIP preprocessing."""
        all_embeddings = []
        for i in range(0, len(images), batch_size):
            batch_imgs = images[i : i + batch_size]
            tensors = torch.stack(
                [self.preprocess(img.convert("RGB")) for img in batch_imgs]
            ).to(self._device)
            with torch.no_grad():
                embs = self.model.encode_image(tensors)
                embs = embs / embs.norm(dim=-1, keepdim=True)
            all_embeddings.append(embs.cpu().numpy())
        return np.concatenate(all_embeddings, axis=0)

    # ── Letterbox + multi-crop (query) ───────────────────

    def _letterbox_preprocess(self, img: Image.Image, input_size: int = 224) -> torch.Tensor:
        """Resize to fit within input_size×input_size, pad to square, normalize.

        Preserves full aspect ratio — no center-crop.
        """
        img = img.convert("RGB")
        w, h = img.size
        scale = input_size / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        img = img.resize((new_w, new_h), Image.BICUBIC)

        # Pad to square with CLIP mean pixel color
        padded = Image.new("RGB", (input_size, input_size), _CLIP_MEAN_PIX)
        paste_x = (input_size - new_w) // 2
        paste_y = (input_size - new_h) // 2
        padded.paste(img, (paste_x, paste_y))

        return self._letterbox_transform(padded)

    def embed_multicrop(self, img_or_path) -> np.ndarray:
        """Multi-crop query: letterbox the original + padded variants, average embeddings.

        Generates 3 variants:
          1. Tight crop (original image, letterboxed)
          2. +10% border padding (simulates wider crop)
          3. +25% border padding (catches more edge context)
        Padding uses the image's average color as fill.
        """
        if isinstance(img_or_path, (str, Path)):
            img = Image.open(img_or_path).convert("RGB")
        else:
            img = img_or_path.convert("RGB")

        fill = _edge_avg(img)
        w, h = img.size
        variants = [img]

        for pct in (0.10, 0.25):
            pw = max(1, int(w * pct))
            ph = max(1, int(h * pct))
            variants.append(ImageOps.expand(img, border=(pw, ph), fill=fill))

        tensors = torch.stack(
            [self._letterbox_preprocess(v) for v in variants]
        ).to(self._device)

        with torch.no_grad():
            embs = self.model.encode_image(tensors)
            embs = embs / embs.norm(dim=-1, keepdim=True)

        avg = embs.mean(dim=0, keepdim=True)
        avg = avg / avg.norm(dim=-1, keepdim=True)
        return avg.cpu().numpy().flatten()
