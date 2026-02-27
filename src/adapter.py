"""Lightweight CLIP-Adapter: residual bottleneck MLP on top of frozen CLIP features.

Architecture (CLIP-Adapter, Gao et al. 2021):
    adapted = x + sigmoid(alpha) * MLP(x)
    output  = L2_normalise(adapted)

The adapter is trained with Supervised Contrastive loss on the GroZi-120
inVitro catalog images (see scripts/finetune_clip.py).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CLIPAdapter(nn.Module):
    """Residual bottleneck adapter for CLIP visual embeddings.

    Args:
        dim: Input/output embedding dimension (768 for ViT-L/14).
        bottleneck: Hidden dimension of the bottleneck MLP.
    """

    def __init__(self, dim: int = 768, bottleneck: int = 256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(dim, bottleneck),
            nn.GELU(),
            nn.Linear(bottleneck, dim),
        )
        # Learned residual blending weight; initialised near zero so the
        # adapter starts as an identity and grows from there.
        self.log_alpha = nn.Parameter(torch.tensor(-2.0))

    @property
    def alpha(self) -> torch.Tensor:
        return torch.sigmoid(self.log_alpha)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply adapter and L2-normalise.

        Args:
            x: (B, dim) L2-normalised CLIP embeddings.

        Returns:
            (B, dim) adapted and re-normalised embeddings.
        """
        adapted = x + self.alpha * self.mlp(x)
        return F.normalize(adapted, dim=-1)
