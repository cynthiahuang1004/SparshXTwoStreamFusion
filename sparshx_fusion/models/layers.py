from __future__ import annotations

import math

import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """Image to patch tokens.

    Input:  [B, C, H, W]
    Output: [B, N, D], where N = H / patch * W / patch
    """

    def __init__(self, image_size: int = 224, patch_size: int = 16, in_chans: int = 3, embed_dim: int = 192):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.image_size = image_size
        self.patch_size = patch_size
        self.grid_size = image_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class MLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MLPHead(nn.Module):
    """Configurable MLP decoder head for classification/regression.

    Structure: (num_layers - 1) x [Linear -> GELU -> Dropout] -> Linear(out_dim).
    The fused embedding is already LayerNorm'd upstream, so no extra input norm.
    With num_layers=1 this degenerates to a single linear probe.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        hidden_dim: int | None = None,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        layers: list[nn.Module] = []
        dim = in_dim
        for _ in range(max(0, num_layers - 1)):
            layers += [nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
            dim = hidden_dim
        layers.append(nn.Linear(dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    """Pre-norm self-attention transformer block."""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = MLP(dim, int(dim * mlp_ratio), dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y, _ = self.attn(self.norm1(x), self.norm1(x), self.norm1(x), need_weights=False)
        x = x + y
        x = x + self.mlp(self.norm2(x))
        return x


class SharedBottleneckFusionLayer(nn.Module):
    """Symmetric two-stream bottleneck fusion layer.

    For each modality, prepend the same bottleneck tokens, run a stream-specific
    transformer block, then average the updated bottleneck tokens across streams.
    """

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0, dropout: float = 0.0):
        super().__init__()
        self.rgb_block = TransformerBlock(dim, num_heads, mlp_ratio, dropout)
        self.tactile_block = TransformerBlock(dim, num_heads, mlp_ratio, dropout)

    def forward(
        self,
        rgb_tokens: torch.Tensor,
        tactile_tokens: torch.Tensor,
        bottleneck: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        num_bottleneck = bottleneck.shape[1]

        rgb_joint = torch.cat([bottleneck, rgb_tokens], dim=1)
        tactile_joint = torch.cat([bottleneck, tactile_tokens], dim=1)

        rgb_joint = self.rgb_block(rgb_joint)
        tactile_joint = self.tactile_block(tactile_joint)

        rgb_bottleneck = rgb_joint[:, :num_bottleneck]
        tactile_bottleneck = tactile_joint[:, :num_bottleneck]
        bottleneck = torch.stack([rgb_bottleneck, tactile_bottleneck], dim=0).mean(dim=0)

        rgb_tokens = rgb_joint[:, num_bottleneck:]
        tactile_tokens = tactile_joint[:, num_bottleneck:]
        return rgb_tokens, tactile_tokens, bottleneck


def init_vit_weights(module: nn.Module) -> None:
    if isinstance(module, nn.Linear):
        nn.init.trunc_normal_(module.weight, std=0.02)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.Conv2d):
        fan_out = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
        fan_out //= module.groups
        module.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.LayerNorm):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

