"""Frozen image encoders behind a common interface.

Every encoder exposes:
    embed_dim        : int          (channel width, e.g. 1024 for ViT-L)
    num_patches      : int          (spatial tokens, e.g. 196 for p16@224)
    multiscale_layers: list[int]    (layer indices tapped for multi-scale features)
    forward(x)            -> (patch [B, N, E], cls [B, 1, E] | None)
    forward_multiscale(x) -> list of K  [B, N, E]  tensors

Supported encoders:
  dinov3_vitl16    DINOv3 ViT-L/16
  mae_vitl16       MAE ViT-L/16
  t3_large         T3 sensor encoder + shared trunk
  (no checkpoint)  MockEncoder
"""
from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────
#  Utilities
# ──────────────────────────────────────────────────────────────────────

def _resolve_multiscale(layers, depth, k=4):
    if layers is not None and all(i < depth for i in layers):
        return sorted(layers)
    step = depth / k
    idx = sorted({min(depth - 1, int(round((i + 1) * step)) - 1) for i in range(k)})
    while len(idx) < k:
        for cand in range(depth - 1, -1, -1):
            if cand not in idx:
                idx.append(cand)
                break
        idx = sorted(set(idx))
    return list(idx[-k:])


# ──────────────────────────────────────────────────────────────────────
#  Shared ViT block (matches standard MAE / T3 state-dict keys)
# ──────────────────────────────────────────────────────────────────────

class _Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.Module()
        self.attn.qkv = nn.Linear(dim, 3 * dim)
        self.attn.proj = nn.Linear(dim, dim)
        self._nh, self._hd = num_heads, dim // num_heads
        self.norm2 = nn.LayerNorm(dim)
        h = int(dim * mlp_ratio)
        self.mlp = nn.Module()
        self.mlp.fc1 = nn.Linear(dim, h)
        self.mlp.fc2 = nn.Linear(h, dim)

    def forward(self, x):
        B, N, C = x.shape
        y = self.attn.qkv(self.norm1(x)).reshape(B, N, 3, self._nh, self._hd)
        q, k, v = y.permute(2, 0, 3, 1, 4).unbind(0)
        y = F.scaled_dot_product_attention(q, k, v)
        y = self.attn.proj(y.transpose(1, 2).reshape(B, N, C))
        x = x + y
        y = self.mlp.fc2(F.gelu(self.mlp.fc1(self.norm2(x))))
        x = x + y
        return x


# ──────────────────────────────────────────────────────────────────────
#  DINOv3 ViT-L/16
# ──────────────────────────────────────────────────────────────────────

def _infer_dinov3_config(sd):
    from transformers import DINOv3ViTConfig

    pe = sd["embeddings.patch_embeddings.weight"]
    hidden = pe.shape[0]
    patch = pe.shape[-1]
    num_register = sd["embeddings.register_tokens"].shape[1]
    num_layers = 1 + max(int(k.split(".")[1]) for k in sd if k.startswith("layer."))
    intermediate = sd["layer.0.mlp.up_proj.weight"].shape[0]
    gated = any("gate" in k for k in sd if k.startswith("layer.0.mlp"))
    return DINOv3ViTConfig(
        patch_size=patch,
        hidden_size=hidden,
        intermediate_size=intermediate,
        num_hidden_layers=num_layers,
        num_attention_heads=hidden // 64,
        num_register_tokens=num_register,
        image_size=224,
        use_gated_mlp=gated,
    )


class DINOv3Encoder(nn.Module):
    def __init__(self, weights, image_size=224, multiscale_layers=None):
        super().__init__()
        if weights is None:
            raise ValueError("DINOv3 weights are gated. Pass a local checkpoint path.")
        from transformers import DINOv3ViTModel

        print(f"  [encoder] loading DINOv3 weights from {weights}")
        sd = torch.load(weights, map_location="cpu", weights_only=True)
        cfg = _infer_dinov3_config(sd)
        self.dinov3 = DINOv3ViTModel(cfg)
        remap = {(f"model.{k}" if k.startswith("layer.") else k): v for k, v in sd.items()}
        self.dinov3.load_state_dict(remap, strict=True)

        self.embed_dim = cfg.hidden_size
        self.patch_size = cfg.patch_size
        self.num_register = cfg.num_register_tokens
        self._patch_start = 1 + self.num_register
        self.num_patches = (image_size // cfg.patch_size) ** 2
        self.multiscale_layers = _resolve_multiscale(multiscale_layers, cfg.num_hidden_layers)

        for p in self.dinov3.parameters():
            p.requires_grad = False

    def train(self, mode=True):
        super().train(mode)
        self.dinov3.eval()
        return self

    @torch.no_grad()
    def forward(self, x):
        tokens = self.dinov3(x).last_hidden_state
        return tokens[:, self._patch_start:], tokens[:, :1]

    @torch.no_grad()
    def forward_multiscale(self, x):
        hs = self.dinov3(x, output_hidden_states=True).hidden_states
        return [hs[i + 1][:, self._patch_start:] for i in self.multiscale_layers]


# ──────────────────────────────────────────────────────────────────────
#  MAE ViT-L/16
# ──────────────────────────────────────────────────────────────────────

class MAEEncoder(nn.Module):
    """Frozen MAE ViT-L/16 (Meta, patch=16, dim=1024, 24 layers, has CLS)."""

    def __init__(self, weights, multiscale_layers=None, image_size=224):
        super().__init__()
        print(f"  [encoder] loading MAE weights from {weights}")
        sd = torch.load(weights, map_location="cpu", weights_only=True)
        if "model" in sd:
            sd = sd["model"]

        dim, patch, depth, heads = 1024, 16, 24, 16
        grid = image_size // patch
        self.embed_dim = dim
        self.num_patches = grid * grid

        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(3, dim, patch, patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, dim))
        self.blocks = nn.ModuleList([_Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

        self.load_state_dict(sd, strict=False)

        self.multiscale_layers = _resolve_multiscale(multiscale_layers, depth)
        for p in self.parameters():
            p.requires_grad = False

    def train(self, mode=True):
        super().train(False)
        return self

    def _embed(self, x):
        B = x.shape[0]
        t = self.patch_embed.proj(x).flatten(2).transpose(1, 2)
        return torch.cat([self.cls_token.expand(B, -1, -1), t], dim=1) + self.pos_embed

    @torch.no_grad()
    def forward(self, x):
        t = self._embed(x)
        for blk in self.blocks:
            t = blk(t)
        t = self.norm(t)
        return t[:, 1:], t[:, :1]

    @torch.no_grad()
    def forward_multiscale(self, x):
        t = self._embed(x)
        taps = []
        for i, blk in enumerate(self.blocks):
            t = blk(t)
            if i in self.multiscale_layers:
                taps.append(t[:, 1:])
        return taps


# ──────────────────────────────────────────────────────────────────────
#  T3 (sensor encoder + shared trunk)
# ──────────────────────────────────────────────────────────────────────

class T3Encoder(nn.Module):
    """Frozen T3 encoder (sensor-specific blocks + shared trunk, patch=16, dim=1024, has CLS).

    Loads from two files inside weights_dir:
      - encoder_mini.pth: patch embed, CLS, pos_embed, and the first N sensor blocks.
      - trunk.pth: shared trunk blocks and final LayerNorm.
    """

    def __init__(self, weights_dir, multiscale_layers=None, image_size=224):
        super().__init__()
        print(f"  [encoder] loading T3 from {weights_dir}")
        enc_sd = torch.load(os.path.join(weights_dir, "encoder_mini.pth"),
                            map_location="cpu", weights_only=True)
        trunk_sd = torch.load(os.path.join(weights_dir, "trunk.pth"),
                              map_location="cpu", weights_only=True)

        dim, patch, heads = 1024, 16, 16
        n_enc = 1 + max(int(k.split(".")[1]) for k in enc_sd if k.startswith("blocks."))
        n_trunk = 1 + max(int(k.split(".")[1]) for k in trunk_sd if k.startswith("blocks."))
        depth = n_enc + n_trunk
        grid = image_size // patch

        self.embed_dim = dim
        self.num_patches = grid * grid

        self.patch_embed = nn.Module()
        self.patch_embed.proj = nn.Conv2d(3, dim, patch, patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.num_patches, dim))
        self.blocks = nn.ModuleList([_Block(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)

        combined = {k: v for k, v in enc_sd.items()}
        for k, v in trunk_sd.items():
            if k.startswith("blocks."):
                parts = k.split(".", 2)
                combined[f"blocks.{int(parts[1]) + n_enc}.{parts[2]}"] = v
            else:
                combined[k] = v
        self.load_state_dict(combined, strict=True)

        self.multiscale_layers = _resolve_multiscale(multiscale_layers, depth)
        for p in self.parameters():
            p.requires_grad = False

    def train(self, mode=True):
        super().train(False)
        return self

    def _embed(self, x):
        B = x.shape[0]
        t = self.patch_embed.proj(x).flatten(2).transpose(1, 2)
        return torch.cat([self.cls_token.expand(B, -1, -1), t], dim=1) + self.pos_embed

    @torch.no_grad()
    def forward(self, x):
        t = self._embed(x)
        for blk in self.blocks:
            t = blk(t)
        t = self.norm(t)
        return t[:, 1:], t[:, :1]

    @torch.no_grad()
    def forward_multiscale(self, x):
        t = self._embed(x)
        taps = []
        for i, blk in enumerate(self.blocks):
            t = blk(t)
            if i in self.multiscale_layers:
                taps.append(t[:, 1:])
        return taps


# ──────────────────────────────────────────────────────────────────────
#  MockEncoder (testing stand-in)
# ──────────────────────────────────────────────────────────────────────

class MockEncoder(nn.Module):
    def __init__(self, embed_dim=1024, patch_size=16, image_size=224, multiscale_k=4):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.grid = image_size // patch_size
        self.num_patches = self.grid * self.grid
        self.multiscale_layers = list(range(multiscale_k))

        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_proj = nn.Linear(embed_dim, embed_dim)
        self.scale_projs = nn.ModuleList(
            [nn.Linear(embed_dim, embed_dim) for _ in range(multiscale_k)]
        )
        for p in self.parameters():
            p.requires_grad = False

    def _tokens(self, x):
        return self.patch_embed(x).flatten(2).transpose(1, 2)

    @torch.no_grad()
    def forward(self, x):
        patch = self._tokens(x)
        cls = self.cls_proj(patch.mean(dim=1, keepdim=True))
        return patch, cls

    @torch.no_grad()
    def forward_multiscale(self, x):
        patch = self._tokens(x)
        return [proj(patch) for proj in self.scale_projs]


# ──────────────────────────────────────────────────────────────────────
#  Factory
# ──────────────────────────────────────────────────────────────────────

def build_encoder(enc_cfg, image_size):
    enc_cfg = enc_cfg or {}
    checkpoint = enc_cfg.get("checkpoint", None)
    if not checkpoint:
        return MockEncoder(
            embed_dim=enc_cfg.get("embed_dim", 1024),
            patch_size=enc_cfg.get("patch_size", 16),
            image_size=image_size,
        )
    name = enc_cfg.get("name", "dinov3_vitl16")
    ms = enc_cfg.get("multiscale_layers", None)

    if name == "dinov3_vitl16":
        return DINOv3Encoder(weights=checkpoint, image_size=image_size, multiscale_layers=ms)
    if name == "mae_vitl16":
        return MAEEncoder(weights=checkpoint, multiscale_layers=ms, image_size=image_size)
    if name == "t3_large":
        return T3Encoder(weights_dir=checkpoint, multiscale_layers=ms, image_size=image_size)

    raise ValueError(
        f"Unknown encoder name {name!r}. Available: dinov3_vitl16, mae_vitl16, t3_large"
    )
