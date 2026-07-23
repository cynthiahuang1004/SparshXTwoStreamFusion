"""Frozen image encoders (same DINO method as VisTacFusion).

Two interchangeable encoders behind one interface:
  - DINOv3Encoder : real frozen DINOv3 (HuggingFace `transformers`). The gated weights load
                    from a local HF-format state dict (`embeddings.*`, `layer.N.*`) with no
                    architecture download; config is inferred from the checkpoint.
  - MockEncoder   : deterministic patch-embed stand-in with identical output shapes, so the
                    pipeline runs on CPU before/without the gated weights.

Interface (tokens at the encoder's native dim E):
    forward(x) -> (patch [B, N, E], cls [B, 1, E])

build_encoder() returns the real encoder when a checkpoint is set, else the mock.
"""
from __future__ import annotations

import torch
import torch.nn as nn


def _infer_dinov3_config(sd):
    """Infer a DINOv3ViTConfig from an HF-format state dict (works for any ViT size)."""
    from transformers import DINOv3ViTConfig

    pe = sd["embeddings.patch_embeddings.weight"]          # [hidden, 3, patch, patch]
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
        num_attention_heads=hidden // 64,                  # DINOv3 head_dim = 64
        num_register_tokens=num_register,
        image_size=224,
        use_gated_mlp=gated,
    )


class DINOv3Encoder(nn.Module):
    """Frozen DINOv3 (HuggingFace `transformers`).

    Loads the gated HF-format checkpoint directly. Token layout is
    [CLS, register x R, patch x N]; we expose patch tokens and CLS. All params frozen.
    """

    def __init__(self, weights, image_size=224):
        super().__init__()
        if weights is None:
            raise ValueError(
                "DINOv3 weights are gated. Pass a local checkpoint path, or use MockEncoder "
                "(set encoder.checkpoint: null) for scaffolding/tests."
            )
        from transformers import DINOv3ViTModel

        print(f"  [encoder] loading DINOv3 weights from {weights}")
        sd = torch.load(weights, map_location="cpu", weights_only=True)
        cfg = _infer_dinov3_config(sd)
        self.dinov3 = DINOv3ViTModel(cfg)
        # HF nests the transformer layers under `model.`; embeddings/norm stay top-level.
        remap = {(f"model.{k}" if k.startswith("layer.") else k): v for k, v in sd.items()}
        self.dinov3.load_state_dict(remap, strict=True)

        self.embed_dim = cfg.hidden_size
        self.patch_size = cfg.patch_size
        self.num_register = cfg.num_register_tokens
        self._patch_start = 1 + self.num_register               # skip CLS + registers
        self.num_patches = (image_size // cfg.patch_size) ** 2

        for p in self.dinov3.parameters():
            p.requires_grad = False

    def train(self, mode=True):
        # Keep the frozen backbone in eval mode regardless of the parent's train/eval.
        super().train(mode)
        self.dinov3.eval()
        return self

    @torch.no_grad()
    def forward(self, x):
        tokens = self.dinov3(x).last_hidden_state          # [B, 1+R+N, E]
        patch = tokens[:, self._patch_start:]              # [B, N, E]
        cls = tokens[:, :1]                                # [B, 1, E]
        return patch, cls


class MockEncoder(nn.Module):
    """Deterministic frozen stand-in for DINOv3 with identical output shapes.

    Patch-embed conv -> N tokens; CLS = linear(mean of tokens). All params frozen.
    """

    def __init__(self, embed_dim=1024, patch_size=16, image_size=224):
        super().__init__()
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.grid = image_size // patch_size
        self.num_patches = self.grid * self.grid

        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_proj = nn.Linear(embed_dim, embed_dim)
        for p in self.parameters():
            p.requires_grad = False

    def _tokens(self, x):
        f = self.patch_embed(x)                       # [B, E, g, g]
        return f.flatten(2).transpose(1, 2)           # [B, N, E]

    @torch.no_grad()
    def forward(self, x):
        patch = self._tokens(x)                                # [B, N, E]
        cls = self.cls_proj(patch.mean(dim=1, keepdim=True))   # [B, 1, E]
        return patch, cls


def build_encoder(enc_cfg, image_size):
    """Factory: real DINOv3 if a checkpoint is set, else the MockEncoder for testing."""
    enc_cfg = enc_cfg or {}
    checkpoint = enc_cfg.get("checkpoint", None)
    if checkpoint:
        return DINOv3Encoder(weights=checkpoint, image_size=image_size)
    return MockEncoder(
        embed_dim=enc_cfg.get("embed_dim", 1024),
        patch_size=enc_cfg.get("patch_size", 16),
        image_size=image_size,
    )
