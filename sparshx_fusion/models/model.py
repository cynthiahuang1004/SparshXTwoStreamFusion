from __future__ import annotations

import random
from dataclasses import dataclass

import torch
import torch.nn as nn

from .encoders import build_encoder
from .heads import DPTHead, PoseHead
from .layers import MLPHead, SharedBottleneckFusionLayer, TransformerBlock, init_vit_weights

VALID_CONFIGS = ("both", "tactile", "rgb")


@dataclass
class ModelOutput:
    rgb_tokens: torch.Tensor
    tactile_tokens: torch.Tensor
    bottleneck_tokens: torch.Tensor
    fused_embedding: torch.Tensor | None = None
    logits: torch.Tensor | None = None
    regression: torch.Tensor | None = None
    depth: torch.Tensor | None = None
    normal: torch.Tensor | None = None
    se2: torch.Tensor | None = None


class SparshXTwoStreamFusionModel(nn.Module):
    """Two-stream visuo-tactile transformer with symmetric shared bottleneck fusion.

    Architecture (v2 — DPT decoupled from fusion trunk):
      - Frozen DINOv3 encoder (shared) produces patch tokens + multiscale taps
      - Patch tokens projected E→D, refined by independent blocks, then fused
      - DPT path: encoder multiscale taps @ E dim (decoupled from fusion)
      - Pose path: mean-pooled post-fusion tactile tokens + spatial pool @ D dim
      - Modality dropout: train with both/tactile/rgb configs; eval per-config
    """

    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_chans: int = 3,
        embed_dim: int = 192,
        depth: int = 8,
        num_heads: int = 3,
        mlp_ratio: float = 4.0,
        independent_layers: int = 4,
        fusion_layers: int = 4,
        num_bottleneck_tokens: int = 8,
        dropout: float = 0.0,
        task: str = "classification",
        num_classes: int = 4,
        regression_dim: int = 3,
        head_hidden_dim: int | None = None,
        head_layers: int = 2,
        head_dropout: float = 0.1,
        encoder: dict | None = None,
        dpt: dict | None = None,
        pose: dict | None = None,
    ):
        super().__init__()
        if independent_layers + fusion_layers != depth:
            raise ValueError("independent_layers + fusion_layers must equal depth")
        if task not in {"classification", "regression", "embedding", "reconstruction"}:
            raise ValueError(
                "task must be one of: classification, regression, embedding, reconstruction"
            )

        self.task = task
        self.image_size = image_size
        self.embed_dim = embed_dim
        self.num_bottleneck_tokens = num_bottleneck_tokens

        # ---- Frozen DINO encoders (shared or two instances) ----
        enc_cfg = dict(encoder or {})
        self.tactile_encoder = build_encoder(enc_cfg, image_size)
        if enc_cfg.get("share_encoder_weights", True):
            self.rgb_encoder = self.tactile_encoder
        else:
            self.rgb_encoder = build_encoder(enc_cfg, image_size)
        enc_dim = self.tactile_encoder.embed_dim
        num_patches = self.tactile_encoder.num_patches

        # ---- Projections (E -> D) ----
        self.rgb_proj = nn.Linear(enc_dim, embed_dim)
        self.tactile_proj = nn.Linear(enc_dim, embed_dim)

        self.rgb_pos = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.tactile_pos = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        self.rgb_modality = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.tactile_modality = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.bottleneck = nn.Parameter(torch.zeros(1, num_bottleneck_tokens, embed_dim))

        # Learnable mask tokens for modality dropout
        self.rgb_mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.tactile_mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        self.rgb_blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(independent_layers)]
        )
        self.tactile_blocks = nn.ModuleList(
            [TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(independent_layers)]
        )
        self.fusion_blocks = nn.ModuleList(
            [SharedBottleneckFusionLayer(embed_dim, num_heads, mlp_ratio, dropout) for _ in range(fusion_layers)]
        )

        fused_dim = embed_dim * 3
        self.norm = nn.LayerNorm(fused_dim) if task != "reconstruction" else None
        self.classifier = (
            MLPHead(fused_dim, num_classes, head_hidden_dim, head_layers, head_dropout)
            if task == "classification"
            else None
        )
        self.regressor = (
            MLPHead(fused_dim, regression_dim, head_hidden_dim, head_layers, head_dropout)
            if task == "regression"
            else None
        )

        # ---- Dense DPT decoder: uses encoder multiscale taps @ enc_dim ----
        dpt_cfg = dict(dpt or {})
        pose_cfg = dict(pose or {})
        self.dpt = (
            DPTHead(
                embed_dim=enc_dim,
                features=dpt_cfg.get("features", 256),
                dropout=dpt_cfg.get("dropout", 0.0),
                out_depth_channels=dpt_cfg.get("out_depth_channels", 1),
                out_normal_channels=dpt_cfg.get("out_normal_channels", 3),
            )
            if task == "reconstruction"
            else None
        )
        # Spatial pos embedding for DPT taps (at encoder dim)
        self.dpt_pos = (
            nn.Parameter(torch.zeros(1, num_patches, enc_dim))
            if task == "reconstruction"
            else None
        )
        self.pose_head = (
            PoseHead(
                dim=embed_dim,
                hidden_dim=pose_cfg.get("hidden_dim", 256),
                dropout=pose_cfg.get("dropout", 0.1),
                pose_mode=pose_cfg.get("pose_mode", "regression"),
                rot_num_bins=pose_cfg.get("rot_num_bins", 72),
                use_spatial_pool=pose_cfg.get("use_spatial_pool", True),
            )
            if task == "reconstruction"
            else None
        )

        # Initialize ONLY the trainable modules; never touch the (pretrained/frozen) encoders.
        trainable = [
            self.rgb_proj,
            self.tactile_proj,
            self.rgb_blocks,
            self.tactile_blocks,
            self.fusion_blocks,
        ]
        for m in [self.norm, self.classifier, self.regressor, self.dpt, self.pose_head]:
            if m is not None:
                trainable.append(m)
        for module in trainable:
            module.apply(init_vit_weights)
        nn.init.trunc_normal_(self.rgb_pos, std=0.02)
        nn.init.trunc_normal_(self.tactile_pos, std=0.02)
        nn.init.trunc_normal_(self.rgb_modality, std=0.02)
        nn.init.trunc_normal_(self.tactile_modality, std=0.02)
        nn.init.trunc_normal_(self.bottleneck, std=0.02)
        nn.init.trunc_normal_(self.rgb_mask_token, std=0.02)
        nn.init.trunc_normal_(self.tactile_mask_token, std=0.02)
        if self.dpt_pos is not None:
            nn.init.trunc_normal_(self.dpt_pos, std=0.02)

    def forward(self, rgb: torch.Tensor, tactile: torch.Tensor,
                config: str = "both") -> ModelOutput:
        if config not in VALID_CONFIGS:
            raise ValueError(f"config must be one of {VALID_CONFIGS}, got {config!r}")
        use_rgb = config in ("both", "rgb")
        use_tactile = config in ("both", "tactile")

        B = rgb.shape[0]

        # ---- Encode (frozen) ----
        if use_rgb:
            rgb_patch, _ = self.rgb_encoder(rgb)
        if use_tactile:
            tactile_patch, _ = self.tactile_encoder(tactile)

        # ---- Project to fusion dim, add pos + modality embeddings ----
        if use_rgb:
            rgb_tokens = self.rgb_proj(rgb_patch) + self.rgb_pos + self.rgb_modality
        else:
            rgb_tokens = self.rgb_mask_token.expand(B, self.rgb_pos.shape[1], -1) + self.rgb_pos + self.rgb_modality

        if use_tactile:
            tactile_tokens = self.tactile_proj(tactile_patch) + self.tactile_pos + self.tactile_modality
        else:
            tactile_tokens = self.tactile_mask_token.expand(B, self.tactile_pos.shape[1], -1) + self.tactile_pos + self.tactile_modality

        # ---- Independent pre-fusion refinement ----
        for rgb_block, tactile_block in zip(self.rgb_blocks, self.tactile_blocks):
            rgb_tokens = rgb_block(rgb_tokens)
            tactile_tokens = tactile_block(tactile_tokens)

        # ---- Shared bottleneck fusion ----
        bottleneck = self.bottleneck.expand(B, -1, -1)
        for fusion_block in self.fusion_blocks:
            rgb_tokens, tactile_tokens, bottleneck = fusion_block(rgb_tokens, tactile_tokens, bottleneck)

        # ---- DPT path: encoder multiscale taps (decoupled from fusion trunk) ----
        depth = normal = se2 = None
        if self.dpt is not None:
            if use_tactile:
                dpt_taps = self.tactile_encoder.forward_multiscale(tactile)
            else:
                dpt_taps = self.rgb_encoder.forward_multiscale(rgb)
            dpt_taps = [t + self.dpt_pos for t in dpt_taps]
            depth, normal = self.dpt(dpt_taps, out_hw=(self.image_size, self.image_size))

        # ---- Pose path: from post-fusion tokens ----
        if self.pose_head is not None:
            if use_tactile:
                pose_token = tactile_tokens.mean(dim=1, keepdim=True)
                spatial_queries = tactile_tokens
            else:
                pose_token = rgb_tokens.mean(dim=1, keepdim=True)
                spatial_queries = rgb_tokens
            # Concatenate bottleneck info for richer pose signal
            pose_token = pose_token + bottleneck.mean(dim=1, keepdim=True)
            se2 = self.pose_head(pose_token, spatial_queries=spatial_queries)["se2"]

        # The pooled global embedding / classifier / regressor are not used by the dense
        # reconstruction head, so skip them (and their cost) for that task.
        fused = logits = regression = None
        if self.task != "reconstruction":
            fused = self.pool(rgb_tokens, tactile_tokens, bottleneck)
            logits = self.classifier(fused) if self.classifier is not None else None
            regression = self.regressor(fused) if self.regressor is not None else None

        return ModelOutput(
            rgb_tokens=rgb_tokens,
            tactile_tokens=tactile_tokens,
            bottleneck_tokens=bottleneck,
            fused_embedding=fused,
            logits=logits,
            regression=regression,
            depth=depth,
            normal=normal,
            se2=se2,
        )

    def pool(self, rgb_tokens: torch.Tensor, tactile_tokens: torch.Tensor, bottleneck: torch.Tensor) -> torch.Tensor:
        rgb_feat = rgb_tokens.mean(dim=1)
        tactile_feat = tactile_tokens.mean(dim=1)
        bottleneck_feat = bottleneck.mean(dim=1)
        fused = torch.cat([rgb_feat, tactile_feat, bottleneck_feat], dim=-1)
        return self.norm(fused) if self.norm is not None else fused
