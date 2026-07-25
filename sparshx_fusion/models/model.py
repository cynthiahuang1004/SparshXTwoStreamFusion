"""MBT fusion ablation baseline — identical to VisTacFusion except the fusion trunk.

Dense (DPT) and Pose paths are decoupled (same as VisTacFusion v3):

  DPT path:  encoder multiscale taps at native dim (1024) → RGB injection via bottleneck
             → DPT Reassemble(1024→256) → depth/normal.

  Pose path: encoder → Projection(1024→768) → MBT Fusion Trunk(768) → Pose Head.
             Modality dropout (both/tactile/rgb) only affects this path.

The ONLY difference from VisTacFusion:
  - Fusion trunk uses symmetric shared-bottleneck (MBT) instead of asymmetric cross-attention.
  - MBT: both streams self-attend with shared bottleneck tokens, then bottleneck is averaged.
  - VisTacFusion: bottleneck cross-attends to RGB memory, then queries cross-attend to bottleneck.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .encoders import build_encoder
from .heads import DPTHead, PoseHead
from .layers import SharedBottleneckFusionLayer, init_vit_weights

VALID_CONFIGS = ("both", "tactile", "rgb")


@dataclass
class ModelOutput:
    depth: torch.Tensor | None = None
    normal: torch.Tensor | None = None
    se2: torch.Tensor | None = None


class BranchProjection(nn.Module):
    def __init__(self, in_dim=1024, out_dim=768):
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim)
        self.modality_emb = nn.Parameter(torch.zeros(1, 1, out_dim))
        nn.init.normal_(self.modality_emb, std=0.02)

    def forward(self, tokens):
        return self.proj(tokens) + self.modality_emb


class SpatialPosEmbedding(nn.Module):
    def __init__(self, num_tokens=196, dim=768):
        super().__init__()
        self.pos = nn.Parameter(torch.zeros(1, num_tokens, dim))
        nn.init.trunc_normal_(self.pos, std=0.02)

    def forward(self, x):
        return x + self.pos


class TapInjection(nn.Module):
    """Per-tap residual RGB injection: tap += gate * CrossAttn(Q=tap, K=V=bottleneck).

    Identical to VisTacFusion's TapInjection.
    Q is at encoder dim (e.g. 1024), KV at trunk dim (e.g. 768).
    ReZero gate init 0 → starts as pure encoder taps, learns to inject.
    """

    def __init__(self, q_dim, kv_dim, num_heads, dropout=0.0, gate_init=0.0):
        super().__init__()
        self.norm_q = nn.LayerNorm(q_dim)
        self.norm_kv = nn.LayerNorm(kv_dim)
        self.q_proj = nn.Linear(q_dim, q_dim)
        self.k_proj = nn.Linear(kv_dim, q_dim)
        self.v_proj = nn.Linear(kv_dim, q_dim)
        self.out_proj = nn.Linear(q_dim, q_dim)
        self.num_heads = num_heads
        self.head_dim = q_dim // num_heads
        self.gate = nn.Parameter(torch.tensor(float(gate_init)))

    def forward(self, tap, bottleneck):
        B, N, _ = tap.shape
        q = self.q_proj(self.norm_q(tap))
        kv_normed = self.norm_kv(bottleneck)
        k = self.k_proj(kv_normed)
        v = self.v_proj(kv_normed)

        q = q.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, -1, self.num_heads, self.head_dim).transpose(1, 2)

        attn = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        attn = attn.transpose(1, 2).reshape(B, N, -1)
        return tap + self.gate * self.out_proj(attn)


class MBTFusionTrunk(nn.Module):
    """Symmetric shared-bottleneck (MBT) fusion trunk.

    Same interface as VisTacFusion's FusionTrunk:
      forward(queries, memory, use_rgb) -> (taps, pose_token, bottleneck)

    Instead of asymmetric cross-attention, uses shared bottleneck:
      1. Prepend bottleneck to both query and memory streams
      2. Self-attend within each stream
      3. Average bottleneck from both streams
    """

    def __init__(self, num_layers=4, dim=768, num_heads=8, num_bottleneck_tokens=4,
                 mlp_ratio=4, dropout=0.1, tap_layers=None):
        super().__init__()
        self.num_layers = num_layers
        self.dim = dim
        self.num_bottleneck = num_bottleneck_tokens
        self.tap_layers = tap_layers or list(range(num_layers))
        assert len(self.tap_layers) == 4, "DPT needs exactly 4 taps."

        self.bottleneck = nn.Parameter(torch.zeros(1, num_bottleneck_tokens, dim))
        nn.init.trunc_normal_(self.bottleneck, std=0.02)

        self.layers = nn.ModuleList([
            SharedBottleneckFusionLayer(dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])

    def forward(self, queries, memory, use_rgb):
        B = queries.shape[0]
        bottleneck = self.bottleneck.expand(B, -1, -1)

        if not use_rgb:
            memory = torch.zeros(B, queries.shape[1], self.dim,
                                 device=queries.device, dtype=queries.dtype)

        taps = []
        for i, layer in enumerate(self.layers):
            queries, memory, bottleneck = layer(queries, memory, bottleneck)
            if i in self.tap_layers:
                taps.append(queries[:, :-1])

        pose_token = queries[:, -1:]
        return taps, pose_token, bottleneck


class SparshXTwoStreamFusionModel(nn.Module):
    """MBT fusion ablation — VisTacFusion with symmetric shared-bottleneck trunk.

    Everything except the fusion trunk is identical to VisTacFusion.
    """

    def __init__(
        self,
        image_size: int = 224,
        trunk_dim: int = 768,
        encoder: dict | None = None,
        fusion_trunk: dict | None = None,
        dpt: dict | None = None,
        pose: dict | None = None,
        inject_gate_init: float = 0.0,
        **kwargs,
    ):
        super().__init__()
        self.image_size = image_size
        self.trunk_dim = trunk_dim

        # ---- Encoder (frozen) ----
        enc_cfg = dict(encoder or {})
        multiscale_layers = enc_cfg.pop("multiscale_layers", [5, 11, 17, 23])
        enc_cfg["multiscale_layers"] = multiscale_layers
        self.tactile_encoder = build_encoder(enc_cfg, image_size)
        if enc_cfg.get("share_encoder_weights", True):
            self.rgb_encoder = self.tactile_encoder
        else:
            self.rgb_encoder = build_encoder(enc_cfg, image_size)
        self.enc_dim = self.tactile_encoder.embed_dim
        self.num_spatial = self.tactile_encoder.num_patches

        # ---- Pose path: projection (enc_dim→trunk_dim) ----
        self.tactile_proj = BranchProjection(self.enc_dim, trunk_dim)
        self.rgb_proj = BranchProjection(self.enc_dim, trunk_dim)
        self.spatial_pos = SpatialPosEmbedding(self.num_spatial, trunk_dim)

        self.spatial_mask = nn.Parameter(torch.zeros(1, self.num_spatial, trunk_dim))
        self.pose_mask = nn.Parameter(torch.zeros(1, 1, trunk_dim))
        nn.init.trunc_normal_(self.spatial_mask, std=0.02)
        nn.init.trunc_normal_(self.pose_mask, std=0.02)

        # ---- MBT Fusion Trunk (the ONLY difference from VisTacFusion) ----
        ft_cfg = dict(fusion_trunk or {})
        self.trunk = MBTFusionTrunk(
            num_layers=ft_cfg.get("num_layers", 4),
            dim=trunk_dim,
            num_heads=ft_cfg.get("num_heads", 8),
            num_bottleneck_tokens=ft_cfg.get("num_bottleneck_tokens", 4),
            mlp_ratio=ft_cfg.get("mlp_ratio", 4),
            dropout=ft_cfg.get("dropout", 0.1),
            tap_layers=ft_cfg.get("tap_layers", [0, 1, 2, 3]),
        )

        # ---- DPT path: encoder multiscale taps (1024) + RGB injection ----
        self.dpt_pos = SpatialPosEmbedding(self.num_spatial, self.enc_dim)

        dpt_cfg = dict(dpt or {})
        num_heads = ft_cfg.get("num_heads", 8)
        self.tap_inject = nn.ModuleList([
            TapInjection(
                q_dim=self.enc_dim,
                kv_dim=trunk_dim,
                num_heads=num_heads,
                dropout=ft_cfg.get("dropout", 0.1),
                gate_init=inject_gate_init,
            )
            for _ in range(4)
        ])

        self.dpt = DPTHead(
            embed_dim=self.enc_dim,
            features=dpt_cfg.get("features", 256),
            dropout=dpt_cfg.get("dropout", 0.1),
            out_depth_channels=dpt_cfg.get("out_depth_channels", 1),
            out_normal_channels=dpt_cfg.get("out_normal_channels", 3),
        )

        # ---- Pose head (identical to VisTacFusion) ----
        pose_cfg = dict(pose or {})
        self.pose_head = PoseHead(
            dim=trunk_dim,
            hidden_dim=pose_cfg.get("hidden_dim", 256),
            dropout=pose_cfg.get("dropout", 0.1),
            pose_mode=pose_cfg.get("pose_mode", "regression"),
            rot_num_bins=pose_cfg.get("rot_num_bins", 72),
            use_spatial_pool=pose_cfg.get("use_spatial_pool", True),
        )

        # ---- Initialize trainable modules (match VisTacFusion: only trunk + projections) ----
        for m in [self.tactile_proj, self.rgb_proj, self.trunk]:
            m.apply(init_vit_weights)

    def _build_pose_memory(self, rgb_patch, rgb_cls):
        patch = self.rgb_proj(rgb_patch)
        cls = self.rgb_proj(rgb_cls) if rgb_cls is not None else self.pose_mask.expand(patch.shape[0], -1, -1)
        return torch.cat([patch, cls], dim=1)

    def _build_pose_queries(self, tac_patch, tac_cls, use_tactile, B, device):
        if use_tactile:
            spatial = self.spatial_pos(self.tactile_proj(tac_patch))
            pose_q = self.tactile_proj(tac_cls) if tac_cls is not None else self.pose_mask.expand(B, -1, -1)
        else:
            spatial = self.spatial_pos(self.spatial_mask.expand(B, -1, -1))
            pose_q = self.pose_mask.expand(B, -1, -1)
        return torch.cat([spatial, pose_q], dim=1)

    def forward(self, rgb, tactile, config="both"):
        if config not in VALID_CONFIGS:
            raise ValueError(f"config must be one of {VALID_CONFIGS}, got {config!r}")
        use_rgb = config in ("both", "rgb")
        use_tactile = config in ("both", "tactile")

        ref = tactile if use_tactile else rgb
        B, device = ref.shape[0], ref.device

        # ---- Encode (frozen) ----
        tac_patch = tac_cls = rgb_patch = rgb_cls = None
        if use_tactile:
            tac_patch, tac_cls = self.tactile_encoder(tactile)
        if use_rgb:
            rgb_patch, rgb_cls = self.rgb_encoder(rgb)

        # ---- Pose path: projection → MBT trunk → pose head ----
        pose_memory = self._build_pose_memory(rgb_patch, rgb_cls) if use_rgb else None
        pose_queries = self._build_pose_queries(tac_patch, tac_cls, use_tactile, B, device)

        trunk_taps, pose_token, bottleneck = self.trunk(pose_queries, pose_memory, use_rgb)

        spatial_queries = trunk_taps[-1]
        pose_out = self.pose_head(pose_token, spatial_queries=spatial_queries)

        # ---- DPT path: encoder multiscale taps (1024) + RGB injection ----
        if use_tactile:
            ms = self.tactile_encoder.forward_multiscale(tactile)
        else:
            ms = self.rgb_encoder.forward_multiscale(rgb)
        dpt_taps = [self.dpt_pos(t) for t in ms]

        if use_rgb:
            dpt_taps = [inj(t, bottleneck)
                        for inj, t in zip(self.tap_inject, dpt_taps)]

        depth, normal = self.dpt(dpt_taps, out_hw=(self.image_size, self.image_size))

        return ModelOutput(depth=depth, normal=normal, se2=pose_out["se2"])
