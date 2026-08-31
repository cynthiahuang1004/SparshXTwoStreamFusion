"""SparshX two-stream (MBT) fusion baseline built on VisTacFusion components.

Everything except the fusion trunk is imported from ``vistacfusion`` (frozen encoders,
projection, positional embeddings, object/domain embeddings, TapInjection, DPT head,
Pose head) and the model follows ``VisuoTactileModel.forward`` exactly (modality configs,
decoupled RGB->DPT injection flag, encoder cache, rgb-only DPT fallback). It is trained
with VisTacFusion's own ``vistacfusion.engine.train`` via ``model_type: external``, so
data processing, losses, modality dropout, optimizer, evaluation and checkpoint rules are
literally the same code as the VisTacFusion runs it is compared against.

The ONLY difference: ``MBTFusionTrunk`` (symmetric shared-bottleneck averaging, MBT /
Nagrani et al. 2021) replaces VisTacFusion's asymmetric cross-attention ``FusionTrunk``.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from vistacfusion.models.encoders import build_encoder
from vistacfusion.models.heads.dpt import DPTHead
from vistacfusion.models.heads.pose import PoseHead
from vistacfusion.models.model import VALID_CONFIGS, TapInjection, _config_flags, _resample_tokens
from vistacfusion.models.projection import BranchProjection, SpatialPosEmbedding

from .layers import init_vit_weights
from .model import MBTFusionTrunk


class SparshXTwoStreamFusionVTF(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.image_size = cfg.image_size
        self.trunk_dim = cfg.trunk_dim

        # ---- Frozen encoders (identical construction to VisuoTactileModel) ----
        enc_cfg = dict(cfg.encoder)
        enc_cfg["multiscale_layers"] = list(cfg.heads.dpt.encoder_tap_layers)
        self.tactile_encoder = build_encoder(enc_cfg, self.image_size)
        rgb_cfg = cfg.get("rgb_encoder", None)
        if rgb_cfg is not None:
            rgb_enc_cfg = dict(rgb_cfg)
            if "multiscale_layers" not in rgb_enc_cfg:
                rgb_enc_cfg["multiscale_layers"] = list(cfg.heads.dpt.encoder_tap_layers)
            self.rgb_encoder = build_encoder(rgb_enc_cfg, self.image_size)
        elif enc_cfg.get("share_encoder_weights", True):
            self.rgb_encoder = self.tactile_encoder
        else:
            self.rgb_encoder = build_encoder(enc_cfg, self.image_size)

        self.enc_dim = self.tactile_encoder.embed_dim
        self.rgb_enc_dim = self.rgb_encoder.embed_dim
        self.num_spatial = self.tactile_encoder.num_patches

        # ---- Pose path: projection -> MBT trunk ----
        self.tactile_proj = BranchProjection(self.enc_dim, self.trunk_dim)
        self.rgb_proj = BranchProjection(self.rgb_enc_dim, self.trunk_dim)
        self.spatial_pos = SpatialPosEmbedding(self.num_spatial, self.trunk_dim)
        self.spatial_mask = nn.Parameter(torch.zeros(1, self.num_spatial, self.trunk_dim))
        self.pose_mask = nn.Parameter(torch.zeros(1, 1, self.trunk_dim))
        nn.init.trunc_normal_(self.spatial_mask, std=0.02)
        nn.init.trunc_normal_(self.pose_mask, std=0.02)

        ft = cfg.fusion_trunk
        self.trunk = MBTFusionTrunk(
            num_layers=ft.get("num_layers", 4), dim=self.trunk_dim,
            num_heads=ft.get("num_heads", 8),
            num_bottleneck_tokens=ft.get("num_bottleneck_tokens", 4),
            mlp_ratio=ft.get("ffn_mult", ft.get("mlp_ratio", 4)),
            dropout=ft.get("dropout", 0.1),
            tap_layers=list(ft.get("tap_layers", [0, 1, 2, 3])),
        )

        self.use_obj_emb = cfg.tokens.get("object_embedding", False)
        if self.use_obj_emb:
            self.obj_embedding = nn.Embedding(cfg.tokens.get("num_objects", 20), self.trunk_dim)
        self.use_domain_emb = cfg.tokens.get("domain_embedding", False)
        if self.use_domain_emb:
            self.domain_embedding = nn.Embedding(2, self.trunk_dim)

        # ---- DPT path: encoder multiscale taps + gated RGB injection (as VisTacFusion) ----
        self.dpt_pos = SpatialPosEmbedding(self.num_spatial, self.enc_dim)
        self.tap_inject = nn.ModuleList([
            TapInjection(q_dim=self.enc_dim, kv_dim=self.trunk_dim,
                         num_heads=ft.get("num_heads", 8), dropout=ft.get("dropout", 0.1),
                         gate_init=cfg.heads.dpt.get("inject_gate_init", 0.0))
            for _ in range(4)
        ])
        d = cfg.heads.dpt
        self.dpt = DPTHead(embed_dim=self.enc_dim, features=d.get("features", 256),
                           dropout=d.get("dropout", 0.0),
                           out_depth_channels=d.get("out_depth_channels", 1),
                           out_normal_channels=d.get("out_normal_channels", 3))
        p = cfg.heads.pose
        self.pose_head = PoseHead(dim=self.trunk_dim, hidden_dim=p.get("hidden_dim", 256),
                                  dropout=p.get("dropout", 0.0),
                                  pose_mode=p.get("pose_mode", "regression"),
                                  rot_num_bins=p.get("rot_num_bins", 72),
                                  use_spatial_pool=p.get("use_spatial_pool", True))

        # MBT trunk init (VisTacFusion's FusionTrunk uses PyTorch defaults + trunc_normal
        # bottleneck; the MBT blocks are timm-style nn.Linear/LayerNorm so init them as ViT)
        self.trunk.layers.apply(init_vit_weights)

    # -- identical helpers to VisuoTactileModel --
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

    def forward(self, rgb, tactile, config="both", inject_rgb_to_dpt=None,
                encoder_cache=None, object_ids=None, domain_ids=None):
        if config not in VALID_CONFIGS:
            raise ValueError(f"config must be one of {VALID_CONFIGS}, got {config!r}")
        use_rgb, use_tactile = _config_flags(config)
        ref = tactile if use_tactile else rgb
        B, device = ref.shape[0], ref.device
        if inject_rgb_to_dpt is None:
            inject_rgb_to_dpt = use_rgb

        tac_enc = encoder_cache.get("tactile") if encoder_cache else None
        rgb_enc = encoder_cache.get("rgb") if encoder_cache else None
        tac_patch = tac_cls = rgb_patch = rgb_cls = None
        if use_tactile:
            tac_patch, tac_cls = tac_enc if tac_enc else self.tactile_encoder(tactile)
        if use_rgb:
            rgb_patch, rgb_cls = rgb_enc if rgb_enc else self.rgb_encoder(rgb)

        pose_memory = self._build_pose_memory(rgb_patch, rgb_cls) if use_rgb else None
        pose_queries = self._build_pose_queries(tac_patch, tac_cls, use_tactile, B, device)
        if self.use_obj_emb and object_ids is not None:
            pose_queries = pose_queries + self.obj_embedding(object_ids).unsqueeze(1)
        if self.use_domain_emb and domain_ids is not None:
            pose_queries = pose_queries + self.domain_embedding(domain_ids).unsqueeze(1)

        trunk_taps, pose_token, bottleneck = self.trunk(pose_queries, pose_memory, use_rgb)
        pose = self.pose_head(pose_token, spatial_queries=trunk_taps[-1])

        tac_ms = encoder_cache.get("tactile_ms") if encoder_cache else None
        rgb_ms = encoder_cache.get("rgb_ms") if encoder_cache else None
        if use_tactile:
            ms = tac_ms if tac_ms is not None else self.tactile_encoder.forward_multiscale(tactile)
            dpt_taps = [self.dpt_pos(t) for t in ms]
        elif self.rgb_enc_dim == self.enc_dim:   # token grid resampled if patch sizes differ (as VTF)
            ms = rgb_ms if rgb_ms is not None else self.rgb_encoder.forward_multiscale(rgb)
            ms = [_resample_tokens(t, self.num_spatial) for t in ms]
            dpt_taps = [self.dpt_pos(t) for t in ms]
        else:
            dpt_taps = None

        if dpt_taps is not None:
            if inject_rgb_to_dpt and use_rgb:
                dpt_taps = [inj(t, bottleneck) for inj, t in zip(self.tap_inject, dpt_taps)]
            depth, normal = self.dpt(dpt_taps, out_hw=(self.image_size, self.image_size))
        else:
            depth = torch.zeros(B, 1, self.image_size, self.image_size, device=device)
            normal = torch.zeros(B, 3, self.image_size, self.image_size, device=device)

        out = {"depth": depth, "normal": normal}
        out.update(pose)
        return out


def build_model(cfg):
    """Entry point for VisTacFusion's ``model_type: external`` hook."""
    return SparshXTwoStreamFusionVTF(cfg)
