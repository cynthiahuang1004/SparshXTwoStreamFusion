"""Pose head: SE(2), 3 DoF.

Takes pose token [B, 1, D] + spatial queries [B, 196, D] (via pooling).
Rotation: regression (cos/sin) or classification into bins with soft-argmax.
Translation: regression (tx, ty).
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PoseHead(nn.Module):
    def __init__(self, dim=768, hidden_dim=256, dropout=0.0,
                 pose_mode="regression", rot_num_bins=72,
                 use_spatial_pool=True):
        super().__init__()
        self.pose_mode = pose_mode
        self.rot_num_bins = rot_num_bins
        self.use_spatial_pool = use_spatial_pool

        in_dim = dim * 2 if use_spatial_pool else dim

        if pose_mode == "classification":
            self.rot_head = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, rot_num_bins),
            )
            self.trans_head = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 2),
            )
            bin_centers = torch.linspace(-math.pi, math.pi, rot_num_bins + 1)[:-1]
            bin_centers = bin_centers + (math.pi / rot_num_bins)
            self.register_buffer("bin_centers", bin_centers)
        else:
            out_dim = 4
            self.net = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Linear(in_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, out_dim),
            )

    def _pool_input(self, pose_token, spatial_queries=None):
        x = pose_token.squeeze(1)
        if self.use_spatial_pool and spatial_queries is not None:
            pool = spatial_queries.mean(dim=1)
            x = torch.cat([x, pool], dim=-1)
        return x

    def forward(self, pose_token, spatial_queries=None):
        """
        pose_token: [B, 1, D]
        spatial_queries: [B, 196, D] (optional, for spatial pooling)

        classification -> dict(rot_logits=[B,bins], se2=[B,4], trans=[B,2])
        regression     -> dict(se2=[B,4])
        """
        x = self._pool_input(pose_token, spatial_queries)

        if self.pose_mode == "classification":
            rot_logits = self.rot_head(x)
            trans = self.trans_head(x)
            probs = F.softmax(rot_logits, dim=-1)
            cos = (probs * torch.cos(self.bin_centers)).sum(dim=-1)
            sin = (probs * torch.sin(self.bin_centers)).sum(dim=-1)
            cos_sin = F.normalize(torch.stack([cos, sin], dim=-1), dim=-1)
            se2 = torch.cat([cos_sin, trans], dim=-1)
            return {"rot_logits": rot_logits, "se2": se2, "trans": trans}

        out = self.net(x)
        ab = out[:, :2]
        txy = out[:, 2:]
        cos_sin = F.normalize(ab, dim=-1, eps=1e-6)
        return {"se2": torch.cat([cos_sin, txy], dim=-1)}
