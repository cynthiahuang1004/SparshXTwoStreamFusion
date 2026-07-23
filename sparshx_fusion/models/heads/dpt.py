"""DPT dense decoder head (adapted from VisTacFusion).

Input : 4 taps, each [B, N, D]  (N must be a square grid, e.g. 196 = 14x14).
Output: depth [B, 1, H, W]  (+ optional normal [B, 3, H, W]), upsampled to `out_hw`.

Reassemble scales {4, 2, 1, 0.5} build a multi-scale pyramid; FeatureFusion blocks merge
coarse->fine; a CNN head predicts depth (and optionally surface normals).

Compared with the VisTacFusion head this version makes the normal head optional
(`out_normal_channels=0` disables it) since the SparshX reconstruction task supervises
depth only by default.
"""
from __future__ import annotations

import torch.nn as nn
import torch.nn.functional as F


class Reassemble(nn.Module):
    def __init__(self, embed_dim, out_channels, scale_factor):
        super().__init__()
        self.norm = nn.LayerNorm(embed_dim)
        self.proj = nn.Conv2d(embed_dim, out_channels, kernel_size=1)
        self.scale_factor = scale_factor

    def forward(self, tokens):
        B, N, D = tokens.shape
        h = w = int(N ** 0.5)
        assert h * w == N, f"DPT Reassemble needs a square token grid, got N={N}"
        tokens = self.norm(tokens)
        x = tokens.permute(0, 2, 1).reshape(B, D, h, w)
        x = self.proj(x)
        if self.scale_factor != 1.0:
            x = F.interpolate(x, scale_factor=self.scale_factor,
                              mode="bilinear", align_corners=True)
        return x


class ResidualConvUnit(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(features)
        self.conv2 = nn.Conv2d(features, features, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(features)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(x)
        out = self.bn1(self.conv1(out))
        out = self.relu(out)
        out = self.bn2(self.conv2(out))
        return out + x


class FeatureFusionBlock(nn.Module):
    def __init__(self, features):
        super().__init__()
        self.rcu1 = ResidualConvUnit(features)
        self.rcu2 = ResidualConvUnit(features)

    def forward(self, x, skip=None):
        if skip is not None:
            x = x + self.rcu1(skip)
        x = self.rcu2(x)
        x = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=True)
        return x


class DPTHead(nn.Module):
    def __init__(self, embed_dim=768, features=256, dropout=0.0,
                 out_depth_channels=1, out_normal_channels=0):
        super().__init__()
        self.reassemble = nn.ModuleList([
            Reassemble(embed_dim, features, scale_factor=4.0),
            Reassemble(embed_dim, features, scale_factor=2.0),
            Reassemble(embed_dim, features, scale_factor=1.0),
            Reassemble(embed_dim, features, scale_factor=0.5),
        ])
        self.fusion = nn.ModuleList([FeatureFusionBlock(features) for _ in range(4)])
        self.drop = nn.Dropout2d(p=dropout)
        self.depth_head = self._make_head(features, out_depth_channels)
        self.normal_head = (
            self._make_head(features, out_normal_channels) if out_normal_channels > 0 else None
        )

    @staticmethod
    def _make_head(features, out_channels):
        return nn.Sequential(
            nn.Conv2d(features, features // 2, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True),
            nn.Conv2d(features // 2, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_channels, 1),
        )

    def forward(self, taps, out_hw):
        # taps: list of 4, each [B, N, D]  (shallow->deep ordering matches reassemble scales)
        maps = [r(t) for t, r in zip(taps, self.reassemble)]
        x = self.fusion[0](maps[3])
        x = self.fusion[1](x, maps[2])
        x = self.fusion[2](x, maps[1])
        x = self.fusion[3](x, maps[0])
        x = self.drop(x)
        depth = self.depth_head(x)
        normal = self.normal_head(x) if self.normal_head is not None else None
        out_hw = (int(out_hw[0]), int(out_hw[1]))
        if depth.shape[2:] != out_hw:
            depth = F.interpolate(depth, size=out_hw, mode="bilinear", align_corners=True)
            if normal is not None:
                normal = F.interpolate(normal, size=out_hw, mode="bilinear", align_corners=True)
        return depth, normal
