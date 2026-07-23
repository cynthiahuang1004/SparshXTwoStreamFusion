"""Depth reconstruction loss (adapted from VisTacFusion).

- Scale-shift-invariant (SSI / MiDaS-style) on normalized depth -- robust for sim2real.
- Optional gradient-matching term for sharp edges.
- Plain MSE / L1 fallbacks.

An optional mask restricts the loss to the contact region (mask > 0). When no mask is
passed the loss is computed over the full depth map (background is zero depth), matching
the VisTacFusion default.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _normalize_ssi(x, mask, eps=1e-6):
    """Per-sample shift (median) + scale (mean abs dev) normalization (MiDaS-style)."""
    B = x.shape[0]
    x_flat = x.view(B, -1)
    m_flat = mask.view(B, -1)
    out = torch.zeros_like(x_flat)
    for i in range(B):
        v = x_flat[i][m_flat[i] > 0.5]
        if v.numel() == 0:
            continue
        t = v.median()
        s = (v - t).abs().mean().clamp_min(eps)
        out[i] = (x_flat[i] - t) / s
    return out.view_as(x)


def gradient_matching_loss(pred, gt, scales=4):
    """Multi-scale gradient L1 (edge sharpness)."""
    total = 0.0
    for k in range(scales):
        p = pred[:, :, :: 2 ** k, :: 2 ** k]
        g = gt[:, :, :: 2 ** k, :: 2 ** k]
        dx = (p[:, :, :, 1:] - p[:, :, :, :-1]) - (g[:, :, :, 1:] - g[:, :, :, :-1])
        dy = (p[:, :, 1:, :] - p[:, :, :-1, :]) - (g[:, :, 1:, :] - g[:, :, :-1, :])
        total = total + dx.abs().mean() + dy.abs().mean()
    return total / scales


class DepthLoss(nn.Module):
    def __init__(self, kind="mse", grad_matching_weight=0.0):
        super().__init__()
        self.kind = kind
        self.grad_w = grad_matching_weight

    def forward(self, pred, gt, mask=None):
        if mask is None:
            mask = torch.ones_like(gt)
        if self.kind == "ssi":
            p = _normalize_ssi(pred, mask)
            g = _normalize_ssi(gt, mask)
            loss = (F.l1_loss(p, g, reduction="none") * mask).sum() / mask.sum().clamp_min(1)
        elif self.kind == "mse":
            loss = (F.mse_loss(pred, gt, reduction="none") * mask).sum() / mask.sum().clamp_min(1)
        else:  # l1
            loss = (F.l1_loss(pred, gt, reduction="none") * mask).sum() / mask.sum().clamp_min(1)
        if self.grad_w > 0:
            loss = loss + self.grad_w * gradient_matching_loss(pred, gt)
        return loss
