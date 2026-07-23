"""Normal loss: MSE or cosine/angular, selectable by kind."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class NormalLoss(nn.Module):
    def __init__(self, kind="mse", eps=1e-6):
        super().__init__()
        self.kind = kind
        self.eps = eps

    def forward(self, pred, gt, mask=None):
        if self.kind == "mse":
            loss = F.mse_loss(pred, gt, reduction="none")
            if mask is not None:
                return (loss * mask).sum() / (mask.sum().clamp_min(1) * pred.shape[1])
            return loss.mean()
        # cosine: L2-normalize both, then 1 - cos
        p = F.normalize(pred, dim=1, eps=self.eps)
        g = F.normalize(gt, dim=1, eps=self.eps)
        cos = (p * g).sum(dim=1, keepdim=True)
        loss = 1.0 - cos
        if mask is not None:
            return (loss * mask).sum() / mask.sum().clamp_min(1)
        return loss.mean()
