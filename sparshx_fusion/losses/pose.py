"""Pose loss (SE(2)).

Classification: cross-entropy over rotation bins + L1 translation.
Regression: 1 - cos(theta_pred - theta_gt) + L1 translation.
Both modes also output se2 for the eval metrics.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PoseLoss(nn.Module):
    def __init__(self, pose_mode="classification", rot_num_bins=72):
        super().__init__()
        self.pose_mode = pose_mode
        self.rot_num_bins = rot_num_bins

    def _theta_to_bin(self, theta):
        """Map angle [-pi, pi] to bin index [0, rot_num_bins-1]."""
        bin_size = 2 * math.pi / self.rot_num_bins
        bins = ((theta + math.pi) / bin_size).long()
        return bins.clamp(0, self.rot_num_bins - 1)

    def forward(self, pred, gt):
        """gt: [B, 4] = (cos, sin, t_x, t_y). Returns (l_rot, l_trans) separately."""
        cos_gt, sin_gt, txy_gt = gt[:, 0], gt[:, 1], gt[:, 2:]

        if self.pose_mode == "classification":
            theta_gt = torch.atan2(sin_gt, cos_gt)
            target_bins = self._theta_to_bin(theta_gt)
            l_rot = F.cross_entropy(pred["rot_logits"], target_bins)
            l_trans = F.l1_loss(pred["trans"], txy_gt)
        else:
            se2 = pred["se2"]
            cos_p, sin_p, txy_p = se2[:, 0], se2[:, 1], se2[:, 2:]
            l_rot = (1.0 - (cos_p * cos_gt + sin_p * sin_gt)).mean()
            l_trans = F.l1_loss(txy_p, txy_gt)

        return l_rot, l_trans
