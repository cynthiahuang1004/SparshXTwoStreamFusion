"""Total multi-task loss.

Grouped Kendall uncertainty weighting: dense group (depth + normal) and pose group
(rot + trans) each have independent learned log-variances that auto-balance WITHIN
the group. The two groups are combined with a fixed `dense_pose_ratio` so that pose
improvements cannot steal gradient from dense tasks.

Legacy mode (uncertainty_weighting: true, grouped_uncertainty: false) keeps the old
4-way global uncertainty for checkpoint compatibility.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .depth import DepthLoss
from .normal import NormalLoss
from .pose import PoseLoss


class MultiTaskLoss(nn.Module):
    def __init__(self, loss_cfg, pose_mode="regression", rot_num_bins=72):
        super().__init__()
        depth_cfg = loss_cfg.get("depth", {})
        normal_cfg = loss_cfg.get("normal", {})
        pose_cfg = loss_cfg.get("pose", {})

        self.w_depth = depth_cfg.get("weight", 1.0)
        self.w_normal = normal_cfg.get("weight", 1.0)
        self.w_rot = pose_cfg.get("rot_weight", 1.0)
        self.w_trans = pose_cfg.get("trans_weight", 1.0)
        self.uncertainty = loss_cfg.get("uncertainty_weighting", False)
        self.grouped = loss_cfg.get("grouped_uncertainty", False)
        self.dense_pose_ratio = loss_cfg.get("dense_pose_ratio", 1.0)

        self.depth_loss = DepthLoss(
            kind=depth_cfg.get("type", "mse"),
            grad_matching_weight=depth_cfg.get("grad_matching_weight", 0.0),
        )
        self.normal_loss = NormalLoss(kind=normal_cfg.get("type", "mse"))
        self.pose_loss = PoseLoss(
            pose_mode=pose_mode,
            rot_num_bins=rot_num_bins,
        )
        if self.uncertainty and self.grouped:
            self.log_var_dense = nn.Parameter(torch.zeros(2))  # depth, normal
            self.log_var_pose = nn.Parameter(torch.zeros(2))   # rot, trans
        elif self.uncertainty:
            self.log_var = nn.Parameter(torch.zeros(4))

    def forward(self, pred, gt, supervise_dense=True):
        """pred: model output dict. gt: dict with depth/normal/pose/mask."""
        comps = {}
        terms = []
        weights = []

        if supervise_dense:
            l_depth = self.depth_loss(pred["depth"], gt["depth"])
            l_normal = self.normal_loss(pred["normal"], gt["normal"])
            comps["depth"] = l_depth.detach()
            comps["normal"] = l_normal.detach()
            terms += [l_depth, l_normal]
            weights += [self.w_depth, self.w_normal]

        l_rot, l_trans = self.pose_loss(pred, gt["pose"])
        comps["pose_rot"] = l_rot.detach()
        comps["pose_trans"] = l_trans.detach()
        terms += [l_rot, l_trans]
        weights += [self.w_rot, self.w_trans]

        if self.uncertainty and self.grouped:
            # --- Grouped uncertainty: dense and pose balanced independently ---
            pose_total = (torch.exp(-self.log_var_pose[0]) * l_rot
                          + 0.5 * self.log_var_pose[0]
                          + torch.exp(-self.log_var_pose[1]) * l_trans
                          + 0.5 * self.log_var_pose[1])
            if supervise_dense:
                dense_total = (torch.exp(-self.log_var_dense[0]) * l_depth
                               + 0.5 * self.log_var_dense[0]
                               + torch.exp(-self.log_var_dense[1]) * l_normal
                               + 0.5 * self.log_var_dense[1])
                total = self.dense_pose_ratio * dense_total + pose_total
            else:
                total = pose_total
        elif self.uncertainty:
            idx = ([0, 1] if supervise_dense else []) + [2, 3]
            total = 0.0
            for t, j in zip(terms, idx):
                total = total + torch.exp(-self.log_var[j]) * t + 0.5 * self.log_var[j]
        else:
            total = sum(w * t for w, t in zip(weights, terms))

        comps["total"] = total.detach()
        return total, comps
