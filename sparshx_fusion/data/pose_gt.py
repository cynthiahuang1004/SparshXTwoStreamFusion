"""SE(2) pose ground truth for gs_blender (matches updated VisTacFusion).

Uses mesh-based session center, relative delta_rz vs session_000, and object-frame
translation normalized by target_size — same logic as VisTacFusion vistacfusion/data/dataset.py
after commit 4f31084 (pose_calculation.py).
"""
from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np
import torch


def load_object_pose_info(root: str | Path, mesh_dir: str | Path) -> dict[str, dict]:
    """Pre-load mesh + session_000 info for each object under `root`."""
    root = Path(root)
    mesh_dir = Path(mesh_dir)
    obj_pose_info: dict[str, dict] = {}
    try:
        import trimesh  # noqa: F401
    except ImportError:
        print("[WARN] trimesh not installed — pose labels will fall back to zeros")
        return obj_pose_info

    for obj_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        obj_name = obj_dir.name
        mesh_path = mesh_dir / f"{obj_name}.obj"
        s0_path = obj_dir / "session_000" / "session.json"
        if not mesh_path.exists() or not s0_path.exists():
            continue
        mesh = __import__("trimesh").load(str(mesh_path), force="mesh")
        with s0_path.open() as f:
            d0 = json.load(f)
        fixed_scale = d0["fixed_scale"]
        target_size = d0.get("_target_size_mm", 82.0)
        half = target_size / 2.0 / 1000.0
        rz0 = d0["base_rotation"][2]
        obj_pose_info[obj_name] = {
            "vertices": mesh.vertices,
            "fixed_scale": fixed_scale,
            "half": half,
            "rz0": rz0,
        }
    return obj_pose_info


def get_session_center(vertices, fixed_scale, base_rotation) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    R_3d = Rotation.from_euler("xyz", base_rotation)
    v = R_3d.apply(vertices) / fixed_scale
    cx = (v[:, 0].min() + v[:, 0].max()) / 2.0
    cy = (v[:, 1].min() + v[:, 1].max()) / 2.0
    return np.array([cx, cy])


def build_unit_pose_meta(
    session_json: str | Path,
    obj_name: str,
    obj_pose_info: dict[str, dict],
    image_size: int,
) -> dict | None:
    """Build per-sensor-unit metadata for depth normals + SE(2) pose."""
    info = obj_pose_info.get(obj_name)
    if info is None:
        return None
    with Path(session_json).open() as f:
        sess = json.load(f)
    x_min, x_max = sess["X_MIN"], sess["X_MAX"]
    y_min, y_max = sess["Y_MIN"], sess["Y_MAX"]
    base_rot = sess["base_rotation"]
    delta_rz = base_rot[2] - info["rz0"]
    session_center = get_session_center(info["vertices"], info["fixed_scale"], base_rot)
    return {
        "pixel_size_x": (x_max - x_min) / image_size,
        "pixel_size_y": (y_max - y_min) / image_size,
        "delta_rz": delta_rz,
        "rz0": info["rz0"],
        "session_center": session_center,
        "half": info["half"],
    }


def build_unit_meta_map(
    samples: list[dict[str, str]],
    obj_pose_info: dict[str, dict],
    image_size: int,
) -> dict[str, dict]:
    """unit path -> meta dict (pixel sizes + pose fields)."""
    meta: dict[str, dict] = {}
    for s in samples:
        unit = s.get("unit") or ""
        if not unit or unit in meta:
            continue
        sj = s.get("session_json") or ""
        obj = s.get("object") or ""
        if not sj or not obj:
            continue
        unit_meta = build_unit_pose_meta(sj, obj, obj_pose_info, image_size)
        if unit_meta is not None:
            meta[unit] = unit_meta
    return meta


def load_pose_se2(pose_path: str | Path, meta: dict) -> torch.Tensor:
    """SE(2) label: (cos delta_rz, sin delta_rz, tx_norm, ty_norm) in object frame.

    Translation: R(+theta) @ (sx, sy) / half, matching VisTacFusion's corrected
    coordinate convention (sample_x/sample_y are press offsets from session center
    with x negated in world axes).

    Uses per-sample rotation_euler[2] from the pose json (not session-level delta_rz).
    In sim all samples in a session share the same rotation_euler[2] ==
    base_rotation[2], so this is equivalent. In real data each sample has its own
    rotation_euler[2] within a single session.
    """
    with Path(pose_path).open() as f:
        data = json.load(f)

    half = meta["half"]

    # Per-sample rotation from pose json instead of session-level delta_rz
    delta_rz = data["rotation_euler"][2] - meta["rz0"]

    cos_rz = math.cos(delta_rz)
    sin_rz = math.sin(delta_rz)
    sx, sy = data["sample_x"], data["sample_y"]
    x_norm = (cos_rz * sx - sin_rz * sy) / max(half, 1e-8)
    y_norm = (sin_rz * sx + cos_rz * sy) / max(half, 1e-8)

    return torch.tensor(
        [cos_rz, sin_rz, x_norm, y_norm],
        dtype=torch.float32,
    )


def filter_samples_with_pose_meta(
    samples: list[dict[str, str]],
    unit_meta: dict[str, dict],
) -> list[dict[str, str]]:
    return [s for s in samples if s.get("unit") in unit_meta]
