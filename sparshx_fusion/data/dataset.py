from __future__ import annotations

import csv
import json
import math
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset, DistributedSampler, random_split
from torchvision import transforms

from .pose_gt import (
    build_unit_meta_map,
    filter_samples_with_pose_meta,
    load_object_pose_info,
    load_pose_se2,
)
from .transforms import (
    FIXED_CROP,
    RGBPhotometricAug,
    TactilePhotometricAug,
    fixed_center_crop,
    rotate_gel_spin,
    to_tensor_imagenet,
)

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class SyntheticPairDataset(Dataset):
    """Deterministic random paired-image dataset for smoke tests."""

    def __init__(
        self,
        num_samples: int,
        image_size: int = 224,
        num_classes: int = 4,
        regression_dim: int = 3,
        seed: int = 0,
        task: str = "classification",
    ):
        self.num_samples = num_samples
        self.image_size = image_size
        self.num_classes = num_classes
        self.regression_dim = regression_dim
        self.seed = seed
        self.task = task

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        g = torch.Generator().manual_seed(self.seed * 100_003 + idx)
        rgb = torch.randn(3, self.image_size, self.image_size, generator=g)
        tactile = torch.randn(3, self.image_size, self.image_size, generator=g)
        label = torch.tensor(idx % self.num_classes, dtype=torch.long)
        target = torch.randn(self.regression_dim, generator=g)
        # Dense depth + contact mask + normal + SE(2) pose for multi-task reconstruction.
        depth = torch.rand(1, self.image_size, self.image_size, generator=g)
        mask = (depth > 0.3).float()
        normal = torch.randn(3, self.image_size, self.image_size, generator=g)
        normal = normal / normal.norm(dim=0, keepdim=True).clamp_min(1e-6)
        theta = torch.rand(1, generator=g).item() * 2 * math.pi - math.pi
        tx = torch.rand(1, generator=g).item() * 2 - 1
        ty = torch.rand(1, generator=g).item() * 2 - 1
        pose = torch.tensor([math.cos(theta), math.sin(theta), tx, ty], dtype=torch.float32)
        return {
            "rgb": rgb,
            "tactile": tactile,
            "label": label,
            "target": target,
            "depth": depth,
            "normal": normal,
            "mask": mask,
            "pose": pose,
        }


class PairedFolderDataset(Dataset):
    """Dataset for paired RGB/tactile images with matching relative filenames."""

    def __init__(
        self,
        root: str | Path,
        rgb_dir: str = "rgb",
        tactile_dir: str = "tactile",
        labels_csv: str | None = "labels.csv",
        image_size: int = 224,
        task: str = "classification",
    ):
        self.root = Path(root)
        self.rgb_root = self.root / rgb_dir
        self.tactile_root = self.root / tactile_dir
        self.task = task
        if not self.rgb_root.exists():
            raise FileNotFoundError(f"Missing RGB directory: {self.rgb_root}")
        if not self.tactile_root.exists():
            raise FileNotFoundError(f"Missing tactile directory: {self.tactile_root}")

        self.transform = transforms.Compose(
            [
                transforms.Resize((image_size, image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ]
        )
        self.labels = self._read_labels(self.root / labels_csv) if labels_csv else {}
        self.samples = self._match_samples()
        if not self.samples:
            raise RuntimeError(f"No paired samples found under {self.root}")

    @staticmethod
    def _image_files(root: Path) -> list[Path]:
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
        return sorted(p for p in root.rglob("*") if p.suffix.lower() in exts)

    def _read_labels(self, path: Path) -> dict[str, dict[str, str]]:
        if not path.exists():
            return {}
        with path.open(newline="") as f:
            reader = csv.DictReader(f)
            return {row["name"]: row for row in reader}

    def _match_samples(self) -> list[tuple[Path, Path, str]]:
        rgb_files = self._image_files(self.rgb_root)
        samples = []
        for rgb_path in rgb_files:
            rel = rgb_path.relative_to(self.rgb_root)
            tactile_path = self.tactile_root / rel
            if tactile_path.exists():
                samples.append((rgb_path, tactile_path, rel.as_posix()))
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def _load_image(self, path: Path) -> torch.Tensor:
        img = Image.open(path).convert("RGB")
        return self.transform(img)

    def _make_supervision(self, name: str) -> dict[str, torch.Tensor]:
        row = self.labels.get(name) or self.labels.get(Path(name).name)
        if row is None:
            return {
                "label": torch.tensor(0, dtype=torch.long),
                "target": torch.zeros(1, dtype=torch.float32),
            }
        label = torch.tensor(int(row.get("label", 0)), dtype=torch.long)
        target_cols = [k for k in row if k.startswith("target_")]
        target_cols = sorted(target_cols, key=lambda k: int(k.split("_")[-1]))
        if target_cols:
            target = torch.tensor([float(row[k]) for k in target_cols], dtype=torch.float32)
        else:
            target = torch.zeros(1, dtype=torch.float32)
        return {"label": label, "target": target}

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        rgb_path, tactile_path, name = self.samples[idx]
        sample = {
            "rgb": self._load_image(rgb_path),
            "tactile": self._load_image(tactile_path),
        }
        sample.update(self._make_supervision(name))
        return sample


def _extract_pose_target(pose: dict, target_key: str) -> list[float]:
    """Pull a numeric target vector out of a gs_blender pose json dict."""
    if target_key == "location":
        return list(pose["location"])
    if target_key == "sample_xyz":
        return [pose["sample_x"], pose["sample_y"], pose["sample_z"]]
    if target_key == "sample_xy":
        return [pose["sample_x"], pose["sample_y"]]
    if target_key == "rotation_euler":
        return list(pose["rotation_euler"])
    raise ValueError(f"Unknown target_key: {target_key}")


def scan_gs_blender(
    root: str | Path,
    rgb_dir: str = "rgb",
    tactile_dir: str = "samples",
    raw_dir: str = "raw_data",
    require_depth: bool = False,
    use_gt_depth: bool = True,
) -> dict[tuple[str, str], list[dict[str, str]]]:
    """Scan the gs_blender renders layout into (object, session) -> samples.

    Expected layout:
        root/<object>/session_*/sensor_*/{rgb,samples}/XXXX.png
        root/<object>/session_*/sensor_*/raw_data/XXXX_pose.json
        root/<object>/session_*/sensor_*/raw_data/XXXX{_gt}.npy   (depth, optional)

    When `require_depth` is set, only samples whose depth .npy exists are kept and a
    "depth" path is recorded. `use_gt_depth` picks the clean GT map (`XXXX_gt.npy`) over
    the noisy render (`XXXX.npy`).
    """
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(f"gs_blender root does not exist: {root}")
    depth_suffix = "_gt" if use_gt_depth else ""
    groups: dict[tuple[str, str], list[dict[str, str]]] = {}
    for obj_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        for session_dir in sorted(obj_dir.glob("session_*")):
            for sensor_dir in sorted(session_dir.glob("sensor_*")):
                rgb_root = sensor_dir / rgb_dir
                tac_root = sensor_dir / tactile_dir
                raw_root = sensor_dir / raw_dir
                if not (rgb_root.is_dir() and tac_root.is_dir() and raw_root.is_dir()):
                    continue
                for rgb_path in sorted(rgb_root.glob("*.png")):
                    stem = rgb_path.stem
                    tac_path = tac_root / f"{stem}.png"
                    pose_path = raw_root / f"{stem}_pose.json"
                    depth_path = raw_root / f"{stem}{depth_suffix}.npy"
                    if not (tac_path.exists() and pose_path.exists()):
                        continue
                    if require_depth and not depth_path.exists():
                        continue
                    session_json = session_dir / "session.json"
                    key = (obj_dir.name, session_dir.name)
                    groups.setdefault(key, []).append(
                        {
                            "rgb": str(rgb_path),
                            "tactile": str(tac_path),
                            "pose": str(pose_path),
                            "depth": str(depth_path),
                            "unit": str(sensor_dir),
                            "session_json": str(session_json) if session_json.exists() else "",
                            "object": obj_dir.name,
                            "session": session_dir.name,
                        }
                    )
    if not groups:
        raise RuntimeError(f"No paired gs_blender samples found under {root}")
    return groups


def compute_target_stats(
    samples: list[dict[str, str]],
    target_key: str,
    max_n: int = 5000,
    seed: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate per-dimension target mean/std from a subset of samples."""
    rng = random.Random(seed)
    subset = samples if len(samples) <= max_n else rng.sample(samples, max_n)
    vals = []
    for s in subset:
        with open(s["pose"]) as f:
            pose = json.load(f)
        vals.append(_extract_pose_target(pose, target_key))
    t = torch.tensor(vals, dtype=torch.float32)
    mean = t.mean(dim=0)
    std = t.std(dim=0).clamp_min(1e-6)
    return mean, std


class GsBlenderPoseDataset(Dataset):
    """Paired RGB (object appearance) + tactile image dataset with pose targets.

    Returns standardized regression targets when target_mean/std are provided.
    """

    def __init__(
        self,
        samples: list[dict[str, str]],
        image_size: int = 224,
        target_key: str = "location",
        target_mean: torch.Tensor | None = None,
        target_std: torch.Tensor | None = None,
        augment: bool = False,
        tactile_aug_params: dict | None = None,
    ):
        self.samples = samples
        self.image_size = image_size
        self.target_key = target_key
        self.target_mean = target_mean
        self.target_std = target_std
        self.augment = augment
        # Label-preserving photometric augmentation (train only); geometry left untouched.
        self.tactile_aug = TactilePhotometricAug(tactile_aug_params) if augment else None
        self.rgb_aug = RGBPhotometricAug() if augment else None

    def __len__(self) -> int:
        return len(self.samples)

    def _load_np(self, path: str) -> np.ndarray:
        return np.array(Image.open(path).convert("RGB"), dtype=np.float32)  # HWC, 0-255

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s = self.samples[idx]
        with open(s["pose"]) as f:
            pose = json.load(f)
        raw_target = torch.tensor(_extract_pose_target(pose, self.target_key), dtype=torch.float32)
        target = raw_target
        if self.target_mean is not None and self.target_std is not None:
            target = (raw_target - self.target_mean) / self.target_std

        rgb = self._load_np(s["rgb"])
        tactile = self._load_np(s["tactile"])
        if self.augment:
            tactile = self.tactile_aug(tactile)
            rgb = self.rgb_aug(rgb)
        out_hw = (self.image_size, self.image_size)
        return {
            "rgb": to_tensor_imagenet(rgb, out_hw),
            "tactile": to_tensor_imagenet(tactile, out_hw),
            "target": target,
            "raw_target": raw_target,
            "label": torch.tensor(0, dtype=torch.long),
        }


def rotate_pose_theta(pose, dtheta_rad):
    """Shift the pose theta by dtheta_rad; (x, y) unchanged (gel spins in place)."""
    cos_t, sin_t = pose[0].item(), pose[1].item()
    c, s = math.cos(dtheta_rad), math.sin(dtheta_rad)
    cos_new = cos_t * c - sin_t * s
    sin_new = sin_t * c + cos_t * s
    return torch.tensor([cos_new, sin_new, pose[2].item(), pose[3].item()],
                        dtype=torch.float32)


class GsBlenderDepthDataset(Dataset):
    """Paired RGB + tactile with depth, normal, and SE(2) pose (VisTacFusion-style targets).

    Depth from raw_data/XXXX{_gt}.npy (x1000); normals from depth finite differences;
    pose = mesh-based delta_rz + object-frame translation.
    Gel-spin rotation augmentation + fixed 1/sqrt(2) center crop (matching VisTacFusion).
    """

    def __init__(
        self,
        samples: list[dict[str, str]],
        root: str | Path,
        image_size: int = 224,
        depth_scale: float = 1000.0,
        mesh_dir: str | Path | None = None,
        augment: bool = False,
        tactile_aug_params: dict | None = None,
        rot_augment: bool = True,
        rot_augment_max_deg: float = 180.0,
        gel_view_m: float = 0.017502,
    ):
        root = Path(root)
        self.image_size = image_size
        self.depth_scale = depth_scale
        self.augment = augment
        self.tactile_aug = TactilePhotometricAug(tactile_aug_params) if augment else None
        self.rgb_aug = RGBPhotometricAug() if augment else None
        self.rot_aug = augment and rot_augment
        self.rot_aug_max_deg = rot_augment_max_deg
        self.pixel_size = gel_view_m * FIXED_CROP / image_size

        mesh_dir = Path(mesh_dir) if mesh_dir else root.parent / "meshes"
        obj_pose_info = load_object_pose_info(root, mesh_dir)
        self.unit_meta = build_unit_meta_map(samples, obj_pose_info, image_size)
        self.samples = filter_samples_with_pose_meta(samples, self.unit_meta)
        if not self.samples:
            raise RuntimeError(
                f"No samples with valid pose metadata (check mesh_dir={mesh_dir} and session_000)"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def _load_np(self, path: str) -> np.ndarray:
        return np.array(Image.open(path).convert("RGB"), dtype=np.float32)

    def _load_depth_raw(self, path: str) -> np.ndarray:
        depth = np.load(path).astype(np.float32)
        t = torch.from_numpy(np.ascontiguousarray(depth)).unsqueeze(0).unsqueeze(0)
        out_hw = (self.image_size, self.image_size)
        if (t.shape[2], t.shape[3]) != out_hw:
            t = F.interpolate(t, size=out_hw, mode="nearest")
        return t.squeeze(0).squeeze(0).numpy()

    @staticmethod
    def depth_to_normal(depth, pixel_size_x, pixel_size_y):
        dz_dx = np.zeros_like(depth)
        dz_dy = np.zeros_like(depth)
        dz_dx[:, 1:-1] = (depth[:, 2:] - depth[:, :-2]) / (2.0 * pixel_size_x)
        dz_dy[1:-1, :] = (depth[2:, :] - depth[:-2, :]) / (2.0 * pixel_size_y)
        dz_dx[:, 0] = (depth[:, 1] - depth[:, 0]) / pixel_size_x
        dz_dx[:, -1] = (depth[:, -1] - depth[:, -2]) / pixel_size_x
        dz_dy[0, :] = (depth[1, :] - depth[0, :]) / pixel_size_y
        dz_dy[-1, :] = (depth[-1, :] - depth[-2, :]) / pixel_size_y
        normal = np.stack([-dz_dx, -dz_dy, np.ones_like(depth)], axis=-1)
        norm = np.linalg.norm(normal, axis=-1, keepdims=True).clip(min=1e-8)
        return (normal / norm).astype(np.float32)

    def _load_pose_se2(self, pose_path: str, unit: str) -> torch.Tensor:
        return load_pose_se2(pose_path, self.unit_meta[unit])

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        s = self.samples[idx]
        rgb = self._load_np(s["rgb"])
        tactile = self._load_np(s["tactile"])
        depth_raw = self._load_depth_raw(s["depth"])
        pose = self._load_pose_se2(s["pose"], s["unit"])

        if self.rot_aug:
            phi_deg = random.uniform(-self.rot_aug_max_deg, self.rot_aug_max_deg)
            tactile, rgb, depth_raw = rotate_gel_spin(tactile, rgb, depth_raw, phi_deg)
            pose = rotate_pose_theta(pose, -math.radians(phi_deg))

        tactile = fixed_center_crop(tactile)
        rgb = fixed_center_crop(rgb)
        depth_raw = fixed_center_crop(depth_raw)

        if self.tactile_aug is not None:
            tactile = self.tactile_aug(tactile)
        if self.rgb_aug is not None:
            rgb = self.rgb_aug(rgb)

        normal = self.depth_to_normal(depth_raw, self.pixel_size, self.pixel_size)
        mask = (depth_raw > 0).astype(np.float32)
        depth = depth_raw * self.depth_scale
        out_hw = (self.image_size, self.image_size)

        return {
            "rgb": to_tensor_imagenet(rgb, out_hw),
            "tactile": to_tensor_imagenet(tactile, out_hw),
            "depth": torch.from_numpy(depth).unsqueeze(0).float(),
            "normal": torch.from_numpy(np.ascontiguousarray(normal)).permute(2, 0, 1).float(),
            "mask": torch.from_numpy(mask).unsqueeze(0).float(),
            "pose": pose,
        }


def _build_gs_blender_recon(cfg: Any, image_size: int):
    """Build (train_ds, val_ds) for the gs_blender depth-reconstruction task.

    Per-session split: every val_every-th sample (by index) is val, the rest train.
    All objects/sessions appear in both splits — different press samples.
    """
    gb = cfg["data"]["gs_blender"]
    groups = scan_gs_blender(
        gb["root"],
        rgb_dir=gb.get("rgb_dir", "rgb"),
        tactile_dir=gb.get("tactile_dir", "samples"),
        raw_dir=gb.get("raw_dir", "raw_data"),
        require_depth=True,
        use_gt_depth=gb.get("use_gt_depth", True),
    )
    val_every = gb.get("val_every", 20)
    all_samples = [s for k in sorted(groups.keys()) for s in groups[k]]
    train_samples = []
    val_samples = []
    for s in all_samples:
        stem = Path(s["rgb"]).stem
        idx = int(stem)
        if idx % val_every == 0:
            val_samples.append(s)
        else:
            train_samples.append(s)
    print(
        f"gs_blender(recon): {len(all_samples)} total samples "
        f"(val_every={val_every}) -> train={len(train_samples)} val={len(val_samples)}"
    )

    augment = gb.get("augment", True)
    aug_params = gb.get("tactile_aug_params", None)
    depth_scale = gb.get("depth_scale", 1000.0)
    mesh_dir = gb.get("mesh_dir")
    rot_augment = gb.get("rot_augment", True)
    rot_augment_max_deg = gb.get("rot_augment_max_deg", 180.0)
    gel_view_m = gb.get("gel_view_m", 0.017502)
    train_ds = GsBlenderDepthDataset(
        train_samples,
        root=gb["root"],
        image_size=image_size,
        depth_scale=depth_scale,
        mesh_dir=mesh_dir,
        augment=augment,
        tactile_aug_params=aug_params,
        rot_augment=rot_augment,
        rot_augment_max_deg=rot_augment_max_deg,
        gel_view_m=gel_view_m,
    )
    val_ds = GsBlenderDepthDataset(
        val_samples,
        root=gb["root"],
        image_size=image_size,
        depth_scale=depth_scale,
        mesh_dir=mesh_dir,
        augment=False,
        gel_view_m=gel_view_m,
    )
    print(
        f"gs_blender(recon) augment(train)={augment} depth_scale={depth_scale} "
        f"use_gt_depth={gb.get('use_gt_depth', True)} mesh_dir={mesh_dir or gb['root'] + '/../meshes'}"
    )
    return train_ds, val_ds


def build_dataloaders(cfg: Any, distributed: bool = False) -> tuple[DataLoader, DataLoader]:
    data_cfg = cfg["data"]
    train_cfg = cfg["train"]
    task = cfg["model"]["task"]
    image_size = data_cfg["image_size"]

    if data_cfg["dataset"] == "synthetic":
        syn = data_cfg["synthetic"]
        train_ds = SyntheticPairDataset(
            syn["train_samples"],
            image_size=image_size,
            num_classes=data_cfg["num_classes"],
            regression_dim=data_cfg["regression_dim"],
            seed=0,
            task=task,
        )
        val_ds = SyntheticPairDataset(
            syn["val_samples"],
            image_size=image_size,
            num_classes=data_cfg["num_classes"],
            regression_dim=data_cfg["regression_dim"],
            seed=1,
            task=task,
        )
    elif data_cfg["dataset"] == "paired_folder":
        pf = data_cfg["paired_folder"]
        full_ds = PairedFolderDataset(
            root=pf["root"],
            rgb_dir=pf.get("rgb_dir", "rgb"),
            tactile_dir=pf.get("tactile_dir", "tactile"),
            labels_csv=pf.get("labels_csv", "labels.csv"),
            image_size=image_size,
            task=task,
        )
        n_val = max(1, int(0.2 * len(full_ds)))
        n_train = len(full_ds) - n_val
        train_ds, val_ds = random_split(full_ds, [n_train, n_val], generator=torch.Generator().manual_seed(cfg["seed"]))
    elif data_cfg["dataset"] == "gs_blender" and task == "reconstruction":
        train_ds, val_ds = _build_gs_blender_recon(cfg, image_size)
    elif data_cfg["dataset"] == "gs_blender":
        gb = data_cfg["gs_blender"]
        target_key = gb.get("target_key", "location")
        groups = scan_gs_blender(
            gb["root"],
            rgb_dir=gb.get("rgb_dir", "rgb"),
            tactile_dir=gb.get("tactile_dir", "samples"),
            raw_dir=gb.get("raw_dir", "raw_data"),
        )
        # Split by (object, session) group to avoid adjacent-frame leakage.
        keys = sorted(groups.keys())
        rng = random.Random(cfg["seed"])
        rng.shuffle(keys)
        n_val = max(1, int(gb.get("val_ratio", 0.2) * len(keys)))
        val_keys = set(keys[:n_val])
        train_samples = [s for k in keys[n_val:] for s in groups[k]]
        val_samples = [s for k in sorted(val_keys) for s in groups[k]]

        mean, std = compute_target_stats(
            train_samples, target_key, max_n=gb.get("stats_samples", 5000), seed=cfg["seed"]
        )
        cfg["_target_mean"] = mean.tolist()
        cfg["_target_std"] = std.tolist()
        print(
            f"gs_blender: {len(keys)} session-groups -> train_groups={len(keys) - n_val} "
            f"val_groups={n_val} | train={len(train_samples)} val={len(val_samples)}"
        )
        print(f"gs_blender target='{target_key}' mean={mean.tolist()} std={std.tolist()}")

        augment = gb.get("augment", True)
        aug_params = gb.get("tactile_aug_params", None)
        train_ds = GsBlenderPoseDataset(
            train_samples, image_size, target_key, mean, std, augment=augment, tactile_aug_params=aug_params
        )
        val_ds = GsBlenderPoseDataset(val_samples, image_size, target_key, mean, std, augment=False)
        print(f"gs_blender augment(train)={augment}")
    else:
        raise ValueError(f"Unknown dataset: {data_cfg['dataset']}")

    train_sampler = DistributedSampler(train_ds, shuffle=True) if distributed else None
    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=train_cfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg["batch_size"],
        shuffle=False,
        num_workers=train_cfg["num_workers"],
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader

