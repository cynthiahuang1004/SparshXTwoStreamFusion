"""Depth-map inference for the trained 3D-reconstruction model.

Loads a `best.pt` checkpoint (trainable weights only -- the frozen DINOv3 encoder is
rebuilt from the path stored in the checkpoint's config), runs a forward pass on a paired
RGB + tactile image, and writes the predicted depth map (in the original raw units, i.e.
un-scaled by the training `depth_scale`).

NOTE: the model is a two-stream RGB+tactile fusion net. Depth is decoded from the tactile
stream, but the tactile tokens are influenced by the RGB stream through the shared
bottleneck, and this checkpoint was trained WITHOUT modality dropout (always paired inputs).
So for in-distribution results pass the paired RGB (object-appearance) image via --rgb.
If --rgb is omitted a zero image is used (out-of-distribution; quality will degrade).

Examples:
    python -m sparshx_fusion.engine.inference \
        --tactile path/to/samples/0000.png --rgb path/to/rgb/0000.png --out pred_0000

    # depth only (npy + normalized png) with the default best checkpoint:
    python -m sparshx_fusion.engine.inference --tactile 0000.png --rgb 0000.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from sparshx_fusion.data.transforms import to_tensor_imagenet
from sparshx_fusion.models import SparshXTwoStreamFusionModel
from sparshx_fusion.utils.misc import resolve_device


def load_model(checkpoint: str, device: torch.device) -> tuple[SparshXTwoStreamFusionModel, dict]:
    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    model = SparshXTwoStreamFusionModel(**cfg["model"])   # builds + loads the frozen encoder
    # The checkpoint stores only trainable weights; the encoder is already loaded above, so
    # its keys legitimately show up as "missing". Anything else missing is a real problem.
    missing, unexpected = model.load_state_dict(ckpt["model"], strict=False)
    real_missing = [k for k in missing if not k.startswith(("rgb_encoder.", "tactile_encoder."))]
    if real_missing:
        raise RuntimeError(f"Missing non-encoder weights in checkpoint: {real_missing[:8]} ...")
    if unexpected:
        raise RuntimeError(f"Unexpected weights in checkpoint: {unexpected[:8]} ...")
    model.eval().to(device)
    print(f"[inference] loaded {checkpoint} (epoch {ckpt.get('epoch')}), task={cfg['model']['task']}")
    return model, cfg


def load_image(path: str, image_size: int) -> torch.Tensor:
    arr = np.array(Image.open(path).convert("RGB"), dtype=np.float32)   # HWC, 0-255
    return to_tensor_imagenet(arr, (image_size, image_size))            # [3, H, W]


def save_depth_png(depth: np.ndarray, path: str) -> None:
    """Normalize a depth map to 0-255 (over its positive/contact values) and save as PNG."""
    d = depth.astype(np.float32)
    pos = d[d > 0]
    if pos.size > 0:
        lo, hi = float(pos.min()), float(pos.max())
        norm = np.clip((d - lo) / max(hi - lo, 1e-8), 0, 1)
        norm[d <= 0] = 0.0
    else:
        norm = np.zeros_like(d)
    Image.fromarray((norm * 255).astype(np.uint8)).save(path)


@torch.no_grad()
def predict_depth(model, cfg, tactile_path, rgb_path, device):
    image_size = cfg["model"]["image_size"]
    depth_scale = cfg.get("data", {}).get("gs_blender", {}).get("depth_scale", 1000.0)

    tactile = load_image(tactile_path, image_size)
    if rgb_path is not None:
        rgb = load_image(rgb_path, image_size)
    else:
        print("[inference] WARNING: no --rgb given; using a zero RGB image (out-of-distribution).")
        rgb = torch.zeros_like(tactile)

    out = model(rgb.unsqueeze(0).to(device), tactile.unsqueeze(0).to(device))
    depth = out.depth[0, 0].float().cpu().numpy() / depth_scale   # back to original raw units
    return depth


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default="outputs/gs_blender_recon/best.pt")
    ap.add_argument("--tactile", required=True, help="tactile image path")
    ap.add_argument("--rgb", default=None, help="paired RGB (object appearance) image path")
    ap.add_argument("--out", default="depth_pred", help="output path stem (.npy + .png)")
    ap.add_argument("--device", default="auto")
    args = ap.parse_args()

    device = resolve_device(args.device)
    model, cfg = load_model(args.checkpoint, device)
    depth = predict_depth(model, cfg, args.tactile, args.rgb, device)

    out_stem = Path(args.out)
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    np.save(out_stem.with_suffix(".npy"), depth)
    save_depth_png(depth, str(out_stem.with_suffix(".png")))
    nz = depth[depth > 0]
    print(f"[inference] depth shape={depth.shape} contact_px={int((depth > 0).sum())} "
          f"depth_range=({nz.min() if nz.size else 0:.6f}, {depth.max():.6f})")
    print(f"[inference] saved {out_stem.with_suffix('.npy')} and {out_stem.with_suffix('.png')}")


if __name__ == "__main__":
    main()
