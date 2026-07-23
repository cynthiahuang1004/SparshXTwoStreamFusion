"""Input augmentation (sim2real domain randomization), adapted from VisTacFusion.

- TactilePhotometricAug : heavy photometric domain randomization on the tactile image.
- rotate_gel_spin       : gel-spin rotation aug -- rotates tactile+rgb+depth by the same
  angle; GT theta shifts by MINUS the image angle.
- fixed_center_crop     : fixed 1/sqrt(2) center crop applied to EVERY sample.
- RGBPhotometricAug     : light photometric jitter on the RGB context image.
- to_tensor_imagenet    : float32 HWC (0-255) -> ImageNet-normalized CHW tensor.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

FIXED_CROP = 1.0 / math.sqrt(2.0)

DEFAULT_TACTILE_AUG = {
    "gain": 0.5,
    "bias": 45.0,
    "bright": 25.0,
    "grad": 0.7,
    "resid": 20.0,
    "noise": 6.0,
}


def fixed_center_crop(img, out_size=None):
    """Center-crop to FIXED_CROP of the side length, resize back to original (or out_size)."""
    H, W = img.shape[:2]
    side = int(math.floor(min(H, W) * FIXED_CROP))
    off_y = (H - side) // 2
    off_x = (W - side) // 2
    crop = img[off_y:off_y + side, off_x:off_x + side]
    out = out_size or (W, H)
    if isinstance(out, int):
        out = (out, out)
    return cv2.resize(crop, out, interpolation=cv2.INTER_LINEAR)


def rotate_gel_spin(tactile, rgb, depth, angle_deg):
    """Rotate tactile + rgb + depth by angle_deg around the image center.

    Simulates spinning the gel in place: image content rotates, theta changes
    by -angle_deg, (x, y) unchanged.
    """
    H, W = tactile.shape[:2]
    M = cv2.getRotationMatrix2D((W / 2, H / 2), angle_deg, 1.0)
    flags, border = cv2.INTER_LINEAR, cv2.BORDER_REFLECT_101
    tac = cv2.warpAffine(tactile, M, (W, H), flags=flags, borderMode=border)
    rgb_r = cv2.warpAffine(rgb, M, (W, H), flags=flags, borderMode=border)
    dep = cv2.warpAffine(depth, M, (W, H), flags=flags, borderMode=border)
    return tac, rgb_r, dep


class TactilePhotometricAug:
    """Heavy sim2real photometric randomization on a tactile image (float32 HWC, 0-255)."""

    def __init__(self, params: dict | None = None):
        self.p = {**DEFAULT_TACTILE_AUG, **(params or {})}

    def __call__(self, img: np.ndarray) -> np.ndarray:
        p = self.p
        h, w = img.shape[:2]
        if p["gain"] > 0:
            g = np.random.uniform(1 - p["gain"], 1 + p["gain"], size=(1, 1, 3)).astype(np.float32)
            img = img * g
        if p["bias"] > 0:
            img = img + np.random.uniform(-p["bias"], p["bias"], size=(1, 1, 3)).astype(np.float32)
        if p["bright"] > 0:
            img = img + np.float32(np.random.uniform(-p["bright"], p["bright"]))
        if p["grad"] > 0:
            angle = np.random.uniform(0, 2 * np.pi)
            ys = np.linspace(-1, 1, h, dtype=np.float32).reshape(-1, 1)
            xs = np.linspace(-1, 1, w, dtype=np.float32).reshape(1, -1)
            grad_map = np.float32(np.cos(angle)) * xs + np.float32(np.sin(angle)) * ys
            amp = np.random.uniform(0, p["grad"], size=(1, 1, 3)).astype(np.float32)
            img = img + grad_map[..., None] * amp * np.float32(50.0)
        if p["resid"] > 0:
            raw = np.random.randn(16, 16, 3).astype(np.float32)
            smooth = cv2.resize(raw, (w, h), interpolation=cv2.INTER_LINEAR)
            smooth = cv2.GaussianBlur(smooth, (0, 0), sigmaX=h / 8.0)
            std = np.float32(smooth.std())
            if std > 1e-6:
                smooth = smooth / std * np.float32(p["resid"])
            img = img + smooth
        if p["noise"] > 0:
            img = img + np.random.normal(0, p["noise"], img.shape).astype(np.float32)
        return img


class RGBPhotometricAug:
    """Light photometric jitter on the RGB context image (float32 HWC, 0-255).
    All gain/bias/brightness are UNIFORM across channels to preserve marker hue."""

    def __init__(self, gain: float = 0.3, bias: float = 20.0, bright: float = 15.0,
                 grad: float = 0.4, noise: float = 4.0):
        self.gain = gain
        self.bias = bias
        self.bright = bright
        self.grad = grad
        self.noise = noise

    def __call__(self, img: np.ndarray) -> np.ndarray:
        H, W = img.shape[:2]
        if self.gain > 0:
            g = np.float32(np.random.uniform(1 - self.gain, 1 + self.gain))
            img = img * g
        if self.bias > 0:
            b = np.float32(np.random.uniform(-self.bias, self.bias))
            img = img + b
        if self.bright > 0:
            img = img + np.float32(np.random.uniform(-self.bright, self.bright))
        if self.grad > 0:
            angle = np.random.uniform(0, 2 * np.pi)
            ys = np.linspace(-1, 1, H, dtype=np.float32).reshape(-1, 1)
            xs = np.linspace(-1, 1, W, dtype=np.float32).reshape(1, -1)
            grad_map = np.float32(np.cos(angle)) * xs + np.float32(np.sin(angle)) * ys
            img = img + grad_map[..., None] * np.float32(self.grad * 30.0)
        if self.noise > 0:
            img = img + np.random.normal(0, self.noise, img.shape).astype(np.float32)
        return img


def to_tensor_imagenet(img: np.ndarray, out_hw: tuple[int, int]) -> torch.Tensor:
    """float32 HWC (0-255) -> ImageNet-normalized CHW tensor, bilinear-resized to out_hw."""
    t = torch.from_numpy(np.ascontiguousarray(img)).float().permute(2, 0, 1) / 255.0
    out_hw = (int(out_hw[0]), int(out_hw[1]))
    if (t.shape[1], t.shape[2]) != out_hw:
        t = F.interpolate(t.unsqueeze(0), size=out_hw, mode="bilinear", align_corners=False).squeeze(0)
    mean = torch.tensor(IMAGENET_MEAN).view(-1, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(-1, 1, 1)
    return (t - mean) / std
