"""Training loop with modality dropout + multi-GPU DDP support.

Single GPU:
    python -m sparshx_fusion.engine.train --config configs/gs_blender_recon.yaml

Multi-GPU (e.g. 2 GPUs):
    torchrun --nproc_per_node=2 -m sparshx_fusion.engine.train \
           --config configs/gs_blender_recon.yaml
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler

from sparshx_fusion.data import build_dataloaders
from sparshx_fusion.losses import MultiTaskLoss
from sparshx_fusion.models import SparshXTwoStreamFusionModel
from sparshx_fusion.utils.config import load_config, update_epochs
from sparshx_fusion.utils.misc import resolve_device, set_seed


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def is_main_process():
    return not is_distributed() or dist.get_rank() == 0


def setup_distributed():
    if "RANK" not in os.environ:
        return None, 0, 1
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = dist.get_world_size()
    torch.cuda.set_device(local_rank)
    return torch.device(f"cuda:{local_rank}"), rank, world_size


def sample_config(md_cfg: dict) -> str:
    if not md_cfg.get("enabled", False):
        return "both"
    r = random.random()
    p_both = md_cfg.get("p_both", 0.5)
    p_tac = md_cfg.get("p_tactile_only", 0.25)
    if r < p_both:
        return "both"
    if r < p_both + p_tac:
        return "tactile"
    return "rgb"


def sync_config(config: str, device) -> str:
    if not is_distributed():
        return config
    mapping = {"both": 0, "tactile": 1, "rgb": 2}
    reverse = {0: "both", 1: "tactile", 2: "rgb"}
    t = torch.tensor([mapping[config]], device=device)
    dist.broadcast(t, src=0)
    return reverse[t.item()]


def build_model_from_cfg(cfg: dict) -> SparshXTwoStreamFusionModel:
    mcfg = cfg["model"]
    return SparshXTwoStreamFusionModel(
        image_size=mcfg.get("image_size", 224),
        trunk_dim=mcfg.get("trunk_dim", 768),
        encoder=mcfg.get("encoder"),
        rgb_encoder=mcfg.get("rgb_encoder"),
        fusion_trunk=mcfg.get("fusion_trunk"),
        dpt=mcfg.get("dpt"),
        pose=mcfg.get("pose"),
        inject_gate_init=mcfg.get("inject_gate_init", 0.0),
    )


def build_multitask_criterion(cfg: dict) -> MultiTaskLoss | None:
    if cfg["model"]["task"] != "reconstruction":
        return None
    pose_cfg = cfg["model"].get("pose", {})
    return MultiTaskLoss(
        cfg.get("loss", {}),
        pose_mode=pose_cfg.get("pose_mode", "regression"),
        rot_num_bins=pose_cfg.get("rot_num_bins", 72),
    )


def _model_pred(out):
    if isinstance(out, dict):
        return {"depth": out["depth"], "normal": out["normal"], "se2": out["se2"]}
    return {"depth": out.depth, "normal": out.normal, "se2": out.se2}


def compute_loss(out, batch, task: str, criterion=None) -> torch.Tensor:
    if task == "classification":
        return F.cross_entropy(out.logits, batch["label"])
    if task == "regression":
        return F.smooth_l1_loss(out.regression, batch["target"])
    if task == "embedding":
        return out.fused_embedding.pow(2).mean()
    if task == "reconstruction":
        loss, _ = criterion(_model_pred(out), batch)
        return loss
    raise ValueError(f"Unknown task: {task}")


@torch.no_grad()
def _depth_metrics(pred, gt, mask, eps: float = 1e-6):
    if mask is None:
        mask = (gt.abs() > eps).float()
    valid = mask.sum().clamp_min(1)
    absrel = ((pred - gt).abs() * mask / gt.abs().clamp_min(eps)).sum() / valid
    mse = ((pred - gt) ** 2 * mask).sum() / valid
    rmse = torch.sqrt(mse)
    return absrel.item(), rmse.item(), mse.item()


@torch.no_grad()
def _normal_angle_deg(pred, gt, mask=None, eps: float = 1e-6):
    p = F.normalize(pred, dim=1, eps=eps)
    g = F.normalize(gt, dim=1, eps=eps)
    cos = (p * g).sum(dim=1, keepdim=True).clamp(-1 + 1e-6, 1 - 1e-6)
    angle = torch.acos(cos)
    if mask is not None:
        valid = mask.sum().clamp_min(1)
        return (angle * mask).sum().item() / valid.item() * 180.0 / math.pi
    return angle.mean().item() * 180.0 / math.pi


@torch.no_grad()
def _pose_rot_deg(pred_se2, gt):
    cosp, sinp = pred_se2[:, 0], pred_se2[:, 1]
    cosg, sing = gt[:, 0], gt[:, 1]
    dcos = (cosp * cosg + sinp * sing).clamp(-1 + 1e-6, 1 - 1e-6)
    return (torch.acos(dcos).mean().item() * 180.0 / math.pi)


def build_scheduler(optimizer, warmup_steps, total_steps, min_lr_ratio=0.01):
    """Cosine decay with linear warmup."""
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def train_one_epoch(model, loader, optimizer, scaler, device, task: str, amp: bool, log_every: int,
                    epoch: int, criterion=None, scheduler=None, md_cfg=None):
    model.train()
    md_cfg = md_cfg or {}
    running = {"total": 0.0}
    for step, batch in enumerate(loader):
        batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                 for k, v in batch.items()}

        config = sample_config(md_cfg)
        config = sync_config(config, device)

        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            out = model(batch["rgb"], batch["tactile"], config=config)
            if task == "reconstruction":
                loss, comps = criterion(_model_pred(out), batch)
            else:
                loss = compute_loss(out, batch, task)
                comps = {"total": loss.detach()}
        if not torch.isfinite(loss):
            if is_main_process():
                print(f"  [WARN] non-finite loss at epoch={epoch} step={step}, skipping")
            loss = loss * 0.0
            scaler.scale(loss).backward()
            optimizer.zero_grad(set_to_none=True)
            if scheduler is not None:
                scheduler.step()
            continue
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        if scheduler is not None:
            scheduler.step()
        running["total"] += loss.item()
        for k, v in comps.items():
            running[k] = running.get(k, 0.0) + float(v)
        if is_main_process() and step % log_every == 0:
            msg = f"epoch={epoch:03d} step={step:04d}/{len(loader)} cfg={config:7s} loss={loss.item():.4f}"
            if task == "reconstruction":
                parts = " ".join(f"{k}={comps[k].item():.4f}" for k in sorted(comps) if k != "total")
                if parts:
                    msg += f" ({parts})"
            print(msg)
    n = max(1, len(loader))
    return {k: v / n for k, v in running.items()}


@torch.no_grad()
def _evaluate_one_config(raw_model, loader, device, task, criterion, config="both"):
    running = {"loss": 0.0}
    correct = count = 0
    abs_err_sum = None
    n_samples = 0
    absrel_sum = rmse_sum = mse_sum = 0.0
    normal_angle_sum = 0.0
    pose_rot_sum = 0.0
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) if torch.is_tensor(v) else v
                 for k, v in batch.items()}
        out = raw_model(batch["rgb"], batch["tactile"], config=config)
        if task == "reconstruction":
            loss, comps = criterion(_model_pred(out), batch)
            running["loss"] += loss.item()
            for k, v in comps.items():
                running[k] = running.get(k, 0.0) + float(v)
            absrel, rmse, mse = _depth_metrics(out.depth, batch["depth"], batch.get("mask"))
            absrel_sum += absrel
            rmse_sum += rmse
            mse_sum += mse
            normal_angle_sum += _normal_angle_deg(out.normal, batch["normal"], batch.get("mask"))
            pose_rot_sum += _pose_rot_deg(out.se2, batch["pose"])
        else:
            loss = compute_loss(out, batch, task)
            running["loss"] += loss.item()
        if task == "classification":
            pred = out.logits.argmax(dim=-1)
            correct += (pred == batch["label"]).sum().item()
            count += batch["label"].numel()
        elif task == "regression":
            abs_err = (out.regression - batch["target"]).abs().sum(dim=0)
            abs_err_sum = abs_err if abs_err_sum is None else abs_err_sum + abs_err
            n_samples += batch["target"].shape[0]

    n = max(1, len(loader))
    metrics = {"loss": running["loss"] / n}
    if task == "classification":
        metrics["accuracy"] = correct / max(1, count)
    elif task == "reconstruction":
        metrics["depth_absrel"] = round(absrel_sum / n, 5)
        metrics["depth_rmse"] = round(rmse_sum / n, 5)
        metrics["depth_mse"] = round(mse_sum / n, 6)
        metrics["normal_mean_angle"] = round(normal_angle_sum / n, 3)
        metrics["pose_rot_deg"] = round(pose_rot_sum / n, 3)
        for k in ("depth", "normal", "pose_rot", "pose_trans", "total"):
            if k in running:
                metrics[f"loss_{k}"] = round(running[k] / n, 5)
    elif task == "regression" and abs_err_sum is not None:
        mae_norm = abs_err_sum / max(1, n_samples)
        metrics["mae_norm"] = round(mae_norm.mean().item(), 5)
    return metrics


@torch.no_grad()
def evaluate(model, loader, device, task: str, target_std=None, criterion=None):
    model.eval()
    raw_model = model.module if isinstance(model, DDP) else model
    if task != "reconstruction":
        return _evaluate_one_config(raw_model, loader, device, task, criterion)
    results = {}
    for config in ("both", "tactile", "rgb"):
        results[config] = _evaluate_one_config(raw_model, loader, device, task, criterion, config)
    return results


def plot_curves(history: list[dict], output_dir: Path) -> None:
    """Save loss-curve PNGs from accumulated epoch history, per-config."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(exist_ok=True)

    # --- 1. Combined train/val loss ---
    train_loss = [h["train_loss"] for h in history]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, train_loss, label="train loss")
    configs = ("both", "tactile", "rgb")
    for cfg_name in configs:
        vals = []
        for h in history:
            v = h.get("val", {})
            if isinstance(v, dict) and cfg_name in v:
                vals.append(v[cfg_name].get("loss", None))
            elif cfg_name == "both":
                vals.append(h.get("val_loss"))
            else:
                vals.append(None)
        if vals[0] is not None:
            ax.plot(epochs, vals, label=f"val loss ({cfg_name})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Train / Val Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(plots_dir / "loss_curves.png", dpi=150)
    plt.close(fig)

    # --- 2. Per-config metric plots (reconstruction only) ---
    metric_keys = {
        "depth_mse": ("Depth MSE", "Depth MSE"),
        "normal_mean_angle": ("Normal Mean Angle (deg)", "Normal Mean Angle"),
        "pose_rot_deg": ("Pose Rotation Error (deg)", "Pose Rotation Error"),
    }
    for key, (ylabel, title) in metric_keys.items():
        fig, ax = plt.subplots(figsize=(8, 5))
        has_data = False
        for cfg_name in configs:
            vals = []
            for h in history:
                v = h.get("val", {})
                if isinstance(v, dict) and cfg_name in v:
                    vals.append(v[cfg_name].get(key))
                elif cfg_name == "both":
                    vals.append(h.get(key))
                else:
                    vals.append(None)
            if vals[0] is not None:
                ax.plot(epochs, vals, marker="o", markersize=2, label=cfg_name)
                has_data = True
        if not has_data:
            plt.close(fig)
            continue
        ax.set_xlabel("Epoch")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(plots_dir / f"{key}.png", dpi=150)
        plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()

    ddp_device, rank, world_size = setup_distributed()

    cfg = update_epochs(load_config(args.config), args.epochs)
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir

    set_seed(cfg["seed"] + rank)

    if ddp_device is not None:
        device = ddp_device
    else:
        device = resolve_device(cfg["device"])

    if is_main_process():
        print(f"Device: {device} | World size: {world_size}")

    train_loader, val_loader = build_dataloaders(cfg, distributed=is_distributed())
    target_std = cfg.get("_target_std")
    model = build_model_from_cfg(cfg).to(device)
    task = cfg["model"]["task"]

    if is_distributed():
        model = DDP(model, device_ids=[device.index], find_unused_parameters=True)

    criterion = build_multitask_criterion(cfg)
    if criterion is not None:
        criterion = criterion.to(device)

    raw_model = model.module if isinstance(model, DDP) else model
    params = list(p for p in raw_model.parameters() if p.requires_grad)
    if criterion is not None:
        params += list(p for p in criterion.parameters() if p.requires_grad)
    optimizer = torch.optim.AdamW(
        params,
        lr=cfg["train"]["lr"],
        weight_decay=cfg["train"]["weight_decay"],
    )
    scaler = torch.amp.GradScaler("cuda", enabled=cfg["train"]["amp"] and device.type == "cuda")

    train_cfg = cfg["train"]
    total_steps = len(train_loader) * train_cfg["epochs"]
    warmup_steps = train_cfg.get("warmup_steps", 1000)
    scheduler = build_scheduler(optimizer, warmup_steps, total_steps,
                                min_lr_ratio=train_cfg.get("min_lr_ratio", 0.01))

    from datetime import datetime
    base_output_dir = cfg["output_dir"]
    date_prefix = datetime.now().strftime("%Y%m%d")
    if not Path(base_output_dir).name.startswith(date_prefix):
        base_output_dir = str(Path(base_output_dir).parent / f"{date_prefix}_SparshX")
    output_dir = Path(base_output_dir)
    if is_main_process():
        output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Resume from checkpoint ----
    start_epoch = 0
    best_depth = float("inf")
    best_pose = float("inf")
    history: list[dict] = []
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device, weights_only=False)
        raw_model.load_state_dict(ckpt["model"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])
        scaler.load_state_dict(ckpt["scaler"])
        if criterion is not None and "criterion" in ckpt:
            criterion.load_state_dict(ckpt["criterion"])
        start_epoch = ckpt["epoch"] + 1
        history_path = output_dir / "history.json"
        if history_path.exists():
            with open(history_path) as f:
                history = json.load(f)
            history = [h for h in history if h["epoch"] < start_epoch]
        if is_main_process():
            print(f"Resumed from {args.resume}, starting at epoch {start_epoch} (history: {len(history)} entries)")

    md_cfg = cfg.get("modality_dropout", {})
    for epoch in range(start_epoch, train_cfg["epochs"]):
        if is_distributed():
            train_loader.sampler.set_epoch(epoch)

        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scaler,
            device,
            task=task,
            amp=train_cfg["amp"],
            log_every=train_cfg["log_every"],
            epoch=epoch,
            criterion=criterion,
            scheduler=scheduler,
            md_cfg=md_cfg,
        )

        if is_main_process():
            val_result = evaluate(model, val_loader, device, task, target_std=target_std, criterion=criterion)
            train_loss = train_metrics.get("total", train_metrics.get("loss", 0.0))
            print(f"epoch={epoch:03d} train_loss={train_loss:.4f} val={val_result}")

            def _save_ckpt(path):
                trainable_state = {
                    k: v
                    for k, v in raw_model.state_dict().items()
                    if not k.startswith(("rgb_encoder.", "tactile_encoder."))
                }
                save_obj = {
                    "model": trainable_state,
                    "cfg": cfg,
                    "epoch": epoch,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "scaler": scaler.state_dict(),
                }
                if criterion is not None:
                    save_obj["criterion"] = criterion.state_dict()
                torch.save(save_obj, path)

            if task == "reconstruction":
                both_metrics = val_result.get("both", val_result)
                depth_score = both_metrics.get("depth_mse", float("inf"))
                pose_score = both_metrics.get("pose_rot_deg", float("inf"))
                if depth_score < best_depth:
                    best_depth = depth_score
                    _save_ckpt(output_dir / "best_depth.pt")
                    print(f"  ** new best depth: mse={best_depth:.6f}")
                if pose_score < best_pose:
                    best_pose = pose_score
                    _save_ckpt(output_dir / "best_pose.pt")
                    print(f"  ** new best pose: rot_deg={best_pose:.3f}")
            else:
                val_loss = val_result["loss"]
                if val_loss < best_depth:
                    best_depth = val_loss
                    _save_ckpt(output_dir / "best.pt")
                    print(f"saved best checkpoint to {output_dir / 'best.pt'}")

            # --- Record history and plot ---
            record = {
                "epoch": epoch,
                "train_loss": round(train_loss, 6),
                "val": val_result,
            }
            if task == "reconstruction" and isinstance(val_result, dict) and "both" in val_result:
                both_m = val_result["both"]
                record["val_loss"] = round(both_m.get("loss", 0), 6)
                for k in ("depth_mse", "normal_mean_angle", "pose_rot_deg"):
                    if k in both_m:
                        record[k] = both_m[k]
            history.append(record)
            with open(output_dir / "history.json", "w") as f:
                json.dump(history, f, indent=2)
            try:
                plot_curves(history, output_dir)
            except Exception as e:
                print(f"  [WARN] plot_curves failed: {e}")

        if is_distributed():
            dist.barrier()

    if is_main_process():
        print("Training complete.")
    if is_distributed():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
