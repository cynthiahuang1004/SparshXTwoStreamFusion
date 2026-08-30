"""Train the SparshX MBT-fusion baseline with VisTacFusion's training pipeline.

Thin wrapper around ``vistacfusion.engine.train`` so that data processing, losses,
modality dropout, optimizer/schedule, evaluation and checkpoint selection are exactly the
VisTacFusion code (only the model differs, see ``sparshx_fusion/models/vtf_model.py``).

Usage (from this repo, env ``vistacfusion``):
    CUDA_VISIBLE_DEVICES=3 python -m sparshx_fusion.engine.train \\
        --model configs/vtf_model_sparshx_t3mae.yaml \\
        --train <VTF>/configs/train_bs32.yaml \\
        --data  <VTF>/ablation/simqty_gtac/data_ratio_g3s_sim348_transfilt_zoom115_crop816.yaml \\
        [--output-dir outputs/<name>]   # default: outputs/YYYYMMDD_SparshX
or with a bundle yaml holding those three paths:
    python -m sparshx_fusion.engine.train --config configs/run_sim348_crop816.yaml
Relative --train/--data paths are resolved against the VisTacFusion repo root, relative
--model / --output-dir paths against this repo. Extra args are passed through unchanged
(e.g. --resume, --finetune). The legacy standalone trainer is kept as train_legacy.py.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime

import yaml

VTF_ROOT = os.environ.get("VTF_ROOT", "/media/hdd2/ihsuan/VisTacFusion")
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _resolve(path, root):
    return path if os.path.isabs(path) else os.path.abspath(os.path.join(root, path))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="bundle yaml with keys model / train / data [/ output_dir]")
    ap.add_argument("--model", default=None)
    ap.add_argument("--train", default=None)
    ap.add_argument("--data", default=None)
    ap.add_argument("--output-dir", default=None)
    args, passthrough = ap.parse_known_args()

    bundle = {}
    if args.config:
        with open(_resolve(args.config, REPO_ROOT)) as f:
            bundle = yaml.safe_load(f) or {}
    model = args.model or bundle.get("model")
    train = args.train or bundle.get("train", "configs/train_bs32.yaml")
    data = args.data or bundle.get("data")
    out = args.output_dir or bundle.get("output_dir") or f"outputs/{datetime.now():%Y%m%d}_SparshX"
    if not model or not data:
        ap.error("--model and --data are required (directly or via --config)")

    model, train, data, out = (_resolve(model, REPO_ROOT), _resolve(train, VTF_ROOT),
                               _resolve(data, VTF_ROOT), _resolve(out, REPO_ROOT))
    if REPO_ROOT not in sys.path:
        sys.path.insert(0, REPO_ROOT)
    if VTF_ROOT not in sys.path:
        sys.path.insert(0, VTF_ROOT)
    os.chdir(VTF_ROOT)   # VTF configs use repo-relative paths (pretrained_encoders/, ablation/)

    from vistacfusion.engine import train as vtf_train
    sys.argv = [sys.argv[0], "--model", model, "--train", train, "--data", data,
                "--output-dir", out] + passthrough
    print(f"[sparshx] VisTacFusion trainer @ {VTF_ROOT}\n  model={model}\n  train={train}\n  data={data}\n  out={out}", flush=True)
    vtf_train.main()


if __name__ == "__main__":
    main()
