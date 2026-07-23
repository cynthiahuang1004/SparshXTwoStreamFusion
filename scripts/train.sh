#!/usr/bin/env bash
set -euo pipefail

# Single GPU:
#   bash scripts/train.sh --config configs/gs_blender_recon.yaml
#
# Multi-GPU (e.g. 2 GPUs on GPU 2,3):
#   CUDA_VISIBLE_DEVICES=2,3 bash scripts/train.sh --nproc 2 --config configs/gs_blender_recon.yaml

NPROC=1
ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --nproc) NPROC="$2"; shift 2 ;;
        *) ARGS+=("$1"); shift ;;
    esac
done

if [[ "$NPROC" -gt 1 ]]; then
    torchrun --nproc_per_node="$NPROC" -m sparshx_fusion.engine.train "${ARGS[@]}"
else
    python -m sparshx_fusion.engine.train "${ARGS[@]}"
fi
