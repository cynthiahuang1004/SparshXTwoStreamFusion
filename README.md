# SparshXTwoStreamFusion

Minimal experimental framework for a symmetric two-stream visuo-tactile transformer with shared bottleneck fusion.

The intended inputs are two same-sized RGB-like images:

- `rgb`: external object image from the tactile sensor's internal camera
- `tactile`: contact tactile image

The model does not assume pixel alignment between the two images. Each stream is patch-embedded and processed independently first, then the streams exchange information through shared bottleneck tokens.

## Architecture

```text
rgb image      -> RGB PatchEmbed     -> RGB tokens
tactile image  -> Tactile PatchEmbed -> tactile tokens

RGB tokens      -> independent transformer blocks
tactile tokens  -> independent transformer blocks

RGB tokens + shared bottleneck      -> fusion transformer block
tactile tokens + shared bottleneck  -> fusion transformer block
updated bottlenecks are averaged across modalities

pooled RGB + pooled tactile + pooled bottleneck -> task head
```

Compared with asymmetric tactile-anchor fusion, this model treats RGB and tactile as peer modalities. Both streams keep their own tokens and both contribute to the shared bottleneck.

## Quick Start

Run a synthetic smoke training job:

```bash
cd SparshXTwoStreamFusion
python -m sparshx_fusion.engine.train --config configs/default.yaml --epochs 1
```

Run shape tests:

```bash
python tests/test_shapes.py
```

## Real Data Layout

The folder dataset expects paired files with matching relative names:

```text
data_root/
  rgb/
    sample_000.png
    sample_001.png
  tactile/
    sample_000.png
    sample_001.png
  labels.csv          # optional
```

`labels.csv` format:

```csv
name,label
sample_000.png,0
sample_001.png,1
```

For regression, use columns `target_0`, `target_1`, etc. The dataset code is intentionally small so it can be adapted to your exact MuxGel/real capture naming convention.

## Aligned training with VisTacFusion (current workflow)

`sparshx_fusion/engine/train.py` is now a thin wrapper around **VisTacFusion's** trainer
(`vistacfusion.engine.train`, `model_type: external`). Data processing (fixed crop, sim RGB
zoom, translation filter, rotation alignment, gel-spin augmentation with camera-frame xy
labels, sim oversampling), losses, modality dropout, optimizer/schedule, evaluation and
checkpoint selection are therefore *the same code* as the VisTacFusion runs this baseline is
compared against. The model (`sparshx_fusion/models/vtf_model.py`) is built from VisTacFusion
components (frozen encoders, projection, DPT/Pose heads, object embedding, TapInjection);
the only difference is the symmetric shared-bottleneck **MBT trunk** (`MBTFusionTrunk`).

```bash
cd /media/hdd2/ihsuan/SparshXTwoStreamFusion && conda activate vistacfusion
# bundle = model config here + train/data configs taken from the VisTacFusion repo
CUDA_VISIBLE_DEVICES=3 python -m sparshx_fusion.engine.train --config configs/run_sim348_crop816.yaml
# or explicitly (relative --train/--data resolve against $VTF_ROOT = /media/hdd2/ihsuan/VisTacFusion)
python -m sparshx_fusion.engine.train --model configs/vtf_model_sparshx_t3mae.yaml \
    --train configs/train_bs32.yaml \
    --data ablation/simqty_gtac/data_ratio_g3s_sim348_transfilt_zoom115_crop816.yaml \
    --output-dir outputs/base_sparshx_c816_sim348
```
Outputs (history.json, best_depth.pt, best_pose.pt, latest.pt) have the VisTacFusion layout, so
`VisTacFusion/scripts/plot_ratio_ladder.py::load_best_per_mode` and
`python -m vistacfusion.engine.inference --model configs/vtf_model_sparshx_t3mae.yaml ...`
work unchanged. Large outputs live on `/media/hdd/ihsuan/SparshX_outputs/` (symlinked from `outputs/`).
The previous standalone trainer is kept as `sparshx_fusion/engine/train_legacy.py`.
