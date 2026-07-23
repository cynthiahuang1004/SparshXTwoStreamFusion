# CLAUDE.md — SparshXTwoStreamFusion (Sparsh-X Baseline)

> **Purpose**: Fair fusion-method baseline for comparison against **VisTacFusion**
> (`/media/hdd2/ihsuan/VisTacFusion`). Same encoder, same data, same task heads,
> same training recipe — only the **fusion mechanism** differs.

---

## 0. Relationship to VisTacFusion

This repo implements the **Sparsh-X style symmetric shared-bottleneck fusion** as a
controlled baseline. Everything except the fusion method is matched to VisTacFusion:

| Component | VisTacFusion | This repo (SparshX baseline) | Status |
|---|---|---|---|
| Encoder | Frozen DINOv3 ViT-L/16 | Frozen DINOv3 ViT-L/16 | **Same** |
| Fusion | Asymmetric cross-attn bottleneck | Symmetric shared-bottleneck (MBT) | **Different** (the comparison target) |
| DPT head | Reassemble + FeatureFusion | Reassemble + FeatureFusion | **Same architecture** |
| Pose head | PoseHead (CLS token) | PoseHead (mean pool) | **Different** (fusion consequence) |
| DPT tap source | Encoder multiscale @1024 dim | Post-fusion tactile stream @768 dim | **Different** (fusion consequence) |
| Dataset / GT | SimVisuoTactileDataset | GsBlenderDepthDataset | **Matched** |
| Augmentation | TactileAugment + RGB aug + gel-spin | TactilePhotometricAug + RGB aug + gel-spin | **Matched** |
| Loss | Grouped uncertainty (depth+normal / rot+trans) | Grouped uncertainty (depth+normal / rot+trans) | **Matched** |
| Training | AdamW, lr=2e-4, cosine, 150 epochs, seed=0 | AdamW, lr=2e-4, cosine, 150 epochs, seed=0 | **Matched** |
| Modality dropout | Yes (both/tactile/rgb) | No | **Different** (architecture doesn't support it) |

---

## 1. Architecture overview

```
RGB image      ──► Frozen DINOv3 ──► Linear(1024→768) + pos_emb + mod_emb ──► 4× independent self-attn blocks ──┐
                                                                                                                  ├──► 4× SharedBottleneckFusionLayer ──► DPT taps (tactile stream)
Tactile image  ──► Frozen DINOv3 ──► Linear(1024→768) + pos_emb + mod_emb ──► 4× independent self-attn blocks ──┘        │
                                                                                                                          ├──► DPT head ──► depth (B×1×224×224) + normal (B×3×224×224)
                                                                                                              mean(tactile_tokens) ──► PoseHead ──► SE(2) (cos,sin,tx,ty)
```

### SharedBottleneckFusionLayer (the key difference from VisTacFusion)

Each fusion layer:
1. Prepend shared bottleneck tokens to both RGB and tactile token sequences
2. Run stream-specific self-attention block on each (bottleneck + stream tokens)
3. **Average** the bottleneck portions from both streams → updated shared bottleneck
4. Strip bottleneck from each stream → updated RGB/tactile tokens

This is **symmetric** (both modalities contribute equally to bottleneck) vs VisTacFusion's
**asymmetric** approach (RGB → bottleneck → tactile queries, tactile is the anchor).

### Key architectural consequences of the fusion method

- **DPT taps**: come from post-fusion tactile stream at 768 dim (not encoder multiscale at 1024)
- **Pose readout**: mean pool of all 196 tactile tokens (no dedicated CLS/pose token)
- **Pre-fusion refinement**: 4 independent self-attn blocks per stream before fusion
  (VisTacFusion has no independent pre-fusion refinement)
- **No modality dropout**: model always requires both RGB and tactile

---

## 2. Shapes

- `B` = batch size. `D = 768` = fusion trunk dim. `E = 1024` = encoder dim.
- Image: `224×224`, patch 16 → 14×14 = **196 patch tokens**.
- Bottleneck: `16` tokens (configurable via `num_bottleneck_tokens`).
- Independent blocks: `4` layers per stream. Fusion blocks: `4` layers. Total depth: `8`.
- DPT taps: `4 × (B×196×768)` from tactile stream after fusion.
- Pose input: `(B×1×768)` mean-pooled tactile + `(B×196×768)` spatial pool.

---

## 3. Data

Uses the same sim data as VisTacFusion:
- **Sim root**: `/media/hdd2/ihsuan/gs_blender/renders`
- **Meshes**: `/media/hdd2/ihsuan/gs_blender/meshes`
- Train/val split: every 20th sample is val (`val_every: 20`)
- GT: depth from `_gt.npy`, normals computed from depth, pose SE(2) from per-sample
  `rotation_euler[2]` in pose json

---

## 4. Repo structure

```
SparshXTwoStreamFusion/
  CLAUDE.md                           # this file
  configs/
    gs_blender_recon.yaml             # main training config (reconstruction task)
    default.yaml, smoke.yaml, ...    # other task configs (not used for comparison)
  sparshx_fusion/                     # Python package
    models/
      model.py                        # SparshXTwoStreamFusionModel (main model)
      layers.py                       # SharedBottleneckFusionLayer, TransformerBlock
      encoders.py                     # DINOv3Encoder (same as VisTacFusion)
      heads/
        dpt.py                        # DPT decoder (same as VisTacFusion)
        pose.py                       # PoseHead (same as VisTacFusion)
    data/
      dataset.py                      # GsBlenderDepthDataset
      transforms.py                   # augmentation (matched to VisTacFusion)
      pose_gt.py                      # per-sample SE(2) GT computation
    losses/
      total.py                        # MultiTaskLoss with grouped uncertainty
      depth.py, normal.py, pose.py    # individual losses (same as VisTacFusion)
    engine/
      train.py                        # training loop + evaluation
      inference.py                    # single-image / batch inference
    utils/
      config.py, misc.py
  weights -> /media/hdd2/ihsuan/VisTacFusion/weights  # symlink to shared DINOv3 weights
```

---

## 5. Training

```bash
cd /media/hdd2/ihsuan/SparshXTwoStreamFusion

# Single GPU
CUDA_VISIBLE_DEVICES=0 python -m sparshx_fusion.engine.train \
  --config configs/gs_blender_recon.yaml

# Multi-GPU (DDP)
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 \
  -m sparshx_fusion.engine.train \
  --config configs/gs_blender_recon.yaml

# Resume
CUDA_VISIBLE_DEVICES=2,3 torchrun --nproc_per_node=2 \
  -m sparshx_fusion.engine.train \
  --config configs/gs_blender_recon.yaml \
  --resume outputs/gs_blender_recon/epoch_050.pt
```

Output goes to `outputs/gs_blender_recon/` (checkpoints, history.json, plots, tb).

---

## 6. Conda environment

Uses the same environment as VisTacFusion: `vistacfusion` (Python 3.11, PyTorch 2.5.1+cu121).

```bash
conda activate vistacfusion
pip install -e .   # install sparshx_fusion package
```

---

## 7. Comparison results

Run VisTacFusion and this baseline with the same data/config, then compare:

| Metric | VisTacFusion | SparshX baseline |
|---|---|---|
| depth_mse (both) | | |
| normal_mse (both) | | |
| pose_rot_deg (both) | | |
| pose_trans_l1 (both) | | |

Fill in after training completes.

---

## 8. Future ablation matrix

To disentangle encoder vs fusion contributions:

| | Sparsh encoder | DINOv3 encoder |
|---|---|---|
| **Sparsh fusion** (symmetric MBT) | Full Sparsh system | **This repo** |
| **VisTacFusion fusion** (asymmetric cross-attn) | Ablation (TODO) | **VisTacFusion** |
