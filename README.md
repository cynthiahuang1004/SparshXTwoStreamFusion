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

