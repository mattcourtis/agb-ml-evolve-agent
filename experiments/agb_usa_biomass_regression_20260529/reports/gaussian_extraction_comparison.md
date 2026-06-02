# Gaussian Extraction Comparison

**Experiment:** Gaussian-weighted vs point-centroid extraction for 7 co-features
(chm_m, topo_elevation, topo_slope, topo_aspect_cos, topo_aspect_sin, topo_tpi,
dist_years_since). AEF embeddings and coarse features (GEDI L4B, TerraClimate) unchanged.

**Kernel parameters:**
- 10 m features (CHM): σ=15 m, radius=45 m
- 30 m features (SRTM topo, Hansen): σ=25 m, radius=75 m

**Model:** LightGBM baseline (num_leaves=31, lr=0.05, min_child_samples=20, early_stopping=50)
**CV:** 23-project LOPO

## Results

| Config | R² | RMSE | Bias | Q1 | Q2 | Q3 | Q4 | Q5 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| point_centroid (iter3 baseline) | 0.4274 | 56.13 | +0.87 | +36.0 | +32.1 | +18.0 | -9.5 | -72.2 |
| gaussian_weighted (σ=15/25 m) | 0.4239 | 56.3 | +0.51 | +35.3 | +31.8 | +17.6 | -9.8 | -72.3 |

**Δ R²:**   -0.0035
**Δ RMSE:** +0.17 tCO₂/acre

## Feature value comparison (Gaussian vs point centroid)

| Feature | Point mean | Gauss mean | Mean |Δ| | % plots changed |
|---|---:|---:|---:|---:|
| chm_m             | 20.24 | 20.24 | 0.41 | 95.2% |
| topo_elevation    | 410.24 | 410.20 | 0.56 | 98.2% |
| topo_slope        | 5.84° | 5.82° | 0.75 | 98.7% |
| topo_aspect_cos   | 0.042 | 0.044 | 0.19 | 94.0% |
| topo_aspect_sin   | −0.023 | −0.019 | 0.17 | 93.9% |
| topo_tpi          | 0.87 | 0.83 | 0.56 | 98.3% |
| dist_years_since  | 84.8 | 85.2 | 5.71 | 38.1% |

Means are nearly identical; the differences are per-plot noise reductions,
not systematic biases. 95–99% of plots have a modified value for the spatial
features, confirming the Gaussian smoothing is active and effective.

## Interpretation

❌ Gaussian extraction degraded R²

R² change of -0.0035 is meaningful (+0.001 threshold).

The smoothing slightly degrades performance, possibly because averaging over neighbouring pixels introduces off-plot stand information that is not representative of the measured plot.
