"""
ANEW GT applicability — raster AOA maps over project boundaries.  [TODO / STUB]

This is the deferred wall-to-wall companion to make_maps.py's plot-level point maps. Where
make_maps.py colours the *plots*, this fills each project's *full extent* with per-pixel DI
and an inside/outside-AOA mask, so coverage gaps between plots are visible.

The DI maths is already done — DISpace.di_fast (scripts/trust/di.py) computes raster-scale DI
via chunked BLAS. What is missing is the imagery: the AlphaEarth embedding raster sampled over
the user-supplied project boundaries, mapped into training-codec space.

Intended pipeline (needs: project boundary polygons + a GEE / Earth Engine session):
  1. For each project boundary, sample GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL bands A00..A63
     at 10 m for the relevant survey year (export to a local array / GeoTIFF stack).
  2. Apply the production codec affine emb_j = a_j * A_j + c_j, coefficients at
     experiments/agb_ireland_biomass_regression_20260608/preprocessing/aef_affine.parquet
     (cf. that experiment's refit_aef_affine_production.py) — RAW GEE float -> codec space.
  3. dsp = di.load_di_space("embonly")  (or rebuild via compute_di_folds' LOPO fit) and
     di = dsp.di_fast(X_codec_pixels) over the flattened raster; threshold at
     dsp.threshold_cast for the AOA mask.
  4. Write per-project DI + AOA GeoTIFFs (reuse the raster profile from step 1).

Until boundaries + a GEE extraction step are wired in, this raises NotImplementedError.

Run (once implemented):
    uv run --project /home/mattc/code/agb-ml-agent-evolve \
        python scripts/anew_gt_applicability/extract_aoa_rasters.py
"""

from __future__ import annotations


def main() -> None:
    raise NotImplementedError(
        "Raster AOA maps are a TODO: needs project boundary polygons and a GEE extraction "
        "of the AlphaEarth embedding raster. See the module docstring for the full pipeline; "
        "the DI maths is ready via di.DISpace.di_fast."
    )


if __name__ == "__main__":
    main()
