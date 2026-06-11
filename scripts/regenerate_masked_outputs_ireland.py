"""
Regenerate Ireland AGB outputs with the Dynamic World forest mask APPLIED.

Reads the per-stand DW mask checkpoints written by scripts/apply_forest_mask_ireland.py
(preprocessing/_dw_mask_y{2022,2023,2024,survey}/<Location>.parquet, columns lon/lat/trees/
forest/pred_masked_tco2_acre) and:

  1. Re-aggregates to stand level: masked_density = mean over ALL pixels (non-forest = 0)
     = forest_fraction * mean(forest preds); forest_fraction = mean(trees >= 0.5).
  2. ADDS masked columns to the existing outputs (NEVER overwrites the unmasked numbers):
       final/ireland_agb_yearmatched.{parquet,csv,gpkg}:
         our_{2022,2023,2024}_masked_tCO2_acre (+_Mg_ha),
         our_mean_2022_24_masked_tCO2_acre (+_Mg_ha), forest_frac_{2022,2023,2024}
       final/ireland_agb_pixel.{parquet,csv,gpkg}:
         pred_pixel_masked_tCO2_acre (+_Mg_ha), pred_pixel_masked_total_t, forest_fraction
  3. Writes masked GeoTIFFs (non-forest pixels = 0 in band1 tCO2/acre + band2 Mg/ha, EPSG:2157,
     10 m, nodata -9999 outside polygon) under
       final/ireland_pixel_tiffs_{2022,2023,2024}_masked/<Location>.tif (+ per-year VRT)
       final/ireland_pixel_tiffs_masked/<Location>.tif (survey-year) (+ VRT)
  4. Refreshes the dual-scale comparison figure
       final/figures/ireland_vs_deepbiomass_yearmatched_masked.png

Mg/ha = tCO2/acre / 0.6977.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import geopandas as gpd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import from_origin
from shapely.geometry import Point

REPO = Path("/home/mattc/code/agb-ml-agent-evolve")
EXPDIR = REPO / "experiments/agb_ireland_biomass_regression_20260608"
PREP = EXPDIR / "preprocessing"
FINAL = EXPDIR / "final"

DISSOLVED = PREP / "ireland_locations_dissolved.gpkg"
YEARS = [2022, 2023, 2024]
TCO2ACRE_TO_MGHA = 1.0 / 0.6977
RES = 10.0
NODATA = -9999.0


def build_masked_geotiff(name: str, pix: pd.DataFrame, tifdir: Path) -> int:
    """Rasterise MASKED per-pixel preds to a 10 m EPSG:2157 2-band GeoTIFF.

    Non-forest pixels carry value 0 (already in pred_masked_tco2_acre); pixels outside the
    polygon are nodata. Identical grid snapping to aggregate_pixel_outputs.build_geotiff.
    """
    pts = gpd.GeoSeries([Point(xy) for xy in zip(pix.lon, pix.lat)], crs="EPSG:4326").to_crs(
        "EPSG:2157"
    )
    x = np.array([p.x for p in pts])
    y = np.array([p.y for p in pts])
    pred = pix["pred_masked_tco2_acre"].to_numpy()
    x0 = np.floor(x.min() / RES) * RES - RES / 2
    y1 = np.ceil(y.max() / RES) * RES + RES / 2
    col = np.round((x - (x0 + RES / 2)) / RES).astype(int)
    row = np.round(((y1 - RES / 2) - y) / RES).astype(int)
    ncol = int(col.max()) + 1
    nrow = int(row.max()) + 1
    b1 = np.full((nrow, ncol), NODATA, dtype=np.float32)
    b2 = np.full((nrow, ncol), NODATA, dtype=np.float32)
    b1[row, col] = pred.astype(np.float32)
    b2[row, col] = (pred * TCO2ACRE_TO_MGHA).astype(np.float32)
    transform = from_origin(x0, y1, RES, RES)
    out = tifdir / f"{name}.tif"
    with rasterio.open(
        out,
        "w",
        driver="GTiff",
        height=nrow,
        width=ncol,
        count=2,
        dtype="float32",
        crs="EPSG:2157",
        transform=transform,
        nodata=NODATA,
        compress="deflate",
    ) as ds:
        ds.write(b1, 1)
        ds.write(b2, 2)
        ds.set_band_description(1, "pred_AGB_tCO2_per_acre_forestmasked")
        ds.set_band_description(2, "pred_AGB_Mg_per_ha_forestmasked")
    return int((b1 != NODATA).sum())


def build_vrt(tifdir: Path) -> None:
    tifs = sorted(str(p) for p in tifdir.glob("*.tif"))
    listf = tifdir / "_tiflist.txt"
    listf.write_text("\n".join(tifs) + "\n")
    subprocess.run(
        ["gdalbuildvrt", "-b", "1", "-input_file_list", str(listf), str(tifdir / "_index.vrt")],
        check=True,
        capture_output=True,
    )


def aggregate_mask_set(mask_dir: Path, tifdir: Path | None, make_tiffs: bool) -> pd.DataFrame:
    """Aggregate one DW-mask checkpoint set to stand level; optionally write masked GeoTIFFs."""
    sumdir = mask_dir / "_summary"
    rows = []
    if make_tiffs and tifdir is not None:
        tifdir.mkdir(parents=True, exist_ok=True)
    for sumf in sorted(sumdir.glob("*.json")):
        name = sumf.stem
        summ = json.loads(sumf.read_text())
        if make_tiffs and tifdir is not None:
            pix = pd.read_parquet(mask_dir / f"{name}.parquet")
            build_masked_geotiff(name, pix, tifdir)
        rows.append(summ)
    if make_tiffs and tifdir is not None:
        build_vrt(tifdir)
    return pd.DataFrame(rows).set_index("Location_Name")


def main() -> None:
    gdf = gpd.read_file(DISSOLVED)

    # ---------- YEAR-MATCHED SET ----------
    ym = pd.read_parquet(FINAL / "ireland_agb_yearmatched.parquet")
    assert len(ym) == 141, f"yearmatched n={len(ym)}"
    ym_unmasked_cols = list(ym.columns)
    ymi = ym.set_index("Location_Name")

    masked_per_year = {}
    for y in YEARS:
        mdir = PREP / f"_dw_mask_y{y}"
        tdir = FINAL / f"ireland_pixel_tiffs_{y}_masked"
        agg = aggregate_mask_set(mdir, tdir, make_tiffs=True)
        masked_per_year[y] = agg
        ymi[f"forest_frac_{y}"] = agg["forest_fraction"]
        ymi[f"our_{y}_masked_tCO2_acre"] = agg["masked_pred_pixel_tCO2_acre"]
        ymi[f"our_{y}_masked_Mg_ha"] = agg["masked_pred_pixel_tCO2_acre"] * TCO2ACRE_TO_MGHA
        # re-aggregation identity check: masked == forest_frac * mean(forest preds)
        # masked_density already = mean over ALL pixels with non-forest=0, by construction.
    ymi["our_mean_2022_24_masked_tCO2_acre"] = ymi[
        [f"our_{y}_masked_tCO2_acre" for y in YEARS]
    ].mean(axis=1)
    ymi["our_mean_2022_24_masked_Mg_ha"] = (
        ymi["our_mean_2022_24_masked_tCO2_acre"] * TCO2ACRE_TO_MGHA
    )

    new_ym_cols = (
        [f"forest_frac_{y}" for y in YEARS]
        + [f"our_{y}_masked_tCO2_acre" for y in YEARS]
        + [f"our_{y}_masked_Mg_ha" for y in YEARS]
        + ["our_mean_2022_24_masked_tCO2_acre", "our_mean_2022_24_masked_Mg_ha"]
    )
    ym_out = ymi.reset_index()[ym_unmasked_cols + new_ym_cols]
    # verify unmasked columns unchanged
    assert ym_out[ym_unmasked_cols].reset_index(drop=True).equals(ym.reset_index(drop=True)), (
        "unmasked yearmatched columns changed!"
    )
    ym_out.to_parquet(FINAL / "ireland_agb_yearmatched.parquet", index=False)
    ym_out.to_csv(FINAL / "ireland_agb_yearmatched.csv", index=False)
    gpoly = gdf[["Location_Name", "geometry"]].to_crs("EPSG:2157")
    gpd.GeoDataFrame(
        gpoly.merge(ym_out, on="Location_Name", how="inner"), geometry="geometry", crs="EPSG:2157"
    ).to_file(FINAL / "ireland_agb_yearmatched.gpkg", driver="GPKG")

    # ---------- SURVEY-YEAR (PIXEL) SET ----------
    px = pd.read_parquet(FINAL / "ireland_agb_pixel.parquet")
    assert len(px) == 141, f"pixel n={len(px)}"
    px_unmasked_cols = list(px.columns)
    pxi = px.set_index("Location_Name")

    sdir = PREP / "_dw_mask_ysurvey"
    stif = FINAL / "ireland_pixel_tiffs_masked"
    sagg = aggregate_mask_set(sdir, stif, make_tiffs=True)
    pxi["forest_fraction"] = sagg["forest_fraction"]
    pxi["pred_pixel_masked_tCO2_acre"] = sagg["masked_pred_pixel_tCO2_acre"]
    pxi["pred_pixel_masked_Mg_ha"] = sagg["masked_pred_pixel_tCO2_acre"] * TCO2ACRE_TO_MGHA
    pxi["pred_pixel_masked_total_t"] = pxi["pred_pixel_masked_Mg_ha"] * pxi["area_ha"]

    new_px_cols = [
        "forest_fraction",
        "pred_pixel_masked_tCO2_acre",
        "pred_pixel_masked_Mg_ha",
        "pred_pixel_masked_total_t",
    ]
    px_out = pxi.reset_index()[px_unmasked_cols + new_px_cols]
    assert px_out[px_unmasked_cols].reset_index(drop=True).equals(px.reset_index(drop=True)), (
        "unmasked pixel columns changed!"
    )
    px_out.to_parquet(FINAL / "ireland_agb_pixel.parquet", index=False)
    px_out.to_csv(FINAL / "ireland_agb_pixel.csv", index=False)
    gpd.GeoDataFrame(
        gpoly.merge(px_out, on="Location_Name", how="inner"), geometry="geometry", crs="EPSG:2157"
    ).to_file(FINAL / "ireland_agb_pixel.gpkg", driver="GPKG")

    # ---------- FIGURE (dual-scale, masked vs unmasked, vs DB) ----------
    o = ym_out
    fig, axes = plt.subplots(2, 3, figsize=(17, 10))
    lims = [
        0,
        max(
            o[[f"our_{y}_tCO2_acre" for y in YEARS]].max().max(),
            o[[f"db_{y}_tCO2_acre" for y in YEARS]].max().max(),
        )
        * 1.05,
    ]

    def mgha_secondary(ax):
        sec = ax.secondary_yaxis(
            "right", functions=(lambda v: v * TCO2ACRE_TO_MGHA, lambda v: v / TCO2ACRE_TO_MGHA)
        )
        sec.set_ylabel("Mg/ha", fontsize=8)

    for ax, y in zip(axes[0], YEARS):
        ax.scatter(
            o[f"db_{y}_tCO2_acre"],
            o[f"our_{y}_tCO2_acre"],
            s=16,
            alpha=0.35,
            color="0.6",
            label="unmasked",
        )
        ax.scatter(
            o[f"db_{y}_tCO2_acre"],
            o[f"our_{y}_masked_tCO2_acre"],
            s=18,
            alpha=0.7,
            color="C0",
            label="masked",
        )
        ax.plot(lims, lims, "k--", lw=1)
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        rmask = o[f"our_{y}_masked_tCO2_acre"].mean() / o[f"db_{y}_tCO2_acre"].mean()
        ax.set_xlabel(f"Deep Biomass {y} (tCO2/acre)")
        ax.set_ylabel(f"Our {y} (tCO2/acre)")
        ax.set_title(f"{y}: masked ratio {rmask:.2f}x")
        ax.legend(loc="upper left", fontsize=7)
        mgha_secondary(ax)

    # trajectory
    axt = axes[1, 0]
    our_um = [o[f"our_{y}_tCO2_acre"].mean() for y in YEARS]
    our_m = [o[f"our_{y}_masked_tCO2_acre"].mean() for y in YEARS]
    db_t = [o[f"db_{y}_tCO2_acre"].mean() for y in YEARS]
    axt.plot(YEARS, our_um, "o--", color="0.6", label="Our unmasked")
    axt.plot(YEARS, our_m, "o-", color="C0", label="Our masked")
    axt.plot(YEARS, db_t, "s-", color="C3", label="Deep Biomass")
    axt.set_xticks(YEARS)
    axt.set_xlabel("Year")
    axt.set_ylabel("Portfolio mean (tCO2/acre)")
    axt.set_title("Portfolio trajectory (masked vs unmasked)")
    axt.legend(fontsize=7)
    mgha_secondary(axt)

    # 3yr-mean scatter
    axm = axes[1, 1]
    axm.scatter(
        o["db_mean_2022_24_tCO2_acre"],
        o["our_mean_2022_24_tCO2_acre"],
        s=16,
        alpha=0.35,
        color="0.6",
        label="unmasked",
    )
    axm.scatter(
        o["db_mean_2022_24_tCO2_acre"],
        o["our_mean_2022_24_masked_tCO2_acre"],
        s=18,
        alpha=0.7,
        color="C2",
        label="masked",
    )
    axm.plot(lims, lims, "k--", lw=1)
    axm.set_xlim(lims)
    axm.set_ylim(lims)
    rmean = o["our_mean_2022_24_masked_tCO2_acre"].mean() / o["db_mean_2022_24_tCO2_acre"].mean()
    axm.set_xlabel("Deep Biomass 2022-24 mean (tCO2/acre)")
    axm.set_ylabel("Our 2022-24 mean (tCO2/acre)")
    axm.set_title(f"3yr mean: masked ratio {rmean:.2f}x")
    axm.legend(loc="upper left", fontsize=7)
    mgha_secondary(axm)

    # forest_fraction vs age
    axf = axes[1, 2]
    ff_mean = o[[f"forest_frac_{y}" for y in YEARS]].mean(axis=1)
    axf.scatter(o["age_at_survey"], ff_mean, s=18, alpha=0.6, color="C5")
    axf.set_xlabel("Dasos age_at_survey (yr)")
    axf.set_ylabel("mean forest_fraction (2022-24)")
    axf.set_title("Forest fraction vs stand age")
    axf.axhline(0.5, color="k", ls=":", lw=0.8)

    fig.suptitle(
        "Ireland AGB vs Deep Biomass, YEAR-MATCHED, DW forest-masked (non-forest -> 0)", fontsize=13
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    (FINAL / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(FINAL / "figures/ireland_vs_deepbiomass_yearmatched_masked.png", dpi=130)
    plt.close(fig)

    print("Wrote masked yearmatched + pixel outputs, GeoTIFFs, VRTs, figure.")
    # portfolio quicklook
    for y in YEARS:
        print(
            f"{y}: unmasked={o[f'our_{y}_tCO2_acre'].mean():.2f} "
            f"masked={o[f'our_{y}_masked_tCO2_acre'].mean():.2f} "
            f"db={o[f'db_{y}_tCO2_acre'].mean():.2f} ff_mean={o[f'forest_frac_{y}'].mean():.3f}"
        )
    print(
        f"3yr mean: unmasked={o['our_mean_2022_24_tCO2_acre'].mean():.2f} "
        f"masked={o['our_mean_2022_24_masked_tCO2_acre'].mean():.2f} "
        f"db={o['db_mean_2022_24_tCO2_acre'].mean():.2f}"
    )


if __name__ == "__main__":
    main()
