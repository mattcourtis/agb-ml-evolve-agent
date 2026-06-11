"""
Aggregate the three year-matched per-pixel runs (2022/2023/2024) to stand level, join the
year-matched Deep Biomass, run the cross-check vs the accepted single-year run, and emit:
  final/ireland_agb_yearmatched.{csv,parquet,gpkg}  (per stand, all three years + 3yr mean + DB + deltas)
  final/ireland_pixel_tiffs_2022/ , _2023/ , _2024/   (per-stand 2-band rasters + per-year VRT)
  final/figures/ireland_vs_deepbiomass_yearmatched.png (per-year scatter, trajectory, 3yr-mean scatter)

mean(f) = stand AGB density tCO2/acre. Mg/ha = tCO2/acre / 0.6977.
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
DB_YM = PREP / "db_yearmatched.parquet"
ACCEPTED = FINAL / "ireland_agb_pixel.parquet"

YEARS = [2022, 2023, 2024]
TCO2ACRE_TO_MGHA = 1.0 / 0.6977
RES = 10.0
NODATA = -9999.0


def build_geotiff(name: str, pix: pd.DataFrame, tifdir: Path) -> int:
    pts = gpd.GeoSeries([Point(xy) for xy in zip(pix.lon, pix.lat)], crs="EPSG:4326").to_crs(
        "EPSG:2157"
    )
    x = np.array([p.x for p in pts])
    y = np.array([p.y for p in pts])
    pred = pix.pred_tco2_acre.to_numpy()
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
        ds.set_band_description(1, "pred_AGB_tCO2_per_acre")
        ds.set_band_description(2, "pred_AGB_Mg_per_ha")
    return int((b1 != NODATA).sum())


def load_year(year: int, make_tiffs: bool = True) -> pd.DataFrame:
    ckpt = PREP / f"_pixel_pred_y{year}"
    sumdir = ckpt / "_summary"
    tifdir = FINAL / f"ireland_pixel_tiffs_{year}"
    tifdir.mkdir(parents=True, exist_ok=True)
    rows = []
    for sumf in sorted(sumdir.glob("*.json")):
        name = sumf.stem
        summ = json.loads(sumf.read_text())
        if make_tiffs:
            pix = pd.read_parquet(ckpt / f"{name}.parquet")
            build_geotiff(name, pix, tifdir)
        rows.append(
            {
                "Location_Name": name,
                f"our_{year}_tCO2_acre": summ["pred_pixel_tCO2_acre"],
                f"our_{year}_Mg_ha": summ["pred_pixel_Mg_ha"],
                f"n_pixels_{year}": summ["n_pixels"],
            }
        )
    # per-year VRT over band1
    tifs = sorted(str(p) for p in tifdir.glob("*.tif"))
    if tifs:
        listf = tifdir / "_tiflist.txt"
        listf.write_text("\n".join(tifs) + "\n")
        subprocess.run(
            ["gdalbuildvrt", "-b", "1", "-input_file_list", str(listf), str(tifdir / "_index.vrt")],
            check=True,
            capture_output=True,
        )
    return pd.DataFrame(rows)


def main() -> None:
    gdf = gpd.read_file(DISSOLVED)
    cov = gdf.set_index("Location_Name")[
        ["area_ha", "MainSp", "age_at_survey", "Hdom", "YC", "survey_year"]
    ]
    db = pd.read_parquet(DB_YM).set_index("Location_Name")

    per_year = [load_year(y).set_index("Location_Name") for y in YEARS]
    out = pd.concat(per_year, axis=1)
    out = out.join(cov, how="left").join(db, how="left")

    # our 3-year mean
    out["our_mean_2022_24_tCO2_acre"] = out[[f"our_{y}_tCO2_acre" for y in YEARS]].mean(axis=1)
    out["our_mean_2022_24_Mg_ha"] = out[[f"our_{y}_Mg_ha" for y in YEARS]].mean(axis=1)

    # per-year + mean deltas (our - db)
    for y in YEARS:
        out[f"delta_{y}_tCO2_acre"] = out[f"our_{y}_tCO2_acre"] - out[f"db_{y}_tCO2_acre"]
    out["delta_mean_tCO2_acre"] = (
        out["our_mean_2022_24_tCO2_acre"] - out["db_mean_2022_24_tCO2_acre"]
    )

    out = out.reset_index().rename(columns={"index": "Location_Name"})
    out = out.sort_values("Location_Name").reset_index(drop=True)

    col_order = (
        ["Location_Name", "area_ha", "survey_year", "MainSp", "age_at_survey", "Hdom", "YC"]
        + [f"n_pixels_{y}" for y in YEARS]
        + [f"our_{y}_tCO2_acre" for y in YEARS]
        + [f"our_{y}_Mg_ha" for y in YEARS]
        + ["our_mean_2022_24_tCO2_acre", "our_mean_2022_24_Mg_ha"]
        + [f"db_{y}_tCO2_acre" for y in YEARS]
        + [f"db_{y}_Mg_ha" for y in YEARS]
        + ["db_mean_2022_24_tCO2_acre", "db_mean_2022_24_Mg_ha"]
        + [f"delta_{y}_tCO2_acre" for y in YEARS]
        + ["delta_mean_tCO2_acre"]
    )
    out = out[col_order]
    out.to_csv(FINAL / "ireland_agb_yearmatched.csv", index=False)
    out.to_parquet(FINAL / "ireland_agb_yearmatched.parquet", index=False)

    gpoly = gdf[["Location_Name", "geometry"]].to_crs("EPSG:2157")
    gout = gpoly.merge(out, on="Location_Name", how="inner")
    gpd.GeoDataFrame(gout, geometry="geometry", crs="EPSG:2157").to_file(
        FINAL / "ireland_agb_yearmatched.gpkg", driver="GPKG"
    )

    # ---- cross-check vs accepted single-year run ----
    acc = pd.read_parquet(ACCEPTED)[
        ["Location_Name", "survey_year", "pred_pixel_tCO2_acre"]
    ].set_index("Location_Name")
    oi = out.set_index("Location_Name")
    xc_rows = []
    for name, ar in acc.iterrows():
        sy = int(ar["survey_year"])
        if sy in YEARS and name in oi.index:
            ours = oi.loc[name, f"our_{sy}_tCO2_acre"]
            xc_rows.append(
                {
                    "Location_Name": name,
                    "survey_year": sy,
                    "accepted": ar["pred_pixel_tCO2_acre"],
                    "yearmatched": ours,
                    "abs_diff": abs(ar["pred_pixel_tCO2_acre"] - ours),
                }
            )
    xc = pd.DataFrame(xc_rows)
    xc.to_csv(FINAL / "_yearmatched_crosscheck.csv", index=False)

    # ---- portfolio stats ----
    stats = {"n_stands": len(out)}
    for y in YEARS:
        stats[f"our_{y}"] = float(out[f"our_{y}_tCO2_acre"].mean())
        stats[f"db_{y}"] = float(out[f"db_{y}_tCO2_acre"].mean())
        stats[f"ratio_{y}"] = stats[f"our_{y}"] / stats[f"db_{y}"]
        stats[f"H1_{y}"] = float((out[f"our_{y}_tCO2_acre"] >= out[f"db_{y}_tCO2_acre"]).mean())
    stats["our_mean"] = float(out["our_mean_2022_24_tCO2_acre"].mean())
    stats["db_mean"] = float(out["db_mean_2022_24_tCO2_acre"].mean())
    stats["ratio_mean"] = stats["our_mean"] / stats["db_mean"]
    stats["H1_mean"] = float(
        (out["our_mean_2022_24_tCO2_acre"] >= out["db_mean_2022_24_tCO2_acre"]).mean()
    )
    stats["xcheck_n"] = int(len(xc))
    stats["xcheck_max_abs_diff"] = float(xc["abs_diff"].max()) if len(xc) else None
    stats["xcheck_mean_abs_diff"] = float(xc["abs_diff"].mean()) if len(xc) else None
    (FINAL / "_yearmatched_stats.json").write_text(json.dumps(stats, indent=2))

    # ---- figure ----
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    lims = [
        0,
        max(
            out[[f"our_{y}_tCO2_acre" for y in YEARS]].max().max(),
            out[[f"db_{y}_tCO2_acre" for y in YEARS]].max().max(),
        )
        * 1.05,
    ]
    for ax, y in zip(axes[0], YEARS):
        ax.scatter(out[f"db_{y}_tCO2_acre"], out[f"our_{y}_tCO2_acre"], s=18, alpha=0.6, color="C0")
        ax.plot(lims, lims, "k--", lw=1, label="1:1")
        ax.set_xlim(lims)
        ax.set_ylim(lims)
        ax.set_xlabel(f"Deep Biomass {y} (tCO2/acre)")
        ax.set_ylabel(f"Our mean(f) {y} (tCO2/acre)")
        ax.set_title(f"{y}: ratio {stats[f'ratio_{y}']:.2f}x, H1={stats[f'H1_{y}']:.2f}")
        ax.legend(loc="upper left", fontsize=8)

    # trajectory
    axt = axes[1, 0]
    our_traj = [stats[f"our_{y}"] for y in YEARS]
    db_traj = [stats[f"db_{y}"] for y in YEARS]
    axt.plot(YEARS, our_traj, "o-", color="C0", label="Our mean(f)")
    axt.plot(YEARS, db_traj, "s-", color="C3", label="Deep Biomass")
    for xx, yy in zip(YEARS, our_traj):
        axt.annotate(f"{yy:.1f}", (xx, yy), textcoords="offset points", xytext=(0, 6), fontsize=8)
    for xx, yy in zip(YEARS, db_traj):
        axt.annotate(f"{yy:.1f}", (xx, yy), textcoords="offset points", xytext=(0, -12), fontsize=8)
    axt.set_xticks(YEARS)
    axt.set_xlabel("Year")
    axt.set_ylabel("Portfolio mean (tCO2/acre)")
    axt.set_title("Portfolio trajectory 2022->2024")
    axt.legend(fontsize=8)

    # 3yr-mean scatter
    axm = axes[1, 1]
    axm.scatter(
        out["db_mean_2022_24_tCO2_acre"],
        out["our_mean_2022_24_tCO2_acre"],
        s=18,
        alpha=0.6,
        color="C2",
    )
    axm.plot(lims, lims, "k--", lw=1, label="1:1")
    axm.set_xlim(lims)
    axm.set_ylim(lims)
    axm.set_xlabel("Deep Biomass 2022-24 mean (tCO2/acre)")
    axm.set_ylabel("Our 2022-24 mean (tCO2/acre)")
    axm.set_title(f"3yr mean: ratio {stats['ratio_mean']:.2f}x, H1={stats['H1_mean']:.2f}")
    axm.legend(loc="upper left", fontsize=8)

    # cross-check panel
    axc = axes[1, 2]
    if len(xc):
        axc.scatter(xc["accepted"], xc["yearmatched"], s=18, alpha=0.6, color="C4")
        cl = [0, max(xc["accepted"].max(), xc["yearmatched"].max()) * 1.05]
        axc.plot(cl, cl, "k--", lw=1, label="1:1")
        axc.set_xlim(cl)
        axc.set_ylim(cl)
        axc.set_xlabel("Accepted single-year mean(f)")
        axc.set_ylabel("Year-matched mean(f) (same year)")
        axc.set_title(f"Cross-check n={len(xc)}, max|d|={xc['abs_diff'].max():.3f}")
        axc.legend(loc="upper left", fontsize=8)
    fig.suptitle(
        "Ireland AGB: our model vs Deep Biomass, YEAR-MATCHED (2022/2023/2024)", fontsize=13
    )
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    (FINAL / "figures").mkdir(parents=True, exist_ok=True)
    fig.savefig(FINAL / "figures/ireland_vs_deepbiomass_yearmatched.png", dpi=130)
    print("Wrote tabular + gpkg + figure.")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
