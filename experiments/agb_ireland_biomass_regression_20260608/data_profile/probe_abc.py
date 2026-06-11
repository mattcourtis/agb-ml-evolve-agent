"""Profile DB CSV (A), Dasos gpkg + crosswalk (B), training parquet encoding (C)."""

from __future__ import annotations

import re
import warnings
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 60)

CSV = Path(
    "/home/mattc/data-space/carbonmap-embeddings/dasos-ireland/"
    "deepbiomass-model-outputs/Deep Biomass - Aggregated Data & Portfolio Summary.csv"
)
GPKG = Path("/home/mattc/data-space/carbonmap-embeddings/boundary-files/dasos_fgl_2025ye.gpkg")
PARQUET = Path(
    "/home/mattc/code/tf-deep-landcover/experiments/agb/usa_v1_pilot_joint_v2/features.parquet"
)

YEARS = [str(y) for y in range(2013, 2025)]


def part_a() -> pd.DataFrame:
    print("=" * 80)
    print("PART A — Deep Biomass CSV")
    print("=" * 80)
    raw = pd.read_csv(CSV, thousands=",")
    print(f"raw shape: {raw.shape}; columns: {list(raw.columns)}")
    loc = raw[raw["Location No"].notna()].copy()
    foot = raw[raw["Location No"].isna()].copy()
    print(f"\nLocation rows: {len(loc)}  (expect 141)")
    print(f"Footer rows: {len(foot)}")
    print("Footer:")
    print(foot.to_string(index=False))

    def tonum(s):
        return pd.to_numeric(
            s.astype(str).str.replace(",", "", regex=False).str.strip(), errors="coerce"
        )

    loc["Area Ha"] = tonum(loc["Area Ha"])
    for y in YEARS:
        loc[y] = tonum(loc[y])
    print(f"\nArea_Ha sum (locations): {loc['Area Ha'].sum():.2f} (footer says 3367.63)")
    print(f"unique Location Names: {loc['Location Name'].nunique()}")

    # cells = total tonnes AGB = Mg/ha * Area_ha  -> recover Mg/ha
    mgha = loc[YEARS].astype(float).div(loc["Area Ha"], axis=0)
    print("\n--- Per-Location Mg/ha = cell / Area_Ha : distribution per year ---")
    desc = mgha.describe().T[["mean", "50%", "min", "max", "std"]]
    print(desc.to_string())
    print("\nPortfolio mean of (per-location Mg/ha) per year:")
    print(mgha.mean().round(2).to_string())
    print("\nCHECK vs footer ton/ha (area-weighted): total tonnes / total area")
    aw = loc[YEARS].astype(float).sum() / loc["Area Ha"].sum()
    print(aw.round(1).to_string())

    print("\n2020-2024 mean per-Location Mg/ha distribution:")
    m2024 = mgha[["2020", "2021", "2022", "2023", "2024"]].mean(axis=1)
    print(m2024.describe().round(2).to_string())

    # spurious flags
    print("\n--- Noise/spurious flags ---")
    print("Ahalahana 2015 cell:", loc.loc[loc["Location Name"] == "Ahalahana", "2015"].values)
    tiny = (loc[YEARS].astype(float) < 5).sum().sum()
    print(f"cells with total < 5 ton (likely spurious): {int(tiny)}")
    # year-on-year jumps
    ratio = mgha[YEARS].max(axis=1) / mgha[YEARS].replace(0, np.nan).min(axis=1)
    print(f"max/min Mg/ha ratio across years: median={ratio.median():.1f} max={ratio.max():.1f}")
    return loc


def part_b(loc: pd.DataFrame) -> None:
    print("\n" + "=" * 80)
    print("PART B — Dasos gpkg")
    print("=" * 80)
    import pyogrio

    layers = pyogrio.list_layers(str(GPKG))
    print(f"layers: {layers.tolist()}")
    gdf = gpd.read_file(GPKG)
    print(f"shape: {gdf.shape}  (expect 1053 sub-compartments)")
    print(f"CRS: {gdf.crs}")
    print(f"geom types: {gdf.geom_type.value_counts().to_dict()}")
    print(f"all valid: {gdf.geometry.is_valid.all()}  invalid={(~gdf.geometry.is_valid).sum()}")
    print(f"\ncolumns ({len(gdf.columns)}):")
    for c in gdf.columns:
        print(f"  {c:18s} {str(gdf[c].dtype):10s} nmiss={gdf[c].isna().sum()}")

    print(f"\nSiteName unique: {gdf['SiteName'].nunique()}")

    # --- Crosswalk Location Name -> SiteName ---
    print("\n--- Crosswalk Location Name -> SiteName ---")
    sites = set(gdf["SiteName"].dropna().unique())
    rows = []
    for nm in loc["Location Name"]:
        if nm in sites:
            rows.append((nm, nm, "direct"))
        elif "_" in nm and nm.replace("_", "/") in sites:
            # CSV uses Group_Site, gpkg SiteName uses Group/Site
            rows.append((nm, nm.replace("_", "/"), "underscore_to_slash"))
        elif "_" in nm and nm.split("_", 1)[1] in sites:
            rows.append((nm, nm.split("_", 1)[1], "underscore_split_suffix"))
        else:
            rows.append((nm, None, "FAIL"))
    cw = pd.DataFrame(rows, columns=["Location_Name", "SiteName", "method"])
    print(cw["method"].value_counts().to_string())
    fails = cw[cw["method"] == "FAIL"]
    print(f"resolved: {(cw['method'] != 'FAIL').sum()} / {len(cw)}")
    if len(fails):
        print("FAILURES:")
        print(fails.to_string(index=False))
    cw.to_csv(Path(__file__).parent / "crosswalk_location_to_sitename.csv", index=False)

    # --- dissolved area per Location vs CSV Area_Ha ---
    print("\n--- Dissolved area per Location vs CSV Area_Ha ---")
    g = gdf.copy()
    # area in ITM (EPSG:2157) for metric area
    g_itm = g.to_crs(2157)
    g["area_ha_geom"] = g_itm.geometry.area / 1e4
    site_area = g.groupby("SiteName")["area_ha_geom"].sum()
    cw2 = cw.merge(
        loc[["Location Name", "Area Ha"]],
        left_on="Location_Name",
        right_on="Location Name",
        how="left",
    )
    cw2["geom_area_ha"] = cw2["SiteName"].map(site_area)
    cw2["diff_pct"] = 100 * (cw2["geom_area_ha"] - cw2["Area Ha"]) / cw2["Area Ha"]
    print(f"total geom area (ITM): {g['area_ha_geom'].sum():.1f} ha (CSV 3367.63)")
    print(cw2["diff_pct"].describe().round(2).to_string())
    print(f"|diff|>10%: {(cw2['diff_pct'].abs() > 10).sum()} locations")

    # --- covariate profiling ---
    print("\n--- Covariate profiles (eval cuts) ---")
    if "MainSp" in gdf:
        print("MainSp top:")
        print(gdf["MainSp"].value_counts().head(10).to_string())
        ss = (gdf["MainSp"] == "SS").sum()
        print(f"Sitka SS: {ss}/{len(gdf)} ({100 * ss / len(gdf):.0f}%)")
    for col in [
        "PlantingYe",
        "SurveyDate",
        "YC",
        "Hmean",
        "Hdom",
        "BA_Conifer",
        "Thinned",
        "MgtRegime",
        "SecSp",
        "GrossArea",
        "ProdArea",
    ]:
        if col in gdf:
            s = gdf[col]
            if pd.api.types.is_numeric_dtype(s):
                print(
                    f"{col:12s} num  min={s.min()} max={s.max()} "
                    f"mean={s.mean():.2f} nmiss={s.isna().sum()}"
                )
            else:
                vc = s.value_counts().head(6).to_dict()
                print(f"{col:12s} cat  nmiss={s.isna().sum()} top={vc}")
    # SurveyDate range parse
    if "SurveyDate" in gdf:
        sd = pd.to_datetime(gdf["SurveyDate"], errors="coerce")
        print(f"\nSurveyDate parsed range: {sd.min()} .. {sd.max()}; year vc:")
        print(sd.dt.year.value_counts(dropna=False).to_string())


def part_c() -> None:
    print("\n" + "=" * 80)
    print("PART C — Training parquet encoding reference")
    print("=" * 80)
    print(f"parquet: {PARQUET}  exists={PARQUET.exists()}")
    df = pd.read_parquet(PARQUET)
    print(f"shape: {df.shape}")
    emb_cols = [c for c in df.columns if re.fullmatch(r"emb_\d{2}", c)]
    print(f"emb columns: {len(emb_cols)} (expect 64); dtype={df[emb_cols[0]].dtype}")
    arr = df[emb_cols].to_numpy(dtype=float)
    print(
        f"value range: min={arr.min():.3f} max={arr.max():.3f} "
        f"mean={arr.mean():.3f} std={arr.std():.3f}"
    )
    # integer-valued?
    frac_int = np.mean(arr == np.round(arr))
    print(f"fraction integer-valued: {frac_int:.3f}")
    print(f"is in int8 range [-128,127]: {arr.min() >= -128 and arr.max() <= 127}")
    print(f"is in dequant float range ~[-1,1]: {arr.min() >= -1.5 and arr.max() <= 1.5}")
    # key cols for part D
    for c in ["lon", "lat", "project_name", "year", "survey_year"]:
        if c in df.columns:
            print(f"col {c}: dtype={df[c].dtype} sample={df[c].dropna().head(2).tolist()}")
    if "project_name" in df.columns:
        print("\nproject_name vc:")
        print(df["project_name"].value_counts().head(15).to_string())
    # save a small sample for part D
    keep = [c for c in ["lon", "lat", "project_name", "year", "survey_year"] if c in df.columns]
    df[keep + emb_cols].to_parquet(Path(__file__).parent / "train_emb_sample.parquet")
    print(f"\nsaved train_emb_sample.parquet ({len(df)} rows)")


if __name__ == "__main__":
    loc = part_a()
    part_b(loc)
    part_c()
