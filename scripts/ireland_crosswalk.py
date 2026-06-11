"""
Ireland AGB transfer — crosswalk + dissolve.

Build the Deep-Biomass `Location Name` -> Dasos gpkg `SiteName` crosswalk (direct match +
underscore->slash), assert all 141 resolve, then DISSOLVE the 1,053 sub-compartments up to 141
`SiteName` Locations carrying area-weighted structural covariates and a representative survey year.

Outputs (under the experiment preprocessing/ dir):
  - ireland_locations_dissolved.gpkg  : 141 Location MultiPolygons (EPSG:4326) + covariates
  - db_reference.parquet              : per-Location Deep-Biomass reference (Mg/ha + tCO2/acre)

Run:
    uv run --project /home/mattc/code/agb-ml-agent-evolve python scripts/ireland_crosswalk.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

import geopandas as gpd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

REPO = Path("/home/mattc/code/agb-ml-agent-evolve")
EXPDIR = REPO / "experiments/agb_ireland_biomass_regression_20260608"
PREP = EXPDIR / "preprocessing"

DB_CSV = Path(
    "/home/mattc/data-space/carbonmap-embeddings/dasos-ireland/deepbiomass-model-outputs/"
    "Deep Biomass - Aggregated Data & Portfolio Summary.csv"
)
GPKG = Path("/home/mattc/data-space/carbonmap-embeddings/boundary-files/dasos_fgl_2025ye.gpkg")
GPKG_LAYER = "fgl_2025ye_"

ITM = "EPSG:2157"  # Irish Transverse Mercator (metric area)
WGS84 = "EPSG:4326"

# Deep Biomass -> tCO2/acre conversion (0.47 IPCC C fraction x 3.667 CO2/C x 0.4047 ha/acre).
MGHA_TO_TCO2ACRE = 0.6977

# AlphaEarth annual embedding coverage.
AEF_YEAR_MIN, AEF_YEAR_MAX = 2017, 2025

# Covariates carried to the Location level (area-weighted where numeric).
NUMERIC_COVARS = ["PlantingYe", "Hmean", "Hdom", "YC", "BA_Conifer"]


# ---------------------------------------------------------------------------
# Deep Biomass CSV
# ---------------------------------------------------------------------------


def _coerce_numeric(s: pd.Series) -> pd.Series:
    """CSV cells are thousands-separated strings inside quotes (e.g. "1,074") -> float."""
    return pd.to_numeric(
        s.astype(str).str.replace(",", "", regex=False).str.strip(), errors="coerce"
    )


def load_db() -> pd.DataFrame:
    df = pd.read_csv(DB_CSV)
    # 141 Location rows have a numeric Location No; 2 footer rows have it blank.
    df = df[pd.to_numeric(df["Location No"], errors="coerce").notna()].copy()
    assert len(df) == 141, f"expected 141 DB Location rows, got {len(df)}"
    year_cols = [c for c in df.columns if c.strip().isdigit()]
    for c in ["Area Ha", *year_cols]:
        df[c] = _coerce_numeric(df[c])
    df["Location_Name"] = df["Location Name"].astype(str).str.strip()
    return df


# ---------------------------------------------------------------------------
# Crosswalk
# ---------------------------------------------------------------------------


def build_crosswalk(db: pd.DataFrame, gdf: gpd.GeoDataFrame) -> pd.DataFrame:
    sitenames = set(gdf["SiteName"].astype(str))
    rows = []
    for nm in db["Location_Name"]:
        if nm in sitenames:
            rows.append((nm, nm, "direct"))
        elif nm.replace("_", "/") in sitenames:
            rows.append((nm, nm.replace("_", "/"), "underscore_to_slash"))
        else:
            rows.append((nm, None, "UNRESOLVED"))
    cw = pd.DataFrame(rows, columns=["Location_Name", "SiteName", "method"])
    n_unres = (cw["method"] == "UNRESOLVED").sum()
    assert n_unres == 0, f"{n_unres} Locations failed to resolve: " + ", ".join(
        cw.loc[cw["method"] == "UNRESOLVED", "Location_Name"]
    )
    assert len(cw) == 141
    print(
        f"Crosswalk: {len(cw)}/141 resolve "
        f"({(cw['method'] == 'direct').sum()} direct, "
        f"{(cw['method'] == 'underscore_to_slash').sum()} underscore->slash)."
    )
    return cw


# ---------------------------------------------------------------------------
# Survey-year alignment (per Location)
# ---------------------------------------------------------------------------


def location_survey_year(sub: gpd.GeoDataFrame) -> tuple[int, int, bool]:
    """Return (survey_year_clamped, survey_year_raw_mode, used_pre2017_fallback).

    Representative year = area-weighted mode of the sub-compartment SurveyDate years; if every
    sub-compartment year is missing, fall back to AEF_YEAR_MIN. Clamp to AEF coverage; flag when
    the chosen year was < 2017 (pre-AlphaEarth) and had to be lifted to 2017.
    """
    yrs = pd.to_datetime(sub["SurveyDate"], format="%d/%m/%Y", errors="coerce").dt.year
    area = sub.geometry.area  # already in metric CRS when called
    valid = yrs.notna()
    if not valid.any():
        return AEF_YEAR_MIN, -1, True
    # area-weighted vote per year
    vote = (
        pd.DataFrame({"year": yrs[valid].astype(int), "w": area[valid].to_numpy()})
        .groupby("year")["w"]
        .sum()
    )
    raw = int(vote.idxmax())
    clamped = min(max(raw, AEF_YEAR_MIN), AEF_YEAR_MAX)
    return clamped, raw, raw < AEF_YEAR_MIN


# ---------------------------------------------------------------------------
# Dissolve + covariates
# ---------------------------------------------------------------------------


def dissolve_locations(gdf: gpd.GeoDataFrame, cw: pd.DataFrame) -> gpd.GeoDataFrame:
    gdf_m = gdf.to_crs(ITM)
    sitename_to_loc = dict(zip(cw["SiteName"], cw["Location_Name"]))

    records = []
    geoms = []
    for sitename, loc in sitename_to_loc.items():
        sub = gdf_m[gdf_m["SiteName"] == sitename]
        assert len(sub) > 0, f"no sub-compartments for SiteName {sitename}"
        area = sub.geometry.area
        tot_area = float(area.sum())

        rec: dict = {"Location_Name": loc, "SiteName": sitename, "n_subcpt": len(sub)}

        # area-weighted numeric covariates (PlantingYe later -> age at survey)
        for col in NUMERIC_COVARS:
            v = pd.to_numeric(sub[col], errors="coerce")
            mask = v.notna()
            rec[col] = (
                float((v[mask] * area[mask]).sum() / area[mask].sum()) if mask.any() else np.nan
            )

        # dominant species by area share
        sp = sub[["MainSp"]].copy()
        sp["w"] = area.to_numpy()
        sp = sp[sp["MainSp"].notna()]
        if len(sp):
            share = sp.groupby("MainSp")["w"].sum()
            rec["MainSp"] = str(share.idxmax())
            rec["MainSp_area_share"] = float(share.max() / tot_area)
        else:
            rec["MainSp"] = None
            rec["MainSp_area_share"] = np.nan

        sy, sy_raw, fallback = location_survey_year(sub)
        rec["survey_year"] = sy
        rec["survey_year_raw_mode"] = sy_raw
        rec["pre2017_fallback"] = fallback
        rec["area_ha"] = tot_area / 1e4
        # stand age at survey: clamp future PlantingYe to survey year (replant scheduling).
        if np.isfinite(rec["PlantingYe"]):
            rec["age_at_survey"] = max(sy - min(rec["PlantingYe"], sy), 0.0)
        else:
            rec["age_at_survey"] = np.nan

        records.append(rec)
        geoms.append(sub.geometry.union_all())

    out = gpd.GeoDataFrame(records, geometry=geoms, crs=ITM)
    return out.to_crs(WGS84)


# ---------------------------------------------------------------------------
# Deep Biomass reference per Location
# ---------------------------------------------------------------------------


def db_reference(db: pd.DataFrame) -> pd.DataFrame:
    """Per-Location DB density (Mg/ha) = cell tonnes / Area_Ha; 2020-24 mean & 2024-only.

    cells = total tonnes AGB = Mg/ha x Area_Ha (data_profile A, verified vs footer).
    """
    win = ["2020", "2021", "2022", "2023", "2024"]
    rows = []
    for _, r in db.iterrows():
        area = r["Area Ha"]
        dens = {y: (r[y] / area if area and np.isfinite(r[y]) else np.nan) for y in win}
        mean_2024 = dens["2024"]
        mean_window = np.nanmean([dens[y] for y in win])
        rows.append(
            {
                "Location_Name": r["Location_Name"],
                "Area_Ha": area,
                "db_mgha_2020_2024_mean": mean_window,
                "db_mgha_2024": mean_2024,
                "db_tco2acre_2020_2024_mean": mean_window * MGHA_TO_TCO2ACRE,
                "db_tco2acre_2024": mean_2024 * MGHA_TO_TCO2ACRE,
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    PREP.mkdir(parents=True, exist_ok=True)

    db = load_db()
    gdf = gpd.read_file(GPKG, layer=GPKG_LAYER)
    assert len(gdf) == 1053, f"expected 1053 sub-compartments, got {len(gdf)}"
    print(f"Loaded DB ({len(db)} Locations) + gpkg ({len(gdf)} sub-cpts).")

    cw = build_crosswalk(db, gdf)

    dissolved = dissolve_locations(gdf, cw)
    assert len(dissolved) == 141, f"expected 141 dissolved Locations, got {len(dissolved)}"
    n_fallback = int(dissolved["pre2017_fallback"].sum())
    print(f"Dissolved -> {len(dissolved)} Locations. pre-2017 fallbacks: {n_fallback}")
    print("survey_year distribution:")
    print(dissolved["survey_year"].value_counts().sort_index().to_string())

    out_gpkg = PREP / "ireland_locations_dissolved.gpkg"
    dissolved.to_file(out_gpkg, layer="locations", driver="GPKG")
    print(f"Wrote {out_gpkg}")

    ref = db_reference(db)
    ref_path = PREP / "db_reference.parquet"
    ref.to_parquet(ref_path, index=False)
    print(f"Wrote {ref_path}  (2020-24 mean Mg/ha: {ref['db_mgha_2020_2024_mean'].mean():.2f})")

    cw.to_csv(PREP / "crosswalk_location_to_sitename.csv", index=False)


if __name__ == "__main__":
    main()
