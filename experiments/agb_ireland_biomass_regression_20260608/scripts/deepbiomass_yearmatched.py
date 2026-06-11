"""
Recompute per-Location Deep Biomass for the fixed years 2022, 2023, 2024 + the 2022-2024 mean.

DB CSV holds per-Location annual TOTAL TONNES. Per year Y:
    Mg/ha   = tonnes_Y / Area_Ha
    tCO2/acre = Mg/ha * 0.6977
The 'Location Name' column matches our Location_Name directly (141/141 join).

Writes preprocessing/db_yearmatched.parquet:
    Location_Name, Area_Ha,
    db_2022_tCO2_acre/.._Mg_ha, db_2023_*, db_2024_*, db_mean_2022_24_tCO2_acre/.._Mg_ha
"""

from __future__ import annotations

from pathlib import Path

import geopandas as gpd
import pandas as pd

REPO = Path("/home/mattc/code/agb-ml-agent-evolve")
EXPDIR = REPO / "experiments/agb_ireland_biomass_regression_20260608"
PREP = EXPDIR / "preprocessing"
DB_CSV = Path(
    "/home/mattc/data-space/carbonmap-embeddings/dasos-ireland/"
    "deepbiomass-model-outputs/Deep Biomass - Aggregated Data & Portfolio Summary.csv"
)
DISSOLVED = PREP / "ireland_locations_dissolved.gpkg"
MGHA_TO_TCO2ACRE = 0.6977
YEARS = [2022, 2023, 2024]


def main() -> None:
    db = pd.read_csv(DB_CSV)
    # numeric: strip thousands commas
    for col in ["Area Ha", *[str(y) for y in YEARS]]:
        db[col] = pd.to_numeric(
            db[col].astype(str).str.replace(",", "", regex=False), errors="coerce"
        )
    db = db.rename(columns={"Location Name": "Location_Name", "Area Ha": "Area_Ha"})

    g = gpd.read_file(DISSOLVED)[["Location_Name"]]
    out = g.merge(
        db[["Location_Name", "Area_Ha", *[str(y) for y in YEARS]]], on="Location_Name", how="left"
    )
    missing = out[out["Area_Ha"].isna()]["Location_Name"].tolist()
    assert not missing, f"DB missing for: {missing}"

    for y in YEARS:
        mgha = out[str(y)] / out["Area_Ha"]
        out[f"db_{y}_Mg_ha"] = mgha
        out[f"db_{y}_tCO2_acre"] = mgha * MGHA_TO_TCO2ACRE
    out["db_mean_2022_24_Mg_ha"] = out[[f"db_{y}_Mg_ha" for y in YEARS]].mean(axis=1)
    out["db_mean_2022_24_tCO2_acre"] = out[[f"db_{y}_tCO2_acre" for y in YEARS]].mean(axis=1)

    keep = (
        ["Location_Name", "Area_Ha"]
        + [c for y in YEARS for c in (f"db_{y}_tCO2_acre", f"db_{y}_Mg_ha")]
        + ["db_mean_2022_24_tCO2_acre", "db_mean_2022_24_Mg_ha"]
    )
    out = out[keep]
    out.to_parquet(PREP / "db_yearmatched.parquet", index=False)
    print(f"Wrote db_yearmatched.parquet ({len(out)} stands)")
    print("Portfolio DB means (tCO2/acre):")
    for y in YEARS:
        print(f"  {y}: {out[f'db_{y}_tCO2_acre'].mean():.3f}")
    print(f"  2022-24 mean: {out['db_mean_2022_24_tCO2_acre'].mean():.3f}")


if __name__ == "__main__":
    main()
