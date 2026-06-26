"""
CANONICAL CODEC STORE — all 52 ANEW projects in one training-codec embedding table.

No GEE. Takes the raw-GEE pool embeddings (all 12,837 ANEW plots, every project) and
applies the per-band affine (aef_affine.parquet) to land them in the deployed model's
training-codec space, then joins lon/lat from the ANEW GT gpkg. This is the substrate
for the codec-space all-projects overlays and the 29-project DI→error validation.

Embeddings only (+ CO2 label, eco/biome, lon/lat) — co-features (chm/topo/dstx) are NOT
extracted (no-GEE scope). Ireland is handled separately in the overlay step (it is a
prediction target, not ANEW training data, and is already codec in ireland_features.parquet).

Output (data-space, gitignored):
    {DATASPACE}/agb_trust_aoa_20260626/preprocessing/anew_canonical_codec.parquet
Provenance (repo, tracked):
    experiments/agb_trust_aoa_20260626/preprocessing/{feature_schema.json,data_version.txt}
    experiments/agb_trust_aoa_20260626/final/DATA_STORE.md

Run:
    uv run --project /home/mattc/code/agb-ml-agent-evolve python scripts/trust/build_canonical_codec.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

REPO = Path("/home/mattc/code/agb-ml-agent-evolve")
DATASPACE = Path("/home/mattc/data-space/carbonmap-embeddings")
IRE_PREP = DATASPACE / "agb_ireland_biomass_regression_20260608/preprocessing"
USA_PREP = REPO / "experiments/agb_usa_biomass_regression_20260529/preprocessing"

POOL = IRE_PREP / "iter1_pool_embeddings.parquet"
AFFINE = IRE_PREP / "aef_affine.parquet"
GT_GPKG = DATASPACE / "training-data/anew_gt_with_eco_info.gpkg"
TRAIN = USA_PREP / "features_iter3.parquet"

OUT_PARQUET = DATASPACE / "agb_trust_aoa_20260626/preprocessing/anew_canonical_codec.parquet"
PREP_PROV = REPO / "experiments/agb_trust_aoa_20260626/preprocessing"
SCHEMA = PREP_PROV / "feature_schema.json"
DATA_VERSION = PREP_PROV / "data_version.txt"
DATA_STORE = REPO / "experiments/agb_trust_aoa_20260626/final/DATA_STORE.md"

EMB = [f"emb_{i:02d}" for i in range(64)]
BANDS = [f"A{i:02d}" for i in range(64)]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def key(df: pd.DataFrame, pid: str = "plot_id") -> pd.Series:
    return df["project_name"].astype(str) + "|" + df[pid].astype(str)


def main() -> None:
    OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)
    PREP_PROV.mkdir(parents=True, exist_ok=True)
    DATA_STORE.parent.mkdir(parents=True, exist_ok=True)

    pool = pd.read_parquet(POOL)
    anew = pool[pool["project_name"] != "Ireland"].copy()
    anew["k"] = key(anew)
    n_dup = int(anew["k"].duplicated().sum())
    anew = anew.drop_duplicates("k").reset_index(drop=True)

    # apply affine: raw GEE -> training codec
    affine = pd.read_parquet(AFFINE)
    a = affine.set_index("band")["a"]
    c = affine.set_index("band")["c"]
    for b, e in zip(BANDS, EMB):
        anew[e] = anew[e].to_numpy(float) * a[b] + c[b]

    # lon/lat from the GT gpkg (authoritative geometry)
    g = gpd.read_file(GT_GPKG).to_crs(4326)
    g["k"] = g["project_name"].astype(str) + "|" + g["Plot_ID"].astype(float).astype(str)
    g["lon"] = g.geometry.x
    g["lat"] = g.geometry.y
    anew = anew.merge(g[["k", "lon", "lat"]].drop_duplicates("k"), on="k", how="left")

    # modelled flag + region from the deployed training set
    train = pd.read_parquet(TRAIN)
    train["k"] = key(train)
    reg = train.drop_duplicates("k").set_index("k")["region"]
    anew["modelled"] = anew["k"].isin(set(train["k"]))
    anew["region"] = anew["k"].map(reg)

    cols = [
        "project_name",
        "plot_id",
        "location_id",
        "lon",
        "lat",
        "survey_year",
        "CO2",
        "ECO_NAME",
        "BIOME_NAME",
        "ECO_ID",
        "region",
        "modelled",
    ] + EMB
    out = anew[[col for col in cols if col in anew.columns]].copy()
    out.to_parquet(OUT_PARQUET, index=False)

    n_no_lonlat = int(out["lon"].isna().sum())
    emb_absmax = float(np.nanmax(np.abs(out[EMB].to_numpy(float))))
    print(f"Wrote {OUT_PARQUET}")
    print(
        f"  rows={len(out)} projects={out['project_name'].nunique()} "
        f"modelled={int(out['modelled'].sum())} dropped_dups={n_dup} no_lonlat={n_no_lonlat}"
    )
    print(f"  emb encoding check: |max|={emb_absmax:.1f} (codec expected ~50-210)")

    # ---- provenance ----
    schema = {
        "schema_version": "1.0",
        "experiment_id": "agb_trust_aoa_20260626",
        "description": "All 52 ANEW projects in training-codec embedding space (affine-mapped from "
        "raw pool). Embeddings only + CO2 + eco + lon/lat. For DI/AOA and overlays.",
        "encoding": "training_codec",
        "encoding_note": "raw GEE float pool emb mapped via aef_affine.parquet (emb=a*GEE+c); "
        "verified within Temperate Broadleaf (audit corr>=0.95); UNVERIFIED for conifer/boreal/"
        "tundra/grassland biomes (no codec anchor) — see audit/data_audit.md.",
        "n_rows": int(len(out)),
        "n_projects": int(out["project_name"].nunique()),
        "n_modelled": int(out["modelled"].sum()),
        "target_column": "CO2",
        "target_meaning": "CO2 standing stock, tCO2/acre",
        "id_columns": ["project_name", "plot_id"],
        "cv_partition_key": "project_name",
        "embedding": {
            "count": 64,
            "names_pattern": "emb_00..emb_63",
            "dtype": "float (codec scale)",
        },
        "co_features": "none (no-GEE scope; chm/topo/dstx not extracted for the 29 unused projects)",
    }
    SCHEMA.write_text(json.dumps(schema, indent=2))

    DATA_VERSION.write_text(
        "experiment: agb_trust_aoa_20260626\n"
        "artifact: anew_canonical_codec.parquet\n"
        f"output_path: {OUT_PARQUET}\n"
        "snapshot_timestamp_utc: 2026-06-26\n"
        "inputs:\n"
        f"  pool_iter1_pool_embeddings_sha256: {sha256(POOL)}\n"
        f"  aef_affine_sha256: {sha256(AFFINE)}\n"
        f"  anew_gt_gpkg_sha256: {sha256(GT_GPKG)}\n"
        f"dropped_duplicate_keys: {n_dup}\n"
        f"rows_without_lonlat: {n_no_lonlat}\n"
        "method: raw pool emb -> per-band affine -> codec; lon/lat join from GT gpkg on "
        "(project_name, Plot_ID); modelled flag + region from features_iter3.parquet.\n"
    )

    DATA_STORE.write_text(
        "# Data store — agb_trust_aoa_20260626\n\n"
        f"Outputs live in the data-space (not git): `{OUT_PARQUET.parent}`\n\n"
        "| artifact | rows | contents |\n"
        "|---|--:|---|\n"
        f"| preprocessing/anew_canonical_codec.parquet | {len(out)} | "
        "52 ANEW projects, codec embeddings + CO2 + eco + lon/lat |\n"
        f"| audit/data_audit_summary.json | — | machine-readable audit summary |\n"
    )
    print(f"Wrote provenance: {SCHEMA.name}, {DATA_VERSION.name}, {DATA_STORE}")


if __name__ == "__main__":
    main()
