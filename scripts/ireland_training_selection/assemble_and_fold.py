"""
PART 2 — Training-set assembly + CV fold design.

Consumes the Part 1 ranking and turns it into nested candidate training sets for an
Ireland-bound model, plus project-intact CV folds. No model is trained here.

The DI-to-Ireland ranking has only ~4 genuinely-closer projects, then a flat shelf where
~40 projects of all biomes are equidistant from Ireland (gaps < FLAT_GAP). So selection is
gap-based, not a fixed top-K or a plot-floor:

  - core           — recommended. Contiguous top of the ranking while each step still clears
                     FLAT_GAP (the oceanic-conifer/maritime cluster that is actually closer).
  - extended       — core padded down the flat shelf to anti-overfit floors (more plots/variation,
                     but no closer to Ireland). For the bake-off to test whether padding helps.
  - all_minus_err  — every eligible project (baseline).

CV folds keep whole projects together: core uses leave-one-project-out (its natural scheme);
extended uses a grouped spatial K-fold (KMeans on project centroids in EPSG:5070).

Run:
    uv run --project /home/mattc/code/agb-ml-agent-evolve \
        python scripts/ireland_training_selection/assemble_and_fold.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from pyproj import Transformer
from sklearn.cluster import KMeans

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "trust"))
import common  # noqa: E402

EXP = common.REPO / "experiments/agb_ireland_training_selection_20260626"
ANALYSIS_DIR = EXP / "analysis"
DATA_OUT = common.DATASPACE / "agb_ireland_training_selection_20260626"
RANKING = DATA_OUT / "analysis/project_ranking.parquet"

# gap cut: a step in median DI-to-Ireland below this is "flat shelf" (can't discriminate).
FLAT_GAP = 0.10
MIN_CORE, MAX_CORE = 3, 8
# anti-overfit floors for the (robustness) extended alternative.
EXT_MIN_PROJECTS = 8
EXT_MIN_PLOTS = 2000
N_SPATIAL_FOLDS = 5


def gap_cut(di: np.ndarray) -> int:
    """Contiguous top of a sorted-ascending DI curve while each step clears FLAT_GAP."""
    di = np.asarray(di, dtype=float)
    k = 1
    while k < len(di) and (di[k] - di[k - 1]) >= FLAT_GAP:
        k += 1
    return int(np.clip(k, MIN_CORE, min(MAX_CORE, len(di))))


def project_centroids(projects: list[str]) -> pd.DataFrame:
    canon = common.load_canonical()
    g = canon[canon.project_name.isin(projects)].groupby("project_name")[["lon", "lat"]].mean()
    tf = Transformer.from_crs("EPSG:4326", "EPSG:5070", always_xy=True)
    xs, ys = tf.transform(g["lon"].to_numpy(), g["lat"].to_numpy())
    return pd.DataFrame({"project_name": g.index, "x_m": xs, "y_m": ys}).reset_index(drop=True)


def spatial_folds(projects: list[str], k: int) -> dict[str, int]:
    """Grouped spatial K-fold: cluster project centroids (EPSG:5070) so projects stay intact."""
    cen = project_centroids(projects)
    k = min(k, len(projects))
    km = KMeans(n_clusters=k, random_state=42, n_init=10).fit(cen[["x_m", "y_m"]].to_numpy())
    return dict(zip(cen["project_name"], km.labels_.astype(int)))


def fold_composition(projects: list[str], fold_map: dict[str, int], fold_col: str) -> pd.DataFrame:
    canon = common.load_canonical()
    sel = canon[canon.project_name.isin(projects)].copy()
    sel[fold_col] = sel["project_name"].map(fold_map)
    return (
        sel.groupby(fold_col)
        .agg(
            projects=("project_name", "nunique"),
            plots=("project_name", "size"),
            co2_med=("CO2", "median"),
            co2_min=("CO2", "min"),
            co2_max=("CO2", "max"),
        )
        .reset_index()
    )


def main() -> None:
    (DATA_OUT / "preprocessing").mkdir(parents=True, exist_ok=True)
    (DATA_OUT / "final").mkdir(parents=True, exist_ok=True)
    rank = pd.read_parquet(RANKING)
    eligible = rank.loc[~rank["co2_flag"]].reset_index(drop=True)  # erroneous dropped
    di = eligible["median_di_to_ireland"].to_numpy()

    # --- core (recommended): gap cut ---
    k_core = gap_cut(di)
    core = eligible["project_name"].head(k_core).tolist()
    n_core_plots = int(eligible["n"].head(k_core).sum())

    # --- extended (robustness alternative): pad down the shelf to the floors ---
    k_ext = max(k_core, EXT_MIN_PROJECTS)
    while k_ext < len(eligible) and int(eligible["n"].head(k_ext).sum()) < EXT_MIN_PLOTS:
        k_ext += 1
    extended = eligible["project_name"].head(k_ext).tolist()

    all_minus_err = eligible["project_name"].tolist()

    # --- CV folds (whole projects intact) ---
    core_lopo = {p: i for i, p in enumerate(core)}  # core's natural scheme = LOPO
    ext_spatial = spatial_folds(extended, N_SPATIAL_FOLDS)

    # --- manifest: one row per project (all 52) ---
    man = rank.copy()
    man["erroneous_excluded"] = man["co2_flag"]
    man["in_core"] = man["project_name"].isin(core)
    man["in_extended"] = man["project_name"].isin(extended)
    man["in_all_minus_err"] = man["project_name"].isin(all_minus_err)
    man["core_lopo_fold"] = man["project_name"].map(core_lopo).fillna(-1).astype(int)
    man["extended_spatial_fold"] = man["project_name"].map(ext_spatial).fillna(-1).astype(int)
    man = man.sort_values(["in_core", "median_di_to_ireland"], ascending=[False, True])
    man.to_parquet(DATA_OUT / "preprocessing/selected_projects.parquet", index=False)

    ext_tab = fold_composition(extended, ext_spatial, "extended_spatial_fold")
    core_tab = eligible.head(k_core)[["project_name", "biome", "n", "median_di_to_ireland"]]
    core_di = di[:k_core]
    shelf_di = di[k_core:]
    n_core_biomes = man.loc[man["in_core"], "biome"].nunique()

    # --- provenance ---
    schema = {
        "schema_version": "ireland-training-selection-v0",
        "encoding": "codec",
        "feature_space_for_selection": "embonly-64",
        "id_columns": ["project_name", "plot_id"],
        "cv_partition_key": "project_name",
        "selection_metric": "Ireland-anchored importance-weighted CAST DI (emb-only)",
        "selection_rule": f"gap cut (FLAT_GAP={FLAT_GAP}) on median DI to Ireland",
        "recommended_set": "in_core",
        "n_core_projects": len(core),
        "n_core_plots": n_core_plots,
        "n_extended_projects": len(extended),
        "n_extended_plots": int(eligible["n"].head(k_ext).sum()),
        "n_extended_spatial_folds": int(ext_tab.shape[0]),
        "erroneous_excluded": man.loc[man["erroneous_excluded"], "project_name"].tolist(),
        "candidate_sets": {
            "core": core,
            "extended": extended,
            "all_minus_err": all_minus_err,
        },
        "cv": {
            "core": "leave-one-project-out (core_lopo_fold)",
            "extended": "grouped spatial K-fold (extended_spatial_fold)",
        },
    }
    (DATA_OUT / "preprocessing/feature_schema.json").write_text(json.dumps(schema, indent=2))
    (DATA_OUT / "preprocessing/data_version.txt").write_text(
        "ireland-training-selection-v0\n"
        f"source_ranking={RANKING}\n"
        f"canonical_store={common.CANONICAL}\n"
        "encoding=codec; selection_space=embonly-64; partition=project_name\n"
    )
    (DATA_OUT / "final/DATA_STORE.md").write_text(
        "# Ireland training-selection outputs\n\n"
        "- `analysis/project_ranking.parquet` — all 52 projects ranked by DI to Ireland.\n"
        "- `preprocessing/selected_projects.parquet` — manifest: set membership + fold ids.\n"
        "- `preprocessing/feature_schema.json`, `data_version.txt` — provenance (encoding=codec).\n"
    )

    # --- append to the Part 1 report ---
    md = ["\n## Part 2 — Assembly + CV folds\n"]
    md.append(
        f"Gap cut (step ≥ {FLAT_GAP} in median DI) → **core = {len(core)} projects, {n_core_plots} plots**, "
        f"{n_core_biomes} biome(s). Core DI {core_di.min():.2f}–{core_di.max():.2f} sits clearly below the "
        f"flat shelf ({shelf_di.min():.2f}–{shelf_di.max():.2f}), where ~{len(shelf_di)} projects of all "
        "biomes are equidistant from Ireland and cannot be told apart.\n"
    )
    md.append("\n### Recommended set — core (closest, gap-defined)\n")
    md.append(core_tab.to_markdown(index=False, floatfmt=".2f") + "\n")
    md.append("\n### Candidate sets for the next-phase bake-off (nested)\n")
    md.append(
        f"- **core** ({len(core)}): gap-defined closest cluster. Recommended; CV = leave-one-project-out.\n"
    )
    md.append(
        f"- **extended** ({len(extended)}, {int(eligible['n'].head(k_ext).sum())} plots): core padded down "
        f"the shelf to anti-overfit floors (≥{EXT_MIN_PROJECTS} projects, ≥{EXT_MIN_PLOTS} plots). Tests "
        "whether extra plots/variation beat staying closest. CV = grouped spatial K-fold.\n"
    )
    md.append(f"- **all_minus_err** ({len(all_minus_err)}): all eligible projects (baseline).\n")
    md.append("\n### Spatial CV folds — extended set (whole projects intact)\n")
    md.append(ext_tab.to_markdown(index=False, floatfmt=".1f") + "\n")
    md.append(
        "\nManifest `preprocessing/selected_projects.parquet`: `in_core` / `in_extended` / "
        "`in_all_minus_err` + `core_lopo_fold` + `extended_spatial_fold`.\n"
    )
    with (ANALYSIS_DIR / "feature_space.md").open("a") as fh:
        fh.write("\n".join(md))

    # --- verification asserts ---
    assert MIN_CORE <= len(core) <= MAX_CORE, "core size outside bounds"
    assert man.loc[man["in_core"], "core_lopo_fold"].nunique() == len(core), (
        "core LOPO folds not 1:1"
    )
    assert man.loc[man["in_extended"], "extended_spatial_fold"].ge(0).all(), (
        "extended project missing a fold"
    )
    assert "Quinte" in schema["erroneous_excluded"], "Quinte not excluded"

    print(f"core ({len(core)} proj / {n_core_plots} plots / {n_core_biomes} biomes): {core}")
    print(
        f"  core DI {core_di.min():.2f}-{core_di.max():.2f} vs shelf {shelf_di.min():.2f}-{shelf_di.max():.2f}"
    )
    print(
        f"extended ({len(extended)} proj / {int(eligible['n'].head(k_ext).sum())} plots): {extended}"
    )
    print(f"\nextended spatial folds ({ext_tab.shape[0]}):")
    print(ext_tab.to_string(index=False))
    print(f"\nSaved manifest -> {DATA_OUT / 'preprocessing/selected_projects.parquet'}")


if __name__ == "__main__":
    main()
