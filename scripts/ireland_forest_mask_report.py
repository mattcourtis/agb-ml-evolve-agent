"""
Build the Ireland forest-mask report + validation: final/ireland_forest_mask.md.

Consumes the DW mask checkpoints/summaries (preprocessing/_dw_mask_y*) and the regenerated
outputs (final/ireland_agb_yearmatched.parquet, ireland_agb_pixel.parquet). Computes:
  - Portfolio masked vs unmasked (tCO2/acre + Mg/ha) per year + 3yr mean; masked DB ratio.
  - New prediction floor (min stand / min pixel reachable) vs old ~16 tCO2/acre.
  - Young-stand validation: masked AGB + forest_fraction for the 9 known age-0/Hdom~0 stands.
  - forest_fraction vs Dasos age_at_survey cross-tab.
  - DW sensitivity: extra pixels a Hansen lossyear==prediction-year rule would zero beyond DW
    (year-matched 2022/2023/2024); flag DW vs Dasos-age disagreements. DW-only is the applied
    mask; Hansen is sensitivity ONLY (not combined).
  - Comparison effect on young stands (our masked vs DB).

Hansen sampling reuses the per-pixel lossyear sample (same sampleRegions pattern as the DW mask).
"""

from __future__ import annotations

from pathlib import Path

import ee
import geopandas as gpd
import numpy as np
import pandas as pd

REPO = Path("/home/mattc/code/agb-ml-agent-evolve")
EXPDIR = REPO / "experiments/agb_ireland_biomass_regression_20260608"
PREP = EXPDIR / "preprocessing"
FINAL = EXPDIR / "final"
DISSOLVED = PREP / "ireland_locations_dissolved.gpkg"

YEARS = [2022, 2023, 2024]
TCO2ACRE_TO_MGHA = 1.0 / 0.6977
FOREST_THRESH = 0.5
HANSEN_ASSET = "UMD/hansen/global_forest_change_2025_v1_13"
DW_SCALE = 10
TILE_SCALE = 8
POINT_BATCH = 2500
OLD_FLOOR = 16.0  # tCO2/acre, the pre-mask low-end floor

YOUNG = [
    "Moyne",
    "Peak",
    "Tawran",
    "Carrigeeny",
    "Carrowreagh",
    "Erriblagh",
    "Carrowkeel",
    "Rathcahill West",
    "Cashel",
]


def sample_lossyear(lon, lat) -> list[int | None]:
    """Sample Hansen lossyear (1..25 = 2001..2025; 0 = no loss) at each 10 m pixel centre."""
    ly = ee.Image(HANSEN_ASSET).select("lossyear")
    out: list[int | None] = [None] * len(lon)
    for s in range(0, len(lon), POINT_BATCH):
        idx = list(range(s, min(s + POINT_BATCH, len(lon))))
        feats = [
            ee.Feature(
                ee.Geometry.Point([float(lon[i]), float(lat[i])], proj="EPSG:4326"), {"pidx": i}
            )
            for i in idx
        ]
        fc = ee.FeatureCollection(feats)
        res = ly.sampleRegions(
            collection=fc, scale=DW_SCALE, tileScale=TILE_SCALE, geometries=False
        ).getInfo()["features"]
        for feat in res:
            p = feat["properties"]
            out[int(p["pidx"])] = p.get("lossyear")
    return out


def hansen_sensitivity() -> dict:
    """For each year-matched set, count pixels DW-non-forest, Hansen-loss==Y, and union/extra."""
    ee.Initialize()
    res = {}
    per_stand_rows = []
    for y in YEARS:
        mdir = PREP / f"_dw_mask_y{y}"
        code = y - 2000
        dw_nonforest = hansen_loss_y = extra = total = 0
        for f in sorted(mdir.glob("*.parquet")):
            name = f.stem
            pix = pd.read_parquet(f)
            ly = sample_lossyear(pix["lon"].to_numpy(), pix["lat"].to_numpy())
            ly = np.array([(-1 if v is None else int(v)) for v in ly])
            dw_nf = ~pix["forest"].to_numpy()
            h_loss = ly == code  # clearfell in the prediction year
            total += len(pix)
            dw_nonforest += int(dw_nf.sum())
            hansen_loss_y += int(h_loss.sum())
            extra += int((h_loss & ~dw_nf).sum())  # Hansen would zero but DW kept
            per_stand_rows.append(
                {
                    "Location_Name": name,
                    "year": y,
                    "n_pixels": len(pix),
                    "dw_nonforest": int(dw_nf.sum()),
                    "hansen_loss_y": int(h_loss.sum()),
                    "hansen_extra_beyond_dw": int((h_loss & ~dw_nf).sum()),
                }
            )
        res[y] = {
            "total": total,
            "dw_nonforest": dw_nonforest,
            "hansen_loss_y": hansen_loss_y,
            "hansen_extra_beyond_dw": extra,
            "union": dw_nonforest + extra,
        }
    pd.DataFrame(per_stand_rows).to_parquet(PREP / "_hansen_sensitivity.parquet", index=False)
    return res


def fmt(v, d=2):
    return f"{v:.{d}f}"


def main() -> None:
    gdf = gpd.read_file(DISSOLVED).set_index("Location_Name")
    ym = pd.read_parquet(FINAL / "ireland_agb_yearmatched.parquet").set_index("Location_Name")
    px = pd.read_parquet(FINAL / "ireland_agb_pixel.parquet").set_index("Location_Name")

    L = []
    L.append("# Ireland AGB — Dynamic World forest/clearfell mask\n")
    L.append(
        "Applied a Dynamic World V1 `trees` forest mask (growing-season Apr–Sep median, "
        "threshold ≥ 0.5 = forest) to the existing 141-stand per-pixel AGB predictions. "
        "Non-forest pixels are set to 0; masked stand AGB = mean over all pixels "
        "= forest_fraction × mean(forest preds). DW year is aligned to the prediction year "
        "(year-matched set) or each stand's survey_year (clamped ≥ 2016 for the survey-year "
        "set). Unmasked columns are preserved alongside the new masked columns. DB is left "
        "unmasked (separate model).\n"
    )

    # ---- Portfolio masked vs unmasked ----
    L.append("## Portfolio: masked vs unmasked (year-matched)\n")
    L.append(
        "| Year | unmasked tCO2/acre | masked tCO2/acre | unmasked Mg/ha | masked Mg/ha | "
        "DB tCO2/acre | masked/DB | mean forest_frac |"
    )
    L.append("|---|---|---|---|---|---|---|---|")
    for y in YEARS:
        um = ym[f"our_{y}_tCO2_acre"].mean()
        m = ym[f"our_{y}_masked_tCO2_acre"].mean()
        db = ym[f"db_{y}_tCO2_acre"].mean()
        ff = ym[f"forest_frac_{y}"].mean()
        L.append(
            f"| {y} | {fmt(um)} | {fmt(m)} | {fmt(um * TCO2ACRE_TO_MGHA)} | "
            f"{fmt(m * TCO2ACRE_TO_MGHA)} | {fmt(db)} | {fmt(m / db)}x | {fmt(ff, 3)} |"
        )
    um = ym["our_mean_2022_24_tCO2_acre"].mean()
    m = ym["our_mean_2022_24_masked_tCO2_acre"].mean()
    db = ym["db_mean_2022_24_tCO2_acre"].mean()
    ffm = ym[[f"forest_frac_{y}" for y in YEARS]].mean(axis=1).mean()
    L.append(
        f"| **3yr mean** | {fmt(um)} | {fmt(m)} | {fmt(um * TCO2ACRE_TO_MGHA)} | "
        f"{fmt(m * TCO2ACRE_TO_MGHA)} | {fmt(db)} | {fmt(m / db)}x | {fmt(ffm, 3)} |"
    )
    L.append("")
    L.append(
        f"Survey-year (pixel) set: unmasked {fmt(px['pred_pixel_tCO2_acre'].mean())} → "
        f"masked {fmt(px['pred_pixel_masked_tCO2_acre'].mean())} tCO2/acre "
        f"({fmt(px['pred_pixel_masked_Mg_ha'].mean())} Mg/ha); "
        f"mean forest_fraction {fmt(px['forest_fraction'].mean(), 3)}.\n"
    )

    # ---- New floor ----
    floor_stand_ym = ym[[f"our_{y}_masked_tCO2_acre" for y in YEARS]].min().min()
    floor_stand_px = px["pred_pixel_masked_tCO2_acre"].min()
    # min reachable pixel = 0 by construction for any non-forest pixel
    n_zero_stands_ym = int(
        (ym[[f"our_{y}_masked_tCO2_acre" for y in YEARS]] == 0).any(axis=1).sum()
    )
    L.append("## Prediction floor\n")
    L.append(
        f"The old per-pixel/stand floor was ~{OLD_FLOOR:.0f} tCO2/acre (no Irish pixel "
        "predicted below it). After masking, any non-forest pixel is exactly 0, so the "
        "per-pixel minimum is now **0** tCO2/acre."
    )
    L.append(
        f"- Minimum masked stand density (year-matched): {fmt(floor_stand_ym)} tCO2/acre "
        f"(was ≥ {OLD_FLOOR:.0f})."
    )
    L.append(f"- Minimum masked stand density (survey-year): {fmt(floor_stand_px)} tCO2/acre.")
    L.append(
        f"- Stands reaching exactly 0 in at least one year (fully non-forest): "
        f"{n_zero_stands_ym}.\n"
    )

    # ---- Young-stand validation ----
    L.append("## Validation — young / clearfell stands (the key test)\n")
    L.append(
        "Known age-0 / Hdom~0 stands should drop toward ~0. Masked density is from the "
        "year-matched set at the stand's survey year where in {2022,2023,2024}, else the "
        "nearest available year; forest_fraction reported for the same year.\n"
    )
    L.append(
        "| Stand | survey_year | age_at_survey | Hdom | unmasked tCO2/acre | masked tCO2/acre "
        "| forest_fraction | DB tCO2/acre |"
    )
    L.append("|---|---|---|---|---|---|---|---|")
    for name in YOUNG:
        if name not in ym.index:
            continue
        sy = int(gdf.loc[name, "survey_year"])
        yy = sy if sy in YEARS else min(YEARS, key=lambda c: abs(c - sy))
        um = ym.loc[name, f"our_{yy}_tCO2_acre"]
        m = ym.loc[name, f"our_{yy}_masked_tCO2_acre"]
        ff = ym.loc[name, f"forest_frac_{yy}"]
        db = ym.loc[name, f"db_{yy}_tCO2_acre"]
        L.append(
            f"| {name} | {sy} (DW {yy}) | {fmt(gdf.loc[name, 'age_at_survey'], 1)} | "
            f"{fmt(gdf.loc[name, 'Hdom'], 1)} | {fmt(um)} | {fmt(m)} | {fmt(ff, 3)} | {fmt(db)} |"
        )
    L.append("")

    # ---- forest_fraction vs age cross-tab ----
    L.append("## forest_fraction vs Dasos age_at_survey (year-matched, mean 2022–24)\n")
    ffm_s = ym[[f"forest_frac_{y}" for y in YEARS]].mean(axis=1)
    age = gdf["age_at_survey"].reindex(ym.index)
    bins = [-0.01, 1, 3, 6, 10, 100]
    labels = ["0-1", "1-3", "3-6", "6-10", "10+"]
    ag = pd.cut(age, bins=bins, labels=labels)
    ct = pd.DataFrame({"age_band": ag, "forest_frac": ffm_s}).groupby("age_band", observed=True)
    L.append("| age band (yr) | n stands | mean forest_fraction | median forest_fraction |")
    L.append("|---|---|---|---|")
    for band, g in ct:
        L.append(
            f"| {band} | {len(g)} | {fmt(g['forest_frac'].mean(), 3)} | "
            f"{fmt(g['forest_frac'].median(), 3)} |"
        )
    L.append("")
    L.append(
        "Expectation: low forest_fraction for young clearfell (0–3 yr), high for established "
        "stands. "
    )
    L.append("")

    # ---- Hansen sensitivity ----
    L.append("## DW sensitivity — Hansen loss==prediction-year (sensitivity only, NOT applied)\n")
    hs = hansen_sensitivity()
    L.append(
        "| Year | total pixels | DW non-forest (applied) | Hansen loss==Y | "
        "Hansen extra beyond DW | DW∪Hansen |"
    )
    L.append("|---|---|---|---|---|---|")
    for y in YEARS:
        r = hs[y]
        L.append(
            f"| {y} | {r['total']} | {r['dw_nonforest']} | {r['hansen_loss_y']} | "
            f"{r['hansen_extra_beyond_dw']} | {r['union']} |"
        )
    L.append("")
    L.append(
        "The **applied mask is DW-only**. A Hansen `loss == prediction-year` clearfell rule "
        "would additionally zero the 'Hansen extra beyond DW' pixels (these are pixels DW "
        "read as tree but Hansen flagged as same-year loss). Hansen and DW are reported "
        "separately and NOT combined.\n"
    )

    # flag DW vs Dasos-age disagreements
    L.append("### DW vs Dasos-age disagreements (mean forest_frac 2022–24)\n")
    dis = pd.DataFrame({"forest_frac": ffm_s, "age": age, "Hdom": gdf["Hdom"].reindex(ym.index)})
    dw_forest_but_young = dis[(dis["forest_frac"] >= 0.5) & (dis["age"] <= 3)]
    dw_nonforest_but_old = dis[(dis["forest_frac"] < 0.5) & (dis["age"] >= 10)]
    L.append(
        f"- DW says forest (ff≥0.5) but Dasos age ≤ 3 yr: **{len(dw_forest_but_young)}** stands"
        + (
            ": "
            + ", ".join(
                f"{i} (ff={fmt(r.forest_frac, 2)}, age={fmt(r.age, 1)})"
                for i, r in dw_forest_but_young.iterrows()
            )
            if len(dw_forest_but_young)
            else ""
        )
    )
    L.append(
        f"- DW says non-forest (ff<0.5) but Dasos age ≥ 10 yr: **{len(dw_nonforest_but_old)}** "
        "stands"
        + (
            ": "
            + ", ".join(
                f"{i} (ff={fmt(r.forest_frac, 2)}, age={fmt(r.age, 1)})"
                for i, r in dw_nonforest_but_old.iterrows()
            )
            if len(dw_nonforest_but_old)
            else ""
        )
    )
    L.append("")

    # ---- Comparison effect ----
    L.append("## Comparison effect — masking moves young stands below DB\n")
    L.append(
        "DB also floors ~20–30 tCO2/acre and is left unmasked, so masked young stands now "
        "often read *below* DB. For the young stands above:"
    )
    below = above = 0
    for name in YOUNG:
        if name not in ym.index:
            continue
        sy = int(gdf.loc[name, "survey_year"])
        yy = sy if sy in YEARS else min(YEARS, key=lambda c: abs(c - sy))
        m = ym.loc[name, f"our_{yy}_masked_tCO2_acre"]
        db = ym.loc[name, f"db_{yy}_tCO2_acre"]
        if m < db:
            below += 1
        else:
            above += 1
    L.append(f"- masked < DB: {below} / {below + above}; masked ≥ DB: {above} / {below + above}.")
    L.append(
        "Pre-mask, every young stand read 44–60 tCO2/acre (well above DB); the mask flips "
        "the structurally-bare ones below DB.\n"
    )

    # ---- caveats ----
    L.append("## Caveats\n")
    L.append(
        "- DW `trees` is optical (saturation-correlated): can misread young plantation as "
        "non-tree (false zeros) or read clearfell-with-ground-veg as tree (missed zeros)."
    )
    L.append(
        "- The mask is binary and fixes only the *structural* zero (bare/clearfell); the "
        "in-domain regression floor for stocked-but-young stands needs the deferred retrain."
    )
    L.append("- DW gap pixels (no growing-season obs) are treated as non-forest (conservative).")
    L.append(
        "- Absolute values remain OOD; DW V1 coverage starts 2015-06 so the survey-year DW "
        "year is clamped to ≥ 2016."
    )
    L.append("")

    out = FINAL / "ireland_forest_mask.md"
    out.write_text("\n".join(L))
    print(f"Wrote {out}")
    print("\n".join(L[:40]))


if __name__ == "__main__":
    main()
