# Iteration 1 — Embedding-space analog-subset selection (Tier-1 results)

Analog Selection Actor, seed 42. Native-float 64-D AEF space
(`GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`), no model training. Inputs:
`preprocessing/iter1_pool_embeddings.parquet` (12,837 ANEW plots / 52 projects + 141 Ireland
Locations). Script: `scripts/iter1_analog_selection.py`.
Run: `uv run --with pyyaml --with umap-learn python scripts/iter1_analog_selection.py`.

## Headline

**No US (ANEW) subset materially reduces Irish OOD.** All four candidate subsets (S0-S3)
keep **100% of the 141 Irish Locations beyond their own 99th-percentile Mahalanobis radius**
and a **domain-classifier AUC of ~1.0**, indistinguishable from the iter0 full-CONUS baseline
(100% beyond / AUC 0.999998). The Irish embedding cloud sits entirely outside the US manifold
regardless of how aggressively we subset by similarity or ecoregion. This is the design's
**ESCALATE** case: "no true US analog exists -> need in-region data / SAR-CHM lever". **STOP
before Tier-2 training** — none of the hard subsets is worth a retrain on its own.

## 1. Triangulated similarity measures (native-float 64-D)

All four similarity routes were computed:
- **Domain-classifier Ireland-likeness** — HistGBM, 5-fold `cross_val_predict`, seed 42.
  Full-pool USA-vs-Ireland **AUC = 1.000000** (perfect separation). Per-ANEW-plot P(Ireland-like)
  is essentially 0 for every plot (max ~2.5e-5) — the classifier finds the domains trivially
  separable, so the soft analog score is near-degenerate (a symptom, not a defect).
- **Mahalanobis to the Irish distribution** (Irish mean + Ledoit-Wolf shrunk covariance, to
  regularise n=141 vs 64-dim). Used as the primary per-plot analog score.
- **k-NN distance to the Irish manifold** (mean distance to k=5 nearest Irish embeddings).
- **Per-project MMD (RBF, median-heuristic bandwidth) and energy distance** vs Ireland.

### Project-level rank correlations (Spearman)

| pair | rho |
|---|---:|
| MMD vs energy | **0.998** |
| MMD vs kNN | 0.831 |
| energy vs kNN | 0.821 |
| kNN vs Mahalanobis | 0.696 |
| Mahalanobis vs MMD | 0.462 |
| Mahalanobis vs P(Ireland) | 0.138 |
| MMD vs P(Ireland) | 0.474 |

MMD/energy/kNN agree strongly and define a stable similarity ordering. Mahalanobis (mean-centred,
single Gaussian) and the classifier probability agree less — expected, because with AUC≈1.0 the
classifier probabilities collapse to ~0 and carry little ranking signal. The distribution-level
measures (MMD/energy/kNN) are therefore the trustworthy ranking; they are used to define S2 and,
blended with Mahalanobis, S3.

### Top ANEW projects most similar to Ireland (by MMD, lowest = most similar)

| rank | project | n | MMD² | energy | mean Mahal→IE | CO2 median | CO2 max | ecoregion |
|---:|---|---:|---:|---:|---:|---:|---:|---|
| 1 | **Kootznoowoo** | 166 | 0.475 | 0.875 | 44.7 | 137 | 591 | Northern Pacific Alaskan coastal forests |
| 2 | **RainierGateway** | 204 | 0.648 | 1.154 | 56.4 | 244 | 1011 | Central-Southern Cascades Forests |
| 3 | **FundyBay** | 201 | 0.660 | 1.179 | 77.2 | 141 | 571 | New England-Acadian forests |
| 4 | EagleMountain | 144 | 0.707 | 1.252 | 75.8 | 137 | 420 | New England-Acadian forests |
| 5 | Soterra | 286 | 0.713 | 1.258 | 85.9 | 115 | 468 | Southeast US conifer savannas |
| 6 | Cassidy | 343 | 0.731 | 1.293 | 82.6 | 112 | 347 | New England-Acadian forests |
| 7 | Manistique | 164 | 0.756 | 1.335 | 90.3 | 73 | 271 | Western Great Lakes forests |
| 8 | 100MileWilderness | 204 | 0.756 | 1.346 | 82.8 | 76 | 268 | New England-Acadian forests |

The most Ireland-like US projects are the **Alaskan/Pacific coastal-maritime conifer
(Kootznoowoo, RainierGateway)** and **New England-Acadian maritime (FundyBay, EagleMountain,
Cassidy, 100MileWilderness)** clusters — exactly the climate analogues the S1 heuristic targets,
which is a reassuring cross-check. But note: even Kootznoowoo's *mean* Mahalanobis-to-Ireland is
**~45**, versus a within-Ireland scale where the closest single Irish Location is ~21 from the
S3 subset centroid. Relative ordering exists; absolute proximity does not.

## 2. Candidate subsets

Irish-relevant biomass band (research §1): rotation-end ~150-376 Mg/ha × 0.6977 ⇒ high-relevant
band **[105, 262] tCO2/acre** (saturation-onset → 45-yr rotation-end), full plausible span
**[0, 262]** (young restock → rotation-end). All subsets comfortably span this band; **biomass
coverage is NOT the binding constraint** — similarity is.

| subset | definition | n_plots | n_proj | CO2 min/med/max | high-band decile coverage |
|---|---|---:|---:|---|---:|
| **S0_full** | all ANEW (baseline) | 12,837 | 52 | 0 / 117 / 1262 | 10/10 |
| **S1_climate_ecoregion** | New England-Acadian + Pacific/Cascades/Alaskan conifer (ECO_NAME + HighCascades/RainierGateway/Kootznoowoo/Doyon/LongviewRanch) | 3,275 | 15 | 0 / 100 / 1011 | 10/10 |
| **S2_project_nearest** | top-k=8 projects by MMD to Ireland | 1,712 | 8 | 0 / 119 / 1011 | 10/10 |
| **S3_plot_coverage** | most-similar plots (blended Mahal+kNN−P z-score), coverage-constrained: top 30% within each decile of [0,262] | 3,540 | 48 | 0 / 108 / 262 | 10/10 |

- **k=8 for S2** chosen by growing the MMD ranking until the subset spans the high band
  (co2_max ≥ 262 and a project median ≥ 105) AND ≥5 projects remain for LOPO. k=8 satisfies
  both with margin (8 projects, co2_max 1011). S2 projects: Kootznoowoo, RainierGateway,
  FundyBay, EagleMountain, Soterra, Cassidy, Manistique, 100MileWilderness.
- **S3 selection rule:** rank ANEW plots by a composite similarity score
  `z(Mahal→IE) + z(kNN→IE) − z(P_Ireland)` (lower = more similar); within each decile of the
  Irish-relevant band [0,262] keep the 30% most-similar plots. This deliberately prevents the
  similarity filter from truncating the biomass range — and it works (full 10/10 decile
  coverage, co2 up to the band ceiling), so S3 is the cleanest similarity-vs-coverage trade.
- **S4 importance weights:** per-plot density-ratio weights `P/(1−P)` over the full pool,
  top-1% clipped, mean-normalised to 1 — saved (no hard cut) in
  `preprocessing/iter1_similarity_scores.parquet` (`importance_weight`). Because P(Ireland-like)
  ≈ 0 for every US plot, the weights are near-uniform (the density ratio carries almost no
  Ireland signal) — S4 cannot rescue the transfer either.

## 3. Tier-1 OOD diagnostics (Ireland vs each subset)

| subset | n | proj | subset 99pct radius | Irish Mahal min / med / max | **frac beyond 99pct** | **domain AUC** | high-band cov |
|---|---:|---:|---:|---|---:|---:|---:|
| **iter0 baseline (CONUS train)** | 4,636 | — | 14.79 | 27.80 / 32.42 / 38.86 | **1.0000** | **0.999998** | — |
| S0_full | 12,837 | 52 | 15.17 | 21.68 / 27.06 / 35.80 | **1.0000** | **1.000000** | 10/10 |
| S1_climate_ecoregion | 3,275 | 15 | 13.58 | 26.80 / 32.85 / 43.44 | **1.0000** | **1.000000** | 10/10 |
| S2_project_nearest | 1,712 | 8 | 13.07 | 21.48 / 26.53 / 35.26 | **1.0000** | **0.999983** | 10/10 |
| S3_plot_coverage | 3,540 | 48 | 13.59 | 20.59 / 25.71 / 34.40 | **1.0000** | **0.999992** | 10/10 |

Figures (`evaluation/figures/`): `iter1_pca_overlap_subsets.png`,
`iter1_umap_overlap_subsets.png`, `iter1_project_mmd_ranking.png`. The PCA/UMAP overlays show the
Irish cloud as a compact island fully **disjoint** from every US subset — no boundary overlap.

**Interpretation of the numbers.** The most-similar subsets (S2, S3) do pull the *minimum* Irish
Mahalanobis down slightly (≈21 vs the iter0 baseline ≈28) and shave the domain AUC by a few
parts in 10⁵ — a faint, real signal that S2/S3 select genuinely-closer US plots. But the
subset 99th-pct radius shrinks in lockstep (≈13 vs ≈15) because a tighter subset has a tighter
own-distribution, so **every single Irish Location remains beyond it: fraction-beyond stays
pinned at 1.0000 and AUC stays ~1.0 for all four subsets.** Biomass coverage holds perfectly
across all subsets (10/10 deciles of the Irish band), confirming coverage is not the limiting
factor — the US embedding manifold simply does not reach Ireland.

## 4. Decision

**ESCALATE — no adequate US analog; STOP before Tier-2.**

Against the design decision rule: Tier-1 OOD is **not** materially reduced by any hard subset
(frac-beyond 1.0, AUC ~1.0 across S0-S3, matching the iter0 baseline), and the importance-weight
lever (S4) is degenerate because P(Ireland-like)≈0 pool-wide. Biomass coverage was never the
problem. Per design §"ESCALATE", the best subset still leaves Ireland far outside the manifold,
so there is **no true US analog** to retrain on. Training a LightGBM head on S2/S3 would only
re-fit US optical/embedding structure that the Irish maritime-plantation distribution does not
share — it would not deliver calibrated Irish absolute biomass.

**Recommendation to the Improvement Planner:**
1. Do **not** proceed to Tier-2 subset retraining on S0-S3 (would not reduce OOD).
2. Escalate the **in-region data / SAR-CHM lever**: the absolute-level transfer needs either
   (a) Irish (or maritime-plantation) labelled plots to anchor a fine-tune/calibration, or
   (b) a structural height channel (SAR/Sentinel-1, GEDI/CHM) that the embedding-only head drops
   — the documented strongest lever against optical saturation (research §3).
3. If any subset is carried forward at all, it should be as an **ordering prior** (S2/S3 are the
   least-OOD US plots and best biomass-coverage candidates) for a domain-adaptation or
   calibration approach, **not** as a standalone retraining set.

## Artefacts

- `preprocessing/iter1_similarity_scores.parquet` — per-ANEW-plot P(Ireland-like),
  Mahalanobis-to-Ireland, kNN-to-Ireland, importance_weight.
- `configs/iter1_subset_membership.json` — plot row_ids + project lists per subset (S0-S3),
  S4 weights note.
- `evaluation/iter1_tier1_ood.yaml` — machine-readable diagnostics + rank correlations.
- `evaluation/figures/iter1_*.png` — PCA/UMAP overlap, project MMD ranking.
- `preprocessing/_iter1_project_ranking.csv` — full 52-project similarity table.

### Assumptions / reproducibility
- seed 42 throughout; Ledoit-Wolf shrinkage for the Irish covariance (n=141 < 64²); RBF-MMD
  bandwidth = median heuristic on a 2,000-point subsample; importance weights top-1% clipped.
- Irish band [105, 262] tCO2/acre from research §1 (376 Mg/ha rotation-end × 0.6977).
- pyyaml + umap-learn supplied via `uv run --with` (not in base env).
