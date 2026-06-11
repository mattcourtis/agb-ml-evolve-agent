# Iteration 1 — decision: ESCALATE (no US analog; stop before Tier-2)

## Outcome
The embedding-space analog-selection study (Tier-1) shows that **no subset of the ANEW pool reduces
Ireland's out-of-distribution gap**. The analog-subset retrain that iteration 0 recommended therefore
cannot, on its own, deliver calibrated Irish biomass. Iteration 1 stops before Tier-2 training and
escalates to a strategy change.

## Evidence (all ACCEPTED, Critic-reproduced incl. a non-circular metric)
- Most Ireland-like ANEW projects are the maritime/coastal-conifer cluster — Kootznoowoo (Alaskan
  coastal), RainierGateway (Cascades), New England-Acadian (FundyBay, EagleMountain, Cassidy,
  100MileWilderness) — agreed across MMD / energy / kNN (Spearman 0.83-0.998). Ecologically sensible.
- But every candidate subset (S0 full-CONUS, S1 climate/ecoregion heuristic, S2 project-nearest,
  S3 plot-nearest coverage-constrained) leaves **100% of the 141 Irish Locations beyond its 99th-pct
  Mahalanobis radius**, domain-classifier AUC **0.99998-1.0** — i.e. no better than the iter0 baseline.
- Non-circular check (decisive): on absolute Euclidean distance, the nearest Irish point is 0.612 from
  any subset point while subset members sit a median ~0.22-0.24 from their own neighbours — **0/141
  Irish points fall within a subset's own internal nearest-neighbour 99th-pct**. The gap is real, not an
  artefact of a shrinking subset radius.
- Biomass coverage is NOT the binding constraint: S3 covers 10/10 deciles of the Irish [105,262]
  tCO2/acre band. The problem is purely manifold location, not range.
- S4 importance weights are degenerate (all ~1.0; max ANEW P(Ireland-like) = 4.96e-3) — re-weighting the
  US pool cannot rescue the transfer either.

## Interpretation
The US ANEW embedding manifold does not reach Ireland's maritime even-aged Sitka plantation domain.
Because a tree head cannot extrapolate, retraining its leaves on the closest US analogs would still leave
Irish inputs uncalibrated. The limitation is the **training domain**, not the model family or the feature
encoding (which is validated).

## Revised improvement path (supersedes improvement_plan.md action 1)
1. **Acquire in-region labelled data** — Irish (or closely comparable maritime Sitka-plantation) field
   plots / inventory. This is now the primary lever: only in-domain labels can calibrate the levels.
   Even a modest set enables fine-tuning or domain-adaptation.
2. **Add a non-optical structural feature** (L-band PALSAR-2 / Sentinel-1, and/or a CHM) — reduces
   reliance on the optical embedding axes that are most domain-shifted, and may shrink the manifold gap.
3. **Retain S2/S3 as an ordering prior** — although they do not calibrate levels, the maritime/coastal
   conifer analogs are the best available prior for any domain-adaptation / isotonic-calibration step
   once in-region anchors exist.
4. **Do NOT** proceed with a pure US analog-subset retrain expecting calibrated Irish absolute biomass —
   this study shows it will not close the OOD gap.

## Status
- iter1_train_eval (Tier-2): SKIPPED by design rule (ESCALATE branch fired at Tier-1).
- The iteration-0 product stands: zero-shot predictions are usable for RELATIVE/ranking purposes
  (structural tracking is real) but NOT for absolute Irish carbon accounting. Unchanged by iter1.
- Next iteration is blocked on in-region data acquisition (a data dependency, not a modelling step).
