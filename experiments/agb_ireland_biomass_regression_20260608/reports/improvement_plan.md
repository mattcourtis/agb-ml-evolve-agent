# Improvement plan

Authored by Orchestrator (fast-track); follows directly from the ACCEPTED `error_analysis/error_analysis.md`
and the `RETRAIN_WARRANTED` decision in `evaluation/`. Ordered by expected uncertainty reduction.

## Diagnosis (from accepted upstream)
The zero-shot embdstx transfer is encoding-valid (gate PASS) and structurally sensible (rank-tracks
age/Hdom far better than Deep Biomass), and reads ~3.35× DB (consistent with DB under-estimation). The
two binding limitations are: (1) **severe OOD** — 100% of Irish Locations lie beyond the 99th-pct
training Mahalanobis radius (domain AUC ≈1.0), so absolute levels are extrapolation; (2) **H2 is
unprovable from this comparison** — the non-widening high-biomass gap is a DB-self-quintiling artefact,
not our head saturating. **Weakest upstream stage = model selection / training pool**: an optical-only
head with no vertical-structure lever and no maritime/plantation analogs in its training pool.

## Prioritised actions
1. **Analog-subset retrain (highest priority; the conditional path from plan v1 §F).** Rebuild the head
   on the maritime-temperate + high-biomass-conifer subset of the 52-project ANEW pool (New
   England-Acadian + Pacific-coastal/Cascades/Alaskan conifer), which spans Ireland's biomass range.
   Re-run the dual evaluation (held-out USA LOPO + zero-shot Ireland). This directly attacks the OOD gap.
2. **Add a non-optical structural lever.** L-band SAR (PALSAR-2) / Sentinel-1 — the deferred arm — to
   give the head a vertical-structure signal it currently lacks (no CHM in embdstx). Re-test whether the
   high-biomass band gains genuine discrimination once a structural feature is present.
3. **Re-cut H2 on age/Hdom, not DB magnitude.** Since DB magnitude is a poor biomass proxy (DB-vs-age
   ρ=0.11 ns), repeat the saturation-resistance test binning by stand age / dominant height (real
   structure) to obtain a saturation read that is not confounded by DB's own bias shape.
4. **Do NOT apply post-hoc calibration** to absolute levels yet. With no ground truth and severe OOD,
   1-D calibration against an under-estimator would encode DB's bias. Calibration is only sensible once
   the analog-subset head is validated against in-range USA holdout.

## Not warranted now
- Fine-tuning on Irish "labels" — none exist (DB is a reference model, not truth).
- Wall-to-wall Irish inference — premature until the analog-subset head reduces the OOD gap.

status: ACCEPTED (orchestrator-authored; consistent with ACCEPTED error_analysis + evaluation decision)
