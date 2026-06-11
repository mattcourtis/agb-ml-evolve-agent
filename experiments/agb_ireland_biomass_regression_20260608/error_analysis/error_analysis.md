# Ireland AGB — Error / Divergence Analysis (embdstx head vs Deep Biomass)

**Stage:** error_analysis (single atomic stage). **Mode:** zero-shot transfer of the pre-trained
`embdstx` head to 141 Irish Dasos Locations, characterised against the Deep Biomass (DB) reference.
**There is NO ground truth.** Everything below is *divergence / behaviour* analysis. DB is a known
under-estimator used as a directional lower bound only; "error" never means error-vs-truth.

> **Missing reference note.** The required gate file
> `/home/mattc/.claude/skills/biomass-ml-agent-evolve/references/error_analysis.md` does not exist
> (the entire `.../biomass-ml-agent-evolve/references/` directory is absent — the same absence the
> experiment_design and data_profile actors flagged). I proceed using the conventions in the ACCEPTED
> upstream artefacts (`configs/experiment_design.md` §7 decision rule, `evaluation/bias_characterisation.md`)
> and flag the absence here for the Critic.

All new computation: `uv run python`, seed 42, on saved artefacts
`evaluation/ireland_predictions.parquet`, `preprocessing/ireland_features.parquet` (64 `emb_*`),
`data_profile/train_emb_sample.parquet` (4636 valid US training plots after dropping 10 NaN rows).
Mahalanobis uses the US training covariance in the 64-embedding space (matches the upstream 99th-pct
radius 14.79 exactly, confirming reproduction). Diagnostics merged to `error_analysis/_merged_diag.parquet`.

---

## 0. Verdict up front

- **WHY H2 FAILED: the comparison cannot reveal saturation resistance — it is a measurement artefact,
  NOT our head saturating.** Within the top DB quintile our pred still rank-tracks age (ρ=0.57) and
  height (ρ=0.42) and rises to 138.4 tCO₂/acre; the head does not plateau. The gap *shrinks* at high DB
  purely because the quintiles are cut on DB itself, so DB's own spread is mechanically maximised across
  them (DB Q1→Q5 +31 pts vs pred +24 pts), and because DB magnitude is a poor proxy for true biomass
  (DB vs age ρ=0.11 ns, DB vs Hdom ρ=0.20). H2 as designed tests a quantity the data cannot expose.
- **OOD consequence:** every Location is deep in extrapolation (min Mahalanobis 27.8 = **1.9× the
  training 99th-pct radius**; train median 7.3). Trust **rankings/structure (H3), not absolute levels.**
  Critically, divergence does **not** grow with OOD distance (|delta| vs Mahalanobis ρ=−0.21), so the
  large pred−DB gap is a *level* offset reflecting DB under-reading, not OOD noise injection.
- **Top recommendation / weakest stage:** the earliest responsible stage is **model selection** — the
  `embdstx` head has no vertical-structure lever (optical AEF embeddings only, no CHM/SAR), so it cannot
  prove saturation resistance and its top-end absolute level is unanchored. Primary recommendation:
  **analog-subset retrain (maritime + high-biomass conifer)** to pull the Irish manifold inside the
  training support, plus add a structural lever (SAR/CHM) if pursuing absolute levels. This confirms the
  upstream **RETRAIN_WARRANTED** decision.

---

## 1. Divergence structure (pred − DB delta)

Reproduced headline: pred mean 91.6 vs DB 27.3 tCO₂/acre (3.35×); signed Δ (2020–24) mean +64.3,
98.6% of Locations pred ≥ DB. The structure of *where* it diverges:

**By DB-magnitude quintile** (the H2 cut), with the structural covariates added:

| DB quintile | DB mean | pred mean | signed Δ | age (yr) | Hdom (m) |
|---|---|---|---|---|---|
| Q1 (lowest DB) | 14.4 | 78.3 | +63.9 | 15.7 | 8.7 |
| Q2 | 19.9 | 94.4 | +74.6 | 18.2 | 10.7 |
| Q3 | 25.5 | 89.7 | +64.2 | 16.0 | 10.1 |
| Q4 | 32.1 | 94.0 | +61.9 | 17.9 | 11.1 |
| Q5 (highest DB) | 45.3 | 102.1 | +56.9 | 19.4 | 12.2 |

**Mechanism (the systematic patterns).**
- **H1 dominance** holds because DB sits flat at ~14–45 tCO₂/acre regardless of stand structure while
  our head responds to it. DB rises only weakly with structure (DB vs age ρ=0.11 ns; DB vs Hdom ρ=0.20),
  i.e. DB is essentially a near-constant low band — exactly the "flat ~20–40 regardless of structure"
  behaviour expected of a saturating under-estimator.
- **Our head reads structure** strongly: pred vs age ρ=0.55, pred vs Hdom ρ=0.56 (both p<1e-12),
  monotone across age and Hdom bins (H3). This is the strongest evidence the transfer is signal, not noise.
- **H2 non-monotone with Q5-lowest gap** is a direct consequence of (i) and (ii): see §2.

**Which Locations diverge most/least.** Only **2 Locations have pred < DB** (negative Δ): *Rathcahill
West* (pred 30.7, DB 35.8, age 2, Hdom 0 — a near-newly-planted stand where DB's floor sits above our
near-zero-structure prediction) and *Bunrevagh* (pred 27.5, DB 35.8, age 4, Hdom 4.8). Both are very
young / low-structure — exactly where our head correctly predicts low biomass and DB's noisy floor
happens to exceed it. The largest Δ Locations are the broadleaf/older-Sitka stands (delta_mean +85 for
broadleaf, +81 for age 30–40). Divergence is *smallest* in the youngest stands and at the top of the DB
range, both for principled reasons (low structure on our side; DB-self-quintiling on the other).

Figure: `figures/absdelta_vs_mahalanobis.png` (colour = pred level).

---

## 2. Why H2 failed — the key analytical question

The design (experiment_design §1, H2) expected the signed gap to **widen** Q1→Q5 because DB compounds
optical saturation at high biomass while our head, carrying a disturbance-timing age proxy, was expected
to under-read less. Observed: gap is flat-to-declining, **Q5 lowest** (56.9). Three candidate causes:

- **(a) Our head compresses at the top (optical saturation in the embeddings; no CHM lever).**
  **REJECTED as the driver.** Within Q5 (n=28) our pred still rank-tracks age (ρ=0.57) and Hdom (ρ=0.42)
  — i.e. it keeps discriminating exactly where a saturated model would flatten. Among the 99 oldest
  stands (age≥15) pred mean is 103 and reaches 138.4; among the 28 tallest (Hdom≥15) pred mean 108,
  max 138.4. There is **no plateau**: pred max 138.4 is only **26.6% of the training range** (520.95)
  and only 1% of Locations exceed 130. Against the research envelope (a 19-yr Irish Sitka stand ≈110
  tCO₂/acre; rotation-end ≈262), our top values are *plausible-to-mildly-low*, not collapsed. So some
  optical compression at the genuine high tail cannot be ruled out (the head is optical-only), but it is
  demonstrably **not** what produces the Q5-lowest gap. Figures:
  `figures/pred_vs_age_by_dbquintile.png`, `figures/topquintile_plateau_check.png`.

- **(b) DB's own bias shape — DB rises in its top quintile so the gap shrinks.** **This is the dominant
  mechanism, and it is partly tautological.** The quintiles are cut *on DB magnitude*, so DB's spread is
  maximised across them by construction: DB climbs +30.8 pts (14.4→45.3) Q1→Q5 while pred climbs only
  +23.8 pts (78.3→102.1). When the denominator series (DB) rises faster than the numerator series (pred)
  *because you sorted on the denominator*, the signed gap pred−DB must compress at the top. The Q5 gap
  being smallest is therefore an artefact of the cut, not evidence about saturation resistance.

- **(c) DB-magnitude quintiles are a poor proxy for true biomass.** **Confirmed, and this is why H2 is
  uninformative.** DB barely tracks the independent structural covariates (DB vs age ρ=0.11 ns; DB vs
  Hdom ρ=0.20), and the Q5 stands are only modestly older/taller than Q1 (age 19.4 vs 15.7; Hdom 12.2 vs
  8.7). So "high DB quintile" ≠ "high true biomass"; it largely captures DB's own noise. A test that
  bins by DB magnitude cannot isolate the high-true-biomass stands where saturation resistance would
  manifest.

**Conclusion.** H2 fails for reasons (b)+(c): **the comparison cannot reveal saturation resistance**,
not because our head saturates (a). "Our model saturates" is *not* demonstrated; "the DB-quintile
comparison is structurally unable to show saturation resistance" *is*. A truth-free test that *could*
expose saturation would bin by an **independent structural axis** (age / Hdom), and on that axis our
head keeps rising while DB stays flat — the opposite of co-saturation (counter-hypothesis C1 is not
supported). The honest statement: we cannot prove added saturation resistance, but we can rule out the
co-saturation failure mode.

---

## 3. OOD consequences

- **How far out.** Irish Mahalanobis (64-emb, US-train covariance): min 27.8 / median 32.4 / max 38.9.
  The training 99th-pct radius is 14.79 and the training median is 7.3. The **closest** Irish Location is
  **1.9× the 99th-pct radius** and ~3.8× the training median; 100% of Locations are beyond the 99th pct.
  This is not marginal extrapolation — the entire portfolio sits in a region with effectively zero
  training support. Domain-classifier AUC ≈1.0 corroborates (USA vs Ireland trivially separable).
- **What it means for trust.** Absolute tCO₂/acre levels are unanchored extrapolation and should not be
  reported as calibrated. **Rankings/structure are the trustworthy signal** (H3: pred tracks age/Hdom at
  ρ≈0.55). Reassuringly, **divergence does not grow with OOD distance**: Spearman |delta| vs Mahalanobis
  = **−0.21** (p=0.011) and pred vs Mahalanobis = −0.36 — if anything the most-OOD Locations have
  *smaller* gaps and slightly *lower* preds. So the pred−DB offset is a coherent level shift (DB
  under-reading + genuine structure), not OOD-driven garbage (counter-hypothesis C2 is the *severity*
  caveat, but the predictions still behave sensibly — they are not noise).
- **Which embedding dimensions drive the shift** (per-dim contribution to the squared Mahalanobis of the
  Irish centroid; cheap diagonal-of-the-quadratic-form attribution):

  | emb dim | Mahalanobis contribution | raw mean shift (train-SD units) |
  |---|---|---|
  | emb_26 | +161 | +4.7 |
  | emb_50 | +153 | −6.3 |
  | emb_23 | +92 | +3.6 |
  | emb_55 | +72 | −4.6 |
  | emb_18 | +56 | +1.5 |
  | emb_53 | +53 | −3.1 |
  | emb_58 | +49 | −2.3 |
  | emb_11 | +48 | +3.3 |

  The top 8 of 64 dimensions account for **68%** of the centroid OOD shift. A handful of AEF dimensions
  (notably emb_26, emb_50, emb_23) carry the maritime-temperate-plantation signature that has no US
  analog — these are the concrete targets for an analog-subset that would shrink the gap. Figure:
  `figures/perdim_ood_contribution.png`.

---

## 4. The 17 pre-2017 fallback Locations

17 / 141 Locations (surveyed 2015–2016) use the AEF year clamped to 2017. They are **not**
over-represented among large divergences or anomalies:

- |Δ| mean for fallback 60.1 vs non-fallback 65.1 (fallback slightly *lower*).
- Only **2 of the top-20 |Δ|** Locations are fallback (base rate 17/141 = 12.1%) — i.e. under-represented.
- Mahalanobis and pred levels are indistinguishable from the rest (fallback pred mean 84 vs 93).
- Of the 2 negative-Δ anomalies, 1 is fallback (Rathcahill West) — but that is driven by its near-zero
  stand structure, not the temporal clamp.

Conclusion: the 2017 fallback is a minor temporal misalignment with **no detectable distortion** of the
divergence structure; it does not need special handling in the improvement plan.

---

## 5. Weakest upstream stage (feeds the improvement plan)

Ranking the stages by responsibility for the limitation:

1. **Model selection (EARLIEST + PRIMARY).** The `embdstx` head is optical-AEF-only with no vertical
   structure lever (no CHM, no SAR). Two consequences flow directly from this single choice: (i) it
   *cannot* prove saturation resistance against an optical under-estimator — there is no independent
   physical signal that resists optical saturation, so H2 is unanswerable in principle for this head; and
   (ii) its high-biomass absolute level is unconstrained. This is the root cause of the H2 outcome.
2. **Training pool.** 4636 US plots with no maritime-temperate / high-biomass-plantation analogs → the
   entire Irish manifold is OOD (§3). This is what makes absolute levels untrustworthy and drives the
   severe domain shift. Second-most responsible.
3. **Evaluation design (H2 metric).** Binning by DB magnitude (a necessary truth-free choice) embeds the
   artefact of §2(b)/(c). Not "wrong" given no truth, but it means the H2 NOT_SUPPORTED verdict should be
   read as "uninformative", not "our head failed". An age/Hdom-binned divergence read (already available
   via H3) is the more honest saturation diagnostic and should be promoted.

Encoding/preprocessing is **not** implicated — the gate passed (corr 0.986, post-affine slope ≈1.0),
the Mahalanobis reproduces exactly, and the per-dim shift is a genuine geography signal, not a codec bug.

---

## 6. Concrete, prioritised recommendations (tied to RETRAIN_WARRANTED)

1. **(Highest leverage) Analog-subset retrain — maritime + high-biomass conifer.** Augment/replace the
   training pool with maritime-temperate and high-biomass plantation analogs so the Irish manifold moves
   inside training support. Target the dimensions carrying the shift (emb_26/50/23/55 etc., §3). This is
   the single action that most reduces the OOD uncertainty on absolute levels and directly executes the
   upstream RETRAIN_WARRANTED escalation (experiment_design §7b, plan §F).
2. **Add a vertical-structure lever (SAR and/or CHM) to the feature set.** This is what would let a
   future evaluation *actually* test saturation resistance (the current head structurally cannot). Pair
   with (1); it addresses the model-selection root cause.
3. **Reframe / re-cut H2 onto an independent structural axis.** Re-run the saturation diagnostic binning
   by **age and Hdom** (not DB magnitude). On the data we have, our head keeps rising on those axes while
   DB stays flat — promote this as the truth-free saturation-resistance read and retire the DB-quintile
   cut as the headline H2 metric. Cheap, immediate, no new data.
4. **Post-hoc calibration is NOT recommended as a substitute for (1).** With 100% OOD and no Irish truth
   there is nothing to calibrate against; any affine recalibration to DB would import DB's under-estimate.
   Calibration only becomes meaningful once Irish ground-truth plots exist.

The pattern — gate PASS, H1 SUPPORTED, H3 (age/Hdom) SUPPORTED, H2 uninformative-not-failed, OOD
SEVERE-but-coherent — confirms the upstream decision: **RETRAIN_WARRANTED**, escalate the analog-subset
retrain. Not "halt" (structure is tracked, predictions are coherent); not "fully credible" (absolute
levels are deep extrapolation).

---

## Assumptions, commands, seeds

- **Assumptions.** No ground truth; DB = directional lower bound (never truth). Mahalanobis uses US
  training covariance (4636 valid plots, 10 NaN rows dropped) in raw 64-emb space — same convention as
  upstream (reproduces r99=14.79). Per-dim OOD contribution = `d_j · (Σ⁻¹ d)_j` for the centroid
  difference `d` (diagonal attribution of the quadratic form; signs can offset, used for ranking only).
  DB-quintiles cut on `db_2020_24_tco2`. Age = survey_year − PlantingYe (1 missing).
- **Seed.** 42 (no stochastic step in the diagnostics; LightGBM predictions are deterministic and were
  taken as-is from the predictions parquet).
- **Commands.** `uv run python /tmp/diag.py` (Mahalanobis, per-dim shift, Spearman correlations,
  quintile/structure table, fallback analysis, figures). Inputs: `evaluation/ireland_predictions.parquet`,
  `preprocessing/ireland_features.parquet`, `data_profile/train_emb_sample.parquet`. Output diag table:
  `error_analysis/_merged_diag.parquet`.
- **Figures** (`error_analysis/figures/`): `absdelta_vs_mahalanobis.png`,
  `pred_vs_age_by_dbquintile.png`, `topquintile_plateau_check.png`, `perdim_ood_contribution.png`.

## Limitations

- No ground truth: all "divergence" is vs an under-estimator. We can rule out co-saturation and show
  structural tracking, but cannot positively *prove* high-end accuracy or saturation resistance.
- Per-dim Mahalanobis attribution is a diagonal approximation of a quadratic form (ranking aid only).
- Q5 / age≥15 / Hdom≥15 subsets are modest n (28–99); top-end statements are suggestive.
- Bayfield is in-sample to the head, so the encoding gate proves codec fidelity, not Irish accuracy
  (carried from upstream).
