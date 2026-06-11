# Deep Research — agb_ireland biomass regression (zero-shot transfer + model-vs-model)

## Iteration 0 — Bias-characterisation anchor (Research Actor, 2026-06-08)

### Scope and mode

This is **NOT** a train-from-scratch task and **NOT** a GT-anchored accuracy task. The
existing pre-trained head `models/inference_model_embdstx.txt` (64 AEF embeddings + 3
survey-relative Hansen disturbance-timing features; target **tCO₂/acre**; training range
**[0, 520.95]**) is applied **zero-shot** to 141 Irish Sitka-spruce-dominated forestry
Locations (Dasos portfolio) and compared **directionally** against the "Deep Biomass" (DB)
model output. There is **no ground truth**; therefore **no accuracy claim** is made and the
deep_research "benchmark table → realistic/stretch R²/RMSE threshold" machinery is
**reframed** (see Actor addendum). The research deliverable is the **plausible biomass
envelope** that lets the Evaluation/Error-analysis Actors judge whether our model's
divergence from DB is in the **expected direction** (our ≥ DB, growing with biomass) and
whether DB's known under-estimation at high biomass is structurally consistent.

Unit conversion used throughout (from the approved plan, Section D):
**tCO₂/acre = (AGB Mg/ha) × 0.6977**, where 0.6977 = 0.47 (IPCC carbon fraction) ×
3.667 (CO₂/C) × 0.4047 (ha→acre). Both DB and the ANEW CO₂ training column are **AGB-only**,
so pools match. Inverse: **AGB Mg/ha = tCO₂/acre ÷ 0.6977**.

---

## 1. Plausible AGB envelope for Irish Sitka spruce plantation (the core anchor)

**Strongest single anchor — Irish Sitka chronosequence, destructive/allometric.**
Black, K. et al. (2009), studying a first-rotation Sitka spruce chronosequence on
surface-water gley soils in Ireland (*Forestry* 82(3):255, DOI 10.1093/forestry/cpp005),
report **above-ground biomass carbon rising from 1.7 Mg C/ha (pre-afforestation grassland)
to a maximum of 176.5 Mg C/ha in the 45-year-old stand** [S1]. Converting carbon → biomass
using the project's fixed **0.47** IPCC carbon fraction (the one baked into the ×0.6977
factor): **AGB ≈ 176.5 / 0.47 ≈ 376 Mg/ha at 45 yr → ≈ 262 tCO₂/acre** (376 × 0.6977).
For comparison, under a typically-cited ~0.50 conifer fraction the same carbon stock implies
**≈ 353 Mg/ha → ≈ 246 tCO₂/acre** — so the rotation-end CO₂ anchor spans **≈ 246–262
tCO₂/acre** (246 at fraction 0.50, 262 at fraction 0.47), with **0.47 / 262 tCO₂/acre as the
primary value** for consistency with the model's units. Either way a mature Irish Sitka stand
sits roughly **half-way up** the model's training range (max 520.95), not beyond it. This is
the **plausible rotation-age high-end** anchor and it is *strong* evidence (peer-reviewed,
Irish, species-exact, AGB-explicit).

**Supporting Irish allometric points (consistent, weaker individually).**
- A **19-year-old** unthinned Irish Sitka stand: total above-ground biomass carbon ≈
  **74 Mg C/ha** (≈ 157 Mg/ha AGB at 0.47; ≈ 110 tCO₂/acre), ±7% [S2]. This shows even a young
  pole-stage stand already exceeds the ~150 Mg/ha optical-saturation onset (Section 3).
- Biomass-expansion-factor (BEF) work for Irish Sitka [S3]: BEF declines from ~5.0 (very
  young) toward a near-constant ~1.4 between ages ~20–46 — i.e. by mid-rotation, standing
  volume → AGB is a roughly fixed multiplier, so high stem volume implies high AGB.

**Yield-class / volume cross-check (independent route to the same magnitude).**
- Irish Sitka national mean weighted **yield class ≈ 17 m³/ha/yr** [S5]; Irish stands are
  among the most productive globally, with YC 18–24 common and exceptional sites cited at
  "34 t/ha/yr stemwood" (effective YC 40+) [S6]. GB typical YC 14, productive 16–20 [S7,S8].
- Rotation length **35–50 yr**; commercial sawlog 40–44 yr (≈30 yr on the fastest sites)
  [S6, S8].
- Green-volume → biomass: **~1.1 m³ fresh spruce ≈ 1 tonne**; basic/dry wood density
  ≈ **0.33–0.40 t/m³** (green density higher) [S9, S10]. A mature YC 18–24 Irish stand
  carries a *standing* stem volume on the order of **400–700 m³/ha** at clearfell, which at
  ~0.35–0.40 t dry/m³ stem plus crown (BEF≈1.4) gives **stem-AGB ~150–280 t/ha → total AGB
  ~200–380 Mg/ha** — consistent with the Black et al. destructive maximum of ~353–376 Mg/ha.
  *(The 400–700 m³/ha standing-volume figure is a synthesis from YC/rotation context, not a
  single cited cell; treated as MODERATE evidence and used only as a corroborating range.)*

**Implied plausible envelope for this portfolio (in our model's units):**

| Stand stage | AGB (Mg/ha) | tCO₂/acre (×0.6977) | Evidence |
|---|---:|---:|---|
| Recent clearfell / restock (age 0–5) | ~0–10 | ~0–7 | structural / definitional |
| Young pole stage (~19 yr) | ~157 | ~110 | S2 (strong, single stand) |
| Mid-rotation (~30 yr) | ~200–280 | ~140–195 | volume/BEF synthesis (moderate) |
| **Rotation-end maximum (~45 yr, productive)** | **~353–376** | **~246–262** | **S1 (strong, Irish, AGB-explicit)** |
| Exceptional high-YC, unthinned, near-clearfell | up to ~400–450 | up to ~280–315 | upper-bound extrapolation (weak) |

**Anchor for the portfolio.** DB 2024 portfolio mean = **44.6 Mg/ha (≈ 31 tCO₂/acre)**;
DB individual stands up to ~120 Mg/ha (≈ 84 tCO₂/acre). Given the Dasos covariates (Sitka-
dominant, `PlantingYe` 1986–2026 → many stands 25–40 yr old by the 2023–24 survey,
YC context above), the **true** portfolio mean is very likely **substantially higher than
44.6 Mg/ha** — a portfolio of largely mid- to late-rotation high-YC Irish Sitka would be
expected to sit in the low-hundreds of Mg/ha on average, with the oldest/highest-YC stands
plausibly approaching the ~350 Mg/ha rotation-end anchor. **This makes DB's 44.6 mean look
like a strong under-estimate**, consistent with the user's prior. It also means a large
fraction of the portfolio likely sits **above the ~150 Mg/ha optical-saturation onset**,
which is the central transfer risk (Section 3).

---

## 2. AlphaEarth (AEF) / satellite-embedding transferability to plantation/conifer biomass

**Embeddings *can* carry a biomass signal, but with the usual saturation ceiling.**
A 2026 study assessing satellite-embedding features (AEF-style annual embeddings) for AGB
prediction in **subtropical forests** with ML found them useful but not a saturation cure —
embedding-only models still compress dynamic range at the high-biomass tail [S11]. This is
consistent with the internal USA finding (predicted-range discrimination 0.19–0.47;
systematic under-prediction of high-biomass plots) carried over from the agb_usa experiment.

**Cross-region transfer is AEF's documented weak point — directly relevant here.**
Benchmarking work on AEF for downstream tasks reports **stable cross-*year* transfer but
limited cross-*region* generalisation**: AEF embedding distributions shift markedly between
ecoregions, with little overlap except along region boundaries, and AEF-based models
*under-performed* RS-feature models in regression (yield) transfer even while competitive in
classification [S12, S13]. Cross-region irrigated-cropland mapping (Guanzhong Plain, China vs
Kansas, USA) showed strong within-region but degraded cross-region transferability under
domain shift [S12]. **Implication for this experiment:** a head trained on **eastern+western US** plots
applied to **maritime-temperate Irish plantation** is exactly the cross-region/cross-biome
transfer AEF is documented to handle *worst*. This elevates the importance of the
**encoding-consistency gate** (plan Section B) and the **OOD diagnostics** (Mahalanobis
fraction + USA-vs-Ireland domain-classifier AUC) — expect non-trivial domain shift; treat
predictions as bias-characterisation, not calibrated estimates. *(Strong evidence that the
risk exists; the magnitude for *this* specific head is unknown and must be measured, not
assumed.)*

---

## 3. Optical saturation above ~150–250 Mg/ha and whether age/disturbance proxies mitigate it

**Saturation onset ~150–200 Mg/ha for optical (Sentinel-2/Landsat-class) sensors.**
Multiple studies place optical spectral-reflectance saturation at **~150–200 Mg/ha**, with
documented **under-estimation of coniferous AGB above ~150 Mg/ha** [S14, S15]. AEF embeddings
are derived from optical (+ other) inputs and inherit this ceiling; the internal USA error
analysis put the practical optical ceiling at **~80 tCO₂/acre (≈115 Mg/ha)** for this exact
feature family. **Every plausible mid-to-late-rotation Irish Sitka stand (Section 1) is at
or above this onset**, so saturation-driven under-prediction is the expected failure mode and
the *primary reason our model may itself under-read the true high-biomass stands* — though
likely *less severely than DB* if disturbance-timing adds a stand-age signal.

**Do age / disturbance-timing proxies mitigate it? Partially.**
- Landsat-derived **disturbance-history / time-since-disturbance** features measurably improve
  forest-structure/AGB estimates by encoding successional stage that converged canopy
  reflectance cannot separate (mixed-conifer, Oregon) [S16]. This is the mechanism behind the
  `dstx_*` features in the embdstx head and the internal +0.013 R² disturbance-timing finding.
- Combining **structural + spectral** information improves conifer AGB by ~10% and performs
  best in coniferous forest [S17] — but the strongest structural lever is **canopy height
  (LiDAR/CHM)**, which the embdstx head **deliberately drops**. So saturation is **only
  partially mitigated**: disturbance-timing gives an age proxy (helps separate young from old
  where a Hansen loss is recorded) but cannot recover absolute biomass once optical saturates,
  and **Irish plantation harvest may be poorly captured by Hansen loss-year** for stands with
  no detected loss in the 2000–2024 record (older un-clearfelled stands → `dstx` near-null).
- Caveat [S16, S18]: stand-age proxies are imperfect — Hansen lossyear records *detected*
  disturbance, not true stand-replacing age; relying on it across a different management
  regime (Irish even-aged clearfell vs US) is itself a transfer assumption to flag.

---

## 4. The "Deep Biomass" reference product and why such products under-estimate high conifer stands

**No authoritative public spec for a product named "Deep Biomass" was located** via web
search (the phrase returns generic "deep-learning biomass" literature, not a named product)
[search negative result]. It is therefore treated, per the plan, as an **external proprietary
model used as a directional lower bound, not a citable benchmark**. The user's standing
characterisation — that DB **systematically under-estimates** — is the operative assumption,
and the portfolio numbers support it: a DB mean of **44.6 Mg/ha** for a Sitka-dominant, largely
mid/late-rotation Irish portfolio is far below the ~150–350 Mg/ha that Section 1 implies for
such stands.

**Why satellite/deep-learning AGB products generically under-estimate high-biomass conifer
stands (well-documented, applies to DB by class):**
1. **Optical/SAR saturation** at high biomass (Section 3) → the predictor variables stop
   increasing while true biomass keeps rising [S14, S15].
2. **Regression-to-the-mean / range compression**: models trained over a wide range
   over-predict low AGB and under-predict high AGB — the explicit signature reported in
   deep-learning AGB models ("overestimating low AGB and underestimating high AGB") [S19].
3. **GEDI/LiDAR reference under-sampling of dense canopy**: where products are GEDI-anchored,
   limited penetration in dense canopy biases reference labels low at the high tail [S19].
4. **Sparse high-biomass training coverage**: high-biomass plantation stands are rare in
   global training sets, so the learned mapping is poorly constrained there.
All four push a global/regional product *down* exactly where Irish Sitka sits (high biomass),
making the **directional expectation our-model ≥ DB, with the gap widening up the biomass
range** the central hypothesis to test in the evaluation stage.

---

## Actor addendum (reframed for no-GT transfer)

- **Realistic default threshold:** N/A — no ground truth, no accuracy target. *Reframed
  success* = (a) encoding-consistency gate passes (corr > 0.8 vs training-parquet encoding);
  (b) our per-Location predictions are **directionally ≥ DB**, with the **signed delta growing
  across DB quintiles** (Q1→Q5); (c) the delta pattern tracks structural covariates
  (older `PlantingYe`, higher `Hdom`, higher `YC` → higher predicted biomass).
- **Stretch threshold:** N/A. *Reframed stretch* = our portfolio mean lands within the
  literature-plausible band for a mid/late-rotation Irish Sitka portfolio (**~100–250 Mg/ha**,
  i.e. ~70–175 tCO₂/acre) rather than collapsing toward DB's 44.6 Mg/ha — which would indicate
  our model also saturates rather than resisting it.
- **Stop-if-unmet / escalation:** if the **encoding gate fails**, halt — no downstream number
  is trustworthy. If predictions are **not directionally ≥ DB** *and* **do not track the Dasos
  structural covariates**, escalate to the Improvement Planner to decide on the conditional
  analog-subset (maritime+conifer) retrain (plan Section F) rather than trusting the zero-shot
  output.
- **Benchmark adjustment vs anchor table:** the deep_research.md anchor table (CONUS plot-scale
  R²/RMSE targets) is **not applicable** — re-anchored from "accuracy vs truth" to "plausible
  AGB envelope + directional/structural consistency vs an under-estimating reference". The
  biomass envelope (Section 1) is the substitute anchor.
- **Access feasibility:** all literature sources are public (some paywalled abstracts; figures
  corroborated across ≥2 sources). DB has no public spec (treated as proprietary reference).
  AEF Irish extraction is the open technical risk handled by data_profile/preprocessing (GEE
  `GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL`, EPSG:2157), not by this artefact.

## Evidence strength summary

- **STRONG:** Black et al. 2009 Irish Sitka above-ground biomass C max 176.5 Mg C/ha @45 yr
  (*Forestry* 82(3), DOI verified) [S1]; optical saturation onset ~150–200 Mg/ha and conifer
  under-estimation >150 Mg/ha [S14,S15]; AEF limited cross-region transfer [S12,S13] — both
  spot-checked and resolvable (rs18071065 DOI resolves; arXiv 2601.00857 resolves; findings
  match); range-compression / low-high bias in AGB models [S19].
- **MODERATE:** 19-yr unthinned stand 74 t C/ha (Green et al. 2007, verified) [S2] and Irish
  BEF behaviour [S3,S4]; YC/rotation context [S5,S6,S7,S8]; disturbance-history and
  structural+spectral mitigation [S16,S17]; AEF-embedding biomass-prediction utility [S11]
  (Remote Sensing 18(3):436, published 2026-01-30 — confirmed extant via search; the article
  page returned HTTP 403 to direct fetch (MDPI anti-bot), so treat as VERIFIED-BY-SEARCH
  rather than full-text-checked).
- **WEAK / SYNTHESISED:** the 400–700 m³/ha standing-volume range and the exceptional
  ~400–450 Mg/ha upper bound (extrapolation, not a single cited cell); wood-density conversion
  [S9,S10].
- **NEGATIVE:** no public "Deep Biomass" product specification found.

## Sources

- [S1] Black, K., Byrne, K.A., Mencuccini, M., Tobin, B., Nieuwenhuis, M. et al. (2009)
  "Carbon stock and stock changes across a Sitka spruce chronosequence on surface-water gley
  soils", *Forestry: An International Journal of Forest Research* 82(3):255, DOI
  10.1093/forestry/cpp005 — first-rotation Irish Sitka chronosequence; above-ground biomass
  carbon 1.7 → 176.5 Mg C/ha (max at 45 yr). PRIMARY source for the rotation-end anchor.
  https://academic.oup.com/forestry/article/82/3/255/596814
  (NB: the figure is also relayed second-hand by Wellock, M.L. et al. (2014) "Changes in
  ecosystem carbon stocks in a grassland ash (*Fraxinus excelsior*) afforestation
  chronosequence in Ireland", *J. Plant Ecology* 7(5):429,
  https://academic.oup.com/jpe/article/7/5/429/939324 — but that is an ASH, not Sitka, paper
  and must NOT be used as the primary Sitka citation.)
- [S2] Green, C., Tobin, B., O'Shea, M. et al. (2007) "Above- and belowground biomass
  measurements in an unthinned stand of Sitka spruce (*Picea sitchensis* (Bong.) Carr.)",
  *Eur. J. Forest Research* 126:179–188, DOI 10.1007/s10342-005-0093-3 — 19-yr unthinned
  stand, total above-ground biomass 74 t C/ha (±7%), root:shoot 0.23.
  https://link.springer.com/article/10.1007/s10342-005-0093-3
- [S3] Tobin, B. & Nieuwenhuis, M. (2007) "Biomass expansion factors for Sitka spruce (*Picea
  sitchensis* (Bong.) Carr.) in Ireland", *Eur. J. Forest Research* 126:189–196, DOI
  10.1007/s10342-005-0105-3 — BEF declines ~5.0 (young) → ~1.4 (age ~20–46).
  https://link.springer.com/article/10.1007/s10342-005-0105-3
- [S4] "Improved estimates of biomass expansion factors for Sitka spruce in Ireland"
  (companion BEF work) — corroborates the BEF age trajectory.
  https://www.researchgate.net/publication/265402438
- [S5] "The distribution and productivity of Sitka spruce in Ireland", Irish Forestry Vol 66
  — national weighted mean YC ≈ 17. https://www.researchgate.net/publication/228461645
- [S6] Teagasc, "Part one: Thinning of high-yielding Sitka spruce" — Irish YC 18–20+, rotation
  /sawlog ages, exceptional ~34 t/ha/yr stemwood.
  https://www.teagasc.ie/news--events/daily/forestry/part-one-thinning-of-high-yielding-sitka-spruce---an-overview-and-results.php
- [S7] Forest Research, "Sitka spruce (SS)" species database — GB YC typical 14, up to 24+.
  https://www.forestresearch.gov.uk/tools-and-resources/tree-species-database/131584-sitka-spruce-ss-2/
- [S8] Napier CWST, "Rate of growth" — GB YC 14–20, rotation 35–45 yr, top height 16–23 m.
  https://blogs.napier.ac.uk/cwst/speed-of-growth/
- [S9] Thunder Said Energy, "Sitka spruce top facts" — ~1.1 m³ fresh ≈ 1 t; density context.
  https://thundersaidenergy.com/2022/01/26/sitka-spruce-top-fact/
- [S10] "Models for predicting wood density of British-grown Sitka spruce", Forestry 84(2):119
  (2011) — basic density. https://academic.oup.com/forestry/article/84/2/119/552169
- [S11] "Assessing the Utility of Satellite Embedding Features for Biomass Prediction in
  Subtropical Forests with Machine Learning", Remote Sensing (MDPI) 18(3):436, published
  2026-01-30 (Yunhe Forestry Station, Zhejiang, China; 89 plots; RF/SVR/MLPNN/GPR on AEF
  embeddings). VERIFIED-BY-SEARCH (extant; direct article fetch returned HTTP 403 MDPI
  anti-bot, not full-text checked). https://www.mdpi.com/2072-4292/18/3/436
- [S12] "Evaluating the Performance of AlphaEarth Foundation Embeddings for Irrigated Cropland
  Mapping Across Regions and Years", Remote Sensing (MDPI) 18(7):1065 — Guanzhong Plain (China)
  and Kansas (USA); strong within-region, degraded cross-region transfer. VERIFIED (DOI
  resolves). https://doi.org/10.3390/rs18071065
- [S13] "Harvesting AlphaEarth: Benchmarking the Geospatial Foundation Model for Agricultural
  Downstream Tasks", arXiv 2601.00857 (Ma et al., 2025-12-30; also Int. J. Appl. Earth Obs.
  Geoinf., ScienceDirect S1569843226001743) — competitive when trained locally but limited
  spatial transferability vs RS-based models. VERIFIED (arXiv ID resolves; findings match).
  https://arxiv.org/abs/2601.00857
- [S14] "Examining Spectral Reflectance Saturation in Landsat Imagery ... to Improve Forest AGB
  Estimation" — optical saturation ~150–200 Mg/ha.
  https://discovery.researcher.life/article/examining-spectral-reflectance-saturation-in-landsat-imagery-and-corresponding-solutions-to-improve-forest-aboveground-biomass-estimation/98ade324ba49340bbe80d575871e66b6
- [S15] "Improved estimation of AGB of regional coniferous forests integrating UAV-LiDAR,
  Sentinel-1 and Sentinel-2", Plant Methods (2023) — conifer under-estimation >150 Mg/ha.
  https://plantmethods.biomedcentral.com/articles/10.1186/s13007-023-01043-9
- [S16] Pflugmacher et al., "Using Landsat-derived disturbance history (1972–2010) to ..."
  USDA-FS — disturbance time-series improves forest-structure estimation (mixed conifer).
  https://www.fs.usda.gov/pnw/pubs/journals/pnw_2012_pflugmacher.pdf
- [S17] "Allometry-based estimation of forest AGB combining LiDAR canopy height attributes and
  optical spectral indexes", Sci. Remote Sensing (2022) — structural+spectral +~10%, best in
  conifer. https://www.sciencedirect.com/science/article/pii/S2197562022000598
- [S18] "Average Stand Age from Forest Inventory Plots Does Not Describe Historical Fire
  Regimes ... Mixed-Conifer Forests" (2016) — stand-age-proxy caveat.
  https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4873010/
- [S19] "Forest aboveground biomass estimation using GEDI and earth observation data through
  attention-based deep learning", arXiv 2311.03067 — over-predict low / under-predict high AGB;
  GEDI dense-canopy penetration limit. https://arxiv.org/pdf/2311.03067

## Reproducibility footer

- input references: web literature (peer-reviewed forestry & remote-sensing; arXiv preprints
  where noted); approved plan `plans/ireland-agb-test-v1.md`; experiment IMPLEMENTATION_PLAN.md.
- method: WebSearch + WebFetch fan-out; cross-corroboration of every numeric anchor across ≥2
  sources where possible; provenance recorded per claim with strength rating.
- key numeric anchor: Black et al. 2009 (*Forestry* 82(3):255) — Irish Sitka above-ground
  biomass C 176.5 Mg C/ha @45 yr → AGB ~376 Mg/ha at fraction 0.47 (primary) / ~353 Mg/ha at
  0.50 → ~262 tCO₂/acre (primary, 0.47) / ~246 (0.50); reported range 246–262.
- libraries: n/a (literature synthesis). seed: n/a.
- conducted by: Research Actor (automated). timestamp_utc: 2026-06-08T08:30:00Z.

## Revision log (attempt 2 — Critic-required fixes)

1. **[S1] mis-citation fixed.** The previously-cited URL (rtt060.pdf, *J. Plant Ecology*
   7(5):429) was confirmed via search to be **Wellock et al. (2014)** — a grassland-to-**ash**
   afforestation chronosequence, not Sitka and not Black 2009. The actual primary source was
   located and verified: **Black, K. et al. (2009) "Carbon stock and stock changes across a
   Sitka spruce chronosequence on surface-water gley soils", *Forestry* 82(3):255, DOI
   10.1093/forestry/cpp005** (author list and Sitka/Irish/gley species-site detail confirmed
   by WebFetch). [S1] now cites the Forestry paper as primary; the Wellock 2014 relay is noted
   parenthetically and explicitly flagged as ash/secondary so it cannot be mistaken for the
   primary. Author/year/species/venue now all agree.
2. **tCO₂/acre conversion reconciled.** The contradictory inline "≈246 tCO₂/acre (=176.5 ÷
   0.47 × 0.6977)" (which actually computes to 262) is removed. Section 1 now presents both
   fractions explicitly — 0.47 → 376 Mg/ha → 262 tCO₂/acre (PRIMARY, the project's fixed
   fraction) and 0.50 → 353 Mg/ha → 246 tCO₂/acre — as a 246–262 range with each endpoint tied
   to its fraction. Envelope table, evidence summary, and footer updated to 246–262 / 262
   primary.
3. **2026/forthcoming citations spot-checked.** All three resolve: [S12] DOI 10.3390/rs18071065
   resolves; [S13] arXiv 2601.00857 resolves (Ma et al.; findings on limited spatial
   transferability confirmed). [S11] Remote Sensing 18(3):436 is confirmed extant (published
   2026-01-30) but the article page returned HTTP 403 to direct fetch, so it is downgraded to
   VERIFIED-BY-SEARCH and listed under MODERATE, not relied on for the STRONG cross-region
   claim. The STRONG cross-region-transfer rating now rests only on the two fully-resolvable
   sources [S12,S13]. Also corrected a factual slip: the [S12] regions are Guanzhong Plain /
   Kansas (not the previously stated Germany/Japan/Michigan).
4. **[S2/S3] bracket split.** Now three distinct, individually resolvable references: [S2] =
   **Green, C., Tobin, B., O'Shea, M. et al. (2007)**, *Eur. J. Forest Research* 126:179–188,
   DOI 10.1007/s10342-005-0093-3 — the single 19-yr unthinned stand, 74 t C/ha ±7% (verified);
   [S3] = **Tobin & Nieuwenhuis (2007)**, *Eur. J. Forest Research* 126:189–196, DOI
   10.1007/s10342-005-0105-3 — the BEF trajectory; [S4] = companion improved-BEF work. The
   74 Mg C/ha anchor now traces to one paper [S2]. (Also corrected: lead author is Green and
   year is 2007, not the previously stated 2005/Tobin.)
