# Forest-probability strategy for AGB inference — decision & considerations

Status: decided 2026-06-04. Context: the Bayfield County wall-to-wall AGB map (30 m, EPSG:32615)
reads directionally well, but has a low-end **floor of ~30 tCO₂/acre** — it never predicts low,
even where the ground is clearly non-forest or recently harvested. This note records how we will
use the Dynamic World forest probability to deal with that, why, and what to develop next.

## 1. The problem — the floor has two distinct sources

| source | description | who can fix it |
| --- | --- | --- |
| (a) **Non-forest extrapolation** | fields, clearings, scrub, developed land get forest-level biomass because the model was trained only on forest plots and has never seen "this isn't forest" | a forest mask / scoping the domain |
| (b) **Within-forest compression** | genuinely low / recently-cut *forest* is over-predicted (LOPO Q1 over-prediction of **+35.5** tCO₂/acre; LightGBM L2-loss leaf averaging shrinks the tails) | better low-end features + low-biomass training data |

Conflating these is what made earlier single-fix attempts disappointing. They need different tools.

## 2. Evidence gathered this session

- **DW probability as a feature, alone → no effect.** Adding DW bands to the model gave
  **ΔR² = +0.0005** and left Q1 bias unchanged. Reason: every training plot is forest
  (`dw_trees` ≈ 0.73 with negligible variance), so the feature has **no in-domain variance and no
  low examples** — a tree model cannot learn "low trees-prob → low biomass" from data that
  contains no such cases. (`reports/dynamic_world_experiment.md`)
- **Hard forest mask → fixes non-forest only.** Zeroing DW `trees` < 0.5 set **4.6%** of the
  county to 0; the map min went to 0 for non-forest, but the *within-forest* floor was untouched.
  (`predictions/README.md`)
- **Stale CHM was propping the floor up.** Embeddings-only floor = **12.4** vs full-model **26.6**;
  the ETH-2020 canopy height reads tall over 2021–2023 cuts and pushes them up. Dropping it lowers
  the floor (but loses some accuracy).
- **Disturbance features raise, not lower, the map floor.** `emb+dstx` min = 32 (vs emb-only 12.4):
  the dstx features only pull predictions down where a disturbance is *detected*; everywhere else
  they read "undisturbed" and nudge up. They sharpen detected cuts but don't fix the floor.

Net: a forest mask cleanly fixes (a); nothing we have yet meaningfully fixes (b).

## 3. Options considered & trade-offs

| | feature space | soft post-process (×ramp) | hard clip-to-zero |
| --- | --- | --- | --- |
| fixes non-forest (a) | ✓ (needs augmentation) | ✓ soft | ✓ hard |
| fixes within-forest low (b) | ✓ (needs augmentation) | partial (edges) | ✗ |
| smooth / mixed-pixel handling | best | good | poor (edge artifacts) |
| threshold sensitivity | none | low | high |
| physical defensibility | high | low (heuristic) | medium (standard product) |
| retrain needed | yes | no | no |
| works on *current* data | no (needs augmentation) | yes | yes (proven) |
| operational separability | low (coupled input) | medium | high (swappable layer) |

## 4. Chosen strategy

**Use the forest probability in *two complementary roles*, and scope the model to forest:**

1. **As a model feature** — DW `trees` probability (and optionally the other DW probability bands)
   enters the biomass feature space, so the model can genuinely discriminate low- vs high-biomass
   *within* the forest/forest-management domain.
2. **As a hard-clip forest mask at inference** — predict only where it is forest; **assume 0
   tCO₂/acre everywhere else**. This deterministically removes "definitely not forest" locations
   and defines the model's job as *forest / forest-management land only*.

**Why both, and why they are not redundant:**
- The **clip** enforces the *domain boundary* deterministically — a clean, swappable land-cover
  decision independent of the biomass model. It is the operational guardrail and is standard
  practice for forest-carbon products (predict within a forest mask, zero outside).
- The **feature** does a different job: it helps the model rank biomass *inside* the kept domain
  (sparse vs stocked, edge vs interior, recently harvested vs intact) — which the clip cannot do.
- **Caveat (important):** as shown in §2, the feature only adds value **once the training set
  contains low/non-forest variance**. On today's all-forest plots the feature is inert. So the
  feature role is contingent on the augmentation in §5 — until then, the *clip* is doing all the
  real work and the feature is a near-no-op (harmless, but not yet earning its place).

This framing also reframes the whole floor problem cleanly: **(a) is solved by assumption** (non-
forest ≡ 0 via the mask), leaving **(b) — the within-forest low end — as the only remaining
modelling problem**, which §5 targets directly.

## 5. Roadmap / further development

1. **Low-biomass augmentation (the key unlock).** Add training points that the model currently
   lacks: confirmed **harvested / recently-cut forest** and **near-zero-biomass forested**
   locations (e.g. Hansen current-year clearcuts, sparse/regenerating stands, NLCD/ESA-WorldCover
   non-forest within the project AOIs) with biomass ≈ 0 and features extracted identically. This
   gives the DW feature in-domain variance so it finally bites, and directly attacks the +35.5 Q1
   over-prediction.
2. **Retrain with DW prob in the feature set** once augmentation exists; re-measure Q1 bias and the
   map floor (expect both to drop).
3. **Recalibrate the mask** — tune the DW threshold (0.5 is provisional), and/or combine sources
   (DW + ESA WorldCover) to reduce single-product error. Keep the mask layer epoch-matched to the
   embedding/inference year (DW 2023 here).
4. **Revisit a current-epoch canopy height** (GEDI-derived / Meta 1 m) to replace the stale ETH
   2020 CHM — it would *help* the low end rather than prop it up.

## 6. Risks & considerations

- **DW classification error & threshold sensitivity.** A hard clip turns DW mistakes into hard
  biomass errors (a mis-called forest pixel → forced 0; a mis-called non-forest → keeps full
  biomass). Mitigate with threshold tuning, multi-source consensus, and (later) the soft-ramp
  variant if edge artifacts matter.
- **Augmentation balance.** Too many synthetic zeros will bias the model low; the augmented class
  fraction must be tuned and validated.
- **Epoch alignment.** Mask, embeddings, disturbance and (future) CHM features must all reference
  the inference year, or recent change is mis-handled (the stale-CHM lesson).
- **Double-counting.** Feature + clip are complementary, not additive on the same signal — fine as
  long as the feature is given in-domain variance (post-augmentation); before that it is inert.
- **Evaluation gap.** LOPO on the forest plots cannot measure the low end we care about — we need
  held-out **non-forest / harvested** validation points to quantify whether the floor is actually
  fixed. Report per-quintile bias *and* low-class performance, not just overall R².

## 7. Current artefacts (Bayfield)

In `predictions/`: `bayfield_agb_30m.tif` (full), `_embonly_`, `_embdstx_`, `_forestmasked`
variants, `bayfield_agb_embonly_30m_forestmasked.tif` (best low-end so far), and
`bayfield_dw_trees_prob_30m.tif` (the DW mask layer). Mask logic: `scripts/apply_forest_mask.py`.
Models: `models/inference_model{,_embonly,_embdstx}.txt`.
