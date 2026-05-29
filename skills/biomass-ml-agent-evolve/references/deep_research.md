# Deep Research

## Purpose
Produce task-specific research that constrains the experiment.

## Required phases
- read `references/source_registry.md` and use it as the source hierarchy unless the task requires a better official source
- task framing and target definition (standing stock vs. annual increment vs. change; units)
- state-of-the-art review (GEDI L4 product literature, NFI inter-comparisons, fusion studies)
- benchmark performance discovery
- data-source suitability review (coverage, latency, licensing, plot-density)
- remote-sensing feature review (optical embeddings, LiDAR vertical structure, SAR backscatter, DEM topography)
- ground-truth review (plot networks, label-noise floor, GPS error vs. footprint)
- evaluation-method review
- known failure modes (feature ceiling, ecoregion mismatch, GPS-error label noise, GEDI-coverage gaps)
- deployment and monitoring considerations

## Required output
`research/deep_research.md` containing:
- task framing
- benchmark table
- recommended data sources
- recommended feature families
- recommended baseline ladder
- failure modes
- deployment caveats
- benchmark-derived default target if user target missing
- access feasibility note for any non-trivial source (GEDI auth, NEON AOP licensing, Source Coop reachability)

## Benchmark rules
- include at least 3 relevant studies or benchmark sources; if fewer exist, the artefact must say so explicitly and explain why
- for each cited benchmark, record `{title, venue_or_org, year, url_or_doi, reported_metric, reported_value, geography, split_type}`
- a benchmark whose `split_type` is `random` cannot be used to set the operational default threshold for a task that requires spatial or temporal generalisation
- separate local high-water-mark results from realistic operational defaults
- clearly mark uncertainty
- when citing aggregated biomass products (e.g., ESA CCI Biomass, GEDI L4B) as benchmarks, distinguish their *gridded* validation R² (typically 0.5–0.7 at 1 km) from *plot-level* expectations (typically 0.3–0.5 at sub-hectare scale) — these are not interchangeable

## Default-threshold anchors (examples; re-anchor per task)

These are defaults, not promises. The Research Actor must re-anchor for the exact subject, geography, label quality, and resolution, and document the adjustment.

| Example task | Realistic default target | Stretch target | Source class |
|---|---|---|---|
| CONUS forest AGB regression at field-plot scale, project-LOPO, embeddings-only | R² ≥ 0.40, RMSE ≤ 60 tCO₂/acre, \|bias\| ≤ 5 | R² ≥ 0.55, RMSE ≤ 45 tCO₂/acre, predicted_range_discrimination ≥ 0.6 | Internal AGB pilot baselines (project-internal `docs/runs/agb_usa.md`); GEDI L4B v2 validation literature for the gridded reference. |
| CONUS forest AGB regression at field-plot scale, project-LOPO, with GEDI canopy-height co-feature | R² ≥ 0.55, RMSE ≤ 50 tCO₂/acre, predicted_range_discrimination ≥ 0.6 | R² ≥ 0.65, RMSE ≤ 40 tCO₂/acre, predicted_range_discrimination ≥ 0.75 | Optical + GEDI fusion literature; co-target supervision studies. Anchor down to plot scale; do not borrow gridded-product R² directly. |
| Canopy height regression at plot scale (RH98 vs. GEDI / NEON) | R² ≥ 0.50, RMSE ≤ 6 m | R² ≥ 0.65, RMSE ≤ 4 m | NEON AOP CHM literature; published GEDI canopy-height products. |

If the user supplied a target, keep it and record this table only as background. If the user did not, propose the realistic default and offer the stretch.

## Actor addendum
Must state:
- "realistic default threshold" (numeric, with units)
- "stretch threshold" (numeric, with units)
- "stop if benchmark unmet" condition (which iteration cap triggers escalation)
- explicit benchmark adjustment vs the anchor table above
- access-feasibility note for any required source that is not anonymous-public (GEDI tier, NEON licensing, internal gpkg path)

## Critic addendum
Reject:
- benchmark claims without a citation from the structured row schema above
- use of a `random`-split benchmark to set the operational default for a task that requires spatial or temporal generalisation
- borrowing gridded-product validation R² to set a plot-scale operational default
- stale or non-primary sourcing where a stronger primary source is reachable
- silent omission of one of the Actor-addendum fields (realistic, stretch, stop-if-unmet, anchor adjustment, access feasibility)
