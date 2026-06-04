# Disturbance-Timing Contamination Audit

Dataset: `features_iter3.parquet` ⨝ `disturbance_timing_features.csv`, 4636 modelled plots. Survey years: [np.int64(2022), np.int64(2023)]. Buckets by Hansen loss year relative to plot survey year.

## 1. Bucket counts

| bucket | n | % |
| --- | --- | --- |
| undisturbed | 3222 | 69.5% |
| pre_at_survey | 1317 | 28.4% |
| post_survey | 97 | 2.1% |

**Post-survey contamination: 97 plots (2.1%)** — harvested *after* their field survey, so their field biomass is legitimately high but 'current land cover' reads non-forest.

## 2. Bucket × region

| region | post_survey | pre_at_survey | undisturbed |
| --- | --- | --- | --- |
| mw | 71 | 629 | 1931 |
| ne | 16 | 577 | 814 |
| wv | 10 | 111 | 477 |

## 3. Target (tCO₂/acre) by bucket

| bucket | n | mean | median | Q1-share | Q5-share |
| --- | --- | --- | --- | --- | --- |
| undisturbed | 3222 | 125.6 | 117.3 | 10% | 25% |
| pre_at_survey | 1317 | 64.4 | 51.7 | 44% | 6% |
| post_survey | 97 | 132.9 | 129.3 | 12% | 29% |

*Hypothesis check:* pre-or-at-survey harvest should skew low (high Q1-share); post-survey plots should sit higher (measured before the cut).

## 4. How the existing `dist_years_since` encodes each bucket

| bucket | mean dist_years_since | % with years_since==0 |
| --- | --- | --- |
| undisturbed | 100.0 | 0% |
| pre_at_survey | 50.0 | 1% |
| post_survey | 51.6 | 46% |

Post-survey plots receiving `years_since==0` are the feature–label inversion: a 'just disturbed' signal on a high-biomass plot.

## 5. Hansen vs LandTrendr (pre/at-survey detection)

- Hansen pre/at-survey loss: 1324 plots
- LandTrendr pre/at-survey disturbance (mag>150): 835 plots
- both: 550; LandTrendr-only: 285 (partial harvest / degradation Hansen's stand-replacing loss misses)

