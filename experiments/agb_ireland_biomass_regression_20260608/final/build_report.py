#!/usr/bin/env python3
"""Build a self-contained HTML report (PNGs embedded as base64) for the Ireland
AGB zero-shot transfer experiment: our carbon map vs Deep Biomass, with a focus
on low-biomass behaviour, limitations and avenues to explore."""

import base64
from pathlib import Path

EXP = Path(__file__).resolve().parent.parent  # experiment root
OUT = EXP / "final" / "ireland_agb_report.html"


def img(path: str, alt: str, caption: str) -> str:
    """Return a <figure> with the PNG inlined as a base64 data URI."""
    p = EXP / path
    b64 = base64.b64encode(p.read_bytes()).decode("ascii")
    return (
        f"<figure>\n"
        f'  <img src="data:image/png;base64,{b64}" alt="{alt}" loading="lazy">\n'
        f"  <figcaption>{caption}</figcaption>\n"
        f"</figure>"
    )


FIG_MAIN = img(
    "final/figures/ireland_vs_deepbiomass_yearmatched.png",
    "Our model vs Deep Biomass, year-matched 2022/2023/2024",
    "<b>Year-matched comparison, our carbon map vs Deep Biomass.</b> Per-year scatters "
    "(2022 / 2023 / 2024), the 3-yr-mean scatter and the portfolio trajectory. Every point "
    "sits well above the 1:1 line: our portfolio mean is <b>91.6 tCO₂/acre</b> against "
    "DB’s <b>26.8</b> — a persistent <b>3.4×</b> level offset. The bottom-right "
    "panel is the reproducibility cross-check (73 stands, max |Δ| = 0.000).",
)
FIG_MASKED = img(
    "final/figures/ireland_vs_deepbiomass_yearmatched_masked.png",
    "Forest-masked model vs Deep Biomass",
    "<b>After the Dynamic World forest/clearfell mask.</b> Non-forest pixels are set to 0. "
    "The bottom-right panel (coloured by forest fraction) shows the mask pulling low-forest-fraction "
    "(young / clearfelled) stands down toward zero. The masked ratio falls to <b>3.05×</b> and, "
    "for the first time, 5 stands read <i>below</i> DB — the structural zero is fixed, but the "
    "in-domain regression floor for stocked-but-young stands remains.",
)
FIG_AGE = img(
    "evaluation/figures/pred_db_vs_age.png",
    "Predicted carbon and Deep Biomass against stand age",
    "<b>H3 — structural rank-tracking.</b> Our predictions (blue) rise with stand age "
    "(ρ = 0.55), while Deep Biomass (orange) stays flat (ρ = 0.11, ns). This is the "
    "strongest evidence the transfer is reading real stand structure, not noise.",
)
FIG_DELTA = img(
    "evaluation/figures/delta_histogram.png",
    "Signed delta distribution",
    "<b>Signed divergence (pred − DB).</b> 98.6% of Locations lie to the right of zero "
    "(median +72.5 tCO₂/acre). The few points near/below zero are the very young, "
    "near-zero-structure stands discussed under low-biomass behaviour.",
)
FIG_QUINT = img(
    "evaluation/figures/quintile_signed_bias.png",
    "Per-quintile signed bias",
    "<b>H2 — saturation resistance (NOT supported, but uninformative).</b> The gap is "
    "flat-to-declining across DB quintiles rather than widening. This is a measurement artefact "
    "of cutting quintiles on DB magnitude, <i>not</i> our head saturating (see text).",
)
FIG_OOD = img(
    "error_analysis/figures/perdim_ood_contribution.png",
    "Per-dimension contribution to OOD shift",
    "<b>What drives the domain shift.</b> 8 of the 64 AlphaEarth embedding dimensions "
    "(notably emb_26, emb_50, emb_23) carry 68% of the Irish OOD centroid shift — the "
    "maritime-temperate-plantation signature with no US analog. These are the concrete targets "
    "for an analog-subset retrain.",
)

HTML = f"""<!DOCTYPE html>
<html lang="en-GB">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ireland AGB — our carbon map vs Deep Biomass</title>
<style>
  :root {{
    --ink: #1a2b22; --muted: #5b6b62; --line: #d7e0da;
    --accent: #1f7a4d; --warn: #b25b00; --bad: #b3261e; --good: #1f7a4d;
    --bg: #fbfdfb; --card: #ffffff;
  }}
  * {{ box-sizing: border-box; }}
  body {{
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
    color: var(--ink); background: var(--bg); margin: 0; line-height: 1.55;
  }}
  .wrap {{ max-width: 980px; margin: 0 auto; padding: 2.4rem 1.4rem 4rem; }}
  header.title {{ border-bottom: 3px solid var(--accent); padding-bottom: 1.2rem; margin-bottom: 2rem; }}
  h1 {{ font-size: 1.9rem; margin: 0 0 .4rem; line-height: 1.2; }}
  h2 {{ font-size: 1.35rem; margin: 2.6rem 0 .8rem; padding-top: .4rem; border-top: 1px solid var(--line); }}
  h3 {{ font-size: 1.08rem; margin: 1.6rem 0 .5rem; color: var(--accent); }}
  p, li {{ font-size: .98rem; }}
  .sub {{ color: var(--muted); font-size: .95rem; margin: 0; }}
  .pills {{ display: flex; flex-wrap: wrap; gap: .5rem; margin: 1rem 0 0; }}
  .pill {{ font-size: .8rem; padding: .25rem .65rem; border-radius: 999px; background: #eef4f0;
           border: 1px solid var(--line); color: var(--ink); }}
  .pill.bad {{ background: #fce9e7; border-color: #f3c4bf; color: var(--bad); }}
  .pill.warn {{ background: #fbf0e1; border-color: #f0d9b6; color: var(--warn); }}
  .pill.good {{ background: #e6f4ec; border-color: #bfe2cd; color: var(--good); }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(190px, 1fr)); gap: 1rem; margin: 1.4rem 0; }}
  .kpi {{ background: var(--card); border: 1px solid var(--line); border-radius: 10px; padding: 1rem; }}
  .kpi .n {{ font-size: 1.7rem; font-weight: 700; color: var(--accent); line-height: 1; }}
  .kpi .l {{ font-size: .82rem; color: var(--muted); margin-top: .35rem; }}
  figure {{ margin: 1.6rem 0; background: var(--card); border: 1px solid var(--line);
            border-radius: 10px; padding: 1rem; }}
  figure img {{ width: 100%; height: auto; border-radius: 6px; display: block; }}
  figcaption {{ font-size: .86rem; color: var(--muted); margin-top: .7rem; }}
  table {{ border-collapse: collapse; width: 100%; margin: 1.2rem 0; font-size: .9rem; }}
  th, td {{ border: 1px solid var(--line); padding: .5rem .6rem; text-align: left; }}
  th {{ background: #eef4f0; font-weight: 600; }}
  td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums; }}
  .callout {{ border-left: 4px solid var(--accent); background: #f1f7f3; padding: .9rem 1.1rem;
              border-radius: 0 8px 8px 0; margin: 1.3rem 0; }}
  .callout.warn {{ border-color: var(--warn); background: #fbf4ea; }}
  .callout.bad {{ border-color: var(--bad); background: #fcefed; }}
  .callout p {{ margin: .3rem 0; }}
  .tag {{ font-weight: 700; }}
  .tag.bad {{ color: var(--bad); }} .tag.warn {{ color: var(--warn); }} .tag.good {{ color: var(--good); }}
  ol.avenues > li {{ margin: .7rem 0; }}
  code {{ background: #eef4f0; padding: .08rem .35rem; border-radius: 4px; font-size: .85em; }}
  footer {{ margin-top: 3rem; padding-top: 1.2rem; border-top: 1px solid var(--line);
            color: var(--muted); font-size: .82rem; }}
</style>
</head>
<body>
<div class="wrap">

<header class="title">
  <h1>Ireland AGB — our carbon map vs Deep Biomass</h1>
  <p class="sub">Zero-shot transfer of the pre-trained <code>embdstx</code> head (64 AlphaEarth embeddings
  + 3 disturbance-timing features; target CO₂ standing stock, tCO₂/acre) to 141 Irish Dasos
  forestry Locations (Sitka-dominant maritime-temperate plantation). Model-vs-model comparison —
  <b>no Irish ground truth</b>. Deep Biomass (DB) is a known under-estimator used only as a directional
  lower bound. Generated {OUT.name} from accepted experiment artefacts.</p>
  <div class="pills">
    <span class="pill good">Encoding gate: PASS (corr 0.986)</span>
    <span class="pill good">H1 dominance: SUPPORTED</span>
    <span class="pill warn">H2 saturation: uninformative</span>
    <span class="pill good">H3 structure: SUPPORTED</span>
    <span class="pill bad">OOD: SEVERE (100% beyond 99th-pct)</span>
    <span class="pill bad">Decision: RETRAIN_WARRANTED</span>
  </div>
</header>

<h2>1. Headline</h2>
<p>The transfer is <b>encoding-valid and structurally sensible</b> — our predictions track stand age
and height far better than Deep Biomass — <b>but it operates in a severe extrapolation regime</b>, so
absolute carbon levels are not yet trustworthy. Trust the <b>ranking / structure</b>, not the absolute
tCO₂/acre. The portfolio reads ~3.4× above DB; most of that gap is genuine (DB under-reads this
high-biomass Sitka plantation), but part is unquantifiable OOD extrapolation.</p>

<div class="grid">
  <div class="kpi"><div class="n">91.6</div><div class="l">Our portfolio mean (tCO₂/acre); 131 Mg/ha</div></div>
  <div class="kpi"><div class="n">26.8</div><div class="l">Deep Biomass mean (tCO₂/acre); 38 Mg/ha</div></div>
  <div class="kpi"><div class="n">3.4×</div><div class="l">Level ratio (ours / DB), year-matched</div></div>
  <div class="kpi"><div class="n">98.6%</div><div class="l">Locations with pred ≥ DB (H1)</div></div>
  <div class="kpi"><div class="n">100%</div><div class="l">Locations beyond the training 99th-pct radius (OOD)</div></div>
  <div class="kpi"><div class="n">141 / 141</div><div class="l">Stands inferred (593,754 native 10 m pixels/yr)</div></div>
</div>

<h2>2. Our carbon map vs Deep Biomass</h2>
<p>Both sides are fixed to the same year (2022 / 2023 / 2024) and use the same pixel-then-aggregate
estimator, removing the temporal confound of the earlier comparison. The ~3.4× offset is stable across
years and alignment methods — it is a genuine, robust level difference driven by the CONUS→Ireland
domain shift and by DB’s known tendency to under-read, <i>not</i> an artefact of mismatched years.</p>

{FIG_MAIN}

<p>The divergence is overwhelmingly one-directional: 98.6% of Locations read higher than DB, median
+72.5 tCO₂/acre. The left tail (points near or below zero) is the low-biomass story of §4.</p>

{FIG_DELTA}

<h3>Why we believe the ranking even without ground truth</h3>
<p>Our head responds to real stand structure; DB barely does. This is the single strongest piece of
truth-free evidence that the transfer is signal rather than noise.</p>

{FIG_AGE}

<table>
  <thead><tr><th>Structural axis</th><th class="num">ρ (our pred)</th><th class="num">ρ (Deep Biomass)</th><th>Read</th></tr></thead>
  <tbody>
    <tr><td>Stand age</td><td class="num">0.553</td><td class="num">0.113 (ns)</td><td>Ours tracks age strongly; DB flat</td></tr>
    <tr><td>Top height (Hdom)</td><td class="num">0.556</td><td class="num">0.200</td><td>Ours tracks height strongly; DB weak</td></tr>
    <tr><td>Yield class (YC)</td><td class="num">−0.077 (ns)</td><td class="num">0.031 (ns)</td><td>Null for both (YC ≠ standing stock)</td></tr>
  </tbody>
</table>

<h2>3. Low biomass — the key consideration</h2>
<p>Low-biomass stands (young plantation, recent clearfell, bare ground) are where this product is
<b>weakest</b>, for two distinct reasons that must not be conflated:</p>

<div class="callout warn">
  <p><span class="tag warn">(a) The regression floor.</span> The US-trained head has a hard per-pixel
  floor of ~16 tCO₂/acre and a stand-level minimum of ~30.5 tCO₂/acre. It cannot natively predict
  near-zero biomass: a freshly clearfelled or newly planted stand is reported at ~30–50 tCO₂/acre,
  which is physically wrong. This is an <i>in-domain</i> limitation independent of the OOD problem.</p>
  <p><span class="tag warn">(b) The structural zero.</span> A Dynamic World forest/clearfell mask sets
  non-forest pixels to 0, so a stand’s density becomes <code>forest_fraction × mean(forest-pixel preds)</code>.
  This correctly collapses bare / clearfelled area toward zero.</p>
</div>

<p>The mask removes ~11% of portfolio density (91.6 → 81.5 tCO₂/acre) and pulls the DB ratio from
3.42× down to 3.05×. Crucially it fixes (b) but <b>not</b> (a): a stand that is genuinely forested but
young still sits on the ~16–30 tCO₂/acre floor. After masking, 5 stands finally read <i>below</i> DB.</p>

{FIG_MASKED}

<h3>Validation on known age-0 / Hdom≈0 stands</h3>
<p>These should collapse toward zero — and they do, in proportion to their (low) forest fraction:</p>
<table>
  <thead><tr><th>Stand</th><th class="num">age</th><th class="num">Hdom</th><th class="num">forest frac</th><th class="num">unmasked</th><th class="num">masked</th></tr></thead>
  <tbody>
    <tr><td>Moyne</td><td class="num">0.0</td><td class="num">0.0</td><td class="num">0.080</td><td class="num">44.2</td><td class="num">5.7</td></tr>
    <tr><td>Peak</td><td class="num">0.0</td><td class="num">0.0</td><td class="num">0.022</td><td class="num">48.6</td><td class="num">2.2</td></tr>
    <tr><td>Carrigeeny</td><td class="num">0.0</td><td class="num">0.0</td><td class="num">0.031</td><td class="num">50.9</td><td class="num">1.8</td></tr>
    <tr><td>Carrowkeel</td><td class="num">2.4</td><td class="num">1.1</td><td class="num">0.000</td><td class="num">40.6</td><td class="num">0.0</td></tr>
    <tr><td>Rathcahill West</td><td class="num">2.0</td><td class="num">0.0</td><td class="num">0.000</td><td class="num">39.0</td><td class="num">0.0</td></tr>
    <tr><td>Cashel ⚠</td><td class="num">2.6</td><td class="num">0.0</td><td class="num">0.968</td><td class="num">107.3</td><td class="num">105.2</td></tr>
  </tbody>
</table>

<div class="callout bad">
  <p><span class="tag bad">The mask is noisy at the margin.</span> Dynamic World <code>trees</code> is
  optical and binary at 0.5, so it makes two kinds of mistake on exactly the low-biomass stands that
  matter most:</p>
  <p><b>Missed clearfell</b> — ground vegetation / residual cover read as trees, so young stands stay
  too high: <i>Cashel</i> (age 2.6, ff 0.968, masked 105.2), <i>Benmore</i> (age 0, ff 0.932, masked 84.1).</p>
  <p><b>False non-forest</b> — mature canopy under-detected, so the mask over-penalises:
  <i>Cummeen Upper</i> (age 21.6, Hdom 11.3, ff 0.000 → masked 0.0, clearly wrong),
  <i>Knocknahooan</i> (age 18, ff 0.331).</p>
</div>

<p>Reassuringly, forest fraction rises monotonically with stand age (0–3 yr: mean ff 0.31; 15–25 yr:
0.86; 25+ yr: 0.89), so the mask is behaving correctly in aggregate — it is only individual stand
margins where DW disagrees with the Dasos age.</p>

<h2>4. Limitations</h2>
<table>
  <thead><tr><th>Limitation</th><th>Detail / evidence</th></tr></thead>
  <tbody>
    <tr><td><b>No ground truth</b></td><td>All divergence is measured against an under-estimator. We can rule out co-saturation and show structural tracking, but cannot positively prove high-end accuracy.</td></tr>
    <tr><td><b>Severe OOD (the central caveat)</b></td><td>Irish Mahalanobis min 27.8 = 1.9× the training 99th-pct radius (14.79); 100% of Locations beyond it. Domain-classifier AUC ≈ 1.0. Absolute levels are deep extrapolation.</td></tr>
    <tr><td><b>Low-biomass regression floor</b></td><td>~16 tCO₂/acre per pixel, ~30.5 stand-level minimum — the head cannot natively read near-zero biomass (§3a).</td></tr>
    <tr><td><b>No vertical-structure lever</b></td><td>The head is optical-AEF-only (no CHM / SAR). It structurally <i>cannot</i> prove saturation resistance against an optical under-estimator.</td></tr>
    <tr><td><b>H2 uninformative, not failed</b></td><td>Binning by DB magnitude mechanically maximises DB’s own spread, so the gap compresses at high DB. Within the top quintile our head still rank-tracks age (ρ 0.57) and reaches 138.4 (only 27% of the training range) — no plateau.</td></tr>
    <tr><td><b>Mask noise</b></td><td>Optical DW mis-labels young-with-ground-veg and mature-canopy-gap stands (§3).</td></tr>
    <tr><td><b>Calibration is not valid</b></td><td>With 100% OOD and no Irish truth there is nothing to calibrate to; recalibrating to DB would import DB’s under-estimate.</td></tr>
  </tbody>
</table>

{FIG_QUINT}
{FIG_OOD}

<h2>5. Avenues to explore</h2>
<ol class="avenues">
  <li><b>Analog-subset retrain (highest leverage).</b> Rebuild the head on the maritime-temperate +
  high-biomass-conifer subset of the training pool (New England–Acadian + Pacific-coastal / Cascades /
  Alaskan conifer) to pull the Irish manifold inside training support. Target the OOD dimensions
  emb_26 / 50 / 23 / 55. This most reduces the absolute-level uncertainty.</li>

  <li><b>A low-biomass hurdle / two-stage model.</b> Pair a clearfell/young classifier with the
  regressor so the ~16–30 tCO₂/acre floor is replaced by a genuine near-zero prediction for young
  and clearfelled stands — directly fixing limitation §3a, which the DW mask cannot.</li>

  <li><b>Add a vertical-structure lever (L-band SAR / Sentinel-1 and/or a CHM).</b> The only way a
  future evaluation could <i>actually</i> test saturation resistance, and it anchors high-biomass
  absolute levels physically rather than optically.</li>

  <li><b>A better forest mask.</b> Replace / blend the binary optical DW mask with a multi-source mask
  (Hansen loss-year + SAR + Dasos clearfell records) to cut the missed-clearfell and false-non-forest
  errors at stand margins (§3).</li>

  <li><b>Re-cut the saturation diagnostic on an independent axis.</b> Bin by age / Hdom instead of DB
  magnitude (free, immediate). On those axes our head keeps rising while DB stays flat — promote this
  as the honest truth-free saturation read.</li>

  <li><b>Acquire even a small set of Irish ground-truth plots.</b> The single thing that would convert
  every “trust the ranking, not the level” caveat into a calibrated, validated product. Until then,
  no post-hoc calibration is warranted.</li>
</ol>

<footer>
  <p><b>Provenance.</b> Built from accepted artefacts in
  <code>experiments/agb_ireland_biomass_regression_20260608/</code>:
  <code>final/experiment_report.md</code>, <code>final/ireland_yearmatched_comparison.md</code>,
  <code>final/ireland_forest_mask.md</code>, <code>error_analysis/error_analysis.md</code>,
  <code>evaluation/bias_characterisation.md</code>. Figures embedded as base64 from
  <code>final/figures/</code>, <code>evaluation/figures/</code>, <code>error_analysis/figures/</code>.
  Inference-only zero-shot transfer; no model trained, no ground truth. seed 42; AEF
  <code>GOOGLE/SATELLITE_EMBEDDING/V1/ANNUAL</code>. Mg/ha = tCO₂/acre / 0.6977.</p>
</footer>

</div>
</body>
</html>
"""

OUT.write_text(HTML, encoding="utf-8")
print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")
