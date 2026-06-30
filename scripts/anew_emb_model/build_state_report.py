"""
Build a self-contained HTML current-state report (figures embedded as base64 data URIs).

The repo gitignores experiments/**/*.png, so figures don't travel in git. This assembles a
single committable, standalone current-state.html (no external assets) that embeds the curated
figures inline, so the report carries its own visuals. Prose mirrors current-state.md.

Run:
    uv run --project /home/mattc/code/agb-ml-agent-evolve \
        python scripts/anew_emb_model/build_state_report.py
"""

from __future__ import annotations

import base64
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "trust"))
import common  # noqa: E402

REPO = common.REPO
GT = REPO / "experiments/agb_anew_gt_applicability_20260626/figures"
MOD = REPO / "experiments/agb_anew_emb_weighted_20260630/figures"
OUT = REPO / "current-state.html"

# (section, figure path, caption)
FIGURES = [
    (
        "1. GT-space DI/AOA applicability",
        GT / "aoa_national_map.png",
        "Self-referential AOA over the 51-project GT: 94% of plots inside; the AK/PNW conifer/tundra "
        "frontier (yellow, out-of-AOA) is the only region the GT cannot vouch for from its interior.",
    ),
    (
        "1. GT-space DI/AOA applicability",
        GT / "di_ranking_bar.png",
        "Per-project dissimilarity ranking (interior → frontier), coloured by spatial bloc. The dense "
        "broadleaf interior is highly redundant; five conifer/tundra projects form the frontier.",
    ),
    (
        "2. Emb-only global model + trust layer",
        MOD / "scheme_per_tier_rmse.png",
        "Frontier-aware weighting (S1–S4) does not beat unweighted S0 on any tier — no frontier signal "
        "for weighting to exploit, so S0 ships.",
    ),
    (
        "2. Emb-only global model + trust layer",
        MOD / "error_vs_di.png",
        "Trust layer: monotone DI→expected-RMSE curve. Every prediction carries a DI, an "
        "inside/outside-AOA flag (threshold 0.558), and an expected error.",
    ),
    (
        "3. Low/zero-biomass de-biasing",
        MOD / "low_end_band_bias.png",
        "Conditional bias by biomass band. log1p (bold) roughly halves the <100 over-prediction; "
        "S0/hurdle/calibration overlap near the top (they don't de-bias). High end dips lowest under "
        "log1p — the accepted, quantified trade.",
    ),
    (
        "3. Low/zero-biomass de-biasing",
        MOD / "low_end_pred_vs_true.png",
        "Predicted vs true (LOPO OOF): log1p lifts low-end predictions toward the 1:1 line versus the "
        "mean-seeking raw-L2 baseline.",
    ),
    (
        "3. Low/zero-biomass de-biasing",
        MOD / "low_end_roc_lt50.png",
        "Separating true < 50 tCO2/acre: AUC ~0.88. Low biomass is moderately separable and the "
        "regressor already exploits it — the two-stage hurdle (overlapping curve) adds nothing.",
    ),
]


def img_tag(path: Path) -> str:
    b64 = base64.b64encode(path.read_bytes()).decode()
    return f'<img src="data:image/png;base64,{b64}" alt="{path.stem}">'


def main() -> None:
    missing = [f for _, f, _ in FIGURES if not f.exists()]
    assert not missing, f"missing figures: {missing}"

    sections = {}
    for sec, path, cap in FIGURES:
        sections.setdefault(sec, []).append((path, cap))
    fig_html = ""
    for sec, items in sections.items():
        fig_html += f"<h3>{sec}</h3>\n"
        for path, cap in items:
            fig_html += f"<figure>{img_tag(path)}<figcaption>{cap}</figcaption></figure>\n"

    css = """
    body{font:16px/1.6 -apple-system,Segoe UI,Roboto,Helvetica,Arial,sans-serif;color:#1a1a1a;
      max-width:900px;margin:2rem auto;padding:0 1.2rem}
    h1{font-size:1.7rem;margin-bottom:.2rem} h2{margin-top:2.2rem;border-bottom:2px solid #eee;padding-bottom:.3rem}
    h3{margin-top:1.6rem;color:#333} .banner{background:#eef6ff;border-left:4px solid #2b6cb0;
      padding:.8rem 1rem;border-radius:4px;margin:1rem 0}
    table{border-collapse:collapse;width:100%;margin:1rem 0;font-size:.92rem}
    th,td{border:1px solid #ddd;padding:.4rem .6rem;text-align:left} th{background:#f5f5f5}
    .win{color:#1a7f37;font-weight:600} figure{margin:1.2rem 0}
    img{max-width:100%;border:1px solid #eee;border-radius:4px} figcaption{font-size:.86rem;color:#555;margin-top:.4rem}
    code{background:#f3f3f3;padding:.1rem .35rem;border-radius:3px;font-size:.88rem}
    .muted{color:#666;font-size:.9rem}
    """

    html = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>AGB ANEW model — current state</title><style>{css}</style></head><body>
<h1>AGB ANEW model — current state</h1>
<p class="muted">Self-contained report · 2026-06-30 · figures embedded · companion to <code>current-state.md</code></p>
<div class="banner"><b>Current baseline:</b> a <b>log1p, emb-only LightGBM over all 51 eligible
projects</b>, shipped with a self-referential <b>DI/AOA trust layer</b>. Emb-only feature-limited
(R²≈0.4); its value is coverage + calibrated applicability + a de-biased low end. Further discovery
branches from here.</div>

<h2>Current baseline</h2>
<table>
<tr><th>Item</th><th>Value</th></tr>
<tr><td>Data</td><td>51 ANEW projects, 12,636 plots (Quinte dropped)</td></tr>
<tr><td>Features</td><td>64 codec embeddings <code>emb_00..emb_63</code> (no GEE)</td></tr>
<tr><td>Target</td><td><code>CO2</code> tCO2/acre, raw; <b>log1p</b> transform</td></tr>
<tr><td>Predict</td><td><code>clip(expm1(booster.predict(X)), 0)</code></td></tr>
<tr><td>Model</td><td>LightGBM, num_leaves 31, lr 0.05, n_estimators 172; weighting none (S0)</td></tr>
<tr><td>Trust layer</td><td>CAST DI, AOA threshold 0.558, isotonic DI→expected-RMSE curve</td></tr>
<tr><td>Artifact</td><td><code>anew_emb51_log1p_model.txt</code> — data-space candidate, current baseline (not auto-promoted)</td></tr>
</table>

<h2>Headline metrics (LOPO)</h2>
<table>
<tr><th>Metric</th><th>S0 raw-L2</th><th>log1p (current)</th></tr>
<tr><td>RMSE (all)</td><td>73.7</td><td>77.3</td></tr>
<tr><td>bias, true&lt;100</td><td>+43.0</td><td class="win">+23.8</td></tr>
<tr><td>RMSE, true&lt;100</td><td>58.5</td><td class="win">44.4</td></tr>
<tr><td>zero-detection recall</td><td>0.54</td><td class="win">0.74</td></tr>
<tr><td>discrimination &lt;100 (Spearman)</td><td>0.574</td><td>0.599</td></tr>
<tr><td>separability AUC, true&lt;50</td><td>0.874</td><td>0.881</td></tr>
<tr><td>bias, true&gt;150 (accepted trade)</td><td>−65.6</td><td>−87.7</td></tr>
</table>

<h2>How we got here</h2>
<p><b>1. DI/AOA applicability</b> — the GT is one dense broadleaf interior (44 projects, highly
redundant) plus a small conifer/tundra frontier; a regional-dependence axis separates
"no-nearby-analogue" frontiers from intrinsically isolated ones. This justified a single global
model and the project/bloc/biome CV grouping.</p>
<p><b>2. Emb-only model + trust</b> — LightGBM ≈ XGBoost ≈ ridge (ceiling is feature-driven, not
learner-driven); frontier-aware weighting did not help (S0 ships); trust layer maps DI → expected error.</p>
<p><b>3. Low-end de-biasing</b> — log1p halves the &lt;100 over-prediction and improves
zero-detection; the two-stage hurdle did not help (separability already exploited); post-hoc
calibration cannot de-bias (it re-imposes the conditional mean).</p>

<h2>Figures</h2>
{fig_html}

<h2>Known ceiling & next discovery</h2>
<p>The residual +24 low-end bias and the range compression are the floor reachable on emb-only
features. Highest-leverage next step: <b>GEE extraction of CHM/topo/disturbance for the 28
non-modelled projects → full-feature model over all 51</b>, adding the vertical-structure signal
needed to separate biomass <i>level</i> within similar-spectra stands. Evaluate any such work
against the metrics above on the same LOPO/bloc/biome CV.</p>

<h2>References</h2>
<ul>
<li>Trust toolkit: <code>scripts/trust/</code></li>
<li>GT applicability: <code>scripts/anew_gt_applicability/</code>, <code>experiments/agb_anew_gt_applicability_20260626/</code></li>
<li>Model + low-end: <code>scripts/anew_emb_model/</code>, <code>experiments/agb_anew_emb_weighted_20260630/</code></li>
<li>Data-space outputs (gitignored): <code>/home/mattc/data-space/carbonmap-embeddings/agb_anew_emb_weighted_20260630/</code></li>
</ul>
</body></html>
"""
    OUT.write_text(html)
    kb = OUT.stat().st_size / 1024
    print(f"Wrote {OUT} ({kb:.0f} KB, {len(FIGURES)} figures embedded)")


if __name__ == "__main__":
    main()
