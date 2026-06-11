"""Dual-scale figure: masked our-values vs Deep Biomass (year-matched) + masked-vs-unmasked panel.
Primary axis tCO2/acre; secondary axis Mg/ha (= tCO2/acre / 0.6977)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

EXPDIR = Path(
    "/home/mattc/code/agb-ml-agent-evolve/experiments/agb_ireland_biomass_regression_20260608"
)
FINAL = EXPDIR / "final"
YEARS = [2022, 2023, 2024]
MGHA = 1.0 / 0.6977

ym = pd.read_parquet(FINAL / "ireland_agb_yearmatched.parquet")


def add_mgha_axis(ax, xaxis=True):
    sec = ax.secondary_yaxis("right", functions=(lambda v: v * MGHA, lambda v: v / MGHA))
    sec.set_ylabel("Mg/ha")
    if xaxis:
        ax.secondary_xaxis("top", functions=(lambda v: v * MGHA, lambda v: v / MGHA))


fig, axes = plt.subplots(2, 3, figsize=(17, 11))

lims = [
    0,
    max(
        ym[[f"our_{y}_masked_tCO2_acre" for y in YEARS]].max().max(),
        ym[[f"db_{y}_tCO2_acre" for y in YEARS]].max().max(),
    )
    * 1.05,
]

# row 0: masked vs DB per year
for ax, y in zip(axes[0], YEARS):
    ax.scatter(
        ym[f"db_{y}_tCO2_acre"], ym[f"our_{y}_masked_tCO2_acre"], s=20, alpha=0.6, color="C0"
    )
    ax.plot(lims, lims, "k--", lw=1, label="1:1")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.set_xlabel(f"Deep Biomass {y} (tCO2/acre)")
    ax.set_ylabel(f"Our MASKED {y} (tCO2/acre)")
    m = ym[f"our_{y}_masked_tCO2_acre"].mean()
    db = ym[f"db_{y}_tCO2_acre"].mean()
    ax.set_title(f"{y}: masked ratio {m / db:.2f}x (mean ff {ym[f'forest_frac_{y}'].mean():.2f})")
    ax.legend(loc="upper left", fontsize=8)
    add_mgha_axis(ax)

# row 1 panel 0: portfolio trajectory masked vs unmasked vs DB
axt = axes[1, 0]
um = [ym[f"our_{y}_tCO2_acre"].mean() for y in YEARS]
mm = [ym[f"our_{y}_masked_tCO2_acre"].mean() for y in YEARS]
db = [ym[f"db_{y}_tCO2_acre"].mean() for y in YEARS]
axt.plot(YEARS, um, "o--", color="C7", label="Our UNmasked")
axt.plot(YEARS, mm, "o-", color="C0", label="Our MASKED")
axt.plot(YEARS, db, "s-", color="C3", label="Deep Biomass")
for arr, dy in [(um, 8), (mm, -14), (db, 6)]:
    for xx, yy in zip(YEARS, arr):
        axt.annotate(f"{yy:.0f}", (xx, yy), textcoords="offset points", xytext=(0, dy), fontsize=8)
axt.set_xticks(YEARS)
axt.set_xlabel("Year")
axt.set_ylabel("Portfolio mean (tCO2/acre)")
axt.set_title("Portfolio trajectory: masked vs unmasked vs DB")
axt.legend(fontsize=8)
add_mgha_axis(axt, xaxis=False)

# row 1 panel 1: 3yr-mean masked vs DB
axm = axes[1, 1]
axm.scatter(
    ym["db_mean_2022_24_tCO2_acre"],
    ym["our_mean_2022_24_masked_tCO2_acre"],
    s=20,
    alpha=0.6,
    color="C2",
)
axm.plot(lims, lims, "k--", lw=1, label="1:1")
axm.set_xlim(lims)
axm.set_ylim(lims)
axm.set_xlabel("Deep Biomass 2022-24 mean (tCO2/acre)")
axm.set_ylabel("Our MASKED 2022-24 mean (tCO2/acre)")
rm = ym["our_mean_2022_24_masked_tCO2_acre"].mean() / ym["db_mean_2022_24_tCO2_acre"].mean()
nbelow = (ym["our_mean_2022_24_masked_tCO2_acre"] < ym["db_mean_2022_24_tCO2_acre"]).sum()
axm.set_title(f"3yr mean: masked ratio {rm:.2f}x; {nbelow} stands below DB")
axm.legend(loc="upper left", fontsize=8)
add_mgha_axis(axm)

# row 1 panel 2: masked vs unmasked our-values (3yr mean), coloured by forest_fraction
axu = axes[1, 2]
ff = ym[[f"forest_frac_{y}" for y in YEARS]].mean(axis=1)
sc = axu.scatter(
    ym["our_mean_2022_24_tCO2_acre"],
    ym["our_mean_2022_24_masked_tCO2_acre"],
    c=ff,
    cmap="viridis",
    s=24,
    alpha=0.85,
    vmin=0,
    vmax=1,
)
ul = [0, ym["our_mean_2022_24_tCO2_acre"].max() * 1.05]
axu.plot(ul, ul, "k--", lw=1, label="1:1 (no mask effect)")
axu.set_xlim(ul)
axu.set_ylim(ul)
axu.set_xlabel("Our UNmasked 3yr mean (tCO2/acre)")
axu.set_ylabel("Our MASKED 3yr mean (tCO2/acre)")
axu.set_title("Masking pulls low-forest-fraction stands down")
axu.legend(loc="upper left", fontsize=8)
cb = fig.colorbar(sc, ax=axu, fraction=0.046, pad=0.10)
cb.set_label("mean forest fraction")
add_mgha_axis(axu)

fig.suptitle(
    "Ireland AGB: forest-MASKED model vs Deep Biomass (year-matched 2022/2023/2024)\n"
    "primary axes tCO2/acre, secondary axes Mg/ha",
    fontsize=13,
)
fig.tight_layout(rect=[0, 0, 1, 0.95])
(FINAL / "figures").mkdir(parents=True, exist_ok=True)
out = FINAL / "figures/ireland_vs_deepbiomass_yearmatched_masked.png"
fig.savefig(out, dpi=130)
print("Wrote", out)
