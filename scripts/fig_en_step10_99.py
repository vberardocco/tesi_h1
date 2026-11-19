"""English figure of the Isomap advantage on the single trial with 99 twins (mean and 95% range), from the CSVs of
literature_step10_twins99.py. Same style as step10_isomap_single_en.png.
Usage: PYTHONPATH=~/thesis_h1_git/src python scripts/fig_en_step10_99.py <output folder>
"""
import sys
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from thesis_h1 import palette, results

D_FIG = 10
c = pd.read_csv(results.RESULTS / "literature" / "step10_single_99_curve.csv")
fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.8), sharey=True)
for ax, (name, g) in zip(axs, c.groupby("session", sort=False)):
    palette.style_axes(ax)
    g = g[g.d <= D_FIG]
    ax.fill_between(g.d, g.twins_p2_5, g.twins_p97_5, color=palette.NEUTRAL, alpha=0.25, lw=0,
                    label="Gaussian twins (99), 95% range")
    ax.plot(g.d, g.twins_mean, color=palette.NEUTRAL, lw=1.4, label="Gaussian twins (99), mean")
    ax.plot(g.d, g.data, "o-", color=palette.color("Isomap"), lw=2, ms=4, label="Data")
    ax.axhline(0, color=palette.INK2, lw=0.8)
    ax.set_title(name, fontsize=10.5, color=palette.INK)
    ax.set_xlabel("Embedding dimension d", fontsize=9, color=palette.INK2)
axs[0].set_ylabel("Residual variance, PCA − Isomap (points)", fontsize=9, color=palette.INK2)
axs[0].legend(fontsize=7.5, frameon=False, loc="lower left")
fig.tight_layout()
out = Path(sys.argv[1]) / "step10_isomap_single_99_en.png"
fig.savefig(out, dpi=200, bbox_inches="tight")
print(out)
