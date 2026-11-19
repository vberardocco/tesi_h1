"""Results figure on the latent structure: phase decoding and co-smoothing per method, in the three replication
sessions, with the common method colours (src/thesis_h1/palette.py).

No recomputation: the numbers are read from rep/tables/comparison.csv (extracted from the outputs of the replication
notebooks by rep/tables.py, FALCON bins). Columns used: "ref. S1", "S2", "S5". For CEBRA and LFADS the value is the mean
over seeds and the bar goes from the minimum to the maximum (for LFADS in co-smoothing, the interval reported in the table).

Output: results/replica/fig_latent_structure.png and latent_structure.csv (the numbers of the figure).
Usage: python scripts/fig_latent_structure.py
"""
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from thesis_h1 import palette as pal  # noqa: E402

SRC = REPO / "rep/tables/comparison.csv"
OUT = REPO / "results/replica"
SESS = ["ref. S1", "S2", "S5"]
PHASE = "§12: fase, tetto con tutti i 25 fattori (%)"
PHASE_A = "Fase riconosciuta (%), caso 12,5"
COBPS = "§12: bit/spike sui 40 canali nascosti, trial nuovi"
# (panel, section, measure in the table, label, family for the colour, hatching)
ROWS = [
    ("phase", PHASE, "25 PC smussati (parte A)", "PCA, 25 smoothed PCs", "PCA", None),
    ("phase", PHASE, "FA", "FA, 25 factors", "FA", None),
    ("phase", PHASE, "GPFA alla pari", "GPFA, 25 latents", "GPFA", None),
    ("phase", PHASE_A, "CEBRA-Time, 4 dimensioni", "CEBRA-Time, 4 dims", "CEBRA", None),
    ("phase", PHASE, "LFADS (5 semi)", "LFADS, 25 factors", "LFADS", None),
    ("cobps", COBPS, "smoothing + GLM (reference)", "smoothing + GLM (reference)", "reference", None),
    ("cobps", COBPS, "FA", "FA", "FA", None),
    ("cobps", COBPS, "GPFA alla pari", "GPFA", "GPFA", None),
    ("cobps", COBPS, "GPFA + regressione di Poisson", "GPFA + Poisson regression", "GPFA", "////"),
    ("cobps", COBPS, "LFADS (5 semi)", "LFADS", "LFADS", None),
]
CLOCK = (PHASE_A, "orologio (solo il tempo dall'inizio del trial)")
NUM = r"-?\d+(?:\.\d+)?"


def parse(cell):
    """'46.5 (44.7-47.7)' or '0.0388 [0.0383, 0.0393]' or '42.2' -> (value, minimum, maximum)."""
    s = str(cell).strip()
    m = re.fullmatch(rf"({NUM})\s*[\(\[]\s*({NUM})\s*[-,]\s*({NUM})\s*[\)\]]", s)
    if m:
        return tuple(float(x) for x in m.groups())
    v = float(re.fullmatch(NUM, s).group(0))
    return v, np.nan, np.nan


def table():
    T = pd.read_csv(SRC)
    T = T[T.notebook == 4]
    rows = []
    for panel, sec, mis, lab, fam, hatch in ROWS + [("clock", *CLOCK, "clock (time since trial start)", "reference", None)]:
        r = T[(T.section == sec) & (T.measure == mis)]
        assert len(r) == 1, (sec, mis, len(r))
        for s in SESS:
            v, lo, hi = parse(r.iloc[0][s])
            rows.append(dict(panel=panel, label=lab, family=fam, hatch=hatch, session=s, value=v, low=lo, high=hi,
                             source=f"{sec} / {mis}"))
    return pd.DataFrame(rows)


def figure(D):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.4), layout="constrained", sharex="row")
    titles = {"phase": "phase recognized on new trials (%)", "cobps": "co-smoothing, bits/spike (40 held-out channels)"}
    for i, panel in enumerate(("phase", "cobps")):
        labels = [r[3] for r in ROWS if r[0] == panel]
        for j, s in enumerate(SESS):
            ax = axes[i, j]
            pal.style_axes(ax)
            ax.grid(False, axis="y")
            d = D[(D.panel == panel) & (D.session == s)].set_index("label").loc[labels]
            y = np.arange(len(labels))[::-1]
            for yy, (lab, r) in zip(y, d.iterrows()):
                c = pal.color(r.family) if r.family != "reference" else pal.NEUTRAL
                ax.barh(yy, r.value, height=0.62, color=c if not isinstance(r.hatch, str) else "white",
                        edgecolor=c, hatch=r.hatch if isinstance(r.hatch, str) else None, lw=1.2)
                if np.isfinite(r.low):
                    ax.plot([r.low, r.high], [yy, yy], color=pal.INK, lw=1.2)
                fmt = f"{r.value:.1f}" if panel == "phase" else f"{r.value:.3f}"
                end = r.high if np.isfinite(r.high) else r.value
                ax.text(end, yy, "  " + fmt, va="center", ha="left", fontsize=7.5, color=pal.INK2)
            ax.set_yticks(y, labels if j == 0 else [""] * len(labels), fontsize=8.5, color=pal.INK)
            if panel == "phase":
                ax.axvline(12.5, color=pal.INK2, lw=1, ls=":")
                ck = D[(D.panel == "clock") & (D.session == s)].value.iloc[0]
                ax.axvline(ck, color=pal.NEUTRAL, lw=1.4, ls="--")
                ax.text(ck, len(labels) - 0.45, f"clock {ck:.1f}", fontsize=7.5, color=pal.INK2, ha="center")
                ax.text(12.5, len(labels) - 0.45, "chance", fontsize=7.5, color=pal.INK2, ha="center")
                ax.set_xlim(0, 72)
                ax.set_ylim(-0.6, len(labels) - 0.1)
            else:
                ax.set_xlim(0, D[D.panel == "cobps"][["value", "high"]].max().max() * 1.25)
            ax.set_title(f"{s}: {titles[panel]}", fontsize=9, color=pal.INK, loc="left")
    fig.suptitle("Latent structure in the three replica sessions (FALCON bins, same hyperparameters). Bars: mean over "
                 "seeds; black lines: min–max over seeds (CEBRA 5 seeds, LFADS 5 seeds).\nDotted line = chance (12.5%); "
                 "dashed line = clock, i.e. the phase guessed from the time since trial start only.", fontsize=8.5,
                 color=pal.INK, x=0.01, ha="left")
    fig.savefig(OUT / "fig_latent_structure.png", dpi=160)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    D = table()
    D.to_csv(OUT / "latent_structure.csv", index=False)
    figure(D)
    print(f"{OUT / 'fig_latent_structure.png'}; {len(D)} numbers from {SRC.relative_to(REPO)}")
