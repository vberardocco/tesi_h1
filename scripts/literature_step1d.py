"""Step 1d (literature comparison): saturation curve with the number of channels, S1 only (decided at Step 0), rule R6.

N = 20, 40, ..., 160 active channels, 10 draws per N (seeds 0-9; the same channels for the three measures):
- components for 80% (pipeline 1b: counts, Gaussian σ = 30 ms, no z-score, trial bins);
- variance of the first 12 PCs (pipeline 1a: square root, Gaussian σ = 50 ms, no z-score);
- FA dimensions for 95% of the shared variance (pipeline 1c: z-scored counts at 45 ms); K by
  5-fold contiguous-block cross-validation on 3 draws (seeds 0-2), grid 2-20 step 2 restricted to the Ks that
  satisfy the Ledermann bound (N − K)² ≥ N + K.
Three models (linear, power, exponential plateau) are fitted to the 8 medians and the AICs are compared:
"saturates" if the plateau has an AIC lower by at least 2 than both the others (R6).

Outputs: ~/thesis_h1_work/results/literature/step1d_draws.csv, step1d_models.csv, step1_saturation.png;
rows in the run registry. Usage: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=~/thesis_h1_git/src python scripts/literature_step1d.py
"""
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                               # noqa: E402
import numpy as np                                                            # noqa: E402
import pandas as pd                                                           # noqa: E402
from joblib import Parallel, delayed                                          # noqa: E402
from sklearn.model_selection import KFold                                     # noqa: E402

from thesis_h1 import palette, results, literature                                  # noqa: E402

OUT = results.RESULTS / "literature"
OUT.mkdir(parents=True, exist_ok=True)
N_LIST = list(range(20, 161, 20))
SEMI, SEMI_CV = range(10), range(3)
GRID = list(range(2, 21, 2))
N_JOBS = 6
ONLY = "--only-figure" in sys.argv          # redraw only the figure from step1d_draws.csv (no FA, no run registry)
T0 = time.time()


def log(msg):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {msg}", flush=True)


def zscore(X):
    return (X - X.mean(0)) / X.std(0)


def cv_one(X, K, tr, te):
    return literature.fa_fit(X[tr], K).score(X[te])


def dim95(X, K):
    return literature.dim_shared(literature.fa_fit(X, K))


if ONLY:
    df = pd.read_csv(OUT / "step1d_draws.csv")
else:
    s = literature.load("S1")
    m = s.mask
    Ya = literature.pipeline_gallego(s.counts)[m]
    Yb = literature.pipeline_silvernagel(s.counts)[m]
    X45, _ = literature.counts_45ms(s)
    Ntot = s.counts.shape[1]
    log(f"S1: {Ntot} channels, {m.sum()} trial bins, {len(X45)} bins of 45 ms")

    rows, kstar = [], {}
    with Parallel(n_jobs=N_JOBS) as par:
        for N in N_LIST:
            cols = {seed: literature.extract(Ntot, N, np.random.default_rng(seed)) for seed in SEMI}
            grid = [K for K in GRID if literature.ledermann(N, K)]
            folds = list(KFold(n_splits=5, shuffle=False).split(X45))
            ll = par(delayed(cv_one)(zscore(X45[:, cols[seed]]), K, tr, te)
                     for seed in SEMI_CV for K in grid for tr, te in folds)
            ll = np.array(ll).reshape(len(SEMI_CV), len(grid), 5).mean(2).mean(0)
            K = grid[int(np.argmax(ll))]
            kstar[N] = K
            unique = {tuple(c) for c in cols.values()}                       # with N = 160 the draws coincide
            d = dict(zip(unique, par(delayed(dim95)(zscore(X45[:, list(c)]), K) for c in unique)))
            for seed in SEMI:
                c = cols[seed]
                rows.append(dict(N=N, seed=seed, K_star=K, var12=literature.r1(Ya[:, c]),
                                  n80=literature.n_per(literature.eigenvalues(Yb[:, c]), 0.80), dim95=d[tuple(c)]))
            sel = [r for r in rows if r["N"] == N]
            log(f"N = {N:3d}: K* {K:2d} (grid {grid[0]}-{grid[-1]}); medians var12 "
                f"{np.median([r['var12'] for r in sel]):.1f}, n80 {np.median([r['n80'] for r in sel]):.1f}, "
                f"dim95 {np.median([r['dim95'] for r in sel]):.1f}")
            literature.log([{**r, "seed": r["seed"]} for r in sel], ["var12", "n80", "dim95", "K_star"],
                            experiment="literature", method=None, latent_dim=None, decoder=None, session=s.key,
                            bins="falcon", split="session", condition=f"1d_saturation_N{N}", fold=None, k_shot=None,
                            H_ms=None)

    df = pd.DataFrame(rows)
    df.to_csv(OUT / "step1d_draws.csv", index=False)

MEASURES = {"n80": "components for 80% (1b)", "var12": "variance of the first 12 PCs, % (1a)",
          "dim95": "FA dimensions for 95% shared (1c)"}
med = df.groupby("N")[list(MEASURES)].median()
models = []
fits = {}
for mis in MEASURES:
    out = literature.models_saturation(med.index.to_numpy(), med[mis].to_numpy())
    fits[mis] = out
    for name in ("linear", "power", "plateau"):
        models.append(dict(measure=mis, model=name, rss=out[name]["rss"], aic=out[name]["aic"],
                            params=" ".join(f"{p:.6g}" for p in out[name]["par"])))
    if ONLY:
        continue
    results.log_run({**{f"aic_{k}": out[k]["aic"] for k in ("linear", "power", "plateau")},
                     **{f"rss_{k}": out[k]["rss"] for k in ("linear", "power", "plateau")},
                     "R6_saturates": float(out["saturates"])},
                    experiment="literature", method=None, latent_dim=None, decoder=None, session="S1_set_1",
                    bins="falcon", split="session", condition=f"1d_saturation_{mis}", fold=None, k_shot=None,
                    H_ms=None, seed=None)
for mis, out in fits.items():
    log(f"{mis}: AIC linear {out['linear']['aic']:.2f}, power {out['power']['aic']:.2f}, plateau "
        f"{out['plateau']['aic']:.2f} -> {'saturates' if out['saturates'] else 'does not saturate in the observed range'}")
if not ONLY:
    pd.DataFrame(models).to_csv(OUT / "step1d_models.csv", index=False)

# Figure: H1 (S1) medians with [min, max], three models, literature values in gray
STILE = {"linear": (palette.CAT[0], "-"), "power": (palette.CAT[1], "--"), "plateau": (palette.CAT[2], ":")}
LIT = {"n80": [(96, 31, 38, "Silvernagel 2026\n(96 channels)", "below")],
        "var12": [(66, 60, 60, "Gallego 2018: ≥ 60%\n(66 channels)", "above")],
        "dim95": [(90, 5, 11, "Oby 2019\n(90 channels)", "below"), (75, 4, 8, "Degenhart 2020\n(75 channels)", "below left")]}
fig, axs = plt.subplots(1, 3, figsize=(12.5, 4.0))
xx = np.linspace(20, 160, 200)
for ax, (mis, lab) in zip(axs, MEASURES.items()):
    palette.style_axes(ax)
    g = df.groupby("N")[mis]
    ax.errorbar(med.index, med[mis], yerr=[med[mis] - g.min(), g.max() - med[mis]], fmt="o", ms=5,
                color=palette.INK, ecolor=palette.INK2, elinewidth=1, capsize=2, label="H1, S1: median [min, max]",
                zorder=3)
    for name, (col, ls) in STILE.items():
        p = fits[mis][name]["par"]
        y = (p[0] + p[1] * xx if name == "linear" else p[0] * xx ** p[1] if name == "power"
             else p[0] + p[1] * np.exp(-xx / p[2]))
        ax.plot(xx, y, color=col, ls=ls, lw=2, label=f"{name} (AIC {fits[mis][name]['aic']:.1f})")
    for x, lo, hi, txt, where in LIT[mis]:
        ax.plot([x, x], [lo, hi], color=palette.NEUTRAL, lw=6, alpha=0.5, solid_capstyle="butt", zorder=2)
        if lo == hi:
            ax.plot([x], [lo], marker="_", ms=16, mew=3, color=palette.NEUTRAL, zorder=2)
        if where == "above":
            ax.annotate(txt, (x, hi), xytext=(4, 4), textcoords="offset points", fontsize=7.5, color=palette.INK2)
        else:
            sx = where.endswith("left")
            ax.annotate(txt, (x, lo), xytext=(-4 if sx else 4, -4), textcoords="offset points", fontsize=7.5,
                        color=palette.INK2, va="top", ha="right" if sx else "left")
    verdict = "saturates" if fits[mis]["saturates"] else "does not saturate in the observed range"
    ax.set_title(f"{lab}\nR6: {verdict}", fontsize=9.5, color=palette.INK)
    ax.set_xlabel("number of channels N", fontsize=9, color=palette.INK2)
    ax.set_xticks(N_LIST)
    ax.legend(fontsize=7.5, frameon=False, loc="best")
fig.suptitle("Dimensionality and number of channels, S1 (10 draws per N)", fontsize=11, color=palette.INK)
fig.tight_layout()
fig.savefig(OUT / "step1_saturation.png", dpi=200, bbox_inches="tight")
log(f"saved {OUT / 'step1_saturation.png'}, step1d_draws.csv, step1d_models.csv")
