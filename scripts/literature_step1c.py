"""Step 1c (literature comparison): Oby 2019 (90 channels) and Degenhart 2020 (75 channels), rules R3 and R4.

For S1, S2 and S5: counts rebinned at 45 ms from the spike times (raw counts, only whole bins with the
center in the trial bins). FA with svd_method="lapack"; K by 5-fold contiguous-block cross-validation on the
z-scored counts, grid 2-20 step 2 (notebook 02, § 9.1). K* = maximum of the mean curve over 5
draws (seeds 0-4), then used for all 20 draws (seeds 0-19, the same channels).
For each draw:
- dimensions for 95% of the shared variance (eigenvalues of W Wᵀ) with K*, K = 10 and K = 20, on z-scored
  counts (convention of notebook 02 and of the 14-dimension check); for comparison also without z-score;
- population shared fraction trace(W Wᵀ) / trace(W Wᵀ + Ψ) with K*, on non-z-scored counts (R4);
- median of the per-channel shared fraction (with K*; it does not depend on the z-score).

Outputs: ~/thesis_h1_work/results/literature/step1c_draws.csv and step1c_cv.csv; rows in the run registry.
Usage: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=~/thesis_h1_git/src python scripts/literature_step1c.py
"""
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import KFold

from thesis_h1 import results, literature

OUT = results.RESULTS / "literature"
OUT.mkdir(parents=True, exist_ok=True)
GRID = list(range(2, 21, 2))
SEMI_CV, SEMI = range(5), range(20)
FONTI = {"oby": 90, "degenhart": 75}
N_JOBS = 6
T0 = time.time()


def log(msg):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {msg}", flush=True)


def zscore(X):
    return (X - X.mean(0)) / X.std(0)


def cv_one(X, K, tr, te):
    return literature.fa_fit(X[tr], K).score(X[te])


def estimates(X, K_star):
    """The FAs of one draw: K*, 10, 20 on z-score; K* without z-score."""
    Xz = zscore(X)
    out = {}
    for K in sorted({K_star, 10, 20}):
        f = literature.fa_fit(Xz, K)
        out[f"dim95_K{K}"] = literature.dim_shared(f)
        if K == K_star:
            out.update(dim95_Kstar=literature.dim_shared(f), share_channels_median=float(np.median(literature.share_channels(f))),
                       share_pop_z=literature.share_population(f), pr_shared=literature.participation_ratio(
                           literature.eigenvalues_shared(f)), n_iter=f.n_iter_)
    f = literature.fa_fit(X, K_star)
    out.update(share_pop=literature.share_population(f), dim95_Kstar_noz=literature.dim_shared(f))
    return out


rows, curve = [], []
with Parallel(n_jobs=N_JOBS) as par:
    for name in literature.SESSIONS:
        s = literature.load(name)
        X45, m45 = literature.counts_45ms(s)
        log(f"{name}: {X45.shape[0]} trial bins of 45 ms, {X45.shape[1]} channels, {int(X45.sum())} spikes")
        for source, n in FONTI.items():
            cols = {seed: literature.extract(X45.shape[1], n, np.random.default_rng(seed)) for seed in SEMI}
            folds = list(KFold(n_splits=5, shuffle=False).split(X45))
            Xz = {seed: zscore(X45[:, cols[seed]]) for seed in SEMI_CV}
            ll = par(delayed(cv_one)(Xz[seed], K, tr, te) for seed in SEMI_CV for K in GRID for tr, te in folds)
            ll = np.array(ll).reshape(len(SEMI_CV), len(GRID), 5).mean(2)
            K_star = GRID[int(np.argmax(ll.mean(0)))]
            K_draw = [GRID[int(i)] for i in ll.argmax(1)]
            for i, seed in enumerate(SEMI_CV):
                curve += [dict(session=name, source=source, n_channels=n, seed=seed, K=K, ll=ll[i, j])
                          for j, K in enumerate(GRID)]
            keys = dict(experiment="literature", method="FA", latent_dim=None, decoder=None, session=s.key,
                        bins="falcon", split="cv5_contiguous", condition=f"1c_{source}_cv", fold=None, k_shot=None,
                        H_ms=None, seed=None)
            results.log_run({"K_star": K_star, **{f"ll_mean_K{K}": float(v) for K, v in zip(GRID, ll.mean(0))},
                             **{f"K_star_seed{seed}": K for seed, K in zip(SEMI_CV, K_draw)}}, **keys)
            log(f"  {source} ({n} channels): K* = {K_star} (per draw: {K_draw})")
            res = par(delayed(estimates)(X45[:, cols[seed]], K_star) for seed in SEMI)
            recs = [dict(seed=seed, n_channels=n, K_star=K_star, n_bin=len(X45), **r) for seed, r in zip(SEMI, res)]
            metrics = ["dim95_Kstar", "dim95_K10", "dim95_K20", "dim95_Kstar_noz", "share_pop", "share_pop_z",
                        "share_channels_median", "pr_shared", "n_iter", "K_star", "n_channels", "n_bin"]
            summ = literature.log(recs, metrics, experiment="literature", method="FA", latent_dim=K_star,
                                   decoder=None, session=s.key, bins="falcon", split="session",
                                   condition=f"1c_{source}", fold=None, k_shot=None, H_ms=None)
            for r in recs:
                r.update(session=name, source=source)
            rows += recs
            log(f"  {source}: dimensions for 95% {summ['dim95_Kstar_median']:.1f} "
                f"[{summ['dim95_Kstar_min']:.0f}-{summ['dim95_Kstar_max']:.0f}] con K*; K=10 "
                f"{summ['dim95_K10_median']:.1f}, K=20 {summ['dim95_K20_median']:.1f}; without z-score "
                f"{summ['dim95_Kstar_noz_median']:.1f}; population shared fraction {summ['share_pop_median']:.3f}; "
                f"per-channel shared fraction {summ['share_channels_median_median']:.3f}")

pd.DataFrame(curve).to_csv(OUT / "step1c_cv.csv", index=False)
df = pd.DataFrame(rows)
cols = ["session", "source", "n_channels", "seed", "K_star"]
df[cols + [c for c in df.columns if c not in cols]].to_csv(OUT / "step1c_draws.csv", index=False)
log(f"saved {OUT / 'step1c_draws.csv'} and step1c_cv.csv")
