"""Step 5d (literature comparison, optional, approved at STOP 3): GPFA on 61 channels, for the comparison with Yu 2009.

S1 only (time: about 27 minutes for 10 draws; rule 8 forbids fewer than 10 draws). The "61 channels" option is used
and not "a single phase": per phase there are about 15 windows per session, too few for
5-fold cross-validation.
As in notebook 02 (§ 10): the 119 windows of 1 s from the start of each movement phase, square root of the
counts at 20 ms (FALCON bins), elephant GPFA with 500 iterations and fixed seed; 5-fold contiguous-block
cross-validation of windows, score = log-likelihood per bin on the held-out windows; d chosen with the rule of the
notebook (the smallest d ≤ best d that the best does not beat in all blocks), and also the best d.
61 channels drawn at random among the 160 active ones, 10 draws (seeds 0-9); grid d = 2, 5, 8, 10, 12, 15, 20, 25.
Reference: the replication chooses d = 25 with all 160 channels. Yu 2009: p* = 10 (8-12 per target), with 61 units
and a different criterion (leave-neuron-out error). In the tests a linear GPFA on Poisson data with an exponential
chooses more factors than the true latents: the chosen d is a ceiling for the latent dimension.

Outputs: ~/thesis_h1_work/results/literature/step5d_gpfa.csv and step5d_gpfa.md; rows in the run registry.
Usage: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=~/thesis_h1_git/src python scripts/literature_step5d.py
"""
import pickle
import sys
import time

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.model_selection import KFold

from thesis_h1 import io, results, literature

OUT = results.RESULTS / "literature"
OUT.mkdir(parents=True, exist_ok=True)
GRID = [2, 5, 8, 10, 12, 15, 20, 25]
SEMI, N_CHANNELS, N_JOBS = range(10), 61, 6
ONLY_TABLE = "--only-table" in sys.argv    # rewrite only step5d_gpfa.md from step5d_gpfa.csv (no estimates, no run registry)
T0 = time.time()


def log(msg):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {msg}", flush=True)


def one(X, fin, col, d, tr, te):
    p = literature.gpfa_fit(literature.sequences_gpfa(X, [fin[i] for i in tr], col), d)
    return literature.gpfa_ll_per_bin(p, literature.sequences_gpfa(X, [fin[i] for i in te], col))


s = literature.load("S1")
ses = io.load_session(literature.DATA / "sub-HumanPitt-held-in-calib" / f"sub-HumanPitt-held-in-calib_ses-{s.sid}.nwb",
                      bins="falcon")
fin = io.phase_windows(ses, win_s=1.0, align="start")
X = s.counts
d_ref = int(pickle.load(open(literature.REP / s.sid / "02_rep.pkl", "rb"))["numbers"]["D_GP"])
folds = list(KFold(n_splits=5, shuffle=False).split(np.arange(len(fin))))
cols = {seed: literature.extract(X.shape[1], N_CHANNELS, np.random.default_rng(seed)) for seed in SEMI}
log(f"S1: {X.shape[1]} active channels, {len(fin)} windows; replication reference with all channels: d = {d_ref}")

if ONLY_TABLE:
    df = pd.read_csv(OUT / "step5d_gpfa.csv")
else:
    tasks = [(seed, d, k) for seed in SEMI for d in GRID for k in range(5)]
    ll = Parallel(n_jobs=N_JOBS, verbose=0)(delayed(one)(X, fin, cols[seed], d, *folds[k]) for seed, d, k in tasks)
    LL = np.array(ll).reshape(len(SEMI), len(GRID), 5)
    log("estimates done")
    rows = []
    for i, seed in enumerate(SEMI):
        best, d = literature.choose_d(LL[i], GRID)
        rows.append(dict(seed=seed, n_channels=N_CHANNELS, d_best=best, d_chosen=d,
                          **{f"ll_d{g}": float(v) for g, v in zip(GRID, LL[i].mean(1))}))
        log(f"draw {seed}: best d {best}, chosen d {d}")
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "step5d_gpfa.csv", index=False)
    literature.log(rows, ["d_best", "d_chosen", "n_channels"] + [f"ll_d{g}" for g in GRID],
                    experiment="literature", method="GPFA", latent_dim=None, decoder=None, session=s.key,
                    bins="falcon", split="cv5_windows", condition="5d_gpfa_61channels", fold=None, k_shot=None,
                    H_ms=None)
summ = {f"{m}_{k}": f(df[m]) for m in ("d_best", "d_chosen")
        for k, f in (("median", np.median), ("min", np.min), ("max", np.max))}
al_edge = int((df.d_best == GRID[-1]).sum())
md = ["# Step 5d · GPFA on 61 channels (optional, S1 only)", "",
      f"Commit {results.code_version()[0]}. Procedure of notebook 02 (§ 10.2) on {N_CHANNELS} channels drawn at random "
      f"among the 160 active ones, 10 draws; {len(fin)} windows of 1 s, square root of the counts, 500 iterations, 5 contiguous "
      "blocks of windows, score = log-likelihood per bin on the held-out windows.", "",
      "| | chosen d (notebook rule) | d with the highest mean score |", "| --- | --- | --- |",
      f"| H1, S1, 61 channels (median [min-max] over 10 draws) | {summ['d_chosen_median']:.1f} "
      f"[{summ['d_chosen_min']:.0f}-{summ['d_chosen_max']:.0f}] | {summ['d_best_median']:.1f} "
      f"[{summ['d_best_min']:.0f}-{summ['d_best_max']:.0f}] |",
      f"| H1, S1, 160 channels (replication) | {d_ref} | — |",
      "| Yu 2009, 61 units (18 single units + 43 multiunits), leave-neuron-out | p* = 10 (8-12 per target) | — |", "",
      "Mean score on the held-out windows, per d (median over draws, difference from d = 2):", "",
      "| " + " | ".join(f"d = {g}" for g in GRID) + " |", "| " + " | ".join("---" for _ in GRID) + " |",
      "| " + " | ".join(f"{np.median(df[f'll_d{g}'] - df['ll_d2']):+.4f}" for g in GRID) + " |", "",
      "Caveats: Yu 2009 chooses p* with the leave-neuron-out error, here with the log-likelihood on the "
      "held-out windows; Yu estimates one target at a time, here all phases together. In the tests (tests/test_literature.py) the "
      "same procedure on Poisson counts with an exponential of 3 latents chooses 8 factors: with nonlinear "
      "data the d chosen by cross-validation is a ceiling for the latent dimension, not an estimate."]
if al_edge:
    md += ["", f"**At the grid edge**: the highest mean score is at d = {GRID[-1]} in {al_edge} draws out of "
           f"{len(df)}, and the curve still rises between d = {GRID[-2]} and {GRID[-1]}. The d of H1 with 61 channels is therefore "
           "at least the one reported: a lower bound, not a closed estimate. To find the maximum, higher values "
           "are needed (notebook 02 went up to 50)."]
(OUT / "step5d_gpfa.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
