"""Step 4 (literature comparison): robustness checks.

4a · 1 Hz threshold (Cunningham & Yu 2014), S1 only: channels lost relative to 0.5 Hz; PCA PR with the pipeline
     of notebook 02 (Gaussian σ = 50 ms, z-score, all bins, all active channels); K* of the FA by
     cross-validation as in § 9.1 of notebook 02 (z-scored raw counts at 20 ms, grid 2, 5, 10, 15,
     20, 30, 5 contiguous blocks, svd_method="lapack"); dimensions for 95% of the shared variance with K*.
     The same computation at 0.5 Hz serves as reference (expected: PR 106.1, K* 15, 14 dimensions).
4b · Per-condition means (§ 8 of notebook 02), S1, S2, S5: 16 conditions (8 movement phases × odd
     and even trials), 1 s windows from the start of the phase on smoothed (σ = 50 ms) and z-scored data, 800 × N matrix;
     cumulative variance of the first 2, 6 and 10 PCs. The matrix is compared with the replication PSTH (02_rep.pkl).
     The first 2 PCs give the maximum variance on a plane: they are a ceiling for any jPCA plane.

Outputs: ~/thesis_h1_work/results/literature/step4_robustness.csv and step4_robustness.md; rows in the run registry.
Usage: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=~/thesis_h1_git/src python scripts/literature_step4.py
"""
import pickle
import sys
import time

import numpy as np
import pandas as pd

from thesis_h1 import io, results, literature

OUT = results.RESULTS / "literature"
OUT.mkdir(parents=True, exist_ok=True)
GRID = [2, 5, 10, 15, 20, 30]
T0 = time.time()
KEYS = dict(experiment="literature", latent_dim=None, decoder=None, bins="falcon", split="session", fold=None,
            k_shot=None, H_ms=None, seed=None)


def log(msg):
    print(f"[{(time.time() - T0) / 60:4.1f} min] {msg}", flush=True)


def zscore(X):
    return (X - X.mean(0)) / X.std(0)


def session(name):
    sid = literature.SESSIONS[name]
    return io.load_session(literature.DATA / "sub-HumanPitt-held-in-calib" / f"sub-HumanPitt-held-in-calib_ses-{sid}.nwb",
                           bins="falcon")


# 4a · 1 Hz threshold, S1
s = session("S1")
rows_a = []
for hz in (0.5, 1.0):
    act = io.active_units(s, min_hz=hz)
    X = s.spikes[:, act].astype(float)
    ev = literature.eigenvalues(literature.pipeline_02(X), zscore=True)
    cv = literature.fa_cv(zscore(X), GRID, n_jobs=6)
    K = GRID[int(np.argmax(cv))]
    f = literature.fa_fit(zscore(X), K)
    r = dict(literature_hz=hz, n_channels=len(act), PR=literature.participation_ratio(ev), n80=literature.n_per(ev, 0.8), K_star=K,
             dim95=literature.dim_shared(f), **{f"ll_K{k}": float(v) for k, v in zip(GRID, cv)})
    rows_a.append(r)
    results.log_run({k: v for k, v in r.items() if k != "literature_hz"}, method="PCA+FA", session="S1_set_1",
                    condition=f"4a_literature_{hz:g}hz", **KEYS)
    log(f"4a, {hz:g} Hz: {len(act)} channels, PR {r['PR']:.1f}, K* {K}, dimensions for 95% {r['dim95']}")
ref, one = rows_a
if round(ref["PR"], 1) != 106.1 or ref["dim95"] != 14:
    sys.exit("4a: the 0.5 Hz reference does not reproduce PR 106.1 and 14 dimensions: stopping")
lost = np.setdiff1d(io.active_units(s, 0.5), io.active_units(s, 1.0))
log(f"4a: lost {len(lost)} channels (indices {lost.tolist()})")

# 4b · per-condition means, S1, S2, S5
rows_b = []
for name, sid in literature.SESSIONS.items():
    s = session(name)
    act = io.active_units(s, 0.5)
    Z = zscore(literature.pipeline_02(s.spikes[:, act]))
    win = io.phase_windows(s, win_s=1.0, align="start")
    conds = [(ph, v) for ph in io.MOVEMENT_PHASES for v in ("odd", "even")]
    reps = {c: np.stack([Z[w["start"]:w["stop"]] for w in win
                         if (w["tag"], "odd" if w["trial"] % 2 == 1 else "even") == c]) for c in conds}
    PSTH = np.concatenate([reps[c].mean(0) for c in conds])
    ref_psth = pickle.load(open(literature.REP / sid / "02_rep.pkl", "rb"))["arrays"]["PSTH"]
    diff = float(np.abs(PSTH - ref_psth).max())
    if diff > 1e-6:
        sys.exit(f"4b, {name}: PSTH differs from the replication one (maximum difference {diff:.2e}): stopping")
    ev = literature.eigenvalues(PSTH)
    r = dict(session=name, n_channels=len(act), n_rows=len(PSTH), repetitions_min=min(len(v) for v in reps.values()),
             repetitions_max=max(len(v) for v in reps.values()), var2=100 * literature.share_first(ev, 2),
             var3=100 * literature.share_first(ev, 3), var6=100 * literature.share_first(ev, 6),
             var10=100 * literature.share_first(ev, 10), PR=literature.participation_ratio(ev), diff_replica=diff)
    rows_b.append(r)
    results.log_run({k: v for k, v in r.items() if k != "session"}, method="PCA", session=literature.KEYS[name],
                    condition="4b_means_condition", **KEYS)
    log(f"4b, {name}: PSTH {PSTH.shape} (equal to the replication, diff {diff:.1e}); first 2 PCs {r['var2']:.1f}%, "
        f"6 PC {r['var6']:.1f}%, 10 PC {r['var10']:.1f}%")

A, B = pd.DataFrame(rows_a), pd.DataFrame(rows_b)
pd.concat([A.assign(analysis="4a"), B.assign(analysis="4b")]).to_csv(OUT / "step4_robustness.csv", index=False)
md = ["# Step 4 · Robustness checks", "", f"Commit {results.code_version()[0]}.", "",
      "## 4a · Activity threshold of 1 Hz instead of 0.5 Hz (S1)", "",
      "Cunningham & Yu 2014 (p. 1506): neurons below about 1 spike/s \"should typically be excluded\".", "",
      "| Threshold | channels | PR (σ 50 ms, z-score) | components for 80% | K* (FA, CV) | FA dimensions for 95% |",
      "| --- | --- | --- | --- | --- | --- |"]
for r in rows_a:
    md.append(f"| {r['literature_hz']:g} Hz | {r['n_channels']} | {r['PR']:.1f} | {r['n80']} | {r['K_star']} | {r['dim95']} |")
md += ["", f"Differences (1 Hz − 0.5 Hz): channels {one['n_channels'] - ref['n_channels']:+d}, PR {one['PR'] - ref['PR']:+.1f}, "
       f"components for 80% {one['n80'] - ref['n80']:+d}, K* {one['K_star'] - ref['K_star']:+d}, dimensions for 95% "
       f"{one['dim95'] - ref['dim95']:+d}. Channels lost (original indices): {lost.tolist()}.", "",
       "Cross-validation log-likelihood per K (mean over the 5 blocks):", "",
       "| Threshold | " + " | ".join(f"K = {k}" for k in GRID) + " |", "| --- | " + " | ".join("---" for _ in GRID) + " |"]
for r in rows_a:
    md.append(f"| {r['literature_hz']:g} Hz | " + " | ".join(f"{r[f'll_K{k}']:.4f}" for k in GRID) + " |")
md += ["", "## 4b · Per-condition means (16 conditions × 50 bins, z-score, σ 50 ms)", "",
       "| Session | channels | repetitions per condition | first 2 PCs | first 3 PCs | first 6 PCs | first 10 PCs | PR |",
       "| --- | --- | --- | --- | --- | --- | --- | --- |"]
for r in rows_b:
    md.append(f"| {r['session']} | {r['n_channels']} | {r['repetitions_min']}-{r['repetitions_max']} | {r['var2']:.1f}% | "
              f"{r['var3']:.1f}% | {r['var6']:.1f}% | {r['var10']:.1f}% | {r['PR']:.1f} |")
md += ["", "Comparisons: Churchland 2012 uses 6 PCs (k = 6) and 2-3 jPCA planes capture 50-70% of the total variance; "
       "Pandarinath 2015: the first jPCA plane captures 61% (T6) and 27% (T7), 28% on average in the macaque. The variance "
       "of the first 2 PCs is the maximum on any plane, hence it is a ceiling for the first jPCA plane. Conditions, "
       "windows and preprocessing differ (16 conditions, 1 s from the start of the phase, z-score): the comparison is "
       "only indicative."]
(OUT / "step4_robustness.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
