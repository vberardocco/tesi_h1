"""Step 1a, 1b and 1e (literature comparison): Gallego 2018, Silvernagel 2026 and PCs for 60%, with the circular-shift null.

For S1, S2 and S5, trial bins (without Intertrial and without the bin with empty label):
- 1a: square root, Gaussian σ = 50 ms, PCA without z-score on 66 channels (20 draws, seeds 0-19):
      variance of the first 12 PCs (R1). Also with z-score, and with all active channels (with and without z-score).
- 1b: counts, Gaussian σ = 30 ms, PCA without z-score on 96 channels: components for 80% (R2). Also with
      z-score and with all active channels.
- 1e (post hoc): components for 60% with the same pipelines and draws, and with all channels.
Null: for each draw, 20 circular-shift surrogates of the same channels (criteria of Step 2, median trial
duration of the session); the median of the surrogates and the difference data − null are reported.

Outputs: ~/thesis_h1_work/results/literature/step1ab_draws.csv, rows in the run registry (experiment "literature").
Usage: PYTHONPATH=~/thesis_h1_git/src python scripts/literature_step1ab.py
"""
import time

import pandas as pd

from thesis_h1 import results, literature

OUT = results.RESULTS / "literature"
OUT.mkdir(parents=True, exist_ok=True)
SEMI, N_SURR = range(20), 20
METRICS = [f"{m}{s}" for m in literature.MEASURES for s in ("", "_null", "_diff", "_null_min", "_null_max")] \
    + ["n_channels"]
T0 = time.time()

ANALYSIS = [  # (condition, pipeline, channels per draw, z-score)
    ("1a_gallego", literature.pipeline_gallego, 66, False),
    ("1a_gallego_z", literature.pipeline_gallego, 66, True),
    ("1a_gallego_all", literature.pipeline_gallego, None, False),
    ("1a_gallego_all_z", literature.pipeline_gallego, None, True),
    ("1b_silvernagel", literature.pipeline_silvernagel, 96, False),
    ("1b_silvernagel_z", literature.pipeline_silvernagel, 96, True),
    ("1b_silvernagel_all", literature.pipeline_silvernagel, None, False),
    ("1b_silvernagel_all_z", literature.pipeline_silvernagel, None, True),
]

rows = []
for name in literature.SESSIONS:
    s = literature.load(name)
    m, D = s.mask, s.duration_trial
    _, escl = literature.shifts_allowed(len(s.counts), D)
    print(f"[{(time.time() - T0) / 60:4.1f} min] {name}: {s.counts.shape[1]} channels, {m.sum()} trial bins "
          f"out of {len(m)}, median trial duration {D * literature.BIN_S:.2f} s, shifts excluded {escl:.1%}",
          flush=True)
    Y = {}
    for cond, pipe, n, z in ANALYSIS:
        if pipe not in Y:
            Y[pipe] = pipe(s.counts)
        recs = literature.draws_con_null(Y[pipe], m, n, SEMI if n else [0], D, zscore=z, n_surr=N_SURR)
        for r in recs:
            r.update(session=name, condition=cond, zscore=z, duration_trial_s=D * literature.BIN_S,
                     fraction_excluded=escl, n_bin=int(m.sum()))
        rows += recs
        summ = literature.log(recs, METRICS + ["duration_trial_s", "fraction_excluded", "n_bin"],
                               experiment="literature", method="PCA", latent_dim=None, decoder=None,
                               session=s.key, bins="falcon", split="session", condition=cond, fold=None,
                               k_shot=None, H_ms=None)
        print(f"   {cond:24s} var12 {summ['var12_median']:5.1f} (null {summ['var12_null_median']:5.1f}, "
              f"diff {summ['var12_diff_median']:+5.1f})  n80 {summ['n80_median']:5.1f} "
              f"(null {summ['n80_null_median']:5.1f})  n60 {summ['n60_median']:5.1f} "
              f"(null {summ['n60_null_median']:5.1f})", flush=True)

cols = ["session", "condition", "zscore", "seed", "n_channels", "n_bin", "duration_trial_s", "fraction_excluded"]
df = pd.DataFrame(rows)
df = df[cols + [c for c in df.columns if c not in cols]]
df.to_csv(OUT / "step1ab_draws.csv", index=False)
print(f"[{(time.time() - T0) / 60:4.1f} min] saved {OUT / 'step1ab_draws.csv'} ({len(df)} rows)")
