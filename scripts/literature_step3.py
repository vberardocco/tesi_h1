"""Step 3 (literature comparison): clock + neurons, rule R7.

Same trials, blocks and classifier as the phase decoding in the replication with FALCON bins
(notebook 04, § 1.2 and 12.2): points = bins of the 8 movement phases with index a multiple of 5; 5 blocks of 3
trials (GroupKFold); 10 neighbors; balanced accuracy; chance = 1 / 8 classes = 12.5%.
Inputs: (a) clock only (s from the trial start); (b) the 25 GPFA latents of the block (F_GP12[k], as in the
replication); (c) clock + 25 latents: standardized clock × w, standardized latents / √25 (statistics of the
training points), with w chosen by inner cross-validation on the training trials only (grid
0, 0.25, 0.5, 1, 2, 4, 8, ∞; ties go to the highest weight). Variants of (c) with fixed weight (w = 1: clock and
latents with the same total variance; w = 0.2: z-score of all 26 columns), without verdict.
Null for (c) − (a): in the test trials of each block the latents are permuted across whole trials (uniform
permutation of the 3 trials, identity included), at the same time from the trial start, clock fixed, 200 times
(seed 0); p = (1 + #{null ≥ observed}) / 201.
R7: (c) − (a) > 0 in 3 sessions out of 3, with p < 0.05 in each. Optional: accuracy within deciles
of the clock.
Check: (a) and (b) must coincide with ACC and ACC12 of the replication; otherwise the script stops without logging.

Outputs: ~/thesis_h1_work/results/literature/step3_clock.csv and step3_clock.md; rows in the run registry.
Usage: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=~/thesis_h1_git/src python scripts/literature_step3.py
"""
import pickle
import sys
import time

import numpy as np
import pandas as pd

from thesis_h1 import results, literature

OUT = results.RESULTS / "literature"
OUT.mkdir(parents=True, exist_ok=True)
T0 = time.time()
KEYS = dict(experiment="literature", method="GPFA", latent_dim=25, decoder="knn10", bins="falcon", split="cv5_trial",
            condition="3_clock", fold=None, k_shot=None, H_ms=None, seed=0)

rows = []
for name, sid in literature.SESSIONS.items():
    d = pickle.load(open(literature.REP / sid / "04_rep.pkl", "rb"))
    a, D = d["arrays"], d["dicts"]
    clock, phase, trial = a["clock"], a["phase"], a["trial"]
    F = {k: D["F_GP12"][k] for k in range(5)}
    ref_a = float(D["ACC"]["l'orologio da solo (1 numero)"])
    ref_b = float(D["ACC12"][("GPFA alla pari", "tetto: tutti i 25 fattori")][0])
    del d, a, D
    out = literature.clock_neurons(clock, F, phase, trial, n_perm=200, seed=0)
    print(f"[{(time.time() - T0) / 60:4.1f} min] {name}: (a) {out['acc_a']:.2f} (replication {ref_a:.2f}), (b) "
          f"{out['acc_b']:.2f} (replication {ref_b:.2f}), (c) {out['acc_c']:.2f}, (c) - (a) {out['diff_c_a']:+.2f}, "
          f"p {out['p']:.3f}; weights {[out[f'weight_block{k}'] for k in range(5)]}", flush=True)
    if abs(out["acc_a"] - ref_a) > 0.01 or abs(out["acc_b"] - ref_b) > 0.01:
        sys.exit(f"{name}: (a) or (b) differ from the replication: stopping without logging")
    out.update(ref_replica_a=ref_a, ref_replica_b=ref_b)
    results.log_run({k: v for k, v in out.items()}, session=literature.KEYS[name], **KEYS)
    rows.append(dict(session=name, **out))

df = pd.DataFrame(rows)
df.to_csv(OUT / "step3_clock.csv", index=False)
r7 = bool(((df.diff_c_a > 0) & (df.p < 0.05)).all())
results.log_run({"R7_si": float(r7), "n_sessions_ok": float(((df.diff_c_a > 0) & (df.p < 0.05)).sum())},
                **{**KEYS, "condition": "3_clock_R7", "session": None, "seed": None})

md = ["# Step 3 · Clock + neurons (R7)", "",
      f"Commit {results.code_version()[0]}. Same points, blocks and classifier as the replication (8 movement "
      "phases, one point every 100 ms, 5 blocks of 3 trials, 10 neighbors, balanced accuracy, chance "
      f"{df.chance.iloc[0]:.1f}%). (c): clock weight chosen by cross-validation on the training trials only. "
      "Null: latents permuted across whole test trials (at the same time from the trial start), clock "
      "fixed, 200 times.", "",
      "| Session | points | (a) clock | (b) GPFA 25 | (c) clock + GPFA | (c) − (a) | null of (c): median [95th] | p | "
      "clock weight per block |",
      "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
for r in rows:
    weights = ", ".join("∞" if np.isinf(r[f"weight_block{k}"]) else f"{r[f'weight_block{k}']:g}" for k in range(5))
    md.append(f"| {r['session']} | {r['n_points']} | {r['acc_a']:.1f} | {r['acc_b']:.1f} | {r['acc_c']:.1f} | "
              f"{r['diff_c_a']:+.1f} | {r['null_c_median']:.1f} [{r['null_c_p95']:.1f}] | {r['p']:.3f} | {weights} |")
md += ["", f"**R7**: {'sì' if r7 else 'no'} ((c) − (a) > 0 with p < 0.05 in "
       f"{int(((df.diff_c_a > 0) & (df.p < 0.05)).sum())} sessions out of 3).", "",
       "Variants of (c) with fixed weight (without verdict):", "",
       "| Session | (c), w = 1 | − (a) | p | (c), z-score of all columns (w = 0.2) | − (a) | p |",
       "| --- | --- | --- | --- | --- | --- | --- |"]
for r in rows:
    md.append(f"| {r['session']} | {r['acc_c_w1']:.1f} | {r['diff_c_a_w1']:+.1f} | {r['p_w1']:.3f} | "
              f"{r['acc_c_w0.2']:.1f} | {r['diff_c_a_w0.2']:+.1f} | {r['p_w0.2']:.3f} |")
md += ["", "Optional: balanced accuracy within the deciles of the time from the trial start "
       "((a) / (b) / (c)):", "", "| Decile | " + " | ".join(r["session"] for r in rows) + " |",
       "| --- | " + " | ".join("---" for _ in rows) + " |"]
for i in range(1, 11):
    md.append(f"| {i} | " + " | ".join(f"{r[f'decile{i}_a']:.0f} / {r[f'decile{i}_b']:.0f} / {r[f'decile{i}_c']:.0f}"
                                     for r in rows) + " |")
(OUT / "step3_clock.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
