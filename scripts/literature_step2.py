"""Step 2 (literature comparison): Parallel Analysis on S1, S2 and S5, rule R5.

Pipeline, all active channels:
- 02:      counts at 20 ms, Gaussian σ = 50 ms, z-score; all bins of the session (as in notebook 02);
- gallego: square root, Gaussian σ = 50 ms, no z-score; trial bins (without Intertrial and empty bin).
Nulls, 200 surrogates each, over the whole session:
- shift:   independent circular shift per channel, uniform displacement between 2 s and T − 2 s, excluding ±2 s around
           the multiples of the median trial duration of the session (main null, seed 0);
- shuffle: independent permutation of the bins of each channel (Altan 2021, seed 1).
k_PA = data eigenvalues above the 95th percentile of the eigenvalue of the same rank in the surrogates:
k_seq counts the first consecutive ones (it stops at the first one below the threshold, main definition), k_all all
ranks above the threshold. R5: "yes" if k_seq >= 1 with the circular shift in all sessions and in both pipelines.

Outputs: ~/thesis_h1_work/results/literature/step2_pa.csv, step2_pa.md, step2_spectra.csv; rows in the run registry.
Usage: PYTHONPATH=~/thesis_h1_git/src python scripts/literature_step2.py
"""
import time

import numpy as np
import pandas as pd

from thesis_h1 import results, literature

OUT = results.RESULTS / "literature"
OUT.mkdir(parents=True, exist_ok=True)
N_SURR = 200
T0 = time.time()
PIPE = {"02": (literature.pipeline_02, True, False), "gallego": (literature.pipeline_gallego, False, True)}

rows, spectra = [], []
for name in literature.SESSIONS:
    s = literature.load(name)
    D = s.duration_trial
    allowed, escl = literature.shifts_allowed(len(s.counts), D)
    for pipe, (f, z, only_trial) in PIPE.items():
        Y = f(s.counts)
        m = s.mask if only_trial else np.ones(len(Y), bool)
        for null, seed in (("shift", 0), ("shuffle", 1)):
            rng = np.random.default_rng(seed)
            ev, sur = literature.spectra(Y, m, z, rng, allowed, N_SURR, null)
            k_seq, k_all, thresholds = literature.parallel_analysis(ev, sur)
            N = Y.shape[1]
            r = dict(session=name, pipeline=pipe, null=null, N=N, n_bin=int(m.sum()), k_seq=k_seq,
                     k_seq_su_N=k_seq / N, k_all=k_all, k_all_su_N=k_all / N, duration_trial_s=D * literature.BIN_S,
                     fraction_excluded=escl if null == "shift" else np.nan, n_surr=N_SURR, seed=seed)
            rows.append(r)
            spectra += [dict(session=name, pipeline=pipe, null=null, rank=i + 1, data=ev[i], literature95=thresholds[i],
                             null_median=float(np.median(sur[:, i]))) for i in range(N)]
            results.log_run({k: v for k, v in r.items() if k not in ("session", "pipeline", "null", "seed")
                             and np.isfinite(v)},
                            experiment="literature", method="PCA", latent_dim=None, decoder=None, session=s.key,
                            bins="falcon", split="session", condition=f"2_pa_{pipe}_{null}", fold=None,
                            k_shot=None, H_ms=None, seed=seed)
            print(f"[{(time.time() - T0) / 60:4.1f} min] {name} {pipe:8s} {null:8s} N {N}: k_seq {k_seq:3d} "
                  f"({k_seq / N:.1%}), k_all {k_all:3d}", flush=True)

df = pd.DataFrame(rows)
df.to_csv(OUT / "step2_pa.csv", index=False)
pd.DataFrame(spectra).to_csv(OUT / "step2_spectra.csv", index=False)

shift = df[df.null == "shift"]
r5 = bool((shift.k_seq >= 1).all())
results.log_run({"R5_si": float(r5), "k_seq_min": float(shift.k_seq.min())},
                experiment="literature", method="PCA", latent_dim=None, decoder=None, session=None, bins="falcon",
                split="session", condition="2_pa_R5", fold=None, k_shot=None, H_ms=None, seed=None)

rows_md = ["# Step 2 · Parallel Analysis", "",
            f"Commit {results.code_version()[0]}. 200 surrogates per null; threshold = 95th percentile of the same rank.",
            "k_seq = first consecutive eigenvalues above the threshold (main definition); k_all = all ranks "
            "above the threshold.", "",
            "| Session | Pipeline | Null | N | bins | k_seq | k_seq/N | k_all | k_all/N | trial duration (s) | "
            "shifts excluded |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
for r in rows:
    escl = f"{r['fraction_excluded']:.1%}" if r["null"] == "shift" else "—"
    rows_md.append(f"| {r['session']} | {r['pipeline']} | {r['null']} | {r['N']} | {r['n_bin']} | {r['k_seq']} | "
                    f"{r['k_seq_su_N']:.1%} | {r['k_all']} | {r['k_all_su_N']:.1%} | {r['duration_trial_s']:.2f} | "
                    f"{escl} |")
diff = df.pivot_table(index=["session", "pipeline"], columns="null", values="k_seq")
rows_md += ["", "Difference shuffle − circular shift (k_seq): " +
             "; ".join(f"{s} {p}: {int(v['shuffle'] - v['shift']):+d}" for (s, p), v in diff.iterrows()), "",
             f"**R5**: {'sì' if r5 else 'no'} (k_seq ≥ 1 with the circular shift in all three sessions and in "
             f"both pipelines; minimum {int(shift.k_seq.min())}). Minimal threshold: k_PA and k_PA/N are reported."]
(OUT / "step2_pa.md").write_text("\n".join(rows_md) + "\n")
print("\n".join(rows_md[-3:]))
print(f"[{(time.time() - T0) / 60:4.1f} min] saved {OUT / 'step2_pa.csv'}, step2_pa.md, step2_spectra.csv")
