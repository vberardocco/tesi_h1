"""Step 1f (literature comparison, exploratory, added on 3 Nov after seeing the results): two groups of channels.

Working hypothesis, not a fact: the cut at 88 (original index < 88 versus ≥ 88) might separate the two
arrays, but the NWB files have no electrode table. The R1 verdict remains that of 1a, on all active
channels: no group is chosen to reach the literature values.

For S1, S2, S5 and for each group (below88, da88, all): pipeline 1a (square root, Gaussian σ = 50 ms,
trial bins), 66 channels, 20 draws (seeds 0-19), circular-shift null with 20 surrogates per draw.
- median firing rate of the channels of the group (Hz); share of the variance in the 5 most variable channels of the group, and
  median over the draws of the same share among the 66 drawn channels;
- R1 (variance of the first 12 PCs) on data and null, difference per draw: without z-score, with z-score, and without
  the 10 most variable channels of the group (draws from the remaining channels, without z-score).

Outputs: ~/thesis_h1_work/results/literature/step1f.csv and step1f.md; rows in the run registry.
Usage: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=~/thesis_h1_git/src python scripts/literature_step1f.py
"""
import time

import numpy as np
import pandas as pd

from thesis_h1 import results, literature

OUT = results.RESULTS / "literature"
OUT.mkdir(parents=True, exist_ok=True)
SEMI, N_SURR, N = range(20), 20, 66
T0 = time.time()
VARIANTI = {"noz": (False, False), "z": (True, False), "without10": (False, True)}

rows, synthesis = [], []
for name in literature.SESSIONS:
    s = literature.load(name)
    m, D = s.mask, s.duration_trial
    Y = literature.pipeline_gallego(s.counts)
    var = Y[m].var(0, ddof=1)
    hz = s.rate_hz
    groups = {"below88": np.flatnonzero(s.active < 88), "da88": np.flatnonzero(s.active >= 88),
              "all": np.arange(len(s.active))}
    for g, col in groups.items():
        base = dict(session=name, group=g, n_channels_group=len(col), freq_median_hz=float(np.median(hz[col])),
                    share5_group=100 * literature.share_more_variables(var[col], 5))
        for v, (z, without10) in VARIANTI.items():
            c = col[np.argsort(var[col])[:-10]] if without10 else col
            c = np.sort(c)
            recs = literature.draws_con_null(Y[:, c], m, N, SEMI, D, zscore=z, n_surr=N_SURR,
                                               measures={"var12": literature.MEASURES["var12"]})
            for r in recs:
                sel = c[literature.extract(len(c), N, np.random.default_rng(r["seed"]))]
                r.update(share5_draw=100 * literature.share_more_variables(var[sel], 5), variant=v, **base)
            rows += recs
            summ = literature.log(recs, ["var12", "var12_null", "var12_diff", "share5_draw",
                                          "freq_median_hz", "share5_group", "n_channels_group"],
                                   experiment="literature", method="PCA", latent_dim=None, decoder=None,
                                   session=s.key, bins="falcon", split="session", condition=f"1f_{g}_{v}",
                                   fold=None, k_shot=None, H_ms=None)
            synthesis.append(dict(**base, variant=v, **summ))
            print(f"[{(time.time() - T0) / 60:4.1f} min] {name} {g:8s} {v:8s} ({len(c)} channels): R1 "
                  f"{summ['var12_median']:5.1f}, null {summ['var12_null_median']:5.1f}, diff "
                  f"{summ['var12_diff_median']:+5.1f}; freq {base['freq_median_hz']:.1f} Hz, "
                  f"share5 {base['share5_group']:.1f}%", flush=True)

pd.DataFrame(rows).to_csv(OUT / "step1f.csv", index=False)
S = pd.DataFrame(synthesis)


def mmm(r, k, fmt="{:.1f}"):
    return f"{fmt.format(r[k + '_median'])} [{fmt.format(r[k + '_min'])}-{fmt.format(r[k + '_max'])}]"


md = ["# Step 1f · Two groups of channels (exploratory, post hoc)", "",
      f"Commit {results.code_version()[0]}. Analysis added on 3 Nov after seeing the results: it does not change the "
      "R1 verdict, which remains computed on all active channels (1a). The cut at 88 is a hypothesis: the NWB files "
      "have no electrode table, so the groups must not be called 'the two arrays'.", "",
      "Pipeline 1a, 66 channels, 20 draws, circular-shift null (20 surrogates per draw). Values: median "
      "[min-max] over the draws; the difference is data − null, draw by draw.", "",
      "| Session | Group | channels | median rate (Hz) | share of 5 most variable (%) | variant | R1 data (%) | "
      "R1 null (%) | data − null (points) |",
      "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
for _, r in S.iterrows():
    md.append(f"| {r.session} | {r.group} | {r.n_channels_group} | {r.freq_median_hz:.1f} | {r.share5_group:.1f} | "
              f"{r.variant} | {mmm(r, 'var12')} | {mmm(r, 'var12_null')} | {mmm(r, 'var12_diff', '{:+.1f}')} |")
md += ["", "Variants: noz = without z-score; z = with z-score; without10 = without the 10 most variable channels of the group "
       "(without z-score)."]
(OUT / "step1f.md").write_text("\n".join(md) + "\n")
print(f"saved {OUT / 'step1f.csv'} and step1f.md")
