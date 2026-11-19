"""Step 5a (literature comparison, optional, approved at STOP 3): CCA as in Gallego 2020 between S1, S2 and S5.

For each session: counts in 30 ms bins rebinned from the spike times (active channels ≥ 0.5 Hz), square
root, Gaussian σ = 50 ms; PCA to 10 dimensions without z-score, fitted on the windows of the session.
Windows: 33 bins (0.99 s) from the start of each movement phase; 16 conditions = phase × odd/even trials (§ 8
of notebook 02). For each draw (seeds 0-19), in each session and condition the repetitions are shuffled and two
disjoint halves of 3 are taken (A and B):
- same day: CCA between half A and half B of the same session (reference for the normalization);
- between sessions: CCA between half A of one session and half A of the other, same conditions in the same order;
- control: between sessions with the conditions of the second one swapped (permutation without fixed points);
- post hoc variant (added after seeing the single-trial values): the same CCAs on the means of the 3
  repetitions of each condition (16 mean windows per half), to separate single-trial noise.
Measures: mean of the 10 canonical correlations (aligned) and mean of the |correlations| between PC i of one session
and PC i of the other (non-aligned); normalized = between sessions / mean of the two same-day values.
They are compared with 0.93 ± 0.03 (aligned, normalized) and 0.38 ± 0.14 (non-aligned) of Gallego 2020: the
comparison is qualitative (other data, windows and conditions).

Outputs: ~/thesis_h1_work/results/literature/step5a_cca.csv and step5a_cca.md; rows in the run registry.
Usage: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=~/thesis_h1_git/src python scripts/literature_step5a.py
"""
import itertools
import time

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

from thesis_h1 import io, results, literature

OUT = results.RESULTS / "literature"
OUT.mkdir(parents=True, exist_ok=True)
BIN, N_BIN, D, N_REP = 0.03, 33, 10, 3
SEMI = range(20)
DAYS = {"S1": 8, "S2": 13, "S5": 20}                       # January 1925 (dates anonymized by DANDI)
CONDS = [(f, v) for f in literature.PHASES_MOVEMENT for v in ("odd", "even")]
T0 = time.time()
KEYS = dict(experiment="literature", method="PCA", latent_dim=D, decoder=None, bins="falcon", split="windows_1s",
            fold=None, k_shot=None, H_ms=None)


def log(msg):
    print(f"[{(time.time() - T0) / 60:4.1f} min] {msg}", flush=True)


data = {}
for name, sid in literature.SESSIONS.items():
    s = literature.load(name)
    ses = io.load_session(literature.DATA / "sub-HumanPitt-held-in-calib" / f"sub-HumanPitt-held-in-calib_ses-{sid}.nwb",
                          bins="falcon")
    c30, edges, complete = literature.ribinna(s.spike_times, s.t0, s.t1, BIN)
    n_tot = int(complete.sum())
    Y = literature.smooth(np.sqrt(c30[:n_tot]), 50, bin_s=BIN)
    W = literature.windows_phase(ses.epochs, lambda t: ses.trial_num[min(io.time_to_bin(ses, t), ses.n_bins - 1)],
                             s.t0, BIN, N_BIN, n_tot)
    rip = {c: [i for i, w in enumerate([w for w in W if w["cond"] == c])] for c in CONDS}
    pca = PCA(D).fit(np.concatenate([Y[w["start"]:w["stop"]] for w in W]))
    L = pca.transform(Y)
    data[name] = dict(L=L, W=W, n_rip={c: len(v) for c, v in rip.items()}, var10=100 * pca.explained_variance_ratio_.sum())
    log(f"{name}: {Y.shape[1]} channels, {len(W)} windows, repetitions per condition "
        f"{min(data[name]['n_rip'].values())}-{max(data[name]['n_rip'].values())}, variance of the 10 PCs "
        f"{data[name]['var10']:.1f}%")
assert min(min(d["n_rip"].values()) for d in data.values()) >= 2 * N_REP

rows = []
for seed in SEMI:
    rng = np.random.default_rng(seed)
    meta = {}
    for name, d in data.items():
        A, B = {}, {}
        for c in CONDS:
            p = rng.permutation(d["n_rip"][c])
            A[c], B[c] = list(p[:N_REP]), list(p[N_REP:2 * N_REP])
        meta[name] = (literature.block_conditions(d["L"], d["W"], A), literature.block_conditions(d["L"], d["W"], B), A,
                      literature.means_conditions(d["L"], d["W"], A), literature.means_conditions(d["L"], d["W"], B))
        cc = literature.correlations_canonical(meta[name][0], meta[name][1])
        nc = literature.correlations_non_aligned(meta[name][0], meta[name][1])
        rows.append(dict(seed=seed, kind="stesso giorno", pair=name, days=0, aligned=cc.mean(),
                          non_aligned=nc.mean(),
                          means_aligned=literature.correlations_canonical(meta[name][3], meta[name][4]).mean(),
                          means_non_aligned=literature.correlations_non_aligned(meta[name][3], meta[name][4]).mean(),
                          **{f"cc{i + 1}": v for i, v in enumerate(cc)}))
    for x, y in itertools.combinations(data, 2):
        Ax, Ay = meta[x][0], meta[y][0]
        cc = literature.correlations_canonical(Ax, Ay)
        nc = literature.correlations_non_aligned(Ax, Ay)
        while True:                                                   # conditions swapped, none in its place
            perm = rng.permutation(len(CONDS))
            if (perm != np.arange(len(CONDS))).all():
                break
        swapped = {CONDS[j]: meta[y][2][CONDS[j]] for j in perm}
        cs = literature.correlations_canonical(Ax, literature.block_conditions(data[y]["L"], data[y]["W"], swapped))
        ref = [r for r in rows if r["seed"] == seed and r["kind"] == "stesso giorno" and r["pair"] in (x, y)]
        ref_a = np.mean([r["aligned"] for r in ref])
        ref_n = np.mean([r["non_aligned"] for r in ref])
        ma = literature.correlations_canonical(meta[x][3], meta[y][3]).mean()
        mn = literature.correlations_non_aligned(meta[x][3], meta[y][3]).mean()
        ms = literature.correlations_canonical(meta[x][3], literature.means_conditions(data[y]["L"], data[y]["W"], swapped)).mean()
        rows.append(dict(seed=seed, kind="tra sessioni", pair=f"{x}-{y}", days=DAYS[y] - DAYS[x],
                          aligned=cc.mean(), non_aligned=nc.mean(), aligned_norm=cc.mean() / ref_a,
                          non_aligned_norm=nc.mean() / ref_n, conditions_swapped=cs.mean(),
                          means_aligned=ma, means_non_aligned=mn,
                          means_aligned_norm=ma / np.mean([r["means_aligned"] for r in ref]),
                          means_not_aligned_norm=mn / np.mean([r["means_non_aligned"] for r in ref]),
                          means_conditions_swapped=ms,
                          **{f"cc{i + 1}": v for i, v in enumerate(cc)}))
df = pd.DataFrame(rows)
df.to_csv(OUT / "step5a_cca.csv", index=False)
METR = ["aligned", "non_aligned", "aligned_norm", "non_aligned_norm", "conditions_swapped", "days",
        "means_aligned", "means_non_aligned", "means_aligned_norm", "means_not_aligned_norm",
        "means_conditions_swapped"] + \
       [f"cc{i + 1}" for i in range(D)]
for pair, g in df.groupby("pair", sort=False):
    rec = g.drop(columns=["kind", "pair"]).to_dict("records")
    same = g.kind.iloc[0] == "stesso giorno"
    literature.log(rec, [m for m in METR if m in g and g[m].notna().all()],
                    session=literature.KEYS[pair] if same else None,
                    condition="5a_cca_same_day" if same else f"5a_cca_{pair.replace('-', '_')}", **KEYS)


def mmm(v, f="{:.2f}"):
    med, lo, hi = literature.summary(v)
    return f"{f.format(med)} [{f.format(lo)}-{f.format(hi)}]"


md = ["# Step 5a · CCA as in Gallego 2020 (optional)", "",
      f"Commit {results.code_version()[0]}. Square root, 30 ms bins, Gaussian σ 50 ms, PCA to 10 dimensions per "
      f"session; 0.99 s windows from the start of the movement phases, 16 conditions, {N_REP} repetitions per "
      "condition and per half; 20 draws (median [min-max]). Normalized = between sessions / mean of the two "
      "same-day values.", "",
      "| Comparison | days | aligned (CCA) | non-aligned | aligned, normalized | non-aligned, "
      "normalized | conditions swapped (control) |", "| --- | --- | --- | --- | --- | --- | --- |"]
for pair, g in df.groupby("pair", sort=False):
    if g.kind.iloc[0] == "stesso giorno":
        md.append(f"| {pair}, same day | 0 | {mmm(g.aligned)} | {mmm(g.non_aligned)} | — | — | — |")
    else:
        md.append(f"| {pair} | {int(g.days.iloc[0])} | {mmm(g.aligned)} | {mmm(g.non_aligned)} | "
                  f"{mmm(g.aligned_norm)} | {mmm(g.non_aligned_norm)} | {mmm(g.conditions_swapped)} |")
md += ["", "Post hoc variant: the same CCAs on the means of the 3 repetitions of each condition (16 mean "
       "windows per half, 528 rows).", "",
       "| Comparison | days | aligned (CCA) | non-aligned | aligned, normalized | non-aligned, "
       "normalized | conditions swapped (control) |", "| --- | --- | --- | --- | --- | --- | --- |"]
for pair, g in df.groupby("pair", sort=False):
    if g.kind.iloc[0] == "stesso giorno":
        md.append(f"| {pair}, same day | 0 | {mmm(g.means_aligned)} | {mmm(g.means_non_aligned)} | — | — | — |")
    else:
        md.append(f"| {pair} | {int(g.days.iloc[0])} | {mmm(g.means_aligned)} | {mmm(g.means_non_aligned)} | "
                  f"{mmm(g.means_aligned_norm)} | {mmm(g.means_not_aligned_norm)} | "
                  f"{mmm(g.means_conditions_swapped)} |")
md += ["", "Canonical correlations per component, single trials (median over draws):", "",
       "| Comparison | " + " | ".join(f"CC{i + 1}" for i in range(D)) + " |", "| --- | " + " | ".join("---" for _ in range(D)) + " |"]
for pair, g in df.groupby("pair", sort=False):
    md.append(f"| {pair} | " + " | ".join(f"{g[f'cc{i + 1}'].median():.2f}" for i in range(D)) + " |")
md += ["", "Gallego 2020 (monkey CL, up to ~750 days, M1 at 10 dimensions): \"the normalized similarity of the "
       "latent dynamics after alignment with CCA was 0.93 ± 0.03\", against 0.38 ± 0.14 without alignment. The "
       "comparison with H1 is qualitative: other data, conditions (targets versus phases) and windows.",
       "Variance explained by the 10 PCs on the windows: " + ", ".join(f"{n} {d['var10']:.1f}%" for n, d in data.items()) + "."]
(OUT / "step5a_cca.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
