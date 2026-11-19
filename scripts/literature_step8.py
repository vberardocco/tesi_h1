"""Phase 2, R8 (literature comparison): sensitivity of the Parallel Analysis to preprocessing; shared fraction of slide 5 with FALCON
bins.

R8 (rules in docs/literature/rules_decision.md, commit 1aa7c5d; clarification P8):
- pipeline of notebook 02: Gaussian σ ∈ {20, 50, 100} ms, in Hz, with and without z-score, all bins of the session;
- Gallego pipeline: square root, Gaussian σ ∈ {20, 50, 100} ms, no z-score, trial bins;
- null: independent circular shift per channel as in Step 2 (2 s - T − 2 s, excluding ±2 s around the multiples
  of the median trial duration), 200 surrogates, seed 0 for each combination; k_PA = first consecutive eigenvalues
  above the 95th percentile of the same rank.
- Check before everything else: 02 with σ = 50 ms and z-score must give 19 / 17 / 15, Gallego with σ = 50 ms
  11 / 9 / 11. If it does not match, the script stops without logging anything.
- Verdict: "robust existence" if k_PA ≥ 1 in all combinations and in all 3 sessions; the number is reported
  as a min-max range over the grid, per session.

Shared fraction (slide 5, no verdict): as in § 9.5 of notebook 02, FA with K by cross-validation (5 contiguous blocks,
svd_method="lapack") on the z-scored raw counts (grid 2, 5, 10, 15, 20, 30) and on the smoothed data with
σ = 20, 50, 100 ms and z-scored (grid 5, 10, 15, 20, 30); median of the per-channel shared fraction. Check: with the
legacy bins in S1 notebook 02 gave 3.8% (counts) and 16.7% (σ = 50 ms).

Outputs: ~/thesis_h1_work/results/literature/step8_sigma.csv, step8_share.csv, step8_sigma.md, step8_sigma.png; rows
in the run registry (condition 8_sigma_*).
Usage: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=~/thesis_h1_git/src python scripts/literature_step8.py
"""
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                                # noqa: E402
import numpy as np                                                             # noqa: E402
import pandas as pd                                                            # noqa: E402

from thesis_h1 import io, palette, results, literature, literature2                      # noqa: E402

OUT = results.RESULTS / "literature"
OUT.mkdir(parents=True, exist_ok=True)
N_SURR = 200
ATTESI = {"02_s50_z": {"S1": 19, "S2": 17, "S5": 15}, "gallego_s50": {"S1": 11, "S2": 9, "S5": 11}}
GRID_GREZZI = [2, 5, 10, 15, 20, 30]
GRID_SMUSSATI = [5, 10, 15, 20, 30]
SHARE_LEGACY_02 = {0: 3.8, 50: 16.7}           # notebook 02_new, § 9.5 (S1, legacy bins)
KEYS = dict(experiment="literature", latent_dim=None, decoder=None, split="session", fold=None, k_shot=None,
            H_ms=None)
T0 = time.time()


def log(msg):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {msg}", flush=True)


def comma(x, nd=1):
    return f"{x:.{nd}f}".replace(".", ",")


sess = {name: literature.load(name) for name in literature.SESSIONS}
grid = literature2.grid_r8()


def compute(s, comb):
    name, pipe, sigma, z, only_trial = comb
    f = literature2.pipeline_02_sigma if pipe == "02" else literature2.pipeline_gallego_sigma
    Y = f(s.counts, sigma)
    m = s.mask if only_trial else np.ones(len(Y), bool)
    allowed, escl = literature.shifts_allowed(len(s.counts), s.duration_trial)
    k = literature2.k_pa(Y, m, z, allowed, N_SURR, seed=0)
    N = Y.shape[1]
    log(f"{s.name} {name:12s}: k_PA {k:3d} ({k / N:.1%})")
    return dict(session=s.name, combination=name, pipeline=pipe, sigma_ms=sigma, zscore=z, N=N, n_bin=int(m.sum()),
                k_pa=k, k_pa_su_N=k / N, duration_trial_s=s.duration_trial * literature.BIN_S, fraction_excluded=escl,
                n_surr=N_SURR, seed=0)


# 1 · reproduction check (Step 2)
rows = []
for comb in grid:
    if comb[0] in ATTESI:
        rows += [compute(s, comb) for s in sess.values()]
wrong = [f"{r['session']} {r['combination']}: {r['k_pa']} instead of {ATTESI[r['combination']][r['session']]}"
             for r in rows if r["k_pa"] != ATTESI[r["combination"]][r["session"]]]
if wrong:
    msg = "R8 check failed, stopping without logging: " + "; ".join(wrong)
    (OUT / "step8_check_failed.md").write_text(msg + "\n")
    sys.exit(msg)
log("check passed: 19 / 17 / 15 and 11 / 9 / 11 reproduced")

# 2 · the rest of the grid
for comb in grid:
    if comb[0] not in ATTESI:
        rows += [compute(s, comb) for s in sess.values()]
order = {c[0]: i for i, c in enumerate(grid)}
df = pd.DataFrame(rows)
df["order"] = df.combination.map(order)
df = df.sort_values(["session", "order"]).drop(columns="order").reset_index(drop=True)
df.to_csv(OUT / "step8_sigma.csv", index=False)
for r in df.to_dict("records"):
    results.log_run({k: float(r[k]) for k in ("k_pa", "k_pa_su_N", "N", "n_bin", "sigma_ms", "zscore",
                                              "fraction_excluded", "n_surr")},
                    method="PCA", session=literature.KEYS[r["session"]], bins="falcon",
                    condition=f"8_sigma_{r['combination']}", seed=0, **KEYS)
robust = bool((df.k_pa >= 1).all())
interv = df.groupby("session").agg(k_min=("k_pa", "min"), k_max=("k_pa", "max"), kN_min=("k_pa_su_N", "min"),
                                    kN_max=("k_pa_su_N", "max"))
for name, r in interv.iterrows():
    results.log_run({k: float(v) for k, v in r.items()}, method="PCA", session=literature.KEYS[name], bins="falcon",
                    condition="8_sigma_interval", seed=0, **KEYS)
results.log_run({"R8_existence_robust": float(robust), "k_min": float(df.k_pa.min()),
                 "combinations_con_k0": float((df.k_pa < 1).sum())},
                method="PCA", session=None, bins="falcon", condition="8_sigma_R8", seed=None, **KEYS)
log(f"R8: {'robust existence' if robust else 'NOT robust'}; ranges {interv.to_dict('index')}")


# 3 · figure
COL = {("02", True): palette.CAT[0], ("02", False): palette.CAT[1], ("gallego", False): palette.CAT[2]}
fig, axs = plt.subplots(3, 1, figsize=(10.5, 8.2), sharex=True, sharey=True)
etich = [f"{'02' if p == '02' else 'Gallego'}\nσ {sg} ms" + ("" if p == "gallego" else ("\nz-score" if z else "\nno z"))
         for _, p, sg, z, _ in grid]
for ax, (name, g) in zip(axs, df.groupby("session", sort=False)):
    palette.style_axes(ax)
    g = g.set_index("combination").loc[[c[0] for c in grid]]
    x = np.arange(len(g))
    cols = [COL[(p, z)] for p, z in zip(g.pipeline, g.zscore)]
    ax.bar(x, g.k_pa, color=cols, width=0.7,
           edgecolor=[palette.INK if c in ATTESI else "none" for c in g.index], linewidth=1.4)
    for xi, (k, kn) in enumerate(zip(g.k_pa, g.k_pa_su_N)):
        ax.annotate(f"{k}\n{comma(100 * kn)}%", (xi, k), xytext=(0, 3), textcoords="offset points", ha="center",
                    fontsize=7.5, color=palette.INK2)
    ax.axhline(1, color=palette.NEUTRAL, lw=1, ls="--")
    ax.set_ylabel(f"{name}: k_PA", fontsize=9, color=palette.INK2)
    ax.set_title(f"{name} (N = {int(g.N.iloc[0])}): k_PA from {int(g.k_pa.min())} to {int(g.k_pa.max())}",
                 fontsize=9.5, color=palette.INK, loc="left")
axs[-1].set_xticks(np.arange(len(grid)))
axs[-1].set_xticklabels(etich, fontsize=7.5)
axs[0].set_ylim(0, df.k_pa.max() * 1.3)
fig.suptitle("R8 · Parallel Analysis (circular shift, 200 surrogates) as preprocessing varies — "
             f"{'robust existence' if robust else 'existence NOT robust'}\n"
             "black edge = Step 2 combinations (check); dashed = k_PA = 1; above the bars k_PA and k_PA/N",
             fontsize=10, color=palette.INK)
fig.tight_layout()
fig.savefig(OUT / "step8_sigma.png", dpi=200, bbox_inches="tight")
log("figure saved")


# 4 · shared fraction of slide 5 (FA, K by cross-validation)
def zscore(X):
    return (X - X.mean(0)) / X.std(0)


def share(counts, session, bins):
    out = []
    X = np.asarray(counts, float)
    for sigma in (0, 20, 50, 100):
        if bins == "legacy" and sigma not in SHARE_LEGACY_02:
            continue
        Z = zscore(X if sigma == 0 else literature.smooth(X, sigma))
        gr = GRID_GREZZI if sigma == 0 else GRID_SMUSSATI
        cv = literature.fa_cv(Z, gr, n_jobs=6)
        K = gr[int(np.argmax(cv))]
        f = literature.fa_fit(Z, K)
        r = dict(session=session, bins=bins, sigma_ms=sigma, K_star=K, K_al_edge=K == gr[-1],
                 share_channel_median=100 * float(np.median(literature.share_channels(f))),
                 share_population=100 * literature.share_population(f), dim95=literature.dim_shared(f),
                 PR_shared=literature.participation_ratio(literature.eigenvalues_shared(f)))
        log(f"shared fraction {session} {bins} σ {sigma:3d}: K* {K}, per-channel median shared fraction {r['share_channel_median']:.1f}%")
        out.append(r)
    return out


rq = []
for name, s in sess.items():
    rq += share(s.counts, name, "falcon")
sl = io.load_session(literature.DATA / "sub-HumanPitt-held-in-calib" /
                     f"sub-HumanPitt-held-in-calib_ses-{literature.SESSIONS['S1']}.nwb", bins="legacy")
rq += share(sl.spikes[:, io.active_units(sl, min_hz=0.5)], "S1", "legacy")
dq = pd.DataFrame(rq)
dq.to_csv(OUT / "step8_share.csv", index=False)
for r in dq.to_dict("records"):
    results.log_run({k: float(r[k]) for k in ("K_star", "K_al_edge", "share_channel_median", "share_population",
                                              "dim95", "PR_shared")},
                    method="FA", session=literature.KEYS[r["session"]], bins=r["bins"],
                    condition=f"8_sigma_share_s{r['sigma_ms']}", seed=0, **KEYS)
leg = dq[dq.bins == "legacy"].set_index("sigma_ms").share_channel_median
ctrl_q = {s: round(float(leg[s]), 1) == v for s, v in SHARE_LEGACY_02.items()}
log(f"legacy shared-fraction check S1: {dict(leg.round(1))} against {SHARE_LEGACY_02}: {ctrl_q}")

# 5 · markdown
md = ["# Phase 2 · R8 · Parallel Analysis as preprocessing varies", "",
      f"Commit {results.code_version()[0]}. Rules: docs/literature/rules_decision.md (commit 1aa7c5d). Circular shift, "
      f"{N_SURR} surrogates, seed 0 for each combination; k_PA = first consecutive eigenvalues above the 95th percentile.",
      "", "Check: 02 with σ = 50 ms and z-score = 19 / 17 / 15, Gallego with σ = 50 ms = 11 / 9 / 11: reproduced.", "",
      "| Session | " + " | ".join(e.replace("\n", " ") for e in etich) + " |",
      "| --- | " + " | ".join("---" for _ in grid) + " |"]
for name, g in df.groupby("session", sort=False):
    g = g.set_index("combination").loc[[c[0] for c in grid]]
    md.append(f"| {name} (N = {int(g.N.iloc[0])}) | " +
              " | ".join(f"{k} ({comma(100 * kn)}%)" for k, kn in zip(g.k_pa, g.k_pa_su_N)) + " |")
md += ["", "Min-max range over the grid, per session (k_PA; k_PA/N):", ""]
for name, r in interv.iterrows():
    md.append(f"- {name}: {int(r.k_min)}-{int(r.k_max)}; {comma(100 * r.kN_min)}-{comma(100 * r.kN_max)}%")
md += ["", f"**R8**: {'robust existence' if robust else 'existence NOT robust'} (k_PA ≥ 1 in "
       f"{int((df.k_pa >= 1).sum())} combinations out of {len(df)}).", "",
       "## Per-channel shared fraction (FA, K by cross-validation), for slide 5", "",
       "As in § 9.5 of notebook 02. No verdict.", "",
       "| Session | bins | σ (ms) | K* | per-channel shared fraction, median | population shared fraction | dimensions for 95% |",
       "| --- | --- | --- | --- | --- | --- | --- |"]
for r in dq.to_dict("records"):
    md.append(f"| {r['session']} | {r['bins']} | {'counts' if r['sigma_ms'] == 0 else r['sigma_ms']} | "
              f"{r['K_star']}{' (edge)' if r['K_al_edge'] else ''} | {comma(r['share_channel_median'])}% | "
              f"{comma(r['share_population'])}% | {r['dim95']} |")
md += ["", "Check with the legacy bins (S1) against notebook 02 (3.8% and 16.7%): " +
       ", ".join(f"σ {s}: {comma(float(leg[s]))}% ({'reproduced' if ok else 'NOT reproduced'})"
                 for s, ok in ctrl_q.items()) + "."]
(OUT / "step8_sigma.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
log("end")
