"""Phase 2, R10 (literature comparison): TME null for nonlinear structure; Isomap advantage on the single trial with FALCON bins.

Rules in docs/literature/rules_decision.md (commit 1aa7c5d), clarifications P3, P4, P5.

A · Single trial (updates +3.6 and −9.6 of slide 8): pipeline of notebook 03 (io.smooth_rates σ = 50 ms, z-score,
    PCA to 25 fitted on the bins of the movement phases, one bin in 5, Isomap with k = 10), data against 5 random-phase
    twins (seeds 0-4), d = 1..15. Check: with the legacy bins in S1, +3.6 (data) and −9.6 (twin mean) must be
    recovered at d = 3, as in notebook 03.
B · TME null (R10): tensor of per-condition means of Step 4b (8 phases × odd/even trials, 1 s windows
    from the start of the phase, Gaussian 50 ms, z-score), T = 50 × N × C = 16, compared with the PSTH of the replication.
    TME of type TNC (src/thesis_h1/tme.py, port of the official MATLAB code). Validation (P3): expected
    marginal covariances of the fitted distribution against those of the data, relative Frobenius error < 5% in all
    three modes; the error of the mean of the 100 surrogates and the median error of the single surrogate are also reported.
    Statistic: Isomap − PCA advantage at d = 3 (PCA to 25 of the points, Isomap k = 10) on the 800 points, data and 100
    surrogates (seeds 0-99); p = (1 + #{surrogates ≥ data}) / 101. Verdict: "nonlinear structure beyond the marginals"
    if p < 0.05 in 3 sessions out of 3, "explained by the marginals" if p ≥ 0.05 in 3 out of 3, "mixed" otherwise;
    "not applicable" if the validation fails.
Note: by construction the TME preserves the covariance between neurons, hence the PCA spectrum: it does not test linear
dimensionality, only what goes beyond the marginal covariances.

Outputs: ~/thesis_h1_work/results/literature/step10_tme.csv, step10_null.csv, step10_single.csv, step10_tme.md,
step10_tme.png, step10_isomap_single.png; the first 5 surrogates for R9 in ~/thesis_h1_work/literature_cache/phase2/;
rows in the run registry (condition 10_tme_*).
Usage: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=~/thesis_h1_git/src python scripts/literature_step10.py
"""
import pickle
import sys
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                                # noqa: E402
import numpy as np                                                             # noqa: E402
import pandas as pd                                                            # noqa: E402
from sklearn.decomposition import PCA                                          # noqa: E402
from sklearn.preprocessing import StandardScaler                               # noqa: E402

from thesis_h1 import io, palette, results, literature, literature2, tme                 # noqa: E402

OUT = results.RESULTS / "literature"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = results.WORK / "literature_cache" / "phase2"
CACHE.mkdir(parents=True, exist_ok=True)
N_SUR = 100
N_GEM = 5
D_STAT = 3
D_FIG = 10
EXPECTED_LEGACY = {"data": 3.6, "twins_mean": -9.6, "twins_max": -5.3}   # notebook 03, § 5.2 (d = 3)
KEYS = dict(experiment="literature", latent_dim=None, decoder=None, split="session", fold=None, k_shot=None, H_ms=None)
T0 = time.time()


def log(msg):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {msg}", flush=True)


def comma(x, nd=1, sign=False):
    s = f"{x:+.{nd}f}" if sign else f"{x:.{nd}f}"
    return s.replace(".", ",").replace("-", "−")


def nwb(sid):
    return literature.DATA / "sub-HumanPitt-held-in-calib" / f"sub-HumanPitt-held-in-calib_ses-{sid}.nwb"


def single_trial(counts, phase):
    """§ 5.2 of notebook 03: Isomap advantage (points) for d = 1..15, data and twins."""
    Z = StandardScaler().fit_transform(io.smooth_rates(np.asarray(counts), sigma_ms=50).astype(float))
    mov = np.isin(phase, io.MOVEMENT_PHASES)
    bins_all = np.flatnonzero(mov)
    bins = bins_all[bins_all % literature2.STEP == 0]
    Z25 = PCA(n_components=literature2.K_PCA).fit(Z[bins_all]).transform(Z)
    ri, rp = literature2.curve_rv(Z25[bins])
    gem = [literature2.curve_rv(literature2.phases_random(Z25, g)[bins]) for g in range(N_GEM)]
    return 100 * (rp - ri), np.array([100 * (p - i) for i, p in gem]), len(bins)


# A0 · check with the legacy bins (S1)
sl = io.load_session(nwb(literature.SESSIONS["S1"]), bins="legacy")
act = io.active_units(sl, min_hz=0.5)
phase_l = np.array([literature.name_base(p) for p in io.phase_per_bin(sl)], dtype=object)
adv_l, gem_l, n_l = single_trial(sl.spikes[:, act], phase_l)
legacy = {"data": adv_l[D_STAT - 1], "twins_mean": gem_l[:, D_STAT - 1].mean(),
          "twins_max": gem_l[:, D_STAT - 1].max()}
ctrl_legacy = {k: round(float(v), 1) == EXPECTED_LEGACY[k] for k, v in legacy.items()}
log(f"legacy check S1 ({n_l} points): " + ", ".join(f"{k} {v:+.2f} (expected {EXPECTED_LEGACY[k]:+.1f})"
                                                      for k, v in legacy.items()))
results.log_run({**{f"{k}_d3": float(v) for k, v in legacy.items()}, "reproduced": float(all(ctrl_legacy.values())),
                 "n_points": float(n_l)}, method="Isomap", session="S1_set_1", bins="legacy",
                condition="10_tme_single_check", seed=0, **KEYS)
del sl

rows, nulls, single = [], [], []
for name, sid in literature.SESSIONS.items():
    s = literature.load(name)
    # A · single trial, FALCON bins
    adv, gem, n_pt = single_trial(s.counts, s.phase)
    for d in range(1, literature2.D_MAX + 1):
        single.append(dict(session=name, d=d, data=adv[d - 1], **{f"twin{g}": gem[g, d - 1] for g in range(N_GEM)}))
    g3 = gem[:, D_STAT - 1]
    log(f"{name} single trial ({n_pt} points): data {adv[D_STAT - 1]:+.2f}, twins mean {g3.mean():+.2f} "
        f"[{g3.min():+.2f}, {g3.max():+.2f}]")
    results.log_run({"data_d3": float(adv[D_STAT - 1]), "twins_mean_d3": float(g3.mean()),
                     "twins_min_d3": float(g3.min()), "twins_max_d3": float(g3.max()),
                     "twins_below_data_d3": float((g3 < adv[D_STAT - 1]).sum()), "n_points": float(n_pt),
                     **{f"data_d{d}": float(adv[d - 1]) for d in range(1, D_FIG + 1)},
                     **{f"twins_mean_d{d}": float(gem[:, d - 1].mean()) for d in range(1, D_FIG + 1)}},
                    method="Isomap", session=s.key, bins="falcon", condition="10_tme_single", seed=0, **KEYS)

    # B · tensor of per-condition means and TME
    si = io.load_session(nwb(sid), bins="falcon")
    Zc = literature2.pipeline_02_sigma(s.counts, 50)
    Zc = (Zc - Zc.mean(0)) / Zc.std(0)
    X, rip = literature2.tensor_conditions(Zc, io.phase_windows(si, win_s=1.0, align="start"))
    ref = pickle.load(open(literature.REP / sid / "02_rep.pkl", "rb"))["arrays"]["PSTH"]
    diff = float(np.abs(literature2.impila(X) - ref).max())
    if diff > 1e-6:
        sys.exit(f"{name}: tensor of means differs from the replication PSTH ({diff:.2e}): stopping")
    np.savez_compressed(CACHE / f"cond_{name}.npz", X=X)
    sig, M = tme.marginals(X)
    me = tme.fit(sig, M)
    sur = [tme.campiona(me, np.random.default_rng(seed)) for seed in range(N_SUR)]
    val = tme.validation(me, sig, sur)
    valid = all(val[f"expected_{m}"] < 0.05 for m in "TNC")
    log(f"{name} TME {X.shape}: objective {me.objective_log:.1e}, iterations {me.iterations}, "
        + ", ".join(f"{k} {100 * v:.2f}%" for k, v in val.items()) + f" -> {'valid' if valid else 'NOT valid'}")
    results.log_run({**{f"err_{k}": float(v) for k, v in val.items()}, "valid": float(valid),
                     "objective_log": me.objective_log, "cost_original": tme.cost_original(me),
                     "iterations": float(me.iterations), "convergence": float(me.convergence),
                     "T": float(X.shape[0]), "N": float(X.shape[1]), "C": float(X.shape[2]),
                     "repetitions_min": float(rip.min()), "repetitions_max": float(rip.max())},
                    method="TME", session=s.key, bins="falcon", condition="10_tme_validation", seed=None, **KEYS)
    stat = literature2.advantage_isomap(literature2.pca25(literature2.impila(X)), d=D_STAT)
    r = dict(session=name, N=X.shape[1], T=X.shape[0], C=X.shape[2], valid=valid,
             objective_log=me.objective_log, **{f"err_{k}": v for k, v in val.items()}, stat_data=stat,
             single_points=n_pt, single_data_d3=adv[D_STAT - 1], single_twins_mean_d3=g3.mean(), single_twins_min_d3=g3.min(),
             single_twins_max_d3=g3.max())
    if valid:
        ns = np.array([literature2.advantage_isomap(literature2.pca25(literature2.impila(Y)), d=D_STAT) for Y in sur])
        p = literature2.p_permutation(stat, ns)
        nulls += [dict(session=name, seed=i, stat=v) for i, v in enumerate(ns)]
        r.update(null_median=float(np.median(ns)), null_p95=float(np.percentile(ns, 95)), null_min=float(ns.min()),
                 null_max=float(ns.max()), p=p, n_sur=N_SUR)
        np.savez_compressed(CACHE / f"tme_{name}.npz", sur=np.stack(sur[:5]), semi=np.arange(5))
        log(f"{name} TME: data {stat:+.2f}, null median {np.median(ns):+.2f} [95th {np.percentile(ns, 95):+.2f}], "
            f"p {p:.3f}")
        results.log_run({"stat_data": stat, "null_median": r["null_median"], "null_p95": r["null_p95"],
                         "null_min": r["null_min"], "null_max": r["null_max"], "p": p, "n_sur": float(N_SUR)},
                        method="Isomap", session=s.key, bins="falcon", condition="10_tme_isomap", seed=0, **KEYS)
    rows.append(r)
    del s, si, sur

df = pd.DataFrame(rows)
df.to_csv(OUT / "step10_tme.csv", index=False)
pd.DataFrame(nulls).to_csv(OUT / "step10_null.csv", index=False)
pd.DataFrame(single).to_csv(OUT / "step10_single.csv", index=False)
if not df.valid.all():
    verdict, code = "not_applicable", np.nan
else:
    below = int((df.p < 0.05).sum())
    verdict = {3: "nonlinear structure beyond the marginals", 0: "explained by the marginals"}.get(below, "mixed")
    code = {3: 2.0, 0: 0.0}.get(below, 1.0)
results.log_run({"R10_code": code, "sessions_valid": float(df.valid.sum()),
                 "sessions_p_below_005": float((df.get("p", pd.Series(dtype=float)) < 0.05).sum())},
                method="Isomap", session=None, bins="falcon", condition="10_tme_R10", seed=None, **KEYS)
log(f"R10: {verdict}")

# figures
nd = pd.DataFrame(nulls)
fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.8), sharey=True)
for ax, r in zip(axs, rows):
    palette.style_axes(ax)
    if r["valid"]:
        v = nd[nd.session == r["session"]].stat
        ax.hist(v, bins=20, color=palette.NEUTRAL, alpha=0.7, label=f"{N_SUR} TME surrogates")
        ax.axvline(r["null_p95"], color=palette.INK2, ls=":", lw=1, label="95th percentile of the null")
        ttl = f"{r['session']}: data {comma(r['stat_data'], sign=True)}, p = {comma(r['p'], 3)}"
    else:
        ttl = f"{r['session']}: TME not validated"
    ax.axvline(r["stat_data"], color=palette.color("Isomap"), lw=2.2, label="data")
    ax.set_title(ttl, fontsize=9.5, color=palette.INK)
    ax.set_xlabel("Isomap − PCA advantage at d = 3 (points)", fontsize=9, color=palette.INK2)
axs[0].set_ylabel("surrogates", fontsize=9, color=palette.INK2)
axs[0].legend(fontsize=7.5, frameon=False, loc="upper left")
fig.suptitle(f"R10 · Per-condition means (16 × 50 bins) against the TME null (time, neurons, conditions) — {verdict}",
             fontsize=10.5, color=palette.INK)
fig.tight_layout()
fig.savefig(OUT / "step10_tme.png", dpi=200, bbox_inches="tight")

sd = pd.DataFrame(single)
fig, axs = plt.subplots(1, 3, figsize=(12.5, 3.8), sharey=True)
for ax, (name, g) in zip(axs, sd.groupby("session", sort=False)):
    palette.style_axes(ax)
    g = g[g.d <= D_FIG]
    for k in range(N_GEM):
        ax.plot(g.d, g[f"twin{k}"], color=palette.NEUTRAL, lw=1, label="linear Gaussian twins (5)" if k == 0 else None)
    ax.plot(g.d, g.data, "o-", color=palette.color("Isomap"), lw=2, ms=4, label="data")
    ax.axhline(0, color=palette.INK2, lw=0.8)
    r3 = g[g.d == D_STAT].iloc[0]
    gm = r3[[f"twin{k}" for k in range(N_GEM)]].astype(float)
    ax.set_title(f"{name}: d = 3 data {comma(r3.data, sign=True)}, twins {comma(gm.mean(), sign=True)}",
                 fontsize=9.5, color=palette.INK)
    ax.set_xlabel("dimensions d", fontsize=9, color=palette.INK2)
axs[0].set_ylabel("PCA − Isomap (percentage points)", fontsize=9, color=palette.INK2)
axs[0].legend(fontsize=7.5, frameon=False, loc="lower left")
fig.suptitle("Single trial, FALCON bins: Isomap advantage over PCA (above zero = Isomap better)",
             fontsize=10.5, color=palette.INK)
fig.tight_layout()
fig.savefig(OUT / "step10_isomap_single.png", dpi=200, bbox_inches="tight")
log("figures saved")

# md
md = ["# Phase 2 · R10 · TME null for nonlinear structure", "",
      f"Commit {results.code_version()[0]}. Rules: docs/literature/rules_decision.md (commit 1aa7c5d), clarifications "
      "P3-P5. TME: src/thesis_h1/tme.py, port of the official MATLAB code (github.com/gamaleldin/TME), type TNC.", "",
      "**Note.** By construction the TME preserves the covariance between neurons, hence it also preserves the PCA spectrum. For "
      "this reason it is not used to test linear dimensionality, only what goes beyond the marginal covariances. Here it is "
      "used for the nonlinear structure of slide 8.", "",
      "## Validation of the surrogates (relative Frobenius error on the marginal covariances)", "",
      "| Session | T × N × C | expected: time / neurons / conditions | mean of 100 surrogates | single surrogate (median) | valid |",
      "| --- | --- | --- | --- | --- | --- |"]
for r in rows:
    md.append(f"| {r['session']} | {r['T']} × {r['N']} × {r['C']} | "
              + " / ".join(f"{comma(100 * r[f'err_expected_{m}'], 3)}%" for m in "TNC") + " | "
              + " / ".join(f"{comma(100 * r[f'err_mean_{m}'])}%" for m in "TNC") + " | "
              + " / ".join(f"{comma(100 * r[f'err_single_{m}'])}%" for m in "TNC") + f" | {'sì' if r['valid'] else 'no'} |")
md += ["", "The 5% criterion (P3) applies to the expected covariances of the fitted distribution. The mean of the surrogates "
       "and the single surrogate deviate from them through sampling noise alone (the surrogates are centered on the tensor "
       "of the data means, M.TNC).", "",
       "## Statistic: Isomap − PCA advantage at d = 3 on the per-condition means", "",
       "| Session | data | TME null: median [95th percentile; max] | p |", "| --- | --- | --- | --- |"]
for r in rows:
    if r["valid"]:
        md.append(f"| {r['session']} | {comma(r['stat_data'], sign=True)} | {comma(r['null_median'], sign=True)} "
                  f"[{comma(r['null_p95'], sign=True)}; {comma(r['null_max'], sign=True)}] | {comma(r['p'], 3)} |")
    else:
        md.append(f"| {r['session']} | {comma(r['stat_data'], sign=True)} | not computed (TME not validated) | — |")
md += ["", f"**R10**: {verdict}.", "",
       "## Single trial with FALCON bins (updates +3.6 and −9.6 of slide 8)", "",
       "| Session | points | data, d = 3 | twins, mean [min, max] | twins below data |", "| --- | --- | --- | --- | --- |"]
for r in rows:
    below = int((sd[(sd.session == r["session"]) & (sd.d == D_STAT)][[f"twin{k}" for k in range(N_GEM)]].values
                 < r["single_data_d3"]).sum())
    md.append(f"| {r['session']} | {r['single_points']} | {comma(r['single_data_d3'], sign=True)} | "
              f"{comma(r['single_twins_mean_d3'], sign=True)} [{comma(r['single_twins_min_d3'], sign=True)}, "
              f"{comma(r['single_twins_max_d3'], sign=True)}] | {below} of {N_GEM} |")
md += ["", f"Check with the legacy bins in S1 ({n_l} points) against notebook 03: data {comma(legacy['data'], 2, True)} "
       f"(expected +3.6), twins mean {comma(legacy['twins_mean'], 2, True)} (expected −9.6), best twin "
       f"{comma(legacy['twins_max'], 2, True)} (expected −5.3): "
       f"{'reproduced' if all(ctrl_legacy.values()) else 'NOT reproduced'}."]
(OUT / "step10_tme.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
log("end")
