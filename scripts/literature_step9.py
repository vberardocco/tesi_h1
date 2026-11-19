"""Phase 2, R9 (literature comparison): intrinsic dimension estimators as in Altan et al. 2021.

Rules in docs/literature/rules_decision.md (commit 1aa7c5d), clarifications P1, P2, P6, P7.
Inputs per session (FALCON bins):
  (a) pipeline of notebook 02 (Gaussian 50 ms, z-score), all active channels, bins of the movement phases;
  (b) GPFA latents F_GP12["all"] of the replication (25 dimensions), same bins;
  (c) per-condition means: the stacked 50 × N × 16 tensor of R10 (800 points × N);
  (d) 5 random-phase twins of the whole session (a), seeds 0-4, same bins;
  (e) the first 5 TME surrogates of R10 (seeds 0-4), stacked as (c). Only if R10 validated them.
Estimators (skdim 0.5.0): TwoNN, Levina-Bickel MLE with k = 10 and 20, correlation dimension, DANCo.
Sampling: n ∈ {500, 1000, 2000, 5000, all} (only the values below the number of points, plus all), 10 draws
(seeds 0-9) per n; with all points a single estimate.
Criteria (P2): stable = |med(all) − med(n_min)| / med(n_min) < 20% and IQR < 20% of the median at every n with 10
draws from n_min (1000; 500 for (c), P1); distinguishable = data with all points outside [min, max] of the 5 twins.
Prediction P2 (derived from Altan 2021; the estimator × session grid is ours) for each estimator and session: (a) unstable
and not distinguishable, (b) and (c) stable. "confirmed" if it holds in all cells, otherwise "not confirmed".
Computation: parallel processes (N_PROC); first the four fast estimators, then DANCo with draws 0-4 (and all
points), then 5-9 only if the predicted time of the block stays under 2 hours (P7). Every finite estimate is written immediately to
~/thesis_h1_work/literature_cache/phase2/step9_estimates.jsonl: rerunning the script resumes from there.

Outputs: ~/thesis_h1_work/results/literature/step9_iddim.csv (summary per n), step9_criteria.csv, step9_estimates.csv (all
estimates), step9_iddim.md, step9_iddim.png; rows in the run registry (condition 9_iddim_*).
Usage: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_THREADING_LAYER=workqueue PYTHONPATH=~/thesis_h1_git/src \
     python scripts/literature_step9.py
"""
import json
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                                # noqa: E402
import numpy as np                                                             # noqa: E402
import pandas as pd                                                            # noqa: E402

from thesis_h1 import palette, results, literature, literature2                          # noqa: E402

OUT = results.RESULTS / "literature"
OUT.mkdir(parents=True, exist_ok=True)
CACHE = results.WORK / "literature_cache" / "phase2"
ESTIMATES = CACHE / "step9_estimates.jsonl"
START = CACHE / "step9_start.json"
N_PROC = 6
LIMITE_S = 2 * 3600
VELOCI = ("TwoNN", "MLE10", "MLE20", "CorrInt")
N_GEM = 5
NAMES_ING = {"a": "(a) single trial, channels", "b": "(b) GPFA latents", "c": "(c) per-condition means",
            "d": "(d) twins of (a)", "e": "(e) TME surrogates of (c)"}
KEYS = dict(experiment="literature", decoder=None, bins="falcon", split="session", fold=None, k_shot=None, H_ms=None)
T0 = time.time()


def log(msg):
    print(f"[{(time.time() - T0) / 60:6.1f} min] {msg}", flush=True)


def comma(x, nd=1):
    return f"{x:.{nd}f}".replace(".", ",")


def prepare():
    """Save the inputs to .npy (once only) and return {(session, input, replica): path}."""
    ing = {}
    for name, sid in literature.SESSIONS.items():
        file_ing = {k: CACHE / f"{name}_{k}.npy" for k in ["a", "b", "c"] + [f"d{g}" for g in range(N_GEM)]}
        tme_npz = CACHE / f"tme_{name}.npz"
        if tme_npz.exists():
            file_ing.update({f"e{g}": CACHE / f"{name}_e{g}.npy" for g in range(N_GEM)})
        if not all(p.exists() for p in file_ing.values()):
            s = literature.load(name)
            mov = np.isin(s.phase, literature.PHASES_MOVEMENT)
            Z = literature.pipeline_02(s.counts)
            Z = (Z - Z.mean(0)) / Z.std(0)
            np.save(file_ing["a"], Z[mov])
            for g in range(N_GEM):
                np.save(file_ing[f"d{g}"], literature2.phases_random(Z, g)[mov])
            F = pickle.load(open(literature.REP / sid / "04_rep.pkl", "rb"))["dicts"]["F_GP12"]["all"]
            if len(F) != len(Z):
                raise ValueError(f"{name}: GPFA latents {F.shape} and data {Z.shape} have different lengths")
            np.save(file_ing["b"], np.asarray(F, float)[mov])
            del F
            cond = CACHE / f"cond_{name}.npz"
            if not cond.exists():
                raise FileNotFoundError(f"{cond}: the R10 output is missing (scripts/literature_step10.py)")
            np.save(file_ing["c"], literature2.impila(np.load(cond)["X"]))
            if tme_npz.exists():                                    # only if R10 validated the TME
                t = np.load(tme_npz)
                for g in range(N_GEM):
                    np.save(file_ing[f"e{g}"], literature2.impila(t["sur"][g]))
            log(f"{name}: inputs saved ({int(mov.sum())} movement bins, {Z.shape[1]} channels)")
        for k, p in file_ing.items():
            ing[(name, k[0], int(k[1:]) if len(k) > 1 else 0)] = p
    return ing


def jobs(ing, estimators, semi):
    out = []
    for (name, k, rep), p in ing.items():
        n_tot = len(np.load(p, mmap_mode="r"))
        for n in literature2.values_n(n_tot):
            for j in ([0] if n == n_tot else semi):
                if n == n_tot and 0 not in semi:
                    continue
                out.append(dict(session=name, input=k, replica=rep, n=n, n_tot=n_tot, seed=j, path=str(p),
                                estimators=list(estimators)))
    return sorted(out, key=lambda x: -x["n"] * np.load(x["path"], mmap_mode="r").shape[1])


def key(r, stim):
    return (r["session"], r["input"], r["replica"], int(r["n"]), int(r["seed"]), stim)


def done():
    if not ESTIMATES.exists():
        return {}
    out = {}
    for row in ESTIMATES.read_text().splitlines():
        r = json.loads(row)
        out[key(r, r["estimator"])] = r
    return out


def run(jobs, already):
    todo = []
    for jb in jobs:
        missing = [s for s in jb["estimators"] if key(jb, s) not in already]
        if missing:
            todo.append({**jb, "estimators": missing})
    if not todo:
        return 0.0
    t = time.time()
    with ProcessPoolExecutor(max_workers=N_PROC) as ex, open(ESTIMATES, "a") as f:
        fut = {ex.submit(literature2.estimates_job, jb["path"], jb["n"], jb["seed"], tuple(jb["estimators"])): jb
               for jb in todo}
        for i, fu in enumerate(as_completed(fut), 1):
            jb = fut[fu]
            for stim, v, sec in fu.result():
                r = {k: jb[k] for k in ("session", "input", "replica", "n", "n_tot", "seed")}
                r.update(estimator=stim, estimate=v, seconds=sec)
                f.write(json.dumps(r) + "\n")
                f.flush()
                already[key(r, stim)] = r
            if i % 50 == 0 or i == len(todo):
                log(f"  {i}/{len(todo)} jobs finished")
    return time.time() - t


if __name__ == "__main__":
    CACHE.mkdir(parents=True, exist_ok=True)
    if not START.exists():
        START.write_text(json.dumps({"start": time.time()}))
    start = json.loads(START.read_text())["start"]
    ing = prepare()
    already = done()
    log(f"inputs: {len(ing)}; estimates already done: {len(already)}")

    # 1 · fast estimators
    dt = run(jobs(ing, VELOCI, range(10)), already)
    log(f"fast estimators: {dt / 60:.1f} min")
    # 2 · DANCo, draws 0-4 (and all points)
    dt = run(jobs(ing, ("DANCo",), range(5)), already)
    log(f"DANCo, draws 0-4: {dt / 60:.1f} min")
    # 3 · DANCo, draws 5-9 if the block stays under 2 hours (P7)
    rest = jobs(ing, ("DANCo",), range(5, 10))
    sec = {}
    for r in already.values():
        if r["estimator"] == "DANCo":
            sec.setdefault((r["input"], r["n"], r["n_tot"]), []).append(r["seconds"])
    cost = sum(np.mean(sec.get((jb["input"], jb["n"], jb["n_tot"]), [60])) for jb in rest
                if key(jb, "DANCo") not in already)
    predicted = (time.time() - start) + cost / N_PROC
    draws_danco = 10
    if predicted > LIMITE_S:
        draws_danco = 5
        log(f"DANCo: predicted block time {predicted / 60:.0f} min > 120: stopping at 5 draws (P7)")
    else:
        log(f"DANCo: predicted block time {predicted / 60:.0f} min: running draws 5-9")
        dt = run(rest, already)
        log(f"DANCo, draws 5-9: {dt / 60:.1f} min")
    duration_block = time.time() - start

    # 4 · summaries and criteria
    st = pd.DataFrame(list(already.values()))
    st = st[~((st.estimator == "DANCo") & (st.seed >= draws_danco))]
    st.sort_values(["session", "input", "replica", "estimator", "n", "seed"]).to_csv(OUT / "step9_estimates.csv",
                                                                                         index=False)
    rias = (st.groupby(["session", "input", "replica", "estimator", "n", "n_tot"]).estimate
            .agg(median="median", q25=lambda v: np.percentile(v, 25), q75=lambda v: np.percentile(v, 75),
                 minimum="min", max="max", n_draw="count").reset_index())
    rias.to_csv(OUT / "step9_iddim.csv", index=False)
    estimators = list(literature2.ESTIMATORS)

    def per_n(name, k, rep, stim):
        g = st[(st.session == name) & (st.input == k) & (st.replica == rep) & (st.estimator == stim)]
        return {int(n): v.estimate.to_numpy() for n, v in g.groupby("n")}

    crit = []
    for name in literature.SESSIONS:
        for stim in estimators:
            pa, pb, pc = (per_n(name, k, 0, stim) for k in "abc")
            if not (pa and pb and pc):
                continue
            sa, sb, sc = literature2.stability(pa, 1000), literature2.stability(pb, 1000), literature2.stability(pc, 500)
            ga = [per_n(name, "d", g, stim) for g in range(N_GEM)]
            ge = [per_n(name, "e", g, stim) for g in range(N_GEM)]
            ta = [float(x[max(x)][0]) for x in ga]
            te = [float(x[max(x)][0]) for x in ge if x]
            va, vc = float(pa[max(pa)][0]), float(pc[max(pc)][0])
            da = literature2.distinguishable(va, ta)
            dc = literature2.distinguishable(vc, te) if te else np.nan
            prev = literature2.prediction_altan(sa["stable"], da, sb["stable"], sc["stable"])
            crit.append(dict(session=name, estimator=stim, a_all=va, a_change=sa["change"], a_iqr_max=sa["iqr_max"],
                             a_stable=sa["stable"], twins_min=min(ta), twins_max=max(ta), a_distinguishable=da,
                             twins_stable=sum(literature2.stability(x, 1000)["stable"] for x in ga),
                             b_all=float(pb[max(pb)][0]), b_change=sb["change"], b_iqr_max=sb["iqr_max"],
                             b_stable=sb["stable"], c_all=vc, c_change=sc["change"], c_iqr_max=sc["iqr_max"],
                             c_stable=sc["stable"], tme_min=min(te) if te else np.nan,
                             tme_max=max(te) if te else np.nan, c_distinguishable_tme=dc, prediction_altan=prev,
                             draws=draws_danco if stim == "DANCo" else 10))
    cr = pd.DataFrame(crit)
    cr.to_csv(OUT / "step9_criteria.csv", index=False)
    replica = bool(cr.prediction_altan.all())
    verdict = "prediction P2 confirmed" if replica else "prediction P2 not confirmed"
    log(f"R9: {verdict} ({int(cr.prediction_altan.sum())} cells out of {len(cr)})")

    # run registry: one row per (session, input, estimator) and one for the criteria
    for name in literature.SESSIONS:
        for stim in estimators:
            for k in "abcde":
                g = rias[(rias.session == name) & (rias.input == k) & (rias.estimator == stim)]
                if g.empty:
                    continue
                m = {}
                for rep, gg in g.groupby("replica"):
                    suff = f"_r{rep}" if k in "de" else ""
                    for _, r in gg.iterrows():
                        tag = "all" if r.n == r.n_tot else f"n{int(r.n)}"
                        m[f"median_{tag}{suff}"] = float(r.median)
                        if r.n_draw > 1:
                            m[f"iqr_{tag}{suff}"] = float(r.q75 - r.q25)
                m["n_tot"] = float(g.n_tot.iloc[0])
                dim = np.load(ing[(name, k, 0)], mmap_mode="r").shape[1]
                results.log_run(m, method=stim, latent_dim=int(dim), session=literature.KEYS[name],
                                condition=f"9_iddim_{k}", seed=None, **KEYS)
    for r in cr.to_dict("records"):
        results.log_run({k: float(v) for k, v in r.items() if k not in ("session", "estimator")},
                        method=r["estimator"], latent_dim=None, session=literature.KEYS[r["session"]],
                        condition="9_iddim_criteria", seed=None, **KEYS)
    results.log_run({"R9_replica_altan": float(replica), "cells_ok": float(cr.prediction_altan.sum()),
                     "cells": float(len(cr)), "draws_danco": float(draws_danco),
                     "minutes_block": duration_block / 60},
                    method=None, latent_dim=None, session=None, condition="9_iddim_R9", seed=None, **KEYS)

    # 5 · figure: rows = estimators, columns = inputs (a), (b), (c); color = session
    COLS = dict(zip(literature.SESSIONS, palette.CAT[:3]))
    fig, axs = plt.subplots(len(estimators), 3, figsize=(13, 2.6 * len(estimators)), sharex="col")
    for i, stim in enumerate(estimators):
        for j, k in enumerate("abc"):
            ax = axs[i, j]
            palette.style_axes(ax)
            for name in literature.SESSIONS:
                ref = {"a": "d", "c": "e"}.get(k)
                if ref:
                    for g in range(N_GEM):
                        gg = rias[(rias.session == name) & (rias.input == ref) & (rias.replica == g)
                                  & (rias.estimator == stim)].sort_values("n")
                        ax.plot(gg.n, gg.median, color=palette.NEUTRAL, lw=0.8, alpha=0.8, zorder=1)
                gg = rias[(rias.session == name) & (rias.input == k) & (rias.estimator == stim)].sort_values("n")
                if gg.empty:
                    continue
                ax.fill_between(gg.n, gg.q25, gg.q75, color=COLS[name], alpha=0.25, lw=0, zorder=2)
                ax.plot(gg.n, gg.median, "o-", color=COLS[name], ms=3, lw=1.5, zorder=3,
                        label=name if (i == 0 and j == 0) else None)
            ax.set_xscale("log")
            c = cr[cr.estimator == stim]
            if not c.empty:
                flag = c[f"{k}_stable"].map({True: "sì", False: "no"}).tolist()
                txt = "stable: " + " · ".join(f"{s} {f}" for s, f in zip(c.session, flag))
                if k == "a":
                    txt += "\ndistinguishable: " + " · ".join(
                        f"{s} {'sì' if f else 'no'}" for s, f in zip(c.session, c.a_distinguishable))
                ax.text(0.02, 0.97, txt, transform=ax.transAxes, fontsize=6.8, va="top", color=palette.INK2)
            if i == 0:
                ax.set_title(NAMES_ING[k] + (" — the 5 twins in gray" if k == "a" else
                                            " — 5 TME surrogates in gray" if k == "c" else ""),
                             fontsize=9, color=palette.INK)
            if j == 0:
                ax.set_ylabel(f"{stim}\nstima", fontsize=9, color=palette.INK2)
            if i == len(estimators) - 1:
                ax.set_xlabel("points n (log scale)", fontsize=8.5, color=palette.INK2)
    axs[0, 0].legend(fontsize=7.5, frameon=False, loc="lower right", title="session (color)", title_fontsize=7.5)
    fig.suptitle(f"R9 · Intrinsic dimension against number of points (median and interquartile range over 10 "
                 f"draws{'; DANCo 5' if draws_danco == 5 else ''}) — {verdict}", fontsize=10.5,
                 color=palette.INK)
    fig.tight_layout()
    fig.savefig(OUT / "step9_iddim.png", dpi=200, bbox_inches="tight")
    log("figure saved")

    # 6 · markdown
    md = ["# Phase 2 · R9 · Intrinsic dimension estimators (as in Altan 2021)", "",
          f"Commit {results.code_version()[0]}. Rules: docs/literature/rules_decision.md (commit 1aa7c5d), clarifications "
          f"P1, P2, P6, P7. skdim 0.5.0. Block duration: {duration_block / 60:.0f} min; DANCo with "
          f"{draws_danco} draws per n" + (" (reduction P7)." if draws_danco == 5 else "."), "",
          "Values with all points; in parentheses the change of the median from n_min to all points and the maximum relative "
          "IQR (criterion: both below 20%).", "",
          "| Session | Estimator | (a) data | twins [min, max] | (a) stable | (a) distinguishable | (b) GPFA | "
          "(b) stable | (c) means | (c) stable | TME [min, max] | Altan prediction |",
          "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    sn = {True: "sì", False: "no"}
    for r in crit:
        md.append(f"| {r['session']} | {r['estimator']} | {comma(r['a_all'])} ({comma(100 * r['a_change'], 0)}%, "
                  f"{comma(100 * r['a_iqr_max'], 0)}%) | [{comma(r['twins_min'])}, {comma(r['twins_max'])}] | "
                  f"{sn[r['a_stable']]} | {sn[r['a_distinguishable']]} | {comma(r['b_all'])} "
                  f"({comma(100 * r['b_change'], 0)}%, {comma(100 * r['b_iqr_max'], 0)}%) | {sn[r['b_stable']]} | "
                  f"{comma(r['c_all'])} ({comma(100 * r['c_change'], 0)}%, {comma(100 * r['c_iqr_max'], 0)}%) | "
                  f"{sn[r['c_stable']]} | [{comma(r['tme_min'])}, {comma(r['tme_max'])}] | "
                  f"{'matches' if r['prediction_altan'] else 'does not match'} |")
    md += ["", f"**R9**: {verdict} (holds in {int(cr.prediction_altan.sum())} cells out of {len(cr)}: "
           f"{cr.estimator.nunique()} estimators × {cr.session.nunique()} sessions; the grid is ours, not Altan's)."]
    if not replica:
        no = cr[~cr.prediction_altan]
        md += ["", "Where it does not hold:", ""]
        for r in no.to_dict("records"):
            reasons = []
            if r["a_stable"]:
                reasons.append("(a) stable")
            if r["a_distinguishable"]:
                reasons.append("(a) distinguishable from the twins")
            if not r["b_stable"]:
                reasons.append("(b) unstable")
            if not r["c_stable"]:
                reasons.append("(c) unstable")
            md.append(f"- {r['session']}, {r['estimator']}: " + ", ".join(reasons))
    (OUT / "step9_iddim.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    log("end")
