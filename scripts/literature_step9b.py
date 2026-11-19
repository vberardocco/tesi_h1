"""Phase 2, R9b (literature comparison), POST HOC ANALYSIS: R9 with the Theiler window.

Requested by Valerio on 5 Nov after seeing R9 (prediction P2 not confirmed), to check the reading written in the summary: the
R9 estimates depend strongly on n, probably because with more points the nearest neighbors become consecutive bins
of the same trajectory (the "time chain" of notebook 03). It does not change the R9 verdict and has no verdict.

Same inputs, same draws (seeds 0-9) and same n as R9 (.npy files in ~/thesis_h1_work/literature_cache/phase2/).
Theiler window (Theiler 1986; notebook 03, § 5.3): among the neighbors of a point, points of the same
trial (for (c) and (e): of the same condition) at a time distance ≤ w are not counted; w = 0, 100, 200, 500 ms. Estimators: TwoNN,
MLE (k = 10, 20), correlation dimension (literature2.estimates_theiler, same formulas as skdim 0.5.0). DANCo is not included: skdim
does not accept precomputed distances and its time (~1 h 40 for 5 draws) is not worth it for a check.
Check: with w = 0 the estimates must coincide with those of R9 (step9_estimates.csv); otherwise the script stops without
logging. Threshold: TwoNN and MLE relative error < 1e-10, CorrInt < 1e-3. The first version (commit f66fa99) used 1e-4
for all and stopped: TwoNN and MLE coincided to 1e-15, CorrInt differed in 26 estimates out of 1059 (at most 3.6e-4)
because the radius r1 = median of the distances to the 10th neighbor often coincides with the distance of a pair, and whether that
pair is counted (d < r1) depends on the last binary digit (verified on the worst case). The per-estimator threshold was
chosen AFTER seeing the check fail (post hoc).
Criteria as in R9 (P1, P2), for each window, descriptive only.

Outputs: ~/thesis_h1_work/results/literature/step9b_theiler_estimates.csv, step9b_theiler.csv, step9b_theiler_criteria.csv,
step9b_theiler.md, step9b_theiler.png; rows in the run registry (condition 9b_theiler_*).
Usage: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_THREADING_LAYER=workqueue PYTHONPATH=~/thesis_h1_git/src \
     python scripts/literature_step9b.py
"""
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                                # noqa: E402
import numpy as np                                                             # noqa: E402
import pandas as pd                                                            # noqa: E402

from thesis_h1 import palette, results, literature, literature2                          # noqa: E402

OUT = results.RESULTS / "literature"
CACHE = results.WORK / "literature_cache" / "phase2"
ESTIMATES = CACHE / "step9b_estimates.jsonl"
N_PROC = 6
N_GEM = 5
FIN = {ms: ms // 20 for ms in literature2.WINDOWS_MS}          # ms -> bin
KEYS = dict(experiment="literature", decoder=None, bins="falcon", split="session", fold=None, k_shot=None, H_ms=None)
T0 = time.time()


def log(msg):
    print(f"[{(time.time() - T0) / 60:6.1f} min] {msg}", flush=True)


def comma(x, nd=1):
    return f"{x:.{nd}f}".replace(".", ",").replace("-", "−")


def meta():
    """Group and time (bin) of each point: trial and bin index for (a), (b), (d); condition and bin in the window
    for (c), (e). Return {(session, input): path of the 2 × n_tot .npy}."""
    out = {}
    for name in literature.SESSIONS:
        s = literature.load(name, check_rep=False)
        mov = np.isin(s.phase, literature.PHASES_MOVEMENT)
        p = CACHE / f"{name}_meta_mov.npy"
        np.save(p, np.stack([np.asarray(s.trial)[mov], np.flatnonzero(mov)]).astype(np.int64))
        n_a = len(np.load(CACHE / f"{name}_a.npy", mmap_mode="r"))
        if n_a != int(mov.sum()):
            raise ValueError(f"{name}: {n_a} points in (a) and {int(mov.sum())} movement bins")
        T, C = 50, 16
        pc = CACHE / f"{name}_meta_c.npy"
        np.save(pc, np.stack([np.repeat(np.arange(C), T), np.tile(np.arange(T), C)]).astype(np.int64))
        for k in "abd":
            out[(name, k)] = p
        for k in "ce":
            out[(name, k)] = pc
    return out


def inputs():
    ing = {}
    for name in literature.SESSIONS:
        for k in ["a", "b", "c"] + [f"d{g}" for g in range(N_GEM)] + [f"e{g}" for g in range(N_GEM)]:
            p = CACHE / f"{name}_{k}.npy"
            if p.exists():
                ing[(name, k[0], int(k[1:]) if len(k) > 1 else 0)] = p
    return ing


def key(r):
    return (r["session"], r["input"], int(r["replica"]), int(r["n"]), int(r["seed"]), int(r["window_ms"]),
            r["estimator"])


if __name__ == "__main__":
    mt = meta()
    ing = inputs()
    already = {}
    if ESTIMATES.exists():
        for row in ESTIMATES.read_text().splitlines():
            r = json.loads(row)
            already[key(r)] = r
    jobs = []
    for (name, k, rep), p in ing.items():
        n_tot = len(np.load(p, mmap_mode="r"))
        for n in literature2.values_n(n_tot):
            for j in ([0] if n == n_tot else range(10)):
                done = all(key(dict(session=name, input=k, replica=rep, n=n, seed=j, window_ms=ms,
                                        estimator=s)) in already for ms in FIN for s in literature2.ESTIMATORS_THEILER)
                if not done:
                    jobs.append(dict(session=name, input=k, replica=rep, n=n, n_tot=n_tot, seed=j, path=str(p),
                                     meta=str(mt[(name, k)])))
    jobs.sort(key=lambda x: -x["n"])
    log(f"inputs {len(ing)}, jobs to do {len(jobs)}, estimates already done {len(already)}")
    with ProcessPoolExecutor(max_workers=N_PROC) as ex, open(ESTIMATES, "a") as f:
        fut = {ex.submit(literature2.estimates_job_theiler, jb["path"], jb["meta"], jb["n"], jb["seed"],
                         tuple(FIN.values())): jb for jb in jobs}
        for i, fu in enumerate(as_completed(fut), 1):
            jb = fut[fu]
            for w, s, v, sec in fu.result():
                ms = {b: m for m, b in FIN.items()}[w]
                r = {k: jb[k] for k in ("session", "input", "replica", "n", "n_tot", "seed")}
                r.update(window_ms=ms, estimator=s, estimate=v, seconds=sec)
                f.write(json.dumps(r) + "\n")
                f.flush()
                already[key(r)] = r
            if i % 100 == 0 or i == len(jobs):
                log(f"  {i}/{len(jobs)} jobs finished")

    st = pd.DataFrame(list(already.values()))
    st.sort_values(["session", "input", "replica", "window_ms", "estimator", "n", "seed"]).to_csv(
        OUT / "step9b_theiler_estimates.csv", index=False)

    # check: w = 0 against R9
    r9 = pd.read_csv(OUT / "step9_estimates.csv")
    k9 = ["session", "input", "replica", "estimator", "n", "seed"]
    m = st[st.window_ms == 0].merge(r9[k9 + ["estimate"]], on=k9, suffixes=("", "_r9"))
    m["err"] = np.abs(m.estimate - m.estimate_r9) / np.abs(m.estimate_r9)
    err_s = m.groupby("estimator").err.max().to_dict()
    err = float(m.err.max())
    thresholds = {s: (1e-3 if s == "CorrInt" else 1e-10) for s in literature2.ESTIMATORS_THEILER}
    expected = len(r9[r9.estimator.isin(literature2.ESTIMATORS_THEILER)])
    log(f"check w = 0 against R9: {len(m)} estimates out of {expected}, maximum relative error per estimator "
        + ", ".join(f"{s} {e:.1e}" for s, e in err_s.items()))
    if len(m) != expected or any(err_s[s] > thresholds[s] for s in thresholds):
        msg = f"check failed ({len(m)} estimates out of {expected}, errors {err_s}): stopping without logging"
        (OUT / "step9b_check_failed.md").write_text(msg + "\n")
        sys.exit(msg)

    rias = (st.groupby(["window_ms", "session", "input", "replica", "estimator", "n", "n_tot"]).estimate
            .agg(median="median", q25=lambda v: np.percentile(v, 25), q75=lambda v: np.percentile(v, 75),
                 n_draw="count").reset_index())
    rias.to_csv(OUT / "step9b_theiler.csv", index=False)

    def per_n(ms, name, k, rep, stim):
        g = st[(st.window_ms == ms) & (st.session == name) & (st.input == k) & (st.replica == rep)
               & (st.estimator == stim)]
        return {int(n): v.estimate.to_numpy() for n, v in g.groupby("n")}

    crit = []
    for ms in FIN:
        for name in literature.SESSIONS:
            for stim in literature2.ESTIMATORS_THEILER:
                pa, pb, pc = (per_n(ms, name, k, 0, stim) for k in "abc")
                sa, sb, sc = literature2.stability(pa, 1000), literature2.stability(pb, 1000), literature2.stability(pc, 500)
                ga = [per_n(ms, name, "d", g, stim) for g in range(N_GEM)]
                ge = [per_n(ms, name, "e", g, stim) for g in range(N_GEM)]
                ta = [float(x[max(x)][0]) for x in ga]
                te = [float(x[max(x)][0]) for x in ge]
                va, vc = float(pa[max(pa)][0]), float(pc[max(pc)][0])
                da = literature2.distinguishable(va, ta)
                crit.append(dict(window_ms=ms, session=name, estimator=stim, a_n500=float(np.median(pa[500])),
                                 a_all=va, a_change=sa["change"], a_stable=sa["stable"], twins_min=min(ta),
                                 twins_max=max(ta), a_distinguishable=da, b_n500=float(np.median(pb[500])),
                                 b_all=float(pb[max(pb)][0]), b_change=sb["change"], b_stable=sb["stable"],
                                 c_n500=float(np.median(pc[500])), c_all=vc, c_change=sc["change"],
                                 c_stable=sc["stable"], tme_min=min(te), tme_max=max(te),
                                 c_distinguishable_tme=literature2.distinguishable(vc, te),
                                 prediction_altan=literature2.prediction_altan(sa["stable"], da, sb["stable"],
                                                                           sc["stable"])))
    cr = pd.DataFrame(crit)
    cr.to_csv(OUT / "step9b_theiler_criteria.csv", index=False)
    for r in crit:
        results.log_run({k: float(v) for k, v in r.items() if k not in ("window_ms", "session", "estimator")},
                        method=r["estimator"], latent_dim=None, session=literature.KEYS[r["session"]],
                        condition=f"9b_theiler_w{r['window_ms']}", seed=None, **KEYS)
    results.log_run({"error_max_w0": err, **{f"error_max_w0_{s}": float(e) for s, e in err_s.items()},
                     "corrint_estimates_different_1e6": float((m[m.estimator == "CorrInt"].err > 1e-6).sum()),
                     "estimates_compared": float(len(m))}, method=None, latent_dim=None,
                    session=None, condition="9b_theiler_check", seed=None, **KEYS)

    # figure: S1; rows = estimators, columns = (a), (b), (c); color = window; bands = twins or TME (min-max)
    COLS = dict(zip(FIN, (palette.NEUTRAL, palette.CAT[0], palette.CAT[1], palette.CAT[2])))
    fig, axs = plt.subplots(4, 3, figsize=(13, 10.5), sharex="col")
    for i, stim in enumerate(literature2.ESTIMATORS_THEILER):
        for j, k in enumerate("abc"):
            ax = axs[i, j]
            palette.style_axes(ax)
            for ms in FIN:
                sel = (rias.window_ms == ms) & (rias.session == "S1") & (rias.estimator == stim)
                ref = {"a": "d", "c": "e"}.get(k)
                if ref:
                    gg = rias[sel & (rias.input == ref)].groupby("n").median.agg(["min", "max"]).reset_index()
                    ax.fill_between(gg.n, gg["min"], gg["max"], color=COLS[ms], alpha=0.15, lw=0)
                g = rias[sel & (rias.input == k)].sort_values("n")
                ax.plot(g.n, g.median, "o-", color=COLS[ms], ms=3, lw=1.6,
                        label=f"window {ms} ms" + (" (= R9)" if ms == 0 else "") if (i == 0 and j == 0) else None)
            ax.set_xscale("log")
            if i == 0:
                ax.set_title({"a": "(a) single trial — bands: 5 twins", "b": "(b) GPFA latents",
                              "c": "(c) per-condition means — bands: 5 TME"}[k], fontsize=9, color=palette.INK)
            if j == 0:
                ax.set_ylabel(f"{stim}\nstima", fontsize=9, color=palette.INK2)
            if i == 3:
                ax.set_xlabel("points n (log scale)", fontsize=8.5, color=palette.INK2)
    axs[0, 0].legend(fontsize=7.5, frameon=False, loc="best")
    fig.suptitle("R9b (post hoc) · S1: intrinsic dimension against n with the Theiler window "
                 "(median over 10 draws; S2 and S5 in the table)", fontsize=10.5, color=palette.INK)
    fig.tight_layout()
    fig.savefig(OUT / "step9b_theiler.png", dpi=200, bbox_inches="tight")
    log("figure saved")

    # markdown
    sn = {True: "sì", False: "no"}
    md = ["# Phase 2 · R9b · POST HOC: R9 with the Theiler window", "",
          f"Commit {results.code_version()[0]}. Requested after seeing R9; it does not change the R9 verdict (prediction P2 not confirmed) "
          "and has no verdict. Same inputs, draws and n as R9; DANCo excluded.", "",
          f"Check with window 0 against R9 ({len(m)} estimates): maximum relative error "
          + ", ".join(f"{s} {e:.1e}" for s, e in err_s.items())
          + f". CorrInt differs in {int((m[m.estimator == 'CorrInt'].err > 1e-6).sum())} estimates out of "
          f"{int((m.estimator == 'CorrInt').sum())} because of rounding ties at the radius r1 (a pair at a distance "
          "equal to the radius, counted or not depending on the last binary digit). The first version of the check (threshold "
          "1e-4 for all) stopped for this reason; the per-estimator threshold (1e-10, CorrInt 1e-3) was fixed afterwards "
          "(post hoc).", "",
          "## How much the estimate changes from n = 500 to all points (medians; S1 / S2 / S5)", "",
          "| Estimator | window | (a) single trial | (b) GPFA | (c) means |", "| --- | --- | --- | --- | --- |"]
    for stim in literature2.ESTIMATORS_THEILER:
        for ms in FIN:
            c = cr[(cr.estimator == stim) & (cr.window_ms == ms)].set_index("session").loc[list(literature.SESSIONS)]
            md.append(f"| {stim} | {ms} ms | " + " | ".join(
                " / ".join(f"{comma(a)} → {comma(b)}" for a, b in zip(c[f"{k}_n500"], c[f"{k}_all"]))
                for k in "abc") + " |")
    md += ["", "## R9 criteria per window (estimator × session cells out of 12; DANCo excluded)", "",
           "| window | (a) stable | (a) distinguishable from the twins | (b) stable | (c) stable | Altan prediction |",
           "| --- | --- | --- | --- | --- | --- |"]
    for ms in FIN:
        c = cr[cr.window_ms == ms]
        md.append(f"| {ms} ms | {int(c.a_stable.sum())} | {int(c.a_distinguishable.sum())} | {int(c.b_stable.sum())} | "
                  f"{int(c.c_stable.sum())} | {int(c.prediction_altan.sum())} |")
    md += ["", "## Values with all points, per window (S1 / S2 / S5)", "",
           "| Estimator | window | (a) data | twins [min-max] | (b) GPFA | (c) means | TME [min-max] |",
           "| --- | --- | --- | --- | --- | --- | --- |"]
    for stim in literature2.ESTIMATORS_THEILER:
        for ms in FIN:
            c = cr[(cr.estimator == stim) & (cr.window_ms == ms)].set_index("session").loc[list(literature.SESSIONS)]
            md.append(f"| {stim} | {ms} ms | {' / '.join(comma(x) for x in c.a_all)} | "
                      f"{' / '.join(f'{comma(a)}-{comma(b)}' for a, b in zip(c.twins_min, c.twins_max))} | "
                      f"{' / '.join(comma(x) for x in c.b_all)} | {' / '.join(comma(x) for x in c.c_all)} | "
                      f"{' / '.join(f'{comma(a)}-{comma(b)}' for a, b in zip(c.tme_min, c.tme_max))} |")
    (OUT / "step9b_theiler.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    log("end")
