"""R9 · checks on intrinsic dimension (step 3), POST HOC ANALYSIS: does not change R9.

Rules written before the computations: docs/literature/rules_r9_checks.md (commit 067e8f5), with the post hoc amendment
(option 1) written before the curves: only the 0.167 level decides; the third level is the maximum reachable (0.58). Generator and readings in
src/thesis_h1/literature2.py; g calibrated by scripts/literature_r9c_calibrate.py (literature_cache/r9c/calibration.json).
Datasets: H1 input (a) in S1, S2, S5; the 5 R9 twins per session; the noiseless control (expected rates, at the
g of each level) and with noise (Poisson) at shared fractions 0.167 / 0.45 / 0.58 (maximum reachable); σ_lat 5 and 10 bins; 3 seeds.
Curves: thinning (m = 1, 2, 5, 10, 20, all offsets), random sampling (n as in R9, 10 draws),
Theiler (all points, w = 0-40 bins; group = trial for H1 and twins). Estimators TwoNN and MLE10.
Every finite estimate goes to literature_cache/r9c/estimates.jsonl: rerunning the script resumes from there.
Outputs: ~/thesis_h1_work/results/literature/r9_checks.csv, r9_checks.md, r9_curve.png, r9_checks_estimates.csv;
rows in the run registry (condition 9c_*).
Usage: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_THREADING_LAYER=workqueue PYTHONPATH=src python scripts/literature_r9c_curve.py
"""
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt                                                # noqa: E402
import numpy as np                                                             # noqa: E402
import pandas as pd                                                            # noqa: E402

from thesis_h1 import palette, results, literature, literature2                          # noqa: E402

OUT = results.RESULTS / "literature"
CACHE = results.WORK / "literature_cache"
F2, RC = CACHE / "phase2", CACHE / "r9c"
ESTIMATES = RC / "estimates.jsonl"
CAL = json.loads((RC / "calibration.json").read_text())
FREQ = F2 / "S1_freq_a.npy"
SESS = ("S1", "S2", "S5")
N_GEM, SEMI, SIGMI_LAT = 5, (0, 1, 2), (5, 10)
EMME = (1, 2, 5, 10, 20)
DOBLIU = (0, 1, 3, 5, 10, 20, 40)
STIM = ("TwoNN", "MLE10")
N_PROC = 6
KEYS = dict(experiment="literature", decoder=None, bins="falcon", split="session", fold=None, k_shot=None, H_ms=None)
T0 = time.time()


def log(msg):
    print(f"[{(time.time() - T0) / 60:6.1f} min] {msg}", flush=True)


def comma(x, nd=1):
    return "—" if x is None or not np.isfinite(x) else f"{x:.{nd}f}".replace(".", ",").replace("-", "−")


# --------------------------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------------------------
def datasets():
    """List of datasets: dicts with id, kind (h1, twin, without_noise, noise), session, level, σ_lat, seed."""
    out = []
    for s in SESS:
        out.append(dict(id=f"h1_{s}", kind="h1", session=s, share=None, sigma_lat=None, seed=None))
        for g in range(N_GEM):
            out.append(dict(id=f"gem_{s}_{g}", kind="twin", session=s, share=None, sigma_lat=None, seed=g))
    for sl in SIGMI_LAT:
        for q in literature2.QUOTE_CHECK:
            for seed in SEMI:
                for kind in ("without_noise", "noise"):
                    out.append(dict(id=f"{kind}_q{q}_s{sl}_{seed}", kind=kind, session=None, share=q, sigma_lat=sl,
                                    seed=seed))
    return out


def load(ds):
    """Return (X, group, time) of the dataset, in time order."""
    if ds["kind"] in ("h1", "twin"):
        f = F2 / (f"{ds['session']}_a.npy" if ds["kind"] == "h1" else f"{ds['session']}_d{ds['seed']}.npy")
        X = np.load(f)
        gt = np.load(F2 / f"{ds['session']}_meta_mov.npy")
        return X, gt[0], gt[1]
    c = CAL[f"{ds['share']}_{ds['sigma_lat']}"]
    gen = literature2.generate_check(np.load(FREQ), c["g"], ds["seed"], sigma_lat=ds["sigma_lat"])
    X = literature2.preprocessa_a(gen["rates"] if ds["kind"] == "without_noise" else gen["counts"])
    return X, np.zeros(len(X)), np.arange(len(X))


# --------------------------------------------------------------------------------------------
# Lavori
# --------------------------------------------------------------------------------------------
def jobs(ds):
    """One job = (curve, x): thinning m, sampling n, Theiler (all w together)."""
    n_tot = 5507 if ds["kind"] in ("without_noise", "noise") else len(np.load(F2 / f"{ds['session']}_a.npy",
                                                                            mmap_mode="r"))
    out = [("thinning", m) for m in EMME]
    out += [("random", n) for n in literature2.values_n(n_tot)]
    out += [("theiler", -1)]
    return out


def run(ds, curve, x):
    """Return a list of rows {id, curve, x, rep, estimator, estimate}."""
    X, gr, te = load(ds)
    rows = []
    if curve == "thinning":
        st = literature2.estimates_thinning(X, x, STIM)
        for s in STIM:
            rows += [dict(id=ds["id"], curve=curve, x=x, rip=o, estimator=s, estimate=float(v)) for o, v in enumerate(st[s])]
    elif curve == "random":
        semi = [0] if x == len(X) else range(10)
        for j in semi:
            Xs = X[literature2.draw(len(X), x, j)]
            for s in STIM:
                rows.append(dict(id=ds["id"], curve=curve, x=x, rip=j, estimator=s, estimate=literature2.estimate_id(Xs, s)))
    else:
        st = literature2.estimates_theiler(X, gr, te, list(DOBLIU))
        for w in DOBLIU:
            for s in STIM:
                rows.append(dict(id=ds["id"], curve=curve, x=w, rip=0, estimator=s, estimate=st[(w, s)]))
    return rows


def compute():
    already = set()
    if ESTIMATES.exists():
        for r in map(json.loads, ESTIMATES.read_text().splitlines()):
            already.add((r["id"], r["curve"], -1 if r["curve"] == "theiler" else r["x"]))   # Theiler: one job for all w
    todo = [(ds, c, x) for ds in datasets() for c, x in jobs(ds) if (ds["id"], c, x) not in already]
    log(f"jobs to do: {len(todo)} (already done: {len(already)})")
    with ProcessPoolExecutor(N_PROC) as ex, open(ESTIMATES, "a") as f:
        fut = {ex.submit(run, ds, c, x): (ds["id"], c, x) for ds, c, x in todo}
        for i, fu in enumerate(as_completed(fut), 1):
            for r in fu.result():
                f.write(json.dumps(r) + "\n")
            f.flush()
            if i % 50 == 0 or i == len(todo):
                log(f"{i}/{len(todo)}")


# --------------------------------------------------------------------------------------------
# Summary, readings, figure
# --------------------------------------------------------------------------------------------
def summary():
    st = pd.DataFrame([json.loads(l) for l in ESTIMATES.read_text().splitlines()])
    meta = pd.DataFrame(datasets())
    st = st.merge(meta, on="id")
    st.to_csv(OUT / "r9_checks_estimates.csv", index=False)
    # medians: for H1 and twins per dataset; for the controls over repetitions and seeds together
    st["group"] = np.where(st.kind.isin(["h1", "twin"]), st.id, st.id.str.rsplit("_", n=1).str[0])   # without the seed
    rias = (st.groupby(["group", "kind", "session", "share", "sigma_lat", "curve", "x", "estimator"], dropna=False)
            .estimate.agg(median="median", minimum="min", max="max", n="size").reset_index())
    rias.to_csv(OUT / "r9_checks.csv", index=False)
    return st, rias


def M(rias, group, stim, m):
    r = rias[(rias.group == group) & (rias.estimator == stim) & (rias.curve == "thinning") & (rias.x == m)]
    return float(r.median.iloc[0])


def readings(rias):
    L, tab = {}, []
    for stim in STIM:
        def lit(gr):
            m10, m20 = M(rias, gr, stim, 10), M(rias, gr, stim, 20)
            return dict(M10=m10, M20=m20, **literature2.reading_thinning(m10, m20, 6))
        sens = {(q, sl): lit(f"without_noise_q{q}_s{sl}") for sl in SIGMI_LAT for q in literature2.QUOTE_CHECK}
        ctrl = {(q, sl): lit(f"noise_q{q}_s{sl}") for sl in SIGMI_LAT for q in literature2.QUOTE_CHECK}
        h1 = {s: lit(f"h1_{s}") for s in SESS}
        sens_ok = all(sens[(0.167, sl)]["stable"] and sens[(0.167, sl)]["around"] for sl in SIGMI_LAT)
        c167 = [ctrl[(0.167, sl)] for sl in SIGMI_LAT]
        ctrl_ok = all(v["stable"] and v["around"] for v in c167)
        h1_ok = any(v["stable"] and v["around"] for v in h1.values())
        sentence = "A" if ctrl_ok and not h1_ok else ("B" if not ctrl_ok else "other")
        L[stim] = dict(sensitivity=sens_ok, sentence=sentence, sens=sens, ctrl=ctrl, h1=h1)
        for k, v in sens.items():
            tab.append(dict(estimator=stim, dataset=f"senza rumore, g di q {k[0]}, σ_lat {k[1]}", **v))
        for k, v in ctrl.items():
            tab.append(dict(estimator=stim, dataset=f"rumore, quota {k[0]}, σ_lat {k[1]}", **v))
        for k, v in h1.items():
            tab.append(dict(estimator=stim, dataset=f"H1 {k}", **v))
    return L, pd.DataFrame(tab)


def figure(rias):
    CUR = (("thinning", "one bin every m"), ("random", "drawn points n"), ("theiler", "Theiler window w (bins)"))
    fig, axs = plt.subplots(2, 3, figsize=(14, 8.5), sharey="row")
    col_s = dict(zip(SESS, (palette.CAT[0], palette.CAT[1], palette.CAT[2])))
    col_q = dict(zip(literature2.QUOTE_CHECK, (palette.CAT[3], palette.CAT[4], palette.CAT[6])))
    for i, stim in enumerate(STIM):
        for j, (cur, xl) in enumerate(CUR):
            ax = axs[i, j]
            r = rias[(rias.estimator == stim) & (rias.curve == cur)]
            gem = r[r.kind == "twin"].groupby("x").median.agg(["min", "max"])
            xs = lambda x: np.where(np.asarray(x) == 0, 0.5, x) if cur == "theiler" else np.asarray(x)   # noqa: E731
            ax.fill_between(xs(gem.index), gem["min"], gem["max"], color=palette.GRID, label="twins (3 sessions)")
            for s in SESS:
                g = r[r.group == f"h1_{s}"].sort_values("x")
                ax.plot(xs(g.x), g.median, "o-", color=col_s[s], lw=1.8, ms=4, label=f"H1 {s}")
            for sl, ls in zip(SIGMI_LAT, ("-", ":")):
                g = r[r.group == f"without_noise_q0.167_s{sl}"].sort_values("x")
                ax.plot(xs(g.x), g.median, ls, color=palette.INK, lw=1.2, label=f"noiseless (g of 0.167), σ_lat {sl}")
                for q in literature2.QUOTE_CHECK:
                    g = r[r.group == f"noise_q{q}_s{sl}"].sort_values("x")
                    ax.plot(xs(g.x), g.median, ls, color=col_q[q], lw=1.4,
                            label=f"control share {comma(q, 3 if q == 0.167 else 2)}, σ_lat {sl}")
            ax.axhline(6, color=palette.INK2, ls="--", lw=1, label="d = 6")
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xlabel(xl + (" (0 at left)" if cur == "theiler" else ""), fontsize=9, color=palette.INK)
            if j == 0:
                ax.set_ylabel(f"{stim} (mediana)", fontsize=9, color=palette.INK)
            palette.style_axes(ax)
    h, lab = axs[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=4, fontsize=8, frameon=False)
    fig.suptitle("R9 · POST HOC checks (do not change R9): intrinsic dimension estimated on H1 (a), twins and "
                 "generator with d = 6", fontsize=10.5, color=palette.INK)
    fig.tight_layout(rect=(0, 0.1, 1, 0.96))
    fig.savefig(OUT / "r9_curve.png", dpi=200, bbox_inches="tight")


def main():
    compute()
    st, rias = summary()
    L, tab = readings(rias)
    figure(rias)
    log("figure saved")
    # H1 against twins, for each point of the curves
    hg = []
    for s in SESS:
        for (cur, x, stim), g in rias[rias.group == f"h1_{s}"].groupby(["curve", "x", "estimator"]):
            gm = rias[(rias.kind == "twin") & (rias.session == s) & (rias.curve == cur) & (rias.x == x)
                      & (rias.estimator == stim)].median
            v = float(g.median.iloc[0])
            hg.append(dict(session=s, curve=cur, x=x, estimator=stim, h1=v, gem_min=gm.min(), gem_max=gm.max(),
                           within=bool(gm.min() <= v <= gm.max())))
    hg = pd.DataFrame(hg)
    hg.to_csv(OUT / "r9_checks_h1_twins.csv", index=False)
    sentences = {s: L[s]["sentence"] for s in STIM}
    sentence = sentences["TwoNN"] if len(set(sentences.values())) == 1 else "per estimator: " + ", ".join(
        f"{s} {f}" for s, f in sentences.items())
    sens_ok = all(L[s]["sensitivity"] for s in STIM)
    # run registry
    for _, r in tab.iterrows():
        results.log_run(dict(M10=r.M10, M20=r.M20, stable=float(r.stable), around_6=float(r.around)),
                        method=r.estimator, latent_dim=None, session=None, condition=f"9c_{r.dataset}", seed=None,
                        **KEYS)
    results.log_run({"sensitivity_ok": float(sens_ok), "sentence_A": float(sentence == "A"), "sentence_B": float(sentence == "B"),
                     "h1_within_twins_share": float(hg.within.mean()), "minutes": (time.time() - T0) / 60},
                    method="TwoNN+MLE10", latent_dim=None, session=None, condition="9c_verdict", seed=None, **KEYS)
    # md
    cal = "; ".join(f"share {comma(c['share'], 3)}, σ_lat {c['sigma_lat']}: g = {c['g']:.3f}, FA {comma(c['share_fa'], 3)}"
                    for c in CAL.values())
    fr = np.load(FREQ)
    qv = {k: np.median([literature2.share_true(literature2.generate_check(fr, c["g"], sm, sigma_lat=c["sigma_lat"]))
                        for sm in SEMI]) for k, c in CAL.items()}
    md = ["# R9 · Checks on intrinsic dimension — POST HOC ANALYSIS", "",
          f"Commit {results.code_version()[0]}. Rules: docs/literature/rules_r9_checks.md (commit 067e8f5, before the "
          "computations; post hoc amendment \"option 1\" before the curves: only the 0.167 level decides, the third level is "
          "the maximum reachable 0.58 instead of 0.65, T2 on the medians over seeds). It does not change R9: prediction P2, which we derived from Altan 2021, is not confirmed in any "
          "of the 15 cells (5 estimators × 3 sessions).", "",
          f"Calibration (seed 0): {cal}.", "",
          "True shared fraction of the generator (var of the smoothed expected rate / var of the smoothed counts, median over channels, "
          "median over the 3 seeds): " + "; ".join(f"share {comma(CAL[k]['share'], 3)}, σ_lat {CAL[k]['sigma_lat']}: "
                                              f"{comma(v, 3)}" for k, v in qv.items()) + ".", "",
          f"**Sensitivity** (noiseless control at the g of the 0.167 level, stable and around 6 for TwoNN and MLE10, "
          "both σ_lat): "
          f"{'sì' if sens_ok else 'NO'}.", "",
          f"**Sentence**: {sentence}.", "",
          "## Readings (thinning, medians at m = 10 and m = 20)", "",
          "| estimator | dataset | M10 | M20 | stable | around 6 |", "|---|---|---|---|---|---|"]
    for _, r in tab.iterrows():
        md.append(f"| {r.estimator} | {r.dataset} | {comma(r.M10, 2)} | {comma(r.M20, 2)} | "
                  f"{'sì' if r.stable else 'no'} | {'sì' if r.around else 'no'} |")
    md += ["", "## Curves (median [min, max])", ""]
    for cur in ("thinning", "random", "theiler"):
        md += [f"### {cur}", "", "| estimator | dataset | " + " | ".join(
            str(x) for x in sorted(rias[rias.curve == cur].x.unique())) + " |", "|---|---|" + "---|" * rias[
            rias.curve == cur].x.nunique()]
        for (stim, gr), g in rias[rias.curve == cur].groupby(["estimator", "group"]):
            if gr.startswith("gem_"):
                continue
            g = g.set_index("x").sort_index()
            md.append(f"| {stim} | {gr} | " + " | ".join(
                f"{comma(r.median)} [{comma(r.minimum)}, {comma(r.max)}]" for _, r in g.iterrows()) + " |")
        md.append("")
    md += ["## H1 against twins (median of H1 within [min, max] of the 5 twins)", "",
           f"Within at {int(hg.within.sum())} of {len(hg)} points. Details: r9_checks_h1_twins.csv.", ""]
    for (cur, stim), g in hg.groupby(["curve", "estimator"]):
        md.append(f"- {stim}, {cur}: " + "; ".join(
            f"{s} " + ", ".join(f"{x}:{'d' if d else 'F'}" for x, d in zip(gg.x, gg.within)) for s, gg in g.groupby("session"))
            + " (d = within the twin range, F = outside it)")
    md += ["", f"Total time of this run: {(time.time() - T0) / 60:.1f} min. Outputs: r9_checks.csv, "
           "r9_checks_estimates.csv, r9_checks_h1_twins.csv, r9_curve.png."]
    (OUT / "r9_checks.md").write_text("\n".join(md) + "\n")
    log(f"sensitivity {'ok' if sens_ok else 'NO'}; sentence {sentence}")


if __name__ == "__main__":
    main()
