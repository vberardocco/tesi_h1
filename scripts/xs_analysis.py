"""Analysis of the cross-session protocol (docs/crosssession_protocol.md): tables, tests, figures.

Reads the run registry (results/h1_runs.csv, metric r2) and averages over seeds per session (statistical unit = session).
- experiment "crosssession": the methods at the main dimension (PCA 25, FA 15, GPFA 25, Isomap/UMAP/CEBRA 25 from the second round (31 Oct),
  LFADS 25) plus the references ridge_channels and rates (ridge_rates); the rows of the d = 3 maps of 31 Oct with
  experiment "crosssession" are superseded by those of "crosssession_d3" and are excluded by filtering on the dimension;
- experiment "crosssession_d3": Isomap, UMAP, CEBRA at d = 3 (variant: table and between-method comparisons redone with them);
- experiment "crosssession_mmdiag": moment-matching diagnosis within the held-in sessions.
Writes to RESULTS/crosssession/ tables (csv, md, tex), tests, R²(λ) curves, trial coverage, diagnostics, figures and
report.md. Also works with partial results.

Usage: python scripts/xs_analysis.py
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from statsmodels.stats.multitest import multipletests  # noqa: E402

from thesis_h1 import crosssession as xs  # noqa: E402
from thesis_h1 import palette as pal  # noqa: E402
from thesis_h1 import results as res  # noqa: E402

OUT = res.RESULTS / "crosssession"
METHODS = list(xs.METHODS)
MAIN_DIM = {"PCA": 25, "FA": 15, "GPFA": 25, "Isomap": 25, "UMAP": 25, "CEBRA": 25, "LFADS": 25}
MAPS = ["Isomap", "UMAP", "CEBRA"]
REFS = ["ridge_channels", "rates"]
ROWS = METHODS + REFS
CONDS = ["zero_shot", "recenter", "align", "fa_native", "fewshot", "fewshot_gain", "scratch", "ridge_rates"]
N_SESS = 14
# colours and markers common to all thesis figures (src/tesi_h1/palette.py); text in neutral ink
CAT, COND_STYLE, METHOD_STYLE = pal.CAT, pal.COND_STYLE, pal.METHOD_STYLE
INK, INK2, GRID, RATES = pal.INK, pal.INK2, pal.GRID, pal.NEUTRAL


def chain(m):
    """Comparisons between consecutive conditions (31 Oct plan) + those added in the second round (fewshot_gain, ridge_channels)."""
    if m == "ridge_channels":
        return [("zero_shot", "recenter"), ("recenter", "fewshot"), ("recenter", "fewshot_gain")]
    c = [("zero_shot", "recenter"), ("recenter", "align"), ("align", "fewshot")]
    if m != "LFADS":
        c.append(("fewshot", "scratch"))
    if m == "FA":
        c.append(("align", "fa_native"))
    return c + [("align", "fewshot_gain")]


# --------------------------------------------------------------------------------------------
# Tables per session
# --------------------------------------------------------------------------------------------
def load_registry():
    df = res.load_runs()
    return df[(df.metric == "r2") & df.experiment.str.startswith("crosssession").fillna(False)].copy()


def _select(d, dims):
    keep = np.zeros(len(d), bool)
    for m, dim in dims.items():
        sel = (d.method == m).to_numpy()
        sel &= (d.latent_dim.isna() if dim is None else (d.latent_dim == dim).fillna(False)).to_numpy()
        keep |= sel
    return d[keep]


def per_session(df, experiment, dims):
    """Concatenated R², mean over seeds; dims = {method: d} (None = d not applicable, NA in the registry)."""
    d = _select(df[(df.experiment == experiment) & (df.fold == "concat")], dims)
    if d.empty:
        return pd.DataFrame(columns=["method", "condition", "session", "r2", "n_seeds", "r2_sd_seeds", "latent_dim",
                                     "day", "group", "days_from_held_in"])
    g = (d.groupby(["method", "condition", "session"], dropna=False)
         .agg(r2=("value", "mean"), n_seeds=("value", "size"), r2_sd_seeds=("value", "std"),
              latent_dim=("latent_dim", "first")).reset_index())
    g = res.add_days(g)
    return g.astype({"method": str, "condition": str, "session": str})


def per_fold(df, experiment, dims):
    d = _select(df[(df.experiment == experiment) & (df.fold != "concat")], dims)
    return (d.groupby(["method", "condition", "session", "fold"]).value.mean().rename("r2").reset_index()
            .astype({"method": str, "condition": str, "session": str, "fold": int}))


def label(m, S):
    dd = S[S.method == m].latent_dim.dropna()
    return f"{m} (d = {int(dd.iloc[0])})" if len(dd) else m


def table_conditions(S, name):
    rows = []
    for m in ROWS:
        for c in CONDS:
            v = S[(S.method == m) & (S.condition == c)]["r2"].to_numpy()
            if len(v):
                q1, med, q3 = np.percentile(v, [25, 50, 75])
                rows.append(dict(method=m, label=label(m, S), condition=c, median=med, q1=q1, q3=q3, mean=v.mean(),
                                 n=len(v), n_seeds=int(S[(S.method == m) & (S.condition == c)]["n_seeds"].max())))
    T = pd.DataFrame(rows)
    T.to_csv(OUT / f"{name}.csv", index=False)
    if T.empty:
        return T, "(vuoto)\n"
    cell = T.assign(cell=[f"{a:.3f} [{b:.3f}, {c:.3f}] ({n})" for a, b, c, n in zip(T["median"], T.q1, T.q3, T.n)])
    P = cell.pivot(index="label", columns="condition", values="cell")
    order = [label(m, S) for m in ROWS if label(m, S) in P.index]
    P = P.reindex(index=order, columns=[c for c in CONDS if c in P.columns]).fillna("--")
    md = "| method | " + " | ".join(P.columns) + " |\n|" + "---|" * (len(P.columns) + 1) + "\n"
    md += "\n".join(f"| {m} | " + " | ".join(P.loc[m]) + " |" for m in P.index) + "\n"
    PT = cell.assign(cell=[f"{a:.3f} [{b:.3f}, {c:.3f}]" for a, b, c in zip(T["median"], T.q1, T.q3)]) \
        .pivot(index="label", columns="condition", values="cell").reindex(index=P.index, columns=P.columns)
    PT.fillna("--").to_latex(OUT / f"{name}.tex", escape=True)
    (OUT / f"{name}.md").write_text(md)
    return T, md


# --------------------------------------------------------------------------------------------
# Tests
# --------------------------------------------------------------------------------------------
def wilcoxon_paired(a, b):
    """Two-sided Wilcoxon on b − a; effect size = paired rank-biserial correlation (zero differences excluded)."""
    d = np.asarray(b) - np.asarray(a)
    nz = d[d != 0]
    r = stats.rankdata(np.abs(nz))
    rpos, rneg = r[nz > 0].sum(), r[nz < 0].sum()
    w = stats.wilcoxon(a, b, alternative="two-sided")
    return dict(n=len(d), W=float(w.statistic), p=float(w.pvalue),
                r_rb=float((rpos - rneg) / (rpos + rneg)) if len(nz) else 0.0,
                median_diff=float(np.median(d)), mean_diff=float(np.mean(d)), n_better=int((d > 0).sum()))


def paired(S, m1, c1, m2, c2):
    a = S[(S.method == m1) & (S.condition == c1)].set_index("session")["r2"]
    b = S[(S.method == m2) & (S.condition == c2)].set_index("session")["r2"]
    k = a.index.intersection(b.index)
    return a[k].to_numpy(), b[k].to_numpy()


def stats_within(S, name):
    rows = []
    for m in METHODS + ["ridge_channels"]:
        fam = []
        for c1, c2 in chain(m):
            a, b = paired(S, m, c1, m, c2)
            if len(a) >= 6:
                fam.append(dict(method=m, from_=c1, to=c2, added_v2=(c2 == "fewshot_gain" or m == "ridge_channels"),
                                **wilcoxon_paired(a, b)))
        if fam:
            for f, p in zip(fam, multipletests([f["p"] for f in fam], method="holm")[1]):
                f["p_holm"] = float(p)
            rows += fam
    W = pd.DataFrame(rows).rename(columns={"from_": "from"})
    W.to_csv(OUT / f"{name}.csv", index=False)
    return W


def stats_between(S, name, conds=("align", "fewshot", "fewshot_gain")):
    fr, rows = [], []
    for c in conds:
        cand = METHODS + (["ridge_channels"] if c != "align" else [])
        complete = [m for m in cand if S[(S.method == m) & (S.condition == c)].session.nunique() == N_SESS]
        if len(complete) < 3:
            continue
        M = S[(S.condition == c) & S.method.isin(complete)].pivot(index="session", columns="method", values="r2")[complete]
        chi2, p = stats.friedmanchisquare(*[M[m].to_numpy() for m in complete])
        n, k = M.shape
        ranks = M.rank(axis=1).mean()
        fr.append(dict(condition=c, methods=",".join(complete), n=n, k=k, chi2=float(chi2), p=float(p),
                       kendall_W=float(chi2 / (n * (k - 1))), added_v2=(c == "fewshot_gain"),
                       **{f"meanrank_{m}": float(ranks[m]) for m in complete}))
        fam = []
        for i, m1 in enumerate(complete):
            for m2 in complete[i + 1:]:
                fam.append(dict(condition=c, method_a=m1, method_b=m2, **wilcoxon_paired(M[m1].to_numpy(), M[m2].to_numpy())))
        for f, p_ in zip(fam, multipletests([f["p"] for f in fam], method="holm")[1]):
            f["p_holm"] = float(p_)
        rows += fam
    F, B = pd.DataFrame(fr), pd.DataFrame(rows)
    F.to_csv(OUT / f"{name}_friedman.csv", index=False)
    B.to_csv(OUT / f"{name}_pairs.csv", index=False)
    return F, B


def zero_shot_trend(S):
    rows = []
    for m in ROWS:
        c = "ridge_rates" if m == "rates" else "zero_shot"
        d = S[(S.method == m) & (S.condition == c)]
        if len(d) >= 6:
            x, y = d.days_from_held_in.astype(float).to_numpy(), d.r2.to_numpy()
            lr = stats.linregress(x, y)
            t = stats.t.ppf(0.975, len(x) - 2)
            rho, prho = stats.spearmanr(x, y)
            rows.append(dict(method=m, condition=c, n=len(x), slope_per_day=lr.slope, ci_low=lr.slope - t * lr.stderr,
                             ci_high=lr.slope + t * lr.stderr, spearman_rho=rho, spearman_p=prho,
                             r2_first_day=float(d[d.days_from_held_in == d.days_from_held_in.min()].r2.mean()),
                             r2_last_day=float(d[d.days_from_held_in == d.days_from_held_in.max()].r2.mean())))
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "zero_shot_trend.csv", index=False)
    return T


# --------------------------------------------------------------------------------------------
# Pseudo-few-shot choices and R²(λ) curves
# --------------------------------------------------------------------------------------------
def tune_files():
    out = [(m, d, xs.tune_path(m, d)) for m, d in list(MAIN_DIM.items()) + [(m, 3) for m in MAPS]]
    out += [("ridge_channels", None, xs.tune_path("ridge_channels")), ("rates", None, xs.tune_path("rates"))]
    return [(m, d, p) for m, d, p in out if p.exists()]


def tuning_table():
    rows, curves = [], []
    for m, d, p in tune_files():
        t = json.loads(p.read_text())
        r = dict(method=m, dim=d, alpha_src=t.get("bundle", {}).get("alpha"), lodo_r2=t.get("bundle", {}).get("lodo_r2"))
        for c in ("zero_shot", "recenter", "align", "fa_native", "fewshot_gain", "fewshot", "scratch", "ridge_rates"):
            if c in t:
                r[f"pseudo_{c}"] = t[c]["mean_r2"]
        f = t.get("fewshot", {})
        r.update(lam=f.get("lam"), lam_interior=f.get("interior"), lam_at_edge=f.get("at_edge"),
                 asymptote=f.get("asymptote"), fewshot_minus_asymptote=f.get("gain_over_asymptote"))
        sc = t.get("scratch", t.get("ridge_rates", {}))
        r.update(alpha_scratch=sc.get("alpha"), alpha_at_edge=sc.get("at_edge"))
        rows.append(r)
        for lam, v in f.get("grid", {}).items():
            curves.append(dict(method=m, dim=d, lam=float(lam), mean_r2=v, asymptote_r2=f.get("asymptote_r2"),
                               chosen=bool(np.isclose(float(lam), float(f.get("lam")), rtol=1e-3))))   # JSON keys have 4 digits
    T, Cv = pd.DataFrame(rows), pd.DataFrame(curves)
    T.to_csv(OUT / "tuning.csv", index=False)
    Cv.to_csv(OUT / "lambda_curves.csv", index=False)
    return T, Cv


# --------------------------------------------------------------------------------------------
# Coverage of behaviour in the 2 adaptation trials
# --------------------------------------------------------------------------------------------
MIN_BINS_DIR = 10           # a direction is covered if at least 10 bins (200 ms) exceed the threshold


def coverage():
    data = xs.load_data()
    pool = [t for s in xs.heldin(data) for t in s.trials]
    V = np.concatenate([t.vel[t.mask] for t in pool])
    thr = np.percentile(np.abs(V), 75, axis=0)                  # per-DoF threshold: 75th percentile of |v| in the pool
    var_pool = V.var(0)

    def cov_of(trials):
        conds = {(ph, t.version) for t in trials for ph in t.phases}
        v = np.concatenate([t.vel[t.mask] for t in trials])
        dirs = {(j, sgn) for j in range(7) for sgn in (1, -1) if ((sgn * v[:, j]) > thr[j]).sum() >= MIN_BINS_DIR}
        ratio = v.var(0) / var_pool
        return dict(n_versions=len({t.version for t in trials}), n_phases=len({ph for t in trials for ph in t.phases}),
                    n_conditions=len(conds), n_directions=len(dirs),
                    var_ratio_median=float(np.median(ratio)), var_ratio_min=float(np.min(ratio)))

    rows = [dict(set="pool held-in", session="(13 sessions)", fold=None, **cov_of(pool))]
    for group, sessions in (("held-out", xs.heldout(data)), ("pseudo (S4, S5)", xs.pseudo_sessions(data))):
        for s in sessions:
            for k, ad in xs.folds(s):
                rows.append(dict(set=group, session=s.key, fold=k, test_version=s.trials[k].version,
                                 **cov_of([s.trials[j] for j in ad])))
    for s in xs.heldin(data):                                     # the diagnostic blocks within the session
        for j in range(len(s.trials) // 3):
            blk = list(range(3 * j, 3 * j + 3))
            for k in blk:
                rows.append(dict(set="held-in, diagnosis", session=s.key, fold=k, test_version=s.trials[k].version,
                                 **cov_of([s.trials[i] for i in blk if i != k])))
    C = pd.DataFrame(rows)
    C.to_csv(OUT / "coverage.csv", index=False)
    summ = C.groupby("set", sort=False).agg(
        folds=("n_conditions", "size"), versions_1=("n_versions", lambda x: int((x == 1).sum())),
        conditions_median=("n_conditions", "median"), conditions_min=("n_conditions", "min"),
        directions_median=("n_directions", "median"), directions_min=("n_directions", "min"),
        var_ratio_median=("var_ratio_median", "median")).reset_index()
    summ.to_csv(OUT / "coverage_summary.csv", index=False)
    return C, summ, thr


def align_by_version(F, C, which="held-out", conds=("recenter", "align")):
    """Per fold: Δ = R²(align) − R²(recenter) on the test trial, separating the folds whose 2 trials have the same version."""
    cv = C[C.set == which][["session", "fold", "n_versions"]].dropna().astype({"fold": int})
    rows = []
    for m in METHODS:
        a = F[(F.method == m) & (F.condition == conds[1])].set_index(["session", "fold"]).r2
        r = F[(F.method == m) & (F.condition == conds[0])].set_index(["session", "fold"]).r2
        d = (a - r).dropna().rename("delta").reset_index().merge(cv, on=["session", "fold"])
        for nv, g in d.groupby("n_versions"):
            rows.append(dict(method=m, where=which, n_versions=int(nv), folds=len(g), delta_median=float(g.delta.median()),
                             delta_mean=float(g.delta.mean()), n_better=int((g.delta > 0).sum())))
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------------------------
# Moment-matching diagnosis within the session
# --------------------------------------------------------------------------------------------
def mm_diag(df, S_main):
    D = per_session(df, "crosssession_mmdiag", MAIN_DIM)
    rows = []
    for m in METHODS:
        a, b = paired(D, m, "recenter", m, "align")
        if len(a) >= 6:
            w = wilcoxon_paired(a, b)
            a0, _ = paired(D, m, "no_adapt", m, "align")
            rows.append(dict(method=m, where="held-in, within session", n=w["n"], no_adapt=float(np.median(a0)),
                             recenter=float(np.median(a)), align=float(np.median(b)), delta_median=w["median_diff"],
                             n_better=w["n_better"], W=w["W"], p=w["p"], r_rb=w["r_rb"]))
        a, b = paired(S_main, m, "recenter", m, "align")
        if len(a) >= 6:
            w = wilcoxon_paired(a, b)
            rows.append(dict(method=m, where="held-out (6-20 days)", n=w["n"], recenter=float(np.median(a)),
                             align=float(np.median(b)), delta_median=w["median_diff"], n_better=w["n_better"], W=w["W"],
                             p=w["p"], r_rb=w["r_rb"]))
    T = pd.DataFrame(rows)
    if len(T):
        sub = T["where"] == "held-in, within session"
        if sub.any():
            T.loc[sub, "p_holm"] = multipletests(T.loc[sub, "p"], method="holm")[1]
    T.to_csv(OUT / "mm_diag.csv", index=False)
    D.to_csv(OUT / "mm_diag_sessions.csv", index=False)
    return T, D


# --------------------------------------------------------------------------------------------
# Adaptation bins
# --------------------------------------------------------------------------------------------
def adapt_bins():
    data = xs.load_data()
    n_ch = len(data["active"])
    A = pd.DataFrame([dict(session=s.key, **f) for s in xs.heldout(data) for f in xs.adapt_bins(s)])
    A.to_csv(OUT / "adapt_bins.csv", index=False)
    dims = dict(MAIN_DIM, **{"Isomap/UMAP/CEBRA (d = 3)": 3, "rates": n_ch, "ridge_channels": 176})
    summ = pd.DataFrame([dict(method=m, latent_dim=d, readout_weights_per_dof=xs.H * d + 1,
                              bins_mask_min=int(A.bins_mask.min()), bins_mask_median=float(A.bins_mask.median()),
                              bins_mask_max=int(A.bins_mask.max()), ratio_min=A.bins_mask.min() / (xs.H * d + 1))
                         for m, d in dims.items()])
    summ.to_csv(OUT / "adapt_bins_summary.csv", index=False)
    return A, summ, n_ch


# --------------------------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------------------------
_style = pal.style_axes


def _day_curve(d):
    g = d.groupby("days_from_held_in")["r2"].mean()
    return g.index.astype(float).to_numpy(), g.to_numpy()


def figures(S, Cv, MD):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    if S.empty:
        return
    rates = S[(S.method == "rates") & (S.condition == "ridge_rates")]
    shown = S[S.condition.isin(list(COND_STYLE) + ["ridge_rates"])].r2      # all points of figure A visible
    ylim = (min(-0.2, float(shown.min()) - 0.02), max(0.45, float(shown.max()) + 0.03))
    panels = METHODS + ["ridge_channels"]
    name = {"ridge_channels": "ridge on channels (05 decoder)"}
    # Figure A: one panel per method (+ ridge_channels), one curve per condition
    fig, axes = plt.subplots(3, 3, figsize=(13, 9.2), sharex=True, sharey=True)
    for ax, m in zip(axes.flat, panels):
        _style(ax)
        ax.axhline(0, color=INK2, lw=0.8)
        if len(rates):
            x, y = _day_curve(rates)
            ax.plot(x, y, color=RATES, lw=1.5, ls="--", zorder=1)
        d_m = S[S.method == m]
        for c, (col, mk) in COND_STYLE.items():
            d = d_m[d_m.condition == c]
            if d.empty:
                continue
            ax.scatter(d.days_from_held_in.astype(float), d.r2, s=10, color=col, alpha=0.35, lw=0, zorder=2)
            x, y = _day_curve(d)
            ax.plot(x, y, color=col, lw=2, marker=mk, ms=5, zorder=3)
        ns = d_m.groupby("condition").session.nunique()
        part = "" if len(ns) and ns.min() == N_SESS else " (partial)" if len(ns) else " (not run)"
        ax.set_title(name.get(m, label(m, S)) + part, fontsize=10, color=INK, loc="left")
    leg = axes.flat[-1]
    leg.axis("off")
    handles = [Line2D([], [], color=col, marker=mk, lw=2, ms=5, label=c) for c, (col, mk) in COND_STYLE.items()]
    handles.append(Line2D([], [], color=RATES, ls="--", lw=1.5, label="ridge_rates (2 trials)"))
    leg.legend(handles=handles, loc="center", frameon=False, fontsize=9, title="condition", title_fontsize=9)
    axes[0, 0].set_ylim(*ylim)
    for ax in axes.flat[:-1]:
        ax.tick_params(labelbottom=True)
    for ax in axes[-1]:
        ax.set_xlabel("days since last held-in session", fontsize=9, color=INK2)
    for ax in axes[:, 0]:
        ax.set_ylabel("R² (3 folds concatenated)", fontsize=9, color=INK2)
    fig.suptitle("Velocity decoding on the 14 held-out sessions: lines = mean of the 2 files of each day, dots = sessions; "
                 "seed-averaged", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "fig_A_conditions_by_days.png", dpi=160)
    plt.close(fig)
    # Figure B: fewshot, one curve per method (+ ridge_channels)
    fig, ax = plt.subplots(figsize=(7.8, 4.6))
    _style(ax)
    ax.axhline(0, color=INK2, lw=0.8)
    handles = []
    for m in panels:
        d = S[(S.method == m) & (S.condition == "fewshot")]
        if d.empty:
            continue
        col, mk = METHOD_STYLE[m]
        x, y = _day_curve(d)
        ax.plot(x, y, color=col, lw=2, marker=mk, ms=5)
        handles.append(Line2D([], [], color=col, marker=mk, lw=2, ms=5, label=name.get(m, label(m, S))))
    if len(rates):
        x, y = _day_curve(rates)
        ax.plot(x, y, color=RATES, lw=1.5, ls="--")
        handles.append(Line2D([], [], color=RATES, ls="--", lw=1.5, label="ridge_rates (2 trials)"))
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.set_xlabel("days since last held-in session", fontsize=9, color=INK2)
    ax.set_ylabel("R² (3 folds concatenated)", fontsize=9, color=INK2)
    ax.set_title("fewshot condition (for ridge on channels: recenter + refit): mean of the 2 files of each day",
                 fontsize=9.5, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_B_fewshot_by_days.png", dpi=160)
    plt.close(fig)
    # Figure C: R²(λ) curves on the held-in pseudo-sessions, relative to the asymptote (λ -> ∞)
    if len(Cv):
        fig, ax = plt.subplots(figsize=(7.8, 4.6))
        _style(ax)
        ax.axhline(0, color=INK2, lw=0.8)
        handles = []
        for m in panels:
            d = Cv[(Cv.method == m) & ((Cv.dim == MAIN_DIM.get(m)) | Cv.dim.isna())].sort_values("lam")
            if d.empty:
                continue
            col, mk = METHOD_STYLE[m]
            ax.plot(d.lam, d.mean_r2 - d.asymptote_r2, color=col, lw=2)
            ch = d[d.chosen]
            ax.scatter(ch.lam, ch.mean_r2 - ch.asymptote_r2, color=col, marker=mk, s=40, zorder=3)
            handles.append(Line2D([], [], color=col, marker=mk, lw=2, ms=5, label=name.get(m, m)))
        ax.set_xscale("log")
        top = float((Cv.mean_r2 - Cv.asymptote_r2).max())
        ax.set_ylim(-0.08, max(0.02, top + 0.005))
        ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
        ax.set_xlabel("λ (pull towards the held-in readout); y axis cut at −0.08: at small λ the curves go far below",
                      fontsize=9, color=INK2)
        ax.set_ylabel("R²(λ) − R²(λ → ∞)", fontsize=9, color=INK2)
        ax.set_title("fewshot on the 20 held-in pseudo-sessions, relative to align (ridge on channels: to recenter); "
                     "marker = chosen λ", fontsize=9, color=INK, loc="left")
        fig.tight_layout()
        fig.savefig(OUT / "fig_C_lambda_curves.png", dpi=160)
        plt.close(fig)
    # Figure D: Δ align − recenter within the session (held-in) and on the held-out sessions
    if len(MD):
        fig, ax = plt.subplots(figsize=(7.8, 4.2))
        _style(ax)
        ax.axhline(0, color=INK2, lw=0.8)
        for i, m in enumerate(METHODS):
            for j, (Sx, col) in enumerate(((MD, CAT[0]), (S, CAT[1]))):
                a, b = paired(Sx, m, "recenter", m, "align")
                if not len(a):
                    continue
                x = i + (j - 0.5) * 0.32
                ax.scatter(x + np.random.default_rng(i * 2 + j).uniform(-0.07, 0.07, len(a)), b - a,
                           s=12, color=col, alpha=0.5, lw=0)
                ax.plot([x - 0.11, x + 0.11], [np.median(b - a)] * 2, color=col, lw=2.5)
        ax.set_xticks(range(len(METHODS)), [label(m, S) for m in METHODS], fontsize=8, rotation=20)
        ax.legend(handles=[Line2D([], [], color=CAT[0], marker="o", lw=2.5, label="held-in, within session (no drift), n = 13"),
                           Line2D([], [], color=CAT[1], marker="o", lw=2.5, label="held-out, 6-20 days, n = 14")],
                  frameon=False, fontsize=8, loc="lower left")
        ax.set_ylabel("R²(align) − R²(recenter)", fontsize=9, color=INK2)
        ax.set_title("Moment matching estimated on 2 trials, without and with day-to-day drift (bar = median)",
                     fontsize=9.5, color=INK, loc="left")
        fig.tight_layout()
        fig.savefig(OUT / "fig_D_moment_matching_diagnostic.png", dpi=160)
        plt.close(fig)


def md_table(df, cols, floatfmt="{:.3g}"):
    if df is None or df.empty:
        return "(vuoto)\n"
    cols = [c for c in cols if c in df.columns]
    out = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    for _, r in df.iterrows():
        out += "| " + " | ".join(floatfmt.format(r[c]) if isinstance(r[c], (float, np.floating)) and not pd.isna(r[c])
                                 else str(r[c]) for c in cols) + " |\n"
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    df = load_registry()
    dims_main = dict(MAIN_DIM, ridge_channels=None, rates=None)
    S = per_session(df, "crosssession", dims_main)
    S3 = pd.concat([x for x in (per_session(df, "crosssession_d3", {m: 3 for m in MAPS}), S[~S.method.isin(MAPS)])
                    if len(x)], ignore_index=True)
    F = per_fold(df, "crosssession", dims_main)
    FD = per_fold(df, "crosssession_mmdiag", MAIN_DIM)
    S.to_csv(OUT / "sessions_r2.csv", index=False)
    S3.to_csv(OUT / "sessions_r2_d3.csv", index=False)
    T, md = table_conditions(S, "table_conditions")
    T3, md3 = table_conditions(S3[S3.method.isin(MAPS)], "table_conditions_d3")
    W = stats_within(S, "stats_within")
    W3 = stats_within(S3[S3.method.isin(MAPS)], "stats_within_d3")
    Fr, B = stats_between(S, "stats_between")
    Fr3, B3 = stats_between(S3, "stats_between_d3")
    Z = zero_shot_trend(S)
    A, summ, n_ch = adapt_bins()
    TU, Cv = tuning_table()
    C, CS, _ = coverage()
    AV = pd.concat([align_by_version(F, C, "held-out"), align_by_version(FD, C, "held-in, diagnosis")], ignore_index=True)
    AV.to_csv(OUT / "align_by_version.csv", index=False)
    MDt, MD = mm_diag(df, S)
    figures(S, Cv, MD)
    wcols = ["method", "from", "to", "n", "W", "p", "p_holm", "r_rb", "median_diff", "n_better"]
    bcols = ["condition", "method_a", "method_b", "n", "W", "p", "p_holm", "r_rb", "median_diff", "n_better"]
    fcols = ["condition", "methods", "n", "k", "chi2", "p", "kendall_W"]
    rep = ["# Cross-session protocol: results (generated by scripts/xs_analysis.py)\n\n",
           f"Registry: {res.RUNS}. R² concatenated over the 3 folds, mean over seeds; unit = session. Main dimensions: "
           + ", ".join(f"{m} {d}" for m, d in MAIN_DIM.items()) + ".\n\n",
           "## Median [IQR] (n) per method × condition\n\n", md,
           "\n## Maps at d = 3 (experiment crosssession_d3)\n\n", md3,
           "\n## Wilcoxon between consecutive conditions (Holm within each method)\n\n",
           "Differences = to − from; r_rb > 0 and n_better = sessions in which `to` is best. added_v2 = comparison "
           "added in the second round (not in the plan of the morning of 31 Oct).\n\n", md_table(W, wcols + ["added_v2"]),
           "\nMaps at d = 3:\n\n", md_table(W3, wcols),
           "\n## Friedman between methods\n\n", md_table(Fr, fcols), "\nWith the maps at d = 3:\n\n", md_table(Fr3, fcols),
           "\n## Pairwise Wilcoxon between methods (Holm within each condition)\n\n",
           "Differences = method_b − method_a; r_rb > 0 and n_better = sessions in which method_b is best.\n\n",
           md_table(B, bcols), "\nWith the maps at d = 3:\n\n", md_table(B3, bcols),
           "\n## Zero-shot against days (descriptive)\n\n",
           md_table(Z, ["method", "condition", "n", "slope_per_day", "ci_low", "ci_high", "spearman_rho", "spearman_p",
                        "r2_first_day", "r2_last_day"]),
           "\n## Pseudo-few-shot on the held-in sessions (frozen choices) and R²(λ) curves\n\n",
           "Each pseudo-session (S4, S5) is excluded from the estimation of method and readout, which use only S0-S3. lam_interior = "
           "optimum inside the grid (10^-2 - 10^15); fewshot_minus_asymptote = R²(chosen λ) − R²(λ -> ∞).\n\n",
           md_table(TU, ["method", "dim", "lodo_r2", "pseudo_zero_shot", "pseudo_recenter", "pseudo_align",
                         "pseudo_fewshot_gain", "pseudo_fewshot", "lam", "lam_interior", "fewshot_minus_asymptote",
                         "pseudo_scratch", "pseudo_ridge_rates", "alpha_scratch", "alpha_at_edge"], "{:.4g}"),
           f"\n## Bins of the 2 adaptation trials (active channels: {n_ch})\n\n",
           f"All bins: min {A.bins.min()}, median {A.bins.median():.0f}, max {A.bins.max()}; with eval_mask: min "
           f"{A.bins_mask.min()}, median {A.bins_mask.median():.0f}, max {A.bins_mask.max()} (42 folds).\n\n",
           md_table(summ, ["method", "latent_dim", "readout_weights_per_dof", "bins_mask_min", "ratio_min"]),
           "\n## Coverage of behaviour in the 2 adaptation trials\n\n",
           "Conditions = movement phase (8) × task version (odd/even: g3 has opposite sign in Grasp); directions = "
           f"DoF × sign (14) with at least {MIN_BINS_DIR} bins above the 75th percentile of |v| of the held-in pool; var_ratio = "
           "velocity variance in the 2 trials / variance in the pool (median over DoF). versions_1 = folds whose 2 trials "
           "have the same version.\n\n",
           md_table(CS, ["set", "folds", "versions_1", "conditions_median", "conditions_min", "directions_median",
                         "directions_min", "var_ratio_median"]),
           "\nΔ = R²(align) − R²(recenter) on the test trial, by number of versions in the 2 trials (exploratory):\n\n",
           md_table(AV, ["method", "where", "n_versions", "folds", "delta_median", "delta_mean", "n_better"]),
           "\n## Moment-matching diagnosis (exploratory, added in the second round)\n\n",
           "Within each held-in session: source = the other trials of the same session, blocks of 3 trials with the same "
           "rotation as the held-out sessions (no drift between days). Wilcoxon align vs recenter; Holm over the 7 methods (within "
           "the session only).\n\n",
           md_table(MDt, ["method", "where", "n", "no_adapt", "recenter", "align", "delta_median", "n_better", "W", "p",
                          "p_holm", "r_rb"]),
           "\n## Check on the 3-dimensional latents within a session (31 Oct, exploratory)\n\n",
           (open(OUT / "diag_3d_within_session.csv").read().replace("# ", "") if (OUT / "diag_3d_within_session.csv").exists()
            else "(not present)\n"),
           checks_section()]
    (OUT / "report.md").write_text("".join(rep))
    print(f"written {OUT}")


def checks_section():
    """Exploratory checks after the second round (scripts/xs_checks.py), if present: decomposition of fewshot_gain, gain of each
    trial, within-session decoding and fraction of latent variance explained by the session."""
    out = ["\n## Exploratory checks after the second round (scripts/xs_checks.py; seed 0, main dimensions)\n\n"]
    p = OUT / "check_gain_decomposition.csv"
    if p.exists():
        G = pd.read_csv(p)
        cols = ["base", "gain_offset", "gain_only", "offset_only", "oracle_test"]
        T = G.groupby("method", sort=False)[cols].median().reset_index()
        T["n"] = G.groupby("method", sort=False).size().values
        T["gain_offset_worse"] = G.assign(w=G.gain_offset < G.base).groupby("method", sort=False).w.sum().values
        out += ["**Where the drop of fewshot_gain comes from.** Median over the 14 held-out sessions of the concatenated R². base = align "
                "(ridge_channels: recenter); gain_offset = fewshot_gain; gain_only = per-DoF gain only (offset = "
                "held-in intercept); offset_only = offset only; oracle_test = gain and offset estimated on the test "
                "trial itself (upper bound, not a condition). gain_offset_worse = sessions in which fewshot_gain is "
                "below base.\n\n", md_table(T, ["method", "n"] + cols + ["gain_offset_worse"])]
        q = OUT / "check_gain_per_trial.csv"
        if q.exists():
            P = pd.read_csv(q).drop_duplicates(["method", "session", "trial", "dof"])
            V = (P.groupby(["method", "session", "dof"], sort=False).g.agg(["std", "mean"]).reset_index()
                 .groupby("method", sort=False).agg(g_mean_median=("mean", "median"), g_sd_between_trials_median=("std", "median"))
                 .reset_index())
            F = pd.read_csv(q).drop_duplicates(["method", "session", "fold_test_trial", "dof"])
            V = V.merge(F.groupby("method", sort=False).g_two_trials.agg(
                g2_p5=lambda x: np.percentile(x, 5), g2_median="median", g2_p95=lambda x: np.percentile(x, 95)).reset_index())
            out += ["\nPer-DoF gain estimated on one trial at a time: mean and standard deviation across the 3 trials of the "
                    "same session (medians over sessions and DoF); g2 = gain estimated on the 2 adaptation trials "
                    "(percentiles over folds and DoF).\n\n",
                    md_table(V, ["method", "g_mean_median", "g_sd_between_trials_median", "g2_p5", "g2_median", "g2_p95"])]
    p = OUT / "check_within_session.csv"
    if p.exists():
        W = pd.read_csv(p)
        T = W.groupby(["method", "dim"], sort=False).agg(
            n=("session", "size"), r2_within_median=("r2_within_session", "median"),
            r2_within_min=("r2_within_session", "min"), r2_within_max=("r2_within_session", "max"),
            alpha_at_edge=("alpha_at_edge", "sum")).reset_index()
        q = OUT / "check_session_variance.csv"
        if q.exists():
            T = T.merge(pd.read_csv(q)[["method", "eta2_session", "lodo_r2_heldin"]], on="method", how="left")
            ref = pd.read_csv(q)
            ref = ref[ref.method.str.startswith("velocity")]
        out += ["\n**Within-session decoding and between-session variance of the latents.** In each held-in session: "
                "readout fitted on all trials except the last 3 (leave-one-trial-out α, held-in readout grid), R² "
                "on the last 3. eta2_session = fraction of the variance of the standardized latents explained by the session "
                "(mean over dimensions); lodo_r2_heldin = held-in readout, leave-one-day-out. alpha_at_edge = sessions "
                "with α at the edge of the grid.\n\n",
                md_table(T, ["method", "dim", "n", "r2_within_median", "r2_within_min", "r2_within_max", "alpha_at_edge",
                             "eta2_session", "lodo_r2_heldin"])]
        if q.exists() and len(ref):
            out.append(f"\nReference: for velocity the same fraction is {ref.eta2_session.iloc[0]:.3f}.\n")
    return "".join(out) if len(out) > 1 else ""


if __name__ == "__main__":
    main()
