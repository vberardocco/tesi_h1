"""Analisi del protocollo cross-sessione (docs/crosssession_protocol.md): tabelle, test, figure.

Legge il registro (results/h1_runs.csv, esperimento "crosssession", metrica r2, fold "concat"), fa la media sui semi
per sessione (unità statistica = sessione, n = 14) e scrive in RESULTS/crosssession/:
- table_conditions.{csv,md,tex}: mediana [IQR] dell'R² per metodo × condizione, con n;
- stats_within.csv: Wilcoxon appaiato tra condizioni consecutive dentro ogni metodo, Holm per metodo;
- stats_friedman.csv, stats_between.csv: Friedman tra metodi ad align e fewshot, poi Wilcoxon a coppie con Holm;
- zero_shot_trend.csv: pendenza dell'R² zero-shot contro i giorni (descrittivo);
- adapt_bins.csv: bin dei 2 trial di adattamento rispetto a canali, dimensione latente e pesi del readout;
- tuning.csv: scelte dello pseudo-few-shot (λ, α) e R² medi sulle pseudo-sessioni;
- fig_A_conditions_by_days.png, fig_B_fewshot_by_days.png;
- report.md: tutto quanto sopra in un file da leggere.
Funziona anche con risultati parziali (metodi mancanti o incompleti).

Uso: python scripts/xs_analysis.py
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

from tesi_h1 import crosssession as xs  # noqa: E402
from tesi_h1 import results as res  # noqa: E402

OUT = res.RESULTS / "crosssession"
METHODS = list(xs.METHODS)
ROWS = METHODS + ["rates"]
CONDS = ["zero_shot", "recenter", "align", "fa_native", "fewshot", "scratch", "ridge_rates"]
CHAIN = [("zero_shot", "recenter"), ("recenter", "align"), ("align", "fewshot"), ("fewshot", "scratch")]
N_SESS = 14
# tavolozza categoriale di riferimento (ordine fisso), testo in inchiostro neutro
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
COND_STYLE = {"zero_shot": (CAT[0], "o"), "recenter": (CAT[1], "s"), "align": (CAT[2], "^"),
              "fewshot": (CAT[3], "D"), "scratch": (CAT[4], "v"), "fa_native": (CAT[6], "P")}
METHOD_STYLE = {m: (CAT[i], mk) for i, (m, mk) in enumerate(zip(METHODS, "osD^vPX"))}
INK, INK2, GRID, RATES = "#0b0b0b", "#52514e", "#e4e3df", "#7a7975"


def session_table():
    df = res.load_runs()
    df = df[(df.esperimento == xs.EXPERIMENT) & (df.metrica == "r2") & (df.fold == "concat")]
    if df.empty:
        return pd.DataFrame(columns=["method", "condition", "session", "r2", "n_seeds", "giorni_da_held_in"])
    g = (df.groupby(["method", "condition", "session"], dropna=False)["valore"]
         .agg(r2="mean", n_seeds="size", r2_sd_seeds="std").reset_index())
    g = res.add_days(g)
    return g.astype({"method": str, "condition": str, "session": str})


def fmt3(x):
    return "--" if pd.isna(x) else f"{x:.3f}"


def table_conditions(S):
    rows = []
    for m in ROWS:
        for c in CONDS:
            v = S[(S.method == m) & (S.condition == c)]["r2"].to_numpy()
            if len(v):
                q1, med, q3 = np.percentile(v, [25, 50, 75])
                rows.append(dict(method=m, condition=c, median=med, q1=q1, q3=q3, mean=v.mean(), n=len(v),
                                 n_seeds=int(S[(S.method == m) & (S.condition == c)]["n_seeds"].max())))
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "table_conditions.csv", index=False)
    if T.empty:
        return T, ""
    cell = T.assign(cell=[f"{a:.3f} [{b:.3f}, {c:.3f}] ({n})" for a, b, c, n in zip(T["median"], T.q1, T.q3, T.n)])
    P = cell.pivot(index="method", columns="condition", values="cell")
    P = P.reindex(index=[m for m in ROWS if m in P.index], columns=[c for c in CONDS if c in P.columns]).fillna("--")
    md = "| method | " + " | ".join(P.columns) + " |\n|" + "---|" * (len(P.columns) + 1) + "\n"
    md += "\n".join(f"| {m} | " + " | ".join(P.loc[m]) + " |" for m in P.index)
    tex_cell = cell.assign(cell=[f"{a:.3f} [{b:.3f}, {c:.3f}]" for a, b, c in zip(T["median"], T.q1, T.q3)])
    PT = tex_cell.pivot(index="method", columns="condition", values="cell").reindex(index=P.index, columns=P.columns)
    PT.fillna("--").to_latex(OUT / "table_conditions.tex", escape=True)
    (OUT / "table_conditions.md").write_text(md + "\n")
    return T, md


def wilcoxon_paired(a, b):
    """Wilcoxon a due code su b − a; effect size = correlazione rank-biserial appaiata (differenze nulle escluse)."""
    d = np.asarray(b) - np.asarray(a)
    nz = d[d != 0]
    r = stats.rankdata(np.abs(nz))
    rpos, rneg = r[nz > 0].sum(), r[nz < 0].sum()
    w = stats.wilcoxon(a, b, alternative="two-sided")
    return dict(n=len(d), W=float(w.statistic), p=float(w.pvalue), r_rb=float((rpos - rneg) / (rpos + rneg)) if len(nz) else 0.0,
                median_diff=float(np.median(d)), mean_diff=float(np.mean(d)), n_better=int((d > 0).sum()))


def paired(S, m1, c1, m2, c2):
    a = S[(S.method == m1) & (S.condition == c1)].set_index("session")["r2"]
    b = S[(S.method == m2) & (S.condition == c2)].set_index("session")["r2"]
    k = a.index.intersection(b.index)
    return a[k].to_numpy(), b[k].to_numpy()


def stats_within(S):
    rows = []
    for m in METHODS:
        pairs = [p for p in CHAIN if not (m == "LFADS" and "scratch" in p)] + ([("align", "fa_native")] if m == "FA" else [])
        fam = []
        for c1, c2 in pairs:
            a, b = paired(S, m, c1, m, c2)
            if len(a) >= 6:
                fam.append(dict(method=m, from_=c1, to=c2, **wilcoxon_paired(a, b)))
        if fam:
            ph = multipletests([f["p"] for f in fam], method="holm")[1]
            for f, p in zip(fam, ph):
                f["p_holm"] = float(p)
            rows += fam
    W = pd.DataFrame(rows).rename(columns={"from_": "from"})
    W.to_csv(OUT / "stats_within.csv", index=False)
    return W


def stats_between(S):
    fr, rows = [], []
    for c in ("align", "fewshot"):
        complete = [m for m in METHODS if (S[(S.method == m) & (S.condition == c)].session.nunique() == N_SESS)]
        if len(complete) < 3:
            continue
        M = S[(S.condition == c) & S.method.isin(complete)].pivot(index="session", columns="method", values="r2")[complete]
        chi2, p = stats.friedmanchisquare(*[M[m].to_numpy() for m in complete])
        n, k = M.shape
        ranks = M.rank(axis=1).mean()
        fr.append(dict(condition=c, methods=",".join(complete), n=n, k=k, chi2=float(chi2), p=float(p),
                       kendall_W=float(chi2 / (n * (k - 1))), **{f"meanrank_{m}": float(ranks[m]) for m in complete}))
        fam = []
        for i, m1 in enumerate(complete):
            for m2 in complete[i + 1:]:
                fam.append(dict(condition=c, method_a=m1, method_b=m2,
                                **wilcoxon_paired(M[m1].to_numpy(), M[m2].to_numpy())))
        ph = multipletests([f["p"] for f in fam], method="holm")[1]
        for f, p_ in zip(fam, ph):
            f["p_holm"] = float(p_)
        rows += fam
    F, B = pd.DataFrame(fr), pd.DataFrame(rows)
    F.to_csv(OUT / "stats_friedman.csv", index=False)
    B.to_csv(OUT / "stats_between.csv", index=False)
    return F, B


def zero_shot_trend(S):
    rows = []
    for m in ROWS:
        for c in (["zero_shot"] if m != "rates" else ["ridge_rates"]):
            d = S[(S.method == m) & (S.condition == c)]
            if len(d) >= 6:
                x, y = d.giorni_da_held_in.astype(float).to_numpy(), d.r2.to_numpy()
                lr = stats.linregress(x, y)
                t = stats.t.ppf(0.975, len(x) - 2)
                rho, prho = stats.spearmanr(x, y)
                rows.append(dict(method=m, condition=c, n=len(x), slope_per_day=lr.slope, ci_low=lr.slope - t * lr.stderr,
                                 ci_high=lr.slope + t * lr.stderr, intercept=lr.intercept, spearman_rho=rho, spearman_p=prho,
                                 r2_first_day=float(d[d.giorni_da_held_in == d.giorni_da_held_in.min()].r2.mean()),
                                 r2_last_day=float(d[d.giorni_da_held_in == d.giorni_da_held_in.max()].r2.mean())))
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "zero_shot_trend.csv", index=False)
    return T


def adapt_bins():
    data = xs.load_data()
    n_ch = len(data["active"])
    rows = [dict(session=s.key, **f) for s in xs.heldout(data) for f in xs.adapt_bins(s)]
    A = pd.DataFrame(rows)
    A.to_csv(OUT / "adapt_bins.csv", index=False)
    dims = {"PCA": 25, "FA": 15, "GPFA": 25, "Isomap": 3, "UMAP": 3, "CEBRA": 3, "LFADS": 25, "rates": n_ch}
    summ = pd.DataFrame([dict(method=m, latent_dim=d, readout_weights_per_dof=xs.H * d + 1,
                              bins_mask_min=int(A.bins_mask.min()), bins_mask_median=float(A.bins_mask.median()),
                              bins_mask_max=int(A.bins_mask.max()),
                              ratio_min=A.bins_mask.min() / (xs.H * d + 1), bins_per_channel_min=A.bins.min() / n_ch)
                         for m, d in dims.items()])
    summ.to_csv(OUT / "adapt_bins_summary.csv", index=False)
    return A, summ, n_ch


def tuning_table():
    rows = []
    for m in ROWS:
        p = xs.tune_path(m)
        if not p.exists():
            continue
        t = json.loads(p.read_text())
        r = dict(method=m, alpha_src=t.get("bundle", {}).get("alpha"), lodo_r2=t.get("bundle", {}).get("lodo_r2"))
        for c in ("zero_shot", "recenter", "align", "fa_native", "fewshot", "scratch", "ridge_rates"):
            if c in t:
                r[f"pseudo_{c}"] = t[c]["mean_r2"]
        r["lam"] = t.get("fewshot", {}).get("lam")
        r["lam_at_edge"] = t.get("fewshot", {}).get("at_edge")
        r["alpha_scratch"] = t.get("scratch", t.get("ridge_rates", {})).get("alpha")
        r["alpha_at_edge"] = t.get("scratch", t.get("ridge_rates", {})).get("at_edge")
        hp = xs.XS / "models" / f"{m}_heldin_s0.json"
        if hp.exists():
            h = json.loads(hp.read_text())
            r["heldin_alpha_src"], r["heldin_lodo_r2"] = h.get("alpha"), h.get("lodo_r2")
        rows.append(r)
    T = pd.DataFrame(rows)
    T.to_csv(OUT / "tuning.csv", index=False)
    return T


# --------------------------------------------------------------------------------------------
# Figure
# --------------------------------------------------------------------------------------------
def _style(ax):
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)


def _day_curve(d):
    g = d.groupby("giorni_da_held_in")["r2"].mean()
    return g.index.astype(float).to_numpy(), g.to_numpy()


def figures(S):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    if S.empty:
        return
    rates = S[(S.method == "rates") & (S.condition == "ridge_rates")]
    ylim = (min(-0.2, np.nanpercentile(S.r2, 1) - 0.02), max(0.45, S.r2.max() + 0.03))
    # Figura A
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.2), sharex=True, sharey=True)
    for ax, m in zip(axes.flat, METHODS):
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
            ax.scatter(d.giorni_da_held_in.astype(float), d.r2, s=10, color=col, alpha=0.35, lw=0, zorder=2)
            x, y = _day_curve(d)
            ax.plot(x, y, color=col, lw=2, marker=mk, ms=5, zorder=3)
        ns = d_m.groupby("condition").session.nunique()
        part = "" if len(ns) and ns.min() == N_SESS else " (partial)" if len(ns) else " (not run)"
        dim = {"PCA": 25, "FA": 15, "GPFA": 25, "Isomap": 3, "UMAP": 3, "CEBRA": 3, "LFADS": 25}[m]
        ax.set_title(f"{m} (d = {dim}){part}", fontsize=10, color=INK, loc="left")
    leg = axes.flat[-1]
    leg.axis("off")
    handles = [Line2D([], [], color=col, marker=mk, lw=2, ms=5, label=c) for c, (col, mk) in COND_STYLE.items()]
    handles.append(Line2D([], [], color=RATES, ls="--", lw=1.5, label="ridge_rates (2 trials)"))
    leg.legend(handles=handles, loc="center", frameon=False, fontsize=9, title="condition", title_fontsize=9)
    axes[0, 0].set_ylim(*ylim)
    for ax in axes[1]:
        ax.set_xlabel("days since last held-in session", fontsize=9, color=INK2)
    for ax in axes[:, 0]:
        ax.set_ylabel("R² (3 folds concatenated)", fontsize=9, color=INK2)
    fig.suptitle("Velocity decoding on the 14 held-out sessions: lines = mean of the 2 files of each day, dots = sessions; "
                 "seed-averaged", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(OUT / "fig_A_conditions_by_days.png", dpi=160)
    plt.close(fig)
    # Figura B
    fig, ax = plt.subplots(figsize=(7.5, 4.4))
    _style(ax)
    ax.axhline(0, color=INK2, lw=0.8)
    handles = []
    for m in METHODS:
        d = S[(S.method == m) & (S.condition == "fewshot")]
        if d.empty:
            continue
        col, mk = METHOD_STYLE[m]
        x, y = _day_curve(d)
        ax.plot(x, y, color=col, lw=2, marker=mk, ms=5)
        handles.append(Line2D([], [], color=col, marker=mk, lw=2, ms=5, label=m))
    if len(rates):
        x, y = _day_curve(rates)
        ax.plot(x, y, color=RATES, lw=1.5, ls="--")
        handles.append(Line2D([], [], color=RATES, ls="--", lw=1.5, label="ridge_rates"))
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.set_xlabel("days since last held-in session", fontsize=9, color=INK2)
    ax.set_ylabel("R² (3 folds concatenated)", fontsize=9, color=INK2)
    ax.set_title("fewshot condition (align + readout refit on 2 trials): mean of the 2 files of each day",
                 fontsize=9.5, color=INK, loc="left")
    fig.tight_layout()
    fig.savefig(OUT / "fig_B_fewshot_by_days.png", dpi=160)
    plt.close(fig)


def md_table(df, cols, floatfmt="{:.3g}"):
    if df.empty:
        return "(vuoto)\n"
    out = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    for _, r in df.iterrows():
        out += "| " + " | ".join(floatfmt.format(r[c]) if isinstance(r[c], (float, np.floating)) else str(r[c]) for c in cols) + " |\n"
    return out


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    S = session_table()
    S.to_csv(OUT / "sessions_r2.csv", index=False)
    T, md = table_conditions(S)
    W = stats_within(S)
    F, B = stats_between(S)
    Z = zero_shot_trend(S)
    A, summ, n_ch = adapt_bins()
    TU = tuning_table()
    figures(S)
    rep = ["# Protocollo cross-sessione: risultati (generato da scripts/xs_analysis.py)\n",
           f"Registro: {res.RUNS}. R² concatenato sui 3 fold, media sui semi; unità = sessione.\n",
           "## Mediana [IQR] (n) per metodo × condizione\n", md + "\n",
           "## Wilcoxon tra condizioni consecutive (Holm dentro ogni metodo)\n",
           "Differenze = to − from; r_rb > 0 e n_better = sessioni in cui `to` è migliore.\n\n",
           md_table(W, ["method", "from", "to", "n", "W", "p", "p_holm", "r_rb", "median_diff", "n_better"]) if len(W) else "(vuoto)\n",
           "\n## Friedman tra metodi\n",
           md_table(F, ["condition", "methods", "n", "k", "chi2", "p", "kendall_W"]) if len(F) else "(vuoto)\n",
           "\n## Wilcoxon tra metodi a coppie (Holm dentro ogni condizione)\n",
           "Differenze = method_b − method_a; r_rb > 0 e n_better = sessioni in cui method_b è migliore.\n\n",
           md_table(B, ["condition", "method_a", "method_b", "n", "W", "p", "p_holm", "r_rb", "median_diff", "n_better"]) if len(B) else "(vuoto)\n",
           "\n## Zero-shot contro i giorni (descrittivo)\n",
           md_table(Z, ["method", "condition", "n", "slope_per_day", "ci_low", "ci_high", "spearman_rho", "spearman_p",
                        "r2_first_day", "r2_last_day"]) if len(Z) else "(vuoto)\n",
           f"\n## Bin dei 2 trial di adattamento (canali attivi: {n_ch})\n",
           f"Tutti i bin: min {A.bins.min()}, mediana {A.bins.median():.0f}, max {A.bins.max()}; con eval_mask: "
           f"min {A.bins_mask.min()}, mediana {A.bins_mask.median():.0f}, max {A.bins_mask.max()} (42 fold).\n\n",
           md_table(summ, ["method", "latent_dim", "readout_weights_per_dof", "bins_mask_min", "ratio_min"]),
           "\n## Verifica sui latenti a 3 dimensioni (esplorativa, non nel piano)\n",
           (open(OUT / "diag_3d_within_session.csv").read().replace("# ", "") if (OUT / "diag_3d_within_session.csv").exists()
            else "(non presente)\n"),
           "\n## Pseudo-few-shot sulle held-in (scelte congelate)\n",
           md_table(TU, [c for c in ["method", "heldin_alpha_src", "heldin_lodo_r2", "pseudo_zero_shot", "pseudo_recenter",
                                     "pseudo_align", "pseudo_fa_native", "pseudo_fewshot", "lam", "lam_at_edge",
                                     "pseudo_scratch", "pseudo_ridge_rates", "alpha_scratch", "alpha_at_edge"] if c in TU.columns])
           if len(TU) else "(vuoto)\n"]
    (OUT / "report.md").write_text("".join(rep))
    print(f"scritto {OUT}")


if __name__ == "__main__":
    main()
