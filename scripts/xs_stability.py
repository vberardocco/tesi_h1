"""Stability of the FA subspaces on the held-out sessions and alignment index against zero-shot R²
(docs/crosssession_protocol.md, section "Stability of the FA subspaces on the held-out sessions", 2 Nov 2026).

compute  blocks of 3 trials (54 held-in, 14 held-out), FA K = 15; index and angle for all pairs of blocks
         (normalization a: z-score of the block) and for held-out blocks against the decoder's frozen FA / held-in blocks
         against the leave-one-day-out FA (normalization b: the decoder's); Elsayed-style chance level; registry
         (experiment fa_stability). Writes RESULTS/crosssession/stability/{blocks,pairs,model_pairs}.csv, meta.json.
analysis estimability rule, Mantel, slope, Spearman of figure F, descriptives, figures E and F, report.md.

Usage: python scripts/xs_stability.py [--stage compute|analysis|all]
"""
import argparse
import json
import pickle
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from thesis_h1 import crosssession as xs  # noqa: E402
from thesis_h1 import io  # noqa: E402
from thesis_h1 import palette as pal  # noqa: E402
from thesis_h1 import results as res  # noqa: E402
from thesis_h1 import stability as st  # noqa: E402

OUT = res.RESULTS / "crosssession" / "stability"
CACHE = xs.XS / "stability"
RUN_KEYS = dict(experiment="fa_stability", method="FA", latent_dim=st.K, decoder=None, bins="falcon",
                split="calib3_block", fold=None, k_shot=st.BLOCK, H_ms=None, seed=0)
SEED = 20261102
log = xs.log


# ============================================================================================
# compute
# ============================================================================================
def make_blocks(data):
    """Blocks of 3 trials: held-in as in diag_within (non-overlapping blocks from the start of the file), held-out whole."""
    key = lambda s: (int(s.day[1:]), s.key)  # noqa: E731
    out = []
    for s in sorted(xs.heldin(data), key=key):
        for j in range(len(s.trials) // st.BLOCK):
            out.append(dict(block=f"{s.key}#b{j}", session=s.key, block_index=j, day=s.day, group="held_in",
                            trials=s.trials[st.BLOCK * j:st.BLOCK * (j + 1)]))
    for s in sorted(xs.heldout(data), key=key):
        assert len(s.trials) == st.BLOCK, f"{s.key}: {len(s.trials)} trials"
        out.append(dict(block=s.key, session=s.key, block_index=0, day=s.day, group="held_out", trials=list(s.trials)))
    n_hi = sum(b["group"] == "held_in" for b in out)
    assert n_hi == 54 and len(out) == 68, f"held-in blocks {n_hi}, total {len(out)}"
    return out


def counts(trials):
    return np.concatenate([t.counts for t in trials]).astype(np.float64)


def fa_model(trials):
    """FA with the decoder recipe (FAm): normalization, subspace, covariance of the normalized data."""
    m = xs.FAm(seed=0).fit(trials)
    Z = (counts(trials) - m.mu) / m.sd
    return dict(mu=m.mu, sd=m.sd, L=m.fa.components_.T, C=np.cov(Z, rowvar=False), n_bins=len(Z))


def cond_raw(active):
    """For each held-in session: covariance of the means of the 16 conditions (8 phases × version, 1 s windows
    from the start of the phase) on the counts, and the sd of the session counts (with a minimum sd)."""
    p = CACHE / "cond_raw.pkl"
    if p.exists():
        return pickle.loads(p.read_bytes())
    out = []
    for f in sorted((xs.DATA / "sub-HumanPitt-held-in-calib").glob("*.nwb")):
        s = io.load_session(f, bins="falcon")
        X = s.spikes[:, active].astype(np.float64)
        groups = {}
        for w in io.phase_windows(s, win_s=1.0, align="start"):
            assert w["stop"] - w["start"] == 50
            groups.setdefault((w["tag"], w["trial"] % 2), []).append(X[w["start"]:w["stop"]])
        means = np.concatenate([np.mean(g, 0) for g in groups.values()])
        out.append(dict(file=f.name, n_cond=len(groups), C=np.cov(means, rowvar=False),
                        sd=np.maximum(X.std(0), st.SD_MIN)))
        log(f"  means per condition: {f.name}, {len(groups)} conditions")
    p.write_bytes(pickle.dumps(out))
    return out


def chance_stats(A, B, M_pair, banks, rng):
    """Chance-level summaries for the three versions and the two k. M_pair: 'all bins' sampler of the pair;
    banks[(version, k)] = (Q1, Q2) bases already drawn for the versions with fixed C."""
    out = {}
    for k in st.KS:
        Q1, Q2 = st.draw_bases(M_pair, k, st.N_DRAWS, rng), st.draw_bases(M_pair, k, st.N_DRAWS, rng)
        out.update(st.chance_summary(*st.chance(A, B, k, Q1, Q2), f"allbins_k{k}"))
        for v in ("cond", "unif"):
            out.update(st.chance_summary(*st.chance(A, B, k, *banks[(v, k)]), f"{v}_k{k}"))
    return out


def make_banks(M_cond, M_unif, seed):
    rng = np.random.default_rng(seed)
    return {(v, k): (st.draw_bases(M, k, st.N_DRAWS, rng), st.draw_bases(M, k, st.N_DRAWS, rng))
            for v, M in (("cond", M_cond), ("unif", M_unif)) for k in st.KS}


def compute():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    CACHE.mkdir(parents=True, exist_ok=True)
    data = xs.load_data()
    active = data["active"]
    p = len(active)
    assert p == 170
    blocks = make_blocks(data)
    hi_sessions = xs.heldin(data)
    log(f"{len(blocks)} blocks, {p} channels")

    # the decoder's frozen FA, and a check that the fa_model recipe reproduces it
    with open(xs.bundle_path("FA", 0, "heldin", st.K), "rb") as f:
        bnd = pickle.load(f)
    assert bnd.model.dim == st.K and len(bnd.info["sessions"]) == 13
    frozen = fa_model([t for s in hi_sessions for t in s.trials])
    repro = float(np.abs(frozen["L"] - bnd.model.fa.components_.T).max())
    log(f"frozen FA: recipe reproduced, max |Δ loading| = {repro:.2e}")
    assert repro < 1e-8 and np.allclose(frozen["sd"], bnd.model.sd)
    frozen["L"] = bnd.model.fa.components_.T
    # leave-one-day-out FA, for the reference held-in blocks
    lodo = {}
    for d in sorted({s.day for s in hi_sessions}, key=lambda d: int(d[1:])):
        lp = CACHE / f"lodo_{d}.pkl"
        if lp.exists():
            lodo[d] = pickle.loads(lp.read_bytes())
        else:
            t1 = time.time()
            lodo[d] = fa_model([t for s in hi_sessions if s.day != d for t in s.trials])
            lp.write_bytes(pickle.dumps(lodo[d]))
            log(f"FA without {d}: {(time.time() - t1):.0f} s")
    sub_model = {"frozen": st.Sub.from_loadings(frozen["L"])}
    sub_model.update({d: st.Sub.from_loadings(m["L"]) for d, m in lodo.items()})
    models = dict(lodo, frozen=frozen)

    # condition means: fixed C for all pairs (a: session z-score; b: model normalization)
    craw = cond_raw(active)
    C_cond_a = np.mean([c["C"] / np.outer(c["sd"], c["sd"]) for c in craw], 0)
    C_raw = np.mean([c["C"] for c in craw], 0)
    M_unif = st.sampler(np.eye(p))
    banks_a = make_banks(st.sampler(C_cond_a), M_unif, SEED)
    banks_b = {name: make_banks(st.sampler(C_raw / np.outer(m["sd"], m["sd"])), M_unif, SEED + 1 + i)
               for i, (name, m) in enumerate(models.items())}

    # block subspaces, under the two normalizations
    rows = []
    for b in blocks:
        X = counts(b["trials"])
        Za, n0, nlow = st.zscore_block(X)
        b["sub_a"], b["C_a"] = st.Sub.from_loadings(st.fit_loadings(Za)), np.cov(Za, rowvar=False)
        mname = "frozen" if b["group"] == "held_out" else b["day"]
        m = models[mname]
        Zb = (X - m["mu"]) / m["sd"]
        b["sub_b"], b["C_b"], b["model"] = st.Sub.from_loadings(st.fit_loadings(Zb)), np.cov(Zb, rowvar=False), mname
        rows.append(dict(block=b["block"], session=b["session"], block_index=b["block_index"], day=b["day"], group=b["group"], n_trials=len(b["trials"]),
                         n_bins=len(X), n_sd_zero=n0, n_sd_below_min=nlow, model_b=mname,
                         shared_var_a=float(b["sub_a"].lam.sum()), shared_var_b=float(b["sub_b"].lam.sum())))
    B = pd.DataFrame(rows)
    B.to_csv(OUT / "blocks.csv", index=False)
    log(f"block FAs done; (channel, block) pairs with sd = 0: {int(B.n_sd_zero.sum())}, with sd < {st.SD_MIN}: "
        f"{int(B.n_sd_below_min.sum())}")

    # pairs of blocks (normalization a)
    pairs = []
    n = len(blocks)
    for i in range(n):
        for j in range(i + 1, n):
            a, b = blocks[i], blocks[j]
            rng = np.random.default_rng([SEED, i, j])
            r = dict(a=a["block"], b=b["block"], session_a=a["session"], session_b=b["session"], day_a=a["day"],
                     day_b=b["day"], group_a=a["group"], group_b=b["group"], same_session=a["session"] == b["session"],
                     same_day=a["day"] == b["day"])
            r.update(st.pair_measures(a["sub_a"], b["sub_a"]))
            r.update(chance_stats(a["sub_a"], b["sub_a"], st.sampler((a["C_a"] + b["C_a"]) / 2), banks_a, rng))
            pairs.append(r)
        log(f"  pairs: block {i + 1}/{n}")
    P = pd.DataFrame(pairs)
    P.to_csv(OUT / "pairs.csv", index=False)

    # blocks against the model FA (normalization b)
    mp = []
    for i, b in enumerate(blocks):
        m = models[b["model"]]
        rng = np.random.default_rng([SEED, 999, i])
        r = dict(block=b["block"], session=b["session"], block_index=b["block_index"], day=b["day"], group=b["group"],
                 model=b["model"])
        r.update(st.pair_measures(b["sub_b"], sub_model[b["model"]]))
        r.update(chance_stats(b["sub_b"], sub_model[b["model"]], st.sampler((b["C_b"] + m["C"]) / 2),
                              banks_b[b["model"]], rng))
        r["n_silent_channels"] = int((counts(b["trials"]).std(0) == 0).sum())
        mp.append(r)
    MP = pd.DataFrame(mp)
    MP.to_csv(OUT / "model_pairs.csv", index=False)

    # registry: one row per (block, metric); session = FALCON key of the file, fold = block number within the file
    metr = [c for c in MP.columns if c.startswith(("index_", "angle_", "allbins_", "cond_", "unif_"))] + ["n_silent_channels"]
    keys = {k: v for k, v in RUN_KEYS.items() if k != "fold"}
    for r in MP.itertuples(index=False):
        cond = "vs_frozen_fa" if r.group == "held_out" else "vs_lodo_fa"
        res.log_run({c: getattr(r, c) for c in metr}, session=r.session, condition=cond, fold=int(r.block_index), **keys)
    cross = P[(P.group_a != P.group_b)]
    for ho in B[B.group == "held_out"].block:
        d = cross[(cross.a == ho) | (cross.b == ho)]
        assert len(d) == 54
        res.log_run({f"{c}_median": d[c].median() for c in ("index_k15", "angle_k15", "index_k5", "angle_k5")},
                    session=ho, condition="vs_heldin_blocks", **RUN_KEYS)
    meta = dict(n_blocks=len(blocks), n_heldin_blocks=int((B.group == "held_in").sum()), n_pairs=len(P),
                n_channels=p, K=st.K, K_sens=st.K_SENS, sd_min=st.SD_MIN, n_draws=st.N_DRAWS, seed=SEED,
                frozen_recipe_max_abs_diff=repro, cond_conditions={c["file"]: c["n_cond"] for c in craw},
                n_sd_zero=int(B.n_sd_zero.sum()), n_sd_below_min=int(B.n_sd_below_min.sum()),
                n_silent_b=int(MP.n_silent_channels.sum()), minutes=(time.time() - t0) / 60)
    (OUT / "meta.json").write_text(json.dumps(meta, indent=1))
    log(f"compute finished in {meta['minutes']:.1f} min")


# ============================================================================================
# analysis
# ============================================================================================
def day_table():
    s = pd.read_csv(res.RESULTS / "sessions.csv")
    d = s.groupby("day").data.first().map(pd.Timestamp)
    days = sorted(d.index, key=lambda x: int(x[1:]))
    date = np.array([(d[x] - d[days[0]]).days for x in days])
    return days, date, dict(s[["session", "days_from_held_in"]].values)


def fmt(x, nd=3):
    return "NA" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def analysis():
    B = pd.read_csv(OUT / "blocks.csv")
    P = pd.read_csv(OUT / "pairs.csv")
    MP = pd.read_csv(OUT / "model_pairs.csv")
    meta = json.loads((OUT / "meta.json").read_text())
    days, date, since = day_table()
    di = {d: i for i, d in enumerate(days)}
    P["ddays"] = np.abs(date[P.day_a.map(di)] - date[P.day_b.map(di)])
    kind = {("held_in", "held_in"): "held-in–held-in", ("held_in", "held_out"): "held-in–held-out",
            ("held_out", "held_in"): "held-in–held-out", ("held_out", "held_out"): "held-out–held-out"}
    P["pair_type"] = [kind[(a, b)] for a, b in zip(P.group_a, P.group_b)]
    T = {}

    # 1. estimability rule
    W = P[P.same_session]
    assert len(W) == 95, len(W)
    rule = {}
    for k in st.KS:
        med, med_a = W[f"index_k{k}"].median(), W[f"angle_k{k}"].median()
        thr = {v: W[f"{v}_k{k}_index_p95"].median() for v in st.CHANCE}
        thr_a = {v: W[f"{v}_k{k}_angle_p5"].median() for v in st.CHANCE}
        rule[k] = dict(index_median=med, index_q25=W[f"index_k{k}"].quantile(.25), index_q75=W[f"index_k{k}"].quantile(.75),
                       chance_p95=thr, passed=bool(all(med > t for t in thr.values())),
                       angle_median=med_a, chance_angle_p5=thr_a, angle_passed=bool(all(med_a < t for t in thr_a.values())))
    K0 = st.K if rule[st.K]["passed"] else st.K_SENS
    T["rule"] = {str(k): v for k, v in rule.items()}
    T["primary_k"] = K0

    # 2. Mantel and slope (excluding pairs within the same session)
    X = P[~P.same_session]
    da, db = X.day_a.map(di).to_numpy(), X.day_b.map(di).to_numpy()
    T["n_pairs_mantel"] = len(X)
    T["n_pairs_same_day_other_file"] = int((X.same_day).sum())
    mt = {}
    for k in st.KS:
        for meas in ("index", "angle"):
            col = f"{meas}_k{k}"
            rho, pv = st.mantel(X[col], da, db, date, n_perm=10_000, seed=0)
            sl, lo, hi, nb = st.slope_bootstrap(X[col], da, db, date, n_boot=2_000, seed=0)
            mt[col] = dict(rho=rho, p=pv, slope_per_day=sl, slope_ci95=[lo, hi], n_boot=nb,
                           primary=(meas == "index" and k == K0))
            log(f"Mantel {col}: ρ = {rho:+.3f}, p = {pv:.4f}; slope {sl:+.5f}/day [{lo:+.5f}, {hi:+.5f}]")
    T["mantel"] = mt
    T["by_pair_type"] = {}
    for t, g in X.groupby("pair_type"):
        T["by_pair_type"][t] = dict(n=len(g), index_median=g[f"index_k{K0}"].median(),
                                    angle_median=g[f"angle_k{K0}"].median(),
                                    spearman_vs_days=float(stats.spearmanr(g[f"index_k{K0}"], g.ddays).statistic)
                                    if g.ddays.nunique() > 1 else None, ddays_range=[int(g.ddays.min()), int(g.ddays.max())])
    sd = X[X.same_day]
    T["same_day_other_file"] = {g: dict(n=len(d), index_median=d[f"index_k{K0}"].median(),
                                        angle_median=d[f"angle_k{K0}"].median())
                                for g, d in sd.groupby("pair_type")}

    # 3. figure F: index against the frozen FA and zero-shot R²
    S = pd.read_csv(res.RESULTS / "crosssession" / "sessions_r2.csv")
    r2 = S.pivot_table(index="session", columns=["method", "condition"], values="r2")
    HO = MP[MP.group == "held_out"].set_index("block").sort_index()
    HI = MP[MP.group == "held_in"]
    x = HO[f"index_k{K0}"]
    y = r2[("FA", "zero_shot")].reindex(x.index)
    gday = pd.Series({s: since[s] for s in x.index})
    rho, pv = st.perm_spearman(x, y, n_perm=10_000, seed=0)
    dm = pd.DataFrame(dict(x=x, y=y, g=gday)).groupby("g").mean()
    rho_d, p_d = st.perm_spearman(dm.x, dm.y)
    rho_p, p_p = st.partial_spearman(x, y, gday, n_perm=10_000, seed=0)
    T["fig_F"] = dict(x=f"index_k{K0} vs frozen FA", n=len(x), rho=rho, p=pv, rho_day_means=rho_d, p_day_means=p_d,
                      n_days=len(dm), rho_partial_days=rho_p, p_partial_days=p_p,
                      critical_rho_n14=st.critical_rho(len(x)), critical_rho_n7=st.critical_rho(len(dm)))
    log(f"Figure F: ρ = {rho:+.3f} (p = {pv:.4f}); day means ρ = {rho_d:+.3f} (p = {p_d:.4f}); "
        f"partial ρ = {rho_p:+.3f} (p = {p_p:.4f})")
    T["frozen"] = dict(heldout_index_median=x.median(), heldout_index_range=[x.min(), x.max()],
                       heldout_angle_median=HO[f"angle_k{K0}"].median(),
                       heldin_lodo_index_median=HI[f"index_k{K0}"].median(),
                       heldin_lodo_index_iqr=[HI[f"index_k{K0}"].quantile(.25), HI[f"index_k{K0}"].quantile(.75)],
                       heldin_lodo_angle_median=HI[f"angle_k{K0}"].median(),
                       chance_p95_median={v: HO[f"{v}_k{K0}_index_p95"].median() for v in st.CHANCE},
                       heldout_above_all_chance=int((HO[f"index_k{K0}"] > HO[[f"{v}_k{K0}_index_p95" for v in st.CHANCE]]
                                                     .max(1)).sum()),
                       sensitivity_other_k={f"index_k{k}": dict(zip(["rho", "p"], st.perm_spearman(
                           HO[f"index_k{k}"], y, n_perm=10_000, seed=0))) for k in st.KS if k != K0},
                       secondary_angle=dict(zip(["rho", "p"], st.perm_spearman(HO[f"angle_k{K0}"], y, n_perm=10_000,
                                                                              seed=0))))
    desc = {}
    for m in ("PCA", "GPFA", "LFADS", "ridge_channels"):
        if (m, "zero_shot") in r2:
            desc[f"{m}_zero_shot"] = float(stats.spearmanr(x, r2[(m, "zero_shot")].reindex(x.index)).statistic)
    gain = (r2[("FA", "align")] - r2[("FA", "recenter")]).reindex(x.index)
    desc["FA_align_minus_recenter"] = float(stats.spearmanr(x, gain).statistic)
    cross = P[P.group_a != P.group_b]
    med_blocks = pd.Series({ho: cross[(cross.a == ho) | (cross.b == ho)][f"index_k{K0}"].median() for ho in x.index})
    desc["median_vs_heldin_blocks__FA_zero_shot"] = float(stats.spearmanr(med_blocks, y).statistic)
    desc["median_vs_heldin_blocks__vs_frozen"] = float(stats.spearmanr(med_blocks, x).statistic)
    T["descriptive_spearman"] = desc
    (OUT / "tests.json").write_text(json.dumps(T, indent=1, default=float))
    per = pd.DataFrame(dict(index_vs_frozen=x, angle_vs_frozen=HO[f"angle_k{K0}"], index_median_vs_heldin_blocks=med_blocks,
                            fa_zero_shot=y, fa_align_minus_recenter=gain, days_since_heldin=gday))
    per.to_csv(OUT / "heldout_sessions.csv", index_label="session")
    figures(P, X, W, HO, HI, r2, x, y, gday, gain, med_blocks, K0, T)
    report(T, meta, B, K0)


def figures(P, X, W, HO, HI, r2, x, y, gday, gain, med_blocks, K0, T):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    ink, ink2 = pal.INK, pal.INK2
    tcol = {"held-in–held-in": (pal.CAT[0], "o"), "held-in–held-out": (pal.CAT[1], "s"),
            "held-out–held-out": (pal.CAT[2], "^")}
    chance_ls = {"allbins": ("--", "chance, all-bins cov."), "cond": (":", "chance, condition-mean cov."),
                 "unif": ("-.", "chance, uniform")}
    # Figure E: index and angle against days apart, for k = 15 and k = 5
    for k in st.KS:
        fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), layout="constrained")
        for ax, meas, pc, lab in ((axes[0], "index", "p95", "alignment index (shared covariance)"),
                                  (axes[1], "angle", "p5", "mean principal angle (deg)")):
            pal.style_axes(ax)
            col = f"{meas}_k{k}"
            q1, q2, q3 = W[col].quantile([.25, .5, .75])
            ax.axhspan(q1, q3, color=pal.GRID, alpha=0.9, lw=0, zorder=0)
            ax.axhline(q2, color=ink2, lw=1.2, zorder=1)
            xmax = X.ddays.max()
            for v, (ls, _) in chance_ls.items():
                ax.axhline(W[f"{v}_k{k}_{meas}_{pc}"].median(), color=ink2, lw=1, ls=ls, zorder=1)
            for t, (c, mk) in tcol.items():
                g = X[X.pair_type == t]
                ax.scatter(g.ddays + np.random.default_rng(0).uniform(-0.25, 0.25, len(g)), g[col], s=4, color=c,
                           alpha=0.18, lw=0, zorder=2)
                gm = g.groupby(["day_a", "day_b"]).agg(dd=("ddays", "first"), v=(col, "median"))
                ax.scatter(gm.dd, gm.v, s=42, color=c, marker=mk, edgecolor="white", lw=1.2, zorder=3)
            ax.set_xlim(-1, xmax + 1.5)
            ax.set_xlabel("days between the two blocks", fontsize=9, color=ink2)
            ax.set_ylabel(lab, fontsize=9, color=ink2)
            m = T["mantel"][col]
            tag = "Mantel test" if m["primary"] else "Mantel (secondary)"
            ax.set_title(f"{lab.split(' (')[0]}, k = {k}\n{tag}: ρ = {m['rho']:+.2f}, p = {m['p']:.3g}; "
                         f"slope {m['slope_per_day']:+.4f} per day", fontsize=9, color=ink, loc="left")
            pct = f"{pc[1:]}th perc."
            handles = [Line2D([], [], color=c, marker=mk, lw=0, ms=6, label=f"{t} (median per pair of days)")
                       for t, (c, mk) in tcol.items()]
            handles.append(Patch(color=pal.GRID, label="same session: IQR, line = median"))
            handles += [Line2D([], [], color=ink2, lw=1, ls=ls, label=f"{lab_c} ({pct}, median over pairs)")
                        for ls, lab_c in chance_ls.values()]
            ax.legend(handles=handles, frameon=False, fontsize=7.5, loc="upper right" if meas == "index" else "lower right",
                      ncol=1)
        fig.suptitle(f"FA subspaces (K = 15) of 3-trial blocks: 54 held-in, 14 held-out; faint dots = pairs of blocks "
                     f"(same-session pairs only in the band); k = {k} directions", fontsize=9.5, color=ink)
        fig.savefig(OUT / f"fig_E_stability_by_days_k{k}.png", dpi=160)
        plt.close(fig)

    # Figure F
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 10), layout="constrained")
    cmap = matplotlib.colormaps["Blues"]
    gd = gday.to_numpy(float)
    cn = (gd - 4) / (gd.max() - 4)                       # 6-20 days -> 0.13-1, no tones that are too light
    col_days = cmap(0.35 + 0.65 * cn)
    lo, hi = HI[f"index_k{K0}"].quantile([.25, .75])
    hm = HI[f"index_k{K0}"].median()

    def frame(ax, xlabel, ylabel):
        pal.style_axes(ax)
        ax.set_xlabel(xlabel, fontsize=9, color=pal.INK2)
        ax.set_ylabel(ylabel, fontsize=9, color=pal.INK2)

    def ref(ax):
        ax.axvspan(lo, hi, color=pal.GRID, alpha=0.9, lw=0, zorder=0)
        ax.axvline(hm, color=pal.INK2, lw=1.2, zorder=1)
        c = T["frozen"]["chance_p95_median"]["allbins"]
        ax.axvline(c, color=pal.INK2, lw=1, ls="--", zorder=1)

    xl = f"alignment index with the decoder's frozen FA (k = {K0})"
    ax = axes[0, 0]
    frame(ax, xl, "FA zero-shot R²")
    ref(ax)
    ax.scatter(x, y, s=60, c=col_days, edgecolor="white", lw=1.2, zorder=3)
    f = T["fig_F"]
    ax.set_title(f"(a) test: Spearman ρ = {f['rho']:+.2f}, p = {f['p']:.3g} (n = 14; significant only if |ρ| ≥ "
                 f"{f['critical_rho_n14']:.2f})\nday means: ρ = {f['rho_day_means']:+.2f}, p = {f['p_day_means']:.3g} "
                 f"(n = 7; only if |ρ| ≥ {f['critical_rho_n7']:.2f})\npartial on days since held-in: "
                 f"ρ = {f['rho_partial_days']:+.2f}, p = {f['p_partial_days']:.3g}", fontsize=8.5, color=pal.INK, loc="left")
    ax = axes[0, 1]
    frame(ax, xl, "FA R²(align) − R²(recenter)")
    ref(ax)
    ax.axhline(0, color=pal.INK2, lw=0.8)
    ax.scatter(x, gain, s=60, c=col_days, edgecolor="white", lw=1.2, zorder=3)
    ax.set_title(f"(b) descriptive: ρ = {T['descriptive_spearman']['FA_align_minus_recenter']:+.2f}", fontsize=8.5,
                 color=pal.INK, loc="left")
    ax = axes[1, 0]
    frame(ax, xl, "zero-shot R²")
    ref(ax)
    ax.axhline(0, color=pal.INK2, lw=0.8)
    hs = []
    for m in ("PCA", "GPFA", "LFADS", "ridge_channels"):
        if (m, "zero_shot") not in r2:
            continue
        c, mk = pal.METHOD_STYLE[m]
        ax.scatter(x, r2[(m, "zero_shot")].reindex(x.index), s=36, color=c, marker=mk, edgecolor="white", lw=1, zorder=3)
        hs.append(Line2D([], [], color=c, marker=mk, lw=0, ms=6,
                         label=f"{m} (ρ = {T['descriptive_spearman'][f'{m}_zero_shot']:+.2f})"))
    ax.legend(handles=hs, frameon=False, fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=2)
    ax.set_title("(c) descriptive: same index, zero-shot of the other methods", fontsize=8.5, color=pal.INK, loc="left")
    ax = axes[1, 1]
    frame(ax, f"median alignment index with the 54 held-in blocks (k = {K0}, block z-score)", "FA zero-shot R²")
    ax.scatter(med_blocks, y, s=60, c=col_days, edgecolor="white", lw=1.2, zorder=3)
    ax.set_title(f"(d) descriptive: ρ = {T['descriptive_spearman']['median_vs_heldin_blocks__FA_zero_shot']:+.2f}",
                 fontsize=8.5, color=pal.INK, loc="left")
    sm = matplotlib.cm.ScalarMappable(cmap=matplotlib.colors.ListedColormap(cmap(np.linspace(0.35 + 0.65 * (6 - 4) / 16, 1, 64))),
                                      norm=matplotlib.colors.Normalize(6, 20))
    cb = fig.colorbar(sm, ax=axes[:, 1], shrink=0.5, pad=0.02)
    cb.set_label("days since last held-in session", fontsize=8.5, color=pal.INK2)
    cb.ax.tick_params(labelsize=8, colors=pal.INK2)
    fig.suptitle("14 held-out sessions, colour = days since the last held-in session. Grey band = held-in blocks vs the FA "
                 "fit without their day (IQR, line = median);\ndashed line = chance (all-bins covariance, median 95th "
                 "percentile)", fontsize=9, color=pal.INK, x=0.01, ha="left")
    fig.savefig(OUT / "fig_F_alignment_vs_zero_shot.png", dpi=160)
    plt.close(fig)


def report(T, meta, B, K0):
    r15, r5 = T["rule"][str(st.K)], T["rule"][str(st.K_SENS)]

    def rule_line(k, r):
        c = ", ".join(f"{v} {r['chance_p95'][v]:.3f}" for v in st.CHANCE)
        ca = ", ".join(f"{v} {r['chance_angle_p5'][v]:.1f}°" for v in st.CHANCE)
        return (f"| {k} | {r['index_median']:.3f} [{r['index_q25']:.3f}, {r['index_q75']:.3f}] | {c} | "
                f"{'yes' if r['passed'] else 'NO'} | {r['angle_median']:.1f}° | {ca} | {'yes' if r['angle_passed'] else 'no'} |")

    L = ["# Stability of FA subspaces on the held-out sessions", "",
         "Plan: `docs/crosssession_protocol.md`, section of 2 Nov 2026 (written before any index was computed). "
         "Code: `src/thesis_h1/stability.py`, `scripts/xs_stability.py`, tests in `tests/test_stability.py`.", "",
         "## Data", "",
         f"- {meta['n_blocks']} blocks of 3 consecutive trials ({meta['n_heldin_blocks']} held-in, 14 held-out), "
         f"{meta['n_channels']} channels, FA K = {meta['K']}; {meta['n_pairs']} pairs of blocks.",
         f"- The recipe of the decoder's frozen FA was reproduced exactly (max |Δ loading| = "
         f"{meta['frozen_recipe_max_abs_diff']:.1e}).",
         f"- Silent channels: (channel, block) cases with sd = 0 in the block z-score: {meta['n_sd_zero']}; with sd below "
         f"sd_min = {meta['sd_min']}: {meta['n_sd_below_min']} (of {meta['n_blocks'] * meta['n_channels']}). "
         f"With the decoder normalization, silent channels in the blocks vs the model: {meta['n_silent_b']}.",
         f"- Condition means: conditions per held-in file {sorted(set(meta['cond_conditions'].values()))}.",
         "", "## Estimability rule (same-session reference vs chance)", "",
         "| k | same-session index, median [IQR] (95 pairs) | chance 95th perc. (median over pairs) | passes | "
         "same-session angle, median | chance 5th perc. | passes |", "|---|---|---|---|---|---|---|",
         rule_line(st.K, r15), rule_line(st.K_SENS, r5), "",
         f"Primary k for the tests: **{K0}**" + ("." if K0 == st.K else
                                                  " (the K = 15 subspace is not estimable from 3 trials by the rule)."),
         "", "## Stability across days (Mantel test, same-session pairs excluded)", "",
         f"{T['n_pairs_mantel']} pairs ({T['n_pairs_same_day_other_file']} same day, other file, at distance 0).", "",
         "| measure | Spearman ρ with days apart | p (permutation of days) | slope per day [95% CI, bootstrap on days] |",
         "|---|---|---|---|"]
    for col, m in T["mantel"].items():
        L.append(f"| {col}{' (primary test)' if m['primary'] else ''} | {m['rho']:+.3f} | {m['p']:.4f} | "
                 f"{m['slope_per_day']:+.5f} [{m['slope_ci95'][0]:+.5f}, {m['slope_ci95'][1]:+.5f}] |")
    L += ["", f"By pair type (k = {K0}, descriptive):", "", "| pair type | pairs | days apart | median index | "
          "median angle | Spearman with days |", "|---|---|---|---|---|---|"]
    for t, d in T["by_pair_type"].items():
        L.append(f"| {t} | {d['n']} | {d['ddays_range'][0]}–{d['ddays_range'][1]} | {d['index_median']:.3f} | "
                 f"{d['angle_median']:.1f}° | {fmt(d['spearman_vs_days'], 2)} |")
    L += ["", "Same day, other file: " + "; ".join(f"{g}: {d['n']} pairs, median index {d['index_median']:.3f}, "
                                                    f"angle {d['angle_median']:.1f}°"
                                                    for g, d in T["same_day_other_file"].items()) + ".", ""]
    fr, f = T["frozen"], T["fig_F"]
    L += ["## Alignment with the decoder's frozen FA and zero-shot R² (Fig. F)", "",
          f"- Held-out blocks vs frozen FA (k = {K0}): median index {fr['heldout_index_median']:.3f} "
          f"[{fr['heldout_index_range'][0]:.3f}, {fr['heldout_index_range'][1]:.3f}], median angle "
          f"{fr['heldout_angle_median']:.1f}°.",
          f"- Reference, held-in blocks vs the FA fit without their day: median index "
          f"{fr['heldin_lodo_index_median']:.3f} [IQR {fr['heldin_lodo_index_iqr'][0]:.3f}, "
          f"{fr['heldin_lodo_index_iqr'][1]:.3f}], median angle {fr['heldin_lodo_angle_median']:.1f}°.",
          "- Chance 95th percentile (median over the 14): " +
          ", ".join(f"{v} {c:.3f}" for v, c in fr["chance_p95_median"].items()) +
          f"; held-out sessions above all three: {fr['heldout_above_all_chance']}/14.", "",
          "| test | ρ | p | significant only if abs(ρ) ≥ |", "|---|---|---|---|",
          f"| index vs FA zero-shot R², sessions (n = {f['n']}) — **the test** | {f['rho']:+.3f} | {f['p']:.4f} | "
          f"{f['critical_rho_n14']:.3f} |",
          f"| same, day means (n = {f['n_days']}) | {f['rho_day_means']:+.3f} | {f['p_day_means']:.4f} | "
          f"{f['critical_rho_n7']:.3f} |",
          f"| same, partial on days since held-in | {f['rho_partial_days']:+.3f} | {f['p_partial_days']:.4f} | — |"]
    for k, v in fr["sensitivity_other_k"].items():
        L.append(f"| sensitivity: {k} vs FA zero-shot | {v['rho']:+.3f} | {v['p']:.4f} | — |")
    L.append(f"| secondary: angle_k{K0} vs FA zero-shot | {fr['secondary_angle']['rho']:+.3f} | "
             f"{fr['secondary_angle']['p']:.4f} | — |")
    L += ["", "Descriptive Spearman ρ (no test): " +
          "; ".join(f"{k} {v:+.2f}" for k, v in T["descriptive_spearman"].items()) + ".", "",
          "A null result reads as \"not detectable with these data\" (n = 14 detects only |ρ| ≳ 0.54, n = 7 only "
          "|ρ| ≳ 0.79), not as \"no relation\".", "",
          "Files: `blocks.csv`, `pairs.csv`, `model_pairs.csv`, `heldout_sessions.csv`, `tests.json`, "
          "`fig_E_stability_by_days_k15.png`, `fig_E_stability_by_days_k5.png`, `fig_F_alignment_vs_zero_shot.png`; "
          "registry rows with experiment `fa_stability`."]
    (OUT / "report.md").write_text("\n".join(L) + "\n")
    log(f"report: {OUT / 'report.md'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=("compute", "analysis", "all"), default="all")
    a = ap.parse_args()
    if a.stage in ("compute", "all"):
        compute()
    if a.stage in ("analysis", "all"):
        analysis()
