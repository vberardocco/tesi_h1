"""Velocity R² curve against the number of components (minival, official FALCON evaluator).

Question: does the 0.100 of the ridge on the 15 FA latents (table B), against the 0.448 of the ridge on the channels
(ridge_h50), mean that velocity does not lie in the directions of maximum shared variance, or only that 15 latents are too few?

Pipeline: the same as for the two starting numbers.
- Ridge on channels (0.448): scripts/05_ridge_tune_long.py (α), scripts/05_ridge_fit_final.py (model),
  scripts/05_ridge_minival_h50.py (evaluation); code in src/thesis_h1/decoding.py. FALCON bins (load_nwb), 176 channels
  z-scored with the mean and sd of the held-in calib data, causal history of 50 bins (1000 ms) running through the whole file, ridge with
  free intercept on the eval_mask bins, α chosen with the 80/20 split per calib file (grid 1-10^6, 13 values).
- PCA (this script): PCA of the z-scored channels fitted ONLY on the training data (all bins, like the z-score; for the
  choice of α the first 80% of each file), before the history; scores NOT standardized, then the same 50-bin history and the
  same ridge. α chosen with the same 80/20 procedure for each (variant, k), then a model on all calib data; in addition a
  fixed α of 10^4.5 (sensitivity). "top" = the k PCs of largest variance, "bottom" = the k of smallest variance among the 175 non-null ones
  (one channel is silent in the calib data). With k = 175 the model is exactly ridge_h50: the ridge is invariant under rotation and the
  silent channel has weight 0.
  Each PCA model is rewritten as an equivalent ridge on the 176 channels, W_channels = (I_50 ⊗ V_k) W, and evaluated with the
  streaming decoder of ridge_h50 (decoding.make_streaming_decoder). The X'X sums of the PC history are derived
  exactly from those of the channel history: a single pass over the data per phase.
- FA (this script): the recipe of the frozen model FA_d15_heldin_s0 (crosssession.fit_bundle: 170 active channels,
  held-in z-score, FA with K factors, standardized latents, 50-bin history that restarts at each trial in training,
  leave-one-day-out α on the held-in sessions), one model for each K; exported and evaluated as in scripts/05_fa_minival.py.
- Evaluation: FalconEvaluator(split="h1"), phase="minival", streaming with batch 4 (the reported number); check against the
  block computation and our evaluator (decoding.evaluate), and with batch 1 for k = 175 and FA K = 15.
No choice is made on minival. The registry results/h1_runs.csv is NOT touched (user request of 6 Nov).

Phases (--stage): pca, fa, report; optional: channels (ridge on random subsets of channels), report_channels.
Outputs: results/curve_components/ (CSV, figures, log, summary); models and cache in ~/thesis_h1_work/curve_components/.
Restart: the phases re-read the X'X sums, the α choices and the rows already evaluated.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from thesis_h1 import decoding as dec  # noqa: E402

RAW = Path.home() / "000954"
OUT = REPO / "results" / "curve_components"
WORK = Path(os.environ.get("CURVE_WORK", Path.home() / "thesis_h1_work" / "curve_components"))
CACHE, MODELS = WORK / "cache", WORK / "models"
TMP = Path(os.environ.get("CURVE_TMP", tempfile.mkdtemp()))

H = 50                                     # history: 50 bins = 1000 ms
C = dec.N_CH                               # 176 channels
ALPHAS = np.logspace(0, 6, 13)             # grid of scripts/05_ridge_tune_long.py
ALPHA_H50 = 10 ** 4.5                      # α of ridge_h50 (scripts/05_ridge_fit_final.py)
K_PCA = [175, 5, 10, 15, 25, 50, 100, 160]  # 175 = all non-null PCs, first: it is the check against 0.448
K_FA = [15, 5, 10, 25, 50]                 # 15 first: it is the check against 0.100
N_CH_SUB, N_DRAWS, SEED_DRAWS = [20, 40, 80, 120, 160], 10, 0
R2_H50, R2_FA15, R2_FALCON_SAMPLE = 0.448332, 0.100056, 0.195
H50_MODEL, FA15_MODEL = REPO / "models/ridge_h50.npz", REPO / "models/fa15_ridge_h50.npz"

assert ALPHAS[9] == ALPHA_H50              # 10^4.5 is in the grid, bit-identical

for d in (OUT, CACHE, MODELS, TMP):
    d.mkdir(parents=True, exist_ok=True)
_LOG = open(OUT / "run.log", "a")


def out(*a):
    s = f"[{time.strftime('%H:%M:%S')}] " + " ".join(str(x) for x in a)
    print(s, flush=True)
    _LOG.write(s + "\n")
    _LOG.flush()


# --------------------------------------------------------------------------------------------
# Data and X'X sums of the channels (as in 05_ridge_tune_long.py and 05_ridge_fit_final.py)
# --------------------------------------------------------------------------------------------
def load():
    cal = dec.load_split(RAW, "held-in-calib")
    mv = dec.load_split(RAW, "held-in-minival")
    assert len(cal) == 13 and len(mv) == 13
    return cal, mv


def split_80_20(cal):
    """80% / 20% per file, only bins with eval_mask (scripts/05_ridge_tune_long.py)."""
    cut = [int(0.8 * len(f.counts)) for f in cal]
    first = [np.r_[np.ones(c, bool), np.zeros(len(f.counts) - c, bool)] for f, c in zip(cal, cut)]
    tr = [r & f.mask for r, f in zip(first, cal)]
    va = [~r & f.mask for r, f in zip(first, cal)]
    return cut, tr, va


def channel_model(stage, cal):
    """RidgeHistory of the channels with the X'X sums. 'tune': z-score on the first 80% of each file, rows of the 80% with eval_mask;
    'final': z-score on all calib data, all rows with eval_mask. Cache in WORK/cache."""
    f_xtx, f_rest = CACHE / f"chan_{stage}_XtX.npy", CACHE / f"chan_{stage}_rest.npz"
    m = dec.RidgeHistory(H)
    if f_xtx.exists() and f_rest.exists():
        r = np.load(f_rest)
        m.mu, m.sd = r["mu"], r["sd"]
        m.stats = dict(XtX=np.load(f_xtx, mmap_mode="r"), Xty=r["Xty"], sx=r["sx"], sy=r["sy"], n=int(r["n"]))
        out(f"X'X sums of the channels ({stage}) re-read from the cache")
        return m
    t0 = time.time()
    if stage == "tune":
        cut, tr, _ = split_80_20(cal)
        m.fit_scaler([f.counts[:c] for f, c in zip(cal, cut)])
        m.accumulate([f.counts for f in cal], [f.vel for f in cal], tr)
    else:
        m.fit_scaler([f.counts for f in cal])
        m.accumulate([f.counts for f in cal], [f.vel for f in cal], [f.mask for f in cal])
    s = m.stats
    tmp = f_xtx.with_name(f_xtx.stem + ".tmp")
    with open(tmp, "wb") as fh:
        np.save(fh, s["XtX"])
    os.replace(tmp, f_xtx)
    np.savez(f_rest, mu=m.mu, sd=m.sd, Xty=s["Xty"], sx=s["sx"], sy=s["sy"], n=s["n"])
    out(f"X'X sums of the channels ({stage}): {s['n']} bins, {C * H} columns, {time.time() - t0:.0f} s")
    del m, s
    return channel_model(stage, cal)          # re-read from disk without keeping it in RAM (the Mac has 8 GB)


# --------------------------------------------------------------------------------------------
# PCA and ridge in the PC basis
# --------------------------------------------------------------------------------------------
def pca(m, counts_list):
    """PCA of the z-scored channels (z-score of m) on the given bins: decreasing eigenvalues, eigenvectors, number of non-null PCs."""
    Z = np.concatenate([m.z(c) for c in counts_list]).astype(np.float64)
    Z -= Z.mean(0)
    w, V = np.linalg.eigh(Z.T @ Z / len(Z))
    w, V = w[::-1].copy(), V[:, ::-1].copy()
    return w, V, int((w > 1e-8 * w[0]).sum())


def basis(V, nn, variant, k):
    """The k "top" PCs (largest variance) or "bottom" PCs (smallest variance among the nn non-null ones): 176 × k matrix."""
    idx = np.arange(k) if variant == "top" else np.arange(nn - k, nn)
    return V[:, idx]


def stats_in_basis(s, B):
    """X'X sums of the history of Z @ B (B: 176 × k), derived exactly from those of the history of Z.
    Columns in the order of decoding.lagged: first the lag, then the column."""
    k = B.shape[1]
    G = s["XtX"]
    XtX = np.empty((H * k, H * k))
    for j in range(H):
        L = B.T @ G[j * C:(j + 1) * C]                                  # k × (H·176)
        XtX[j * k:(j + 1) * k] = (L.reshape(k, H, C) @ B).reshape(k, H * k)
    Xty = np.einsum("jcy,cs->jsy", s["Xty"].reshape(H, C, 7), B).reshape(H * k, 7)
    sx = (s["sx"].reshape(H, C) @ B).reshape(H * k)
    return dict(XtX=XtX, Xty=Xty, sx=sx, sy=s["sy"], n=s["n"])


def stats_subset(s, cols):
    """X'X sums for a subset of columns (subsets of channels)."""
    return dict(XtX=s["XtX"][np.ix_(cols, cols)], Xty=s["Xty"][cols], sx=s["sx"][cols], sy=s["sy"], n=s["n"])


def ridge(stats, alphas):
    """The ridge of decoding.RidgeHistory.solve (free intercept, centred X), for each α, with a Cholesky solve
    instead of the eigenvalue decomposition: same solution (difference ~1e-14 in the synthetic test),
    but the decomposition of an 8000 × 8000 matrix with 8 GB of RAM ended up in swap (19 min instead of ~1.5).
    WARNING: modifies stats["XtX"] in place (the sums passed here are copies that are no longer needed)."""
    from scipy.linalg import solve
    n = stats["n"]
    mx, my = stats["sx"] / n, stats["sy"] / n
    Cc = stats["XtX"]
    for i in range(0, len(Cc), 1000):                                    # Cc -= n·mx·mxᵀ in blocks (less memory)
        Cc[i:i + 1000] -= n * np.outer(mx[i:i + 1000], mx)
    r = stats["Xty"] - n * np.outer(mx, my)
    out_ = {}
    for a in alphas:
        A = Cc.copy()
        A[np.diag_indices_from(A)] += a
        W = solve(A, r, assume_a="pos", overwrite_a=True, check_finite=False)
        del A
        out_[float(a)] = (W, my - mx @ W)
    return out_


def to_channels(W, B):
    """Weights on the PC history (H·k × 7) -> equivalent weights on the history of the 176 channels (H·176 × 7)."""
    k = B.shape[1]
    return np.einsum("cs,jsy->jcy", B, W.reshape(H, k, 7)).reshape(H * C, 7)


def z_files(m, cal):
    """Z-scored channels (z-score of m) of each calib file."""
    return [m.z(f.counts).astype(np.float64) for f in cal]


def predict_from(z, W, b, start):
    """= decoding.lagged(z, H)[start:] @ W + b, without building the history matrix (sum over lags)."""
    n = len(z)
    P = np.tile(b, (n - start, 1))
    for j in range(H):
        a = max(start, j)                                               # first row t with t - j >= 0
        if a < n:
            P[a - start:] += z[a - j:n - j] @ W[j * C:(j + 1) * C]
    return P


def val_r2(sol_ch, zs, cal, cut, va):
    """{α: (W_channels, b)} -> {α: R² on the 20% validation part of the calib data} (as in 05_ridge_tune_long.py: the history of the
    validation bins includes the bins before the cut)."""
    al = list(sol_ch)
    Wall = np.hstack([sol_ch[a][0] for a in al])
    ball = np.concatenate([sol_ch[a][1] for a in al])
    P = [predict_from(z, Wall, ball, c) for z, c in zip(zs, cut)]
    res = {}
    for i, a in enumerate(al):
        preds = [np.r_[np.zeros((c, 7)), p[:, 7 * i:7 * (i + 1)]] for p, c in zip(P, cut)]
        res[float(a)] = dec.evaluate([f.day for f in cal], [f.vel for f in cal], preds, va)[0]
    return res


# --------------------------------------------------------------------------------------------
# Official FALCON evaluator (as in 05_ridge_minival_h50.py and 05_fa_minival.py)
# --------------------------------------------------------------------------------------------
def falcon_eval(decoder, mv, tag):
    from falcon_challenge.evaluator import FalconEvaluator
    os.environ["EVAL_DATA_PATH"] = str(Path.home())                  # -> ~/000954/*minival/*.nwb
    pred_p, gt_p = TMP / f"pred_{tag}.pkl", TMP / f"gt_{tag}.pkl"
    os.environ["PREDICTION_PATH_LOCAL"] = str(pred_p)
    os.environ["GT_PATH"] = str(gt_p)
    ev = FalconEvaluator(eval_remote=False, split="h1", dataloader_workers=0)
    r = ev.evaluate(decoder, phase="minival")
    off = r["result"][0]["minival_split_h1"]
    with open(pred_p, "rb") as fh:
        preds = pickle.load(fh)["h1"]
    stream = {f.key: np.asarray(preds[f.key])[: len(f.counts)] for f in mv}   # batches padded to the longest file
    for p in (pred_p, gt_p):
        p.unlink(missing_ok=True)
    return float(off["Held In R2 Mean"]), float(off["Held In R2 Std."]), stream


def evaluate_model(name, block, make_decoder, mv, batches=(4,)):
    """Official R² (streaming, batch 4) with the check against the block computation."""
    r2_b = dec.evaluate([f.day for f in mv], [f.vel for f in mv], [block[f.key] for f in mv], [f.mask for f in mv])[0]
    res = dict(r2_block=r2_b)
    for bs in batches:
        r2, sd, stream = falcon_eval(make_decoder(bs), mv, f"{name.replace(' ', '_')}_b{bs}")
        dmax = max(np.abs(stream[f.key] - block[f.key]).max() for f in mv)
        ok = round(r2, 4) == round(r2_b, 4) and dmax < 1e-4
        out(f"  {name}: FALCON batch {bs} R² {r2:.6f} (SD over days {sd:.4f}); block {r2_b:.6f}; "
            f"max prediction diff. {dmax:.1e}; {'match' if ok else 'DO NOT MATCH'}")
        assert ok, "streaming and block results do not match"
        if bs == 4:
            res.update(r2=r2, r2_sd_days=sd, stream_block_maxdiff=float(dmax))
        else:
            res[f"r2_batch{bs}"] = r2
    return res


# --------------------------------------------------------------------------------------------
# Result rows (one row per evaluated model), for restarts
# --------------------------------------------------------------------------------------------
def rows(stage):
    p = WORK / f"rows_{stage}.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def add_row(stage, row):
    with open(WORK / f"rows_{stage}.jsonl", "a") as fh:
        fh.write(json.dumps(row) + "\n")


def is_done(stage, **key):
    return any(all(r.get(a) == b for a, b in key.items()) for r in rows(stage))


# --------------------------------------------------------------------------------------------
# Phase pca
# --------------------------------------------------------------------------------------------
def stage_pca():
    t_all = time.time()
    cal, mv = load()
    cut, tr, va = split_80_20(cal)
    tune_file = WORK / "tune_pca.json"
    tune = json.loads(tune_file.read_text()) if tune_file.exists() else {}
    todo = [(v, k) for k in K_PCA for v in ("top", "bottom") if f"{v}_{k}" not in tune]
    if todo:
        out("=" * 100)
        out("PCA, choice of α (80/20 per calib file, as in 05_ridge_tune_long.py)")
        m = channel_model("tune", cal)
        w, V, nn = pca(m, [f.counts[:c] for f, c in zip(cal, cut)])
        out(f"PCA on the first 80% of each file: {nn} non-null PCs out of {C}")
        assert nn == max(K_PCA)
        zs = z_files(m, cal)
        for variant, k in todo:
            t0 = time.time()
            B = basis(V, nn, variant, k)
            sol = ridge(stats_in_basis(m.stats, B), ALPHAS)
            cv = val_r2({a: (to_channels(W, B), b) for a, (W, b) in sol.items()}, zs, cal, cut, va)
            best = max(cv, key=cv.get)
            tune[f"{variant}_{k}"] = dict(alpha=best, val_r2=cv[best], at_edge=best in (ALPHAS[0], ALPHAS[-1]),
                                          cv={f"{a:.6g}": v for a, v in cv.items()})
            out(f"  {variant:6s} k = {k:3d}: α = {best:9.1f}{' (AT EDGE)' if tune[f'{variant}_{k}']['at_edge'] else ''}"
                f"  validation R² {cv[best]:.4f}   (α 10^4.5: {cv[float(ALPHA_H50)]:.4f})  {time.time() - t0:.0f} s")
            tune_file.write_text(json.dumps(tune, indent=1))
        del zs, m
    t175 = tune[f"top_{max(K_PCA)}"]
    out(f"check: top k = {max(K_PCA)} chooses α = {t175['alpha']:.1f} (ridge_h50: {ALPHA_H50:.1f}), "
        f"validation R² {t175['val_r2']:.4f} (05_ridge_tune_long.py, H = 50: 0.3922)")

    out("=" * 100)
    out("PCA, models on all calib data and evaluation on minival")
    m = channel_model("final", cal)
    h50 = np.load(H50_MODEL)
    assert np.array_equal(m.mu, h50["mu"]) and np.array_equal(m.sd, h50["sd"]), "z-score differs from ridge_h50"
    w, V, nn = pca(m, [f.counts for f in cal])
    out(f"PCA on all calib data: {nn} non-null PCs; variance of the first 5: {np.round(w[:5], 3)}, "
        f"of the last 5 non-null: {np.round(w[nn - 5:nn], 3)}")
    np.savez(WORK / "pca_final.npz", w=w, V=V, nn=nn)
    rh = dec.RidgeHistory(H)
    rh.mu, rh.sd = m.mu, m.sd
    for k in K_PCA:
        for variant in ("top", "bottom"):
            method = f"PCA_{variant}"
            modes = [md for md in ("tuned", "fixed") if not is_done("pca", method=method, k=k, alpha_mode=md)]
            if not modes:
                continue
            t0 = time.time()
            tk = tune[f"{variant}_{k}"]
            B = basis(V, nn, variant, k)
            alphas = {"tuned": tk["alpha"], "fixed": ALPHA_H50}
            sol = ridge(stats_in_basis(m.stats, B), sorted({alphas[md] for md in modes}))
            frac = float((w[:k] if variant == "top" else w[nn - k:nn]).sum() / w.sum())
            for md in modes:
                a = alphas[md]
                W, b = to_channels(sol[a][0], B), sol[a][1]
                path = MODELS / f"pca_{variant}_k{k}_{md}.npz"
                dec.save_model(path, W, b, m.mu, m.sd, H)
                block = {f.key: rh.predict(f.counts, W, b) for f in mv}
                batches = (4, 1) if (k == nn and md == "tuned") else (4,)
                res = evaluate_model(f"{method} k={k} α={a:g}", block,
                                     lambda bs, p=path: dec.make_streaming_decoder(p, batch_size=bs), mv, batches)
                row = dict(method=method, k=k, alpha_mode=md, alpha=a,
                           alpha_at_edge=bool(tk["at_edge"]) if md == "tuned" else False,
                           val_r2_calib=tk["val_r2"] if md == "tuned" else tk["cv"][f"{ALPHA_H50:.6g}"],
                           var_frac=frac, n_features=H * k, **res)
                if k == nn and md == "tuned":
                    row["maxdiff_W_vs_ridge_h50"] = float(np.abs(W - h50["W"]).max())
                    row["maxdiff_b_vs_ridge_h50"] = float(np.abs(b - h50["b"]).max())
                    out(f"  check k = {nn} ({variant}): equivalent weights on the channels against ridge_h50, max diff. "
                        f"W {row['maxdiff_W_vs_ridge_h50']:.1e}, b {row['maxdiff_b_vs_ridge_h50']:.1e} "
                        f"(|W| max {np.abs(h50['W']).max():.1e}); R² {res['r2']:.6f} against {R2_H50}")
                add_row("pca", row)
                if k == nn and md == "tuned" and abs(res["r2"] - R2_H50) > 0.02:
                    out(f"STOP: with k = {nn} R² {res['r2']:.4f}, off from {R2_H50} by more than 0.02")
                    sys.exit(2)
            out(f"  {method} k = {k}: {time.time() - t0:.0f} s")
    out(f"phase pca finished in {(time.time() - t_all) / 60:.1f} min")


# --------------------------------------------------------------------------------------------
# Phase fa
# --------------------------------------------------------------------------------------------
def stage_fa():
    t_all = time.time()
    from thesis_h1 import crosssession as xs
    from thesis_h1 import fa_decoder as fd
    _, mv = load()
    out("=" * 100)
    out("FA: the FA_d15_heldin_s0 recipe (crosssession.fit_bundle) for each K, evaluation as in 05_fa_minival.py")
    data = xs.load_data()                         # read only: ~/tesi_h1_work/xs/data.pkl
    frozen_bundle = xs.bundle_path("FA", 0, "heldin", 15)
    xs.XS = WORK / "xs"                           # the new bundles go here, not in ~/tesi_h1_work/xs/models
    for k in K_FA:
        if is_done("fa", method="FA_top", k=k):
            continue
        t0 = time.time()
        b = xs.fit_bundle("FA", 0, xs.heldin(data), "heldin", use_cache=True, dim=k)
        path = MODELS / f"fa_k{k}.npz"
        fd.export(b, data["active"], path)
        fm = fd.FALatentRidge(path)
        block = {f.key: fm.predict(f.counts) for f in mv}
        res = evaluate_model(f"FA K={k}", block, lambda bs, p=path: fd.make_streaming_decoder(p, batch_size=bs), mv,
                             (4, 1) if k == 15 else (4,))
        L = b.model.fa.components_                 # K × 170, on the z-scored channels (total variance = 170)
        row = dict(method="FA_top", k=k, alpha_mode="lodo", alpha=float(b.readout.alpha),
                   alpha_at_edge=bool(b.readout.alpha_at_edge), val_r2_calib=float(b.info["lodo_r2"]),
                   var_frac=float((L ** 2).sum() / L.shape[1]), n_features=H * k, **res)
        if k == 15:
            fz, nw = np.load(FA15_MODEL), np.load(path)
            assert sorted(fz.files) == sorted(nw.files)
            row["maxdiff_vs_frozen_npz"] = float(max(np.abs(fz[a].astype(float) - nw[a].astype(float)).max()
                                                     for a in fz.files))
            with open(frozen_bundle, "rb") as fh:
                b0 = pickle.load(fh)
            row["maxdiff_loadings_vs_frozen"] = float(np.abs(b0.model.fa.components_ - L).max())
            out(f"  check K = 15: refit against frozen, max diff. npz {row['maxdiff_vs_frozen_npz']:.1e}, "
                f"loading {row['maxdiff_loadings_vs_frozen']:.1e}; α {row['alpha']:g} (frozen {b0.readout.alpha:g})")
            fzm = fd.FALatentRidge(FA15_MODEL)
            r0 = evaluate_model("FA K=15 frozen (models/fa15_ridge_h50.npz)", {f.key: fzm.predict(f.counts) for f in mv},
                                lambda bs: fd.make_streaming_decoder(FA15_MODEL, batch_size=bs), mv)
            row["r2_frozen_model"] = r0["r2"]
        add_row("fa", row)
        out(f"  FA K = {k}: α {row['alpha']:g}{' (AT EDGE)' if row['alpha_at_edge'] else ''}, held-in LODO R² "
            f"{row['val_r2_calib']:.4f}, minival {row['r2']:.4f}, {time.time() - t0:.0f} s")
    out(f"phase fa finished in {(time.time() - t_all) / 60:.1f} min")


# --------------------------------------------------------------------------------------------
# Phase channels (optional): ridge on random subsets of channels
# --------------------------------------------------------------------------------------------
def channel_draws():
    rng = np.random.default_rng(SEED_DRAWS)
    return [(n, d, np.sort(rng.choice(C, n, replace=False))) for n in N_CH_SUB for d in range(N_DRAWS)]


def cols_of(S):
    return (np.arange(H)[:, None] * C + np.asarray(S)[None, :]).ravel()   # order of decoding.lagged


def stage_channels():
    t_all = time.time()
    cal, mv = load()
    cut, tr, va = split_80_20(cal)
    draws = channel_draws()
    tune_file = WORK / "tune_channels.json"
    tune = json.loads(tune_file.read_text()) if tune_file.exists() else {}
    todo = [(n, d, S) for n, d, S in draws if f"{n}_{d}" not in tune]
    if todo:
        out("=" * 100)
        out("Channel subsets, choice of α (80/20 per calib file)")
        m = channel_model("tune", cal)
        zs = z_files(m, cal)
        for n, d, S in todo:
            t0 = time.time()
            cols = cols_of(S)
            sol = ridge(stats_subset(m.stats, cols), ALPHAS)
            sol_ch = {}
            for a, (W, b) in sol.items():
                Wc = np.zeros((H * C, 7))
                Wc[cols] = W
                sol_ch[a] = (Wc, b)
            cv = val_r2(sol_ch, zs, cal, cut, va)
            best = max(cv, key=cv.get)
            tune[f"{n}_{d}"] = dict(alpha=best, val_r2=cv[best], at_edge=best in (ALPHAS[0], ALPHAS[-1]),
                                    channels=S.tolist())
            out(f"  {n:3d} channels, draw {d}: α = {best:9.1f}  validation R² {cv[best]:.4f}  {time.time() - t0:.0f} s")
            tune_file.write_text(json.dumps(tune, indent=1))
        del zs, m
    out("=" * 100)
    out("Channel subsets, models on all calib data and evaluation on minival")
    m = channel_model("final", cal)
    rh = dec.RidgeHistory(H)
    rh.mu, rh.sd = m.mu, m.sd
    for n, d, S in draws:
        if is_done("channels", n_channels=n, draw=d):
            continue
        tk = tune[f"{n}_{d}"]
        assert tk["channels"] == S.tolist()
        cols = cols_of(S)
        W, b = ridge(stats_subset(m.stats, cols), [tk["alpha"]])[float(tk["alpha"])]
        Wc = np.zeros((H * C, 7))
        Wc[cols] = W
        path = MODELS / f"chan_n{n}_d{d}.npz"
        dec.save_model(path, Wc, b, m.mu, m.sd, H)
        block = {f.key: rh.predict(f.counts, Wc, b) for f in mv}
        res = evaluate_model(f"{n} channels draw {d}", block,
                             lambda bs, p=path: dec.make_streaming_decoder(p, batch_size=bs), mv)
        add_row("channels", dict(n_channels=n, draw=d, alpha=tk["alpha"], alpha_at_edge=bool(tk["at_edge"]),
                                 val_r2_calib=tk["val_r2"], channels=" ".join(map(str, S)), **res))
    out(f"phase channels finished in {(time.time() - t_all) / 60:.1f} min")


# --------------------------------------------------------------------------------------------
# Report: CSV, checks, figure, summary
# --------------------------------------------------------------------------------------------
def _fig_style():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10, "pdf.fonttype": 42,
                         "svg.fonttype": "none"})
    return plt


def k_half(df_top, target):
    """First "top" k with R² ≥ target, and an estimate by linear interpolation in log k between the two points that bracket it."""
    d = df_top.sort_values("k")
    ks, rs = d.k.to_numpy(), d.r2.to_numpy()
    above = np.flatnonzero(rs >= target)
    if not len(above):
        return None, None
    i = above[0]
    if i == 0:
        return int(ks[0]), float(ks[0])
    lk = np.interp(target, [rs[i - 1], rs[i]], np.log([ks[i - 1], ks[i]]))
    return int(ks[i]), float(np.exp(lk))


def stage_report():
    import pandas as pd
    from thesis_h1 import palette as pal
    df = pd.DataFrame(rows("pca") + rows("fa"))
    order = {"PCA_top": 0, "PCA_bottom": 1, "FA_top": 2}
    df = df.sort_values(["method", "k"], key=lambda s: s.map(order) if s.name == "method" else s)
    main = df[df.alpha_mode != "fixed"].copy()
    main["alpha_selection"] = main.alpha_mode.map({"tuned": "80/20 calib (as ridge_h50)",
                                                   "lodo": "leave-one-day-out held-in (as FA_d15)"})
    cols = ["method", "k", "r2", "r2_sd_days", "var_frac", "alpha", "alpha_selection", "alpha_at_edge",
            "val_r2_calib", "r2_block", "n_features"]
    main[cols].to_csv(OUT / "r2_vs_components.csv", index=False, float_format="%.6g")
    fixed = df[df.alpha_mode == "fixed"]
    fixed[["method", "k", "r2", "r2_sd_days", "alpha", "val_r2_calib"]].to_csv(
        OUT / "r2_vs_components_fixed_alpha.csv", index=False, float_format="%.6g")
    pf = np.load(WORK / "pca_final.npz")
    w, nn = pf["w"], int(pf["nn"])
    vf = pd.DataFrame(dict(k=K_PCA, var_frac_top=[w[:k].sum() / w.sum() for k in K_PCA],
                           var_frac_bottom=[w[nn - k:nn].sum() / w.sum() for k in K_PCA])).sort_values("k")
    vf.to_csv(OUT / "pca_variance_fractions.csv", index=False, float_format="%.6g")

    get = lambda meth, k, col="r2": float(main[(main.method == meth) & (main.k == k)][col].iloc[0])  # noqa: E731
    top = main[main.method == "PCA_top"].sort_values("k")
    bot = main[main.method == "PCA_bottom"].sort_values("k")
    fa = main[main.method == "FA_top"].sort_values("k")
    r175t, r175b, rfa15 = get("PCA_top", nn), get("PCA_bottom", nn), get("FA_top", 15)
    row175 = main[(main.method == "PCA_top") & (main.k == nn)].iloc[0]
    rowfa = main[(main.method == "FA_top") & (main.k == 15)].index[0]
    fa_raw = pd.DataFrame(rows("fa")).set_index("k").loc[15]
    pca_raw = pd.DataFrame(rows("pca"))
    p175 = pca_raw[(pca_raw.method == "PCA_top") & (pca_raw.k == nn) & (pca_raw.alpha_mode == "tuned")].iloc[0]
    dtop = np.diff(top.r2.to_numpy())
    chk = [
        "# Checks",
        "",
        f"1. k = {nn} (all non-null PCs; there are 176 channels, one is silent in the calib data): top {r175t:.6f}, bottom "
        f"{r175b:.6f}, |difference| {abs(r175t - r175b):.1e}. Chosen α: top {get('PCA_top', nn, 'alpha'):g}, bottom "
        f"{get('PCA_bottom', nn, 'alpha'):g}.",
        f"2. k = {nn} against ridge_h50 (0.448332): {r175t:.6f}, difference {r175t - R2_H50:+.1e}; equivalent weights on the "
        f"channels against models/ridge_h50.npz: max diff. W {p175['maxdiff_W_vs_ridge_h50']:.1e}, b "
        f"{p175['maxdiff_b_vs_ridge_h50']:.1e}; batch 1 {p175['r2_batch1']:.6f}. 80/20 validation R² "
        f"{row175.val_r2_calib:.4f} (05_ridge_tune_long.py: 0.3922).",
        f"3. FA K = 15 refit with the recipe: {rfa15:.6f}; frozen model models/fa15_ridge_h50.npz evaluated now: "
        f"{fa_raw['r2_frozen_model']:.6f} (table B: 0.100056); max diff. refit - frozen: npz "
        f"{fa_raw['maxdiff_vs_frozen_npz']:.1e}, loading {fa_raw['maxdiff_loadings_vs_frozen']:.1e}.",
        f"4. Top curve, increments between consecutive k ({', '.join(f'{a}→{b}' for a, b in zip(top.k[:-1], top.k[1:]))}): "
        f"{', '.join(f'{x:+.4f}' for x in dtop)}; "
        + ("monotonically increasing." if (dtop > 0).all() else
           f"not monotonic: largest drop {dtop.min():+.4f}."),
        f"5. Every model: streaming batch 4 = block to the 4th digit; max prediction diff. "
        f"{main.stream_block_maxdiff.max():.1e} (all models).",
        f"6. α at the edge of the grid: "
        + (", ".join(f"{r.method} k={r.k} (α {r.alpha:g})" for r in main[main.alpha_at_edge.astype(bool)].itertuples())
           or "none") + ".",
        "",
        "Note: row index of FA K = 15 in the CSV: %d." % rowfa,
    ]
    (OUT / "checks.md").write_text("\n".join(chk[:-2]) + "\n")
    for line in chk[:-2]:
        out(line)

    # ---- figure
    plt = _fig_style()
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    pal.style_axes(ax)
    cP, cF = pal.color("PCA"), pal.color("FA")
    ax.plot(top.k, top.r2, color=cP, lw=2, marker="o", ms=6, label="PCA, top-k components (largest variance)",
            zorder=3)
    ax.plot(bot.k, bot.r2, color=cP, lw=2, ls=(0, (4, 2)), marker="o", ms=6, mfc="white", mew=1.6,
            label="PCA, bottom-k components (smallest variance)", zorder=3)
    ax.plot(fa.k, fa.r2, color=cF, lw=2, marker="s", ms=6, label="FA, k factors", zorder=3)
    for y, txt in ((R2_H50, f"ridge on all channels ({R2_H50:.3f})"),
                   (R2_FALCON_SAMPLE, f"FALCON example decoder ({R2_FALCON_SAMPLE:.3f})")):
        ax.axhline(y, color=pal.NEUTRAL, lw=1.2, ls=(0, (5, 3)), zorder=2)
        ax.text(4.3, y + 0.006, txt, color=pal.INK2, fontsize=8, va="bottom", ha="left")
    ax.set_xscale("log")
    ticks = [5, 10, 15, 25, 50, 100, nn]
    ax.set_xticks(ticks)
    ax.set_xticklabels([str(t) for t in ticks])
    ax.minorticks_off()
    ax.set_xlim(4.2, 215)
    ax.set_ylim(min(0.0, float(main.r2.min()) - 0.02), 0.5)
    ax.set_xlabel("Number of components (log scale)", color=pal.INK)
    ax.set_ylabel("Velocity R² (minival)", color=pal.INK)
    ax.set_title("Velocity R² vs number of components (minival, FALCON evaluator)", color=pal.INK, loc="left")
    leg = ax.legend(loc="lower right", fontsize=8, frameon=True, framealpha=0.95, edgecolor=pal.GRID)
    for t in leg.get_texts():
        t.set_color(pal.INK)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"r2_vs_components.{ext}", dpi=300)
    plt.close(fig)

    # ---- summary
    half = R2_H50 / 2
    k_first, k_interp = k_half(top, half)
    small = [k for k in sorted(top.k) if k <= 25]
    cmp_small = ", ".join(f"k={k}: top {get('PCA_top', k):.3f} / bottom {get('PCA_bottom', k):.3f}" for k in small)
    bottom_wins = [k for k in sorted(top.k) if get("PCA_bottom", k) > get("PCA_top", k)]
    summ = {"k_half_first": k_first, "k_half_interp": k_interp, "bottom_wins_at": bottom_wins, "cmp_small": cmp_small,
            "fa_vs_top": {int(k): (get("FA_top", k), get("PCA_top", k)) for k in K_FA}}
    (OUT / "summary_numbers.json").write_text(json.dumps(summ, indent=1))
    out(f"half of 0.448 ({half:.3f}): first top k that exceeds it {k_first}, interpolated in log k ≈ {k_interp:.1f}")
    out(f"top against bottom for small k: {cmp_small}; bottom > top for k = {bottom_wins}")
    out("FA against top PCA: " + ", ".join(f"K={k}: FA {a:.3f} / PCA {b:.3f}" for k, (a, b) in summ["fa_vs_top"].items()))


def stage_report_channels():
    import pandas as pd
    from thesis_h1 import palette as pal
    d = pd.DataFrame(rows("channels")).sort_values(["n_channels", "draw"])
    d[["n_channels", "draw", "r2", "r2_sd_days", "alpha", "alpha_at_edge", "val_r2_calib", "r2_block",
       "channels"]].to_csv(OUT / "r2_vs_channels_draws.csv", index=False, float_format="%.6g")
    g = d.groupby("n_channels").r2.agg(r2_mean="mean", r2_std=lambda x: x.std(ddof=1), n_draws="count").reset_index()
    full = pd.DataFrame(dict(n_channels=[C], r2_mean=[R2_H50], r2_std=[np.nan], n_draws=[1]))
    g = pd.concat([g, full], ignore_index=True)
    g.to_csv(OUT / "r2_vs_channels.csv", index=False, float_format="%.6g")
    for r in g.itertuples():
        out(f"  {r.n_channels:3d} channels: R² {r.r2_mean:.4f} ± {r.r2_std:.4f} (n = {r.n_draws})")
    plt = _fig_style()
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    pal.style_axes(ax)
    c = pal.color("ridge_channels")
    jit = (d.draw - (N_DRAWS - 1) / 2) * 0.6
    ax.scatter(d.n_channels + jit, d.r2, s=12, color=c, alpha=0.35, lw=0, zorder=2, label="single random draw")
    gs = g[g.n_channels < C]
    ax.errorbar(gs.n_channels, gs.r2_mean, yerr=gs.r2_std, color=c, lw=2, marker="o", ms=6, capsize=3, zorder=3,
                label=f"mean ± SD over {N_DRAWS} draws")
    ax.plot([C], [R2_H50], marker="D", ms=7, color=c, mfc="white", mew=1.6, ls="none", zorder=3,
            label=f"all {C} channels (ridge_h50, {R2_H50:.3f})")
    ax.axhline(R2_FALCON_SAMPLE, color=pal.NEUTRAL, lw=1.2, ls=(0, (5, 3)), zorder=1)
    ax.text(183, R2_FALCON_SAMPLE + 0.006, f"FALCON example decoder ({R2_FALCON_SAMPLE:.3f})", color=pal.INK2,
            fontsize=8, ha="right")
    ax.set_xticks(N_CH_SUB + [C])
    ax.set_xlim(0, 186)
    ax.set_ylim(0, 0.5)
    ax.set_xlabel("Number of channels (random subsets)", color=pal.INK)
    ax.set_ylabel("Velocity R² (minival)", color=pal.INK)
    ax.set_title("Velocity R² vs number of channels (minival, FALCON evaluator)", color=pal.INK, loc="left")
    leg = ax.legend(loc="lower right", fontsize=8, frameon=True, framealpha=0.95, edgecolor=pal.GRID)
    for t in leg.get_texts():
        t.set_color(pal.INK)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"r2_vs_channels.{ext}", dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["pca", "fa", "report", "channels", "report_channels"])
    stage = ap.parse_args().stage
    out(f"##### phase {stage}, {time.strftime('%Y-%m-%d %H:%M')}, temporary files in {TMP}")
    {"pca": stage_pca, "fa": stage_fa, "report": stage_report, "channels": stage_channels,
     "report_channels": stage_report_channels}[stage]()
