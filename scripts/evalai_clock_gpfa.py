"""Local test of Clock + online GPFA as a velocity decoder on FALCON H1, before deciding whether to submit it.

API (checked by reading falcon_challenge 1.0.2, interface.py and evaluator.py): for H1 the evaluator is "continual"
(self.continual = True for h1, m1, m2) and never calls on_done or observe; trial_change is read by load_nwb but
does not reach the decoder. The decoder only receives reset(dataset_tags) at the start of each group of files and
predict(counts) at each bin. Hence no clock: we evaluate (a) ridge on channels and (b) online GPFA.

Phases (from the repo root):
  OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=$HOME/thesis_h1_work/falcon-challenge:src \
      python scripts/evalai_clock_gpfa.py --phase model|latents|alpha|fit|minival

  model    parameters of the frozen cross-session GPFA (GPFA_d25_heldin_s0: d = 25, square root of the counts of the 170
           active channels, 13 held-in calib files) and active channels (xs/data.pkl), no refit; W = default_window,
           τ of each latent -> window_tau.csv
  latents  orthonormalized online latents of each held-in calib and minival file, whole file = one segment (as the
           evaluator sees it, serving files from start to end)
  alpha    ridge on the latents with lags 0..49 (1000 ms), same α choice as ridge_h50 (scripts/05_ridge_tune_long.py):
           80/20 per file, z-score per latent on the first 80%, fit on the eval_mask bins of the first 80%, R² as in FALCON
           (mean over days) on the eval_mask bins of the last 20%, grid 10^0..10^6 (13 values) -> alpha.csv
  fit      final ridge on all held-in calib data with the chosen α -> models/gpfa25_online_ridge_h50.npz
  minival  official FALCON evaluator: (a) models/ridge_h50.npz (must give 0.448), (b) wrapper
           evalai/decode_gpfa_online.py with batch 1 and 4; block + our evaluator; streaming = block check
           (StreamingClockGPFA bin by bin on one session, max |diff| < 1e-9); table in results/evalai_clock_gpfa/

Inputs and cache in ~/thesis_h1_work/evalai_clock_gpfa/. The registry h1_runs.csv is not touched. Nothing is submitted to EvalAI.
"""
import argparse
import importlib.util
import json
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))                       # clock_gpfa_online.py is in the repo root
sys.path.insert(0, str(ROOT / "src"))
import clock_gpfa_online as cg                      # noqa: E402
from thesis_h1 import decoding as dec                 # noqa: E402

HOME = Path.home()
RAW = HOME / "000954"
WORK = HOME / "thesis_h1_work" / "evalai_clock_gpfa"
OUT = ROOT / "results" / "evalai_clock_gpfa"
GPFA_PKL = HOME / "thesis_h1_work" / "xs" / "models" / "GPFA_d25_heldin_s0.pkl"
XS_DATA = HOME / "thesis_h1_work" / "xs" / "data.pkl"
RIDGE_H50 = ROOT / "models" / "ridge_h50.npz"
MODEL = ROOT / "models" / "gpfa25_online_ridge_h50.npz"
WRAPPER = ROOT / "evalai" / "decode_gpfa_online.py"
H = 50                                              # history: 50 bins = 1000 ms (ridge_h50)
ALPHAS = np.logspace(0, 6, 13)                      # grid of scripts/05_ridge_tune_long.py
BIN_MS = 20.0
SESS_CHECK = "19250108T110520"                      # thesis reference session, for the streaming = block check
T0 = time.time()


def log(m):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {m}", flush=True)


def wrapper():
    spec = importlib.util.spec_from_file_location("decode_gpfa_online", WRAPPER)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# --------------------------------------------------------------------------------------------
# model: frozen GPFA parameters, active channels, window, orthonormalized filters
# --------------------------------------------------------------------------------------------
def phase_model():
    with open(GPFA_PKL, "rb") as f:
        b = pickle.load(f)
    p = b.model.params
    params = dict(C=np.asarray(p["C"], float), d=np.asarray(p["d"], float).ravel(),
                  R=np.diag(np.asarray(p["R"], float)).copy(), gamma=np.asarray(p["gamma"], float).ravel(),
                  eps=np.asarray(p["eps"], float).ravel())
    with open(XS_DATA, "rb") as f:
        active = np.asarray(pickle.load(f)["active"], int)
    assert params["C"].shape == (170, 25) and active.size == 170, (params["C"].shape, active.size)
    _, s, Vt = np.linalg.svd(params["C"], full_matrices=False)
    orth = np.diag(s) @ Vt                          # = cg.orthonormalize: x_orth = orth @ x
    W = cg.default_window(params)
    WORK.mkdir(parents=True, exist_ok=True)
    np.savez(WORK / "gpfa.npz", active=active, orth=orth, window=W, **params)
    tau = cg.tau_from_gamma(params["gamma"], BIN_MS)
    df = pd.DataFrame(dict(latent=np.arange(25), tau_ms=tau)).sort_values("tau_ms")
    df["W_bin"], df["W_ms"] = W, W * BIN_MS
    df["W_3tau_without_ceiling_bin"] = int(np.ceil(3 * tau.max() / BIN_MS))
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "window_tau.csv", index=False)
    log(f"GPFA {GPFA_PKL.name}: d = {params['C'].shape[1]}, channels {active.size}, sessions {b.info['sessions']}, "
        f"{b.info['n_windows']} windows of 1 s")
    log(f"W = {W} bins = {W * BIN_MS:.0f} ms (3·τ_max = {3 * tau.max():.0f} ms without ceiling); "
        f"τ = {np.round(np.sort(tau)).astype(int).tolist()} ms")
    t = time.time()
    cg.causal_gpfa_filters(params, W)
    log(f"causal filters L = 1..{W}: {time.time() - t:.1f} s")


def load_gpfa():
    g = np.load(WORK / "gpfa.npz")
    params = {k: g[k] for k in ("C", "d", "R", "gamma", "eps")}
    filters = [np.einsum("ij,kjy->kiy", g["orth"], F) for F in cg.causal_gpfa_filters(params, int(g["window"]))]
    return g["active"].astype(int), params, filters, int(g["window"])


# --------------------------------------------------------------------------------------------
# online latents of each file
# --------------------------------------------------------------------------------------------
def phase_latents():
    active, params, filters, W = load_gpfa()
    for split in ("held-in-calib", "held-in-minival"):
        files = dec.load_split(RAW, split)
        out = []
        for f in files:
            X = cg.gpfa_online(np.sqrt(f.counts.astype(float)[:, active]), params, filters=filters)
            out.append(dict(key=f.key, day=f.day, sid=f.path.stem.split("_ses-")[-1], X=X, vel=f.vel, mask=f.mask))
        with open(WORK / f"latents_{split}.pkl", "wb") as fh:
            pickle.dump(out, fh)
        log(f"{split}: {len(out)} files, {sum(len(o['X']) for o in out)} bins, W = {W}")


def load_latents(split):
    with open(WORK / f"latents_{split}.pkl", "rb") as fh:
        return pickle.load(fh)


def design(X, mu, sd):
    """Lags 0..H-1 of the latents (zeros before the start of the file, as in add_lags), then z-score per latent."""
    return (cg.add_lags(X, H - 1) - np.tile(mu, H)) / np.tile(sd, H)


def ridge_stats(files, mu, sd, rows):
    p = 25 * H
    G, g, sx, sy, n = np.zeros((p, p)), np.zeros((p, 7)), np.zeros(p), np.zeros(7), 0
    for f, r in zip(files, rows):
        D, y = design(f["X"], mu, sd)[r], f["vel"][r]
        G += D.T @ D; g += D.T @ y; sx += D.sum(0); sy += y.sum(0); n += len(y)
    mx, my = sx / n, sy / n
    return G - n * np.outer(mx, mx), g - n * np.outer(mx, my), mx, my


def ridge_solve(C, r, mx, my, alphas):
    """As decoding.RidgeHistory.solve: one decomposition, then every α (free, unpenalized intercept)."""
    w, V = np.linalg.eigh(C)
    Vr = V.T @ r
    out = {}
    for a in alphas:
        Wr = V @ (Vr / (w + a)[:, None])
        out[a] = (Wr, my - mx @ Wr)
    return out


# --------------------------------------------------------------------------------------------
# alpha: 80/20 per file as for ridge_h50
# --------------------------------------------------------------------------------------------
def phase_alpha():
    cal = load_latents("held-in-calib")
    cut = [int(0.8 * len(f["X"])) for f in cal]
    tr = [np.r_[np.ones(c, bool), np.zeros(len(f["X"]) - c, bool)] & f["mask"] for f, c in zip(cal, cut)]
    va = [~np.r_[np.ones(c, bool), np.zeros(len(f["X"]) - c, bool)] & f["mask"] for f, c in zip(cal, cut)]
    Xtr = np.concatenate([f["X"][:c] for f, c in zip(cal, cut)])
    mu, sd = Xtr.mean(0), Xtr.std(0)
    sol = ridge_solve(*ridge_stats(cal, mu, sd, tr), ALPHAS)
    Dva = [design(f["X"], mu, sd)[c:] for f, c in zip(cal, cut)]
    rows = []
    for a, (Wr, b) in sol.items():
        preds = [np.r_[np.zeros((c, 7)), D @ Wr + b] for D, c in zip(Dva, cut)]
        r2, per_day, per_dof = dec.evaluate([f["day"] for f in cal], [f["vel"] for f in cal], preds, va)
        rows.append(dict(alpha=a, log10_alpha=np.log10(a), r2_val=r2, **{f"r2_{k}": v for k, v in per_day.items()},
                          **{f"r2_{d}": v for d, v in zip(dec.DOF, per_dof.mean(0))}))
        log(f"α = 10^{np.log10(a):.1f}: R² validation {r2:.4f}")
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "alpha.csv", index=False)
    best = df.loc[df.r2_val.idxmax()]
    edge = best.alpha in (ALPHAS.min(), ALPHAS.max())
    log(f"chosen α 10^{best.log10_alpha:.1f} = {best.alpha:g}, validation R² {best.r2_val:.4f}"
        f"{' (AT THE EDGE OF THE GRID)' if edge else ''}; ridge_h50 on channels: α 10^4.5, R² 0.3922 (run05/tune_long.log)")


# --------------------------------------------------------------------------------------------
# final fit on all calib data
# --------------------------------------------------------------------------------------------
def phase_fit():
    cal = load_latents("held-in-calib")
    al = pd.read_csv(OUT / "alpha.csv")
    best = al.loc[al.r2_val.idxmax()]
    alpha = float(ALPHAS[np.argmin(np.abs(np.log10(ALPHAS) - best.log10_alpha))])
    Xall = np.concatenate([f["X"] for f in cal])
    mu, sd = Xall.mean(0), Xall.std(0)
    (Wr, b), = ridge_solve(*ridge_stats(cal, mu, sd, [f["mask"] for f in cal]), [alpha]).values()
    g = np.load(WORK / "gpfa.npz")
    np.savez(MODEL, active=g["active"], C=g["C"], d=g["d"], R=g["R"], gamma=g["gamma"], eps=g["eps"], orth=g["orth"],
             window=g["window"], lat_mu=mu, lat_sd=sd, W=Wr, b=b, H=H, alpha=alpha, r2_val=float(best.r2_val),
             alphas=ALPHAS, r2_val_grid=al.r2_val.values,
             origin=np.array(f"GPFA {GPFA_PKL.stem} (no refit) + ridge H={H}, alpha 80/20 per file as for ridge_h50; "
                              f"scripts/evalai_clock_gpfa.py"))
    # check: the wrapper reloaded from the npz reproduces the ridge computed here on a calib file
    w = wrapper()
    mdl = w.load_model(MODEL)
    f0 = dec.load_file(RAW / "sub-HumanPitt-held-in-calib" / f"sub-HumanPitt-held-in-calib_ses-{cal[2]['sid']}.nwb")
    assert f0.key == cal[2]["key"], (f0.key, cal[2]["key"])
    p_wr = w.batch_predict(mdl, f0.counts)
    p_here = design(cal[2]["X"], mu, sd) @ Wr + b
    log(f"saved {MODEL.relative_to(ROOT)}: α = {alpha:g}, {Wr.shape[0]} weights × 7; wrapper = fit on {cal[2]['key']}: "
        f"max |diff| {np.abs(p_wr - p_here).max():.1e}; filters in the wrapper in {mdl['filter_seconds']:.1f} s")


# --------------------------------------------------------------------------------------------
# minival: official evaluator, block, streaming = block check, table
# --------------------------------------------------------------------------------------------
def evaluate_official(decoder, name, mv):
    from falcon_challenge.evaluator import FalconEvaluator
    os.environ["EVAL_DATA_PATH"] = str(HOME)                              # -> ~/000954/*minival/*.nwb
    os.environ["PREDICTION_PATH_LOCAL"] = str(WORK / f"pred_{name}.pkl")
    os.environ["GT_PATH"] = str(WORK / f"gt_{name}.pkl")
    t = time.time()
    res = FalconEvaluator(eval_remote=False, split="h1", dataloader_workers=0).evaluate(decoder, phase="minival")
    off = res["result"][0]["minival_split_h1"]
    preds = pickle.load(open(WORK / f"pred_{name}.pkl", "rb"))["h1"]
    stream = {f.key: np.asarray(preds[f.key])[: len(f.counts)] for f in mv}
    log(f"{name}: official evaluator {float(off['Held In R2 Mean']):.6f} in {time.time() - t:.0f} s")
    return off, stream


def phase_minival():
    import falcon_challenge
    from importlib.metadata import version
    mv = dec.load_split(RAW, "held-in-minival")
    days = [f.day for f in mv]
    rows, ctrl = [], {}
    log(f"falcon_challenge {version('falcon_challenge')} from {Path(falcon_challenge.__file__).parent}; {len(mv)} minival files")

    # (a) ridge on channels, the ridge_h50 model as it is
    m = np.load(RIDGE_H50)
    rh = dec.RidgeHistory(int(m["H"])); rh.mu, rh.sd = m["mu"], m["sd"]
    blk_a = {f.key: rh.predict(f.counts, m["W"], m["b"]) for f in mv}
    off_a, st_a = evaluate_official(dec.make_streaming_decoder(RIDGE_H50, batch_size=1), "a_ridge_channels_b1", mv)
    ctrl["a_stream_vs_block"] = max(np.abs(st_a[k] - blk_a[k]).max() for k in blk_a)

    # (b) online GPFA + ridge on the latents, with the EvalAI wrapper
    w = wrapper()
    from falcon_challenge.config import FalconConfig, FalconTask
    mdl = w.load_model(MODEL)
    blk_b = {f.key: w.batch_predict(mdl, f.counts) for f in mv}
    offs_b = {}
    for B in (1, 4):
        offs_b[B], st = evaluate_official(w.GPFAOnlineDecoder(FalconConfig(task=FalconTask.h1), MODEL, batch_size=B),
                                         f"b_gpfa_online_b{B}", mv)
        ctrl[f"b_stream_b{B}_vs_block"] = max(np.abs(st[k] - blk_b[k]).max() for k in blk_b)

    # check of point 4: StreamingClockGPFA bin by bin = batch version, on the reference session
    f = next(f for f in mv if SESS_CHECK in f.path.stem)
    Y = np.sqrt(f.counts.astype(float)[:, mdl["active"]])
    X = cg.gpfa_online(Y, mdl["params"], filters=mdl["filters"])
    _, pred_batch = cg.evaluate(mdl["model"], "gpfa", X, None, f.vel, n_lags=mdl["n_lags"])
    sc = cg.StreamingClockGPFA(mdl["params"], mdl["model"], variant="gpfa", filters=mdl["filters"], n_lags=mdl["n_lags"])
    pred_stream = np.stack([sc.step(y) for y in Y])
    ctrl["point4_session"] = f"{f.key} ({SESS_CHECK}), {len(Y)} bins"
    ctrl["point4_max_abs_diff"] = float(np.abs(pred_stream - pred_batch).max())
    ctrl["point4_ok"] = bool(ctrl["point4_max_abs_diff"] < 1e-9)
    log(f"point 4: StreamingClockGPFA bin by bin vs batch on {f.key}: max |diff| {ctrl['point4_max_abs_diff']:.2e}")

    al = pd.read_csv(OUT / "alpha.csv")
    for name, blk, off, desc in (
            ("a) ridge on channels", blk_a, off_a, "176 z-scored channels, 1000 ms history, α 10^4.5 (models/ridge_h50.npz)"),
            ("b) GPFA online", blk_b, offs_b[1], f"frozen GPFA d = 25, 170 √ channels, W = {mdl['window']} bins, ridge "
                                                 f"on latents 1000 ms, α {float(np.load(MODEL)['alpha']):g}")):
        r2, per_day, per_dof = dec.evaluate(days, [f.vel for f in mv], [blk[f.key] for f in mv], [f.mask for f in mv])
        rows.append(dict(variant=name, description=desc, r2_minival=float(off["Held In R2 Mean"]),
                          r2_std_days=float(off["Held In R2 Std."]), r2_ours_evaluator_block=r2,
                          normalized_latency=float(off["Normalized Latency"]),
                          **{f"r2_{k}": v for k, v in per_day.items()},
                          **{f"r2_{d}": v for d, v in zip(dec.DOF, per_dof.mean(0))}))
    rows[0]["r2_val_calib_80_20"] = 0.3922                                         # run05/tune_long.log
    rows[1]["r2_val_calib_80_20"] = float(np.load(MODEL)["r2_val"])
    rows[1]["r2_minival_batch4"] = float(offs_b[4]["Held In R2 Mean"])
    for v in ("c) clock + online GPFA", "d) clock alone"):
        rows.append(dict(variant=v, description="not evaluable: for H1 the evaluator does not pass trial boundaries"))
    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "table.csv", index=False)
    ctrl.update(a_official=float(off_a["Held In R2 Mean"]), b_official_b1=float(offs_b[1]["Held In R2 Mean"]),
                b_official_b4=float(offs_b[4]["Held In R2 Mean"]), filters_seconds=mdl["filter_seconds"])
    (OUT / "checks.json").write_text(json.dumps(ctrl, indent=1, ensure_ascii=False) + "\n")

    ft = pd.read_csv(OUT / "window_tau.csv")
    gg = sorted(per_day)
    md = ["# Clock + online GPFA as a velocity decoder on FALCON H1 (minival, local test)", "",
          "R² as in the official FALCON evaluator (`Held In R2 Mean`: variance-weighted R² on the eval_mask bins, per "
          "day, mean over the 6 held-in days), falcon_challenge 1.0.2 evaluator in streaming, batch 1. Nothing was submitted to EvalAI.", "",
          "**API**: for H1 the evaluator is *continual*: it never calls `on_done` or `observe`. The decoder only receives "
          "`reset(dataset_tags)` at the start of each group of files and `predict(counts)` at every 20 ms bin, inter-trial bins "
          "included. No trial boundaries, so c) and d) cannot be evaluated.", "",
          f"**GPFA**: GPFA_d25_heldin_s0 from the cross-session protocol, reused without refit (d = 25, √counts of the 170 active channels, "
          f"13 held-in calib files, fitted on 1 s windows at the start of the movement phases). Online latents with "
          f"`gpfa_online` and `default_window`: W = {int(ft.W_bin[0])} bins = {int(ft.W_ms[0])} ms (default_window ceiling; "
          f"3·τ_max would be {int(ft.W_3tau_without_ceiling_bin[0])} bins). τ from {ft.tau_ms.min():.0f} to {ft.tau_ms.max():.0f} ms "
          f"(median {ft.tau_ms.median():.0f} ms; {(ft.tau_ms > 1000).sum()} latents above 1 s). Orthonormalized latents, "
          f"whole file = one segment.", "",
          "**Ridge on the latents**: lags 0..49 (1000 ms), z-score per latent, fit on the eval_mask bins; α chosen as for "
          "ridge_h50 (80/20 per file, grid 10^0..10^6).", "",
          "| Variant | minival R² | SD over days | " + " | ".join(gg) + " | calib 80/20 validation R² |",
          "| --- | --- | --- | " + " | ".join("---" for _ in gg) + " | --- |"]
    for r in rows[:2]:
        md.append(f"| {r['variant']} | **{r['r2_minival']:.4f}** | {r['r2_std_days']:.4f} | "
                  + " | ".join(f"{r[f'r2_{k}']:.3f}" for k in gg) + f" | {r['r2_val_calib_80_20']:.4f} |")
    for r in rows[2:]:
        md.append(f"| {r['variant']} | not evaluable | | " + " | ".join("" for _ in gg) + " | |")
    md += ["", "R² per degree of freedom (mean over days): " + "; ".join(
        f"{r['variant']}: " + ", ".join(f"{d} {r[f'r2_{d}']:.3f}" for d in dec.DOF) for r in rows[:2]) + ".", "",
           "**Checks**",
           f"- a) official evaluator {ctrl['a_official']:.6f} (expected 0.448332); streaming − block max |diff| "
           f"{ctrl['a_stream_vs_block']:.1e}.",
           f"- b) batch 1 {ctrl['b_official_b1']:.6f}, batch 4 {ctrl['b_official_b4']:.6f}; wrapper in streaming − block "
           f"max |diff| {ctrl['b_stream_b1_vs_block']:.1e} (batch 1), {ctrl['b_stream_b4_vs_block']:.1e} (batch 4).",
           f"- Point 4: `StreamingClockGPFA` bin by bin against the batch version (`gpfa_online` + `design_matrix` + ridge) "
           f"on {ctrl['point4_session']}: max |diff| {ctrl['point4_max_abs_diff']:.1e} "
           f"({'< 1e-9, ok' if ctrl['point4_ok'] else 'NOT < 1e-9'}).",
           f"- Causal filters recomputed at wrapper start-up in {ctrl['filters_seconds']:.0f} s."]
    (OUT / "table.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=("model", "latents", "alpha", "fit", "minival"), required=True)
    {"model": phase_model, "latents": phase_latents, "alpha": phase_alpha, "fit": phase_fit,
     "minival": phase_minival}[ap.parse_args().phase]()
