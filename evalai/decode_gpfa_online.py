"""Decoder for FALCON H1 (EvalAI): online (causal) GPFA + ridge on the latents with 1000 ms of history.

Model: models/gpfa25_online_ridge_h50.npz, produced by scripts/evalai_clock_gpfa.py (model phases, alpha, fit):
  - frozen d = 25 GPFA of the cross-session protocol (GPFA_d25_heldin_s0: square root of the counts of the 170 active channels, 13 held-in
    calib files), no refit;
  - latents = causal posterior mean E[x_t | y_{t-W+1..t}] (clock_gpfa_online.gpfa_online), window
    W = default_window, orthonormalized with the SVD of C (the transformation is already inside the filters);
  - ridge on the latents with lags 0..H-1 (H = 50 bins = 1000 ms), latents z-scored with the calib mean and sd,
    alpha chosen with the 80/20 split per file as for ridge_h50.

No clock. For H1 the falcon_challenge evaluator is "continual" (evaluator.py: self.continual = True for h1, m1,
m2), so it never calls on_done or observe: the decoder only receives reset(dataset_tags) at the start of each group of
files and predict(counts) at every 20 ms bin, including the bins between one trial and the next. StreamingClockGPFA is used with
variant="gpfa" and new_trial() is never called.

Usage (like FALCON's decoder_demos/sklearn_sample.py). The image needs, in /workspace: decode.py (this file),
clock_gpfa_online.py and data/decoder.npz.
  python evalai/decode_gpfa_online.py --evaluation local --model-path models/gpfa25_online_ridge_h50.npz --phase minival
"""
import argparse
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
for _p in (HERE, HERE.parent):                    # clock_gpfa_online.py: next to this file (image) or in the repo root
    if (_p / "clock_gpfa_online.py").exists():
        sys.path.insert(0, str(_p))
        break
import clock_gpfa_online as cg                    # noqa: E402

from falcon_challenge.config import FalconConfig, FalconTask   # noqa: E402
from falcon_challenge.interface import BCIDecoder              # noqa: E402


def ridge_pipeline(mu, sd, W, b, H, alpha):
    """sklearn pipeline (StandardScaler + Ridge) with the numbers of the model, in the form read by StreamingClockGPFA.

    The z-score is per latent (same mean and sd for all lags, like the per-channel standardization of ridge_h50).
    """
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    sc = StandardScaler()
    sc.mean_, sc.scale_ = np.tile(mu, H), np.tile(sd, H)
    sc.var_, sc.n_features_in_, sc.n_samples_seen_ = sc.scale_ ** 2, sc.mean_.size, 1
    r = Ridge(alpha=alpha)
    r.coef_, r.intercept_, r.n_features_in_ = np.asarray(W, float).T.copy(), np.asarray(b, float).copy(), sc.mean_.size
    return make_pipeline(sc, r)


def load_model(path):
    """Read the npz and recompute the causal filters (once, at startup)."""
    m = np.load(path)
    params = {k: m[k] for k in ("C", "d", "R", "gamma", "eps")}
    T = m["orth"]                                 # x_orth = T x, T = diag(s) Vᵀ from the SVD of C
    t0 = time.time()
    filters = [np.einsum("ij,kjy->kiy", T, F) for F in cg.causal_gpfa_filters(params, int(m["window"]))]
    H = int(m["H"])
    return dict(active=m["active"].astype(int), params=params, filters=filters, n_lags=H - 1,
                model=ridge_pipeline(m["lat_mu"], m["lat_sd"], m["W"], m["b"], H, float(m["alpha"])),
                window=int(m["window"]), filter_seconds=time.time() - t0)


def batch_predict(mdl, counts):
    """The same decoder on a whole file (in one block): online latents, lags, ridge."""
    Y = np.sqrt(np.asarray(counts, float)[:, mdl["active"]])
    X = cg.gpfa_online(Y, mdl["params"], filters=mdl["filters"])       # a single segment: the file is continuous
    return mdl["model"].predict(cg.design_matrix("gpfa", X, n_lags=mdl["n_lags"]))


class GPFAOnlineDecoder(BCIDecoder):
    def __init__(self, task_config, model_path, batch_size=1):
        super().__init__(task_config=task_config, batch_size=batch_size)
        self.m = load_model(model_path)
        self.dec = [cg.StreamingClockGPFA(self.m["params"], self.m["model"], variant="gpfa",
                                          filters=self.m["filters"], n_lags=self.m["n_lags"])
                    for _ in range(batch_size)]

    def reset(self, dataset_tags=[""]):
        for d in self.dec:                         # new file: window and lags restart from zero
            d.reset()

    def on_done(self, dones):
        pass                                       # H1 is continual: the evaluator does not call it

    def predict(self, neural_observations):
        x = np.asarray(neural_observations, dtype=np.float64)
        y = np.sqrt(x[:, self.m["active"]])
        return np.stack([self.dec[i].step(y[i]) for i in range(x.shape[0])])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evaluation", type=str, required=True, choices=["local", "remote"])
    ap.add_argument("--model-path", type=str, default="data/decoder.npz")
    ap.add_argument("--split", type=str, choices=["h1"], default="h1")
    ap.add_argument("--phase", choices=["minival", "test"], default="minival")
    ap.add_argument("--batch-size", type=int, default=1)
    args = ap.parse_args()

    from falcon_challenge.evaluator import FalconEvaluator
    decoder = GPFAOnlineDecoder(FalconConfig(task=FalconTask.h1), args.model_path, batch_size=args.batch_size)
    print(f"causal filters: W = {decoder.m['window']} bins, computed in {decoder.m['filter_seconds']:.1f} s", flush=True)

    evaluator = FalconEvaluator(eval_remote=args.evaluation == "remote", split=args.split)
    out = evaluator.evaluate(decoder, phase=args.phase)
    if out is not None:
        print(out)


if __name__ == "__main__":
    main()
