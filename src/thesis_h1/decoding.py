"""Velocity decoder for FALCON H1 (notebook 05).

Three parts:

- loading with the official FALCON loader (`load_nwb`), so that the bins are aligned with the
  benchmark: bin k contains the spikes in [t_k - 20 ms, t_k) and is paired with the velocity at time t_k;
- causal ridge with history: to predict the velocity at bin k it uses the counts of bins k, k-1, ..., k-H+1;
- metric as in the FALCON evaluator: "variance-weighted" R² on the bins with eval_mask only, computed
  per recording day (files of the same day are concatenated) and then averaged over days.

The `StreamingRidge` decoder receives one bin at a time, as in the official evaluator and on EvalAI.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

N_CH = 176                     # H1 channels
DOF = ["tx", "ty", "tz", "rx", "g1", "g2", "g3"]  # the 7 velocity degrees of freedom (FALCON H1)
BIN_MS = 20                    # bin duration


# --------------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------------
@dataclass
class FalconFile:
    path: Path                 # NWB file
    day: str                   # recording day according to FALCON (S0 ... S5)
    key: str                   # file name according to FALCON (e.g. S0_set_1)
    counts: np.ndarray         # counts, bins × 176
    vel: np.ndarray            # velocity, bins × 7
    mask: np.ndarray           # eval_mask, bins (True = evaluated bin)


def load_file(path) -> FalconFile:
    """Read a file with the official FALCON loader."""
    from falcon_challenge.config import FalconConfig, FalconTask
    from falcon_challenge.dataloaders import load_nwb
    path = Path(path)
    counts, vel, _, mask = load_nwb(path, dataset=FalconTask.h1)   # same bins as the benchmark
    key = FalconConfig(FalconTask.h1).hash_dataset(path.stem)      # e.g. "S0_set_1"
    return FalconFile(path, key.split("_")[0], key, counts.astype(np.float32),
                      vel.astype(np.float64), mask.astype(bool))


def load_split(root, split: str) -> list[FalconFile]:
    """All files of a split ("held-in-calib", "held-in-minival", ...), in order."""
    files = sorted(Path(root).glob(f"sub-HumanPitt-{split}/*.nwb"))
    return [load_file(f) for f in files]


# --------------------------------------------------------------------------------------------
# Causal history
# --------------------------------------------------------------------------------------------
def lagged(z: np.ndarray, H: int) -> np.ndarray:
    """Matrix with the history: row k = [z_k, z_{k-1}, ..., z_{k-H+1}] (most recent bin first).

    Bins before the start of the file are 0, i.e. the channel mean (z is already z-scored):
    this is also what the streaming decoder sees when it starts.
    """
    n, c = z.shape
    out = np.zeros((n, c * H), dtype=np.float32)
    for j in range(H):                                   # j = lag in bins
        out[j:, j * c:(j + 1) * c] = z[: n - j]
    return out


# --------------------------------------------------------------------------------------------
# Ridge with many alpha values at once
# --------------------------------------------------------------------------------------------
class RidgeHistory:
    """Ridge on the causal history, fitted by accumulating X'X file by file (without keeping X in memory).

    A single pass over the data with the maximum history H_max is enough for all shorter histories:
    the columns of history H are the first H × 176 of those with H_max.
    """

    def __init__(self, H_max: int):
        self.H_max = H_max

    # per-channel standardization (not per column: lags of the same channel have the same scale)
    def fit_scaler(self, counts_list):
        allc = np.concatenate(counts_list)
        self.mu = allc.mean(0)
        self.sd = allc.std(0)
        self.sd[self.sd == 0] = 1.0
        return self

    def z(self, counts):
        return (counts - self.mu) / self.sd

    def accumulate(self, counts_list, vel_list, rows_list):
        """Sum X'X, X'y and means over the chosen bins (rows) of each file."""
        p = N_CH * self.H_max
        XtX = np.zeros((p, p)); Xty = np.zeros((p, 7)); sx = np.zeros(p); sy = np.zeros(7); n = 0
        for counts, vel, rows in zip(counts_list, vel_list, rows_list):
            X = lagged(self.z(counts), self.H_max)[rows].astype(np.float64)
            y = vel[rows]
            XtX += X.T @ X; Xty += X.T @ y; sx += X.sum(0); sy += y.sum(0); n += len(y)
        self.stats = dict(XtX=XtX, Xty=Xty, sx=sx, sy=sy, n=n)
        return self

    def solve(self, H: int, alphas):
        """Coefficients for history H and for each alpha: dict alpha -> (W, b)."""
        s = self.stats; p = N_CH * H; n = s["n"]
        mx = s["sx"][:p] / n; my = s["sy"] / n
        C = s["XtX"][:p, :p] - n * np.outer(mx, mx)       # X'X with centered X (free intercept)
        r = s["Xty"][:p] - n * np.outer(mx, my)           # centered X'y
        w, V = np.linalg.eigh(C)                          # one decomposition, then each alpha is cheap
        Vr = V.T @ r
        out = {}
        for a in alphas:
            W = V @ (Vr / (w + a)[:, None])
            out[a] = (W, my - mx @ W)
        return out

    def predict(self, counts, W, b):
        H = W.shape[0] // N_CH
        return lagged(self.z(counts), H).astype(np.float64) @ W + b


# --------------------------------------------------------------------------------------------
# Metric (as in the FALCON evaluator)
# --------------------------------------------------------------------------------------------
def r2_dof(y, yhat):
    """R² of each degree of freedom."""
    ss_res = ((y - yhat) ** 2).sum(0)
    ss_tot = ((y - y.mean(0)) ** 2).sum(0)
    return 1 - ss_res / ss_tot


def r2_vw(y, yhat):
    """Variance-weighted R² = sklearn r2_score(multioutput='variance_weighted')."""
    ss_res = ((y - yhat) ** 2).sum()
    ss_tot = ((y - y.mean(0)) ** 2).sum()
    return 1 - ss_res / ss_tot


def evaluate(days, ys, preds, masks):
    """R² as in FALCON: per day (files of the same day concatenated), only bins with eval_mask.

    Returns: mean over days, dict day -> R², and R² per DoF per day (array days × 7).
    """
    per_day, per_dof = {}, {}
    for d in sorted(set(days)):
        idx = [i for i, dd in enumerate(days) if dd == d]
        y = np.concatenate([ys[i][masks[i]] for i in idx])
        yh = np.concatenate([preds[i][masks[i]] for i in idx])
        per_day[d] = r2_vw(y, yh)
        per_dof[d] = r2_dof(y, yh)
    return float(np.mean(list(per_day.values()))), per_day, np.array([per_dof[d] for d in sorted(per_dof)])


# --------------------------------------------------------------------------------------------
# Streaming decoder for the official evaluator (and for EvalAI)
# --------------------------------------------------------------------------------------------
def save_model(path, W, b, mu, sd, H):
    np.savez(path, W=W, b=b, mu=mu, sd=sd, H=H)


def make_streaming_decoder(model_path, batch_size: int = 1):
    """Create the decoder that receives one bin at a time (FALCON BCIDecoder interface)."""
    from falcon_challenge.config import FalconConfig, FalconTask
    from falcon_challenge.interface import BCIDecoder

    class StreamingRidge(BCIDecoder):
        def __init__(self, task_config, model_path, batch_size=1):
            super().__init__(task_config=task_config, batch_size=batch_size)
            m = np.load(model_path)
            self.W, self.b, self.mu, self.sd, self.H = m["W"], m["b"], m["mu"], m["sd"], int(m["H"])
            self.reset()

        def reset(self, dataset_tags=[""]):
            # buffer of the last H z-scored bins, for each element of the batch; 0 = channel mean
            self.buf = np.zeros((self.batch_size, self.H, N_CH))

        def on_done(self, dones):
            pass                                       # H1 is "continuous": no reset between trials

        def predict(self, neural_observations):
            x = np.asarray(neural_observations, dtype=np.float64)
            B = x.shape[0]
            if B < self.batch_size:                    # last, smaller batch: pad with zeros
                x = np.pad(x, ((0, self.batch_size - B), (0, 0)))
            self.buf = np.roll(self.buf, 1, axis=1)    # shift the history by one bin
            self.buf[:, 0] = (x - self.mu) / self.sd   # the bin that just arrived goes in position 0

            out = self.buf.reshape(self.batch_size, -1) @ self.W + self.b
            return out[:B]

    return StreamingRidge(FalconConfig(task=FalconTask.h1), model_path, batch_size)
