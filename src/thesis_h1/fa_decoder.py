"""Ridge on the FA latents (table B): the frozen cross-session FA model as a causal decoder for FALCON H1.

The model is `FA_d15_heldin_s0` of crosssession.py, as it is: FA with K = 15 on the 170 active channels z-scored with
mean and sd of the 13 held-in sessions, latents = posterior mean, standardized with the held-in moments, ridge readout
with a history of 50 bins (1000 ms), alpha chosen leave-one-day-out on the held-in sessions, fitted on the whole held-in calibration set.

Every step is linear and uses only the current bin, so the decoder works one bin at a time:
  z   = (counts[active] − mu) / sd
  lat = (z − fa_mean) @ P,   P = Ψ⁻¹Wᵀ (I + W Ψ⁻¹ Wᵀ)⁻¹   (= FactorAnalysis.transform)
  s   = (lat − lat_m) / lat_s
  v   = [s_k, s_{k−1}, ..., s_{k−49}] @ W + b   (bins before the start of the file = 0, i.e. the mean)

The readout was trained with the history restarting at every trial; the FALCON evaluator for H1 is "continuous" (it
does not call on_done), so here the history runs through the whole file, as for ridge_h50.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .decoding import lagged

N_CH = 176


def export(bundle, active, path):
    """Save the frozen FA model (crosssession Bundle) to an npz, with only the matrices that are needed."""
    M, rd = bundle.model, bundle.readout
    fa = M.fa
    Wpsi = fa.components_ / fa.noise_variance_
    P = Wpsi.T @ np.linalg.inv(np.eye(len(fa.components_)) + Wpsi @ fa.components_.T)
    np.savez(path, active=np.asarray(active, int), mu=M.mu, sd=M.sd, fa_mean=fa.mean_, P=P, lat_m=rd.m, lat_s=rd.s,
             W=rd.W, b=rd.b, H=int(rd.W.shape[0] // P.shape[1]), alpha=float(rd.alpha),
             lodo_r2=float(bundle.info["lodo_r2"]))
    return Path(path)


class FALatentRidge:
    """The model from the npz, applied to a whole file (in one block)."""

    def __init__(self, path):
        m = np.load(path)
        self.active, self.mu, self.sd, self.fa_mean, self.P = m["active"], m["mu"], m["sd"], m["fa_mean"], m["P"]
        self.lat_m, self.lat_s, self.W, self.b, self.H = m["lat_m"], m["lat_s"], m["W"], m["b"], int(m["H"])
        self.alpha, self.lodo_r2 = float(m["alpha"]), float(m["lodo_r2"])

    def latents(self, counts):
        """Standardized latents (bins × K) from the counts of all 176 channels."""
        z = (np.asarray(counts, np.float64)[:, self.active] - self.mu) / self.sd
        return ((z - self.fa_mean) @ self.P - self.lat_m) / self.lat_s

    def predict(self, counts):
        """Predicted velocities for a whole file, with the history running through the whole file."""
        return lagged(self.latents(counts).astype(np.float32), self.H).astype(np.float64) @ self.W + self.b


def make_streaming_decoder(model_path, batch_size: int = 1):
    """Decoder that receives one bin at a time (FALCON BCIDecoder interface), like StreamingRidge in decoding.py."""
    from falcon_challenge.config import FalconConfig, FalconTask
    from falcon_challenge.interface import BCIDecoder

    class StreamingFALatentRidge(BCIDecoder):
        def __init__(self, task_config, model_path, batch_size=1):
            super().__init__(task_config=task_config, batch_size=batch_size)
            self.m = FALatentRidge(model_path)
            self.K = self.m.P.shape[1]
            self.reset()

        def reset(self, dataset_tags=[""]):
            self.buf = np.zeros((self.batch_size, self.m.H, self.K))    # 0 = latent at the held-in mean

        def on_done(self, dones):
            pass                                                        # H1 is continuous: never called


        def predict(self, neural_observations):
            x = np.asarray(neural_observations, dtype=np.float64)
            B = x.shape[0]
            if B < self.batch_size:
                x = np.pad(x, ((0, self.batch_size - B), (0, 0)))
            self.buf = np.roll(self.buf, 1, axis=1)
            self.buf[:, 0] = self.m.latents(x)
            return (self.buf.reshape(self.batch_size, -1) @ self.m.W + self.m.b)[:B]

    return StreamingFALatentRidge(FalconConfig(task=FalconTask.h1), model_path, batch_size)
