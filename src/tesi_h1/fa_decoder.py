"""Ridge sulle latenti FA (tabella B): il modello FA congelato del cross-sessione come decoder causale per FALCON H1.

Il modello è `FA_d15_heldin_s0` di crosssession.py, così com'è: FA con K = 15 sui 170 canali attivi z-scorati con
media e sd delle 13 held-in, latenti = media a posteriori, standardizzate con i momenti held-in, readout ridge con
storia di 50 bin (1000 ms), α scelto leave-one-day-out sulle held-in, stimato su tutto il calib held-in.

Ogni passo è lineare e usa solo il bin corrente, quindi il decoder funziona un bin alla volta:
  z   = (conteggi[attivi] − mu) / sd
  lat = (z − fa_mean) @ P,   P = Ψ⁻¹Wᵀ (I + W Ψ⁻¹ Wᵀ)⁻¹   (= FactorAnalysis.transform)
  s   = (lat − lat_m) / lat_s
  v   = [s_k, s_{k−1}, ..., s_{k−49}] @ W + b   (bin prima dell'inizio del file = 0, cioè la media)

Il readout è stato allenato con la storia che riparte a ogni trial; l'evaluator di FALCON per H1 è "continuo" (non
chiama on_done), quindi qui la storia scorre per tutto il file, come per ridge_h50.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .decoding import lagged

N_CH = 176


def export(bundle, active, path):
    """Salva in un npz il modello FA congelato (Bundle di crosssession) con le sole matrici che servono."""
    M, rd = bundle.model, bundle.readout
    fa = M.fa
    Wpsi = fa.components_ / fa.noise_variance_
    P = Wpsi.T @ np.linalg.inv(np.eye(len(fa.components_)) + Wpsi @ fa.components_.T)
    np.savez(path, active=np.asarray(active, int), mu=M.mu, sd=M.sd, fa_mean=fa.mean_, P=P, lat_m=rd.m, lat_s=rd.s,
             W=rd.W, b=rd.b, H=int(rd.W.shape[0] // P.shape[1]), alpha=float(rd.alpha),
             lodo_r2=float(bundle.info["lodo_r2"]))
    return Path(path)


class FALatentRidge:
    """Il modello dell'npz, applicato a un file intero (in blocco)."""

    def __init__(self, path):
        m = np.load(path)
        self.active, self.mu, self.sd, self.fa_mean, self.P = m["active"], m["mu"], m["sd"], m["fa_mean"], m["P"]
        self.lat_m, self.lat_s, self.W, self.b, self.H = m["lat_m"], m["lat_s"], m["W"], m["b"], int(m["H"])
        self.alpha, self.lodo_r2 = float(m["alpha"]), float(m["lodo_r2"])

    def latents(self, counts):
        """Latenti standardizzate (bin × K) dai conteggi di tutti i 176 canali."""
        z = (np.asarray(counts, np.float64)[:, self.active] - self.mu) / self.sd
        return ((z - self.fa_mean) @ self.P - self.lat_m) / self.lat_s

    def predict(self, counts):
        """Velocità predette per un file intero, con la storia che scorre per tutto il file."""
        return lagged(self.latents(counts).astype(np.float32), self.H).astype(np.float64) @ self.W + self.b


def make_streaming_decoder(model_path, batch_size: int = 1):
    """Decoder che riceve un bin alla volta (interfaccia BCIDecoder di FALCON), come StreamingRidge di decoding.py."""
    from falcon_challenge.config import FalconConfig, FalconTask
    from falcon_challenge.interface import BCIDecoder

    class StreamingFALatentRidge(BCIDecoder):
        def __init__(self, task_config, model_path, batch_size=1):
            super().__init__(task_config=task_config, batch_size=batch_size)
            self.m = FALatentRidge(model_path)
            self.K = self.m.P.shape[1]
            self.reset()

        def reset(self, dataset_tags=[""]):
            self.buf = np.zeros((self.batch_size, self.m.H, self.K))    # 0 = latente nella media held-in

        def on_done(self, dones):
            pass                                                        # H1 è continuo: non viene chiamato

        def predict(self, neural_observations):
            x = np.asarray(neural_observations, dtype=np.float64)
            B = x.shape[0]
            if B < self.batch_size:
                x = np.pad(x, ((0, self.batch_size - B), (0, 0)))
            self.buf = np.roll(self.buf, 1, axis=1)
            self.buf[:, 0] = self.m.latents(x)
            return (self.buf.reshape(self.batch_size, -1) @ self.m.W + self.m.b)[:B]

    return StreamingFALatentRidge(FalconConfig(task=FalconTask.h1), model_path, batch_size)
