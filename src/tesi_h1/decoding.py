"""Decoder della velocità per FALCON H1 (notebook 05).

Tre pezzi:

- caricamento con il caricatore ufficiale di FALCON (`load_nwb`), così i bin sono allineati al
  benchmark: il bin k contiene gli spike in [t_k - 20 ms, t_k) e si abbina alla velocità al tempo t_k;
- ridge causale con storia: per predire la velocità al bin k usa i conteggi dei bin k, k-1, ..., k-H+1;
- metrica come l'evaluator di FALCON: R² "pesato per varianza" sui soli bin con eval_mask, calcolato
  per giorno di registrazione (i file dello stesso giorno vengono concatenati) e poi mediato sui giorni.

Il decoder `StreamingRidge` riceve un bin alla volta, come nell'evaluator ufficiale e su EvalAI.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

N_CH = 176                     # canali di H1
DOF = ["tx", "ty", "tz", "rx", "g1", "g2", "g3"]  # i 7 gradi di libertà della velocità (FALCON H1)
BIN_MS = 20                    # durata di un bin


# --------------------------------------------------------------------------------------------
# Dati
# --------------------------------------------------------------------------------------------
@dataclass
class FalconFile:
    path: Path                 # file NWB
    day: str                   # giorno di registrazione secondo FALCON (S0 ... S5)
    key: str                   # nome del file secondo FALCON (es. S0_set_1)
    counts: np.ndarray         # conteggi, bin × 176
    vel: np.ndarray            # velocità, bin × 7
    mask: np.ndarray           # eval_mask, bin (True = bin valutato)


def load_file(path) -> FalconFile:
    """Legge un file con il caricatore ufficiale di FALCON."""
    from falcon_challenge.config import FalconConfig, FalconTask
    from falcon_challenge.dataloaders import load_nwb
    path = Path(path)
    counts, vel, _, mask = load_nwb(path, dataset=FalconTask.h1)   # stessi bin del benchmark
    key = FalconConfig(FalconTask.h1).hash_dataset(path.stem)      # es. "S0_set_1"
    return FalconFile(path, key.split("_")[0], key, counts.astype(np.float32),
                      vel.astype(np.float64), mask.astype(bool))


def load_split(root, split: str) -> list[FalconFile]:
    """Tutti i file di uno split ("held-in-calib", "held-in-minival", ...), in ordine."""
    files = sorted(Path(root).glob(f"sub-HumanPitt-{split}/*.nwb"))
    return [load_file(f) for f in files]


# --------------------------------------------------------------------------------------------
# Storia causale
# --------------------------------------------------------------------------------------------
def lagged(z: np.ndarray, H: int) -> np.ndarray:
    """Matrice con la storia: riga k = [z_k, z_{k-1}, ..., z_{k-H+1}] (prima il bin più recente).

    I bin prima dell'inizio del file valgono 0, cioè la media del canale (z è già z-scorato):
    è quello che vede anche il decoder in streaming quando parte.
    """
    n, c = z.shape
    out = np.zeros((n, c * H), dtype=np.float32)
    for j in range(H):                                   # j = ritardo in bin
        out[j:, j * c:(j + 1) * c] = z[: n - j]
    return out


# --------------------------------------------------------------------------------------------
# Ridge con molti valori di alpha in una volta
# --------------------------------------------------------------------------------------------
class RidgeHistory:
    """Ridge sulla storia causale, stimata accumulando X'X file per file (senza tenere X in memoria).

    Una sola passata sui dati con la storia massima H_max basta per tutte le storie più corte:
    le colonne della storia H sono le prime H × 176 di quella con H_max.
    """

    def __init__(self, H_max: int):
        self.H_max = H_max

    # standardizzazione per canale (non per colonna: i ritardi dello stesso canale hanno la stessa scala)
    def fit_scaler(self, counts_list):
        allc = np.concatenate(counts_list)
        self.mu = allc.mean(0)
        self.sd = allc.std(0)
        self.sd[self.sd == 0] = 1.0
        return self

    def z(self, counts):
        return (counts - self.mu) / self.sd

    def accumulate(self, counts_list, vel_list, rows_list):
        """Somma X'X, X'y, medie sui bin scelti (rows) di ogni file."""
        p = N_CH * self.H_max
        XtX = np.zeros((p, p)); Xty = np.zeros((p, 7)); sx = np.zeros(p); sy = np.zeros(7); n = 0
        for counts, vel, rows in zip(counts_list, vel_list, rows_list):
            X = lagged(self.z(counts), self.H_max)[rows].astype(np.float64)
            y = vel[rows]
            XtX += X.T @ X; Xty += X.T @ y; sx += X.sum(0); sy += y.sum(0); n += len(y)
        self.stats = dict(XtX=XtX, Xty=Xty, sx=sx, sy=sy, n=n)
        return self

    def solve(self, H: int, alphas):
        """Coefficienti per la storia H e per ogni alpha: dict alpha -> (W, b)."""
        s = self.stats; p = N_CH * H; n = s["n"]
        mx = s["sx"][:p] / n; my = s["sy"] / n
        C = s["XtX"][:p, :p] - n * np.outer(mx, mx)       # X'X con X centrata (intercetta libera)
        r = s["Xty"][:p] - n * np.outer(mx, my)           # X'y centrata
        w, V = np.linalg.eigh(C)                          # una scomposizione, poi ogni alpha costa poco
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
# Metrica (come l'evaluator di FALCON)
# --------------------------------------------------------------------------------------------
def r2_dof(y, yhat):
    """R² di ogni grado di libertà."""
    ss_res = ((y - yhat) ** 2).sum(0)
    ss_tot = ((y - y.mean(0)) ** 2).sum(0)
    return 1 - ss_res / ss_tot


def r2_vw(y, yhat):
    """R² pesato per varianza = sklearn r2_score(multioutput='variance_weighted')."""
    ss_res = ((y - yhat) ** 2).sum()
    ss_tot = ((y - y.mean(0)) ** 2).sum()
    return 1 - ss_res / ss_tot


def evaluate(days, ys, preds, masks):
    """R² come FALCON: per giorno (file dello stesso giorno concatenati), solo bin con eval_mask.

    Restituisce: media sui giorni, dict giorno -> R², e R² per DoF per giorno (array giorni × 7).
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
# Decoder in streaming per l'evaluator ufficiale (e per EvalAI)
# --------------------------------------------------------------------------------------------
def save_model(path, W, b, mu, sd, H):
    np.savez(path, W=W, b=b, mu=mu, sd=sd, H=H)


def make_streaming_decoder(model_path, batch_size: int = 1):
    """Crea il decoder che riceve un bin alla volta (interfaccia BCIDecoder di FALCON)."""
    from falcon_challenge.config import FalconConfig, FalconTask
    from falcon_challenge.interface import BCIDecoder

    class StreamingRidge(BCIDecoder):
        def __init__(self, task_config, model_path, batch_size=1):
            super().__init__(task_config=task_config, batch_size=batch_size)
            m = np.load(model_path)
            self.W, self.b, self.mu, self.sd, self.H = m["W"], m["b"], m["mu"], m["sd"], int(m["H"])
            self.reset()

        def reset(self, dataset_tags=[""]):
            # buffer degli ultimi H bin z-scorati, per ogni elemento del batch; 0 = media del canale
            self.buf = np.zeros((self.batch_size, self.H, N_CH))

        def on_done(self, dones):
            pass                                       # H1 è "continuo": nessun reset tra un trial e l'altro

        def predict(self, neural_observations):
            x = np.asarray(neural_observations, dtype=np.float64)
            B = x.shape[0]
            if B < self.batch_size:                    # ultimo batch più piccolo: completa con zeri
                x = np.pad(x, ((0, self.batch_size - B), (0, 0)))
            self.buf = np.roll(self.buf, 1, axis=1)    # sposta la storia di un bin
            self.buf[:, 0] = (x - self.mu) / self.sd   # il bin appena arrivato va in posizione 0
            out = self.buf.reshape(self.batch_size, -1) @ self.W + self.b
            return out[:B]

    return StreamingRidge(FalconConfig(task=FalconTask.h1), model_path, batch_size)
