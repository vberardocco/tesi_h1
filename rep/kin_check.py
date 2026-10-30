"""Cinematica: variazione della posizione contro velocità, nelle tre sessioni delle repliche.

Nel 04 (§1) la correlazione era su tutti i campioni, confini di trial e pause compresi. Qui:
- solo coppie di campioni consecutivi dello stesso trial e dentro eval_mask di FALCON (entrambi i campioni);
- cross-correlazione: dkin[k] = (pos[k+1] - pos[k]) / 20 ms contro vel[k + 1 + L], L da -10 a +10 campioni,
  con k, k+1 e k+1+L nello stesso trial e in eval_mask. L = 0 è l'allineamento del 04; L > 0 vuol dire che la
  velocità arriva dopo la variazione di posizione.
- la stessa correlazione dopo una media mobile di 100 e 300 ms su entrambi i segnali, la pendenza della retta
  dkin ~ vel e il rapporto tra le deviazioni standard: dicono se lo scarto è rumore veloce o un disaccordo vero;
- autocorrelazione a 1 campione (20 ms) e deviazione standard assoluta di velocità e variazione di posizione, sugli
  stessi campioni. Rumore bianco nella posizione abbassa l'autocorrelazione della variazione di posizione (verso
  -0.5) e ne alza la sd; una velocità più liscia alza l'autocorrelazione della velocità e ne abbassa la sd.
Velocità ed eval_mask: controllate contro il caricatore ufficiale (load_nwb); la posizione non è in load_nwb per H1
e si legge dallo stesso NWB (OpenLoopKinematics) con io.py.

Uso: python kin_check.py  ->  rep/kin/kin_check.csv, rep/kin/kin_lag.png
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

W = Path.home() / "tesi_h1_work"
sys.path.insert(0, str(W / "src"))
from tesi_h1 import io
from falcon_challenge.config import FalconTask
from falcon_challenge.dataloaders import load_nwb

OUT = W / "rep" / "kin"; OUT.mkdir(exist_ok=True)
SIDS = {"19250108T110520": "rif. S1", "19250113T120811": "S2", "19250120T115044": "S5"}
PATH = lambda sid: W / f"data/raw/sub-HumanPitt-held-in-calib/sub-HumanPitt-held-in-calib_ses-{sid}.nwb"
DOF = list(io.KIN_LABELS)
LAGS = np.arange(-10, 11)


def corr(a, b):
    return np.corrcoef(a, b)[0, 1]


rows, curves = [], {}
for sid, name in SIDS.items():
    s = io.load_session(PATH(sid))
    _, vel_f, _, mask_f = load_nwb(PATH(sid), dataset=FalconTask.h1)
    assert np.array_equal(np.asarray(vel_f, float), s.vel) and np.array_equal(np.asarray(mask_f, bool), s.eval_mask), "io e load_nwb diversi"
    pos, vel, tr, em, T = s.kin, s.vel, s.trial_num, s.eval_mask, len(s.kin)
    dkin = np.diff(pos, axis=0) / io.BIN_S                    # dkin[k] tra i campioni k e k+1
    k = np.arange(T - 1)
    base = (tr[k] == tr[k + 1]) & em[k] & em[k + 1]           # dentro un trial e dentro eval_mask
    curves[name] = {}
    for j, d in enumerate(DOF):
        r_all = corr(dkin[:, j], vel[1:, j])                  # come nel 04: tutti i campioni
        r_lag = []
        for L in LAGS:
            t = k + 1 + L
            ok = base & (t >= 0) & (t < T)
            ok[ok] &= (tr[t[ok]] == tr[k[ok]]) & em[t[ok]]
            r_lag.append(corr(dkin[ok, j], vel[t[ok], j]))
        r_lag = np.array(r_lag); curves[name][d] = r_lag
        best = int(np.nanargmax(r_lag))
        jumps = np.abs(dkin[:, j]) > 10 * np.std(dkin[base, j])  # salti (> 10 deviazioni standard) fuori e dentro i trial
        r_sm = {w: corr(uniform_filter1d(dkin[:, j], w)[base], uniform_filter1d(vel[1:, j], w)[base]) for w in (5, 15)}
        pair = base[:-1] & base[1:]                           # campioni k, k+1, k+2 nello stesso trial e in eval_mask
        x, v = dkin[:, j], vel[1:, j]                         # stesso supporto per i due segnali (v[k] = vel[k+1])
        ac_x, ac_v = corr(x[:-1][pair], x[1:][pair]), corr(v[:-1][pair], v[1:][pair])
        rows.append({"sessione": name, "grado": d, "r, tutti i campioni (come il 04)": r_all,
                     "r, dentro i trial e eval_mask (L = 0)": r_lag[LAGS == 0][0],
                     "ritardo migliore (campioni da 20 ms)": int(LAGS[best]), "r al ritardo migliore": r_lag[best],
                     "r, media mobile 100 ms": r_sm[5], "r, media mobile 300 ms": r_sm[15],
                     "pendenza dkin ~ vel": np.polyfit(vel[1:][base, j], dkin[base, j], 1)[0],
                     "sd(dkin) / sd(vel)": dkin[base, j].std() / vel[1:][base, j].std(),
                     "autocorrelazione a 1 bin, velocità": ac_v, "autocorrelazione a 1 bin, variazione di posizione": ac_x,
                     "sd velocità (unità del file)": vel[1:][base, j].std(),
                     "sd variazione di posizione (unità/s)": dkin[base, j].std(),
                     "coppie usate (L = 0)": int(base.sum()),
                     "salti > 10 sd, ai confini o fuori eval_mask": int((jumps & ~base).sum()),
                     "salti > 10 sd, dentro i trial e eval_mask": int((jumps & base).sum())})
    print(f"{name}: {T} campioni, {base.sum()} coppie dentro i trial e in eval_mask; vel e eval_mask uguali a load_nwb")

tab = pd.DataFrame(rows)
tab.to_csv(OUT / "kin_check.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
print(tab.round(3).to_string(index=False))

fig, ax = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for a, (name, cv) in zip(ax, curves.items()):
    for d, r in cv.items():
        a.plot(LAGS * 20, r, marker="o", ms=3, label=d)
    a.axvline(0, color="0.6", lw=0.8); a.set(title=name, xlabel="ritardo della velocità (ms)")
ax[0].set_ylabel("correlazione con la variazione della posizione"); ax[-1].legend(fontsize=8)
fig.suptitle("Dentro i trial e in eval_mask"); fig.tight_layout(); fig.savefig(OUT / "kin_lag.png", dpi=110)
