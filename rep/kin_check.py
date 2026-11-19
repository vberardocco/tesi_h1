"""Kinematics: position change versus velocity, in the three replication sessions.

In 04 (§1) the correlation was computed over all samples, including trial boundaries and pauses. Here:
- only pairs of consecutive samples of the same trial and inside the FALCON eval_mask (both samples);
- cross-correlation: dkin[k] = (pos[k+1] - pos[k]) / 20 ms against vel[k + 1 + L], L from -10 to +10 samples,
  with k, k+1 and k+1+L in the same trial and in eval_mask. L = 0 is the alignment of 04; L > 0 means that the
  velocity arrives after the position change.
- the same correlation after a moving average of 100 and 300 ms on both signals, the slope of the line
  dkin ~ vel and the ratio between the standard deviations: they tell whether the discrepancy is fast noise or a real disagreement;
- 1-sample (20 ms) autocorrelation and absolute standard deviation of velocity and position change, on the
  same samples. White noise in the position lowers the autocorrelation of the position change (towards
  -0.5) and raises its sd; a smoother velocity raises the autocorrelation of the velocity and lowers its sd.
Velocity and eval_mask: checked against the official loader (load_nwb); the position is not in load_nwb for H1
and is read from the same NWB (OpenLoopKinematics) with io.py.

Usage: python kin_check.py  ->  rep/kin/kin_check.csv, rep/kin/kin_lag.png
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

W = Path.home() / "thesis_h1_work"
sys.path.insert(0, str(W / "src"))
from thesis_h1 import io
from falcon_challenge.config import FalconTask
from falcon_challenge.dataloaders import load_nwb

OUT = W / "rep" / "kin"; OUT.mkdir(exist_ok=True)
SIDS = {"19250108T110520": "ref. S1", "19250113T120811": "S2", "19250120T115044": "S5"}
PATH = lambda sid: W / f"data/raw/sub-HumanPitt-held-in-calib/sub-HumanPitt-held-in-calib_ses-{sid}.nwb"
DOF = list(io.KIN_LABELS)
LAGS = np.arange(-10, 11)


def corr(a, b):
    return np.corrcoef(a, b)[0, 1]


rows, curves = [], {}
for sid, name in SIDS.items():
    s = io.load_session(PATH(sid))
    _, vel_f, _, mask_f = load_nwb(PATH(sid), dataset=FalconTask.h1)
    assert np.array_equal(np.asarray(vel_f, float), s.vel) and np.array_equal(np.asarray(mask_f, bool), s.eval_mask), "io and load_nwb differ"
    pos, vel, tr, em, T = s.kin, s.vel, s.trial_num, s.eval_mask, len(s.kin)
    dkin = np.diff(pos, axis=0) / io.BIN_S                    # dkin[k] between samples k and k+1
    k = np.arange(T - 1)
    base = (tr[k] == tr[k + 1]) & em[k] & em[k + 1]           # within a trial and inside eval_mask
    curves[name] = {}
    for j, d in enumerate(DOF):
        r_all = corr(dkin[:, j], vel[1:, j])                  # as in 04: all samples
        r_lag = []
        for L in LAGS:
            t = k + 1 + L
            ok = base & (t >= 0) & (t < T)
            ok[ok] &= (tr[t[ok]] == tr[k[ok]]) & em[t[ok]]
            r_lag.append(corr(dkin[ok, j], vel[t[ok], j]))
        r_lag = np.array(r_lag); curves[name][d] = r_lag
        best = int(np.nanargmax(r_lag))
        jumps = np.abs(dkin[:, j]) > 10 * np.std(dkin[base, j])  # jumps (> 10 standard deviations) outside and inside trials
        r_sm = {w: corr(uniform_filter1d(dkin[:, j], w)[base], uniform_filter1d(vel[1:, j], w)[base]) for w in (5, 15)}
        pair = base[:-1] & base[1:]                           # samples k, k+1, k+2 in the same trial and in eval_mask
        x, v = dkin[:, j], vel[1:, j]                         # same support for the two signals (v[k] = vel[k+1])
        ac_x, ac_v = corr(x[:-1][pair], x[1:][pair]), corr(v[:-1][pair], v[1:][pair])
        rows.append({"session": name, "degree": d, "r, tutti i campioni (come il 04)": r_all,
                     "r, dentro i trial e eval_mask (L = 0)": r_lag[LAGS == 0][0],
                     "ritardo migliore (campioni da 20 ms)": int(LAGS[best]), "r al ritardo migliore": r_lag[best],
                     "r, media mobile 100 ms": r_sm[5], "r, media mobile 300 ms": r_sm[15],
                     "pendenza dkin ~ vel": np.polyfit(vel[1:][base, j], dkin[base, j], 1)[0],
                     "sd(dkin) / sd(vel)": dkin[base, j].std() / vel[1:][base, j].std(),
                     "autocorrelazione a 1 bin, velocità": ac_v, "autocorrelazione a 1 bin, variazione di posizione": ac_x,
                     "sd velocità (unità del file)": vel[1:][base, j].std(),
                     "sd variazione di posizione (units/s)": dkin[base, j].std(),
                     "coppie usate (L = 0)": int(base.sum()),
                     "salti > 10 sd, ai confini o fuori eval_mask": int((jumps & ~base).sum()),
                     "salti > 10 sd, dentro i trial e eval_mask": int((jumps & base).sum())})
    print(f"{name}: {T} samples, {base.sum()} pairs inside trials and in eval_mask; vel and eval_mask equal to load_nwb")

tab = pd.DataFrame(rows)
tab.to_csv(OUT / "kin_check.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
print(tab.round(3).to_string(index=False))

fig, ax = plt.subplots(1, 3, figsize=(15, 4), sharey=True)
for a, (name, cv) in zip(ax, curves.items()):
    for d, r in cv.items():
        a.plot(LAGS * 20, r, marker="o", ms=3, label=d)
    a.axvline(0, color="0.6", lw=0.8); a.set(title=name, xlabel="velocity lag (ms)")
ax[0].set_ylabel("correlation with the position change"); ax[-1].legend(fontsize=8)
fig.suptitle("Inside trials and in eval_mask"); fig.tight_layout(); fig.savefig(OUT / "kin_lag.png", dpi=110)
