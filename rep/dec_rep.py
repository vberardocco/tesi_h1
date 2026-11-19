"""Velocity decoder on one session: causal ridge with 1000 ms of history (decoding.py).

Evaluation always on the final 20% of the session's calibration file, bins with eval_mask. Minival is not used.

- zero-shot: trained on all held-in calib EXCEPT the session's day (both files), alpha 10^4.5;
- held-in, seen day (reference): trained on the initial 80% of all held-in files, alpha 10^4.5
  (this is the model of the 29 Oct validation, restricted to this session);
- session only: trained on the initial 80% of the file, alpha chosen by an inner cross-validation with 5 contiguous
  blocks of trials within that 80%; grid 10^0-10^6, extended by half an order if the best value is at an extreme.

Usage: python dec_rep.py SID   ->  rep/SID/decoder.json
"""
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path.home() / "thesis_h1_work/src"))
from thesis_h1 import decoding as dec
from falcon_challenge.dataloaders import load_nwb
from falcon_challenge.config import FalconTask

SID = sys.argv[1]
OUT = Path.home() / "thesis_h1_work/rep" / SID
OUT.mkdir(parents=True, exist_ok=True)
H, ALPHA = 50, 10 ** 4.5
T0 = time.time()
def log(m): print(f"[{(time.time() - T0) / 60:5.1f} min] {m}", flush=True)

cal = dec.load_split(Path.home() / "000954", "held-in-calib")
me = next(f for f in cal if SID in f.path.name)
n = len(me.counts); cut = int(0.8 * n)
test = np.r_[np.zeros(cut, bool), np.ones(n - cut, bool)] & me.mask      # evaluated bins: final 20%, eval_mask
first80 = lambda f: np.r_[np.ones(int(0.8 * len(f.counts)), bool), np.zeros(len(f.counts) - int(0.8 * len(f.counts)), bool)]
log(f"{SID} = {me.key}, day {me.day}; {n} bins, test {test.sum()} bins")


def score(model, W, b):
    yh = model.predict(me.counts, W, b)
    return dict(r2=float(dec.r2_vw(me.vel[test], yh[test])), r2_dof=dict(zip(dec.DOF, map(float, dec.r2_dof(me.vel[test], yh[test])))))


res = dict(sid=SID, key=me.key, day=me.day, H_ms=H * 20, test_bins=int(test.sum()))

# 1) zero-shot: without the session's day
others = [f for f in cal if f.day != me.day]
m = dec.RidgeHistory(H).fit_scaler([f.counts for f in others])
m.accumulate([f.counts for f in others], [f.vel for f in others], [f.mask for f in others])
(W, b), = m.solve(H, [ALPHA]).values()
res["zero_shot"] = dict(alpha=ALPHA, train_files=[f.key for f in others], **score(m, W, b))
log(f"zero-shot (without {me.day}): R2 = {res['zero_shot']['r2']:.4f}")

# 2) reference: initial 80% of all held-in files (seen day)
m = dec.RidgeHistory(H).fit_scaler([f.counts[first80(f)] for f in cal])
m.accumulate([f.counts for f in cal], [f.vel for f in cal], [first80(f) & f.mask for f in cal])
(W, b), = m.solve(H, [ALPHA]).values()
res["held_in_80"] = dict(alpha=ALPHA, **score(m, W, b))
log(f"held-in 80% (seen day): R2 = {res['held_in_80']['r2']:.4f}")

# 3) session only: inner CV on the 80% to choose alpha
trial = np.cumsum(load_nwb(me.path, dataset=FalconTask.h1)[2])           # trial number of each bin (from trial_change)
tr_trials = np.unique(trial[:cut])
blocks = np.array_split(tr_trials, 5)                                      # 5 contiguous blocks of trials
alphas = list(np.logspace(0, 6, 13))


def cv(alphas):
    out = {a: [] for a in alphas}
    for bl in blocks:
        val = np.isin(trial, bl) & (np.arange(n) < cut)
        trn = ~np.isin(trial, bl) & (np.arange(n) < cut)
        mm = dec.RidgeHistory(H).fit_scaler([me.counts[trn]])
        mm.accumulate([me.counts], [me.vel], [trn & me.mask])
        for a, (W_, b_) in mm.solve(H, alphas).items():
            yh = mm.predict(me.counts, W_, b_)
            out[a].append(dec.r2_vw(me.vel[val & me.mask], yh[val & me.mask]))
    return {a: float(np.mean(v)) for a, v in out.items()}


cv_res = cv(alphas)
for _ in range(4):                                                         # extend the grid if the best value is at an extreme
    best = max(cv_res, key=cv_res.get)
    if best == min(cv_res):
        new = [best / 10 ** 0.5]
    elif best == max(cv_res):
        new = [best * 10 ** 0.5]
    else:
        break
    log(f"best at the extreme ({best:.3g}): adding {new[0]:.3g}")
    cv_res.update(cv(new))
best = max(cv_res, key=cv_res.get)
log("inner CV: " + ", ".join(f"{a:.0e}:{v:.3f}" for a, v in sorted(cv_res.items())) + f" -> alpha {best:.3g}")
m = dec.RidgeHistory(H).fit_scaler([me.counts[:cut]])
m.accumulate([me.counts], [me.vel], [(np.arange(n) < cut) & me.mask])
(W, b), = m.solve(H, [best]).values()
res["within_session"] = dict(alpha=float(best), cv={f"{a:.3g}": v for a, v in sorted(cv_res.items())}, **score(m, W, b))
log(f"session only (80%): R2 = {res['within_session']['r2']:.4f}")

json.dump(res, open(OUT / "decoder.json", "w"), indent=1)
log(f"saved {OUT / 'decoder.json'}")
