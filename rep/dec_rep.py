"""Decoder della velocità su una sessione: ridge causale con 1000 ms di storia (decoding.py).

Valutazione sempre sul 20% finale del file di calibrazione della sessione, bin con eval_mask. Minival non si usa.

- zero-shot: allenata su tutto il calib held-in TRANNE il giorno della sessione (entrambi i file), alpha 10^4.5;
- held-in, giorno visto (riferimento): allenata sull'80% iniziale di tutti i file held-in, alpha 10^4.5
  (è il modello della validazione del 29/10, ristretto a questa sessione);
- solo la sessione: allenata sull'80% iniziale del file, alpha scelto con una cross-validation interna a 5 blocchi
  contigui di trial dentro quell'80%; griglia 10^0-10^6, allargata di mezzo ordine se il migliore è un estremo.

Uso: python dec_rep.py SID   ->  rep/SID/decoder.json
"""
import json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path.home() / "tesi_h1_work/src"))
from tesi_h1 import decoding as dec
from falcon_challenge.dataloaders import load_nwb
from falcon_challenge.config import FalconTask

SID = sys.argv[1]
OUT = Path.home() / "tesi_h1_work/rep" / SID
OUT.mkdir(parents=True, exist_ok=True)
H, ALPHA = 50, 10 ** 4.5
T0 = time.time()
def log(m): print(f"[{(time.time() - T0) / 60:5.1f} min] {m}", flush=True)

cal = dec.load_split(Path.home() / "000954", "held-in-calib")
me = next(f for f in cal if SID in f.path.name)
n = len(me.counts); cut = int(0.8 * n)
test = np.r_[np.zeros(cut, bool), np.ones(n - cut, bool)] & me.mask      # bin valutati: 20% finale, eval_mask
first80 = lambda f: np.r_[np.ones(int(0.8 * len(f.counts)), bool), np.zeros(len(f.counts) - int(0.8 * len(f.counts)), bool)]
log(f"{SID} = {me.key}, giorno {me.day}; {n} bin, test {test.sum()} bin")


def score(model, W, b):
    yh = model.predict(me.counts, W, b)
    return dict(r2=float(dec.r2_vw(me.vel[test], yh[test])), r2_dof=dict(zip(dec.DOF, map(float, dec.r2_dof(me.vel[test], yh[test])))))


res = dict(sid=SID, key=me.key, day=me.day, H_ms=H * 20, test_bins=int(test.sum()))

# 1) zero-shot: senza il giorno della sessione
others = [f for f in cal if f.day != me.day]
m = dec.RidgeHistory(H).fit_scaler([f.counts for f in others])
m.accumulate([f.counts for f in others], [f.vel for f in others], [f.mask for f in others])
(W, b), = m.solve(H, [ALPHA]).values()
res["zero_shot"] = dict(alpha=ALPHA, train_files=[f.key for f in others], **score(m, W, b))
log(f"zero-shot (senza {me.day}): R2 = {res['zero_shot']['r2']:.4f}")

# 2) riferimento: 80% iniziale di tutti i file held-in (giorno visto)
m = dec.RidgeHistory(H).fit_scaler([f.counts[first80(f)] for f in cal])
m.accumulate([f.counts for f in cal], [f.vel for f in cal], [first80(f) & f.mask for f in cal])
(W, b), = m.solve(H, [ALPHA]).values()
res["held_in_80"] = dict(alpha=ALPHA, **score(m, W, b))
log(f"held-in 80% (giorno visto): R2 = {res['held_in_80']['r2']:.4f}")

# 3) solo la sessione: CV interna sull'80% per alpha
trial = np.cumsum(load_nwb(me.path, dataset=FalconTask.h1)[2])           # numero di trial di ogni bin (da trial_change)
tr_trials = np.unique(trial[:cut])
blocks = np.array_split(tr_trials, 5)                                      # 5 blocchi contigui di trial
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
for _ in range(4):                                                         # allarga la griglia se il migliore è un estremo
    best = max(cv_res, key=cv_res.get)
    if best == min(cv_res):
        new = [best / 10 ** 0.5]
    elif best == max(cv_res):
        new = [best * 10 ** 0.5]
    else:
        break
    log(f"migliore all'estremo ({best:.3g}): aggiungo {new[0]:.3g}")
    cv_res.update(cv(new))
best = max(cv_res, key=cv_res.get)
log("CV interna: " + ", ".join(f"{a:.0e}:{v:.3f}" for a, v in sorted(cv_res.items())) + f" -> alpha {best:.3g}")
m = dec.RidgeHistory(H).fit_scaler([me.counts[:cut]])
m.accumulate([me.counts], [me.vel], [(np.arange(n) < cut) & me.mask])
(W, b), = m.solve(H, [best]).values()
res["within_session"] = dict(alpha=float(best), cv={f"{a:.3g}": v for a, v in sorted(cv_res.items())}, **score(m, W, b))
log(f"solo la sessione (80%): R2 = {res['within_session']['r2']:.4f}")

json.dump(res, open(OUT / "decoder.json", "w"), indent=1)
log(f"salvato {OUT / 'decoder.json'}")
