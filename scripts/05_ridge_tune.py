import sys, time, pickle
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path.home() / "tesi_h1_work/src"))
from tesi_h1 import decoding as dec

t0 = time.time()
RAW = Path.home() / "000954"
cal = dec.load_split(RAW, "held-in-calib")
print("caricati", len(cal), f"{time.time()-t0:.0f}s", flush=True)

H_MAX = 30
HS = [1, 2, 3, 5, 10, 15, 20, 25, 30]
ALPHAS = np.logspace(0, 6, 13)

# 80% / 20% per file (come TRAIN_TEST di FALCON)
cut = [int(0.8 * len(f.counts)) for f in cal]
tr_rows = []
for f, c in zip(cal, cut):
    r = np.zeros(len(f.counts), bool); r[:c] = True
    tr_rows.append(r & f.mask)            # training: solo bin con eval_mask

m = dec.RidgeHistory(H_MAX).fit_scaler([f.counts[:c] for f, c in zip(cal, cut)])
m.accumulate([f.counts for f in cal], [f.vel for f in cal], tr_rows)
print("accumulato", f"{time.time()-t0:.0f}s", flush=True)

res = {}
for H in HS:
    sol = m.solve(H, ALPHAS)
    for a, (W, b) in sol.items():
        preds = [m.predict(f.counts, W, b) for f in cal]
        vmask = [np.r_[np.zeros(c, bool), np.ones(len(f.counts) - c, bool)] & f.mask for f, c in zip(cal, cut)]
        r2, per_day, _ = dec.evaluate([f.day for f in cal], [f.vel for f in cal], preds, vmask)
        res[(H, a)] = r2
    best = max(ALPHAS, key=lambda a: res[(H, a)])
    print(f"H={H:2d} ({H*20:3d} ms)  best alpha={best:9.1f}  R2={res[(H, best)]:.4f}  {time.time()-t0:.0f}s", flush=True)

pickle.dump(res, open(Path(__file__).with_name("tune.pkl"), "wb"))
