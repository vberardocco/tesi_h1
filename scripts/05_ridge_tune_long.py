"""Ridge con storia di 600, 800 e 1000 ms sulla validazione del calib (80/20 per file). Minival non si usa."""
import sys, time, pickle
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path.home() / "tesi_h1_work/src"))
from tesi_h1 import decoding as dec

t0 = time.time()
RAW = Path.home() / "000954"
cal = dec.load_split(RAW, "held-in-calib")

H_MAX = 50                                   # 50 bin = 1000 ms
HS = [30, 40, 50]                            # 600, 800, 1000 ms
ALPHAS = np.logspace(0, 6, 13)               # la stessa griglia di stamattina

cut = [int(0.8 * len(f.counts)) for f in cal]                  # 80% / 20% per file
tr_rows = [np.r_[np.ones(c, bool), np.zeros(len(f.counts) - c, bool)] & f.mask for f, c in zip(cal, cut)]
va_rows = [~np.r_[np.ones(c, bool), np.zeros(len(f.counts) - c, bool)] & f.mask for f, c in zip(cal, cut)]

m = dec.RidgeHistory(H_MAX).fit_scaler([f.counts[:c] for f, c in zip(cal, cut)])
m.accumulate([f.counts for f in cal], [f.vel for f in cal], tr_rows)
print(f"accumulato X'X ({dec.N_CH * H_MAX} colonne) in {time.time()-t0:.0f}s", flush=True)

res = {}
for H in HS:
    sol = m.solve(H, ALPHAS)
    Lval = [dec.lagged(m.z(f.counts), H)[c:] for f, c in zip(cal, cut)]   # storia dei bin di validazione, calcolata una volta
    for a, (W, b) in sol.items():
        preds = [np.r_[np.zeros((c, 7)), L.astype(np.float64) @ W + b] for L, c in zip(Lval, cut)]
        r2, per_day, per_dof = dec.evaluate([f.day for f in cal], [f.vel for f in cal], preds, va_rows)
        res[(H, a)] = dict(r2=r2, per_day=per_day, per_dof=per_dof)
    best = max(ALPHAS, key=lambda a: res[(H, a)]["r2"])
    print(f"H={H:2d} ({H*20:4d} ms)  alpha migliore={best:9.1f}  R2={res[(H, best)]['r2']:.4f}  "
          f"per DoF={np.round(res[(H, best)]['per_dof'].mean(0), 3)}  {time.time()-t0:.0f}s", flush=True)

pickle.dump(res, open(Path(__file__).with_name("tune_long.pkl"), "wb"))
