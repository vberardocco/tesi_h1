"""Modello per EvalAI: ridge con 1000 ms di storia, alpha 10^4.5, allenata su tutto il calib held-in.

Minival non si usa. Controllo: il decoder in streaming (un bin alla volta, come nell'evaluator) dà le stesse
predizioni del calcolo in blocco, su un file del calib.
"""
import sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path.home() / "tesi_h1_work/src"))
from tesi_h1 import decoding as dec

t0 = time.time()
H, ALPHA = 50, 10 ** 4.5
cal = dec.load_split(Path.home() / "000954", "held-in-calib")
m = dec.RidgeHistory(H).fit_scaler([f.counts for f in cal])
m.accumulate([f.counts for f in cal], [f.vel for f in cal], [f.mask for f in cal])
(W, b), = m.solve(H, [ALPHA]).values()
out = Path(__file__).with_name("ridge_h50.npz")
dec.save_model(out, W, b, m.mu, m.sd, H)
print(f"salvato {out.name}, {time.time()-t0:.0f}s", flush=True)

f = cal[2]                                            # 19250108T110520, la sessione di riferimento
offline = m.predict(f.counts, W, b)
d = dec.make_streaming_decoder(out); d.reset()
stream = np.stack([d.predict(x[None])[0] for x in f.counts])
print("streaming = blocco:", np.allclose(stream, offline, atol=1e-8), "max diff", np.abs(stream - offline).max())
