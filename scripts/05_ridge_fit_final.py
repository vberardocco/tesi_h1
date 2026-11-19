"""Model for EvalAI: ridge with 1000 ms of history, alpha 10^4.5, trained on all held-in calib data.

Minival is not used. Check: the streaming decoder (one bin at a time, as in the evaluator) gives the same
predictions as the block computation, on one calib file.
"""
import sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path.home() / "thesis_h1_work/src"))
from thesis_h1 import decoding as dec

t0 = time.time()
H, ALPHA = 50, 10 ** 4.5
cal = dec.load_split(Path.home() / "000954", "held-in-calib")
m = dec.RidgeHistory(H).fit_scaler([f.counts for f in cal])
m.accumulate([f.counts for f in cal], [f.vel for f in cal], [f.mask for f in cal])
(W, b), = m.solve(H, [ALPHA]).values()
out = Path(__file__).with_name("ridge_h50.npz")
dec.save_model(out, W, b, m.mu, m.sd, H)
print(f"saved {out.name}, {time.time()-t0:.0f}s", flush=True)

f = cal[2]                                            # 19250108T110520, the reference session
offline = m.predict(f.counts, W, b)
d = dec.make_streaming_decoder(out); d.reset()
stream = np.stack([d.predict(x[None])[0] for x in f.counts])
print("streaming = block:", np.allclose(stream, offline, atol=1e-8), "max diff", np.abs(stream - offline).max())
