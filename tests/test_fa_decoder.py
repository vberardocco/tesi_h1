"""Tests of the FA latent decoder (src/thesis_h1/fa_decoder.py) against the frozen cross-session FA model.

Requires the cache ~/thesis_h1_work/xs (data.pkl, models/FA_d15_heldin_s0.pkl) and falcon_challenge.
Usage: python tests/test_fa_decoder.py
"""
import pickle
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import numpy as np  # noqa: E402

from thesis_h1 import crosssession as xs  # noqa: E402
from thesis_h1 import fa_decoder as fd  # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(f"[{'ok' if ok else 'ERROR'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


data = xs.load_data()
with open(xs.bundle_path("FA", 0, "heldin", 15), "rb") as f:
    b = pickle.load(f)
path = fd.export(b, data["active"], Path(tempfile.mkdtemp()) / "fa.npz")
m = fd.FALatentRidge(path)
check("1. model: H = 50, K = 15, α and leave-one-day-out of the frozen model",
      m.H == 50 and m.P.shape == (170, 15) and m.alpha == b.readout.alpha and m.lodo_r2 == b.info["lodo_r2"],
      f"H {m.H}, α {m.alpha:g}, LODO {m.lodo_r2:.4f}")

trials = [t for s in xs.heldin(data)[:2] for t in s.trials[:2]] + xs.heldout(data)[0].trials[:1]
err_lat, err_pred = 0.0, 0.0
for t in trials:
    ref_lat = (b.model.encode([xs.rep(t, "counts")])[0] - b.readout.m) / b.readout.s
    err_lat = max(err_lat, float(np.abs(m.latents(t.counts_all) - ref_lat).max()))
    ref = b.readout.predict(b.model.encode([xs.rep(t, "counts")])[0])
    err_pred = max(err_pred, float(np.abs(m.predict(t.counts_all) - ref).max()))
check("2. latents = FactorAnalysis.transform of the frozen model", err_lat < 1e-9, f"max diff {err_lat:.2e}")
check("3. one trial as a file: predictions = frozen readout", err_pred < 1e-5, f"max diff {err_pred:.2e}")

seq = [np.concatenate([t.counts_all for t in trials[:3]]), np.concatenate([t.counts_all for t in trials[3:]])]
block = [m.predict(x) for x in seq]
for B in (1, 2):
    dec = fd.make_streaming_decoder(path, batch_size=B)
    dec.reset()
    use = seq[:B]
    n = max(len(x) for x in use)
    pad = [np.pad(x, ((0, n - len(x)), (0, 0))) for x in use]
    out = np.stack([dec.predict(np.stack([p[k] for p in pad])) for k in range(n)], 1)
    err = max(float(np.abs(out[i, :len(x)] - blk).max()) for i, (x, blk) in enumerate(zip(use, block)))
    check(f"4. streaming with batch {B} = block (history continuous across the file)", err < 1e-5, f"max diff {err:.2e}")

first = m.predict(trials[0].counts_all)
cont = m.predict(np.concatenate([trials[0].counts_all, trials[1].counts_all]))[len(trials[0].counts_all):]
alone = m.predict(trials[1].counts_all)
d = np.abs(cont - alone).max(1)
check("5. continuous history: differs from the per-trial history only in the first 49 bins of the following trial",
      d[49:].max() < 1e-5 and d[:49].max() > 1e-6, f"first 49 bins {d[:49].max():.3f}, after {d[49:].max():.1e}")
print()
print("ALL OK" if not FAIL else f"{len(FAIL)} TESTS FAILED: {FAIL}")

sys.exit(1 if FAIL else 0)
