import os, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path.home() / "thesis_h1_work/src"))
from thesis_h1 import decoding as dec

t0 = time.time()
RAW = Path.home() / "000954"
H, ALPHA = 30, 10 ** 4.5
cal = dec.load_split(RAW, "held-in-calib")
mv = dec.load_split(RAW, "held-in-minival")

m = dec.RidgeHistory(H).fit_scaler([f.counts for f in cal])
m.accumulate([f.counts for f in cal], [f.vel for f in cal], [f.mask for f in cal])
(W, b), = m.solve(H, [ALPHA]).values()
print("fit", f"{time.time()-t0:.0f}s", flush=True)

preds = [m.predict(f.counts, W, b) for f in mv]
r2, per_day, per_dof = dec.evaluate([f.day for f in mv], [f.vel for f in mv], preds, [f.mask for f in mv])
print("OUR evaluator  mean R2:", round(r2, 4), {k: round(v, 4) for k, v in per_day.items()})
print("per DoF (mean over days):", dict(zip(dec.DOF, per_dof.mean(0).round(3))))

out = Path(__file__).with_name("ridge_h30.npz")
dec.save_model(out, W, b, m.mu, m.sd, H)

# official evaluator, one bin at a time
os.environ["EVAL_DATA_PATH"] = str(Path.home())          # -> ~/000954/*minival/*.nwb
os.environ["PREDICTION_PATH_LOCAL"] = str(Path(__file__).with_name("pred.pkl"))
os.environ["GT_PATH"] = str(Path(__file__).with_name("gt.pkl"))
from falcon_challenge.evaluator import FalconEvaluator
ev = FalconEvaluator(eval_remote=False, split="h1", dataloader_workers=0)
res = ev.evaluate(dec.make_streaming_decoder(out), phase="minival")
print("FALCON evaluator:", res, f"{time.time()-t0:.0f}s")
