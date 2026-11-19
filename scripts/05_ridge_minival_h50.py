"""Minival evaluation of the final model (ridge, 1000 ms history): run once, only as a number to report.

The model is models/ridge_h50.npz as it is (the one that goes into the Docker image): no retraining, no choice.
Three computations:
  1. block: predict on each whole file + our evaluator (decoding.evaluate);
  2. streaming with the official FALCON evaluator, batch 4 (as in the Docker image / EvalAI);
  3. the same with batch 1.
Check: R² equal to the fourth digit and bin-by-bin predictions with maximum difference ~1e-5
(the block history is built in float32, streaming works in float64).
Output: run05/minival_h50.log (next to the 600 ms number from commit c8fbd9b).
"""
import hashlib, os, pickle, sys, tempfile, time
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from thesis_h1 import decoding as dec

# Model and batch sizes can be changed from the environment (e.g. MODEL=models/ridge_h30.npz BATCHES=4 for the
# comparability check at 600 ms); the defaults are those of the final-model evaluation.
MODEL = REPO / os.environ.get("MODEL", "models/ridge_h50.npz")
BATCHES = [int(x) for x in os.environ.get("BATCHES", "4,1").split(",")]
ORIGIN = {"ridge_h50.npz": "alpha 10^4.5, trained on all held-in calib data (commit 5a45607)",
          "ridge_h30.npz": "alpha 10^4.5, trained on all held-in calib data (scripts/05_ridge_minival.py, commit c8fbd9b)"}
RAW = Path.home() / "000954"
LOG = REPO / "run05/minival_h50.log"
TMP = Path(os.environ.get("MINIVAL_TMP", tempfile.mkdtemp()))

LOG.parent.mkdir(exist_ok=True)
log_f = open(LOG, "a")
def out(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True); log_f.write(s + "\n"); log_f.flush()

import falcon_challenge
from importlib.metadata import version
from falcon_challenge.config import FalconConfig, FalconTask
from falcon_challenge.evaluator import FalconEvaluator

t0 = time.time()
m = np.load(MODEL)
W, b, mu, sd, H = m["W"], m["b"], m["mu"], m["sd"], int(m["H"])
assert W.shape == (dec.N_CH * H, 7)

out("=" * 100)
out(f"Valutazione minival di {MODEL.name}   {datetime.now():%Y-%m-%d %H:%M}")
out(f"model          : {MODEL.relative_to(REPO)}  md5 {hashlib.md5(MODEL.read_bytes()).hexdigest()}")
out(f"history        : H = {H} bin = {H * dec.BIN_MS} ms; {ORIGIN.get(MODEL.name, '')}")
out(f"falcon_challenge: {version('falcon_challenge')}  ({Path(falcon_challenge.__file__).parent})")
out(f"batch size     : streaming with {BATCHES} (4 = BATCH_SIZE of the Docker image)")
out(f"reference       : 600 ms history (H = 30), minival R² 0.3831, commit c8fbd9b")
out("-" * 100)

# 1. block
mv = dec.load_split(RAW, "held-in-minival")
mm =dec.RidgeHistory(H); mm.mu, mm.sd = mu, sd
block = {f.key: mm.predict(f.counts, W, b) for f in mv}
r2_b, day_b, dof_b = dec.evaluate([f.day for f in mv], [f.vel for f in mv], [block[f.key] for f in mv], [f.mask for f in mv])
out(f"[block, our evaluator] {len(mv)} files")
out(f"  mean R² over days: {r2_b:.4f}   ({r2_b:.6f})")
out("  per giorno: " + ", ".join(f"{k} {v:.4f}" for k, v in day_b.items()))
out("  per DoF (media giorni): " + ", ".join(f"{d} {v:.4f}" for d, v in zip(dec.DOF, dof_b.mean(0))))

# 2-3. streaming, official evaluator
os.environ["EVAL_DATA_PATH"] = str(Path.home())           # -> ~/000954/*minival/*.nwb
cfg = FalconConfig(task=FalconTask.h1)
for B in BATCHES:
    os.environ["PREDICTION_PATH_LOCAL"] = str(TMP / f"pred_b{B}.pkl")
    os.environ["GT_PATH"] = str(TMP / f"gt_b{B}.pkl")
    ts = time.time()
    ev = FalconEvaluator(eval_remote=False, split="h1", dataloader_workers=0)
    res = ev.evaluate(dec.make_streaming_decoder(MODEL, batch_size=B), phase="minival")
    preds = pickle.load(open(TMP / f"pred_b{B}.pkl", "rb"))["h1"]
    # batch predictions are padded to the longest file: trim to the file length
    stream = {f.key: np.asarray(preds[f.key])[: len(f.counts)] for f in mv}
    r2_s, day_s, dof_s = dec.evaluate([f.day for f in mv], [f.vel for f in mv], [stream[f.key] for f in mv], [f.mask for f in mv])
    dmax = max(np.abs(stream[k] - block[k]).max() for k in block)
    dmax_mask = max(np.abs(stream[f.key] - block[f.key])[f.mask].max() for f in mv)
    off = res["result"][0]["minival_split_h1"]          # evaluate() restituisce {"result": [...], "submission_result": ...}
    r2_off = off["Held In R2 Mean"]
    out(f"[streaming, official FALCON evaluator, batch {B}] {time.time() - ts:.0f} s")
    out(f"  risultato ufficiale: {off}")
    out(f"  official mean R²   : {r2_off:.4f}   ({r2_off:.6f})")
    out(f"  our evaluator on the streaming predictions: {r2_s:.4f}; per DoF: "
        + ", ".join(f"{d} {v:.4f}" for d, v in zip(dec.DOF, dof_s.mean(0))))
    out(f"  streaming - block, maximum prediction difference: all bins {dmax:.2e}, evaluated bins {dmax_mask:.2e}")
    ok = round(r2_off, 4) == round(r2_b, 4) and dmax < 1e-4
    out(f"  match the block result (R² to the 4th digit, diff < 1e-4): {'SÌ' if ok else 'NO'}")
out(f"tempo totale {time.time() - t0:.0f} s")
