"""Valutazione su minival del modello finale (ridge, storia 1000 ms): una volta sola, solo come numero da riportare.

Il modello è models/ridge_h50.npz così com'è (quello che va nel Docker): nessun riallenamento, nessuna scelta.
Tre calcoli:
  1. in blocco: predict su ogni file intero + il nostro evaluator (decoding.evaluate);
  2. in streaming con l'evaluator ufficiale di FALCON, batch 4 (come nell'immagine Docker / EvalAI);
  3. lo stesso con batch 1.
Controllo: R² uguali alla quarta cifra e predizioni bin per bin con differenza massima ~1e-5
(la storia in blocco è costruita in float32, lo streaming lavora in float64).
Uscita: run05/minival_h50.log (accanto al numero a 600 ms del commit c8fbd9b).
"""
import hashlib, os, pickle, sys, tempfile, time
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tesi_h1 import decoding as dec

MODEL = REPO / "models/ridge_h50.npz"
RAW = Path.home() / "000954"
LOG = REPO / "run05/minival_h50.log"
BATCHES = [4, 1]
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
assert H == 50 and W.shape == (dec.N_CH * H, 7)

out("=" * 100)
out(f"Valutazione minival del modello finale   {datetime.now():%Y-%m-%d %H:%M}")
out(f"modello        : models/ridge_h50.npz  md5 {hashlib.md5(MODEL.read_bytes()).hexdigest()}")
out(f"storia         : H = {H} bin = {H * dec.BIN_MS} ms; alpha 10^4.5 (commit 5a45607); allenata su tutto il calib held-in")
out(f"falcon_challenge: {version('falcon_challenge')}  ({Path(falcon_challenge.__file__).parent})")
out(f"batch size     : streaming con {BATCHES} (4 = BATCH_SIZE dell'immagine Docker)")
out(f"riferimento    : storia 600 ms (H = 30), minival R² 0.3831, commit c8fbd9b")
out("-" * 100)

# 1. in blocco
mv = dec.load_split(RAW, "held-in-minival")
mm =dec.RidgeHistory(H); mm.mu, mm.sd = mu, sd
block = {f.key: mm.predict(f.counts, W, b) for f in mv}
r2_b, day_b, dof_b = dec.evaluate([f.day for f in mv], [f.vel for f in mv], [block[f.key] for f in mv], [f.mask for f in mv])
out(f"[blocco, nostro evaluator] {len(mv)} file")
out(f"  R² medio sui giorni: {r2_b:.4f}   ({r2_b:.6f})")
out("  per giorno: " + ", ".join(f"{k} {v:.4f}" for k, v in day_b.items()))
out("  per DoF (media giorni): " + ", ".join(f"{d} {v:.4f}" for d, v in zip(dec.DOF, dof_b.mean(0))))

# 2-3. streaming, evaluator ufficiale
os.environ["EVAL_DATA_PATH"] = str(Path.home())           # -> ~/000954/*minival/*.nwb
cfg = FalconConfig(task=FalconTask.h1)
for B in BATCHES:
    os.environ["PREDICTION_PATH_LOCAL"] = str(TMP / f"pred_b{B}.pkl")
    os.environ["GT_PATH"] = str(TMP / f"gt_b{B}.pkl")
    ts = time.time()
    ev = FalconEvaluator(eval_remote=False, split="h1", dataloader_workers=0)
    res = ev.evaluate(dec.make_streaming_decoder(MODEL, batch_size=B), phase="minival")
    preds = pickle.load(open(TMP / f"pred_b{B}.pkl", "rb"))["h1"]
    # le predizioni del batch sono allungate al file più lungo: si taglia alla lunghezza del file
    stream = {f.key: np.asarray(preds[f.key])[: len(f.counts)] for f in mv}
    r2_s, day_s, dof_s = dec.evaluate([f.day for f in mv], [f.vel for f in mv], [stream[f.key] for f in mv], [f.mask for f in mv])
    dmax = max(np.abs(stream[k] - block[k]).max() for k in block)
    dmax_mask = max(np.abs(stream[f.key] - block[f.key])[f.mask].max() for f in mv)
    off = res["result"][0]["minival_split_h1"]          # evaluate() restituisce {"result": [...], "submission_result": ...}
    r2_off = off["Held In R2 Mean"]
    out(f"[streaming, evaluator ufficiale FALCON, batch {B}] {time.time() - ts:.0f} s")
    out(f"  risultato ufficiale: {off}")
    out(f"  R² medio ufficiale : {r2_off:.4f}   ({r2_off:.6f})")
    out(f"  nostro evaluator sulle predizioni in streaming: {r2_s:.4f}; per DoF: "
        + ", ".join(f"{d} {v:.4f}" for d, v in zip(dec.DOF, dof_s.mean(0))))
    out(f"  streaming - blocco, differenza massima delle predizioni: tutti i bin {dmax:.2e}, bin valutati {dmax_mask:.2e}")
    ok = round(r2_off, 4) == round(r2_b, 4) and dmax < 1e-4
    out(f"  coincidono con il blocco (R² alla 4a cifra, diff < 1e-4): {'SÌ' if ok else 'NO'}")
out(f"tempo totale {time.time() - t0:.0f} s")
