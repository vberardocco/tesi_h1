"""R9 checks (post hoc), step 2: calibration of the generator gain g (rules: docs/literature/rules_r9_checks.md).
For each shared fraction (0.167; 0.45; 0.65) and σ_lat (5, 10 bins): bisection on g, seed 0. Writes literature_cache/r9c/calibration.json.
Uso: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMBA_THREADING_LAYER=workqueue PYTHONPATH=src python scripts/literature_r9c_calibrate.py
"""
import json
import time
from pathlib import Path

import numpy as np

from thesis_h1 import literature2

C = Path.home() / "thesis_h1_work/literature_cache"
fr = np.load(C / "phase2/S1_freq_a.npy")
OUT = C / "r9c/calibration.json"
cal = json.loads(OUT.read_text()) if OUT.exists() else {}
for sl in (5, 10):
    for q in literature2.QUOTE_CHECK:
        k = f"{q}_{sl}"
        if k in cal:
            continue
        t = time.time()
        g, qm, history = literature2.calibrate_g(fr, q, sl, n_jobs=6)
        cal[k] = dict(share=q, sigma_lat=sl, g=g, share_fa=qm, history=history, seconds=time.time() - t)
        OUT.write_text(json.dumps(cal, indent=1))
        print(k, round(g, 4), round(qm, 4), f"{time.time() - t:.0f} s", flush=True)
