"""Notebook di replica del 02_new (PCA, FA, GPFA) per le tre sessioni.

Tolte le sweep e le alternative: §1-§6.1 sensibilità a σ, §9.1 CV di K (K = 15), §9.5, §10.1 iterazioni,
§10.2 CV dei fattori (d = 25), §10.6, §10.7. La tabella del §10.4 si fa solo con d = 25.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from build_nb import build, header, FOOTER

W = Path.home() / "tesi_h1_work"
REP = W / "rep"
SIDS = ["19250108T110520", "19250113T120811", "19250120T115044"]
SKIP = {29, 30, 43, 44, 51, 52, 59, 60, 61, 62, 63, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83}
for sid in SIDS:
    n = build(W / "notebooks/.src_02_new.py", REP / sid / "02_rep.ipynb", skip=SKIP,
              insert_before={3: [header(sid, REP, 'OUT_NAME = "02_rep"')],
                             45: ["K_FA = 15                                               # REPLICA: il K scelto nel 02_new (§9.1), senza rifare la CV"],
                             55: ["K_GRID = [10]                                           # REPLICA: il K* del 02_new §9.6 (10), senza rifare la CV"],
                             64: ["D_GP = 25                                               # REPLICA: i fattori scelti nel 02_new (§10.2), senza rifare la CV"]},
              replace={7: [("edges = offset + np.arange(n_bins + 1) * BIN_S",
                            "edges = offset + (np.arange(n_bins + 1) - 1) * BIN_S    # REPLICA: bordi di FALCON, il bin k copre [t_k - 20 ms, t_k)")],
                       66: [("D_TAB = [10, 15, 20, 25, 30]", "D_TAB = [D_GP]                                          # REPLICA: solo il valore scelto")]},
              append=[FOOTER],
              title=f"# REPLICA del 02_new sulla sessione {sid}\n\nBin di FALCON; stessi parametri del 02_new "
                    f"(K = 15 per la FA, d = 25 per la GPFA), senza sweep né alternative. Generato da rep/make_02.py.")
    print(sid, n, "celle")
