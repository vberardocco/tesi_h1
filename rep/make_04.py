"""Notebook di replica del 04 (CEBRA e LFADS) per le tre sessioni. Uso: python make_04.py [--prova]

Celle tolte: vedi cfg04.py (esempi sintetici, validazioni dei passi, griglie, sensibilità, alternative, riepiloghi).
Valori fissati invece di ricalcolati: N_ITER = 1000 (§3.2), N_ITER_BEH = 1000 (§6.2), BEST_CFG = dropout 0.05,
KL × 0.3, CD 0.0 e N_EP_LF = 200 (§11b). Il §6 (CEBRA-Behavior, PLS, kin_r2) solo nella sessione di riferimento.
--prova: cache a parte, 20 passi di CEBRA, 3 epoche di LFADS, 3 iterazioni della GPFA (solo per controllare che giri).
"""
import ast, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from build_nb import build, header, FOOTER, parse
from cfg04 import skip

W = Path.home() / "tesi_h1_work"
REP = W / "rep"
SRC = W / "notebooks/.src_04_deep.py"
SIDS = ["19250108T110520", "19250113T120811", "19250120T115044"]
SMOKE = "--prova" in sys.argv
cells = parse(SRC)
lines = open(SRC, encoding="utf-8").read().split("\n")


def stmt(i, name):                                     # l'istruzione della cella i che assegna name
    text = cells[i][1]
    for nd in ast.parse(text).body:
        if isinstance(nd, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in nd.targets):
            return ast.get_source_segment(text, nd)
    raise KeyError(name)


def block(first, last):                                # le righe del sorgente da quella che inizia con first a quella che inizia con last
    a = next(k for k, l in enumerate(lines) if l.startswith(first))
    b = next(k for k in range(a, len(lines)) if lines[k].startswith(last))
    return "\n".join(lines[a:b + 1])


GP_7B = ("# REPLICA: dal §7b, solo la GPFA per blocco (X_GP), che il §12.2 usa come termine di confronto\n"
         + block("Zc = StandardScaler().fit_transform(s.spikes", 'print(f"Fattori GPFA:'))
LF_GLOBALS = "\n".join([
    "# REPLICA: le impostazioni di LFADS definite nelle celle tolte del §10 (stessi valori)",
    "from joblib.externals.loky import ProcessPoolExecutor",
    "POOL_LF = ProcessPoolExecutor(max_workers=6)                # REPLICA: 6 processi invece di 4 (un thread ciascuno, come nel 04): stessi risultati, più core",
    stmt(124, "CACHE_LF"), "CACHE_LF.mkdir(parents=True, exist_ok=True)", stmt(124, "CLIP_LF"),
    "N_NEU = N_CH                                                # REPLICA: serve solo a costruire SYN_HP; n_in, n_out e il KL li sostituisce REAL_HP",
    stmt(119, "SYN_HP"),
    "if SMOKE:\n    lfads_fit.__defaults__ = (None, 3, 200, 64, CLIP_LF, 'cpu', None, None)   # PROVA: 3 epoche"
    "\n    SYN_HP = dict(SYN_HP, ramp_epochs=1)                     # PROVA: rampe di 1 epoca, sennò con 3 epoche non c'è una versione migliore",
])
FIX_11B = "\n".join([
    "# REPLICA: le scelte del §11b, senza rifare la griglia",
    "BEST_CFG = (0.05, 0.3, 0.0)                                 # dropout 0.05, KL × 0.3, CD 0.0",
    "N_EP_LF = 3 if SMOKE else 200                               # epoche per il §12",
    "HI_ACC = {(\"PCA\", 7): decode_map(INP_PC, emb_pca(7))}       # REPLICA: dal §7a, solo la riga della PCA che il §12.2 usa",
    block("SM_RATE = io.smooth_rates(s.spikes", "t_w = (win_ - a0_)"),
])

for sid in SIDS:
    ref = sid == SIDS[0]
    out = REP / sid / ("04_prova.ipynb" if SMOKE else "04_rep.ipynb")
    ins = {3: [header(sid, REP, f'OUT_NAME = "{"04_prova" if SMOKE else "04_rep"}"\nSMOKE = {SMOKE}')],
           51: ["N_ITER = 20 if SMOKE else 1000                            # REPLICA: il numero di passi del §3.2"],
           99: [GP_7B],
           134: [LF_GLOBALS],
           155: [FIX_11B],
           158: ["if SMOKE:\n    gpfa_cached.__defaults__ = (K_PCA, 3)                # PROVA: 3 iterazioni della GPFA"]}
    if ref:
        ins[76] = ["N_ITER_BEH = 20 if SMOKE else 1000                        # REPLICA: il numero di passi del §6.2"]
    n = build(SRC, out, skip=skip(ref),
              replace={38: [('CACHE = Path.home() / "tesi_h1_work" / "cache_04"',
                             'CACHE = (REP / "cache_prova") if SMOKE else Path.home() / "tesi_h1_work" / "cache_04"  # REPLICA: in prova, cache a parte')]},
              insert_before=ins, append=[FOOTER],
              title=f"# REPLICA del 04 sulla sessione {sid}\n\nBin di FALCON; stessi iperparametri del 04, senza sweep né "
                    f"alternative ({'con' if ref else 'senza'} il §6). Generato da rep/make_04.py.")
    print(sid, n, "celle ->", out.name)
