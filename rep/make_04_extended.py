"""Extended 04 notebook on the reference session (night of 16 Nov). Usage: python make_04_extended.py [--trial]

As make_04.py (which is left untouched), with these differences:
- source ~/thesis_h1_git/notebooks/.src_04_deep.py (same 235 cells, same types as the source of 29 Oct);
- FALCON bins from io.load_session (`bins="falcon"` instead of `bins="legacy"` in the loading cell and in
  the cell of the EXTRA sessions of §14d); header WITHOUT falcon_patch.py (superseded: with the new io.py it would shift
  the bins a second time);
- cells removed: S2 | S3 | S10 | S11 | S6_ITER, i.e. §7-§8 (S7_8) and §13.5-§15 (S13B_15) are put back;
- values fixed as in the replication: N_ITER 1000, N_ITER_BEH 1000, BEST_CFG (0.05, 0.3, 0.0), N_EP_LF 200, 6 processes
  for LFADS; §3 and the §11b grid are NOT redone (chosen with the legacy bins);
- two insertions of make_04.py removed because they would be wrong with §7 included: GP_7B (§7b is there) and the line
  HI_ACC = {("PCA", 7): ...} of FIX_11B (it would overwrite the full table of §7a);
- outputs with new names: 04_rep_extended.ipynb / .out.ipynb / .pkl and figures in figures_extended/;
- hook for the English figures (thesis_h1.fig_en.install) in the first added cell, before any figure.
--check-cache (Step 0e): the same notebook without §7-§8 and §13.5-§15 (like the replication of 29 Oct, with GP_7B and
HI_ACC of make_04.py), which trains NOTHING: if the cells already run by the replication contain a CEBRA, LFADS or
GPFA model to train, it stops with "VERIFICA CACHE" and the number; the fingerprints do not change (only the code that
launches the trainings is touched, which is not part of the fingerprints). Outputs 04_check_cache.*, no figure in the repo.
--trial: separate cache (cache_trial_extended), 20 CEBRA steps (also for the variants with 500-5000 steps of §7a),
3 LFADS epochs, 3 GPFA iterations (also that of §7b), outputs 04_trial_extended.*.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from build_nb import build, parse
from cfg04 import S2, S3, S6_ITER, S10, S11
import ast

W = Path.home() / "thesis_h1_work"
REP = W / "rep"
SRC = Path.home() / "thesis_h1_git" / "notebooks" / ".src_04_deep.py"
SID = "19250108T110520"
SMOKE = "--trial" in sys.argv
CHECK = "--check-cache" in sys.argv
NAME = "04_trial_extended" if SMOKE else "04_check_cache" if CHECK else "04_rep_extended"
FIGDIR = "figures_trial_extended" if SMOKE else "figures_check_cache" if CHECK else "figures_extended"
FIG_RUN = "trial_04_extended" if SMOKE else "check_cache_04" if CHECK else "04_rep_extended_S1"
SKIP = S2 | S3 | S10 | S11 | S6_ITER
if CHECK:
    from cfg04 import S7_8, S13B_15
    SKIP = SKIP | S7_8 | S13B_15

cells = parse(SRC)
assert len(cells) == 235, len(cells)
lines = open(SRC, encoding="utf-8").read().split("\n")


def stmt(i, name):                                     # the statement of cell i that assigns name (as in make_04.py)
    text = cells[i][1]
    for nd in ast.parse(text).body:
        if isinstance(nd, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in nd.targets):
            return ast.get_source_segment(text, nd)
    raise KeyError(name)


def block(first, last):                                # the source lines from the one starting with first to the one starting with last
    a = next(k for k, l in enumerate(lines) if l.startswith(first))
    b = next(k for k in range(a, len(lines)) if lines[k].startswith(last))
    return "\n".join(lines[a:b + 1])


# As in make_04.py, identical text (the model fingerprints do not depend on these cells, but the values do)
LF_GLOBALS = "\n".join([
    "# REPLICATION: the LFADS settings defined in the removed cells of §10 (same values)",
    "from joblib.externals.loky import ProcessPoolExecutor",
    "POOL_LF = ProcessPoolExecutor(max_workers=6)                # REPLICATION: 6 processes instead of 4 (one thread each, as in 04): same results, more cores",
    stmt(124, "CACHE_LF"), "CACHE_LF.mkdir(parents=True, exist_ok=True)", stmt(124, "CLIP_LF"),
    "N_NEU = N_CH                                                # REPLICATION: only needed to build SYN_HP; n_in, n_out and the KL are replaced by REAL_HP",
    stmt(119, "SYN_HP"),
    "if SMOKE:\n    lfads_fit.__defaults__ = (None, 3, 200, 64, CLIP_LF, 'cpu', None, None)   # TRIAL: 3 epochs"
    "\n    SYN_HP = dict(SYN_HP, ramp_epochs=1)                     # TRIAL: 1-epoch ramps, otherwise with 3 epochs there is no better version",
])
# As in make_04.py, WITHOUT the line HI_ACC = {("PCA", 7): ...}: with §7a included, HI_ACC is its full table
FIX_11B = "\n".join([
    "# REPLICATION: the choices of §11b, without redoing the grid",
    "BEST_CFG = (0.05, 0.3, 0.0)                                 # dropout 0.05, KL × 0.3, CD 0.0",
    "N_EP_LF = 3 if SMOKE else 200                               # epochs for §12",
    block("SM_RATE = io.smooth_rates(s.spikes", "t_w = (win_ - a0_)"),
])

HEADER = f'''# ===== EXTENDED REPLICATION (added, night of 16 Nov): session {SID}, FALCON bins =====
# The FALCON bins come from io.load_session(bins="falcon") (io.py since 30 Oct); no falcon_patch.py.
SID = "{SID}"
REP = Path("{REP}") / SID                                  # results and figures of this session
REP.mkdir(parents=True, exist_ok=True)
SESS = ROOT / f"data/raw/sub-HumanPitt-held-in-calib/sub-HumanPitt-held-in-calib_ses-{{SID}}.nwb"
FIGS = REP / "{FIGDIR}"; FIGS.mkdir(exist_ok=True)          # new figures: the figures of 29 Oct stay untouched
OUT_NAME = "{NAME}"
SMOKE = {SMOKE}
from thesis_h1 import fig_en                                  # English versions: pickle and strings of every saved figure
print("fig_en:", fig_en.install("{FIG_RUN}"))
assert "bins" in io.load_session.__code__.co_varnames, "io.py without the bins parameter"'''

SMOKE_CEBRA = '''if SMOKE:                                                   # TRIAL: the variants with 500-5000 steps of §7a also run 20 steps
    _run_cebra_cached_full = run_cebra_cached
    def run_cebra_cached(jobs, fn=None):
        jobs = [dict(j, cfg=dict(j["cfg"], max_iterations=min(j["cfg"].get("max_iterations", 20), 20))) for j in jobs]
        return _run_cebra_cached_full(jobs, fn)'''

# Memory (added on 16 Nov after the first trial): the processes of the CEBRA pool (POOL, 2) and of the LFADS pool (POOL_LF, 6)
# stay alive for the whole notebook and grow with every training (in the trial ~3 GB each for CEBRA, ending up in
# swap, which sits on the same disk). Here they are renewed in blocks: same trainings, same order of results,
# same fingerprints (run_cebra and run_lfads_cached are not part of job_key / lf_key).
POOL_CEBRA_RENEW = """# EXTENDED REPLICATION (night of 16 Nov): the 2 CEBRA processes are renewed every 30 trainings (memory)
_run_cebra_pool = run_cebra
def run_cebra(jobs, fn=None):
    global POOL
    out = []
    for i_ in range(0, len(jobs), 30):
        out += _run_cebra_pool(jobs[i_:i_ + 30], fn)
        POOL.shutdown(wait=True)
        POOL = ProcessPoolExecutor(max_workers=2)
    return out"""
POOL_LF_RENEW = """# EXTENDED REPLICATION (night of 16 Nov): the 6 LFADS processes are renewed every 60 trainings (memory)
_run_lfads_cached_pool = run_lfads_cached
def run_lfads_cached(jobs):
    global POOL_LF
    out = []
    for i_ in range(0, len(jobs), 60):
        out += _run_lfads_cached_pool(jobs[i_:i_ + 60])
        POOL_LF.shutdown(wait=True)
        POOL_LF = ProcessPoolExecutor(max_workers=6)
    return out"""

ins = {3: [HEADER],
       27: [POOL_CEBRA_RENEW],
       135: [POOL_LF_RENEW],
       51: ["N_ITER = 20 if SMOKE else 1000                            # REPLICATION: the number of steps of §3.2"],
       76: ["N_ITER_BEH = 20 if SMOKE else 1000                        # REPLICATION: the number of steps of §6.2"],
       85: [SMOKE_CEBRA],
       134: [LF_GLOBALS],
       155: [FIX_11B],
       158: ["if SMOKE:\n    gpfa_cached.__defaults__ = (K_PCA, 3)                # TRIAL: 3 GPFA iterations"]}

replace = {
    0: [("# **Bin di prima del 30/10/2026 (`bins=\"legacy\"`).** Questo notebook carica i dati con",
         "# **Bin di prima del 30/10/2026 (`bins=\"legacy\"`).** *(Not in the extended replication of 16 Nov: here the data are loaded "
         "with `bins=\"falcon\"`; the paragraph applies to the original 04.)* The original 04 loads the data with")],
    4: [('s = io.load_session(SESS, bins="legacy")', 's = io.load_session(SESS, bins="falcon")')],
    38: [('CACHE = Path.home() / "thesis_h1_work" / "cache_04"',
          'CACHE = (REP / "cache_trial_extended") if SMOKE else Path.home() / "thesis_h1_work" / "cache_04"  # REPLICATION: in trial mode, separate cache')],
    90: [("x_dim=K_PCA, bin_width=20.0, em_max_iters=500, tau_init=100.0)",
          "x_dim=K_PCA, bin_width=20.0, em_max_iters=(3 if SMOKE else 500), tau_init=100.0)  # REPLICATION: 3 iterations in trial mode")],
    224: [('SESS.name.replace("19250108T110520", sid), bins="legacy")',
           'SESS.name.replace("19250108T110520", sid), bins="falcon")')],
}

if CHECK:
    GP_7B = ("# CACHE CHECK: from §7b, only the per-block GPFA (X_GP), as GP_7B of make_04.py\n"
             + block("Zc = StandardScaler().fit_transform(s.spikes", 'print(f"Fattori GPFA:'))
    NO_CEBRA = """# CACHE CHECK: no CEBRA training (run_cebra is not part of the fingerprints)
def run_cebra(jobs, fn=None):
    if len(jobs):
        raise RuntimeError(f"VERIFICA CACHE: {len(jobs)} CEBRA trainings to do")
    return []"""
    NO_LFADS = """# CACHE CHECK: no LFADS training (POOL_LF is not part of the fingerprints)
class _NoPool:
    def submit(self, *a, **k):
        raise RuntimeError("VERIFICA CACHE: LFADS training to do")
POOL_LF = _NoPool()"""
    ins[39] = [NO_CEBRA]
    ins[99] = [GP_7B]
    ins[135] = [NO_LFADS]
    ins[155] = [FIX_11B + '\nHI_ACC = {("PCA", 7): decode_map(INP_PC, emb_pca(7))}       # as in make_04.py']
    replace[157] = [("fits = Parallel(n_jobs=3)(delayed(gpfa_fit_whole)(seg_sets[i], x_dim, iters) for i in todo)",
                     "assert not todo, f'VERIFICA CACHE: {len(todo)} GPFA to fit'; fits = []")]

# The FOOTER of build_nb saves numbers, tables, dictionaries and arrays in a single pkl (~0.8-1 GB). In the trial of 16 Nov the
# disk filled up right there and the executed notebook was not saved. Here: first a small pkl (numbers, tables and
# the dictionaries smaller than 50 MB), then the full pkl as in the FOOTER, only if at least 2 GB would remain
# free afterwards; if the write fails the partial file is deleted, so that run_nb.py can save the executed notebook.
FOOTER_SAFE = '''# ===== EXTENDED REPLICATION (added): save the numbers and tables of the notebook (small pkl first, then the full one) =====
import pickle, numbers, os, shutil
_num = {k: float(v) for k, v in globals().items() if not k.startswith("_") and isinstance(v, numbers.Real) and not isinstance(v, bool)}
_tabs = {k: v for k, v in globals().items() if not k.startswith("_") and isinstance(v, pd.DataFrame)}
_dicts, _big = {}, []
for k, v in list(globals().items()):                         # result dictionaries (accuracies, measures): only if they can be pickled
    if not k.startswith("_") and isinstance(v, dict) and k.isupper():
        try:
            _b = pickle.dumps(v)
        except Exception:
            continue
        (_dicts.__setitem__(k, v) if len(_b) < 50_000_000 else _big.append(k))
        del _b
with open(REP / f"{OUT_NAME}_numeri.pkl", "wb") as f:
    pickle.dump(dict(numbers=_num, tables=_tabs, dicts=_dicts, dizionari_grandi_excluded=_big), f)
print(f"small pkl: {len(_num)} numbers, {len(_tabs)} tables, {len(_dicts)} dictionaries (excluded because > 50 MB: {_big})")
_arrs = {k: v for k, v in globals().items() if not k.startswith("_") and isinstance(v, np.ndarray) and v.size <= 2_000_000}
_need = (sum(v.nbytes for v in _arrs.values()) + 50e6 * len(_big)) / 1e9 + 0.3
_free = shutil.disk_usage(REP).free / 1e9
_path = REP / f"{OUT_NAME}.pkl"
if _free - _need < 2.0:
    print(f"full pkl NOT written: {_free:.1f} GB free, about {_need:.1f} would be needed leaving at least 2 GB")
else:
    try:
        _dicts_all = {**_dicts, **{k: globals()[k] for k in _big}}
        with open(_path, "wb") as f:
            pickle.dump(dict(numbers=_num, tables=_tabs, arrays=_arrs, dicts=_dicts_all), f)
        print(f"saved {len(_num)} numbers, {len(_tabs)} tables, {len(_arrs)} arrays, {len(_dicts_all)} dictionaries in {_path}")
    except OSError as e:
        if _path.exists():
            _path.unlink()
        print(f"full pkl NOT written ({e}); partial file deleted")
'''

if __name__ == "__main__":
    from build_nb import FOOTER
    out = REP / SID / f"{NAME}.ipynb"
    n = build(SRC, out, skip=SKIP, replace=replace, insert_before=ins, append=[FOOTER_SAFE],
              title=f"# EXTENDED REPLICATION of 04 on session {SID}\n\nFALCON bins (io.load_session, `bins=\"falcon\"`); "
                    "same hyperparameters as 04 and as the replication of 29 Oct, without the sweeps of §3 and §11b; with §7-§8 and "
                    "§13.5-§15. Generated by rep/make_04_extended.py (night of 16 Nov).")
    print(SID, n, "cells ->", out)
