"""Angoli principali tra i sottospazi della sessione di riferimento e delle due sessioni nuove.

Sottospazi (bin di FALCON, canali attivi in TUTTE e tre le sessioni, così gli spazi sono gli stessi):
- PCA, K = 10 e 25: firing rate smussati (σ = 50 ms), z-score per canale, tutti i bin (come il 02_new §6);
- FA, K = 10, 15 (il K* del 02_new) e 25: conteggi grezzi z-scorati (02_new §9);
- GPFA, d = 25 (02_new §10): radice dei conteggi nelle finestre da 1 s dall'inizio delle fasi; anche le sue prime
  10 direzioni (base ortonormale ordinata per varianza, dalla SVD di C).

Termini di confronto:
- rumore: le due metà della stessa sessione (trial 1-2, 5-6, ... contro 3-4, 7-8, ...: entrambe le versioni del task
  in ogni metà), stesso metodo e stesso K;
- caso: sottospazi casuali presi dalla covarianza dei dati (Elsayed et al. 2016): v = U S^1/2 r, r gaussiano,
  poi ortonormalizzati; covarianza = media di quelle delle due sessioni confrontate, nello spazio del metodo.
  Sottospazi uniformi, con ~150 canali, sarebbero quasi ortogonali per costruzione.

Uso: python angles.py   ->  rep/angles/angles.pkl e figure
"""
import os, sys, time, pickle, contextlib, logging, warnings
from pathlib import Path
import numpy as np
from scipy.linalg import subspace_angles, qr
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import PCA, FactorAnalysis
from joblib import Parallel, delayed
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")
W = Path.home() / "tesi_h1_work"
sys.path.insert(0, str(W / "src"))
from tesi_h1 import io
OUT = W / "rep" / "angles"; (OUT / "figures").mkdir(parents=True, exist_ok=True)
SIDS = ["19250108T110520", "19250113T120811", "19250120T115044"]
NAME = {"19250108T110520": "rif. S1", "19250113T120811": "S2", "19250120T115044": "S5"}
PATH = lambda sid: W / f"data/raw/sub-HumanPitt-held-in-calib/sub-HumanPitt-held-in-calib_ses-{sid}.nwb"
SESS, FIGS = PATH(SIDS[0]), OUT / "figures"
exec(open(W / "rep/falcon_patch.py", encoding="utf-8").read())   # io con i bin di FALCON
T0 = time.time()
def log(m): print(f"[{(time.time() - T0) / 60:5.1f} min] {m}", flush=True)

BIN_S, SIGMA_BINS, N_DRAWS = 0.02, 50 / 20, 500
S = {sid: io.load_session(PATH(sid)) for sid in SIDS}
rate = {sid: s.spikes.sum(0) / (s.n_bins * BIN_S) for sid, s in S.items()}
COMMON = np.flatnonzero(np.all([rate[sid] >= 0.5 for sid in SIDS], axis=0))
log(f"canali attivi in tutte e tre le sessioni: {len(COMMON)}")


def half_of(trial):                                     # metà A: trial 1-2, 5-6, ...; metà B: 3-4, 7-8, ...
    return ((trial - 1) // 2) % 2 == 0


def zs(X):
    sd = X.std(0); sd[sd == 0] = 1
    return (X - X.mean(0)) / sd


def spaces(sid, part):
    """I dati nello spazio di ogni metodo, per la sessione intera o per una metà (part = None, 'A', 'B')."""
    s = S[sid]; C = s.spikes[:, COMMON].astype(float)
    rows = np.ones(s.n_bins, bool) if part is None else (half_of(s.trial_num) == (part == "A"))
    Z = zs(gaussian_filter1d(C, SIGMA_BINS, axis=0, mode="nearest") / BIN_S)[rows]
    Xfa = zs(C)[rows]
    wins = [w for w in io.phase_windows(s, win_s=1.0, align="start")
            if part is None or half_of(np.array([w["trial"]]))[0] == (part == "A")]
    return Z, Xfa, wins


def fit_gpfa(sid, wins, d=25, iters=500, tau_init=100.0):   # come fit_gpfa del 02_new (§10)
    from elephant.gpfa import gpfa_core
    import elephant.utils; elephant.utils.logger.setLevel(logging.ERROR)
    C = S[sid].spikes[:, COMMON]
    seqs = np.array([(w["stop"] - w["start"], np.sqrt(C[w["start"]:w["stop"]].T.astype(float))) for w in wins],
                    dtype=[("T", int), ("y", "O")])
    np.random.seed(0)
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        params, _ = gpfa_core.fit(seqs, x_dim=d, bin_width=BIN_S * 1000, em_max_iters=iters, tau_init=tau_init)
    return params["C"]


def orth(M):
    return qr(M, mode="economic")[0]


def gp_basis(Cg, k):                                   # base ortonormale della GPFA, direzioni ordinate per varianza
    U, sv, _ = np.linalg.svd(Cg, full_matrices=False)
    return U[:, :k]


def angles(A, B):
    return np.sort(np.degrees(subspace_angles(A, B)))


# --- stime
JOBS = [(sid, part) for sid in SIDS for part in (None, "A", "B")]
data = {j: spaces(*j) for j in JOBS}
log("dati pronti; GPFA: 9 stime (3 sessioni × intera e due metà)")
gp = dict(zip(JOBS, Parallel(n_jobs=3)(delayed(fit_gpfa)(sid, data[(sid, part)][2]) for sid, part in JOBS)))
log("GPFA stimate")
BASES = {}
for j in JOBS:
    Z, Xfa, _ = data[j]
    for K in (10, 25):
        BASES[("PCA", K) + j] = PCA(K).fit(Z).components_.T
    for K in (10, 15, 25):
        BASES[("FA", K) + j] = orth(FactorAnalysis(K, svd_method="lapack", random_state=0).fit(Xfa).components_.T)
    for K in (10, 25):
        BASES[("GPFA", K) + j] = gp_basis(gp[j], K)
log("PCA e FA stimate")


def cov_space(method, sid):
    Z, Xfa, wins = data[(sid, None)]
    if method == "PCA":
        return np.cov(Z, rowvar=False)
    if method == "FA":
        return np.cov(Xfa, rowvar=False)
    C = S[sid].spikes[:, COMMON]
    return np.cov(np.sqrt(np.concatenate([C[w["start"]:w["stop"]] for w in wins]).astype(float)), rowvar=False)


def random_angles(method, K, sa, sb, seed=0):          # Elsayed et al. 2016: sottospazi casuali dalla covarianza dei dati
    ev, U = np.linalg.eigh((cov_space(method, sa) + cov_space(method, sb)) / 2)
    ev = np.clip(ev, 0, None); M = U * np.sqrt(ev)
    rng = np.random.default_rng(seed)
    return np.array([angles(orth(M @ rng.normal(size=(len(ev), K))), orth(M @ rng.normal(size=(len(ev), K))))
                     for _ in range(N_DRAWS)])


METHODS = [("PCA", 10), ("PCA", 25), ("FA", 10), ("FA", 15), ("FA", 25), ("GPFA", 10), ("GPFA", 25)]
PAIRS = [(SIDS[0], SIDS[1]), (SIDS[0], SIDS[2]), (SIDS[1], SIDS[2])]
RES = {}
for meth, K in METHODS:
    r = dict(pairs={}, noise={}, random={})
    for a, b in PAIRS:
        r["pairs"][(a, b)] = angles(BASES[(meth, K, a, None)], BASES[(meth, K, b, None)])
        rnd = random_angles(meth, K, a, b)
        r["random"][(a, b)] = dict(p5=np.percentile(rnd, 5, 0), p50=np.percentile(rnd, 50, 0), p95=np.percentile(rnd, 95, 0))
    for sid in SIDS:
        r["noise"][sid] = angles(BASES[(meth, K, sid, "A")], BASES[(meth, K, sid, "B")])
    RES[(meth, K)] = r
    log(f"{meth} K={K}: angolo medio rif-S2 {r['pairs'][PAIRS[0]].mean():.1f}°, rif-S5 {r['pairs'][PAIRS[1]].mean():.1f}°, "
        f"metà-metà {np.mean([v.mean() for v in r['noise'].values()]):.1f}°, caso (mediana) {r['random'][PAIRS[0]]['p50'].mean():.1f}°")

pickle.dump(dict(RES=RES, COMMON=COMMON, SIDS=SIDS, N_DRAWS=N_DRAWS), open(OUT / "angles.pkl", "wb"))

# --- figure: una per metodo e K
for (meth, K), r in RES.items():
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    idx = np.arange(1, K + 1)
    for (a, b), col in zip(PAIRS, ["C0", "C1", "C2"]):
        ax.plot(idx, r["pairs"][(a, b)], "o-", ms=3.5, color=col, label=f"{NAME[a]} – {NAME[b]}")
    rr = r["random"][PAIRS[0]]
    ax.fill_between(idx, rr["p5"], rr["p95"], color="0.85", label="caso dalla covarianza (5°-95° perc.)")
    ax.plot(idx, rr["p50"], color="0.5", lw=1)
    nm = np.mean([v for v in r["noise"].values()], 0)
    ax.plot(idx, nm, "k--", lw=1.2, label="due metà della stessa sessione (media)")
    ax.set(xlabel="angolo principale", ylabel="gradi", ylim=(0, 90),
           title=f"{meth}, K = {K}: angoli principali tra sessioni ({len(COMMON)} canali comuni)")
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout(); fig.savefig(OUT / "figures" / f"angles_{meth}_{K}.png", dpi=110); plt.close(fig)
log("figure salvate; fine")
