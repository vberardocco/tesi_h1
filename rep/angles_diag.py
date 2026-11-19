"""Diagnosis of the random-subspace comparison of the principal angles (angles.py), without redoing the angles.

For each session and method:
- the covariance used by the "from the covariance" chance level (cov_space of angles.py) and that of the data passed to
  the subspace estimation (the same matrix given to PCA, FA, GPFA): channels, participation ratio (Σλ)²/Σλ², fraction
  of variance in the first k eigenvalues, and the maximum difference between the two matrices;
- a sampling check: the covariance of 20000 vectors v = U S^1/2 r must return the starting covariance.
In addition, as a preview to decide (not used in the tables): the chance level with steeper covariances, namely that of
the condition means (as in Elsayed et al. 2016) and the shared covariance of the FA (L Lᵀ).

Usage: python angles_diag.py  ->  rep/angles/diagnosis.csv, rep/angles/preview_chance.csv
"""
import ast
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
SRC = (HERE / "angles.py").read_text(encoding="utf-8")
exec(SRC.split("# --- stime")[0])                            # same sessions (FALCON bins), same channels, same preprocessing
for nd in ast.parse(SRC).body:                                # and the same chance-level functions
    if isinstance(nd, ast.FunctionDef) and nd.name in ("cov_space", "cond_cov", "draw_angles", "random_angles"):
        exec(ast.get_source_segment(SRC, nd))

data = {(sid, None): spaces(sid, None) for sid in SIDS}
KS = {"PCA": (10, 25), "FA": (10, 15, 25), "GPFA": (10, 25)}
PREP = {"PCA": "FALCON 20 ms bins, Gaussian smoothing σ = 50 ms, Hz, per-channel z-score, all bins",
        "FA": "FALCON 20 ms bins, raw counts, per-channel z-score, all bins",
        "GPFA": "FALCON 20 ms bins, square root of the counts, 1 s windows from phase onset (119), no z-score"}


def fit_matrix(method, sid):
    """The matrix passed to the subspace estimation, rebuilt as in angles.py (spaces, fit_gpfa)."""
    Z, Xfa, wins = data[(sid, None)]
    if method == "PCA":
        return Z                                              # PCA(K).fit(Z)
    if method == "FA":
        return Xfa                                            # FactorAnalysis(K).fit(Xfa)
    C = S[sid].spikes[:, COMMON]                              # fit_gpfa: np.sqrt(C[start:stop]) per window
    return np.concatenate([np.sqrt(C[w["start"]:w["stop"]].astype(float)) for w in wins])


def spectrum(Cv, ks):
    ev = np.clip(np.linalg.eigvalsh(Cv)[::-1], 0, None)
    return dict(channels=Cv.shape[0], PR=ev.sum() ** 2 / (ev ** 2).sum(),
                **{f"primi {k} (%)": 100 * ev[:k].sum() / ev.sum() for k in ks})


rows = []
for meth, ks in KS.items():
    for sid in SIDS:
        C_rand = cov_space(meth, sid)
        C_fit = np.cov(fit_matrix(meth, sid), rowvar=False)
        ev, U = np.linalg.eigh(C_rand); M = U * np.sqrt(np.clip(ev, 0, None))   # as in random_angles
        V = M @ np.random.default_rng(1).normal(size=(len(ev), 20000))
        err = np.linalg.norm(np.cov(V) - C_rand) / np.linalg.norm(C_rand)
        for which, Cv in (("chance (cov_space)", C_rand), ("dati dei sottospazi", C_fit)):
            rows.append(dict(method=meth, session=NAME[sid], covarianza=which, **spectrum(Cv, (10, 15, 25)),
                             **{"differenza massima tra le due": np.abs(C_rand - C_fit).max(),
                                "campionamento: errore relativo della covarianza (20000 vettori)": err}))
        C_cond = cond_cov(meth, sid)                          # the condition-means chance level (angles.py)
        rows.append(dict(method=meth, session=NAME[sid], covarianza="medie per condizione", **spectrum(C_cond, (10, 15, 25)),
                         **{"differenza massima tra le due": np.nan,
                            "campionamento: errore relativo della covarianza (20000 vettori)": np.nan}))
diag = pd.DataFrame(rows)
diag.to_csv(OUT / "diagnosis.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
print(diag.round(4).to_string(index=False))
for m, p in PREP.items():
    print(f"preprocessing {m}: {p} (same for the chance level and for the subspaces)")


# --- preview: the chance level with steeper covariances (only to decide)
def cond_means(sid):                                          # condition means (phase × even/odd version) of 02_new, on the common channels
    Z, _, wins = data[(sid, None)]
    groups = {}
    for w in wins:
        groups.setdefault((w["tag"], w["trial"] % 2), []).append(Z[w["start"]:w["start"] + 50])
    return np.concatenate([np.mean([x for x in g if len(x) == 50], 0) for g in groups.values()])


def shared_cov(sid, K):
    L = FactorAnalysis(K, svd_method="lapack", random_state=0).fit(data[(sid, None)][1]).components_
    return L.T @ L


def null_from(Ca, Cb, K, draws=500, seed=0):
    ev, U = np.linalg.eigh((Ca + Cb) / 2); M = U * np.sqrt(np.clip(ev, 0, None))
    rng = np.random.default_rng(seed)
    return np.median([angles(orth(M @ rng.normal(size=(len(ev), K))), orth(M @ rng.normal(size=(len(ev), K)))).mean()
                      for _ in range(draws)])


a, b = SIDS[0], SIDS[1]
CM = {sid: np.cov(cond_means(sid), rowvar=False) for sid in (a, b)}
prev = []
for K in (10, 25):
    prev.append(dict(comparison=f"PCA, K = {K}", **{"caso attuale (tutti i bin)": null_from(cov_space("PCA", a), cov_space("PCA", b), K),
                     "caso dalle medie per condizione": null_from(CM[a], CM[b], K),
                     "PR medie per condizione (ref. / S2)": f"{spectrum(CM[a], ())['PR']:.1f} / {spectrum(CM[b], ())['PR']:.1f}"}))
for K in (10, 15, 25):
    SA, SB = shared_cov(a, K), shared_cov(b, K)
    prev.append(dict(comparison=f"FA, K = {K}", **{"caso attuale (tutti i bin)": null_from(cov_space("FA", a), cov_space("FA", b), K),
                     "caso dalla covarianza condivisa L Lᵀ": null_from(SA, SB, K),
                     "PR condivisa (ref. / S2)": f"{spectrum(SA, ())['PR']:.1f} / {spectrum(SB, ())['PR']:.1f}"}))
prev = pd.DataFrame(prev)
prev.to_csv(OUT / "preview_chance.csv", index=False)
print("\nPreview, ref.-S2 pair (mean angle, median over 500 draws):")
print(prev.round(1).to_string(index=False))
