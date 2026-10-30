"""Diagnosi del confronto casuale degli angoli principali (angles.py), senza rifare gli angoli.

Per ogni sessione e metodo:
- la covarianza usata dal caso "dalla covarianza" (cov_space di angles.py) e quella dei dati passati alla stima dei
  sottospazi (la stessa matrice data a PCA, FA, GPFA): canali, participation ratio (Σλ)²/Σλ², frazione di varianza
  nei primi k autovalori, e la differenza massima tra le due matrici;
- un controllo del campionamento: la covarianza di 20000 vettori v = U S^1/2 r deve ridare la covarianza di partenza.
In più, come anteprima per decidere (non entra nelle tabelle): il caso con covarianze più ripide, cioè quella delle
medie per condizione (come in Elsayed et al. 2016) e quella condivisa della FA (L Lᵀ).

Uso: python angles_diag.py  ->  rep/angles/diagnosi.csv, rep/angles/anteprima_caso.csv
"""
import ast
from pathlib import Path
import numpy as np
import pandas as pd

HERE = Path(__file__).parent
SRC = (HERE / "angles.py").read_text(encoding="utf-8")
exec(SRC.split("# --- stime")[0])                            # stesse sessioni (bin di FALCON), stessi canali, stesso preprocessing
for nd in ast.parse(SRC).body:                                # e le stesse funzioni del caso
    if isinstance(nd, ast.FunctionDef) and nd.name in ("cov_space", "cond_cov", "draw_angles", "random_angles"):
        exec(ast.get_source_segment(SRC, nd))

data = {(sid, None): spaces(sid, None) for sid in SIDS}
KS = {"PCA": (10, 25), "FA": (10, 15, 25), "GPFA": (10, 25)}
PREP = {"PCA": "bin FALCON 20 ms, smoothing gaussiano σ = 50 ms, Hz, z-score per canale, tutti i bin",
        "FA": "bin FALCON 20 ms, conteggi grezzi, z-score per canale, tutti i bin",
        "GPFA": "bin FALCON 20 ms, radice dei conteggi, finestre da 1 s dall'inizio delle fasi (119), niente z-score"}


def fit_matrix(method, sid):
    """La matrice passata alla stima dei sottospazi, ricostruita come in angles.py (spaces, fit_gpfa)."""
    Z, Xfa, wins = data[(sid, None)]
    if method == "PCA":
        return Z                                              # PCA(K).fit(Z)
    if method == "FA":
        return Xfa                                            # FactorAnalysis(K).fit(Xfa)
    C = S[sid].spikes[:, COMMON]                              # fit_gpfa: np.sqrt(C[start:stop]) per finestra
    return np.concatenate([np.sqrt(C[w["start"]:w["stop"]].astype(float)) for w in wins])


def spectrum(Cv, ks):
    ev = np.clip(np.linalg.eigvalsh(Cv)[::-1], 0, None)
    return dict(canali=Cv.shape[0], PR=ev.sum() ** 2 / (ev ** 2).sum(),
                **{f"primi {k} (%)": 100 * ev[:k].sum() / ev.sum() for k in ks})


rows = []
for meth, ks in KS.items():
    for sid in SIDS:
        C_rand = cov_space(meth, sid)
        C_fit = np.cov(fit_matrix(meth, sid), rowvar=False)
        ev, U = np.linalg.eigh(C_rand); M = U * np.sqrt(np.clip(ev, 0, None))   # come random_angles
        V = M @ np.random.default_rng(1).normal(size=(len(ev), 20000))
        err = np.linalg.norm(np.cov(V) - C_rand) / np.linalg.norm(C_rand)
        for which, Cv in (("caso (cov_space)", C_rand), ("dati dei sottospazi", C_fit)):
            rows.append(dict(metodo=meth, sessione=NAME[sid], covarianza=which, **spectrum(Cv, (10, 15, 25)),
                             **{"differenza massima tra le due": np.abs(C_rand - C_fit).max(),
                                "campionamento: errore relativo della covarianza (20000 vettori)": err}))
        C_cond = cond_cov(meth, sid)                          # il caso delle medie per condizione (angles.py)
        rows.append(dict(metodo=meth, sessione=NAME[sid], covarianza="medie per condizione", **spectrum(C_cond, (10, 15, 25)),
                         **{"differenza massima tra le due": np.nan,
                            "campionamento: errore relativo della covarianza (20000 vettori)": np.nan}))
diag = pd.DataFrame(rows)
diag.to_csv(OUT / "diagnosi.csv", index=False)
pd.set_option("display.width", 250); pd.set_option("display.max_columns", 20)
print(diag.round(4).to_string(index=False))
for m, p in PREP.items():
    print(f"preprocessing {m}: {p} (uguale per il caso e per i sottospazi)")


# --- anteprima: il caso con covarianze più ripide (solo per decidere)
def cond_means(sid):                                          # medie per condizione (fase × versione pari/dispari) del 02_new, sui canali comuni
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
    prev.append(dict(confronto=f"PCA, K = {K}", **{"caso attuale (tutti i bin)": null_from(cov_space("PCA", a), cov_space("PCA", b), K),
                     "caso dalle medie per condizione": null_from(CM[a], CM[b], K),
                     "PR medie per condizione (rif. / S2)": f"{spectrum(CM[a], ())['PR']:.1f} / {spectrum(CM[b], ())['PR']:.1f}"}))
for K in (10, 15, 25):
    SA, SB = shared_cov(a, K), shared_cov(b, K)
    prev.append(dict(confronto=f"FA, K = {K}", **{"caso attuale (tutti i bin)": null_from(cov_space("FA", a), cov_space("FA", b), K),
                     "caso dalla covarianza condivisa L Lᵀ": null_from(SA, SB, K),
                     "PR condivisa (rif. / S2)": f"{spectrum(SA, ())['PR']:.1f} / {spectrum(SB, ())['PR']:.1f}"}))
prev = pd.DataFrame(prev)
prev.to_csv(OUT / "anteprima_caso.csv", index=False)
print("\nAnteprima, coppia rif.-S2 (angolo medio, mediana su 500 estrazioni):")
print(prev.round(1).to_string(index=False))
