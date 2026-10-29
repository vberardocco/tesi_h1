"""Controllo sul riconoscimento delle fasi del 04 (sessione di riferimento, bin di io.py come le mappe salvate).

Si rileggono dalla cache le mappe di CEBRA-Time e CEBRA-Behavior (nessun allenamento: run_cebra è bloccata),
si rifanno PLS, PCA, tetto e orologio (pochi secondi), più i controlli nuovi:
1. livello del caso (1/K e classe più frequente), accuratezza semplice e bilanciata;
2. matrici di confusione: errori tra fasi adiacenti o sparsi;
3. contesto temporale: lo stesso classificatore con 1000 ms di storia causale (come la ridge);
4. solo il tempo dall'inizio del trial.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from build_nb import build, parse

W = Path.home() / "tesi_h1_work"
SRC = W / "notebooks/.src_04_deep.py"
OUT = W / "rep/fasi/fasi_04.ipynb"
cells = parse(SRC)
KEEP = {2, 4, 8, 13, 26, 38, 40, 52, 73}
SKIP = set(range(len(cells))) - KEEP - {i for i, (k, _) in enumerate(cells) if k == "md" and i < 3}

GUARD = '''# CONTROLLO FASI: nessun allenamento. Se una mappa non è in cache, ci si ferma invece di allenare.
def run_cebra(jobs, fn=None):
    if jobs:
        raise RuntimeError(f"{len(jobs)} allenamenti non trovati in cache: il controllo non deve allenare")
    return []'''

CEBRA_B = '''# dal §6.3 del 04 (senza UMAP e senza le misure del §6.3 che non servono qui): CEBRA-Behavior dalla cache e PLS
N_ITER_BEH = 1000
''' + "\n".join(cells[77][1].split("\n")[:11])

ANALYSIS = r'''# ===== CONTROLLO FASI =====
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
PH = list(io.MOVEMENT_PHASES); K = len(PH)
y = phase[bins].astype(str)

def predict(inp, embed):                                    # come decode_map, ma restituisce le previsioni di ogni punto
    b, folds = inp
    pred = np.empty(len(b), dtype=object)
    for Xtr, Xte, trn, tst in folds:
        Etr, Ete = embed(Xtr, Xte)
        pred[tst] = KNeighborsClassifier(n_neighbors=K_NN).fit(Etr, phase[b[trn]]).predict(Ete)
    return pred.astype(str)

# --- 3. contesto temporale: conteggi grezzi z-scorati, PCA a 25 per blocco (bin di movimento dei trial di training)
Zc = StandardScaler().fit_transform(s.spikes[:, ACTIVE].astype(float))
PCA_C = [PCA(K_PCA).fit(Zc[bins_all[~np.isin(trial[bins_all], tt)]]) for tt in TEST_TRIALS]
XC = [p.transform(Zc) for p in PCA_C]
def causal_mean(X, L):                                      # media dei bin t-L+1 ... t (all'inizio della sessione: quelli che ci sono)
    c = np.cumsum(np.vstack([np.zeros((1, X.shape[1])), X]), axis=0)
    t = np.arange(len(X)); a = np.maximum(t - L + 1, 0)
    return (c[t + 1] - c[a]) / (t + 1 - a)[:, None]
def causal_stack(X, n_sub=10, sub=5):                       # 10 medie causali da 100 ms, dalla più recente: 1000 ms, 250 numeri
    m = causal_mean(X, sub)
    return np.hstack([np.vstack([np.repeat(m[:1], j * sub, 0), m[:len(m) - j * sub]]) for j in range(n_sub)])

clock = np.zeros(n_bins)
for t in TRIALS:
    clock[trial == t] = np.arange((trial == t).sum()) * io.BIN_S

METHODS = {                                                 # nome -> lista di (input, mappa), una per seme
    "PCA, d = 3": [(INP_PC, emb_pca(3))],
    "tetto: 25 PC smussati, 1 bin": [(INP_PC, emb_none)],
    **{f"CEBRA-Time, {od} dimensioni": [(fold_inp(lambda k, od=od, sd=sd: MAPS4[(od, sd, k)]), emb_none) for sd in SEEDS] for od in (3, 4)},
    **{f"CEBRA-Behavior, {od} dimensioni": [(fold_inp(lambda k, od=od, sd=sd: MAPS6[(od, sd, k)]), emb_none) for sd in SEEDS] for od in (3, 4)},
    **{f"PLS, {nc} componenti": [(fold_inp(lambda k, nc=nc: PLS_F[nc][k].transform(X25[k])), emb_none)] for nc in (2, 3)},
    "orologio (solo il tempo dall'inizio del trial)": [(fold_inp(lambda k: clock[:, None]), emb_none)],
    "contesto: 25 PC dei conteggi, 1 bin (causale)": [(fold_inp(lambda k: XC[k]), emb_none)],
    "contesto: 25 PC dei conteggi, media causale 1000 ms": [(fold_inp(lambda k: causal_mean(XC[k], 50)), emb_none)],
    "contesto: 25 PC dei conteggi, 10 × 100 ms causali (250 numeri)": [(fold_inp(lambda k: causal_stack(XC[k])), emb_none)],
    "contesto: 25 PC smussati, 10 × 100 ms causali (250 numeri)": [(fold_inp(lambda k: causal_stack(X25[k])), emb_none)],
}
PRED = {name: [predict(i_, e_) for i_, e_ in lst] for name, lst in METHODS.items()}

def adjacent_share(yt, yp):                                 # quota degli errori che cadono su una fase adiacente nell'ordine del task
    i_t = np.array([PH.index(v) for v in yt]); i_p = np.array([PH.index(v) for v in yp])
    err = i_t != i_p
    obs = np.mean(np.abs(i_t[err] - i_p[err]) == 1)
    n_adj = np.array([(i > 0) + (i < K - 1) for i in range(K)])  # vicine di ogni fase (Reach e Release ne hanno una)
    exp = np.mean(n_adj[i_t[err]] / (K - 1))                # se gli errori fossero sparsi a caso tra le altre 7 fasi
    return obs * 100, exp * 100

counts = pd.Series(y).value_counts()
rows = {}
for name, preds in PRED.items():
    bal = [balanced_accuracy_score(y, p) * 100 for p in preds]
    acc = [np.mean(y == p) * 100 for p in preds]
    adj = np.array([adjacent_share(y, p) for p in preds])
    rows[name] = {"semi": len(preds), "accuratezza bilanciata (%)": mean_range(bal) if len(bal) > 1 else f"{bal[0]:.1f}",
                  "accuratezza semplice (%)": mean_range(acc) if len(acc) > 1 else f"{acc[0]:.1f}",
                  "errori su fasi adiacenti (%)": f"{adj[:, 0].mean():.0f}",
                  "se sparsi a caso (%)": f"{adj[:, 1].mean():.0f}"}
TAB_FASI = pd.DataFrame(rows).T
TAB_FASI.loc["caso: 1/K"] = ["", f"{100 / K:.1f}", "", "", ""]
TAB_FASI.loc["caso: classe più frequente"] = ["", f"{100 / K:.1f}", f"{counts.max() / len(y) * 100:.1f} ({counts.idxmax()})", "", ""]
display(TAB_FASI)
print(f"Punti di valutazione: {len(y)}; per fase: {counts.reindex(PH).to_dict()}")
print("Nota: la balanced accuracy di un classificatore che dice sempre la classe più frequente è 1/K = 12.5%.")

# --- 2. matrici di confusione (righe: fase vera, normalizzate; somma sui semi)
SHOW = ["CEBRA-Time, 4 dimensioni", "CEBRA-Behavior, 4 dimensioni", "PLS, 3 componenti",
        "orologio (solo il tempo dall'inizio del trial)", "tetto: 25 PC smussati, 1 bin",
        "contesto: 25 PC dei conteggi, 10 × 100 ms causali (250 numeri)"]
fig, ax = plt.subplots(2, 3, figsize=(17, 10.5))
CM = {}
for a_, name in zip(ax.ravel(), SHOW):
    cm = sum(confusion_matrix(y, p, labels=PH) for p in PRED[name]).astype(float)
    cm = cm / cm.sum(1, keepdims=True) * 100; CM[name] = cm
    a_.imshow(cm, cmap="Blues", vmin=0, vmax=100)
    for i in range(K):
        for j in range(K):
            a_.text(j, i, f"{cm[i, j]:.0f}", ha="center", va="center", fontsize=7.5, color="w" if cm[i, j] > 55 else "k")
    a_.set(xticks=range(K), yticks=range(K), xticklabels=PH, yticklabels=PH, title=name[:60])
    a_.tick_params(axis="x", rotation=45, labelsize=8); a_.tick_params(axis="y", labelsize=8); a_.grid(False)
fig.supxlabel("fase prevista"); fig.supylabel("fase vera (righe in %)")
fig.tight_layout(); fig.savefig(FIGS / "fasi_confusione.png", bbox_inches="tight"); plt.show()

import pickle
pickle.dump(dict(TAB_FASI=TAB_FASI, CM=CM, PRED=PRED, y=y, PH=PH), open(FIGS.parent / "fasi_04.pkl", "wb"))
'''

HEAD = '''# CONTROLLO FASI: cartelle di uscita (il 04 originale resta intatto)
FIGS = Path.home() / "tesi_h1_work/rep/fasi/figures"; FIGS.mkdir(parents=True, exist_ok=True)'''

n = build(SRC, OUT, skip=SKIP,
          replace={8: [('    ACC[f"Isomap, d = {d}"] = decode_map(INP_PC, emb_isomap(d))\n', ''),
                       ('    ACC[f"UMAP, d = {d}"] = Parallel(n_jobs=5)(delayed(decode_map)(INP_PC, emb_umap(d, sd)) for sd in SEEDS)  # 5 semi in parallelo\n', '')]},
          insert_before={3: [HEAD], 38: [GUARD], 52: ["N_ITER = 1000                                               # il numero di passi del §3.2"],
                         74: [CEBRA_B]},
          append=[ANALYSIS],
          title="# Controllo sul riconoscimento delle fasi del 04 (sessione di riferimento)\n\nMappe rilette dalla cache del 04 "
                "originale (bin di io.py), nessun allenamento. Generato da rep/make_fasi.py.")
print(n, "celle ->", OUT)
