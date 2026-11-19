"""Check on the phase decoding of 04 (reference session, io.py bins as in the saved maps).

The CEBRA-Time and CEBRA-Behavior maps are reread from the cache (no training: run_cebra is blocked),
PLS, PCA, ceiling and clock are redone (a few seconds), plus the new checks:
1. chance level (1/K and most frequent class), plain and balanced accuracy;
2. confusion matrices: errors between adjacent phases or scattered;
3. temporal context: the same classifier with 1000 ms of causal history (as the ridge);
4. only the time since trial onset.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
from build_nb import build, parse

W = Path.home() / "thesis_h1_work"
SRC = W / "notebooks/.src_04_deep.py"
OUT = W / "rep/phases/phases_04.ipynb"
cells = parse(SRC)
KEEP = {2, 4, 8, 13, 26, 38, 40, 52, 73}
SKIP = set(range(len(cells))) - KEEP - {i for i, (k, _) in enumerate(cells) if k == "md" and i < 3}

GUARD = '''# PHASE CHECK: no training. If a map is not in the cache, stop instead of training.
def run_cebra(jobs, fn=None):
    if jobs:
        raise RuntimeError(f"{len(jobs)} trainings not found in cache: the check must not train")
    return []'''

CEBRA_B = '''# from §6.3 of 04 (without UMAP and without the §6.3 measures that are not needed here): CEBRA-Behavior from the cache and PLS
N_ITER_BEH = 1000
''' + "\n".join(cells[77][1].split("\n")[:11])

ANALYSIS = r'''# ===== PHASE CHECK =====
from sklearn.metrics import confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
PH = list(io.MOVEMENT_PHASES); K = len(PH)
y = phase[bins].astype(str)

def predict(inp, embed):                                    # as decode_map, but returns the predictions for each point
    b, folds = inp
    pred = np.empty(len(b), dtype=object)
    for Xtr, Xte, trn, tst in folds:
        Etr, Ete = embed(Xtr, Xte)
        pred[tst] = KNeighborsClassifier(n_neighbors=K_NN).fit(Etr, phase[b[trn]]).predict(Ete)
    return pred.astype(str)

# --- 3. temporal context: z-scored raw counts, 25-component PCA per block (movement bins of the training trials)
Zc = StandardScaler().fit_transform(s.spikes[:, ACTIVE].astype(float))
PCA_C = [PCA(K_PCA).fit(Zc[bins_all[~np.isin(trial[bins_all], tt)]]) for tt in TEST_TRIALS]
XC = [p.transform(Zc) for p in PCA_C]
def causal_mean(X, L):                                      # mean of bins t-L+1 ... t (at the start of the session: those available)
    c = np.cumsum(np.vstack([np.zeros((1, X.shape[1])), X]), axis=0)
    t = np.arange(len(X)); a = np.maximum(t - L + 1, 0)
    return (c[t + 1] - c[a]) / (t + 1 - a)[:, None]
def causal_stack(X, n_sub=10, sub=5):                       # 10 causal means of 100 ms, from the most recent: 1000 ms, 250 values
    m = causal_mean(X, sub)
    return np.hstack([np.vstack([np.repeat(m[:1], j * sub, 0), m[:len(m) - j * sub]]) for j in range(n_sub)])

clock = np.zeros(n_bins)
for t in TRIALS:
    clock[trial == t] = np.arange((trial == t).sum()) * io.BIN_S

METHODS = {                                                 # name -> list of (input, map), one per seed
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

def adjacent_share(yt, yp):                                 # share of the errors that fall on a phase adjacent in the task order
    i_t = np.array([PH.index(v) for v in yt]); i_p = np.array([PH.index(v) for v in yp])
    err = i_t != i_p
    obs = np.mean(np.abs(i_t[err] - i_p[err]) == 1)
    n_adj = np.array([(i > 0) + (i < K - 1) for i in range(K)])  # neighbors of each phase (Reach and Release have one)
    exp = np.mean(n_adj[i_t[err]] / (K - 1))                # if the errors were scattered at random among the other 7 phases
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
TAB_PHASES = pd.DataFrame(rows).T
TAB_PHASES.loc["caso: 1/K"] = ["", f"{100 / K:.1f}", "", "", ""]
TAB_PHASES.loc["caso: classe più frequente"] = ["", f"{100 / K:.1f}", f"{counts.max() / len(y) * 100:.1f} ({counts.idxmax()})", "", ""]
display(TAB_PHASES)
print(f"Evaluation points: {len(y)}; per phase: {counts.reindex(PH).to_dict()}")
print("Note: the balanced accuracy of a classifier that always predicts the most frequent class is 1/K = 12.5%.")

# --- 2. confusion matrices (rows: true phase, normalized; summed over seeds)
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
fig.supxlabel("predicted phase"); fig.supylabel("true phase (rows in %)")
fig.tight_layout(); fig.savefig(FIGS / "phases_confusione.png", bbox_inches="tight"); plt.show()

import pickle
pickle.dump(dict(TAB_PHASES=TAB_PHASES, CM=CM, PRED=PRED, y=y, PH=PH), open(FIGS.parent / "phases_04.pkl", "wb"))
'''

HEAD = '''# PHASE CHECK: output folders (the original 04 stays untouched)
FIGS = Path.home() / "thesis_h1_work/rep/phases/figures"; FIGS.mkdir(parents=True, exist_ok=True)'''

n = build(SRC, OUT, skip=SKIP,
          replace={8: [('    ACC[f"Isomap, d = {d}"] = decode_map(INP_PC, emb_isomap(d))\n', ''),
                       ('    ACC[f"UMAP, d = {d}"] = Parallel(n_jobs=5)(delayed(decode_map)(INP_PC, emb_umap(d, sd)) for sd in SEEDS)  # 5 semi in parallelo\n', '')]},
          insert_before={3: [HEAD], 38: [GUARD], 52: ["N_ITER = 1000                                               # the number of steps of §3.2"],
                         74: [CEBRA_B]},
          append=[ANALYSIS],
          title="# Check on the phase decoding of 04 (reference session)\n\nMaps reread from the cache of the original 04 "
                "(io.py bins), no training. Generated by rep/make_phases.py.")
print(n, "cells ->", OUT)
