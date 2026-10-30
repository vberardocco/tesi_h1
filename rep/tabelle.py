"""Tabelle delle repliche: una per sessione e una di confronto tra le tre sessioni e i notebook originali.

I numeri del 02 e del 04 si leggono dalle uscite dei notebook eseguiti (originali e repliche) con la stessa
estrazione, così ogni colonna è presa nello stesso modo; i decoder da decoder.json, gli angoli da angles.pkl,
il controllo sulle fasi da fasi/fasi_04.pkl. Nessun ricalcolo.

Colonne: "originale" = notebook 02_new e 04_deep (bin di io.py, sessione di riferimento); "rif. S1", "S2", "S5" =
repliche con i bin di FALCON. Le figure sono quelle salvate dai notebook (rep/<sessione>/figures, figures/ per
gli originali).

Uso: python tabelle.py  ->  rep/tabelle/confronto.md, rep/tabelle/<sessione>.md, rep/tabelle/confronto.csv
"""
import json, pickle, re
from html.parser import HTMLParser
from pathlib import Path
import numpy as np
import pandas as pd

W = Path.home() / "tesi_h1_work"
REP = W / "rep"
OUT = REP / "tabelle"; OUT.mkdir(exist_ok=True)
SIDS = {"19250108T110520": "rif. S1", "19250113T120811": "S2", "19250120T115044": "S5"}
ORIG = "originale (io.py)"
COLS = [ORIG, *SIDS.values()]
NOTEBOOKS = {
    "02": {ORIG: W / "notebooks/02_new.final.ipynb", **{n: REP / s / "02_rep.out.ipynb" for s, n in SIDS.items()}},
    "04": {ORIG: W / "notebooks/04_deep.ipynb", **{n: REP / s / "04_rep.out.ipynb" for s, n in SIDS.items()}},
}
DOF = ["tx", "ty", "tz", "rx", "g1", "g2", "g3"]


# ---------- lettura delle uscite dei notebook

class _Table(HTMLParser):                                     # una tabella HTML di pandas: intestazioni e righe come testo
    def __init__(self):
        super().__init__(); self.rows, self.row, self.cell, self.inside = [], None, None, False
    def handle_starttag(self, tag, attrs):
        if tag == "tr": self.row = []
        elif tag in ("th", "td"): self.cell = []
    def handle_endtag(self, tag):
        if tag in ("th", "td") and self.cell is not None:
            self.row.append(" ".join("".join(self.cell).split())); self.cell = None
        elif tag == "tr" and self.row is not None:
            self.rows.append(self.row); self.row = None
    def handle_data(self, data):
        if self.cell is not None: self.cell.append(data)


def parse_table(html):
    p = _Table(); p.feed(html)
    header = p.rows[0][1:]                                    # la prima riga: le colonne (la prima cella è l'indice)
    body = [r for r in p.rows[1:] if len(r) == len(header) + 1]  # salta la riga col nome dell'indice
    return dict(cols=header, rows=[(r[0], r[1:]) for r in body])


def outputs(path):
    """Per ogni cella di codice (chiave: la sua prima riga), le tabelle mostrate e il testo stampato."""
    out = {}
    for c in json.load(open(path))["cells"]:
        if c["cell_type"] != "code":
            continue
        key = "".join(c["source"]).split("\n")[0].strip()
        o = out.setdefault(key, dict(tabs=[], text=""))
        for x in c.get("outputs", []):
            d = x.get("data", {})
            if "text/html" in d and "dataframe" in "".join(d["text/html"]):
                o["tabs"].append(parse_table("".join(d["text/html"])))
            elif x.get("output_type") == "stream" and x.get("name") == "stdout":
                o["text"] += "".join(x["text"])
    return out


def find(o, key):
    return next((v for k, v in o.items() if k.startswith(key)), None)


def pick(labels, want):                                       # etichetta esatta, altrimenti la prima che inizia così
    if want in labels:
        return labels.index(want)
    return next((i for i, l in enumerate(labels) if l.startswith(want)), None)


def tab(key, row, col, i=0):
    def get(o):
        c = find(o, key)
        if c is None or len(c["tabs"]) <= i:
            return None
        t = c["tabs"][i]
        r, k = pick([l for l, _ in t["rows"]], row), pick(t["cols"], col)
        return None if r is None or k is None or t["rows"][r][1][k] == "" else t["rows"][r][1][k]
    return get


def rx(key, pattern, fmt=None):
    def get(o):
        c = find(o, key); m = c and re.search(pattern, c["text"])
        return (fmt(m) if fmt else m.group(1)) if m else None
    return get


def lead(v):                                                  # il numero iniziale di "40.3 (38.7-41.6)"
    m = v and re.match(r"-?[\d.]+", v)
    return float(m.group()) if m else None


# ---------- le righe delle tabelle: (sezione, misura, notebook, estrazione)

ROWS_02 = [
    ("Dati", "canali attivi (≥ 0,5 Hz)", rx("duration_s = n_bins", r"Canali attivi: (\d+) su")),
    ("Dati", "frequenza media (Hz)", rx("vals, freq = np.unique", r"= ([\d.]+) Hz")),
    ("PCA (σ = 50 ms, tutti i bin)", "participation ratio", rx("PR = lam.sum()", r"Participation ratio: ([\d.]+)")),
    ("PCA (σ = 50 ms, tutti i bin)", "varianza della PC1 (%)", rx("PR = lam.sum()", r"PC1: ([\d.]+)%")),
    ("PCA (σ = 50 ms, tutti i bin)", "varianza delle PC1-3 (%)", rx("PR = lam.sum()", r"PC1-3: ([\d.]+)%")),
    ("PCA (σ = 50 ms, tutti i bin)", "componenti per l'80% / 90% / 95%",
     lambda o: (lambda a, b, c: f"{a} / {b} / {c}" if a else None)(
         rx("PR = lam.sum()", r"per l'80%: (\d+)")(o), rx("PR = lam.sum()", r"per il 90%: (\d+)")(o),
         tab("eig_sh = np.linalg.eigvalsh", "PCA su tutti i bin", "dimensioni per il 95%")(o))),
    ("PCA sulle medie per condizione", "participation ratio", tab("eig_sh = np.linalg.eigvalsh", "PCA sulle medie per condizione", "participation ratio")),
    ("PCA sulle medie per condizione", "componenti per il 95%", tab("eig_sh = np.linalg.eigvalsh", "PCA sulle medie per condizione", "dimensioni per il 95%")),
    ("PCA sulle medie per condizione", "PR con 1 / 7 ripetizioni mediate (finestre allineate)",
     lambda o: (lambda a, b: f"{a} / {b}" if a else None)(tab("rng = np.random.default_rng(0)", "finestre allineate", "1")(o),
                                                         tab("rng = np.random.default_rng(0)", "finestre allineate", "7")(o))),
    ("FA (conteggi, K* = 15)", "PR della parte condivisa", tab("eig_sh = np.linalg.eigvalsh", "FA su tutti i bin", "participation ratio")),
    ("FA (conteggi, K* = 15)", "dimensioni condivise per il 95%", tab("eig_sh = np.linalg.eigvalsh", "FA su tutti i bin", "dimensioni per il 95%")),
    ("FA (conteggi, K* = 15)", "prime 3 direzioni (% della varianza condivisa)", rx("eig_sh = np.linalg.eigvalsh", r"prime 3 direzioni: (\d+)%")),
    ("FA (conteggi, K* = 15)", "quota condivisa per canale, mediana", rx("fa = FactorAnalysis(n_components=K_FA", r"mediana ([\d.]+)")),
    ("FA nelle finestre (K* = 10)", "PR della parte condivisa", tab("in_win = np.zeros", "FA sui bin singoli", "participation ratio")),
    ("GPFA (d = 25)", "PR della parte condivisa", tab("def shared_stats", "25", "GPFA: participation ratio")),
    ("GPFA (d = 25)", "dimensioni condivise per il 95%", tab("def shared_stats", "25", "GPFA: dimensioni per il 95%")),
    ("GPFA (d = 25)", "quota condivisa per canale, mediana (%)", tab("def shared_stats", "25", "GPFA: quota condivisa mediana")),
    ("GPFA (d = 25)", "scala temporale più breve / più lunga (ms)",
     lambda o: (lambda a, b: f"{a} / {b}" if a else None)(tab("params_gp, _ = fit_gpfa", "partenza 100 ms", "1")(o),
                                                         tab("params_gp, _ = fit_gpfa", "partenza 100 ms", "25")(o))),
    ("Traiettorie nelle finestre", "consistenza: PCA smussata", tab("seq_lat, _ = gpfa_core", "PCA, smussati", "consistenza")),
    ("Traiettorie nelle finestre", "consistenza: FA (K = 25)", tab("seq_lat, _ = gpfa_core", "FA, radice", "consistenza")),
    ("Traiettorie nelle finestre", "consistenza: GPFA (d = 25)", tab("seq_lat, _ = gpfa_core", "GPFA, radice", "consistenza")),
    ("Traiettorie nelle finestre", "seghettatura: GPFA (d = 25)", tab("seq_lat, _ = gpfa_core", "GPFA, radice", "seghettatura")),
]

F = "fase riconosciuta (%)"
ROWS_04 = [
    ("Dati", "punti di valutazione", rx("s = io.load_session(SESS)", r"Punti di valutazione: (\d+)")),
    ("Fase riconosciuta (%), caso 12,5", "PCA, d = 3", tab("HALF = 5", "PCA, d = 3", F)),
    ("Fase riconosciuta (%), caso 12,5", "CEBRA-Time, 3 dimensioni", tab("HALF = 5", "CEBRA-Time, 3 dimensioni", F)),
    ("Fase riconosciuta (%), caso 12,5", "CEBRA-Time, 4 dimensioni", tab("HALF = 5", "CEBRA-Time, 4 dimensioni", F)),
    ("Fase riconosciuta (%), caso 12,5", "tetto: 25 PC smussati", tab("HALF = 5", "tetto, 25 dimensioni", F)),
    ("Fase riconosciuta (%), caso 12,5", "tetto alla pari: media sulla finestra", tab("HALF = 5", "tetto alla pari: media", F)),
    ("Fase riconosciuta (%), caso 12,5", "orologio (solo il tempo dall'inizio del trial)", rx("clock = np.zeros", r"riconosce la fase al ([\d.]+)%")),
    ("Fase riconosciuta (%), caso 12,5", "CEBRA-Time 3D sui gemelli", tab("def phase_randomize", "CEBRA-Time, 3 dimensioni, gemelli", F)),
    ("CEBRA-Time", "consistenza tra allenamenti, 3D (R², %)", tab("from sklearn.linear_model import LinearRegression", "CEBRA-Time, 3 dimensioni (stimato", "consistenza")),
    ("CEBRA-Time", "consistenza tra allenamenti, 4D (R², %)", tab("from sklearn.linear_model import LinearRegression", "CEBRA-Time, 4 dimensioni (stimato", "consistenza")),
    ("CEBRA-Time", "vicini nella catena del tempo, Δ = 10 (%)", tab("from scipy.spatial.distance import pdist", "CEBRA-Time, Δ = 10", "catena del tempo")),
    ("CEBRA-Time", "vicini della stessa fase in altri trial, Δ = 10 (%)", tab("from scipy.spatial.distance import pdist", "CEBRA-Time, Δ = 10", "stessa fase")),
    ("CEBRA-Time", "seghettatura, 4D", tab("windows = io.phase_windows", "CEBRA-Time, 4 dimensioni", "seghettatura")),
    ("CEBRA-Time", "consistenza delle traiettorie, 4D", tab("windows = io.phase_windows", "CEBRA-Time, 4 dimensioni", "consistenza, tutte")),
    ("§6, solo rif.: fase (%)", "CEBRA-Behavior, 3 dimensioni", tab("CEBRA_B = dict(CEBRA_STD", "CEBRA-Behavior, 3 dimensioni", F)),
    ("§6, solo rif.: fase (%)", "CEBRA-Behavior, 4 dimensioni", tab("CEBRA_B = dict(CEBRA_STD", "CEBRA-Behavior, 4 dimensioni", F)),
    ("§6, solo rif.: fase (%)", "PLS, 2 componenti", tab("CEBRA_B = dict(CEBRA_STD", "PLS, 2 componenti", F)),
    ("§6, solo rif.: fase (%)", "PLS, 3 componenti", tab("CEBRA_B = dict(CEBRA_STD", "PLS, 3 componenti", F)),
    ("§6, solo rif.: fase (%)", "distacco CEBRA-Time − CEBRA-Behavior, 3D / 4D (punti)",
     lambda o: (lambda t3, b3, t4, b4: f"{t3 - b3:.1f} / {t4 - b4:.1f}" if None not in (t3, b3, t4, b4) else None)(
         *(lead(tab("CEBRA_B = dict(CEBRA_STD", r, F)(o)) for r in
           ("CEBRA-Time, 3 dimensioni", "CEBRA-Behavior, 3 dimensioni", "CEBRA-Time, 4 dimensioni", "CEBRA-Behavior, 4 dimensioni")))),
    ("§6, solo rif.: fase (%)", "CEBRA-Behavior 3D, etichette mescolate nel tempo", tab("def shuffled_labels", "mescolate nel tempo", "fase riconosciuta (%): CEBRA")),
    ("§6, solo rif.: fase (%)", "CEBRA-Behavior 3D, etichette scambiate tra trial", tab("def shuffled_labels", "scambiate tra trial", "fase riconosciuta (%): CEBRA")),
    *[("§6, solo rif.: kin_r2, velocità (R², %)", m, tab("CEBRA_B = dict(CEBRA_STD", m, "velocità ricavata"))
      for m in ("CEBRA-Time, 4 dimensioni", "CEBRA-Behavior, 4 dimensioni", "PLS, 3 componenti", "tetto, 25 dimensioni")],
    *[("§6, solo rif.: kin_r2, posizione (R², %)", m, tab("CEBRA_B = dict(CEBRA_STD", m, "posizione ricavata"))
      for m in ("CEBRA-Time, 4 dimensioni", "CEBRA-Behavior, 4 dimensioni", "PLS, 3 componenti", "tetto, 25 dimensioni")],
    *[("§12: fase, tetto con tutti i 25 fattori (%)", m, tab("GETTERS12", r, "tetto"))
      for m, r in (("FA", "FA"), ("GPFA come nel §7b", "GPFA come nel §7b"), ("GPFA alla pari", "GPFA alla pari"),
                   ("LFADS (5 semi)", "LFADS"), ("25 PC smussati (parte A)", "parte A"))],
    *[("§12: fase, primi 3 fattori (%)", m, tab("GETTERS12", r, "primi 3 fattori"))
      for m, r in (("FA", "FA"), ("GPFA alla pari", "GPFA alla pari"), ("LFADS (5 semi)", "LFADS"), ("25 PC smussati (parte A)", "parte A"))],
    *[("§12: fase senza SnapTo, tetto (%), caso 14,3", m, tab("keep_ns = phase[bins]", r, "tetto"))
      for m, r in (("GPFA alla pari", "GPFA alla pari"), ("LFADS (5 semi)", "LFADS"))],
    *[("§12: bit/spike sui 40 canali nascosti, trial nuovi", m, tab("GP_PIECES", r, "tutta la sessione"))
      for m, r in (("smoothing + GLM (riferimento)", "smoothing + GLM"), ("FA", "FA"), ("GPFA alla pari", "GPFA alla pari"),
                   ("LFADS (5 semi)", "LFADS"), ("GPFA + regressione di Poisson", "variante: GPFA + regressione di Poisson"))],
    ("§12: bit/spike sui 40 canali nascosti, trial nuovi", "LFADS sopra lo smoothing + GLM (blocco × seme)",
     rx("GP_PIECES", r"sopra lo smoothing \+ GLM: (\d+) casi su 25", lambda m: f"{m.group(1)} su 25")),
    *[("§13: bit/spike, trial nuovi / trial visti", m, tab("FULL13", r, "nuovi / visti"))
      for m, r in (("smoothing + GLM", "smoothing + GLM"), ("GPFA alla pari", "GPFA alla pari"), ("LFADS (5 semi)", "LFADS"))],
    *[("§13: bit/spike con canali nascosti spostati nel tempo (trial nuovi)", m, tab("rng_sh = np.random", r, "spostati, trial nuovi"))
      for m, r in (("smoothing + GLM", "smoothing + GLM"), ("GPFA alla pari", "GPFA alla pari"), ("LFADS (3 semi)", "LFADS"))],
    ("§13: gemelli", "LFADS sul gemello, fase con tutti i fattori (%)", tab("SEGS_TW = ", "LFADS sul gemello", "tetto")),
    ("§12: stabilità", "LFADS tra semi, 25 fattori (R², %)", tab("acc_lf = np.array", "LFADS, 25 fattori, stimato su tutti", "R²")),
    ("§12: stabilità", "consistenza delle traiettorie: GPFA alla pari", tab("def traj_of", "GPFA alla pari", "consistenza, tutte")),
    ("§12: stabilità", "consistenza delle traiettorie: LFADS", tab("def traj_of", "LFADS", "consistenza, tutte")),
]


def extract(which, rows):
    outs = {name: outputs(p) for name, p in NOTEBOOKS[which].items()}
    return pd.DataFrame([dict(notebook=which, sezione=s, misura=m, **{c: g(outs[c]) for c in COLS}) for s, m, g in rows])


# ---------- decoder, angoli, controllo fasi

def decoder_table():
    rows = []
    for sid, name in SIDS.items():
        d = json.load(open(REP / sid / "decoder.json"))
        for cond, lab in (("zero_shot", "zero-shot (senza il suo giorno)"), ("held_in_80", "held-in 80% (giorno incluso)"),
                          ("within_session", "solo la sessione, 80%")):
            r = d[cond]
            rows.append({"sessione": f"{name} ({d['key']})", "condizione": lab, "R² medio": f"{r['r2']:.3f}",
                         **{k: f"{r['r2_dof'][k]:.3f}" for k in DOF}, "α": f"10^{np.log10(r['alpha']):.1f}"})
    return pd.DataFrame(rows)


def angles_tables():
    d = pickle.load(open(REP / "angles/angles.pkl", "rb"))
    S = list(SIDS); N = SIDS
    rows = []
    for (meth, K), r in d["RES"].items():
        row = {"metodo": f"{meth}, K = {K}"}
        for a, b in r["pairs"]:
            row[f"{N[a]} – {N[b]}"] = f"{r['pairs'][(a, b)].mean():.1f}"
        row["metà – metà (media)"] = f"{np.mean([v.mean() for v in r['noise'].values()]):.1f}"
        for sid in S:
            row[f"metà – metà {N[sid]}"] = f"{r['noise'][sid].mean():.1f}"
        rnd = r["random"][(S[0], S[1])]
        row["caso dalla covarianza (mediana)"] = f"{rnd['p50'].mean():.1f}"
        rows.append(row)
    first = [{"metodo": f"{meth}, K = {K}", **{f"{N[a]} – {N[b]}": " / ".join(f"{x:.0f}" for x in r["pairs"][(a, b)][:3])
                                                for a, b in r["pairs"]}} for (meth, K), r in d["RES"].items()]
    return pd.DataFrame(rows), pd.DataFrame(first), len(d["COMMON"])


def uniform_baseline(K, N, draws=300, seed=0):              # sottospazi uniformi, solo per dire quanto il caso "dalla covarianza" se ne discosta
    from scipy.linalg import subspace_angles, qr
    rng = np.random.default_rng(seed)
    return np.median([np.degrees(subspace_angles(qr(rng.normal(size=(N, K)), mode="economic")[0],
                                                 qr(rng.normal(size=(N, K)), mode="economic")[0])).mean() for _ in range(draws)])


# ---------- scrittura

def md(df, index=False):
    df = df.reset_index() if index else df
    cells = [[("—" if v is None or (isinstance(v, float) and np.isnan(v)) else str(v)) for v in r] for r in df.values]
    head = "| " + " | ".join(map(str, df.columns)) + " |\n|" + "|".join("---" for _ in df.columns) + "|\n"
    return head + "".join("| " + " | ".join(r) + " |\n" for r in cells)


def by_section(df, cols):
    text = ""
    for s, g in df.groupby("sezione", sort=False):
        text += f"\n**{s}**\n\n" + md(g[["misura", *cols]])
    return text


def figure_grid(names, cols):
    """Una riga per figura: originale, rif. S1, S2, S5 (dove esiste)."""
    src = {ORIG: lambda n: (W / "figures" / n, f"../../figures/{n}"),
           **{name: (lambda n, sid=sid: (REP / sid / "figures" / n, f"../{sid}/figures/{n}")) for sid, name in SIDS.items()}}
    out = "| figura | " + " | ".join(cols) + " |\n|---|" + "|".join("---" for _ in cols) + "|\n"
    for n in names:
        cells = []
        for c in cols:
            path, rel = src[c](n)
            cells.append(f'<img src="{rel}" width="260">' if path.exists() else "—")
        out += f"| `{n}` | " + " | ".join(cells) + " |\n"
    return out


def main():
    t02, t04 = extract("02", ROWS_02), extract("04", ROWS_04)
    dec = decoder_table()
    ang, ang_first, n_common = angles_tables()
    unif = {K: uniform_baseline(K, n_common) for K in (10, 15, 25)}
    ang["uniforme (mediana)"] = [f"{unif[int(m.split('= ')[1])]:.1f}" for m in ang["metodo"]]
    fasi = pickle.load(open(REP / "fasi/fasi_04.pkl", "rb"))["TAB_FASI"].rename_axis("metodo")

    allrows = pd.concat([t02, t04], ignore_index=True)
    allrows.to_csv(OUT / "confronto.csv", index=False)
    dec.to_csv(OUT / "decoder.csv", index=False)
    ang.to_csv(OUT / "angoli.csv", index=False)

    figs = sorted({p.name for sid in SIDS for p in (REP / sid / "figures").glob("*.png")})
    notes = """Note per leggere le tabelle:
- **originale (io.py)**: i notebook 02_new e 04_deep sulla sessione di riferimento, con i bin di io.py (una riga di
  sfasamento rispetto a FALCON). **rif. S1, S2, S5**: le repliche con i bin di FALCON, stessi iperparametri, senza sweep.
- Nel ricalcolo della sessione di riferimento cambiano anche i blocchi della cross-validation del 04: `GroupKFold`
  divide i trial in base al numero di punti di valutazione, che con la riga spostata passa da 1106 a 1099. La
  differenza tra "originale" e "rif. S1" quindi non è solo l'effetto dei bin.
- Il §6 (CEBRA-Behavior, PLS, kin_r2) è stato ricalcolato solo sulla sessione di riferimento: "—" nelle altre.
- Intervalli tra parentesi: minimo e massimo sui semi. Isomap e UMAP restano fuori (esplorativi).
"""
    # --- confronto
    with open(OUT / "confronto.md", "w", encoding="utf-8") as f:
        f.write("# Repliche su 19250113T120811 (S2) e 19250120T115044 (S5): confronto\n\n"
                "Generato da `rep/tabelle.py` dalle uscite dei notebook eseguiti, da `decoder.json` e da `angles.pkl`.\n\n" + notes)
        f.write("\n## 1. Decoder (ridge causale, 1000 ms di storia, R² pesato per varianza sul 20% finale, eval_mask)\n\n"
                "Formato di minival: R² medio e per grado di libertà. Zero-shot: allenata sulle held-in degli altri giorni, "
                "α = 10^4,5. Held-in 80%: primo 80% di tutti i file held-in, giorno incluso, α = 10^4,5. Solo la sessione: "
                "primo 80% della sola sessione, α scelto con cross-validation interna (5 blocchi di trial).\n\n" + md(dec))
        f.write("\n## 2. Dimensionalità e sottospazi (02)\n" + by_section(t02, COLS))
        f.write("\n## 3. CEBRA, LFADS, GPFA (04)\n" + by_section(t04, COLS))
        f.write(f"\n## 4. Angoli principali tra sessioni (angolo medio, gradi; {n_common} canali attivi in tutte e tre)\n\n"
                "Metà – metà: le due metà della stessa sessione (trial 1-2, 5-6, … contro 3-4, 7-8, …), il pavimento del "
                "rumore. Caso dalla covarianza: sottospazi presi dalla covarianza dei dati (Elsayed et al. 2016), 500 "
                "estrazioni. Uniforme: sottospazi casuali uniformi negli stessi canali, per confronto.\n\n" + md(ang) +
                "\nI primi tre angoli principali (gradi):\n\n" + md(ang_first) +
                "\n**Attenzione:** il caso dalla covarianza è quasi uguale a quello uniforme, soprattutto per FA: la "
                "covarianza dei conteggi z-scorati a 20 ms è quasi piatta (domina il rumore di Poisson). Il confronto che "
                "conta qui è quello con le due metà della stessa sessione.\n")
        f.write("\n## 5. Controllo sul riconoscimento delle fasi (04, sessione di riferimento, bin di io.py)\n\n"
                "Mappe rilette dalla cache del 04, nessun allenamento (`rep/fasi`).\n\n" + md(fasi, index=True))
        f.write("\n## 6. Figure\n\nOriginale, poi le tre repliche.\n\n" + figure_grid(figs, COLS))
        f.write("\nAngoli principali:\n\n" + "".join(f'<img src="../angles/figures/{p.name}" width="420"> '
                                                       for p in sorted((REP / "angles/figures").glob("*.png"))) + "\n")

    # --- una tabella per sessione
    for sid, name in SIDS.items():
        cols = [name] + ([ORIG] if sid == list(SIDS)[0] else [])
        d = dec[dec["sessione"].str.startswith(name)].drop(columns="sessione")
        a = ang[["metodo", *[c for c in ang.columns if name in c and "metà – metà (" not in c], "caso dalla covarianza (mediana)"]]
        with open(OUT / f"{sid}.md", "w", encoding="utf-8") as f:
            f.write(f"# Sessione {sid} ({name})\n\nGenerato da `rep/tabelle.py`. Bin di FALCON"
                    + (", con accanto i numeri dei notebook originali (bin di io.py)." if ORIG in cols else ".") + "\n\n")
            f.write("## Decoder\n\n" + md(d))
            f.write("\n## Dimensionalità e sottospazi (02)\n" + by_section(t02, cols))
            t = t04 if sid == list(SIDS)[0] else t04[~t04["sezione"].str.startswith("§6")]
            f.write("\n## CEBRA, LFADS, GPFA (04)\n" + by_section(t, cols))
            f.write("\n## Angoli principali (angolo medio, gradi)\n\n" + md(a))
            f.write("\n## Figure\n\n" + "".join(f'<img src="../{sid}/figures/{p.name}" width="420"> '
                                                for p in sorted((REP / sid / "figures").glob("*.png"))) + "\n")

    print(f"righe 02: {len(t02)}, righe 04: {len(t04)}; figure per riga: {len(figs)}")
    empty = [(r["notebook"], r["misura"], c) for t in (t02, t04) for _, r in t.iterrows() for c in COLS
             if r[c] is None and not (r["sezione"].startswith("§6") and c in ("S2", "S5"))]
    print("valori non trovati:", empty if empty else "nessuno")


if __name__ == "__main__":
    main()
