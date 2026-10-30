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


NULLS = (("random", "noise_random", "caso, tutti i bin"), ("random_cond", "noise_random_cond", "caso, medie per condizione"),
         ("random_unif", "random_unif", "uniforme"))


def angles_tables():
    """Tabella larga (angoli medi osservati e i tre casi della coppia rif.-S2), tabella lunga (ogni angolo osservato
    contro il suo caso: per le coppie, la covarianza media delle due sessioni; per le metà, quella della sessione)
    e i primi tre angoli."""
    d = pickle.load(open(REP / "angles/angles.pkl", "rb"))
    S = list(SIDS); N = SIDS
    ms = lambda s: f"{s['mean_p50']:.1f} ({s['mean_p5']:.1f})"
    wide, long = [], []
    for (meth, K), r in d["RES"].items():
        row = {"metodo": f"{meth}, K = {K}"}
        for a, b in r["pairs"]:
            row[f"{N[a]} – {N[b]}"] = f"{r['pairs'][(a, b)].mean():.1f}"
        row["metà – metà (media)"] = f"{np.mean([v.mean() for v in r['noise'].values()]):.1f}"
        for sid in S:
            row[f"metà – metà {N[sid]}"] = f"{r['noise'][sid].mean():.1f}"
        for kp, _, lab in NULLS:
            row[f"{lab}: mediana (5°)"] = ms(r[kp] if kp == "random_unif" else r[kp][(S[0], S[1])])
        wide.append(row)
        obs = [(f"{N[a]} – {N[b]}", r["pairs"][(a, b)].mean(), {kp: (r[kp] if kp == "random_unif" else r[kp][(a, b)]) for kp, _, _ in NULLS})
               for a, b in r["pairs"]]
        obs += [(f"metà – metà {N[sid]}", r["noise"][sid].mean(), {kp: (r[kn] if kn == "random_unif" else r[kn][sid]) for kp, kn, _ in NULLS})
                for sid in S]
        for name, o, nl in obs:
            row = {"metodo": f"{meth}, K = {K}", "confronto": name, "osservato": f"{o:.1f}"}
            for kp, _, lab in NULLS:
                row[f"{lab}: mediana (5°)"] = ms(nl[kp])
                row[f"sotto il 5° ({lab})"] = "sì" if o < nl[kp]["mean_p5"] else "no"
            long.append(row)
    first = [{"metodo": f"{meth}, K = {K}", **{f"{N[a]} – {N[b]}": " / ".join(f"{x:.0f}" for x in r["pairs"][(a, b)][:3])
                                                for a, b in r["pairs"]}} for (meth, K), r in d["RES"].items()]
    return pd.DataFrame(wide), pd.DataFrame(long), pd.DataFrame(first), len(d["COMMON"])


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


KIN_VERDICT = """**Lettura: rumore nella posizione.** In S5 l'autocorrelazione della variazione di posizione scende in tutti i gradi
(tx 0.29, tz 0.11, contro 0.78-0.89 nelle altre due sessioni) e la sua sd quasi raddoppia in traslazione (tx 0.063
contro 0.033-0.037, tz 0.130 contro 0.059-0.063). La velocità invece è come nelle altre sessioni: autocorrelazione
0.997-0.999 ovunque, S5 non più alta, e sd uguale o un po' più alta (tx 0.00071 contro 0.00057-0.00065), non più
bassa. L'autocorrelazione della velocità è già al soffitto in tutte le sessioni, quindi da sola avrebbe poco margine:
è la sd a escludere una velocità più liscia. Il rumore non domina (l'autocorrelazione non arriva verso -0.5): si
somma a un segnale che resta, e con una media mobile di 300 ms posizione e velocità tornano a 0.98-0.99.
"""


def main():
    t02, t04 = extract("02", ROWS_02), extract("04", ROWS_04)
    dec = decoder_table()
    ang, ang_test, ang_first, n_common = angles_tables()
    fasi = pickle.load(open(REP / "fasi/fasi_04.pkl", "rb"))["TAB_FASI"].rename_axis("metodo")

    allrows = pd.concat([t02, t04], ignore_index=True)
    allrows.to_csv(OUT / "confronto.csv", index=False)
    dec.to_csv(OUT / "decoder.csv", index=False)
    ang.to_csv(OUT / "angoli.csv", index=False)
    ang_test.to_csv(OUT / "angoli_contro_il_caso.csv", index=False)

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
    diag = pd.read_csv(REP / "angles/diagnosi.csv")
    diag = diag[diag["covarianza"] != "dati dei sottospazi"]   # identica a quella del caso "tutti i bin" (differenza 0)
    kk = {"PCA": (10, 25), "FA": (10, 15, 25), "GPFA": (10, 25)}
    lab = {"caso (cov_space)": "tutti i bin", "medie per condizione": "medie per condizione"}
    opt = lambda v, f: "" if pd.isna(v) else format(v, f)
    diag_t = pd.DataFrame([{"metodo": r["metodo"], "sessione": r["sessione"], "covarianza del caso": lab[r["covarianza"]],
                            "canali": r["canali"], "PR": f"{r['PR']:.1f}",
                            **{f"primi {k} (%)": (f"{r[f'primi {k} (%)']:.1f}" if k in kk[r["metodo"]] else "") for k in (10, 15, 25)},
                            "differenza massima con la covarianza dei sottospazi": opt(r["differenza massima tra le due"], ".1g"),
                            "errore relativo, covarianza di 20000 vettori campionati":
                                opt(r["campionamento: errore relativo della covarianza (20000 vettori)"], ".3f")}
                           for _, r in diag.iterrows()])
    kin = pd.read_csv(REP / "kin/kin_check.csv")
    kin_t = kin[["sessione", "grado", "r, tutti i campioni (come il 04)", "r, dentro i trial e eval_mask (L = 0)",
                 "ritardo migliore (campioni da 20 ms)", "r al ritardo migliore", "r, media mobile 100 ms",
                 "r, media mobile 300 ms", "pendenza dkin ~ vel", "sd(dkin) / sd(vel)"]].round(3)
    kin_ac = kin[["sessione", "grado", "autocorrelazione a 1 bin, velocità", "autocorrelazione a 1 bin, variazione di posizione",
                  "sd velocità (unità del file)", "sd variazione di posizione (unità/s)"]].copy()
    for c in kin_ac.columns[2:]:
        kin_ac[c] = kin_ac[c].map(lambda v: f"{v:.4g}")
    n_obs = len(ang_test)
    n_below = {lab: int((ang_test[f"sotto il 5° ({lab})"] == "sì").sum()) for _, _, lab in NULLS}
    changes = f"""**Modifiche del 30/10.**
- §4, angoli principali: **i sottospazi sono stimati su tutti i bin, il caso nuovo sulle medie per condizione.**
  Aggiunto il caso dalla covarianza delle medie delle 16 condizioni (8 fasi × versione), per PCA, FA e GPFA ognuno con
  il suo preprocessing, come in Elsayed et al. 2016; restano il caso "tutti i bin" e quello uniforme. Per ogni caso:
  mediana e 5° percentile dell'angolo medio (500 coppie); §4c confronta ogni angolo osservato, tra sessioni e tra le
  due metà, con il 5° percentile del suo caso. Sotto il 5° percentile: {n_below['caso, medie per condizione']} angoli su
  {n_obs} con il caso delle medie per condizione, {n_below['caso, tutti i bin']} su {n_obs} con quello di tutti i bin,
  {n_below['uniforme']} su {n_obs} con quello uniforme. Il margine più stretto è la PCA a K = 25 (55.5° contro 58.0°
  tra rif. S1 e S5). Gli angoli osservati e il caso "tutti i bin" sono identici a prima (differenza 0).
  Il caso delle medie per condizione è ripido solo per la PCA (PR 15-20), che lavora su firing rate smussati; per FA e
  GPFA, che lavorano su conteggi a 20 ms senza smoothing, la media di 7-8 ripetizioni lascia molto rumore (PR 49-85)
  e il caso resta vicino all'uniforme (§4b).
- §4b: diagnosi delle covarianze dei casi. La covarianza del caso "tutti i bin" è la stessa dei dati dei sottospazi
  (stesso preprocessing, differenza 0) e il campionamento la riproduce; lo spettro è piatto (PR 87-145 su 155 canali),
  per questo quel caso è quasi uniforme. Non era un errore di campionamento.
- §7: cinematica, variazione della posizione contro velocità dentro i trial e in eval_mask, con cross-correlazione. In
  S5 la correlazione resta bassa (tx 0.59, tz 0.46) senza ritardi. §7b: è **rumore nella posizione**, non una
  velocità più liscia (autocorrelazione a 1 bin della variazione di posizione 0.11-0.72 in S5 contro 0.51-0.89; sd
  quasi doppia in traslazione; velocità con la stessa autocorrelazione, 0.997-0.999, e sd non più bassa).

"""
    # --- confronto
    with open(OUT / "confronto.md", "w", encoding="utf-8") as f:
        f.write("# Repliche su 19250113T120811 (S2) e 19250120T115044 (S5): confronto\n\n"
                "Generato da `rep/tabelle.py` dalle uscite dei notebook eseguiti, da `decoder.json` e da `angles.pkl`.\n\n"
                + changes + notes)
        f.write("\n## 1. Decoder (ridge causale, 1000 ms di storia, R² pesato per varianza sul 20% finale, eval_mask)\n\n"
                "Formato di minival: R² medio e per grado di libertà. Zero-shot: allenata sulle held-in degli altri giorni, "
                "α = 10^4,5. Held-in 80%: primo 80% di tutti i file held-in, giorno incluso, α = 10^4,5. Solo la sessione: "
                "primo 80% della sola sessione, α scelto con cross-validation interna (5 blocchi di trial).\n\n" + md(dec))
        f.write("\n## 2. Dimensionalità e sottospazi (02)\n" + by_section(t02, COLS))
        f.write("\n## 3. CEBRA, LFADS, GPFA (04)\n" + by_section(t04, COLS))
        f.write(f"\n## 4. Angoli principali tra sessioni (angolo medio, gradi; {n_common} canali attivi in tutte e tre)\n\n"
                "I sottospazi sono stimati su tutti i bin (PCA, FA) o su tutte le finestre (GPFA). Metà – metà: le due "
                "metà della stessa sessione (trial 1-2, 5-6, … contro 3-4, 7-8, …), il pavimento del rumore. Tre casi, "
                "tutti con sottospazi presi da una covarianza come in Elsayed et al. 2016 (v = U S^1/2 r, poi QR; 500 "
                "coppie): **tutti i bin**, la covarianza dei dati dei sottospazi (spettro piatto, quasi uniforme, §4b); "
                "**medie per condizione**, la covarianza delle medie delle 16 condizioni (8 fasi × versione pari/dispari, "
                "finestre da 1 s dall'inizio della fase) con il preprocessing del metodo, il caso di Elsayed et al.; "
                "**uniforme**, covarianza identità. Per le coppie di sessioni il caso usa la media delle due covarianze, "
                "per le due metà la covarianza della sessione. Ogni caso: mediana e, tra parentesi, 5° percentile "
                "dell'angolo medio.\n\n**4a. Angoli medi osservati e casi (casi della coppia rif. S1 – S2)**\n\n" + md(ang) +
                "\n**4c. Ogni angolo osservato contro il suo caso** (\"sì\": sotto il 5° percentile, cioè più vicino di "
                "quanto capiti per caso in 95 estrazioni su 100)\n\n" + md(ang_test) +
                "\nI primi tre angoli principali (gradi):\n\n" + md(ang_first))
        f.write("\n### 4b. Diagnosi del caso dalla covarianza (`rep/angles_diag.py`)\n\n"
                "Per ogni sessione, la covarianza da cui il caso campiona (nel caso si usa la media delle due sessioni "
                "confrontate). Preprocessing, uguale per il caso e per i sottospazi: PCA, bin FALCON da 20 ms, smoothing "
                "σ = 50 ms, z-score, tutti i bin; FA, conteggi grezzi z-scorati, tutti i bin; GPFA, radice dei conteggi "
                "nelle 119 finestre da 1 s, senza z-score. Campionamento (`angles.py`, righe 122-127): autovalori e "
                "autovettori della covarianza media, M = U S^1/2, K vettori M r con r gaussiano, ortonormalizzati con QR, "
                "500 coppie. L'errore relativo dei 20000 vettori è quello atteso dal campionamento finito.\n\n" + md(diag_t))
        f.write("\n## 5. Controllo sul riconoscimento delle fasi (04, sessione di riferimento, bin di io.py)\n\n"
                "Mappe rilette dalla cache del 04, nessun allenamento (`rep/fasi`).\n\n" + md(fasi, index=True))
        f.write("\n## 6. Figure\n\nOriginale, poi le tre repliche.\n\n" + figure_grid(figs, COLS))
        f.write("\n## 7. Cinematica: variazione della posizione contro velocità (`rep/kin_check.py`)\n\n"
                "dkin[k] = (posizione[k+1] - posizione[k]) / 20 ms contro la velocità del file (target di FALCON). "
                "Tutti i campioni: come nel §1 del 04. Dentro i trial e in eval_mask: solo coppie di campioni "
                "consecutivi dello stesso trial, entrambi in eval_mask. Ritardo: quello che massimizza la correlazione "
                "tra -10 e +10 campioni (negativo: la velocità anticipa). Velocità ed eval_mask uguali a quelle di "
                "`load_nwb`; la posizione viene dallo stesso NWB.\n\n" + md(kin_t) +
                '\n<img src="../kin/kin_lag.png" width="820">\n'
                "\n**7b. Rumore nella posizione o velocità più liscia?** Stessa pendenza e rapporto delle sd più alto si "
                "avrebbero in entrambi i casi. Sugli stessi campioni (k, k+1, k+2 nello stesso trial e in eval_mask): "
                "autocorrelazione a 1 bin (20 ms) e sd assoluta. Rumore bianco nella posizione abbassa l'autocorrelazione "
                "della variazione di posizione (verso -0.5 se domina) e ne alza la sd; una velocità più liscia alza "
                "l'autocorrelazione della velocità e ne abbassa la sd.\n\n" + md(kin_ac) + "\n" + KIN_VERDICT)
        f.write("\nAngoli principali:\n\n" + "".join(f'<img src="../angles/figures/{p.name}" width="420"> '
                                                       for p in sorted((REP / "angles/figures").glob("*.png"))) + "\n")

    # --- una tabella per sessione
    for sid, name in SIDS.items():
        cols = [name] + ([ORIG] if sid == list(SIDS)[0] else [])
        d = dec[dec["sessione"].str.startswith(name)].drop(columns="sessione")
        a = ang_test[ang_test["confronto"].str.contains(name, regex=False)]
        with open(OUT / f"{sid}.md", "w", encoding="utf-8") as f:
            f.write(f"# Sessione {sid} ({name})\n\nGenerato da `rep/tabelle.py`. Bin di FALCON"
                    + (", con accanto i numeri dei notebook originali (bin di io.py)." if ORIG in cols else ".") + "\n\n")
            f.write("## Decoder\n\n" + md(d))
            f.write("\n## Dimensionalità e sottospazi (02)\n" + by_section(t02, cols))
            t = t04 if sid == list(SIDS)[0] else t04[~t04["sezione"].str.startswith("§6")]
            f.write("\n## CEBRA, LFADS, GPFA (04)\n" + by_section(t, cols))
            f.write("\n## Angoli principali (angolo medio, gradi)\n\nSottospazi stimati su tutti i bin; casi: mediana e "
                    "5° percentile dell'angolo medio su 500 coppie (vedi confronto.md, §4).\n\n" + md(a))
            f.write("\n## Figure\n\n" + "".join(f'<img src="../{sid}/figures/{p.name}" width="420"> '
                                                for p in sorted((REP / sid / "figures").glob("*.png"))) + "\n")

    print(f"righe 02: {len(t02)}, righe 04: {len(t04)}; figure per riga: {len(figs)}")
    empty = [(r["notebook"], r["misura"], c) for t in (t02, t04) for _, r in t.iterrows() for c in COLS
             if r[c] is None and not (r["sezione"].startswith("§6") and c in ("S2", "S5"))]
    print("valori non trovati:", empty if empty else "nessuno")


if __name__ == "__main__":
    main()
