"""Replication tables: one per session and one comparing the three sessions with the original notebooks.

The numbers of 02 and 04 are read from the outputs of the executed notebooks (originals and replications) with the
same extraction, so that every column is taken in the same way; the decoders from decoder.json, the angles from
angles.pkl, the phase check from phases/phases_04.pkl. No recomputation.

Columns: "original" = notebooks 02_new and 04_deep (io.py bins, reference session); "ref. S1", "S2", "S5" =
replications with the FALCON bins. The figures are those saved by the notebooks (rep/<session>/figures, figures/ for
the originals).

Usage: python tables.py  ->  rep/tables/comparison.md, rep/tables/<session>.md, rep/tables/comparison.csv
"""
import json, pickle, re
from html.parser import HTMLParser
from pathlib import Path
import numpy as np
import pandas as pd

W = Path.home() / "thesis_h1_work"
REP = W / "rep"
OUT = REP / "tables"; OUT.mkdir(exist_ok=True)
SIDS = {"19250108T110520": "ref. S1", "19250113T120811": "S2", "19250120T115044": "S5"}
ORIG = "original (io.py)"
COLS = [ORIG, *SIDS.values()]
NOTEBOOKS = {
    "02": {ORIG: W / "notebooks/02_new.final.ipynb", **{n: REP / s / "02_rep.out.ipynb" for s, n in SIDS.items()}},
    "04": {ORIG: W / "notebooks/04_deep.ipynb", **{n: REP / s / "04_rep.out.ipynb" for s, n in SIDS.items()}},
}
DOF = ["tx", "ty", "tz", "rx", "g1", "g2", "g3"]


# ---------- reading the notebook outputs

class _Table(HTMLParser):                                     # a pandas HTML table: headers and rows as text
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
    header = p.rows[0][1:]                                    # the first row: the columns (the first cell is the index)
    body = [r for r in p.rows[1:] if len(r) == len(header) + 1]  # skip the row with the index name
    return dict(cols=header, rows=[(r[0], r[1:]) for r in body])


def outputs(path):
    """For each code cell (key: its first line), the tables shown and the printed text."""
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


def pick(labels, want):                                       # exact label, otherwise the first one that starts this way
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


def lead(v):                                                  # the leading number of "40.3 (38.7-41.6)"
    m = v and re.match(r"-?[\d.]+", v)
    return float(m.group()) if m else None


# ---------- the table rows: (section, measure, notebook, extraction)

ROWS_02 = [
    ("Data", "canali attivi (≥ 0,5 Hz)", rx("duration_s = n_bins", r"Canali attivi: (\d+) su")),
    ("Data", "frequenza media (Hz)", rx("vals, freq = np.unique", r"= ([\d.]+) Hz")),
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
    ("Traiettorie nelle finestre", "consistenza: PCA smussata", tab("seq_lat, _ = gpfa_core", "PCA, smussati", "consistency")),
    ("Traiettorie nelle finestre", "consistency: FA (K = 25)", tab("seq_lat, _ = gpfa_core", "FA, radice", "consistency")),
    ("Traiettorie nelle finestre", "consistency: GPFA (d = 25)", tab("seq_lat, _ = gpfa_core", "GPFA, radice", "consistency")),
    ("Traiettorie nelle finestre", "seghettatura: GPFA (d = 25)", tab("seq_lat, _ = gpfa_core", "GPFA, radice", "seghettatura")),
]

F = "fase riconosciuta (%)"
ROWS_04 = [
    ("Data", "punti di valutazione", rx("s = io.load_session(SESS)", r"Punti di valutazione: (\d+)")),
    ("Fase riconosciuta (%), caso 12,5", "PCA, d = 3", tab("HALF = 5", "PCA, d = 3", F)),
    ("Fase riconosciuta (%), caso 12,5", "CEBRA-Time, 3 dimensioni", tab("HALF = 5", "CEBRA-Time, 3 dimensioni", F)),
    ("Fase riconosciuta (%), caso 12,5", "CEBRA-Time, 4 dimensioni", tab("HALF = 5", "CEBRA-Time, 4 dimensioni", F)),
    ("Fase riconosciuta (%), caso 12,5", "tetto: 25 PC smussati", tab("HALF = 5", "tetto, 25 dimensioni", F)),
    ("Fase riconosciuta (%), caso 12,5", "tetto alla pari: media sulla finestra", tab("HALF = 5", "tetto alla pari: media", F)),
    ("Fase riconosciuta (%), caso 12,5", "orologio (solo il tempo dall'inizio del trial)", rx("clock = np.zeros", r"riconosce la fase al ([\d.]+)%")),
    ("Fase riconosciuta (%), caso 12,5", "CEBRA-Time 3D sui gemelli", tab("def phase_randomize", "CEBRA-Time, 3 dimensioni, gemelli", F)),
    ("CEBRA-Time", "consistenza tra allenamenti, 3D (R², %)", tab("from sklearn.linear_model import LinearRegression", "CEBRA-Time, 3 dimensioni (stimato", "consistency")),
    ("CEBRA-Time", "consistenza tra allenamenti, 4D (R², %)", tab("from sklearn.linear_model import LinearRegression", "CEBRA-Time, 4 dimensioni (stimato", "consistency")),
    ("CEBRA-Time", "vicini nella catena del tempo, Δ = 10 (%)", tab("from scipy.spatial.distance import pdist", "CEBRA-Time, Δ = 10", "catena del tempo")),
    ("CEBRA-Time", "vicini della stessa fase in altri trial, Δ = 10 (%)", tab("from scipy.spatial.distance import pdist", "CEBRA-Time, Δ = 10", "stessa fase")),
    ("CEBRA-Time", "seghettatura, 4D", tab("windows = io.phase_windows", "CEBRA-Time, 4 dimensioni", "seghettatura")),
    ("CEBRA-Time", "consistenza delle traiettorie, 4D", tab("windows = io.phase_windows", "CEBRA-Time, 4 dimensioni", "consistenza, tutte")),
    ("§6, solo ref.: fase (%)", "CEBRA-Behavior, 3 dimensioni", tab("CEBRA_B = dict(CEBRA_STD", "CEBRA-Behavior, 3 dimensioni", F)),
    ("§6, solo ref.: fase (%)", "CEBRA-Behavior, 4 dimensioni", tab("CEBRA_B = dict(CEBRA_STD", "CEBRA-Behavior, 4 dimensioni", F)),
    ("§6, solo ref.: fase (%)", "PLS, 2 componenti", tab("CEBRA_B = dict(CEBRA_STD", "PLS, 2 componenti", F)),
    ("§6, solo ref.: fase (%)", "PLS, 3 componenti", tab("CEBRA_B = dict(CEBRA_STD", "PLS, 3 componenti", F)),
    ("§6, solo ref.: fase (%)", "distacco CEBRA-Time − CEBRA-Behavior, 3D / 4D (punti)",
     lambda o: (lambda t3, b3, t4, b4: f"{t3 - b3:.1f} / {t4 - b4:.1f}" if None not in (t3, b3, t4, b4) else None)(
         *(lead(tab("CEBRA_B = dict(CEBRA_STD", r, F)(o)) for r in
           ("CEBRA-Time, 3 dimensioni", "CEBRA-Behavior, 3 dimensioni", "CEBRA-Time, 4 dimensioni", "CEBRA-Behavior, 4 dimensioni")))),
    ("§6, solo ref.: fase (%)", "CEBRA-Behavior 3D, etichette mescolate nel tempo", tab("def shuffled_labels", "mescolate nel tempo", "fase riconosciuta (%): CEBRA")),
    ("§6, solo ref.: fase (%)", "CEBRA-Behavior 3D, etichette scambiate tra trial", tab("def shuffled_labels", "scambiate tra trial", "fase riconosciuta (%): CEBRA")),
    *[("§6, solo ref.: kin_r2, velocità (R², %)", m, tab("CEBRA_B = dict(CEBRA_STD", m, "velocità ricavata"))
      for m in ("CEBRA-Time, 4 dimensioni", "CEBRA-Behavior, 4 dimensioni", "PLS, 3 componenti", "tetto, 25 dimensioni")],
    *[("§6, solo ref.: kin_r2, posizione (R², %)", m, tab("CEBRA_B = dict(CEBRA_STD", m, "posizione ricavata"))
      for m in ("CEBRA-Time, 4 dimensioni", "CEBRA-Behavior, 4 dimensioni", "PLS, 3 componenti", "tetto, 25 dimensioni")],
    *[("§12: fase, tetto con tutti i 25 fattori (%)", m, tab("GETTERS12", r, "ceiling"))
      for m, r in (("FA", "FA"), ("GPFA come nel §7b", "GPFA come nel §7b"), ("GPFA alla pari", "GPFA alla pari"),
                   ("LFADS (5 semi)", "LFADS"), ("25 PC smussati (parte A)", "parte A"))],
    *[("§12: fase, primi 3 fattori (%)", m, tab("GETTERS12", r, "primi 3 fattori"))
      for m, r in (("FA", "FA"), ("GPFA alla pari", "GPFA alla pari"), ("LFADS (5 semi)", "LFADS"), ("25 PC smussati (parte A)", "parte A"))],
    *[("§12: fase senza SnapTo, tetto (%), caso 14,3", m, tab("keep_ns = phase[bins]", r, "ceiling"))
      for m, r in (("GPFA alla pari", "GPFA alla pari"), ("LFADS (5 semi)", "LFADS"))],
    *[("§12: bit/spike sui 40 canali nascosti, trial nuovi", m, tab("GP_PIECES", r, "tutta la sessione"))
      for m, r in (("smoothing + GLM (reference)", "smoothing + GLM"), ("FA", "FA"), ("GPFA alla pari", "GPFA alla pari"),
                   ("LFADS (5 semi)", "LFADS"), ("GPFA + regressione di Poisson", "variante: GPFA + regressione di Poisson"))],
    ("§12: bit/spike sui 40 canali nascosti, trial nuovi", "LFADS sopra lo smoothing + GLM (blocco × seme)",
     rx("GP_PIECES", r"sopra lo smoothing \+ GLM: (\d+) casi su 25", lambda m: f"{m.group(1)} su 25")),
    *[("§13: bit/spike, trial nuovi / trial visti", m, tab("FULL13", r, "nuovi / visti"))
      for m, r in (("smoothing + GLM", "smoothing + GLM"), ("GPFA alla pari", "GPFA alla pari"), ("LFADS (5 semi)", "LFADS"))],
    *[("§13: bit/spike con canali nascosti spostati nel tempo (trial nuovi)", m, tab("rng_sh = np.random", r, "spostati, trial nuovi"))
      for m, r in (("smoothing + GLM", "smoothing + GLM"), ("GPFA alla pari", "GPFA alla pari"), ("LFADS (3 semi)", "LFADS"))],
    ("§13: gemelli", "LFADS sul gemello, fase con tutti i fattori (%)", tab("SEGS_TW = ", "LFADS sul gemello", "ceiling")),
    ("§12: stabilità", "LFADS tra semi, 25 fattori (R², %)", tab("acc_lf = np.array", "LFADS, 25 fattori, stimato su tutti", "R²")),
    ("§12: stabilità", "consistenza delle traiettorie: GPFA alla pari", tab("def traj_of", "GPFA alla pari", "consistenza, tutte")),
    ("§12: stabilità", "consistenza delle traiettorie: LFADS", tab("def traj_of", "LFADS", "consistenza, tutte")),
]


def extract(which, rows):
    outs = {name: outputs(p) for name, p in NOTEBOOKS[which].items()}
    return pd.DataFrame([dict(notebook=which, section=s, measure=m, **{c: g(outs[c]) for c in COLS}) for s, m, g in rows])


# ---------- decoder, angles, phase check

def decoder_table():
    rows = []
    for sid, name in SIDS.items():
        d = json.load(open(REP / sid / "decoder.json"))
        for cond, lab in (("zero_shot", "zero-shot (without its own day)"), ("held_in_80", "held-in 80% (day included)"),
                          ("within_session", "session only, 80%")):
            r = d[cond]
            rows.append({"session": f"{name} ({d['key']})", "condition": lab, "mean R²": f"{r['r2']:.3f}",
                         **{k: f"{r['r2_dof'][k]:.3f}" for k in DOF}, "α": f"10^{np.log10(r['alpha']):.1f}"})
    return pd.DataFrame(rows)


NULLS = (("random", "noise_random", "caso, tutti i bin"), ("random_cond", "noise_random_cond", "caso, medie per condizione"),
         ("random_unif", "random_unif", "uniform"))


def angles_tables():
    """Wide table (observed mean angles and the three chance cases of the ref.-S2 pair), long table (each observed
    angle against its chance case: for pairs, the mean covariance of the two sessions; for halves, that of the session)
    and the first three angles."""
    d = pickle.load(open(REP / "angles/angles.pkl", "rb"))
    S = list(SIDS); N = SIDS
    ms = lambda s: f"{s['mean_p50']:.1f} ({s['mean_p5']:.1f})"
    wide, long = [], []
    for (meth, K), r in d["RES"].items():
        row = {"method": f"{meth}, K = {K}"}
        for a, b in r["pairs"]:
            row[f"{N[a]} – {N[b]}"] = f"{r['pairs'][(a, b)].mean():.1f}"
        row["half – half (media)"] = f"{np.mean([v.mean() for v in r['noise'].values()]):.1f}"
        for sid in S:
            row[f"half – half {N[sid]}"] = f"{r['noise'][sid].mean():.1f}"
        for kp, _, lab in NULLS:
            row[f"{lab}: mediana (5°)"] = ms(r[kp] if kp == "random_unif" else r[kp][(S[0], S[1])])
        wide.append(row)
        obs = [(f"{N[a]} – {N[b]}", r["pairs"][(a, b)].mean(), {kp: (r[kp] if kp == "random_unif" else r[kp][(a, b)]) for kp, _, _ in NULLS})
               for a, b in r["pairs"]]
        obs += [(f"half – half {N[sid]}", r["noise"][sid].mean(), {kp: (r[kn] if kn == "random_unif" else r[kn][sid]) for kp, kn, _ in NULLS})
                for sid in S]
        for name, o, nl in obs:
            row = {"method": f"{meth}, K = {K}", "comparison": name, "observed": f"{o:.1f}"}
            for kp, _, lab in NULLS:
                row[f"{lab}: mediana (5°)"] = ms(nl[kp])
                row[f"sotto il 5° ({lab})"] = "sì" if o < nl[kp]["mean_p5"] else "no"
            long.append(row)
    first = [{"method": f"{meth}, K = {K}", **{f"{N[a]} – {N[b]}": " / ".join(f"{x:.0f}" for x in r["pairs"][(a, b)][:3])
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
    for s, g in df.groupby("section", sort=False):
        text += f"\n**{s}**\n\n" + md(g[["measure", *cols]])
    return text


def figure_grid(names, cols):
    """One row per figure: original, ref. S1, S2, S5 (where it exists)."""
    src = {ORIG: lambda n: (W / "figures" / n, f"../../figures/{n}"),
           **{name: (lambda n, sid=sid: (REP / sid / "figures" / n, f"../{sid}/figures/{n}")) for sid, name in SIDS.items()}}
    out = "| figure | " + " | ".join(cols) + " |\n|---|" + "|".join("---" for _ in cols) + "|\n"
    for n in names:
        cells = []
        for c in cols:
            path, rel = src[c](n)
            cells.append(f'<img src="{rel}" width="260">' if path.exists() else "—")
        out += f"| `{n}` | " + " | ".join(cells) + " |\n"
    return out


KIN_VERDICT = """**Reading: noise in the position.** In S5 the autocorrelation of the position change drops in all degrees of freedom
(tx 0.29, tz 0.11, against 0.78-0.89 in the other two sessions) and its sd almost doubles in translation (tx 0.063
against 0.033-0.037, tz 0.130 against 0.059-0.063). The velocity, instead, is as in the other sessions: autocorrelation
0.997-0.999 everywhere, S5 not higher, and sd equal or slightly higher (tx 0.00071 against 0.00057-0.00065), not
lower. The autocorrelation of the velocity is already at ceiling in all sessions, so on its own it would have little room:
it is the sd that rules out a smoother velocity. The noise does not dominate (the autocorrelation does not approach -0.5): it
adds to a signal that remains, and with a 300 ms moving average position and velocity return to 0.98-0.99.
"""


def main():
    t02, t04 = extract("02", ROWS_02), extract("04", ROWS_04)
    dec = decoder_table()
    ang, ang_test, ang_first, n_common = angles_tables()
    phases = pickle.load(open(REP / "phases/phases_04.pkl", "rb"))["TAB_PHASES"].rename_axis("method")

    allrows = pd.concat([t02, t04], ignore_index=True)
    allrows.to_csv(OUT / "comparison.csv", index=False)
    dec.to_csv(OUT / "decoder.csv", index=False)
    ang.to_csv(OUT / "angles.csv", index=False)
    ang_test.to_csv(OUT / "angles_vs_the_chance.csv", index=False)

    figs = sorted({p.name for sid in SIDS for p in (REP / sid / "figures").glob("*.png")})
    notes = """Notes for reading the tables:
- **original (io.py)**: the notebooks 02_new and 04_deep on the reference session, with the io.py bins (one row of
  offset with respect to FALCON). **ref. S1, S2, S5**: the replications with the FALCON bins, same hyperparameters, no sweep.
- In the recomputation of the reference session the cross-validation blocks of 04 also change: `GroupKFold`
  splits the trials according to the number of evaluation points, which with the shifted row goes from 1106 to 1099. The
  difference between "original" and "ref. S1" is therefore not only the effect of the bins.
- §6 (CEBRA-Behavior, PLS, kin_r2) was recomputed only on the reference session: "—" in the others.
- Intervals in parentheses: minimum and maximum over seeds. Isomap and UMAP are left out (exploratory).
"""
    diag = pd.read_csv(REP / "angles/diagnosis.csv")
    diag = diag[diag["covarianza"] != "dati dei sottospazi"]   # identical to that of the "all bins" chance case (difference 0)
    kk = {"PCA": (10, 25), "FA": (10, 15, 25), "GPFA": (10, 25)}
    lab = {"chance (cov_space)": "all bins", "medie per condizione": "condition means"}
    opt = lambda v, f: "" if pd.isna(v) else format(v, f)
    diag_t = pd.DataFrame([{"method": r["method"], "session": r["session"], "chance covariance": lab[r["covarianza"]],
                            "channels": r["channels"], "PR": f"{r['PR']:.1f}",
                            **{f"first {k} (%)": (f"{r[f'primi {k} (%)']:.1f}" if k in kk[r["method"]] else "") for k in (10, 15, 25)},
                            "maximum difference from the subspace covariance": opt(r["differenza massima tra le due"], ".1g"),
                            "relative error, covariance of 20000 sampled vectors":
                                opt(r["campionamento: errore relativo della covarianza (20000 vettori)"], ".3f")}
                           for _, r in diag.iterrows()])
    kin = pd.read_csv(REP / "kin/kin_check.csv")
    kin_t = kin[["session", "degree", "r, tutti i campioni (come il 04)", "r, dentro i trial e eval_mask (L = 0)",
                 "ritardo migliore (campioni da 20 ms)", "r al ritardo migliore", "r, media mobile 100 ms",
                 "r, media mobile 300 ms", "pendenza dkin ~ vel", "sd(dkin) / sd(vel)"]].round(3)
    kin_ac = kin[["session", "degree", "autocorrelazione a 1 bin, velocità", "autocorrelazione a 1 bin, variazione di posizione",
                  "sd velocità (unità del file)", "sd variazione di posizione (units/s)"]].copy()
    for c in kin_ac.columns[2:]:
        kin_ac[c] = kin_ac[c].map(lambda v: f"{v:.4g}")
    n_obs = len(ang_test)
    n_below = {lab: int((ang_test[f"sotto il 5° ({lab})"] == "sì").sum()) for _, _, lab in NULLS}
    changes = f"""**Changes of 30 Oct.**
- §4, principal angles: **the subspaces are estimated on all bins, the new chance case on the condition means.**
  Added the chance case from the covariance of the means of the 16 conditions (8 phases × version), for PCA, FA and GPFA
  each with its own preprocessing, as in Elsayed et al. 2016; the "all bins" and the uniform chance cases remain. For each
  chance case: median and 5th percentile of the mean angle (500 pairs); §4c compares each observed angle, between sessions
  and between the two halves, with the 5th percentile of its chance case. Below the 5th percentile: {n_below['caso, medie per condizione']} angles out of
  {n_obs} with the condition-means chance case, {n_below['caso, tutti i bin']} of {n_obs} with the all-bins one,
  {n_below['uniform']} of {n_obs} with the uniform one. The tightest margin is PCA at K = 25 (55.5° against 58.0°
  between ref. S1 and S5). The observed angles and the "all bins" chance case are identical to before (difference 0).
  The condition-means chance case is steep only for PCA (PR 15-20), which works on smoothed firing rates; for FA and
  GPFA, which work on 20 ms counts without smoothing, averaging 7-8 repetitions leaves a lot of noise (PR 49-85)
  and the chance case stays close to uniform (§4b).
- §4b: diagnosis of the chance-case covariances. The covariance of the "all bins" chance case is the same as that of the
  subspace data (same preprocessing, difference 0) and the sampling reproduces it; the spectrum is flat (PR 87-145 over
  155 channels), which is why that chance case is almost uniform. It was not a sampling error.
- §7: kinematics, position change versus velocity within trials and in eval_mask, with cross-correlation. In
  S5 the correlation stays low (tx 0.59, tz 0.46) without lags. §7b: it is **noise in the position**, not a
  smoother velocity (1-bin autocorrelation of the position change 0.11-0.72 in S5 against 0.51-0.89; sd
  almost doubled in translation; velocity with the same autocorrelation, 0.997-0.999, and sd not lower).

"""
    # --- comparison
    with open(OUT / "comparison.md", "w", encoding="utf-8") as f:
        f.write("# Replications on 19250113T120811 (S2) and 19250120T115044 (S5): comparison\n\n"
                "Generated by `rep/tables.py` from the outputs of the executed notebooks, from `decoder.json` and from `angles.pkl`.\n\n"
                + changes + notes)
        f.write("\n## 1. Decoder (causal ridge, 1000 ms of history, variance-weighted R² on the final 20%, eval_mask)\n\n"
                "Minival format: mean R² and R² per degree of freedom. Zero-shot: trained on the held-in sessions of the other days, "
                "α = 10^4.5. Held-in 80%: first 80% of all held-in files, own day included, α = 10^4.5. Session only: "
                "first 80% of the session alone, α chosen by inner cross-validation (5 blocks of trials).\n\n" + md(dec))
        f.write("\n## 2. Dimensionality and subspaces (02)\n" + by_section(t02, COLS))
        f.write("\n## 3. CEBRA, LFADS, GPFA (04)\n" + by_section(t04, COLS))
        f.write(f"\n## 4. Principal angles between sessions (mean angle, degrees; {n_common} channels active in all three)\n\n"
                "The subspaces are estimated on all bins (PCA, FA) or on all windows (GPFA). Half – half: the two "
                "halves of the same session (trials 1-2, 5-6, … versus 3-4, 7-8, …), the noise floor. Three chance cases, "
                "all with subspaces drawn from a covariance as in Elsayed et al. 2016 (v = U S^1/2 r, then QR; 500 "
                "pairs): **all bins**, the covariance of the subspace data (flat spectrum, almost uniform, §4b); "
                "**condition means**, the covariance of the means of the 16 conditions (8 phases × even/odd version, "
                "1 s windows from the start of the phase) with the preprocessing of the method, the chance case of Elsayed et al.; "
                "**uniform**, identity covariance. For pairs of sessions the chance case uses the mean of the two covariances, "
                "for the two halves the covariance of the session. Each chance case: median and, in parentheses, 5th percentile "
                "of the mean angle.\n\n**4a. Observed mean angles and chance cases (chance cases of the pair ref. S1 – S2)**\n\n" + md(ang) +
                "\n**4c. Each observed angle against its chance case** (\"sì\": below the 5th percentile, i.e. closer than "
                "happens by chance in 95 draws out of 100)\n\n" + md(ang_test) +
                "\nThe first three principal angles (degrees):\n\n" + md(ang_first))
        f.write("\n### 4b. Diagnosis of the chance case from the covariance (`rep/angles_diag.py`)\n\n"
                "For each session, the covariance from which the chance case samples (for the chance case the mean of the two "
                "compared sessions is used). Preprocessing, the same for the chance case and for the subspaces: PCA, FALCON 20 ms bins, smoothing "
                "σ = 50 ms, z-score, all bins; FA, raw counts z-scored, all bins; GPFA, square root of the counts "
                "in the 119 windows of 1 s, without z-score. Sampling (`angles.py`, lines 122-127): eigenvalues and "
                "eigenvectors of the mean covariance, M = U S^1/2, K vectors M r with Gaussian r, orthonormalized with QR, "
                "500 pairs. The relative error of the 20000 vectors is the one expected from finite sampling.\n\n" + md(diag_t))
        f.write("\n## 5. Check on phase decoding (04, reference session, io.py bins)\n\n"
                "Maps reloaded from the 04 cache, no training (`rep/phases`).\n\n" + md(phases, index=True))
        f.write("\n## 6. Figures\n\nOriginal, then the three replications.\n\n" + figure_grid(figs, COLS))
        f.write("\n## 7. Kinematics: position change versus velocity (`rep/kin_check.py`)\n\n"
                "dkin[k] = (position[k+1] - position[k]) / 20 ms against the velocity of the file (FALCON target). "
                "All samples: as in §1 of 04. Within trials and in eval_mask: only pairs of consecutive samples of the "
                "same trial, both in eval_mask. Lag: the one that maximizes the correlation between -10 and +10 samples "
                "(negative: the velocity leads). Velocity and eval_mask are the same as in "
                "`load_nwb`; the position comes from the same NWB.\n\n" + md(kin_t) +
                '\n<img src="../kin/kin_lag.png" width="820">\n'
                "\n**7b. Noise in the position or smoother velocity?** The same slope and the higher sd ratio would be seen "
                "in both cases. On the same samples (k, k+1, k+2 in the same trial and in eval_mask): "
                "1-bin (20 ms) autocorrelation and absolute sd. White noise in the position lowers the autocorrelation "
                "of the position change (towards -0.5 if it dominates) and raises its sd; a smoother velocity raises "
                "the autocorrelation of the velocity and lowers its sd.\n\n" + md(kin_ac) + "\n" + KIN_VERDICT)
        f.write("\nPrincipal angles:\n\n" + "".join(f'<img src="../angles/figures/{p.name}" width="420"> '
                                                       for p in sorted((REP / "angles/figures").glob("*.png"))) + "\n")

    # --- one table per session
    for sid, name in SIDS.items():
        cols = [name] + ([ORIG] if sid == list(SIDS)[0] else [])
        d = dec[dec["session"].str.startswith(name)].drop(columns="session")
        a = ang_test[ang_test["comparison"].str.contains(name, regex=False)]
        with open(OUT / f"{sid}.md", "w", encoding="utf-8") as f:
            f.write(f"# Session {sid} ({name})\n\nGenerated by `rep/tables.py`. FALCON bins"
                    + (", with the numbers of the original notebooks (io.py bins) alongside." if ORIG in cols else ".") + "\n\n")
            f.write("## Decoder\n\n" + md(d))
            f.write("\n## Dimensionality and subspaces (02)\n" + by_section(t02, cols))
            t = t04 if sid == list(SIDS)[0] else t04[~t04["section"].str.startswith("§6")]
            f.write("\n## CEBRA, LFADS, GPFA (04)\n" + by_section(t, cols))
            f.write("\n## Principal angles (mean angle, degrees)\n\nSubspaces estimated on all bins; chance cases: median and "
                    "5th percentile of the mean angle over 500 pairs (see comparison.md, §4).\n\n" + md(a))
            f.write("\n## Figures\n\n" + "".join(f'<img src="../{sid}/figures/{p.name}" width="420"> '
                                                for p in sorted((REP / sid / "figures").glob("*.png"))) + "\n")

    print(f"rows 02: {len(t02)}, rows 04: {len(t04)}; figures per row: {len(figs)}")
    empty = [(r["notebook"], r["measure"], c) for t in (t02, t04) for _, r in t.iterrows() for c in COLS
             if r[c] is None and not (r["section"].startswith("§6") and c in ("S2", "S5"))]
    print("values not found:", empty if empty else "none")


# ---------- extended 04 (night of 16 Nov): original (io.py bins) versus ref. S1 (FALCON bins), §7-§8 and §13.5-§15

def _cell_rows(t):                                            # {(row, column): value} of a table read by parse_table
    return {(r, c): v for r, vals in t["rows"] for c, v in zip(t["cols"], vals)}


def _merge(ta, tb):
    """A table with the columns of the original: "a → b" where the values differ, "a" where they match."""
    a, b = _cell_rows(ta), _cell_rows(tb)
    rows = list(dict.fromkeys([r for r, _ in ta["rows"]] + [r for r, _ in tb["rows"]]))
    cols = list(dict.fromkeys(ta["cols"] + tb["cols"]))
    out = []
    for r in rows:
        line = {"": r}
        for c in cols:
            x, y = a.get((r, c)), b.get((r, c))
            line[c] = (x if x == y else f"{x if x is not None else '—'} → {y if y is not None else '—'}") if (x or y) else ""
        out.append(line)
    return pd.DataFrame(out, columns=["", *cols])


def extended(sid="19250108T110520"):
    """rep/tables/<sid>_extended.md: for each code cell of §7, §8 and §13.5-§15, the tables and the printed text
    of the original 04 (notebooks/04_deep.ipynb of the repo, bins="legacy") and of the extended 04 (rep/<sid>/04_rep_extended.out.ipynb,
    FALCON bins), side by side: in each table cell "original → ref. S1" where they differ."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from build_nb import parse
    src = Path.home() / "thesis_h1_git" / "notebooks" / ".src_04_deep.py"
    orig = outputs(Path.home() / "thesis_h1_git" / "notebooks" / "04_deep.ipynb")
    new = outputs(REP / sid / "04_rep_extended.out.ipynb")
    cells = parse(src)
    keep = set(range(84, 99)) | set(range(189, len(cells)))
    title, text = "", [f"# Extended 04: original vs ref. S1 ({sid})\n",
                       "Generated by `rep/tables.py --extended`. For each code cell of §7, §8 and §13.5-§15: the tables "
                       "of the original notebook (`notebooks/04_deep.ipynb`, io.py bins before 30 Oct, `bins=\"legacy\"`) and "
                       "of the extended 04 (`rep/" + sid + "/04_rep_extended.out.ipynb`, FALCON bins), with the same seeds and the "
                       "same hyperparameters. In the tables: \"original → ref. S1\" where the values differ, a single value "
                       "where they coincide. Below, the printed lines that differ.\n"]
    missing = []
    for i, (kind, body) in enumerate(cells):
        if kind == "md":
            h = [l for l in body.split("\n") if re.match(r"^# #{2,4} ", l)]
            if h:
                title = re.sub(r"^# #+ ", "", h[0]).strip()
                if i in keep:
                    text.append(f"\n## {title}\n")
            continue
        if i not in keep or not body.strip():
            continue
        key = body.split("\n")[0].strip()
        a, b = orig.get(key), new.get(key)
        if a is None or b is None:
            missing.append((i, key[:60], a is None, b is None)); continue
        if not a["tabs"] and not a["text"] and not b["tabs"] and not b["text"]:
            continue
        text.append(f"\n**Cell {i}** (`{key[:70]}`)\n")
        for j in range(max(len(a["tabs"]), len(b["tabs"]))):
            ta = a["tabs"][j] if j < len(a["tabs"]) else dict(cols=[], rows=[])
            tb = b["tabs"][j] if j < len(b["tabs"]) else dict(cols=[], rows=[])
            text.append("\n" + md(_merge(ta, tb)))
        la, lb = a["text"].strip().split("\n"), b["text"].strip().split("\n")
        if la != lb:
            text.append("\nPrinted text, original:\n\n```\n" + "\n".join(l for l in la if l not in lb)[:6000] +
                        "\n```\n\nref. S1:\n\n```\n" + "\n".join(l for l in lb if l not in la)[:6000] + "\n```\n")
    out = OUT / f"{sid}_extended.md"
    out.write_text("".join(t if t.endswith("\n") else t + "\n" for t in text), encoding="utf-8")
    print(f"{out}: cells without a match {missing if missing else 'none'}")
    return out


if __name__ == "__main__":
    import sys
    if "--extended" in sys.argv:
        extended()
    else:
        main()
