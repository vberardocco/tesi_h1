"""Tabelle dal registro dei run (results/h1_runs.csv, vedi src/tesi_h1/results.py).

Per ogni gruppo: media, deviazione standard e numero di run (n). Si tiene l'ultima riga di ogni run
(un run rieseguito sostituisce il precedente); le righe con NA nelle chiavi restano (groupby con dropna=False).

Esempi:
  python scripts/make_tables.py --by method,latent_dim,k_shot --where fold=concat esperimento=scala
  python scripts/make_tables.py --by session,condition --where esperimento=replica --giorni --latex tab.tex
  python scripts/make_tables.py --elenco          # che cosa c'è nel registro

--where chiave=valore (più di una: tutte vere); chiave=NA tiene solo le righe senza quella chiave.
--giorni unisce results/sessions.csv: si può raggruppare anche per giorno, gruppo, giorni_da_held_in.
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tesi_h1 import results as res  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=None, help=f"registro (default {res.RUNS})")
    ap.add_argument("--sessions", default=None, help=f"sessioni (default {res.SESSIONS})")
    ap.add_argument("--metrica", default="r2")
    ap.add_argument("--by", default="method,latent_dim,k_shot", help="chiavi del raggruppamento, separate da virgole")
    ap.add_argument("--where", nargs="*", default=[], metavar="CHIAVE=VALORE")
    ap.add_argument("--giorni", action="store_true", help="unisce sessions.csv (giorni dall'ultima held-in)")
    ap.add_argument("--decimali", type=int, default=3)
    ap.add_argument("--latex", default=None, help="scrive anche la tabella LaTeX in questo file")
    ap.add_argument("--elenco", action="store_true", help="elenca esperimenti, metriche e numero di righe")
    a = ap.parse_args(argv)

    df = res.load_runs(a.runs)
    if a.elenco:
        print(df.groupby(["esperimento", "metrica"], dropna=False).size().rename("righe").to_string())
        return
    if a.giorni:
        df = res.add_days(df, a.sessions)
    where = {}
    for w in a.where:
        k, sep, v = w.partition("=")
        if not sep or k not in df.columns:
            ap.error(f"--where {w!r}: serve chiave=valore con una colonna del registro")
        where[k] = None if v == "NA" else v
    by = [b.strip() for b in a.by.split(",") if b.strip()]
    bad = [b for b in by if b not in df.columns]
    if bad:
        ap.error(f"--by: colonne che non esistono {bad}")
    tab = res.summary(df, by, a.metrica, **where).round(a.decimali)
    print(tab.to_string())
    if a.latex:
        Path(a.latex).write_text(tab.to_latex(float_format=f"%.{a.decimali}f", na_rep="--"))
        print(f"LaTeX: {a.latex}")
    return tab


if __name__ == "__main__":
    main()
