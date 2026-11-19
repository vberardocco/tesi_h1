"""Tables from the run registry (results/h1_runs.csv, see src/thesis_h1/results.py).

For each group: mean, standard deviation and number of runs (n). The last row of each run is kept
(a rerun replaces the previous one); rows with NA in the keys are kept (groupby with dropna=False).

Examples:
  python scripts/make_tables.py --by method,latent_dim,k_shot --where fold=concat experiment=scala
  python scripts/make_tables.py --by session,condition --where experiment=replica --days --latex tab.tex
  python scripts/make_tables.py --list            # what is in the registry

--where key=value (several allowed: all must hold); key=NA keeps only the rows without that key.
--days merges results/sessions.csv: grouping by day, group and days_from_held_in is then also possible.
"""
import argparse
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from thesis_h1 import results as res  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs", default=None, help=f"run registry (default {res.RUNS})")
    ap.add_argument("--sessions", default=None, help=f"sessions (default {res.SESSIONS})")
    ap.add_argument("--metric", default="r2")
    ap.add_argument("--by", default="method,latent_dim,k_shot", help="grouping keys, separated by commas")
    ap.add_argument("--where", nargs="*", default=[], metavar="KEY=VALUE")
    ap.add_argument("--days", action="store_true", help="merges sessions.csv (days since the last held-in session)")
    ap.add_argument("--decimals", type=int, default=3)
    ap.add_argument("--latex", default=None, help="also writes the LaTeX table to this file")
    ap.add_argument("--list", action="store_true", help="lists experiments, metrics and number of rows")
    a = ap.parse_args(argv)

    df = res.load_runs(a.runs)
    if a.list:
        print(df.groupby(["experiment", "metric"], dropna=False).size().rename("rows").to_string())
        return
    if a.days:
        df = res.add_days(df, a.sessions)
    where = {}
    for w in a.where:
        k, sep, v = w.partition("=")
        if not sep or k not in df.columns:
            ap.error(f"--where {w!r}: expected key=value with a column of the registry")
        where[k] = None if v == "NA" else v
    by = [b.strip() for b in a.by.split(",") if b.strip()]
    bad = [b for b in by if b not in df.columns]
    if bad:
        ap.error(f"--by: columns that do not exist {bad}")
    tab = res.summary(df, by, a.metric, **where).round(a.decimals)
    print(tab.to_string())
    if a.latex:
        Path(a.latex).write_text(tab.to_latex(float_format=f"%.{a.decimals}f", na_rep="--"))
        print(f"LaTeX: {a.latex}")
    return tab


if __name__ == "__main__":
    main()
