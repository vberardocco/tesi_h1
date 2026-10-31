"""Porta nel registro dei run (results/h1_runs.csv) i tre decoder.json delle repliche (rep/<sid>/decoder.json).

Ogni condizione di dec_rep.py diventa un run: esperimento "replica", method "counts" (ridge sui conteggi),
decoder "ridge", bins "falcon", split "test20" (20% finale del calib della sessione, bin con eval_mask),
condition "zero_shot" / "held_in_80" / "within_session", fold, k_shot, seed, latent_dim = NA.
Metriche: r2, r2 per grado di libertà, n_bin (bin valutati). L'alpha resta solo nel decoder.json.

timestamp, commit e script sono quelli del run originale, non di questo import:
- timestamp = ora di modifica del decoder.json;
- commit = l'ultimo commit che ha toccato rep/dec_rep.py, con "-dirty" se rep/dec_rep.py o
  src/tesi_h1/decoding.py (in ~/tesi_h1_work, dove è girato) non coincidono con quel commit o sono stati
  modificati dopo il decoder.json (allora il codice che l'ha prodotto non è più quello sul disco).
  tesi_h1/io.py veniva importato da __init__.py ma dec_rep.py non lo usa: non entra nel controllo.

Uso: python scripts/import_decoder_json.py [--force]   (senza --force non importa due volte)
"""
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tesi_h1 import results as res  # noqa: E402

SIDS = ["19250108T110520", "19250113T120811", "19250120T115044"]
SCRIPT = res.WORK / "rep/dec_rep.py"
CODE = ["rep/dec_rep.py", "src/tesi_h1/decoding.py"]


def git(*args):
    return subprocess.run(["git", "-C", str(res.GIT), *args], capture_output=True, text=True, check=True).stdout.strip()


commit = git("log", "-1", "--format=%h", "--", "rep/dec_rep.py")
if res.RUNS.exists() and "--force" not in sys.argv:
    df = res.load_runs(keep_last=False)
    if ((df["esperimento"] == "replica") & (df["script"] == res._short(SCRIPT))).any():
        sys.exit(f"già importati in {res.RUNS} (--force per importarli di nuovo)")

for sid in SIDS:
    path = res.WORK / "rep" / sid / "decoder.json"
    d = json.load(open(path))
    made = path.stat().st_mtime
    problems = []
    for rel in CODE:
        f = res.WORK / rel
        if res._blob(f.read_bytes()) != git("rev-parse", f"{commit}:{rel}"):
            problems.append(f"{rel} diverso da {commit}")
        if f.stat().st_mtime > made:
            problems.append(f"{rel} modificato dopo {path.name}")
    auto = dict(timestamp=datetime.fromtimestamp(made).isoformat(timespec="seconds"),
                commit=commit + ("-dirty" if problems else ""), script=res._short(SCRIPT))
    for cond in ("zero_shot", "held_in_80", "within_session"):
        r = d[cond]
        metrics = {"r2": r["r2"], **{f"r2_{k}": v for k, v in r["r2_dof"].items()}, "n_bin": d["test_bins"]}
        res.log_run(metrics, _auto=auto, esperimento="replica", method="counts", latent_dim=None, decoder="ridge",
                    session=d["key"], bins="falcon", split="test20", condition=cond, fold=None, k_shot=None,
                    H_ms=d["H_ms"], seed=None)
    print(f"{sid} = {d['key']}: 3 condizioni, commit {auto['commit']}" + (f" ({'; '.join(problems)})" if problems else ""))
print(f"-> {res.RUNS}")
