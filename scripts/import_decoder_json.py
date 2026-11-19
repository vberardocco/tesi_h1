"""Imports into the run registry (results/h1_runs.csv) the three decoder.json files of the replications (rep/<sid>/decoder.json).

Each condition of dec_rep.py becomes one run: experiment "replica", method "counts" (ridge on counts),
decoder "ridge", bins "falcon", split "test20" (final 20% of the session's calib data, bins with eval_mask),
condition "zero_shot" / "held_in_80" / "within_session", fold, k_shot, seed, latent_dim = NA.
Metrics: r2, r2 per degree of freedom, n_bin (evaluated bins). The alpha stays only in decoder.json.

timestamp, commit and script are those of the original run, not of this import:
- timestamp = modification time of decoder.json;
- commit = the last commit that touched rep/dec_rep.py, with "-dirty" if rep/dec_rep.py or
  src/thesis_h1/decoding.py (in ~/thesis_h1_work, where it was run) do not match that commit or were
  modified after decoder.json (then the code that produced it is no longer the code on disk).
  thesis_h1/io.py was imported by __init__.py but dec_rep.py does not use it: it is not part of the check.

Usage: python scripts/import_decoder_json.py [--force]   (without --force it does not import twice)
"""
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from thesis_h1 import results as res  # noqa: E402

SIDS = ["19250108T110520", "19250113T120811", "19250120T115044"]
SCRIPT = res.WORK / "rep/dec_rep.py"
CODE = ["rep/dec_rep.py", "src/thesis_h1/decoding.py"]


def git(*args):
    return subprocess.run(["git", "-C", str(res.GIT), *args], capture_output=True, text=True, check=True).stdout.strip()


commit = git("log", "-1", "--format=%h", "--", "rep/dec_rep.py")
if res.RUNS.exists() and "--force" not in sys.argv:
    df = res.load_runs(keep_last=False)
    if ((df["experiment"] == "replica") & (df["script"] == res._short(SCRIPT))).any():
        sys.exit(f"already imported into {res.RUNS} (--force to import them again)")

for sid in SIDS:
    path = res.WORK / "rep" / sid / "decoder.json"
    d = json.load(open(path))
    made = path.stat().st_mtime
    problems = []
    for rel in CODE:
        f = res.WORK / rel
        if res._blob(f.read_bytes()) != git("rev-parse", f"{commit}:{rel}"):
            problems.append(f"{rel} differs from {commit}")
        if f.stat().st_mtime > made:
            problems.append(f"{rel} modified after {path.name}")
    auto = dict(timestamp=datetime.fromtimestamp(made).isoformat(timespec="seconds"),
                commit=commit + ("-dirty" if problems else ""), script=res._short(SCRIPT))
    for cond in ("zero_shot", "held_in_80", "within_session"):
        r = d[cond]
        metrics = {"r2": r["r2"], **{f"r2_{k}": v for k, v in r["r2_dof"].items()}, "n_bin": d["test_bins"]}
        res.log_run(metrics, _auto=auto, experiment="replica", method="counts", latent_dim=None, decoder="ridge",
                    session=d["key"], bins="falcon", split="test20", condition=cond, fold=None, k_shot=None,
                    H_ms=d["H_ms"], seed=None)
    print(f"{sid} = {d['key']}: 3 conditions, commit {auto['commit']}" + (f" ({'; '.join(problems)})" if problems else ""))
print(f"-> {res.RUNS}")
