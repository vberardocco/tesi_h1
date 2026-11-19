"""Tests of the run registry (src/thesis_h1/results.py) and of scripts/make_tables.py.

Everything in a temporary folder (THESIS_RESULTS): the real registry is not touched.
Usage: python tests/test_results.py [--expect-clean]
  --expect-clean: the rows written by the child processes must have a commit without "-dirty"
                  (valid only after committing this file and results.py).
"""
import csv
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

if len(sys.argv) > 1 and sys.argv[1] == "--worker":      # child process of the parallel test
    _, _, PATH, WID, N, T0 = sys.argv
else:
    TMP = Path(tempfile.mkdtemp(prefix="thesis_results_"))
    os.environ["THESIS_RESULTS"] = str(TMP)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import numpy as np  # noqa: E402
from thesis_h1 import results as res  # noqa: E402
from thesis_h1.decoding import r2_vw  # noqa: E402

BASE = dict(experiment="trial", method="LFADS", latent_dim=8, decoder="ridge", session="S5_set_1",
            bins="falcon", split="test20", condition=None, fold=None, k_shot=20, H_ms=1000, seed=0)

if len(sys.argv) > 1 and sys.argv[1] == "--worker":
    while time.time() < float(T0):                         # start together with the other process
        time.sleep(0.001)
    for i in range(int(N)):
        res.log_run({"r2": int(WID) + i / 1000}, path=PATH, **{**BASE, "seed": int(WID) * 1000 + i})
    sys.exit(0)

import pandas as pd  # noqa: E402

assert res.RESULTS == TMP, "the module does not use the temporary folder"
FAIL = []


def check(name, ok, detail=""):
    print(f"[{'ok' if ok else 'ERROR'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def raises(fn):
    try:
        fn()
    except ValueError as e:
        return str(e)
    return None


header = ",".join(res.COLUMNS)

# 1. two processes x 200 rows in parallel, on the same file that does not exist yet
p1 = TMP / "parallel.csv"
t0 = time.time() + 5
procs = [subprocess.Popen([sys.executable, __file__, "--worker", str(p1), str(w), "200", str(t0)]) for w in (1, 2)]
codes = [p.wait() for p in procs]
lines = p1.read_text().splitlines()
df1 = pd.read_csv(p1)
check("1. processes finished without errors", codes == [0, 0], f"codes {codes}")
check("1. a single header", sum(l == header for l in lines) == 1, f"{sum(l == header for l in lines)} headers")
check("1. 400 rows", len(lines) == 401 and len(df1) == 400, f"{len(lines) - 1} data rows")
check("1. pd.read_csv reads it, all seeds present once",
      sorted(df1["seed"]) == sorted([1000 + i for i in range(200)] + [2000 + i for i in range(200)]))
wid = (df1["seed"] // 1000).to_numpy()
switches = int((np.diff(wid) != 0).sum())
check("1. the two processes really wrote together (alternating rows)", switches > 10, f"{switches} process switches")
check("1. values consistent with the seed",
      bool(np.allclose(df1["value"], df1["seed"] // 1000 + (df1["seed"] % 1000) / 1000)))
commits = sorted(set(df1["commit"]))
scripts = sorted(set(df1["script"]))
print(f"     commits in the rows of the children: {commits}; scripts: {scripts}")
if "--expect-clean" in sys.argv:
    check("1. commit without -dirty (code identical to HEAD)", all(not c.endswith("-dirty") for c in commits), str(commits))

# 2. same call with the arguments in a different order
p2 = TMP / "order.csv"
keys2 = dict(BASE, condition="alignment_fa", fold=2, seed=7)
res.log_run({"r2": 0.111}, path=p2, **keys2)
res.log_run({"r2": 0.222}, path=p2, **dict(reversed(list(keys2.items()))))
rows = list(csv.DictReader(open(p2)))
want = {k: ("" if v is None else str(v)) for k, v in keys2.items()}
check("2. different order -> values in the right columns",
      all({k: r[k] for k in res.KEYS} == want for r in rows) and [r["value"] for r in rows] == ["0.111", "0.222"],
      str(rows[-1]))

# 3. missing or extra key -> ValueError, file unchanged
before = p2.read_bytes()
cases = {
    "missing seed": lambda: res.log_run({"r2": 1}, path=p2, **{k: v for k, v in keys2.items() if k != "seed"}),
    "extra alpha": lambda: res.log_run({"r2": 1}, path=p2, alpha=1e4, **keys2),
    "wrong bins": lambda: res.log_run({"r2": 1}, path=p2, **dict(keys2, bins="io")),
    "session not FALCON": lambda: res.log_run({"r2": 1}, path=p2, **dict(keys2, session="S5")),
    "k_shot not integer": lambda: res.log_run({"r2": 1}, path=p2, **dict(keys2, k_shot=20.0)),
    "invalid fold": lambda: res.log_run({"r2": 1}, path=p2, **dict(keys2, fold="all")),
    "non-numeric metric": lambda: res.log_run({"r2": "high"}, path=p2, **keys2),
    "no metric": lambda: res.log_run({}, path=p2, **keys2),
}
for name, fn in cases.items():
    msg = raises(fn)
    check(f"3. {name} -> ValueError", msg is not None, msg or "no error")
check("3. file unchanged after the errors", p2.read_bytes() == before)
p3 = TMP / "other_header.csv"
p3.write_text("method,r2,timestamp\nLFADS,0.41,T\n")
before3 = p3.read_bytes()
msg = raises(lambda: res.log_run({"r2": 1}, path=p3, **keys2))
check("3. different header in the file -> ValueError, file unchanged", msg is not None and p3.read_bytes() == before3, msg)

# 4. same run twice with different values -> the last one is kept, n = 1
p4 = TMP / "double.csv"
res.log_run({"r2": 0.30}, path=p4, **BASE)
res.log_run({"r2": 0.35}, path=p4, **BASE)
t4 = res.summary(res.load_runs(p4), ["method", "k_shot"])
check("4. both rows in the file", len(res.load_runs(p4, keep_last=False)) == 2)
check("4. the table keeps the last one, n = 1", len(t4) == 1 and t4["n"].iloc[0] == 1 and t4["mean"].iloc[0] == 0.35,
      t4.to_string().replace("\n", " | "))

# 5. row with k_shot NA -> appears in the table
p5 = TMP / "na.csv"
res.log_run({"r2": 0.20}, path=p5, **dict(BASE, k_shot=None))
res.log_run({"r2": 0.40}, path=p5, **dict(BASE, k_shot=20))
res.log_run({"r2": 0.50}, path=p5, **dict(BASE, k_shot=20, seed=1))
t5 = res.summary(res.load_runs(p5), ["method", "k_shot"])
na_rows = t5[t5.index.get_level_values("k_shot").isna()]
check("5. the row with k_shot NA is in the table", len(t5) == 2 and len(na_rows) == 1 and na_rows["mean"].iloc[0] == 0.20,
      t5.to_string().replace("\n", " | "))
check("5. the group k_shot = 20 has n = 2", int(t5.loc[("LFADS", 20), "n"]) == 2)
check("5. select(k_shot=None) finds only the NA row", len(res.select(res.load_runs(p5), k_shot=None)) == 1)
out5 = subprocess.run([sys.executable, str(REPO / "scripts/make_tables.py"), "--runs", str(p5), "--by", "method,k_shot",
                       "--latex", str(TMP / "t5.tex")], capture_output=True, text=True)
tex = (TMP / "t5.tex").read_text() if (TMP / "t5.tex").exists() else ""
check("5. make_tables.py shows the NA row and writes the LaTeX",
      out5.returncode == 0 and "<NA>" in out5.stdout and "\\begin{tabular}" in tex, out5.stdout.strip().replace("\n", " | ") or out5.stderr[-300:])

# 6. three folds, saved predictions, R² on the concatenated predictions
p6, pd6 = TMP / "fold.csv", TMP / "preds"
rng = np.random.default_rng(0)
ys, yhs, keys6 = [], [], dict(BASE, experiment="scale", condition="ricentratura", k_shot=3)
for k in range(3):
    y = rng.normal(size=(500, 7)) * (k + 1)
    yh = y + rng.normal(size=y.shape)
    m = rng.random(500) > 0.2
    res.log_fold(y, yh, m, path=p6, preds_dir=pd6, **dict(keys6, fold=k))
    ys.append(y.astype(np.float32)[m])
    yhs.append(yh.astype(np.float32)[m])
conc = res.log_concat([0, 1, 2], path=p6, preds_dir=pd6, **{k: v for k, v in keys6.items() if k != "fold"})
direct = float(r2_vw(np.concatenate(ys), np.concatenate(yhs)))
d6 = res.load_runs(p6)
r2_folds = d6[(d6.metric == "r2") & (d6.fold != "concat")]["value"]
check("6. three .npz files in preds/", len(list(pd6.glob("*.npz"))) == 3)
check("6. R² concat = R² on the concatenated predictions", abs(conc["r2"] - direct) < 1e-12, f"{conc['r2']:.6f} vs {direct:.6f}")
check("6. row with fold = concat in the registry", ((d6.fold == "concat") & (d6.metric == "r2")).sum() == 1)
check("6. concat differs from the mean of the folds (folds of different scale)", abs(conc["r2"] - r2_folds.mean()) > 1e-3,
      f"concat {conc['r2']:.4f}, mean of the folds {r2_folds.mean():.4f}")
before6 = p6.read_bytes()
msg = raises(lambda: res.log_concat([0, 1, 2, 3], path=p6, preds_dir=pd6, **{k: v for k, v in keys6.items() if k != "fold"}))
check("6. missing fold -> ValueError, no row", msg is not None and p6.read_bytes() == before6, msg)

# 7. commit: hash of HEAD, "-dirty" if the code does not match HEAD
dec_work = res.WORK / "src/thesis_h1/decoding.py"
c, pr = res.code_version([dec_work])
check("7. file of ~/thesis_h1_work identical to HEAD -> clean", not c.endswith("-dirty"), f"{c} {pr}")
root = TMP / "fake_work"
(root / "src/thesis_h1").mkdir(parents=True)
copy = root / "src/thesis_h1/decoding.py"
shutil.copy2(dec_work, copy)
c, pr = res.code_version([copy], work=root, start=copy.stat().st_mtime + 1)
check("7. identical copy, not touched after the start -> clean", not c.endswith("-dirty"), f"{c} {pr}")
c, pr = res.code_version([copy], work=root, start=copy.stat().st_mtime - 1)
check("7. file modified after the process started -> -dirty", c.endswith("-dirty") and "after the process started" in pr[0], f"{c} {pr}")
copy.write_text(copy.read_text() + "\n# edit\n")
c, pr = res.code_version([copy], work=root, start=time.time() + 10)
check("7. content different from HEAD -> -dirty", c.endswith("-dirty") and "different" in pr[0], f"{c} {pr}")
extra = root / "src/thesis_h1/new.py"
extra.write_text("x = 1\n")
c, pr = res.code_version([extra], work=root, start=time.time() + 10)
check("7. file not in the repo -> -dirty", c.endswith("-dirty") and "is not in commit" in pr[0], f"{c} {pr}")
c, pr = res.code_version([TMP / "out.py"], start=time.time() + 10)
check("7. file outside work and git -> -dirty", c.endswith("-dirty") and "outside " in pr[0], f"{c} {pr}")
head = subprocess.run(["git", "-C", str(res.GIT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
check("7. the hash is that of HEAD in ~/thesis_h1_git", c.split("-")[0] == head, f"{c} vs {head}")
import types  # noqa: E402
fake = types.ModuleType("fake_ops")
fake.__file__ = "_ops.py"                                  # like torch._ops / torch._classes: relative name, no file
sys.modules["fake_ops"] = fake
_cwd = os.getcwd()
os.chdir(res.WORK)                                         # the real case: the runs start from ~/tesi_h1_work
check("7. module with relative __file__ (torch._ops) ignored", all(p.name != "_ops.py" for p in res.running_files()),
      str([str(p) for p in res.running_files()]))
os.chdir(_cwd)
del sys.modules["fake_ops"]

# 8. days since the last held-in session, from sessions.csv
shutil.copy2(res.WORK / "results/sessions.csv", TMP / "sessions.csv")
d8 = res.add_days(pd.DataFrame({"session": pd.array(["S2_set_1", "S5_set_1", "S6_set_1", "S12_set_2"], dtype="string")}))
check("8. days: S2 -7, S5 0, S6 6, S12 20", d8["days_from_held_in"].tolist() == [-7, 0, 6, 20], str(d8.to_dict("list")))

# 9. the numbers read back are exactly those written
p9 = TMP / "precision.csv"
vals = rng.normal(size=300).tolist() + [0.1 + 0.2, 1 / 3, 1e-300, -2.8199346252000820]
res.log_run({f"m{i}": v for i, v in enumerate(vals)}, path=p9, **BASE)
back = res.load_runs(p9)["value"].tolist()
check("9. values read back identical to those written", back == vals, f"{sum(a != b for a, b in zip(back, vals))} different")

print()
if FAIL:
    print(f"{len(FAIL)} tests failed: {FAIL}\nfiles left in {TMP}")
    sys.exit(1)
shutil.rmtree(TMP)
print("all tests passed")

