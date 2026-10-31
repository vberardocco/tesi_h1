"""Prove del registro dei run (src/tesi_h1/results.py) e di scripts/make_tables.py.

Tutto in una cartella temporanea (TESI_RESULTS): il registro vero non si tocca.
Uso: python tests/test_results.py [--expect-clean]
  --expect-clean: le righe scritte dai processi figli devono avere un commit senza "-dirty"
                  (vale solo dopo aver committato questo file e results.py).
"""
import csv
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

if len(sys.argv) > 1 and sys.argv[1] == "--worker":      # processo figlio della prova in parallelo
    _, _, PATH, WID, N, T0 = sys.argv
else:
    TMP = Path(tempfile.mkdtemp(prefix="tesi_results_"))
    os.environ["TESI_RESULTS"] = str(TMP)

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import numpy as np  # noqa: E402
from tesi_h1 import results as res  # noqa: E402
from tesi_h1.decoding import r2_vw  # noqa: E402

BASE = dict(esperimento="prova", method="LFADS", latent_dim=8, decoder="ridge", session="S5_set_1",
            bins="falcon", split="test20", condition=None, fold=None, k_shot=20, H_ms=1000, seed=0)

if len(sys.argv) > 1 and sys.argv[1] == "--worker":
    while time.time() < float(T0):                         # partenza insieme all'altro processo
        time.sleep(0.001)
    for i in range(int(N)):
        res.log_run({"r2": int(WID) + i / 1000}, path=PATH, **{**BASE, "seed": int(WID) * 1000 + i})
    sys.exit(0)

import pandas as pd  # noqa: E402

assert res.RESULTS == TMP, "il modulo non usa la cartella temporanea"
FAIL = []


def check(name, ok, detail=""):
    print(f"[{'ok' if ok else 'ERRORE'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def raises(fn):
    try:
        fn()
    except ValueError as e:
        return str(e)
    return None


header = ",".join(res.COLUMNS)

# 1. due processi x 200 righe in parallelo, sullo stesso file che non esiste ancora
p1 = TMP / "parallelo.csv"
t0 = time.time() + 5
procs = [subprocess.Popen([sys.executable, __file__, "--worker", str(p1), str(w), "200", str(t0)]) for w in (1, 2)]
codes = [p.wait() for p in procs]
lines = p1.read_text().splitlines()
df1 = pd.read_csv(p1)
check("1. processi finiti senza errori", codes == [0, 0], f"codici {codes}")
check("1. una sola intestazione", sum(l == header for l in lines) == 1, f"{sum(l == header for l in lines)} intestazioni")
check("1. 400 righe", len(lines) == 401 and len(df1) == 400, f"{len(lines) - 1} righe di dati")
check("1. pd.read_csv legge, tutti i semi presenti una volta",
      sorted(df1["seed"]) == sorted([1000 + i for i in range(200)] + [2000 + i for i in range(200)]))
wid = (df1["seed"] // 1000).to_numpy()
switches = int((np.diff(wid) != 0).sum())
check("1. i due processi hanno scritto davvero insieme (righe alternate)", switches > 10, f"{switches} cambi di processo")
check("1. valori coerenti con il seme",
      bool(np.allclose(df1["valore"], df1["seed"] // 1000 + (df1["seed"] % 1000) / 1000)))
commits = sorted(set(df1["commit"]))
scripts = sorted(set(df1["script"]))
print(f"     commit nelle righe dei figli: {commits}; script: {scripts}")
if "--expect-clean" in sys.argv:
    check("1. commit senza -dirty (codice identico a HEAD)", all(not c.endswith("-dirty") for c in commits), str(commits))

# 2. stessa chiamata con gli argomenti in ordine diverso
p2 = TMP / "ordine.csv"
keys2 = dict(BASE, condition="allineamento_fa", fold=2, seed=7)
res.log_run({"r2": 0.111}, path=p2, **keys2)
res.log_run({"r2": 0.222}, path=p2, **dict(reversed(list(keys2.items()))))
rows = list(csv.DictReader(open(p2)))
want = {k: ("" if v is None else str(v)) for k, v in keys2.items()}
check("2. ordine diverso -> valori nelle colonne giuste",
      all({k: r[k] for k in res.KEYS} == want for r in rows) and [r["valore"] for r in rows] == ["0.111", "0.222"],
      str(rows[-1]))

# 3. chiave mancante o in più -> ValueError, file invariato
before = p2.read_bytes()
cases = {
    "manca seed": lambda: res.log_run({"r2": 1}, path=p2, **{k: v for k, v in keys2.items() if k != "seed"}),
    "in più alpha": lambda: res.log_run({"r2": 1}, path=p2, alpha=1e4, **keys2),
    "bins sbagliato": lambda: res.log_run({"r2": 1}, path=p2, **dict(keys2, bins="io")),
    "session non FALCON": lambda: res.log_run({"r2": 1}, path=p2, **dict(keys2, session="S5")),
    "k_shot non intero": lambda: res.log_run({"r2": 1}, path=p2, **dict(keys2, k_shot=20.0)),
    "fold non valido": lambda: res.log_run({"r2": 1}, path=p2, **dict(keys2, fold="tutti")),
    "metrica non numerica": lambda: res.log_run({"r2": "alto"}, path=p2, **keys2),
    "nessuna metrica": lambda: res.log_run({}, path=p2, **keys2),
}
for name, fn in cases.items():
    msg = raises(fn)
    check(f"3. {name} -> ValueError", msg is not None, msg or "nessun errore")
check("3. file invariato dopo gli errori", p2.read_bytes() == before)
p3 = TMP / "altra_intestazione.csv"
p3.write_text("method,r2,timestamp\nLFADS,0.41,T\n")
before3 = p3.read_bytes()
msg = raises(lambda: res.log_run({"r2": 1}, path=p3, **keys2))
check("3. intestazione diversa nel file -> ValueError, file invariato", msg is not None and p3.read_bytes() == before3, msg)

# 4. stesso run due volte con valori diversi -> si tiene l'ultimo, n = 1
p4 = TMP / "doppio.csv"
res.log_run({"r2": 0.30}, path=p4, **BASE)
res.log_run({"r2": 0.35}, path=p4, **BASE)
t4 = res.summary(res.load_runs(p4), ["method", "k_shot"])
check("4. tutte e due le righe nel file", len(res.load_runs(p4, keep_last=False)) == 2)
check("4. la tabella tiene l'ultimo, n = 1", len(t4) == 1 and t4["n"].iloc[0] == 1 and t4["mean"].iloc[0] == 0.35,
      t4.to_string().replace("\n", " | "))

# 5. riga con k_shot NA -> compare nella tabella
p5 = TMP / "na.csv"
res.log_run({"r2": 0.20}, path=p5, **dict(BASE, k_shot=None))
res.log_run({"r2": 0.40}, path=p5, **dict(BASE, k_shot=20))
res.log_run({"r2": 0.50}, path=p5, **dict(BASE, k_shot=20, seed=1))
t5 = res.summary(res.load_runs(p5), ["method", "k_shot"])
na_rows = t5[t5.index.get_level_values("k_shot").isna()]
check("5. la riga con k_shot NA è nella tabella", len(t5) == 2 and len(na_rows) == 1 and na_rows["mean"].iloc[0] == 0.20,
      t5.to_string().replace("\n", " | "))
check("5. il gruppo k_shot = 20 ha n = 2", int(t5.loc[("LFADS", 20), "n"]) == 2)
check("5. select(k_shot=None) trova solo la riga NA", len(res.select(res.load_runs(p5), k_shot=None)) == 1)
out5 = subprocess.run([sys.executable, str(REPO / "scripts/make_tables.py"), "--runs", str(p5), "--by", "method,k_shot",
                       "--latex", str(TMP / "t5.tex")], capture_output=True, text=True)
tex = (TMP / "t5.tex").read_text() if (TMP / "t5.tex").exists() else ""
check("5. make_tables.py mostra la riga NA e scrive il LaTeX",
      out5.returncode == 0 and "<NA>" in out5.stdout and "\\begin{tabular}" in tex, out5.stdout.strip().replace("\n", " | ") or out5.stderr[-300:])

# 6. tre fold, predizioni salvate, R² sulle predizioni concatenate
p6, pd6 = TMP / "fold.csv", TMP / "preds"
rng = np.random.default_rng(0)
ys, yhs, keys6 = [], [], dict(BASE, esperimento="scala", condition="ricentratura", k_shot=3)
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
r2_folds = d6[(d6.metrica == "r2") & (d6.fold != "concat")]["valore"]
check("6. tre file .npz in preds/", len(list(pd6.glob("*.npz"))) == 3)
check("6. R² concat = R² sulle predizioni concatenate", abs(conc["r2"] - direct) < 1e-12, f"{conc['r2']:.6f} vs {direct:.6f}")
check("6. riga con fold = concat nel registro", ((d6.fold == "concat") & (d6.metrica == "r2")).sum() == 1)
check("6. concat diverso dalla media dei fold (fold di scala diversa)", abs(conc["r2"] - r2_folds.mean()) > 1e-3,
      f"concat {conc['r2']:.4f}, media dei fold {r2_folds.mean():.4f}")
before6 = p6.read_bytes()
msg = raises(lambda: res.log_concat([0, 1, 2, 3], path=p6, preds_dir=pd6, **{k: v for k, v in keys6.items() if k != "fold"}))
check("6. fold mancante -> ValueError, niente riga", msg is not None and p6.read_bytes() == before6, msg)

# 7. commit: hash di HEAD, "-dirty" se il codice non coincide con HEAD
dec_work = res.WORK / "src/tesi_h1/decoding.py"
c, pr = res.code_version([dec_work])
check("7. file di ~/tesi_h1_work identico a HEAD -> pulito", not c.endswith("-dirty"), f"{c} {pr}")
root = TMP / "finto_work"
(root / "src/tesi_h1").mkdir(parents=True)
copy = root / "src/tesi_h1/decoding.py"
shutil.copy2(dec_work, copy)
c, pr = res.code_version([copy], work=root, start=copy.stat().st_mtime + 1)
check("7. copia identica, non toccata dopo l'avvio -> pulito", not c.endswith("-dirty"), f"{c} {pr}")
c, pr = res.code_version([copy], work=root, start=copy.stat().st_mtime - 1)
check("7. file modificato dopo l'avvio del processo -> -dirty", c.endswith("-dirty") and "dopo l'avvio" in pr[0], f"{c} {pr}")
copy.write_text(copy.read_text() + "\n# modifica\n")
c, pr = res.code_version([copy], work=root, start=time.time() + 10)
check("7. contenuto diverso da HEAD -> -dirty", c.endswith("-dirty") and "diverso" in pr[0], f"{c} {pr}")
extra = root / "src/tesi_h1/nuovo.py"
extra.write_text("x = 1\n")
c, pr = res.code_version([extra], work=root, start=time.time() + 10)
check("7. file non nel repo -> -dirty", c.endswith("-dirty") and "non è nel commit" in pr[0], f"{c} {pr}")
c, pr = res.code_version([TMP / "fuori.py"], start=time.time() + 10)
check("7. file fuori da work e git -> -dirty", c.endswith("-dirty") and "fuori da" in pr[0], f"{c} {pr}")
head = subprocess.run(["git", "-C", str(res.GIT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
check("7. l'hash è quello di HEAD in ~/tesi_h1_git", c.split("-")[0] == head, f"{c} vs {head}")

# 8. giorni dall'ultima sessione held-in, da sessions.csv
shutil.copy2(res.WORK / "results/sessions.csv", TMP / "sessions.csv")
d8 = res.add_days(pd.DataFrame({"session": pd.array(["S2_set_1", "S5_set_1", "S6_set_1", "S12_set_2"], dtype="string")}))
check("8. giorni: S2 -7, S5 0, S6 6, S12 20", d8["giorni_da_held_in"].tolist() == [-7, 0, 6, 20], str(d8.to_dict("list")))

# 9. i numeri riletti sono esattamente quelli scritti
p9 = TMP / "precisione.csv"
vals = rng.normal(size=300).tolist() + [0.1 + 0.2, 1 / 3, 1e-300, -2.8199346252000820]
res.log_run({f"m{i}": v for i, v in enumerate(vals)}, path=p9, **BASE)
back = res.load_runs(p9)["valore"].tolist()
check("9. valori riletti identici a quelli scritti", back == vals, f"{sum(a != b for a, b in zip(back, vals))} diversi")

print()
if FAIL:
    print(f"{len(FAIL)} prove fallite: {FAIL}\nfile lasciati in {TMP}")
    sys.exit(1)
shutil.rmtree(TMP)
print("tutte le prove passate")
