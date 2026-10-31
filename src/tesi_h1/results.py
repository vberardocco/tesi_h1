"""Registro dei run: una riga per (run, metrica) in un CSV, da cui le tabelle della tesi si fanno con uno script.

File, nella cartella RESULTS (~/tesi_h1_work/results, oppure la variabile d'ambiente TESI_RESULTS):
- h1_runs.csv   il registro; si scrive solo con log_run / log_fold / log_concat;
- preds/*.npz   y_true, y_pred, mask di ogni fold (log_fold), per ricalcolare le metriche;
- sessions.csv  session, giorno, sid, data, gruppo; si unisce solo quando serve (add_days).

Colonne del registro, in quest'ordine:
- le chiavi del run (KEYS): vanno passate TUTTE a ogni chiamata, None = non pertinente (NA nel CSV);
  una chiave mancante o in più -> ValueError e il file non si tocca.
    esperimento  es. "replica", "scala"
    method       cosa entra nel decoder: "counts" (conteggi), "PCA", "FA", "GPFA", "LFADS", "CEBRA"
    latent_dim   dimensioni latenti (int)
    decoder      es. "ridge"
    session      sessione valutata, chiave FALCON (es. "S5_set_1")
    bins         "falcon" (load_nwb) o "legacy" (vecchi bin di io.py): numeri non confrontabili tra loro
    split        dati della valutazione: "test20" (20% finale del calib della sessione), "val", "cv",
                 "minival", "test"
    condition    es. "zero_shot", "ricentratura", "allineamento_fa", "kin_2trial", "da_zero"
    fold         0, 1, 2, ... oppure "concat" (R² sulle predizioni concatenate dei fold, log_concat)
    k_shot       trial di calibrazione (int)
    H_ms         storia del decoder in ms (int)
    seed         seme (int)
- metrica, valore: una riga per ogni metrica del run;
- timestamp, commit, script: li aggiunge log_run.

commit: hash corto di HEAD nel repo ~/tesi_h1_git, con "-dirty" se uno dei file del progetto che il processo
sta eseguendo (lo script e i moduli importati da ~/tesi_h1_work o ~/tesi_h1_git) non è identico alla sua
versione in HEAD, non è nel repo, o è stato modificato dopo l'avvio del processo. I file di ~/tesi_h1_work si
confrontano con lo stesso percorso relativo nel repo (rep/dec_rep.py, src/tesi_h1/decoding.py, ...).
Notebook e `python -c` non hanno uno script da confrontare: sempre "-dirty".

Rieseguire un run aggiunge righe nuove: load_runs tiene l'ultima per ogni (chiavi, metrica).
Per scoprire subito un errore nelle chiavi, a inizio script: check_keys(**chiavi).
"""
from __future__ import annotations

import csv
import fcntl
import hashlib
import io
import json
import numbers
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

WORK = Path.home() / "tesi_h1_work"                 # dove girano i run (fuori da iCloud)
GIT = Path.home() / "tesi_h1_git"                   # il repo git, allineato con GitHub
RESULTS = Path(os.environ.get("TESI_RESULTS", WORK / "results"))
RUNS = RESULTS / "h1_runs.csv"
PREDS = RESULTS / "preds"
SESSIONS = RESULTS / "sessions.csv"

KEYS = ["esperimento", "method", "latent_dim", "decoder", "session", "bins", "split",
        "condition", "fold", "k_shot", "H_ms", "seed"]
INT_KEYS = {"latent_dim", "k_shot", "H_ms", "seed"}
AUTO = ["timestamp", "commit", "script"]
COLUMNS = KEYS + ["metrica", "valore"] + AUTO
EXCLUDE = [WORK / "falcon-challenge"]               # clone di FALCON per Docker: non è codice nostro

_T_IMPORT = time.time()
_WARNED: set[str] = set()


# --------------------------------------------------------------------------------------------
# Controllo delle chiavi
# --------------------------------------------------------------------------------------------
def check_keys(**keys):
    """Controlla le chiavi di un run (le stesse di log_run). ValueError se qualcosa non va."""
    missing = [k for k in KEYS if k not in keys]
    extra = sorted(k for k in keys if k not in KEYS)
    if missing or extra:
        raise ValueError(f"chiavi mancanti: {missing}; chiavi in più: {extra}")
    for k in KEYS:
        v = keys[k]
        if v is None or (k == "fold" and v == "concat"):
            continue
        if k in INT_KEYS or k == "fold":
            if isinstance(v, bool) or not isinstance(v, numbers.Integral) or v < 0:
                alt = ', "concat"' if k == "fold" else ""
                raise ValueError(f"{k} deve essere un intero ≥ 0{alt} o None, non {v!r}")
        elif not isinstance(v, str) or not v.strip() or any(c in v for c in "\r\n"):
            raise ValueError(f"{k} deve essere una stringa non vuota su una riga o None, non {v!r}")
    if keys["bins"] not in ("falcon", "legacy"):
        raise ValueError(f'bins deve essere "falcon" o "legacy", non {keys["bins"]!r}')
    if keys["session"] is not None and not re.fullmatch(r"S\d+_set_\d+", keys["session"]):
        raise ValueError(f'session va scritta come chiave FALCON (es. "S5_set_1"), non {keys["session"]!r}')


def _check_metrics(metrics) -> dict[str, float]:
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError("metrics deve essere un dict non vuoto {nome: valore}")
    out = {}
    for name, v in metrics.items():
        if not isinstance(name, str) or not name.strip() or any(c in name for c in "\r\n"):
            raise ValueError(f"nome di metrica non valido: {name!r}")
        try:
            out[name] = float(v)
        except (TypeError, ValueError):
            raise ValueError(f"il valore di {name} non è un numero: {v!r}") from None
    return out


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, numbers.Integral):
        return str(int(v))
    if isinstance(v, float):
        return repr(float(v))                        # anche np.float64 come numero semplice
    return str(v)


def _csv_line(values) -> bytes:
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerow(values)
    return buf.getvalue().encode()


# --------------------------------------------------------------------------------------------
# Versione del codice in esecuzione
# --------------------------------------------------------------------------------------------
def _inside(p: Path, root: Path) -> bool:
    try:
        p.relative_to(root)
        return True
    except ValueError:
        return False


def _git(*args, cwd) -> str:
    return subprocess.run(["git", "-C", str(cwd), *args], capture_output=True, text=True, check=True).stdout


def _blob(data: bytes) -> str:
    """Lo stesso hash che git dà al contenuto di un file (git hash-object)."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _process_start() -> float:
    try:
        import psutil
        return psutil.Process().create_time()
    except Exception:                                # senza psutil: l'ora dell'import di questo modulo
        return _T_IMPORT


def _main_file() -> Path | None:
    if "ipykernel" in sys.modules:                   # in un notebook __main__ è il kernel, non il nostro codice
        return None
    f = getattr(sys.modules.get("__main__"), "__file__", None)
    return Path(f).resolve() if f else None


def running_files() -> list[Path]:
    """I file del progetto in esecuzione: lo script principale e i moduli importati da WORK o GIT."""
    roots = [WORK.resolve(), GIT.resolve()]
    excl = [e.resolve() for e in EXCLUDE]
    cands = [_main_file()] + [getattr(m, "__file__", None) for m in list(sys.modules.values())]
    out = set()
    for f in cands:
        if not f or not str(f).endswith(".py"):
            continue
        if not Path(f).is_absolute():                # nomi finti senza file (torch._classes, torch._ops: "_classes.py")
            continue
        p = Path(f).resolve()
        if any(_inside(p, r) for r in roots) and not any(_inside(p, e) for e in excl):
            out.add(p)
    return sorted(out)


def code_version(files=None, *, work=WORK, git=GIT, start=None) -> tuple[str, list[str]]:
    """(commit, problemi): hash corto di HEAD in `git`, con "-dirty" se un file non coincide con HEAD.

    files: i file da controllare (di default quelli in esecuzione, più lo script principale, che deve esserci).
    Un file sotto `work` si confronta con lo stesso percorso relativo nel repo.
    start: ora di avvio del processo; un file modificato dopo potrebbe non essere quello eseguito.
    """
    problems = []
    if files is None:
        if _main_file() is None:
            problems.append("nessuno script (notebook o python -c): codice non verificabile")
        files = running_files()
    start = _process_start() if start is None else start
    try:
        head = _git("rev-parse", "--short", "HEAD", cwd=git).strip()
        tree = {}
        for entry in _git("ls-tree", "-r", "-z", "HEAD", cwd=git).split("\0"):
            if entry:
                meta, rel = entry.split("\t", 1)
                tree[rel] = meta.split()[2]
    except (OSError, subprocess.CalledProcessError) as e:
        return "nogit-dirty", problems + [f"git non disponibile in {git}: {e}"]
    roots = [Path(git).resolve(), Path(work).resolve()]
    for p in map(lambda f: Path(f).resolve(), files):
        rel = next((p.relative_to(r).as_posix() for r in roots if _inside(p, r)), None)
        if rel is None:
            problems.append(f"{p}: fuori da {work} e {git}")
        elif rel not in tree:
            problems.append(f"{p}: {rel} non è nel commit {head}")
        elif _blob(p.read_bytes()) != tree[rel]:
            problems.append(f"{p}: diverso da {head}:{rel}")
        elif p.stat().st_mtime > start:
            problems.append(f"{p}: modificato dopo l'avvio del processo")
    return head + ("-dirty" if problems else ""), problems


def _short(p: Path) -> str:
    s = str(p)
    home = str(Path.home())
    return "~" + s[len(home):] if s.startswith(home + os.sep) else s


def _provenance() -> dict:
    commit, problems = code_version()
    for pr in problems:                              # avviso una volta sola per processo
        if pr not in _WARNED:
            _WARNED.add(pr)
            print(f"[results] commit {commit}: {pr}", file=sys.stderr)
    main = _main_file()
    if "ipykernel" in sys.modules:
        script = os.environ.get("JPY_SESSION_NAME", "notebook")
    else:
        script = _short(main) if main else "interattivo"
    return dict(commit=commit, script=script)


# --------------------------------------------------------------------------------------------
# Scrittura
# --------------------------------------------------------------------------------------------
def log_run(metrics: dict, *, path=None, _auto: dict | None = None, **keys) -> Path:
    """Aggiunge al registro una riga per ogni metrica: log_run({"r2": 0.41}, esperimento=..., ..., seed=0).

    Tutto avviene dentro un lock sul file: il controllo "esiste / ha già l'intestazione" e la scrittura,
    così più processi in parallelo non scrivono due intestazioni né mescolano le righe.
    _auto: solo per importare risultati vecchi (timestamp, commit, script del run originale).
    """
    check_keys(**keys)
    vals = _check_metrics(metrics)
    auto = dict(timestamp=datetime.now().isoformat(timespec="seconds"), **_provenance())
    if _auto:
        bad = set(_auto) - set(AUTO)
        if bad:
            raise ValueError(f"_auto accetta solo {AUTO}, non {sorted(bad)}")
        auto.update(_auto)
    body = b"".join(_csv_line([_fmt(keys[k]) for k in KEYS] + [name, repr(v)] + [auto[k] for k in AUTO])
                    for name, v in vals.items())
    header = _csv_line(COLUMNS)
    path = Path(path or RUNS)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a+b") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0)
            first = f.readline()
            if not first:
                prefix = header
            elif first != header:
                raise ValueError(f"{path}: l'intestazione del file non è quella attesa {COLUMNS}")
            else:
                f.seek(-1, os.SEEK_END)
                prefix = b"" if f.read(1) == b"\n" else b"\n"
            f.write(prefix + body)                   # in modalità "a" la scrittura va sempre in fondo
            f.flush()
            os.fsync(f.fileno())
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)
    return path


def _preds_path(keys: dict, preds_dir) -> Path:
    name = "__".join(f"{k}={'NA' if keys[k] is None else keys[k]}" for k in KEYS)
    return Path(preds_dir) / (re.sub(r"[^A-Za-z0-9_.=+-]", "-", name) + ".npz")


def _r2_metrics(y, yh) -> dict[str, float]:
    from .decoding import DOF, r2_dof, r2_vw
    names = DOF if y.shape[1] == len(DOF) else [f"d{i}" for i in range(y.shape[1])]
    return {"r2": float(r2_vw(y, yh)), **{f"r2_{n}": float(v) for n, v in zip(names, r2_dof(y, yh))},
            "n_bin": float(len(y))}


def log_fold(y_true, y_pred, mask=None, *, extra: dict | None = None, path=None, preds_dir=None, **keys) -> dict:
    """Un fold: salva y_true, y_pred, mask in preds/ (float32) e registra r2 (pesato per varianza, come FALCON),
    r2 per grado di libertà e n_bin, calcolati sui bin con mask. extra: altre metriche dello stesso fold."""
    check_keys(**keys)
    if keys["fold"] is None or keys["fold"] == "concat":
        raise ValueError("log_fold: fold deve essere il numero del fold (0, 1, 2, ...)")
    extra = _check_metrics(extra) if extra else {}
    y, yh = np.asarray(y_true, np.float32), np.asarray(y_pred, np.float32)
    y, yh = (y[:, None], yh[:, None]) if y.ndim == 1 else (y, yh)
    if y.shape != yh.shape or y.ndim != 2:
        raise ValueError(f"y_true {y.shape} e y_pred {yh.shape} devono essere bin × uscite, uguali")
    m = np.ones(len(y), bool) if mask is None else np.asarray(mask, bool)
    if m.shape != (len(y),):
        raise ValueError(f"mask {m.shape} deve avere un valore per bin ({len(y)})")
    p = _preds_path(keys, preds_dir or PREDS)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, y_true=y, y_pred=yh, mask=m, keys=json.dumps(keys))
    os.replace(tmp, p)                               # niente file a metà se il processo si interrompe
    metrics = {**_r2_metrics(y[m], yh[m]), **extra}
    log_run(metrics, path=path, **keys)
    return metrics


def log_concat(folds, *, path=None, preds_dir=None, **keys) -> dict:
    """Finiti i fold: r2 sulle predizioni concatenate (rilette da preds/), registrato con fold="concat".
    folds: i fold attesi (es. [0, 1, 2]); se ne manca uno -> ValueError, niente riga."""
    if "fold" in keys:
        raise ValueError('log_concat: non passare fold, si mette da solo a "concat"')
    check_keys(fold="concat", **keys)
    folds = list(folds)
    if not folds or len(set(folds)) != len(folds):
        raise ValueError(f"folds: lista non vuota senza ripetizioni, non {folds}")
    files = {k: _preds_path({**keys, "fold": k}, preds_dir or PREDS) for k in folds}
    missing = [k for k, p in files.items() if not p.exists()]
    if missing:
        raise ValueError(f"mancano le predizioni dei fold {missing} (attese in {Path(preds_dir or PREDS)})")
    ys, yhs = [], []
    for p in files.values():
        with np.load(p) as z:
            m = z["mask"]
            ys.append(z["y_true"][m])
            yhs.append(z["y_pred"][m])
    metrics = _r2_metrics(np.concatenate(ys), np.concatenate(yhs))
    log_run(metrics, path=path, fold="concat", **keys)
    return metrics


# --------------------------------------------------------------------------------------------
# Lettura
# --------------------------------------------------------------------------------------------
def load_runs(path=None, *, keep_last: bool = True):
    """Il registro come DataFrame; con keep_last tiene solo l'ultima riga di ogni (chiavi, metrica)."""
    import pandas as pd
    dtype = {k: ("Int64" if k in INT_KEYS else "string") for k in KEYS}
    dtype.update(metrica="string", valore="float64", **{k: "string" for k in AUTO})
    # round_trip: rilegge esattamente i numeri scritti (il parser di default sbaglia l'ultima cifra, ~1e-17)
    df = pd.read_csv(path or RUNS, dtype=dtype, float_precision="round_trip")
    if list(df.columns) != COLUMNS:
        raise ValueError(f"colonne del registro {list(df.columns)} diverse da {COLUMNS}")
    if keep_last:
        df = df.drop_duplicates(KEYS + ["metrica"], keep="last")
    return df.reset_index(drop=True)


def select(df, **where):
    """Righe con chiave == valore; valore None = solo le righe con NA in quella chiave."""
    for k, v in where.items():
        df = df[df[k].isna()] if v is None else df[(df[k].astype("string") == str(v)).fillna(False).astype(bool)]
    return df


def summary(df, by, metrica: str = "r2", **where):
    """Media, deviazione standard e numero di run per gruppo; le righe con NA nelle chiavi restano (dropna=False)."""
    d = select(df[df["metrica"] == metrica], **where)
    return d.groupby(list(by), dropna=False)["valore"].agg(mean="mean", std="std", n="count")


def add_days(df, sessions=None):
    """Unisce sessions.csv: giorno, gruppo e giorni_da_held_in (giorni dall'ultima sessione held-in; 0 = S5)."""
    import pandas as pd
    s = pd.read_csv(sessions or SESSIONS, dtype="string")
    data = pd.to_datetime(s["data"])
    s["giorni_da_held_in"] = (data - data[s["gruppo"] == "held_in"].max()).dt.days.astype("Int64")
    return df.merge(s[["session", "giorno", "gruppo", "giorni_da_held_in"]], on="session", how="left")
