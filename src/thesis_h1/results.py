"""Run registry: one row per (run, metric) in a CSV, from which the thesis tables are built with a script.

Files, in the RESULTS folder (~/thesis_h1_work/results, or the environment variable THESIS_RESULTS):
- h1_runs.csv   the registry; written only through log_run / log_fold / log_concat;
- preds/*.npz   y_true, y_pred, mask of each fold (log_fold), to recompute the metrics;
- sessions.csv  session, day, sid, data (date), group; merged only when needed (add_days).

Columns of the registry, in this order:
- the run keys (KEYS): ALL must be passed at every call, None = not applicable (NA in the CSV);
  a missing or extra key -> ValueError and the file is not touched.
    experiment   e.g. "replica", "scala"
    method       what goes into the decoder: "counts", "PCA", "FA", "GPFA", "LFADS", "CEBRA"
    latent_dim   latent dimensions (int)
    decoder      e.g. "ridge"
    session      evaluated session, FALCON key (e.g. "S5_set_1")
    bins         "falcon" (load_nwb) or "legacy" (legacy bins of io.py): numbers not comparable with each other
    split        evaluation data: "test20" (final 20% of the session calibration set), "val", "cv",
                 "minival", "test"
    condition    e.g. "zero_shot", "recenter", "alignment_fa", "kin_2trial", "scratch"
    fold         0, 1, 2, ... or "concat" (R² on the concatenated predictions of the folds, log_concat)
    k_shot       calibration trials (int)
    H_ms         decoder history in ms (int)
    seed         seed (int)
- metric, value: one row for each metric of the run;
- timestamp, commit, script: added by log_run.

commit: short hash of HEAD in the ~/thesis_h1_git repo, with "-dirty" if one of the project files that the process
is running (the script and the modules imported from ~/thesis_h1_work or ~/thesis_h1_git) is not identical to its
version in HEAD, is not in the repo, or was modified after the process started. The files of ~/thesis_h1_work are
compared with the same relative path in the repo (rep/dec_rep.py, src/thesis_h1/decoding.py, ...).
Notebooks and `python -c` have no script to compare: always "-dirty".

Re-running a run appends new rows: load_runs keeps the last one for each (keys, metric).
To catch an error in the keys right away, at the start of a script: check_keys(**keys).
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

WORK = Path.home() / "thesis_h1_work"                 # where the runs are executed (outside iCloud)
GIT = Path.home() / "thesis_h1_git"                   # the git repo, aligned with GitHub
RESULTS = Path(os.environ.get("THESIS_RESULTS", WORK / "results"))
RUNS = RESULTS / "h1_runs.csv"
PREDS = RESULTS / "preds"
SESSIONS = RESULTS / "sessions.csv"

KEYS = ["experiment", "method", "latent_dim", "decoder", "session", "bins", "split",
        "condition", "fold", "k_shot", "H_ms", "seed"]
INT_KEYS = {"latent_dim", "k_shot", "H_ms", "seed"}
AUTO = ["timestamp", "commit", "script"]
COLUMNS = KEYS + ["metric", "value"] + AUTO
EXCLUDE = [WORK / "falcon-challenge"]               # FALCON clone for Docker: not our code

_T_IMPORT = time.time()
_WARNED: set[str] = set()


# --------------------------------------------------------------------------------------------
# Key checks
# --------------------------------------------------------------------------------------------
def check_keys(**keys):
    """Check the keys of a run (the same as log_run). ValueError if something is wrong."""
    missing = [k for k in KEYS if k not in keys]
    extra = sorted(k for k in keys if k not in KEYS)
    if missing or extra:
        raise ValueError(f"missing keys: {missing}; extra keys: {extra}")
    for k in KEYS:
        v = keys[k]
        if v is None or (k == "fold" and v == "concat"):
            continue
        if k in INT_KEYS or k == "fold":
            if isinstance(v, bool) or not isinstance(v, numbers.Integral) or v < 0:
                alt = ', "concat"' if k == "fold" else ""
                raise ValueError(f"{k} must be an integer ≥ 0{alt} or None, not {v!r}")
        elif not isinstance(v, str) or not v.strip() or any(c in v for c in "\r\n"):
            raise ValueError(f"{k} must be a non-empty single-line string or None, not {v!r}")
    if keys["bins"] not in ("falcon", "legacy"):
        raise ValueError(f'bins must be "falcon" or "legacy", not {keys["bins"]!r}')
    if keys["session"] is not None and not re.fullmatch(r"S\d+_set_\d+", keys["session"]):
        raise ValueError(f'session must be written as a FALCON key (e.g. "S5_set_1"), not {keys["session"]!r}')


def _check_metrics(metrics) -> dict[str, float]:
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError("metrics must be a non-empty dict {name: value}")
    out = {}
    for name, v in metrics.items():
        if not isinstance(name, str) or not name.strip() or any(c in name for c in "\r\n"):
            raise ValueError(f"invalid metric name: {name!r}")
        try:
            out[name] = float(v)
        except (TypeError, ValueError):
            raise ValueError(f"the value of {name} is not a number: {v!r}") from None
    return out


def _fmt(v) -> str:
    if v is None:
        return ""
    if isinstance(v, numbers.Integral):
        return str(int(v))
    if isinstance(v, float):
        return repr(float(v))                        # np.float64 too as a plain number
    return str(v)


def _csv_line(values) -> bytes:
    buf = io.StringIO()
    csv.writer(buf, lineterminator="\n").writerow(values)
    return buf.getvalue().encode()


# --------------------------------------------------------------------------------------------
# Version of the running code
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
    """The same hash that git gives to the content of a file (git hash-object)."""
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _process_start() -> float:
    try:
        import psutil
        return psutil.Process().create_time()
    except Exception:                                # without psutil: the time this module was imported
        return _T_IMPORT


def _main_file() -> Path | None:
    if "ipykernel" in sys.modules:                   # in a notebook __main__ is the kernel, not our code
        return None
    f = getattr(sys.modules.get("__main__"), "__file__", None)
    return Path(f).resolve() if f else None


def running_files() -> list[Path]:
    """The project files being run: the main script and the modules imported from WORK or GIT."""
    roots = [WORK.resolve(), GIT.resolve()]
    excl = [e.resolve() for e in EXCLUDE]
    cands = [_main_file()] + [getattr(m, "__file__", None) for m in list(sys.modules.values())]
    out = set()
    for f in cands:
        if not f or not str(f).endswith(".py"):
            continue
        if not Path(f).is_absolute():                # fake names without a file (torch._classes, torch._ops: "_classes.py")
            continue
        p = Path(f).resolve()
        if any(_inside(p, r) for r in roots) and not any(_inside(p, e) for e in excl):
            out.add(p)
    return sorted(out)


def code_version(files=None, *, work=WORK, git=GIT, start=None) -> tuple[str, list[str]]:
    """(commit, problems): short hash of HEAD in `git`, with "-dirty" if a file does not match HEAD.

    files: the files to check (by default those being run, plus the main script, which must be present).
    A file under `work` is compared with the same relative path in the repo.
    start: process start time; a file modified afterwards may not be the one that was run.
    """
    problems = []
    if files is None:
        if _main_file() is None:
            problems.append("no script (notebook or python -c): code cannot be verified")
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
        return "nogit-dirty", problems + [f"git not available in {git}: {e}"]
    roots = [Path(git).resolve(), Path(work).resolve()]
    for p in map(lambda f: Path(f).resolve(), files):
        rel = next((p.relative_to(r).as_posix() for r in roots if _inside(p, r)), None)
        if rel is None:
            problems.append(f"{p}: outside {work} and {git}")
        elif rel not in tree:
            problems.append(f"{p}: {rel} is not in commit {head}")
        elif _blob(p.read_bytes()) != tree[rel]:
            problems.append(f"{p}: different from {head}:{rel}")
        elif p.stat().st_mtime > start:
            problems.append(f"{p}: modified after the process started")
    return head + ("-dirty" if problems else ""), problems


def _short(p: Path) -> str:
    s = str(p)
    home = str(Path.home())
    return "~" + s[len(home):] if s.startswith(home + os.sep) else s


def _provenance() -> dict:
    commit, problems = code_version()
    for pr in problems:                              # warn only once per process
        if pr not in _WARNED:
            _WARNED.add(pr)
            print(f"[results] commit {commit}: {pr}", file=sys.stderr)
    main = _main_file()
    if "ipykernel" in sys.modules:
        script = os.environ.get("JPY_SESSION_NAME", "notebook")
    else:
        script = _short(main) if main else "interactive"
    return dict(commit=commit, script=script)


# --------------------------------------------------------------------------------------------
# Writing
# --------------------------------------------------------------------------------------------
def log_run(metrics: dict, *, path=None, _auto: dict | None = None, **keys) -> Path:
    """Append one row per metric to the registry: log_run({"r2": 0.41}, experiment=..., ..., seed=0).

    Everything happens inside a lock on the file: the check "exists / already has the header" and the write,
    so that several parallel processes neither write two headers nor interleave the rows.
    _auto: only for importing old results (timestamp, commit, script of the original run).
    """
    check_keys(**keys)
    vals = _check_metrics(metrics)
    auto = dict(timestamp=datetime.now().isoformat(timespec="seconds"), **_provenance())
    if _auto:
        bad = set(_auto) - set(AUTO)
        if bad:
            raise ValueError(f"_auto accepts only {AUTO}, not {sorted(bad)}")
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
                raise ValueError(f"{path}: the file header is not the expected {COLUMNS}")
            else:
                f.seek(-1, os.SEEK_END)
                prefix = b"" if f.read(1) == b"\n" else b"\n"
            f.write(prefix + body)                   # in "a" mode the write always goes to the end
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
    """One fold: save y_true, y_pred, mask in preds/ (float32) and log r2 (variance-weighted, as in FALCON),
    r2 per degree of freedom and n_bin, computed on the bins with mask. extra: other metrics of the same fold."""
    check_keys(**keys)
    if keys["fold"] is None or keys["fold"] == "concat":
        raise ValueError("log_fold: fold must be the fold number (0, 1, 2, ...)")
    extra = _check_metrics(extra) if extra else {}
    y, yh = np.asarray(y_true, np.float32), np.asarray(y_pred, np.float32)
    y, yh = (y[:, None], yh[:, None]) if y.ndim == 1 else (y, yh)
    if y.shape != yh.shape or y.ndim != 2:
        raise ValueError(f"y_true {y.shape} and y_pred {yh.shape} must be bins × outputs, and equal")
    m = np.ones(len(y), bool) if mask is None else np.asarray(mask, bool)
    if m.shape != (len(y),):
        raise ValueError(f"mask {m.shape} must have one value per bin ({len(y)})")
    p = _preds_path(keys, preds_dir or PREDS)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    with open(tmp, "wb") as fh:
        np.savez_compressed(fh, y_true=y, y_pred=yh, mask=m, keys=json.dumps(keys))
    os.replace(tmp, p)                               # no half-written file if the process is interrupted
    metrics = {**_r2_metrics(y[m], yh[m]), **extra}
    log_run(metrics, path=path, **keys)
    return metrics


def log_concat(folds, *, path=None, preds_dir=None, **keys) -> dict:
    """After the folds: r2 on the concatenated predictions (re-read from preds/), logged with fold="concat".
    folds: the expected folds (e.g. [0, 1, 2]); if one is missing -> ValueError, no row."""
    if "fold" in keys:
        raise ValueError('log_concat: do not pass fold, it is set to "concat" automatically')
    check_keys(fold="concat", **keys)
    folds = list(folds)
    if not folds or len(set(folds)) != len(folds):
        raise ValueError(f"folds: non-empty list without repetitions, not {folds}")
    files = {k: _preds_path({**keys, "fold": k}, preds_dir or PREDS) for k in folds}
    missing = [k for k, p in files.items() if not p.exists()]
    if missing:
        raise ValueError(f"predictions of folds {missing} are missing (expected in {Path(preds_dir or PREDS)})")
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
# Reading
# --------------------------------------------------------------------------------------------
def load_runs(path=None, *, keep_last: bool = True):
    """The registry as a DataFrame; with keep_last only the last row of each (keys, metric) is kept."""
    import pandas as pd
    dtype = {k: ("Int64" if k in INT_KEYS else "string") for k in KEYS}
    dtype.update(metric="string", value="float64", **{k: "string" for k in AUTO})
    # round_trip: re-reads exactly the numbers written (the default parser gets the last digit wrong, ~1e-17)
    df = pd.read_csv(path or RUNS, dtype=dtype, float_precision="round_trip")
    if list(df.columns) != COLUMNS:
        raise ValueError(f"registry columns {list(df.columns)} differ from {COLUMNS}")
    if keep_last:
        df = df.drop_duplicates(KEYS + ["metric"], keep="last")
    return df.reset_index(drop=True)


def select(df, **where):
    """Rows with key == value; value None = only the rows with NA in that key."""
    for k, v in where.items():
        df = df[df[k].isna()] if v is None else df[(df[k].astype("string") == str(v)).fillna(False).astype(bool)]
    return df


def summary(df, by, metric: str = "r2", **where):
    """Mean, standard deviation and number of runs per group; rows with NA in the keys are kept (dropna=False)."""
    d = select(df[df["metric"] == metric], **where)
    return d.groupby(list(by), dropna=False)["value"].agg(mean="mean", std="std", n="count")


def add_days(df, sessions=None):
    """Merge sessions.csv: day, group and days_from_held_in (days since the last held-in session; 0 = S5)."""

    import pandas as pd
    s = pd.read_csv(sessions or SESSIONS, dtype="string")
    data = pd.to_datetime(s["data"])
    s["days_from_held_in"] = (data - data[s["group"] == "held_in"].max()).dt.days.astype("Int64")
    return df.merge(s[["session", "day", "group", "days_from_held_in"]], on="session", how="left")
