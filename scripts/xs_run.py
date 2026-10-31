"""Protocollo cross-sessione few-shot (src/tesi_h1/crosssession.py): un metodo, un seme.

Uso: python scripts/xs_run.py METODO [--seed S] [--stage tune|main|all] [--n-jobs N] [--sessions K1,K2] [--force]
  METODO: PCA FA GPFA Isomap UMAP CEBRA LFADS rates (rates = il riferimento ridge_rates)
  tune  (solo seme 0): pseudo-few-shot sulle held-in -> XS/tune/METODO.json; se il file c'è già si salta (--force rifà)
  main  : modello held-in sulle 13 sessioni, poi le 14 held-out; ogni sessione finita va subito nel registro
          (results/h1_runs.csv) e le sessioni già presenti si saltano: dopo un'interruzione basta rilanciare.
  all   : tune (se serve) e main.
Le scelte dello pseudo-few-shot (λ, α) si leggono da XS/tune/METODO.json, stimato con il seme 0: valgono per tutti i semi.
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")     # niente seconda copia di OpenMP per UMAP

import numpy as np  # noqa: E402  (numpy prima di torch)
import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tesi_h1 import crosssession as xs  # noqa: E402
from tesi_h1 import results as res  # noqa: E402

log = xs.log


def stage_tune(method, n_jobs, force):
    p = xs.tune_path(method)
    if p.exists() and not force:
        log(f"pseudo-few-shot di {method} già fatto ({p})")
        return json.loads(p.read_text())
    return xs.tune(method, xs.load_data(), seed=0, n_jobs=n_jobs)


def wait_tune(p, wait):
    """Il file dello pseudo-few-shot; con wait > 0 lo aspetta fino a `wait` ore (per stimare intanto il modello held-in)."""
    t0 = time.time()
    while not p.exists():
        if time.time() - t0 > wait * 3600:
            raise SystemExit(f"manca {p}: prima --stage tune")
        time.sleep(60)
    return json.loads(p.read_text())


def stage_main(method, seed, n_jobs, sessions, force, wait=0.0):
    p = xs.tune_path(method)
    if not p.exists() and wait <= 0:
        raise SystemExit(f"manca {p}: prima --stage tune")
    data = xs.load_data()
    targets = [s for s in xs.heldout(data) if not sessions or s.key in sessions]
    done = set() if force else xs.done_runs()
    exp = xs.expected_conditions(method)
    todo = [s for s in targets if not all((method, seed, s.key, c) in done for c in exp)]
    log(f"{method} seme {seed}: {len(targets) - len(todo)} sessioni già nel registro, {len(todo)} da fare")
    if not todo:
        return
    alpha_src = None
    if method == "rates":
        tuned = wait_tune(p, wait)
        dim, alphas = None, [tuned["ridge_rates"]["alpha"]]
        fn = lambda s: xs.eval_rates(s, alphas)  # noqa: E731
    else:
        b = xs.fit_bundle(method, seed, xs.heldin(data), "heldin")
        info = xs.XS / "models" / f"{method}_heldin_s{seed}.json"
        info.parent.mkdir(parents=True, exist_ok=True)
        info.write_text(json.dumps(b.info, indent=1, default=float))
        dim, alpha_src = b.model.dim, b.readout.alpha
        if not p.exists():
            log(f"aspetto {p}")
        tuned = wait_tune(p, wait)
        lams = [tuned["fewshot"]["lam"]]
        alphas = [tuned["scratch"]["alpha"]] if "scratch" in tuned else []
        scratch = method != "LFADS"
        fn = lambda s: xs.eval_session(b, s, lams, alphas, scratch=scratch)  # noqa: E731
    step = max(1, n_jobs)
    for i in range(0, len(todo), step):
        chunk = todo[i:i + step]
        t0 = time.time()
        outs = xs._par(n_jobs, fn, chunk)
        for s, preds in zip(chunk, outs):
            xs.log_session(method, dim, s, preds, seed, alpha_src=alpha_src)
            r2 = {c: xs.concat_r2(s, v) for (c, _), v in preds.items()}
            log(f"{method} seme {seed} {s.key}: " + ", ".join(f"{c} {v:.3f}" for c, v in r2.items()))
        log(f"   ({len(chunk)} sessioni in {(time.time() - t0) / 60:.1f} min; {i + len(chunk)}/{len(todo)})")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("method", choices=list(xs.METHODS) + ["rates"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stage", choices=["tune", "main", "all"], default="all")
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--sessions", default="")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--wait-tune", type=float, default=0.0, help="ore di attesa del file dello pseudo-few-shot (main)")
    a = ap.parse_args()
    commit, problems = res.code_version()
    log(f"xs_run {a.method} seme {a.seed} stage {a.stage}; codice {commit}" + (f" ({problems})" if problems else ""))
    t0 = time.time()
    if a.stage in ("tune", "all"):
        if a.seed != 0:
            raise SystemExit("lo pseudo-few-shot si fa solo con il seme 0")
        stage_tune(a.method, a.n_jobs, a.force)
    if a.stage in ("main", "all"):
        stage_main(a.method, a.seed, a.n_jobs, [k for k in a.sessions.split(",") if k], a.force, a.wait_tune)
    log(f"fine ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
