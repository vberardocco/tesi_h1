"""Protocollo cross-sessione few-shot (src/tesi_h1/crosssession.py): un metodo, un seme, una dimensione.

Uso: python scripts/xs_run.py METODO [--seed S] [--dim D] [--experiment E] [--stage tune|main|all|diag]
                                     [--n-jobs N] [--sessions K1,K2] [--force] [--wait-tune ORE]
  METODO: PCA FA GPFA Isomap UMAP CEBRA LFADS, oppure rates (ridge_rates) o ridge_channels (decoder sui canali del 05)
  --dim: dimensione latente (default: quella del metodo; Isomap, UMAP, CEBRA: 25 dalla seconda tornata (31/10), vedi DIM_RULE)
  --experiment: valore di "esperimento" nel registro (default "crosssession"; le mappe a d = 3: "crosssession_d3")
  tune  (solo seme 0): pseudo-few-shot sulle held-in -> XS/tune/METODO_dD.json; se c'è già si salta (--force rifà)
  main  : modello held-in sulle 13 sessioni, poi le 14 held-out; ogni sessione finita va subito nel registro e le
          sessioni già presenti (stesso esperimento, metodo, d, seme, tutte le condizioni) si saltano
  all   : tune (se serve) e main
  diag  : diagnosi del moment matching dentro le 13 sessioni held-in (esperimento "crosssession_mmdiag")
Le scelte dello pseudo-few-shot (λ, α) valgono per tutti i semi.
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
REF = ("rates", "ridge_channels")


def tune_file(method, dim):
    return xs.tune_path(method, None if method in REF else dim)


def stage_tune(method, dim, n_jobs, force):
    p = tune_file(method, dim)
    if p.exists() and not force:
        log(f"pseudo-few-shot di {method} (d = {dim}) già fatto ({p})")
        return json.loads(p.read_text())
    return xs.tune(method, xs.load_data(), seed=0, n_jobs=n_jobs, dim=dim)


def wait_tune(p, wait):
    """Il file dello pseudo-few-shot; con wait > 0 lo aspetta fino a `wait` ore (per stimare intanto il modello held-in)."""
    t0 = time.time()
    while not p.exists():
        if time.time() - t0 > wait * 3600:
            raise SystemExit(f"manca {p}: prima --stage tune")
        time.sleep(60)
    return json.loads(p.read_text())


def stage_main(method, seed, dim, experiment, n_jobs, sessions, force, wait=0.0):
    p = tune_file(method, dim)
    if not p.exists() and wait <= 0:
        raise SystemExit(f"manca {p}: prima --stage tune")
    data = xs.load_data()
    targets = [s for s in xs.heldout(data) if not sessions or s.key in sessions]
    reg_dim = None if method in REF else dim
    done = set() if force else xs.done_runs(experiment=experiment, dim=reg_dim)
    exp = xs.expected_conditions(method)
    todo = [s for s in targets if not all((method, seed, s.key, c) in done for c in exp)]
    log(f"{method} d = {dim} seme {seed} [{experiment}]: {len(targets) - len(todo)} sessioni già nel registro, "
        f"{len(todo)} da fare")
    if not todo:
        return
    alpha_src = None
    if method == "rates":
        tuned = wait_tune(p, wait)
        alphas = [tuned["ridge_rates"]["alpha"]]
        fn = lambda s: xs.eval_rates(s, alphas)  # noqa: E731
    elif method == "ridge_channels":
        tuned = wait_tune(p, wait)
        dec = xs.ChannelDecoder.load()
        lams = [tuned["fewshot"]["lam"]]
        fn = lambda s: xs.eval_channels(dec, s, lams)  # noqa: E731
    else:
        b = xs.fit_bundle(method, seed, xs.heldin(data), "heldin", dim=dim)
        info = xs.XS / "models" / f"{method}_d{dim}_heldin_s{seed}.json"
        info.parent.mkdir(parents=True, exist_ok=True)
        info.write_text(json.dumps(b.info, indent=1, default=float))
        alpha_src = b.readout.alpha
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
            xs.log_session(method, reg_dim, s, preds, seed, alpha_src=alpha_src, experiment=experiment)
            r2 = {c: xs.concat_r2(s, v) for (c, _), v in preds.items()}
            log(f"{method} d = {dim} seme {seed} {s.key}: " + ", ".join(f"{c} {v:.3f}" for c, v in r2.items()))
        log(f"   ({len(chunk)} sessioni in {(time.time() - t0) / 60:.1f} min; {i + len(chunk)}/{len(todo)})")


def stage_diag(method, seed, dim, n_jobs, force):
    """Moment matching dentro le sessioni held-in (nessuna deriva tra giorni), modello held-in, seme dato."""
    exp_name = "crosssession_mmdiag"
    data = xs.load_data()
    done = set() if force else xs.done_runs(experiment=exp_name, dim=dim)
    todo = [s for s in xs.heldin(data)
            if not all((method, seed, s.key, c) in done for c in ("no_adapt", "recenter", "align"))]
    log(f"diagnosi {method} d = {dim} seme {seed}: {len(todo)} sessioni held-in da fare")
    if not todo:
        return
    b = xs.fit_bundle(method, seed, xs.heldin(data), "heldin", dim=dim)
    fn = lambda s: xs.diag_within(b, s)  # noqa: E731
    step = max(1, n_jobs)
    for i in range(0, len(todo), step):
        chunk = todo[i:i + step]
        for s, preds in zip(chunk, xs._par(n_jobs, fn, chunk)):
            xs.log_session(method, dim, s, preds, seed, experiment=exp_name, split="within_session_loto",
                           decoder="ridge_within")
            r2 = {c: xs.concat_r2(s, v) for (c, _), v in preds.items()}
            log(f"diagnosi {method} {s.key}: " + ", ".join(f"{c} {v:.3f}" for c, v in r2.items()))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("method", choices=list(xs.METHODS) + list(REF))
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dim", type=int, default=None)
    ap.add_argument("--experiment", default=xs.EXPERIMENT)
    ap.add_argument("--stage", choices=["tune", "main", "all", "diag"], default="all")
    ap.add_argument("--n-jobs", type=int, default=1)
    ap.add_argument("--sessions", default="")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--wait-tune", type=float, default=0.0, help="ore di attesa del file dello pseudo-few-shot (main)")
    a = ap.parse_args()
    dim = None if a.method in REF else (a.dim or xs.default_dim(a.method))
    commit, problems = res.code_version()
    log(f"xs_run {a.method} d = {dim} seme {a.seed} stage {a.stage} [{a.experiment}]; codice {commit}"
        + (f" ({problems})" if problems else ""))
    t0 = time.time()
    if a.stage == "diag":
        stage_diag(a.method, a.seed, dim, a.n_jobs, a.force)
    if a.stage in ("tune", "all"):
        if a.seed != 0:
            raise SystemExit("lo pseudo-few-shot si fa solo con il seme 0")
        stage_tune(a.method, dim, a.n_jobs, a.force)
    if a.stage in ("main", "all"):
        stage_main(a.method, a.seed, dim, a.experiment, a.n_jobs, [k for k in a.sessions.split(",") if k], a.force,
                   a.wait_tune)
    log(f"fine ({(time.time() - t0) / 60:.1f} min)")


if __name__ == "__main__":
    main()
