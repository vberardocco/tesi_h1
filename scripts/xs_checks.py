"""Controlli esplorativi dopo la seconda tornata (31/10 sera, non nel piano): spiegano due risultati della seconda tornata.

1. gain  — da dove viene il calo di fewshot_gain rispetto ad align (ridge_channels: rispetto a recenter). Sulle 14
   held-out, stesso fold e stesso modello held-in (seme 0, dimensione principale): guadagno + offset (= fewshot_gain),
   solo guadagno (offset = intercetta del readout held-in), solo offset (guadagno 1), e un "oracolo" con guadagno e
   offset stimati sul trial di test stesso (limite superiore, NON una condizione). In più il guadagno stimato su ogni
   trial da solo, per vedere quanto cambia da un trial all'altro della stessa sessione.
   -> results/crosssession/check_gain_decomposition.csv, check_gain_per_trial.csv
2. within — le mappe (Isomap, UMAP, CEBRA) a d = 25 restano a R² ≈ 0: è un bug o una proprietà dei latenti?
   Sulle 13 sessioni held-in, con il modello held-in (seme 0, dimensione principale):
   a) quota della varianza dei latenti (standardizzati) spiegata dall'identità della sessione (η², media sulle
      dimensioni); come riferimento la stessa quota per la velocità;
   b) decodifica dentro la sessione: readout (stessa storia) stimato sui trial della sessione tranne gli ultimi 3,
      α scelto leave-one-trial-out su quei trial, R² sugli ultimi 3 (eval_mask, concatenati).
   -> results/crosssession/check_within_session.csv, check_session_variance.csv

Uso: python scripts/xs_checks.py [gain|within|all] [--methods PCA,FA,...] [--sessions K1,K2] [--out DIR]
"""
import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")

import numpy as np  # noqa: E402  (numpy prima di torch)
import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tesi_h1 import crosssession as xs  # noqa: E402
from tesi_h1 import results as res  # noqa: E402

METHODS = ["PCA", "FA", "GPFA", "Isomap", "UMAP", "CEBRA", "LFADS"]
log = xs.log


def lstsq_gain(yh, y):
    return xs.fit_gain(yh, y)


def gain_variants(base_te, base_ad, Y_ad, c, y_te, m_te):
    """Previsioni sul trial di test per le varianti del guadagno. base_* = previsioni di align (o recenter) sul test e
    sui bin con eval_mask dei 2 trial di adattamento; c = intercetta del readout held-in."""
    g, o = xs.fit_gain(base_ad, Y_ad)
    gg = ((base_ad - c) * (Y_ad - c)).sum(0) / ((base_ad - c) ** 2).sum(0)
    oo = (Y_ad - base_ad).mean(0)
    go, oo2 = xs.fit_gain(base_te[m_te], y_te[m_te])
    return {"base": base_te, "gain_offset": base_te * g + o, "gain_only": c + gg * (base_te - c),
            "offset_only": base_te + oo, "oracle_test": base_te * go + oo2}, g


def check_gain(methods, sessions, out):
    data = xs.load_data()
    targets = [s for s in xs.heldout(data) if not sessions or s.key in sessions]
    rows, grows = [], []
    for method in methods:
        if method == "ridge_channels":
            dec = xs.ChannelDecoder.load()
        else:
            b = xs.fit_bundle(method, 0, xs.heldin(data), "heldin", dim=xs.default_dim(method))
        for s in targets:
            tr = s.trials
            preds = {}
            for k, ad in xs.folds(s):
                tad = [tr[j] for j in ad]
                Y_ad = np.concatenate([t.vel[t.mask] for t in tad])
                if method == "ridge_channels":
                    mu_a = np.concatenate([t.counts_all for t in tad]).astype(np.float64).mean(0)
                    base_ad = np.concatenate([(dec.feats(t.counts_all, mu_a) @ dec.W + dec.b)[t.mask] for t in tad])
                    base_te = dec.predict(tr[k].counts_all, mu_a)
                    c = dec.b
                    per_trial = [(dec.predict(t.counts_all, mu_a)[t.mask], t.vel[t.mask]) for t in tad]
                else:
                    A = xs.adapt(b, tad, [1e30], [], scratch=False)
                    lat = b.model.encode([xs.rep(t, b.model.rep) - A.mu_a + b.mu_rep for t in tad])
                    ya = [b.readout.predict(A.f(z)) for z in lat]
                    base_ad = np.concatenate([y[t.mask] for y, t in zip(ya, tad)])
                    base_te = xs.predict_trial(b, A, tr[k])[("align", None)]
                    c = b.readout.b
                    per_trial = [(y[t.mask], t.vel[t.mask]) for y, t in zip(ya, tad)]
                v, g = gain_variants(base_te, base_ad, Y_ad, c, tr[k].vel, tr[k].mask)
                for name, p in v.items():
                    preds.setdefault(name, [None] * len(tr))[k] = p
                for j, (yh, y) in zip(ad, per_trial):     # guadagno di ogni trial da solo
                    gj, _ = xs.fit_gain(yh, y)
                    for d in range(7):
                        grows.append(dict(method=method, session=s.key, fold_test_trial=k, trial=j, dof=d, g=gj[d],
                                          g_two_trials=g[d]))
            r = dict(method=method, session=s.key)
            r.update({name: xs.concat_r2(s, p) for name, p in preds.items()})
            rows.append(r)
            log(f"gain {method} {s.key}: " + ", ".join(f"{n} {v:.3f}" for n, v in r.items() if n not in ("method", "session")))
    df = pd.DataFrame(rows)
    df.to_csv(out / "check_gain_decomposition.csv", index=False)
    pd.DataFrame(grows).to_csv(out / "check_gain_per_trial.csv", index=False)
    return df


def eta2(Zs, groups):
    """Quota della varianza (per colonna, poi media) spiegata dalla media di gruppo."""
    Z = np.concatenate(Zs)
    gid = np.concatenate([np.full(len(z), g) for z, g in zip(Zs, groups)])
    mu = Z.mean(0)
    tot = ((Z - mu) ** 2).sum(0)
    btw = np.zeros(Z.shape[1])
    for g in np.unique(gid):
        sel = gid == g
        btw += sel.sum() * (Z[sel].mean(0) - mu) ** 2
    return float(np.mean(btw / np.where(tot > 0, tot, 1.0)))


def check_within(methods, sessions, out, n_test=3):
    data = xs.load_data()
    hi = xs.heldin(data)
    rows, vrows = [], []
    vel = [t.vel for s in hi for t in s.trials]
    sess_of = [s.key for s in hi for _ in s.trials]
    vrows.append(dict(method="velocity (reference)", dim=7, eta2_session=eta2(vel, sess_of)))
    for method in methods:
        dim = xs.default_dim(method)
        b = xs.fit_bundle(method, 0, hi, "heldin", dim=dim)
        Zs = b.model.encode([xs.rep(t, b.model.rep) for s in hi for t in s.trials])
        Zstd = [(z - b.readout.m) / b.readout.s for z in Zs]
        vrows.append(dict(method=method, dim=dim, eta2_session=eta2(Zstd, sess_of),
                          lodo_r2_heldin=b.info["lodo_r2"], alpha_heldin=b.info["alpha"]))
        log(f"within {method}: η² sessione {vrows[-1]['eta2_session']:.3f}")
        i0 = 0
        for s in hi:
            n = len(s.trials)
            Zs_s, i0 = Zs[i0:i0 + n], i0 + n
            if sessions and s.key not in sessions:
                continue
            tr_tr, z_tr = s.trials[:n - n_test], Zs_s[:n - n_test]
            tr_te, z_te = s.trials[n - n_test:], Zs_s[n - n_test:]
            rd = xs.Readout(*xs.zstats(np.concatenate(z_tr)))
            rd = xs.fit_readout_lodo(rd, z_tr, tr_tr, list(range(len(tr_tr))))
            y = np.concatenate([t.vel[t.mask] for t in tr_te])
            yh = np.concatenate([rd.predict(z)[t.mask] for z, t in zip(z_te, tr_te)])
            # stesso test con il readout held-in congelato (stimato anche su questi trial: solo riferimento)
            yh0 = np.concatenate([b.readout.predict(z)[t.mask] for z, t in zip(z_te, tr_te)])
            rows.append(dict(method=method, dim=dim, session=s.key, n_train_trials=len(tr_tr), n_test_trials=n_test,
                             alpha_loto=rd.alpha, alpha_at_edge=rd.alpha_at_edge, r2_within_session=xs.r2_vw(y, yh),
                             r2_heldin_readout=xs.r2_vw(y, yh0)))
            log(f"within {method} {s.key}: R² {rows[-1]['r2_within_session']:.3f} (α {rd.alpha:.3g})")
    pd.DataFrame(rows).to_csv(out / "check_within_session.csv", index=False)
    pd.DataFrame(vrows).to_csv(out / "check_session_variance.csv", index=False)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["gain", "within", "all"])
    ap.add_argument("--methods", default="")
    ap.add_argument("--sessions", default="")
    ap.add_argument("--out", default=str(res.WORK / "results" / "crosssession"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    ses = [k for k in a.sessions.split(",") if k]
    commit, problems = res.code_version()
    log(f"xs_checks {a.what}; codice {commit}" + (f" ({problems})" if problems else ""))
    (out / f"check_{a.what}_provenance.json").write_text(json.dumps(dict(commit=commit, problems=problems), indent=1))
    if a.what in ("gain", "all"):
        ms = [m for m in a.methods.split(",") if m] or METHODS + ["ridge_channels"]
        check_gain(ms, ses, out)
    if a.what in ("within", "all"):
        ms = [m for m in a.methods.split(",") if m] or METHODS
        check_within(ms, ses, out)


if __name__ == "__main__":
    main()
