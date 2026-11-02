"""Tabella B: ridge sulle latenti FA su minival, una volta sola (decisione del 2/11/2026; docs/crosssession_protocol.md,
sezione "Tabella B").

Il modello è models/fa15_ridge_h50.npz, cioè il modello FA congelato del cross-sessione (FA_d15_heldin_s0) esportato
da src/tesi_h1/fa_decoder.py: nessun riallenamento, nessuna scelta. Prima di valutare si controlla che l'npz coincida
con un'esportazione nuova del modello congelato.
Tre calcoli dello stesso modello, come per ridge_h50 (scripts/05_ridge_minival_h50.py):
  1. in blocco: predict su ogni file intero + il nostro evaluator (decoding.evaluate);
  2. in streaming con l'evaluator ufficiale di FALCON, batch 4 (come nell'immagine Docker): è il numero riportato;
  3. lo stesso con batch 1.
Poi: registro (esperimento "tabella_b": FA su minival, leave-one-day-out della FA sulle held-in, e ridge_h50 su
minival importata dal log del 30/10, commit b797b3e) e tabella B in results/table_B.{csv,md,tex}.
Uscita: run05/minival_fa15.log.
"""
import hashlib
import os
import pickle
import re
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from tesi_h1 import crosssession as xs  # noqa: E402
from tesi_h1 import decoding as dec  # noqa: E402
from tesi_h1 import fa_decoder as fd  # noqa: E402
from tesi_h1 import results as res  # noqa: E402

MODEL = REPO / "models/fa15_ridge_h50.npz"
RAW = Path.home() / "000954"
LOG = REPO / "run05/minival_fa15.log"
LOG_H50 = REPO / "run05/minival_h50.log"
TMP = Path(tempfile.mkdtemp())
KEYS = dict(esperimento="tabella_b", decoder="ridge", session=None, bins="falcon", fold=None, k_shot=None, H_ms=1000,
            seed=0)
BATCHES = [4, 1]

LOG.parent.mkdir(exist_ok=True)
log_f = open(LOG, "a")


def out(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    log_f.write(s + "\n")
    log_f.flush()


def check_model():
    """L'npz deve coincidere con un'esportazione nuova del modello FA congelato."""
    data = xs.load_data()
    with open(xs.bundle_path("FA", 0, "heldin", 15), "rb") as f:
        b = pickle.load(f)
    fresh = fd.export(b, data["active"], TMP / "fresh.npz")
    a, c = np.load(MODEL), np.load(fresh)
    assert sorted(a.files) == sorted(c.files) and all(np.array_equal(a[k], c[k]) for k in a.files), \
        "models/fa15_ridge_h50.npz non coincide con il modello FA congelato"
    return b


def main():
    t0 = time.time()
    b = check_model()
    m = fd.FALatentRidge(MODEL)
    out("=" * 100)
    out(f"Valutazione minival della ridge sulle latenti FA   {datetime.now():%Y-%m-%d %H:%M}")
    out(f"modello        : {MODEL.relative_to(REPO)}  md5 {hashlib.md5(MODEL.read_bytes()).hexdigest()} "
        f"(= FA_d15_heldin_s0, controllato)")
    out(f"                 FA K = {m.P.shape[1]} su {len(m.active)} canali attivi, storia H = {m.H} bin = {m.H * 20} ms, "
        f"α = {m.alpha:g} (leave-one-day-out sulle held-in, R² {m.lodo_r2:.4f}), {b.info['n_bins']} bin di "
        f"{len(b.info['sessions'])} sessioni held-in calib")
    out("nota           : readout allenato con la storia che riparte a ogni trial; qui (evaluator H1 continuo) la "
        "storia scorre per tutto il file")
    from importlib.metadata import version

    from falcon_challenge.config import FalconConfig, FalconTask
    from falcon_challenge.evaluator import FalconEvaluator
    out(f"falcon_challenge: {version('falcon_challenge')}")
    out("-" * 100)

    mv = dec.load_split(RAW, "held-in-minival")
    block = {f.key: m.predict(f.counts) for f in mv}
    r2_b, day_b, dof_b = dec.evaluate([f.day for f in mv], [f.vel for f in mv], [block[f.key] for f in mv],
                                      [f.mask for f in mv])
    out(f"[blocco, nostro evaluator] {len(mv)} file")
    out(f"  R² medio sui giorni: {r2_b:.4f}   ({r2_b:.6f})")
    out("  per giorno: " + ", ".join(f"{k} {v:.4f}" for k, v in day_b.items()))
    out("  per DoF (media giorni): " + ", ".join(f"{d} {v:.4f}" for d, v in zip(dec.DOF, dof_b.mean(0))))

    os.environ["EVAL_DATA_PATH"] = str(Path.home())
    official = {}
    for B in BATCHES:
        os.environ["PREDICTION_PATH_LOCAL"] = str(TMP / f"pred_b{B}.pkl")
        os.environ["GT_PATH"] = str(TMP / f"gt_b{B}.pkl")
        ts = time.time()
        ev = FalconEvaluator(eval_remote=False, split="h1", dataloader_workers=0)
        r = ev.evaluate(fd.make_streaming_decoder(MODEL, batch_size=B), phase="minival")
        preds = pickle.load(open(TMP / f"pred_b{B}.pkl", "rb"))["h1"]
        stream = {f.key: np.asarray(preds[f.key])[: len(f.counts)] for f in mv}
        r2_s, _, dof_s = dec.evaluate([f.day for f in mv], [f.vel for f in mv], [stream[f.key] for f in mv],
                                      [f.mask for f in mv])
        dmax = max(np.abs(stream[k] - block[k]).max() for k in block)
        off = r["result"][0]["minival_split_h1"]
        official[B] = off
        out(f"[streaming, evaluator ufficiale FALCON, batch {B}] {time.time() - ts:.0f} s")
        out(f"  risultato ufficiale: {off}")
        out(f"  R² medio ufficiale : {off['Held In R2 Mean']:.4f}   ({off['Held In R2 Mean']:.6f})")
        out(f"  nostro evaluator sulle predizioni in streaming: {r2_s:.4f}")
        ok = round(off["Held In R2 Mean"], 4) == round(r2_b, 4) and dmax < 1e-4
        out(f"  streaming - blocco, differenza massima delle predizioni {dmax:.2e}; coincidono: {'SÌ' if ok else 'NO'}")
        assert ok, "streaming e blocco non coincidono"

    # registro
    off = official[4]
    metrics = {"r2": off["Held In R2 Mean"], "r2_std_days": off["Held In R2 Std."], "r2_block": r2_b,
               "n_files": len(mv)}
    metrics.update({f"r2_{k}": v for k, v in day_b.items()})
    metrics.update({f"r2_{d}": v for d, v in zip(dec.DOF, dof_b.mean(0))})
    res.log_run(metrics, method="FA", latent_dim=15, split="minival", condition="calib_heldin", **KEYS)
    res.log_run({"r2": m.lodo_r2, "alpha": m.alpha}, method="FA", latent_dim=15, split="calib_lodo",
                condition="calib_heldin", **KEYS)
    import_h50()
    table_b()
    out(f"tempo totale {time.time() - t0:.0f} s")


def import_h50():
    """ridge_h50 su minival dal log del 30/10 (commit b797b3e): nessuna nuova valutazione."""
    df = res.load_runs(keep_last=False)
    if ((df.esperimento == "tabella_b") & (df.method == "counts")).any():
        out("ridge_h50: già nel registro")
        return
    txt = LOG_H50.read_text()
    sec = txt.split("Valutazione minival di ridge_h30.npz")[0]
    off = eval(re.search(r"risultato ufficiale: (\{.*\})", sec).group(1), {"np": np})
    day = dict(re.findall(r"(S\d) (\d\.\d{4})", re.search(r"per giorno: (.*)", sec).group(1)))
    dof = dict(re.findall(r"(\w+) (\d\.\d{4})", re.search(r"per DoF \(media giorni\): (.*)", sec).group(1)))
    metrics = {"r2": float(off["Held In R2 Mean"]), "r2_std_days": float(off["Held In R2 Std."])}
    metrics.update({f"r2_{k}": float(v) for k, v in day.items()})
    metrics.update({f"r2_{k}": float(v) for k, v in dof.items()})
    res.log_run(metrics, method="counts", latent_dim=None, split="minival", condition="calib_heldin",
                _auto=dict(timestamp="2026-10-30T13:05:00", commit="b797b3e",
                           script="~/tesi_h1_git/scripts/05_ridge_minival_h50.py"), **KEYS)
    out(f"ridge_h50 importata dal log del 30/10: R² {metrics['r2']:.4f}")


def table_b():
    import pandas as pd
    df = res.load_runs()
    d = df[df.esperimento == "tabella_b"]

    def get(method, split, metrica):
        v = d[(d.method == method) & (d.split == split) & (d.metrica == metrica)].valore
        return float(v.iloc[0]) if len(v) else np.nan

    days = [f"S{i}" for i in range(6)]
    rows = []
    for method, name, inp in (("counts", "Ridge on channels (ridge_h50)", "176 channels"),
                              ("FA", "Ridge on FA latents", "FA, K = 15, 170 active channels")):
        r = dict(decoder=name, input=inp, history_ms=1000, minival_r2=get(method, "minival", "r2"),
                 minival_r2_sd_days=get(method, "minival", "r2_std_days"),
                 heldin_lodo_r2=get(method, "calib_lodo", "r2"))
        r.update({f"minival_{k}": get(method, "minival", f"r2_{k}") for k in days + list(dec.DOF)})
        rows.append(r)
    T = pd.DataFrame(rows)
    out_dir = res.RESULTS
    T.to_csv(out_dir / "table_B.csv", index=False)
    f3 = lambda x: "—" if not np.isfinite(x) else f"{x:.3f}"  # noqa: E731
    md = ["# Table B — causal decoders, calib held-in → minival (FALCON evaluator, streaming)", "",
          "| decoder | input | minival R² (mean of 6 days) | SD over days | held-in calib, leave-one-day-out R² |",
          "|---|---|---|---|---|"]
    md += [f"| {r.decoder} | {r.input}, 1000 ms history | {f3(r.minival_r2)} | {f3(r.minival_r2_sd_days)} | "
           f"{f3(r.heldin_lodo_r2)} |" for r in T.itertuples()]
    md += ["", "Per day and per DoF (minival):", "", "| decoder | " + " | ".join(days + list(dec.DOF)) + " |",
           "|---|" + "---|" * (len(days) + len(dec.DOF))]
    md += [f"| {r['decoder']} | " + " | ".join(f3(r[f'minival_{k}']) for k in days + list(dec.DOF)) + " |"
           for _, r in T.iterrows()]
    md += ["", "- Both models are fit once on all 13 held-in calib files and evaluated once on minival "
           "(ridge_h50: 30/10/2026, commit b797b3e; FA: run of this script).",
           "- Ridge on channels: history and α chosen on an 80/20 split of the calib (0.392 on the last 20%), no "
           "leave-one-day-out value.",
           "- FA latents: the frozen FA model of the cross-session protocol (FA_d15_heldin_s0); α = 10⁶ chosen "
           "leave-one-day-out on the held-in days. Its readout was trained with the history restarting at each trial; "
           "the H1 evaluator is continual, so at evaluation the history runs through the file (as for ridge_h50)."]
    (out_dir / "table_B.md").write_text("\n".join(md) + "\n")
    tex = ["\\begin{tabular}{llrrr}", "\\toprule",
           "Decoder & Input & Minival $R^2$ & SD (days) & LODO $R^2$ \\\\", "\\midrule"]
    tex += [f"{r.decoder} & {r.input} & {f3(r.minival_r2)} & {f3(r.minival_r2_sd_days)} & "
            f"{f3(r.heldin_lodo_r2)} \\\\".replace("—", "--") for r in T.itertuples()]
    tex += ["\\bottomrule", "\\end{tabular}"]
    (out_dir / "table_B.tex").write_text("\n".join(tex) + "\n")
    out(f"tabella B: {out_dir / 'table_B.md'}")
    for r in T.itertuples():
        out(f"  {r.decoder}: minival {f3(r.minival_r2)} (SD giorni {f3(r.minival_r2_sd_days)}), "
            f"LODO held-in {f3(r.heldin_lodo_r2)}")


if __name__ == "__main__":
    main()
