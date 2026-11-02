"""Tabella dei tempi per la sezione sul costo computazionale: solo tempi misurati, letti dai file che li hanno registrati.

Fonti:
- results/crosssession/models/<metodo>_d<d>_heldin_s<seme>.json: stima di ogni metodo sulle 13 sessioni held-in
  (stessi dati per tutti): fit_seconds = solo il metodo; total_seconds = metodo + latenti + readout (α leave-one-day-out);
- results/crosssession/logs/coda.log: inizio e fine di ogni lavoro della coda del cross-sessione (31/10/2026), da cui
  la durata e quanti lavori giravano insieme (i tempi sono "a orologio", con la macchina condivisa);
- run05/tune_long.log (ridge sui canali, scelta della storia), results/crosssession/stability/meta.json,
  run05/minival_h50.log, run05/minival_fa15.log.
I tempi scritti nei notebook ("circa 3 minuti") sono stime, non misure: non entrano.

Uscita: results/table_timing.md, results/table_timing_fits.csv, results/table_timing_jobs.csv.
Uso: python scripts/make_timing_table.py
"""
import json
import platform
import re
import subprocess
import sys
from datetime import datetime
from importlib.metadata import version
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[1]
XS = REPO / "results/crosssession"
OUT = REPO / "results"
MAIN = {"PCA": 25, "FA": 15, "GPFA": 25, "Isomap": 25, "UMAP": 25, "CEBRA": 25, "LFADS": 25}
DEVICE = {"PCA": "CPU", "FA": "CPU", "GPFA": "CPU", "Isomap": "CPU", "UMAP": "CPU", "CEBRA": "GPU (MPS), 1 CPU thread",
          "LFADS": "CPU, 1 thread"}
# il lavoro della coda in cui il modello held-in è stato stimato (gli altri lo rileggono dalla cache)
FIT_JOB = {"PCA": "xs_PCA", "FA": "xs_FA", "GPFA": "xs_GPFA", "Isomap": "v2_Isomap25", "UMAP": "v2_UMAP25_s{s}",
           "CEBRA": "v2_CEBRA25_s{s}", "LFADS": "xs_LFADS_s{s}"}
# lavori che hanno prodotto i risultati riportati (prima tornata per PCA, FA, GPFA, LFADS; seconda per le mappe a d = 25)
RESULT_JOBS = {"rates (ridge_rates)": ["xs_rates"], "ridge_channels": ["v2_ridge_channels"], "PCA": ["xs_PCA"],
               "FA": ["xs_FA"], "GPFA": ["xs_GPFA"], "Isomap": ["v2_Isomap25"], "UMAP": [f"v2_UMAP25_s{s}" for s in range(3)],
               "CEBRA": ["v2_CEBRA25_tune"] + [f"v2_CEBRA25_s{s}" for s in range(3)],
               "LFADS": ["xs_LFADS_tune"] + [f"xs_LFADS_s{s}" for s in range(3)]}


def queue_jobs():
    rows, start = [], {}
    for line in (XS / "logs/coda.log").read_text().splitlines():
        m = re.match(r"(\S+ \S+) \[(\w+)\] (INIZIO|FINE)\s+(\S+)", line)
        if not m:
            continue
        t, lane, what, job = datetime.fromisoformat(m.group(1)), m.group(2), m.group(3), m.group(4)
        if what == "INIZIO":
            start[job] = (t, lane)
        elif job in start:
            t0, lane0 = start.pop(job)
            rows.append(dict(job=job, lane=lane0, start=t0, end=t, minutes=(t - t0).total_seconds() / 60))
    J = pd.DataFrame(rows)
    J = J[~J.job.str.startswith("commit_")].reset_index(drop=True)
    J["concurrent_jobs"] = [max_simultaneous(J, r.start, r.end) for r in J.itertuples()]
    return J


def max_simultaneous(J, t0, t1):
    """Il massimo numero di lavori attivi nello stesso istante tra t0 e t1 (compreso quello che si misura)."""
    times = [t0] + [t for t in J.start if t0 < t < t1]
    return int(max(((J.start <= t) & (J.end > t)).sum() for t in times))


def model_fits(J):
    rows = []
    for m, d in MAIN.items():
        for p in sorted((XS / "models").glob(f"{m}_d{d}_heldin_s*.json")):
            info = json.loads(p.read_text())
            s = int(p.stem.rsplit("_s", 1)[1])
            job = FIT_JOB[m].format(s=s)
            conc = J.loc[J.job == job, "concurrent_jobs"]
            rows.append(dict(method=m, latent_dim=d, seed=s, n_bins=info["n_bins"], fit_s=info["fit_seconds"],
                             total_s=info["total_seconds"], device=DEVICE[m], fit_job=job,
                             concurrent_jobs=int(conc.iloc[0]) if len(conc) else None,
                             best_epoch=info.get("best_epoch")))
    return pd.DataFrame(rows)


def hardware():
    def sc(k):
        try:
            return subprocess.run(["sysctl", "-n", k], capture_output=True, text=True).stdout.strip()
        except OSError:
            return "?"
    pk = {p: version(p) for p in ("numpy", "scipy", "scikit-learn", "torch", "cebra", "elephant", "umap-learn",
                                  "falcon-challenge")}
    return (f"{sc('machdep.cpu.brand_string')}, {sc('hw.perflevel0.physicalcpu')} performance + "
            f"{sc('hw.perflevel1.physicalcpu')} efficiency cores, {int(sc('hw.memsize')) / 2**30:.0f} GB RAM, "
            f"macOS {platform.mac_ver()[0]}; Python {platform.python_version()}; "
            + ", ".join(f"{k} {v}" for k, v in pk.items()))


def other_runs():
    rows = []
    t = (REPO / "run05/tune_long.log").read_text()
    acc = re.search(r"accumulato X'X \((\d+) colonne\) in (\d+)s", t)
    rows.append(("ridge on channels: X'X over the 13 held-in calib files (8800 columns)", float(acc.group(2)) / 60,
                 "run05/tune_long.log"))
    prev = float(acc.group(2))                     # i secondi del log sono cumulativi dall'inizio dello script
    for h, ms, sec in re.findall(r"H=(\d+) \(\s*(\d+) ms\).*?(\d+)s$", t, flags=re.M):
        rows.append((f"ridge on channels: solve and 80/20 validation, {ms} ms history, 13 α values",
                     (float(sec) - prev) / 60, "run05/tune_long.log"))
        prev = float(sec)
    meta = json.loads((XS / "stability/meta.json").read_text())
    rows.append((f"FA subspace stability: {meta['n_blocks']} block FAs, 6 leave-one-day-out FAs, {meta['n_pairs']} pairs × "
                 f"{meta['n_draws']} chance draws", meta["minutes"], "results/crosssession/stability/meta.json"))
    for name, f in (("minival evaluation, ridge on channels (block + streaming batch 4 and 1)", "run05/minival_h50.log"),
                    ("minival evaluation, ridge on FA latents (block + streaming batch 4 and 1)", "run05/minival_fa15.log")):
        sec = re.findall(r"tempo totale (\d+) s", (REPO / f).read_text())
        rows.append((name, float(sec[0]) / 60, f))
    return pd.DataFrame(rows, columns=["run", "minutes", "source"])


def main():
    J = queue_jobs()
    F = model_fits(J)
    O = other_runs()
    J.to_csv(OUT / "table_timing_jobs.csv", index=False)
    F.to_csv(OUT / "table_timing_fits.csv", index=False)
    g = F.groupby("method", sort=False)
    L = ["# Computational cost (measured times)", "", f"Hardware and software: {hardware()}.", "",
         "## Fitting each method on the 13 held-in sessions (135,991 bins, same data for all)", "",
         "Wall-clock seconds from the cross-session runs of 31/10/2026, as stored with each model. Concurrent jobs = queue "
         "jobs running at the same time, at most, while the model was fitted (the machine was shared).", "",
         "| method | latent dim. | seeds | device | fit only (s), median [min–max] | fit + latents + readout (s) | "
         "concurrent jobs |", "|---|---|---|---|---|---|---|"]
    for m, d in g:
        rng = f" [{d.fit_s.min():.0f}–{d.fit_s.max():.0f}]" if len(d) > 1 else ""
        L.append(f"| {m} | {d.latent_dim.iloc[0]} | {len(d)} | {d.device.iloc[0]} | {d.fit_s.median():.0f}{rng} | "
                 f"{d.total_s.median():.0f} | {d.concurrent_jobs.min()}–{d.concurrent_jobs.max()} |")
    L += ["", "## Whole cross-session jobs (λ tuning on pseudo-sessions, held-in fit, 14 held-out sessions × 3 folds)", "",
          "Wall-clock minutes of the queue jobs whose results are reported (first round for PCA, FA, GPFA, LFADS and "
          "ridge_rates; second round for the maps at d = 25 and ridge_channels).", "",
          "| method | jobs | total minutes | concurrent jobs (at most) |", "|---|---|---|---|"]
    for m, jobs in RESULT_JOBS.items():
        d = J[J.job.isin(jobs)]
        assert len(d) == len(jobs), (m, set(jobs) - set(d.job))
        L.append(f"| {m} | {', '.join(d.job)} | {d.minutes.sum():.1f} | {d.concurrent_jobs.min()}–{d.concurrent_jobs.max()} |")
    L += ["", "## Other runs", "", "| run | minutes | source |", "|---|---|---|"]
    L += [f"| {r.run} | {r.minutes:.1f} | `{r.source}` |" for r in O.itertuples()]
    L += ["", "Files: `table_timing_fits.csv` (one row per model and seed), `table_timing_jobs.csv` (every queue job). "
          "Generated by `scripts/make_timing_table.py`."]
    (OUT / "table_timing.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))


if __name__ == "__main__":
    main()
