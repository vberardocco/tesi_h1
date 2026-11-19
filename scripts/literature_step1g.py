"""Step 1g (literature comparison, exploratory, added on 3 Nov after seeing the results): Cheater and Frankenstein.

It does not change any R1-R7 verdict and is not in the committed rules. Pipeline 1a (square root, Gaussian
σ = 50 ms) on S1, S2 and S5, without Intertrial and without the bin with empty label, all trial bins; circular-shift
null with 20 surrogates per draw and the median trial duration of each session.

- Cheater: channels chosen by variance alone (A: 12 most variable + 54 least variable; B: 66 contiguous in
  variance order with the smallest max/min ratio). Share of the variance in the 12 most variable channels of the group,
  R1 on data and null; r² between that share and R1 over 200 random draws of 66 channels. A and B are not
  H1 values: they only show how much R1 depends on differences in variance.
- Frankenstein (20 draws, seeds 0-19): phase segments within each trial (the first one is discarded if it is Release),
  key = phase (Presentation with the following phase) + occurrence in the trial. True data, phases from different trials
  (per channel and segment, the same key from another trial) and Frankenstein (66 distinct indices among the channels
  active in all three sessions, 22 per session, on the trial structure of S1). R1 with and without z-score;
  share = (phases from different trials − null) / (data − null), nonlinear. Data, shuffled and null use the bins of the
  segments (the trial bins without the initial Release of each trial).
- Check: per channel cov(mean of even trials, mean of odd trials) / single-trial variance, median over
  channels, with phase-by-phase alignment and from the trial start only.

Outputs: ~/thesis_h1_work/results/literature/step1g.csv and step1g.md; rows in the run registry.
Usage: OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONPATH=~/thesis_h1_git/src python scripts/literature_step1g.py
"""
import time

import numpy as np
import pandas as pd

from thesis_h1 import results, literature

OUT = results.RESULTS / "literature"
OUT.mkdir(parents=True, exist_ok=True)
SEMI, N_SURR = range(20), 20
T0 = time.time()
KEYS = dict(experiment="literature", method="PCA", latent_dim=None, decoder=None, bins="falcon", split="session",
            fold=None, k_shot=None, H_ms=None)


def log(msg):
    print(f"[{(time.time() - T0) / 60:4.1f} min] {msg}", flush=True)


sess, keys, tricks, checks = {}, {}, {}, {}
for name in literature.SESSIONS:
    s = literature.load(name)
    d = literature.Data1g(Y=literature.pipeline_gallego(s.counts), phase=s.phase, trial=s.trial, active=s.active,
                      mask=s.mask)
    sess[name], keys[name] = d, s.key
    t = literature.cheater(d.Y, d.mask, d.duration, n_surr=N_SURR, n_random=200)
    tricks[name] = t
    results.log_run(t, condition="1g_cheater", session=s.key, seed=None, **KEYS)
    c = {"check_phase": float(np.median(literature.check_even_odd(d.Y, d.segs, "phase"))),
         "check_start": float(np.median(literature.check_even_odd(d.Y, d.segs, "start"))),
         "n_segments": len(d.segs), "n_bin_segments": len(d.rows), "duration_trial_s": d.duration * literature.BIN_S}
    checks[name] = c
    results.log_run(c, condition="1g_check", session=s.key, seed=None, **KEYS)
    log(f"{name}: {len(d.segs)} segments, {len(d.rows)} bins; Cheater A {t['A_data']:.1f}/{t['A_null']:.1f}, "
        f"B {t['B_data']:.1f}/{t['B_null']:.1f}, r² {t['r2']:.2f}; check phase {c['check_phase']:.3f}, "
        f"start {c['check_start']:.3f}")

recs = []
for seed in SEMI:
    recs.append(literature.frankenstein_1g(sess, seed, n_channels=66, n_surr=N_SURR, structure="S1"))
    if seed in (0, 9, 19):
        log(f"Frankenstein, draw {seed} done")
df = pd.DataFrame(recs)
for z in ("", "_z"):                                                   # differences per draw
    for name in sess:
        df[f"{name}_data_minus_null{z}"] = df[f"{name}_data{z}"] - df[f"{name}_null{z}"]
        df[f"{name}_shuffled_minus_null{z}"] = df[f"{name}_shuffled{z}"] - df[f"{name}_null{z}"]
    df[f"frank_minus_null{z}"] = df[f"frank{z}"] - df[f"frank_null{z}"]
df.to_csv(OUT / "step1g.csv", index=False)
for name in sess:
    cols = [c for c in df.columns if c.startswith(name + "_")]
    literature.log(df[["seed"] + cols].rename(columns=lambda c: c.replace(name + "_", "")).to_dict("records"),
                    [c.replace(name + "_", "") for c in cols], condition="1g_frankenstein_session",
                    session=keys[name], **KEYS)
fcols = [c for c in df.columns if c.startswith("frank")]
literature.log(df[["seed"] + fcols].to_dict("records"), fcols, condition="1g_frankenstein", session=None, **KEYS)


def mmm(v, fmt="{:.1f}"):
    med, lo, hi = literature.summary(v)
    return f"{fmt.format(med)} [{fmt.format(lo)}-{fmt.format(hi)}]"


md = ["# Step 1g · Cheater and Frankenstein (exploratory, post hoc)", "",
      f"Commit {results.code_version()[0]}. Added on 3 Nov after seeing the results: it does not change any "
      "R1-R7 verdict and is not in the committed rules. Pipeline 1a; circular-shift null, 20 surrogates per "
      "draw; values in % of the variance (R1 = first 12 PCs).", "",
      "## Cheater (channels chosen by variance alone; these are not H1 values)", "",
      "| Session | Group | share of the 12 most variable (%) | max/min variance | R1 data | R1 null | data − null |",
      "| --- | --- | --- | --- | --- | --- | --- |"]
for name, t in tricks.items():
    for g in ("A", "B"):
        md.append(f"| {name} | {g} | {t[g + '_share12']:.1f} | {t[g + '_ratio_var']:.1f} | {t[g + '_data']:.1f} | "
                  f"{t[g + '_null']:.1f} | {t[g + '_data'] - t[g + '_null']:+.1f} |")
md += ["", "Random draws of 66 channels (200): r² between the share of the 12 most variable and R1 on the data: " +
       "; ".join(f"{n} {t['r2']:.2f} (medians: share {t['random_share12_median']:.1f}%, R1 "
                 f"{t['random_data_median']:.1f}%)" for n, t in tricks.items()), "",
       "## Frankenstein (66 channels, 20 draws; median [min-max])", ""]
for z, title in (("", "without z-score"), ("_z", "with z-score")):
    md += [f"### R1 {title}", "",
           "| Session | data | phases from different trials | null | data − null | shuffled − null | share |",
           "| --- | --- | --- | --- | --- | --- | --- |"]
    for name in sess:
        md.append(f"| {name} | {mmm(df[f'{name}_data{z}'])} | {mmm(df[f'{name}_shuffled{z}'])} | "
                  f"{mmm(df[f'{name}_null{z}'])} | {mmm(df[f'{name}_data_minus_null{z}'], '{:+.1f}')} | "
                  f"{mmm(df[f'{name}_shuffled_minus_null{z}'], '{:+.1f}')} | "
                  f"{mmm(100 * df[f'{name}_share{z}'], '{:.0f}%')} |")
    md += ["", f"Frankenstein (22 channels from each session, S1 structure): {mmm(df[f'frank{z}'])}; its null "
           f"{mmm(df[f'frank_null{z}'])}; difference {mmm(df[f'frank_minus_null{z}'], '{:+.1f}')}.", ""]
cases = {name: {k: int(df[f"{name}_cases_{k}"].sum()) for k in ("key", "phase", "same")} for name in sess}
fc = {k: int(df[f"frank_cases_{k}"].sum()) for k in ("key", "phase", "mean")}
md += ["Cases (channel × segment, sum over the 20 draws). Phases from different trials: " +
       "; ".join(f"{n} same key {c['key']}, same phase {c['phase']}, same trial {c['same']}"
                 for n, c in cases.items()) +
       f". Frankenstein: same key {fc['key']}, same phase {fc['phase']}, channel mean {fc['mean']}.", "",
       "The share is nonlinear: in testing, a scenario halfway between task and internal structure gives about 30%. "
       "The dimensions above the null are not counted in the shuffled versions (channels that draw the same trial "
       "retain a small correlation). Shuffling the trials loses the part that requires simultaneous recording, "
       "but also the signal specific to each condition; slow common drift and noise common "
       "across channels remain possible.", "",
       "## Check: cov(even mean, odd mean) / single-trial variance (median over channels)", "",
       "| Session | phase by phase | from the trial start only |", "| --- | --- | --- |"]
for name, c in checks.items():
    md.append(f"| {name} | {100 * c['check_phase']:.1f}% | {100 * c['check_start']:.1f}% |")
md += ["", "Alignment from the trial start only is not used for the Frankenstein."]
(OUT / "step1g.md").write_text("\n".join(md) + "\n")
log(f"saved {OUT / 'step1g.csv'} and step1g.md")
