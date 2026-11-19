"""Step 1-2 (literature comparison): table of comparisons with the literature and R1-R6 verdicts, from the registry only.

Reads ~/thesis_h1_work/results/h1_runs.csv (experiment "literature", summary rows with empty seed) and writes
~/thesis_h1_work/results/literature/step1_comparisons.csv and step1_comparisons.md. The verdicts follow
docs/literature/rules_decision.md (commit 7cde125) and are logged as numbers (condition "1_verdicts").
The null next to 1a and 1b is an extra column, not in the committed rules. 1e is post hoc.
Usage: PYTHONPATH=~/thesis_h1_git/src python scripts/literature_tables.py
"""
import numpy as np
import pandas as pd

from thesis_h1 import results, literature

OUT = results.RESULTS / "literature"
NAMES = list(literature.SESSIONS)
R = results.load_runs()
R = R[(R.experiment == "literature") & R.seed.isna()]


def val(cond, name, metric):
    d = R[(R.condition == cond) & (R.session == literature.KEYS[name]) & (R.metric == metric)]
    if len(d) != 1:
        raise KeyError(f"{cond} {name} {metric}: {len(d)} rows")
    return float(d.value.iloc[0])


def mmm(cond, name, m):
    return tuple(val(cond, name, f"{m}_{s}") for s in ("median", "min", "max"))


def fmt(t, f="{:.1f}"):
    return f"{f.format(t[0])} [{f.format(t[1])}-{f.format(t[2])}]"


def v_r1(x):
    return "in linea" if x >= 60 else "below"


def v_r2(x):
    return "in linea" if 31 <= x <= 38 else ("più alta" if x > 38 else "più bassa")


def v_r3(x, lo, hi):
    return "in linea" if lo <= x <= hi else ("più alta" if x > hi else "più bassa")


rows = []
# (rule, source, criterion, source value, condition, measure, has_null, verdict, % scale)
SPEC = [
    ("R1", "Gallego 2018", "variance of the first 12 PCs, 66 channels, square root, σ 50 ms, no z-score (%)",
     "≥ 60 (mean 73.4 ± 6.5)", "1a_gallego", "var12", True, v_r1),
    ("R2", "Silvernagel 2026", "components for 80%, 96 channels, σ 30 ms, no z-score", "31-38",
     "1b_silvernagel", "n80", True, v_r2),
    ("R3", "Oby 2019", "FA dimensions for 95% shared, 90 channels, 45 ms, K*", "5-11 (mean 8.4)",
     "1c_oby", "dim95_Kstar", False, lambda x: v_r3(x, 5, 11)),
    ("R3", "Degenhart 2020", "FA dimensions for 95% shared, 75 channels, 45 ms, K*", "4-8",
     "1c_degenhart", "dim95_Kstar", False, lambda x: v_r3(x, 4, 8)),
    ("R4", "Degenhart 2020", "population shared fraction, 75 channels, 45 ms, K*, no z-score", "32%",
     "1c_degenhart", "share_pop", False, None),
    # variants and post hoc analyses (no verdict)
    ("—", "variant 1a", "as R1 with z-score", "—", "1a_gallego_z", "var12", True, None),
    ("—", "variant 1a", "as R1 with all active channels", "—", "1a_gallego_all", "var12", True, None),
    ("—", "variant 1a", "as R1 with all active channels and z-score", "—", "1a_gallego_all_z", "var12", True, None),
    ("—", "variant 1b", "as R2 with z-score", "—", "1b_silvernagel_z", "n80", True, None),
    ("—", "variant 1b", "as R2 with all active channels", "—", "1b_silvernagel_all", "n80", True, None),
    ("—", "variant 1b", "as R2 with all active channels and z-score", "—", "1b_silvernagel_all_z", "n80", True,
     None),
    ("—", "variant 1c", "Oby with K = 10", "—", "1c_oby", "dim95_K10", False, None),
    ("—", "variant 1c", "Oby with K = 20", "—", "1c_oby", "dim95_K20", False, None),
    ("—", "variant 1c", "Oby with K*, no z-score", "—", "1c_oby", "dim95_Kstar_noz", False, None),
    ("—", "variant 1c", "Degenhart with K = 10", "—", "1c_degenhart", "dim95_K10", False, None),
    ("—", "variant 1c", "Degenhart with K = 20", "—", "1c_degenhart", "dim95_K20", False, None),
    ("—", "variant 1c", "Degenhart with K*, no z-score", "—", "1c_degenhart", "dim95_Kstar_noz", False, None),
    ("—", "variant 1c", "population shared fraction, 90 channels", "—", "1c_oby", "share_pop", False, None),
    ("—", "variant 1c", "per-channel shared fraction, median, 75 channels", "—", "1c_degenhart",
     "share_channels_median", False, None),
    ("1e*", "Gallego 2018 / Silvernagel 2026", "PCs for 60%, pipeline 1a, 66 channels",
     "m = 12 (Gallego); 16 and 13 (Silvernagel)", "1a_gallego", "n60", True, None),
    ("1e*", "Silvernagel 2026", "PCs for 60%, pipeline 1b, 96 channels", "16 and 13", "1b_silvernagel", "n60", True,
     None),
    ("1e*", "—", "PCs for 60%, pipeline 1a, all active channels", "—", "1a_gallego_all", "n60", True, None),
    ("1e*", "—", "PCs for 60%, pipeline 1b, all active channels", "—", "1b_silvernagel_all", "n60", True, None),
]

verdicts = {n: {} for n in NAMES}
for rule, source, criterion, vf, cond, m, null, vfun in SPEC:
    for name in NAMES:
        x = mmm(cond, name, m)
        r = dict(rule=rule, source=source, criterion=criterion, value_source=vf, session=name, condition=cond,
                 measure=m, median=x[0], min=x[1], max=x[2])
        if null:
            all = cond.endswith("all") or cond.endswith("all_z")
            if all:                                                     # a single value: null = 20 surrogates
                r.update(null_median=val(cond, name, f"{m}_null_median"),
                         null_min=val(cond, name, f"{m}_null_min_median"),
                         null_max=val(cond, name, f"{m}_null_max_median"))
            else:
                r.update(zip(("null_median", "null_min", "null_max"), mmm(cond, name, f"{m}_null")))
            r.update(zip(("diff_median", "diff_min", "diff_max"), mmm(cond, name, f"{m}_diff")))
        if cond.startswith("1c"):
            r["K_star"] = int(val(cond, name, "K_star_median"))
        if vfun:
            r["verdict"] = vfun(x[0])
            verdicts[name][f"{rule}_{source}"] = r["verdict"]
        if rule == "R1":
            r["in_73_4_pm_2sd"] = bool(60.4 <= x[0] <= 86.4)
        rows.append(r)
df = pd.DataFrame(rows)
df.to_csv(OUT / "step1_comparisons.csv", index=False)

for name in NAMES:                                                       # verdicts in the run registry, as numbers
    v = verdicts[name]
    cod = {"più bassa": -1.0, "below": -1.0, "in linea": 0.0, "più alta": 1.0}
    results.log_run({"R1_gallego": cod[v["R1_Gallego 2018"]], "R2_silvernagel": cod[v["R2_Silvernagel 2026"]],
                     "R3_oby": cod[v["R3_Oby 2019"]], "R3_degenhart": cod[v["R3_Degenhart 2020"]],
                     "R3_both_in_line": float(v["R3_Oby 2019"] == "in linea" and v["R3_Degenhart 2020"] == "in linea"),
                     "R1_in_73_4_pm_2sd": float(60.4 <= val("1a_gallego", name, "var12_median") <= 86.4)},
                    experiment="literature", method=None, latent_dim=None, decoder=None, session=literature.KEYS[name],
                    bins="falcon", split="session", condition="1_verdicts", fold=None, k_shot=None, H_ms=None,
                    seed=None)

# Markdown
pa = pd.read_csv(OUT / "step2_pa.csv")
sat = R[R.condition.str.startswith("1d_saturation_") & ~R.condition.str.contains("_N")]


def cell(r, k="", f="{:.1f}"):
    return fmt((r[f"{k}median"], r[f"{k}min"], r[f"{k}max"]), f)


md = ["# Step 1 · Like-for-like dimensionality comparisons", "",
      f"Commit {results.code_version()[0]}. FALCON bins (replication rep/19250108T110520 and sessions S2, S5), "
      "active channels ≥ 0.5 Hz, trial bins without Intertrial and without the bin with empty label. Values: median "
      "[min-max] over 20 draws of channels (seeds 0-19); with 'all channels' a single value. Null: independent "
      "circular shift per channel (criteria of Step 2), 20 surrogates per draw; the median "
      "over the draws of the median of the surrogates is reported, and the difference data − null computed draw by "
      "draw. The null is an extra column: it is not in the committed rules (7cde125) and does not change the verdicts.",
      "", "## Rules R1-R4", "",
      "| Rule | Source | Criterion | Source value | S1 | S2 | S5 | null S1 / S2 / S5 | data − null S1 / S2 / S5 "
      "| Verdict S1 / S2 / S5 |", "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
for (rule, source, criterion, vf, cond, m, null, vfun) in SPEC:
    if rule not in ("R1", "R2", "R3", "R4"):
        continue
    d = df[(df.condition == cond) & (df.measure == m)].set_index("session")
    f = "{:.1%}" if m.startswith("share") else "{:.1f}"
    nul = " / ".join(f.format(d.loc[n, "null_median"]) for n in NAMES) if null else "—"
    dif = " / ".join(fmt((d.loc[n, "diff_median"], d.loc[n, "diff_min"], d.loc[n, "diff_max"]), "{:+.1f}")
                     for n in NAMES) if null else "—"
    ver = " / ".join(d.loc[n, "verdict"] for n in NAMES) if vfun else "no verdict (R4)"
    extra = f" (K* = {', '.join(str(int(d.loc[n, 'K_star'])) for n in NAMES)})" if cond.startswith("1c") else ""
    md.append(f"| {rule} | {source} | {criterion}{extra} | {vf} | " +
              " | ".join(cell(d.loc[n], "", f) for n in NAMES) + f" | {nul} | {dif} | {ver} |")
d = df[(df.condition == "1a_gallego") & (df.measure == "var12")].set_index("session")
r3 = {n: verdicts[n]["R3_Oby 2019"] == "in linea" and verdicts[n]["R3_Degenhart 2020"] == "in linea" for n in NAMES}
md += ["", "R3 requires both conditions (Oby 5-11 at 90 channels and Degenhart 4-8 at 75 channels): " + ", ".join(
    f"{n} {'in line' if r3[n] else 'not in line'}" for n in NAMES) + ". K* is the maximum of the mean cross-validation "
    "curve over 5 draws (seeds 0-4); the K* of the single draws are in step1c_cv.csv."]
md += ["", "R1, band 73.4 ± 2 × 6.5% (60.4-86.4%): " + ", ".join(
    f"{n} {'within' if d.loc[n, 'in_73_4_pm_2sd'] else 'out'}" for n in NAMES) + ".", "",
    "## Variants (no verdict)", "",
    "| Source | Criterion | S1 | S2 | S5 | null S1 / S2 / S5 | data − null S1 / S2 / S5 |",
    "| --- | --- | --- | --- | --- | --- | --- |"]
for (rule, source, criterion, vf, cond, m, null, vfun) in SPEC:
    if rule != "—":
        continue
    d = df[(df.condition == cond) & (df.measure == m) & (df.criterion == criterion)].set_index("session")
    f = "{:.1%}" if m.startswith("share") else "{:.1f}"
    nul = " / ".join(f.format(d.loc[n, "null_median"]) for n in NAMES) if null else "—"
    dif = " / ".join(f"{d.loc[n, 'diff_median']:+.1f}" for n in NAMES) if null else "—"
    md.append(f"| {source} | {criterion} | " + " | ".join(cell(d.loc[n], "", f) for n in NAMES) + f" | {nul} | {dif} |")
md += ["", "## 1e · PCs for 60% (post hoc analysis, does not replace R1)", "",
       "Gallego 2018 chooses m = 12 because it guarantees \"at least 60%\" in all datasets, with a mean of "
       "73.4%: many of their datasets reach 60% with fewer than 12 PCs. Silvernagel 2026 (p. 7): 16 and 13 for "
       "monkeys O and U with the threshold at 60%.", "",
       "| Criterion | Source value | S1 | S2 | S5 | null S1 / S2 / S5 [min-max] | data − null S1 / S2 / S5 |",
       "| --- | --- | --- | --- | --- | --- | --- |"]
for (rule, source, criterion, vf, cond, m, null, vfun) in SPEC:
    if rule != "1e*":
        continue
    d = df[(df.condition == cond) & (df.measure == m)].set_index("session")
    nul = " / ".join(fmt((d.loc[n, "null_median"], d.loc[n, "null_min"], d.loc[n, "null_max"])) for n in NAMES)
    dif = " / ".join(fmt((d.loc[n, "diff_median"], d.loc[n, "diff_min"], d.loc[n, "diff_max"]), "{:+.1f}")
                     for n in NAMES)
    md.append(f"| {criterion} | {vf} | " + " | ".join(cell(d.loc[n]) for n in NAMES) + f" | {nul} | {dif} |")
md += ["", "With all channels the null is on 20 surrogates: median [min-max] of the surrogates.", "",
       "## R5 · Parallel Analysis (Step 2)", "",
       "| Session | Pipeline | N | k_PA (circular shift) | k_PA/N | k_PA (Altan shuffle) |",
       "| --- | --- | --- | --- | --- | --- |"]
for (n, p), g in pa.groupby(["session", "pipeline"], sort=False):
    sh, su = g[g.null == "shift"].iloc[0], g[g.null == "shuffle"].iloc[0]
    md.append(f"| {n} | {p} | {sh.N} | {sh.k_seq} | {sh.k_seq_su_N:.1%} | {su.k_seq} |")
r5 = (pa[pa.null == "shift"].k_seq >= 1).all()
md += ["", f"R5: {'sì' if r5 else 'no'} (minimal threshold k_PA ≥ 1 in all sessions and pipelines). Details in "
       "step2_pa.md.", "", "## R6 · Saturation (1d, S1 only, N = 20-160)", "",
       "| Measure | AIC linear | AIC power | AIC plateau | Verdict |", "| --- | --- | --- | --- | --- |"]
for cond in sorted(sat.condition.unique()):
    g = sat[sat.condition == cond].set_index("metric").value
    md.append(f"| {cond.replace('1d_saturation_', '')} | {g['aic_linear']:.2f} | {g['aic_power']:.2f} | "
              f"{g['aic_plateau']:.2f} | {'saturates' if g['R6_saturates'] else 'does not saturate in the observed range'} |")
md += ["", "Figure: step1_saturation.png."]
(OUT / "step1_comparisons.md").write_text("\n".join(md) + "\n")
print("\n".join(md))
