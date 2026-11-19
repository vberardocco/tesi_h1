"""Step 10 A with 99 twins (supervisor comment of 19 Nov 2026, point 2).

Same pipeline as literature_step10.py, part A (single trial, FALCON bins), but with 99 random-phase twins
(seeds 0-98, the first 5 coincide with the original ones) instead of 5. With 5 surrogates the minimum p was 1/6.
p = (1 + #{twins >= data}) / (N + 1), at d = 3. Part B (TME, already with 100 surrogates) is untouched and nothing is written to the registry.
Outputs: ~/thesis_h1_work/results/literature/step10_single_99.csv, step10_single_99.md
"""
import time
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from thesis_h1 import io, results, literature, literature2

OUT = results.RESULTS / "literature"
N_GEM = 99
D_STAT = 3
T0 = time.time()


def log(msg):
    print(f"[{(time.time() - T0) / 60:5.1f} min] {msg}", flush=True)


def single_trial(counts, phase):
    Z = StandardScaler().fit_transform(io.smooth_rates(np.asarray(counts), sigma_ms=50).astype(float))
    mov = np.isin(phase, io.MOVEMENT_PHASES)
    bins_all = np.flatnonzero(mov)
    bins = bins_all[bins_all % literature2.STEP == 0]
    Z25 = PCA(n_components=literature2.K_PCA).fit(Z[bins_all]).transform(Z)
    ri, rp = literature2.curve_rv(Z25[bins])
    gem = [literature2.curve_rv(literature2.phases_random(Z25, g)[bins]) for g in range(N_GEM)]
    return 100 * (rp - ri), np.array([100 * (p - i) for i, p in gem]), len(bins)


rows, curve = [], []
for name in literature.SESSIONS:
    s = literature.load(name)
    adv, gem, n_pt = single_trial(s.counts, s.phase)
    g3, a3 = gem[:, D_STAT - 1], adv[D_STAT - 1]
    p = (1 + int((g3 >= a3).sum())) / (N_GEM + 1)
    rows.append(dict(session=name, n_points=n_pt, data_d3=a3, twins_mean_d3=g3.mean(), twins_min_d3=g3.min(),
                      twins_max_d3=g3.max(), twins_mean_first5_d3=g3[:5].mean(),
                      twins_below_data=int((g3 < a3).sum()), p=p))
    for d in range(1, gem.shape[1] + 1):
        curve.append(dict(session=name, d=d, data=adv[d - 1], twins_mean=gem[:, d - 1].mean(),
                          twins_p2_5=np.percentile(gem[:, d - 1], 2.5), twins_p97_5=np.percentile(gem[:, d - 1], 97.5),
                          twins_max=gem[:, d - 1].max()))
    log(f"{name}: data {a3:+.2f}, twins mean {g3.mean():+.2f} [{g3.min():+.2f}, {g3.max():+.2f}], "
        f"below data {int((g3 < a3).sum())}/{N_GEM}, p = {p:.3f}")

df = pd.DataFrame(rows)
df.to_csv(OUT / "step10_single_99.csv", index=False)
pd.DataFrame(curve).to_csv(OUT / "step10_single_99_curve.csv", index=False)
(OUT / "step10_single_99.md").write_text(df.round(3).to_markdown(index=False) if hasattr(df, "to_markdown") else df.round(3).to_string())
log("end")
