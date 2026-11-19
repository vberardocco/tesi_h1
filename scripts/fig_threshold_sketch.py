"""Figure 3.1 in English (17 Nov). Synthetic signal: Gaussian noise (SD 10 uV), 3 spikes of 90 uV (nearby neuron)
and 2 of 52 uV (distant neuron), sampled at 30 kHz for 60 ms. Thresholds at -3, -4.5 and -6 times the noise RMS.
The original figure code is not in the repo: this script rebuilds the figure with the same specifications as the
caption and searches for the first seed that gives 7, 5 and 4 events, as in the original figure (it prints it)."""
import sys, numpy as np, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
FS, T, SD = 30000, 0.060, 10.0
t = np.arange(int(FS * T)) / FS * 1e3
times = [(8, 90), (17, 52), (27, 90), (38, 52), (49, 90)]
def wave(amp):
    s = np.arange(-15, 45) / FS * 1e3
    return -amp * np.exp(-((s - 0.0) / 0.12) ** 2) + 0.35 * amp * np.exp(-((s - 0.45) / 0.25) ** 2)
def signal(seed):
    rng = np.random.default_rng(seed); v = rng.normal(0, SD, t.size)
    for ms, a in times:
        i = int(ms * FS / 1e3); w = wave(a); v[i - 15:i + 45] += w
    return v
def events(v, k):
    th = -k * SD; idx = np.where((v[1:] < th) & (v[:-1] >= th))[0] + 1
    out, last = [], -10**9
    for i in idx:
        if i - last >= 49: out.append(i); last = i
    return out
for seed in range(10000):
    v = signal(seed)
    if [len(events(v, k)) for k in (3, 4.5, 6)] == [7, 5, 4] and all(abs(t[e] - ms) < 1 for e, (ms, _) in zip(events(v, 4.5), times)):
        break
print("seed", seed)
plt.rcParams.update({"font.size": 10, "font.family": "serif"})
fig, ax = plt.subplots(figsize=(6.3, 2.8))
ax.plot(t, v, color="0.35", lw=0.6, label="Voltage (synthetic)")
for k, ls, c in [(3, ":", "0.5"), (4.5, "--", "C3"), (6, "-.", "0.5")]:
    ax.axhline(-k * SD, ls=ls, color=c, lw=1, label=f"$-{k:g}\\times$ RMS: {len(events(v, k))} events")
e = events(v, 4.5); ax.plot(t[e], np.full(len(e), -4.5 * SD), "v", color="C3", ms=6)
ax.set_xlabel("Time (ms)"); ax.set_ylabel("Voltage (µV)")
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.legend(loc="lower center", ncol=2, frameon=False, fontsize=9, bbox_to_anchor=(0.55, -0.01))
fig.subplots_adjust(left=0.14, right=0.99, top=0.97, bottom=0.42)
fig.savefig(sys.argv[1], dpi=300)
