"""Figure 3.3 redone (17 Nov, v3: legend in the temporal order of the phases, read by rows): phases of the first two trials of 19250108T110520, one row per trial.
No computation: re-reads the rectangles from the pickle of the original figure (fig_en/run_01/01_task_phases.pkl)."""
import pickle, sys, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
src, out = sys.argv[1], sys.argv[2]
f = pickle.load(open(src, "rb")); ax0 = f.axes[0]
leg = ax0.get_legend() or f.legends[0]
names = [t.get_text() for t in leg.get_texts()]
cols = [tuple(h.get_facecolor()) for h in leg.legend_handles]
def lab(c):
    d = [sum((a - b) ** 2 for a, b in zip(c[:3], k[:3])) for k in cols]
    return names[d.index(min(d))]
bars = sorted((p.get_x(), p.get_width(), lab(p.get_facecolor()), p.get_facecolor()) for p in ax0.patches)
bound = ax0.get_lines()[0].get_xdata()[0]
trials = [[b for b in bars if b[0] < bound - 1e-9], [(x - bound, w, l, c) for x, w, l, c in bars if x >= bound - 1e-9]]
plt.rcParams.update({"font.size": 10, "font.family": "serif"})
fig, ax = plt.subplots(figsize=(6.3, 2.0))
for i, tr in enumerate(trials):
    y = 1 - i
    for x, w, l, c in tr:
        kw = dict(facecolor=c, edgecolor="none")
        if l == "Presentation": kw = dict(facecolor="0.85", edgecolor="none")
        if l == "Intertrial": kw = dict(facecolor="white", edgecolor="0.5", hatch="////", linewidth=0.5)
        ax.broken_barh([(x, w)], (y - 0.35, 0.7), **kw)
        if w < 0.1 and l not in ("Presentation", "Intertrial"):
            ax.plot(x + w / 2, y + 0.45, marker="v", color=c, markersize=6, clip_on=False)
ax.set_yticks([1, 0]); ax.set_yticklabels(["Trial 1", "Trial 2"])
ax.set_xlabel("Time from trial onset (s)")
ax.set_xlim(0, max(x + w for t in trials for x, w, _, _ in t)); ax.set_ylim(-0.6, 1.6)
for s in ("top", "right", "left"): ax.spines[s].set_visible(False)
ax.tick_params(axis="y", length=0)
ORDER = ["Presentation", "Reach", "Orient", "SnapTo", "Shape", "Grasp", "Carry", "Orient2", "Release", "Intertrial"]
cmap = dict(zip(names, cols))
assert set(ORDER) == set(names), names
seq = []
for n in ORDER:
    if n == "Presentation": seq.append(Patch(facecolor="0.85", label="Cue (Presentation)"))
    elif n == "Intertrial": seq.append(Patch(facecolor="white", edgecolor="0.5", hatch="////", label="Intertrial"))
    else: seq.append(Patch(facecolor=cmap[n], label=n))
NCOL = 5; nrow = -(-len(seq) // NCOL)
# matplotlib fills the legend by columns: reorder so that it reads by rows
h = [seq[r * NCOL + c] for c in range(NCOL) for r in range(nrow) if r * NCOL + c < len(seq)]
fig.legend(handles=h, loc="lower center", ncol=NCOL, frameon=False, fontsize=9, bbox_to_anchor=(0.5, -0.02), handlelength=1.2, columnspacing=1.0)
fig.subplots_adjust(left=0.12, right=0.99, top=0.97, bottom=0.45)
fig.savefig(out, dpi=300)
print("ok", [len(t) for t in trials], bound)
