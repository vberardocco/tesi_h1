# %% [markdown]
# # 01 · Dataset exploration — FALCON H1 (human iBCI, 7 DoF)
#
# **Dataset:** FALCON Benchmark **H1** — *Human 7DoF Reach and Grasp Motor BCI*
# (DANDI:000954, Ye, Collinger & Gaunt, 2024).
#
# **Contents.** Neural activity recorded from **two 96-channel intracortical arrays**
# implanted in the hand area (*hand knob*) of the motor cortex of **one human participant**
# with a brain-computer interface (clinical trial NCT01894802, Univ. of Pittsburgh).
# The data come from **open-loop calibration** sessions: an audiovisual cue indicates a
# *reach-and-grasp* movement to the participant, who **attempts** it while a virtual arm
# shows the reference kinematics. There is no closed-loop control: the kinematics are those
# *prescribed* by the cue and serve as the reference behavioral variable.
#
# **The decoding problem.** From the neural activity (176 multi-unit channels) we must reconstruct the
# **7-degree-of-freedom kinematics**:
#
# | group | dimensions | meaning |
# |---|---|---|
# | translation | `tx`, `ty`, `tz` | 3D end-effector position |
# | orientation | `rx` | wrist rotation (roll) |
# | grasp | `g1`, `g2`, `g3` | flexion-extension of thumb, index, ring/little finger + thumb abduction |
#
# The official benchmark metric is the **R² on the velocity** of the 7 DoF, so the reference
# target is `OpenLoopKinematicsVelocity`.
#
# ### What each NWB file provides (checked on the 40 files of the dandiset)
#
# | NWB object | Meaning | Shape / step |
# |---|---|---|
# | `units` | **multi-unit spiking activity**: spike times per unit | 176 units, times in s |
# | `acquisition['OpenLoopKinematics']` | 7-DoF position | (T, 7) @ 20 ms bins |
# | `acquisition['OpenLoopKinematicsVelocity']` | **7-DoF velocity = official target** | (T, 7) @ 20 ms bins |
# | `acquisition['TrialNum']` | trial index of each bin | (T,) |
# | `acquisition['eval_mask']` | bins to include in the evaluation | (T,) bool |
# | `intervals['epochs']` | **task phases** (Reach, Grasp, Carry, …) | start/stop/tag table |
#
# > **Technical note on the `rate` field.** NWB declares `rate = 0.02`, but the value is the **time
# > step in seconds** (20 ms bins → 50 Hz), not a frequency. The official FALCON loader
# > makes the same assumption, and we keep it to stay bit-compatible with the benchmark.
#
# ### Differences from the previous dataset (FALCON B1, zebra finch song)
#
# | | B1 (zebra finch RA) | **H1 (human, motor cortex)** |
# |---|---|---|
# | neural feature | binary threshold crossings @30 kHz, 85 channels | **spike times** of 176 multi-unit channels |
# | target | audio spectrogram (158 frequency bins) | **7-DoF kinematics** (position + velocity) |
# | trial structure | stereotyped song motif (~2 s) | reach-and-grasp sequence (~16 s) **split into labeled phases** |
# | discrete labels | absent (continuous target only) | **present**: 8 movement phases per trial |
#
# The last row is the methodologically most important difference: B1 had no discrete
# labels (latent trajectories were colored with the audio envelope as a proxy), whereas
# **H1 directly provides the movement phases**, which we will use as conditions to color and
# compare trajectories in notebook 02.

# %%
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# make the 'thesis_h1' package (src/) importable
ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(ROOT / "src"))
from thesis_h1 import io  # noqa: E402

DATA = ROOT / "data" / "raw"
FIGS = ROOT / "figures"
FIGS.mkdir(exist_ok=True)

plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False})

sessions = io.find_sessions(DATA)
print("ROOT:", ROOT)
print("NWB sessions found:", len(sessions))

# %% [markdown]
# ## 1 · Dataset inventory
#
# The 40 files are divided into 3 splits, which the FALCON benchmark defines as follows (and which map
# directly onto the thesis plan):
#
# - **held-in-calib** → "seen" sessions, many trials per session: **training** of manifold and decoder;
# - **held-in-minival** → small subset of the same sessions, for quick **validation** and
#   for checking the submission format;
# - **held-out-calib** → **different sessions** (later days) with few trials each: used
#   for the **few-shot recalibration** experiment, i.e. adapting an already trained decoder to a
#   new session with very little data.
#
# The held-in / held-out distinction is the core of the benchmark: it measures the **stability over time**
# of the decoder, i.e. how much it degrades when the neural signal changes from one day to the next
# (implant *non-stationarity*).

# %%
inv = pd.DataFrame([io.quick_meta(p) for p in sessions])
inv = inv.sort_values(["split", "session"]).reset_index(drop=True)
display(inv)

summary = inv.groupby("split").agg(
    n_sessions=("session", "count"),
    duration_tot_s=("duration_s", "sum"),
    n_trials=("n_trials", "sum"),
    trials_per_session=("n_trials", "mean"),
    units=("n_units", "max"),
).round(1)
display(summary)
print(f"Total: {inv.duration_s.sum()/60:.1f} minutes of recording, "
      f"{inv.n_trials.sum()} trials, {inv.n_units.max()} units (constant across all sessions).")

# %% [markdown]
# **Reading the numbers.**
#
# - The **176 units are identical in all sessions**: the dataset is already aligned channel by
#   channel, so a decoder trained on one session can be applied to another without
#   remapping the inputs. This is what makes the held-out experiment meaningful.
# - `held-in-calib` contains ~13 sessions × ~13 trials ≈ 45 minutes: this is the bulk of the data.
# - `held-out-calib` has only **3 trials per session** (~50 s): very little data, so the
#   readaptation has to be *few-shot*.
# - `eval_frac` ≈ 0.97: almost all bins are evaluable (see §8 for what is excluded).
#
# Note: the session dates (`1925…`) are **anonymized** by the DANDI archive by shifting
# the year back; only the relative ordering matters, which separates held-in days from held-out days.

# %% [markdown]
# ## 2 · A session in detail
#
# We load a `held-in-calib` session and build the **firing-rate matrix**:
#
# 1. spike times are **binned at 20 ms** (the native step of the benchmark: no arbitrary
#    resampling, the bins coincide with those of the kinematics);
# 2. the counts are **smoothed** with a Gaussian kernel (σ = 50 ms) along time;
# 3. counts/bin are converted to **spikes/s** by dividing by 0.02 s.
#
# The result is an *n_bin × 176 units* matrix at 50 Hz, perfectly aligned with the kinematics.

# %%
SESS = DATA / "sub-HumanPitt-held-in-calib" / "sub-HumanPitt-held-in-calib_ses-19250108T110520.nwb"
s = io.load_session(SESS)
rates = io.firing_rates(s, sigma_ms=50)

print(s)
print(f"spikes (20 ms counts): {s.spikes.shape}   total {s.spikes.sum():,} spikes")
print(f"firing_rates:            {rates.shape} @ {1/io.BIN_S:.0f} Hz")
print(f"kinematics position:    {s.kin.shape}   velocity: {s.vel.shape}")
print(f"mean firing rate per unit: {s.spikes.sum(0).mean()/s.duration_s:.1f} Hz  "
      f"(median {np.median(s.spikes.sum(0)/s.duration_s):.1f} Hz)")

slices = io.trial_slices(s)
durs = np.array([(b - a) * io.BIN_S for a, b in slices])
print(f"trials: {len(slices)}  duration {durs.min():.1f}–{durs.max():.1f}s (median {np.median(durs):.1f}s)")

# %% [markdown]
# ## 3 · Structure of a trial: the task phases
#
# Each ~16 s trial is **not** a single movement, but a **stereotyped sequence of phases**,
# labeled in the `epochs` table:
#
# | phase | what happens |
# |---|---|
# | `Presentation`…`Presentation7` | the cue shows the next movement; the virtual arm is still |
# | `Reach` | 3D translation toward the object |
# | `Orient` | wrist rotation |
# | `SnapTo` | the virtual arm "snaps" to the exact pose (automatic correction, not an attempted movement) |
# | `Shape` | hand pre-shaping |
# | `Grasp` | closing the grasp |
# | `Carry` | carrying the object |
# | `Orient2` | second rotation |
# | `Release` | opening the grasp |
#
# This structure is the **methodological key** of the change of dataset: it provides *discrete,
# repeated conditions* (~15 repetitions per phase in a single session, 170 over the whole
# held-in split). They are the equivalent of the "syllables" that B1 lacked, and we will use them in notebook 02
# as aligned segments to estimate and compare manifolds.

# %%
ph_tab = s.epochs.groupby("tag")["dur"].agg(["count", "mean", "min", "max"]).round(2)
ph_tab = ph_tab.reindex([t for t in io.PHASE_COLORS if t in ph_tab.index]
                        + [t for t in ph_tab.index if t not in io.PHASE_COLORS])
display(ph_tab.rename(columns={"count": "n repetitions", "mean": "mean duration (s)",
                               "min": "min (s)", "max": "max (s)"}))

# phase timeline of the first 2 trials
a0, b0 = slices[0][0], slices[1][1]
t0, t1 = a0 * io.BIN_S, b0 * io.BIN_S
fig, ax = plt.subplots(figsize=(12, 2.2))
for _, e in s.epochs.iterrows():
    if e.stop_time <= t0 or e.start_time >= t1:
        continue
    base = e.tag.rstrip("234567") if e.tag.startswith("Presentation") else e.tag
    ax.barh(0, e.stop_time - e.start_time, left=e.start_time - t0, height=0.6,
            color=io.PHASE_COLORS.get(base, "0.8"), edgecolor="w", lw=0.7)
    if e.dur > 0.6 and not e.tag.startswith("Presentation"):
        ax.text(e.start_time - t0 + e.dur / 2, 0, e.tag, ha="center", va="center",
                fontsize=7, color="w", fontweight="bold")
ax.axvline((slices[0][1] - a0) * io.BIN_S, color="k", ls="--", lw=1)
ax.set(yticks=[], xlabel="time (s)", xlim=(0, t1 - t0), ylim=(-0.55, 0.45),
       title="Task phases — first 2 trials (dashed line = trial boundary)")
ax.grid(False)
handles = [Patch(color=c, label=t) for t, c in io.PHASE_COLORS.items()]
ax.legend(handles=handles, ncol=1, fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5),
          frameon=False)
fig.tight_layout(); fig.savefig(FIGS / "01_task_phases.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# ## 4 · Neural activity — raster + population firing rate
#
# Spike raster of a single trial (one dot = one spike of a unit) and firing rate averaged
# over the 176 units. The colored bands in the background are the task phases: the population
# activity is visibly **modulated systematically by the phases**; it is not a stationary background.

# %%
a, b = slices[0]
t0, t1 = a * io.BIN_S, b * io.BIN_S

def shade_phases(ax, alpha=0.18):
    """Colored background per phase, on the time axis relative to t0 (current trial)."""
    for _, e in s.epochs.iterrows():
        if e.stop_time <= t0 or e.start_time >= t1:
            continue
        base = e.tag.rstrip("234567") if e.tag.startswith("Presentation") else e.tag
        if base == "Presentation":
            continue  # leave the cue-only presentation phases white
        ax.axvspan(max(e.start_time, t0) - t0, min(e.stop_time, t1) - t0,
                   color=io.PHASE_COLORS.get(base, "0.8"), alpha=alpha, lw=0, zorder=0)

fig, ax = plt.subplots(2, 1, figsize=(12, 7.5), sharex=True,
                       gridspec_kw={"height_ratios": [3.4, 1]})
order = np.argsort(s.spikes.sum(0))          # units sorted by increasing activity
events = [s.spike_times[u][(s.spike_times[u] >= t0) & (s.spike_times[u] < t1)] - t0
          for u in order]
ax[0].eventplot(events, colors="k", lineoffsets=np.arange(s.n_units),
                linelengths=0.85, linewidths=0.32)
shade_phases(ax[0])
ax[0].set(ylabel="units (sorted by firing rate)", ylim=(-2, s.n_units + 1),
          title=f"Spike raster — trial 1 ({t1-t0:.1f} s, {s.n_units} multi-unit channels)")
ax[0].grid(False)

tt = np.arange(b - a) * io.BIN_S
ax[1].plot(tt, rates[a:b].mean(1), color="C3", lw=1.2)
shade_phases(ax[1])
ax[1].set(xlabel="time in trial (s)", ylabel="Hz", xlim=(0, t1 - t0),
          title="Population firing rate (mean over units, σ=50 ms)")
handles = [Patch(color=io.PHASE_COLORS[t], alpha=0.35, label=t) for t in io.MOVEMENT_PHASES]
ax[1].legend(handles=handles, ncol=8, fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.35))
fig.tight_layout(); fig.savefig(FIGS / "01_raster_trial0.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# ## 5 · The 7-DoF kinematics (the target)
#
# The 7 dimensions of the reference kinematics for the same trial: **position** (top) and
# **velocity** (bottom). Velocity is the official target of the benchmark.
#
# The logic of the task is clearly visible: during `Reach` and `Carry` the translations move
# (`tx, ty, tz`), during `Orient`/`Orient2` the orientation (`rx`), during `Shape`/`Grasp`/`Release`
# the three grasp degrees of freedom (`g1, g2, g3`). During `Presentation` the kinematics are still.

# %%
fig, ax = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True)
cols = plt.cm.tab10(np.linspace(0, 1, 10))
for j, lab in enumerate(io.KIN_LABELS):
    ax[0].plot(tt, s.kin[a:b, j], lw=1.2, color=cols[j], label=lab)
    ax[1].plot(tt, s.vel[a:b, j], lw=1.2, color=cols[j], label=lab)
shade_phases(ax[0]); shade_phases(ax[1])
ax[0].set(ylabel="position (a.u.)", title="7-DoF kinematics — position (trial 1)")
ax[0].legend(ncol=7, fontsize=8, loc="upper right")
ax[1].set(xlabel="time in trial (s)", ylabel="velocity (a.u./bin)", xlim=(0, t1 - t0),
          title="7-DoF kinematics — velocity  (= official FALCON target, R² metric)")
fig.tight_layout()
# Legend of the background bands = task phases. Only the movement phases: the
# `Presentation` phases are not shaded (kinematics still), so they do not go in the legend.
# It must be anchored to the *figure*, not to the axes: tight_layout ignores artists placed
# outside the axes and would crop it off the canvas. subplots_adjust first reserves
# the bottom margin, then fig.legend draws into it.
fig.subplots_adjust(bottom=0.20)
handles = [Patch(color=io.PHASE_COLORS[t], alpha=0.45, label=t) for t in io.MOVEMENT_PHASES]
fig.legend(handles=handles, ncol=8, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 0.015),
           frameon=False, title="task phase", title_fontsize=8.5)
fig.savefig(FIGS / "01_kinematics_trial0.png", bbox_inches="tight"); plt.show()

# which DoF is active in which phase: mean absolute velocity per (phase, DoF)
phase_lab = io.phase_per_bin(s)
mov = pd.DataFrame(np.abs(s.vel), columns=io.KIN_LABELS)
mov["phase"] = [p.rstrip("234567") if p.startswith("Presentation") else p for p in phase_lab]
tab = mov[mov.phase.isin(io.MOVEMENT_PHASES + ("Presentation",))].groupby("phase").mean()
tab = tab.reindex([t for t in io.MOVEMENT_PHASES if t in tab.index] + ["Presentation"])

fig, ax = plt.subplots(figsize=(7, 3.4))
im = ax.imshow(tab.values / tab.values.max(), cmap="viridis", aspect="auto")
ax.set(xticks=range(7), xticklabels=io.KIN_LABELS, yticks=range(len(tab)), yticklabels=tab.index,
       title="Mean |velocity|, normalized — which DoF moves in which phase")
fig.colorbar(im, ax=ax, label="|v| (norm.)", pad=0.02)
ax.grid(False)
fig.tight_layout(); fig.savefig(FIGS / "01_phase_dof_map.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# ## 6 · The decoding problem in one figure
#
# Neural activity of the 176 units (top) ↔ velocity of the 7 DoF (bottom), same time axis.
# This is **exactly** what the pipeline will have to learn: *neural manifold → kinematics*.
# Compared with B1 (neural → spectrogram → audio) the target is much lower-dimensional,
# but the problem remains the same: extracting from 176 noisy channels the few latent dimensions
# that carry the behavioral signal.

# %%
fig, ax = plt.subplots(2, 1, figsize=(12, 6.5), sharex=True,
                       gridspec_kw={"height_ratios": [2, 1]})
im0 = ax[0].pcolormesh(tt, np.arange(s.n_units), rates[a:b][:, order].T,
                       shading="auto", cmap="viridis")
ax[0].set(ylabel="units", title="Neural activity — firing rate, 176 units (σ=50 ms)")
fig.colorbar(im0, ax=ax[0], label="Hz", pad=0.01)
ax[0].grid(False)
im1 = ax[1].pcolormesh(tt, np.arange(7), s.vel[a:b].T, shading="auto", cmap="RdBu_r",
                       vmin=-np.abs(s.vel[a:b]).max(), vmax=np.abs(s.vel[a:b]).max())
ax[1].set(xlabel="time in trial (s)", yticks=range(7), yticklabels=io.KIN_LABELS,
          ylabel="DoF", title="7-DoF velocity (target)")
fig.colorbar(im1, ax=ax[1], label="v", pad=0.01)
ax[1].grid(False)
fig.tight_layout(); fig.savefig(FIGS / "01_neural_vs_kinematics.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# ## 7 · Is the neural activity really modulated by the task?
#
# Before building any model it is worth checking that the signal is there. We compute the
# **population PSTH aligned to the onset of each phase** (mean over all repetitions of the
# phase in the session): if the motor cortex encodes the attempted movement, the firing rate must
# increase in a *phase-specific* way after onset.
#
# We also add a per-unit measure: the **modulation across phases**, i.e. how much the mean firing rate
# of each unit varies from one phase to another relative to its internal variability
# (a one-way ANOVA, reported as the fraction of variance explained by the phase, η²).

# %%
# population PSTH aligned to the onset of each movement phase
PRE, POST = 0.4, 1.0
npre, npost = int(PRE / io.BIN_S), int(POST / io.BIN_S)
pop = rates.mean(1)

fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
for tag in io.MOVEMENT_PHASES:
    onsets = [int(round(e.start_time / io.BIN_S)) for _, e in s.epochs.iterrows() if e.tag == tag]
    segs = [pop[o - npre:o + npost] for o in onsets if o - npre >= 0 and o + npost <= s.n_bins]
    if not segs:
        continue
    m = np.mean(segs, 0)
    ax[0].plot(np.arange(-npre, npost) * io.BIN_S, m, color=io.PHASE_COLORS[tag], lw=1.6, label=tag)
ax[0].axvline(0, color="k", ls="--", lw=1)
ax[0].set(xlabel="time from phase onset (s)", ylabel="Hz",
          title="Population PSTH aligned to phase onset")
ax[0].legend(fontsize=7, ncol=2)

# per-unit modulation: eta^2 of the ANOVA firing rate ~ phase
lab = np.array([p.rstrip("234567") if p.startswith("Presentation") else p for p in phase_lab])
keep = np.isin(lab, io.MOVEMENT_PHASES + ("Presentation",))
X, L = rates[keep], lab[keep]
gm = X.mean(0)
ss_tot = ((X - gm) ** 2).sum(0)
ss_between = np.zeros(s.n_units)
for g in np.unique(L):
    m = L == g
    ss_between += m.sum() * (X[m].mean(0) - gm) ** 2
eta2 = ss_between / np.maximum(ss_tot, 1e-12)
ax[1].hist(eta2, bins=30, color="C0", alpha=0.85)
ax[1].axvline(np.median(eta2), color="C3", ls="--", label=f"median {np.median(eta2):.3f}")
ax[1].set(xlabel="η² (firing-rate variance explained by phase)", ylabel="n units",
          title="Per-unit modulation across task phases")
ax[1].legend()
fig.tight_layout(); fig.savefig(FIGS / "01_task_modulation.png", bbox_inches="tight"); plt.show()
print(f"median η²: {np.median(eta2):.3f}   max: {eta2.max():.3f}")
print(f"Units with η² > 0.05: {(eta2 > 0.05).sum()}/{s.n_units}  "
      f"(> 0.10: {(eta2 > 0.10).sum()})")

# %% [markdown]
# **Interpretation — and an important result for what follows.** The population PSTH shows clear
# **phase-specific** modulations (the hand-manipulation phases — `Shape`, `Release`,
# `Grasp` — produce the largest peaks), but the histogram on the right shows that **a single unit carries
# little information**: median η² ≈ 0.015, and only ~25 units out of 176 exceed 0.05.
#
# This is the typical picture of cortical multi-unit signals: the motor code is **distributed
# over the population**, not readable from single channels. It is exactly the premise that justifies
# the approach of the thesis — looking for the few **shared latent dimensions** (the *neural manifold*)
# in which the behavioral signal emerges, instead of treating the 176 units as independent features.
# Notebook 02 starts from here.

# %% [markdown]
# ## 8 · `eval_mask`: what is actually evaluated
#
# The benchmark excludes some bins from the metric. Let us see **which ones**, because it is a detail that
# changes the results if ignored.

# %%
ev = pd.DataFrame({"phase": lab if len(lab) == s.n_bins else phase_lab, "eval": s.eval_mask})
ev["phase"] = [p.rstrip("234567") if p.startswith("Presentation") else p for p in ev["phase"]]
frac = ev.groupby("phase")["eval"].agg(["mean", "size"]).sort_values("mean")
display(frac.rename(columns={"mean": "fraction of evaluated bins", "size": "n bins"}).round(3))
print(f"Total excluded bins: {(~s.eval_mask).sum()} / {s.n_bins} "
      f"({100*(1-s.eval_mask.mean()):.1f}%)")

# %% [markdown]
# **Interpretation.** The exclusions are concentrated almost entirely in the **`SnapTo`** phase (and
# partly in `Intertrial`). This makes sense: `SnapTo` is the moment in which the virtual arm **snaps
# automatically** to the correct pose — it is a correction imposed by the software, not a movement
# that the participant is attempting. Evaluating the decoder there would mean asking it to predict
# an artifact of the simulation. The same holds for the pauses between trials.
#
# **Practical consequence:** `eval_mask` must be applied when computing R², and it is advisable to exclude those
# bins during training as well, so as not to teach the model a neural↔kinematics relationship
# that does not exist in the task.

# %% [markdown]
# ## 9 · Basic statistics

# %%
unit_hz = s.spikes.sum(0) / s.duration_s
all_phase_dur = s.epochs[s.epochs.tag.isin(io.MOVEMENT_PHASES)]

fig, ax = plt.subplots(1, 3, figsize=(13, 3.4))
ax[0].hist(unit_hz, bins=30, color="C0", alpha=0.85)
ax[0].axvline(unit_hz.mean(), color="C3", ls="--", label=f"mean {unit_hz.mean():.1f} Hz")
ax[0].set(xlabel="mean firing rate (Hz)", ylabel="n units",
          title=f"Rate per unit (n={s.n_units})")
ax[0].legend()

ax[1].hist(durs, bins=10, color="C2", alpha=0.85)
ax[1].set(xlabel="trial duration (s)", ylabel="n trials", title="Trial duration")

data = [all_phase_dur[all_phase_dur.tag == t].dur.values for t in io.MOVEMENT_PHASES]
bp = ax[2].boxplot(data, tick_labels=list(io.MOVEMENT_PHASES), patch_artist=True)
for patch, t in zip(bp["boxes"], io.MOVEMENT_PHASES):
    patch.set_facecolor(io.PHASE_COLORS[t]); patch.set_alpha(0.7)
ax[2].tick_params(axis="x", rotation=60, labelsize=7)
ax[2].set(ylabel="duration (s)", title="Duration of movement phases")
fig.tight_layout(); fig.savefig(FIGS / "01_stats.png", bbox_inches="tight"); plt.show()

n_silent = int((unit_hz < 0.5).sum())
print(f"Nearly silent units (<0.5 Hz): {n_silent}/{s.n_units} → excluded from the manifold analyses.")
print("Most active units (Hz):", np.round(np.sort(unit_hz)[::-1][:5], 1))

# %% [markdown]
# ## 10 · Summary and next steps
#
# **What we established.**
#
# 1. **Neural feature** = spike counts of 176 multi-unit channels, binned at **20 ms** (native
#    step of the benchmark). Firing rates are obtained with Gaussian smoothing σ = 50 ms.
#    ~16 units are nearly silent (<0.5 Hz) and will be excluded from manifold learning.
# 2. **Target** = 7-DoF kinematics; the official metric is the **R² on velocity**, already aligned
#    bin by bin with the neural activity (no synchronization problem to solve).
# 3. **Task structure**: each trial is a sequence of **stereotyped, repeated phases**
#    (Reach, Orient, SnapTo, Shape, Grasp, Carry, Orient2, Release), with ~15 repetitions per
#    session and 170 over the whole held-in split. They are **discrete conditions**, which B1 did not have.
# 4. **The signal is there, but it is distributed**: the *population* firing rate is modulated in a
#    phase-specific way, whereas a single unit explains little variance (median η² ≈ 0.015, 25/176 units
#    above 0.05). This is the direct motivation for the population-level (manifold) analysis.
# 5. **`eval_mask`** mostly excludes the `SnapTo` phase (movement imposed by the software, not
#    attempted by the participant): it must be respected both in evaluation and in training.
# 6. **The splits** are already set up for the key experiment of the benchmark: held-in for training,
#    held-out (sessions from later days, 3 trials each) for **few-shot recalibration**.
#
# **Next step — `02_baseline_manifolds.ipynb`:** extraction of the neural manifold with the two
# classical linear methods, **PCA** and **GPFA**, selection of the latent dimensionality, and quantitative
# comparison of smoothness and single-trial consistency. The task phases will be used as
# conditions to color and align the latent trajectories.
