# %% [markdown]
# # 02_new · From the threshold to PCA, FA and GPFA, step by step (FALCON H1)
#
# **Goal.** Rebuild from scratch the path that leads from the NWB file to PCA, FA and GPFA, explaining
# **every line of code** and **the reason for every choice**. Notebook 02 performed the same steps
# inside the functions of `thesis_h1.io`; here we write them out in full and, at each step, check
# that the result matches that of the project functions.
#
# The full pipeline:
#
# ```
# voltage at 30 kHz ──(threshold −4.5 × RMS: applied upstream, by the recording system)──► spike_times
#
# spike_times ──(1. binning at 20 ms)─────────────────────► counts          T × 176
#             ──(2. active-channel selection ≥ 0.5 Hz)────► counts          T × 160
#             ──(3. Gaussian smoothing σ = 50 ms)─────────► firing rate     T × 160
#             ──(4. per-channel z-score)──────────────────► same scale      T × 160
#
# then, to understand how much of what PCA sees is signal:
#
#   §8  PCA on the repetition means of each phase ──────► is there a signal, and is it simple?
#   §9  Factor Analysis on the raw counts ───────────────► shared signal vs private noise
#   §10 GPFA on the phase windows ───────────────────────► FA + a model of time: single-trial trajectories
# ```
#
# Only the threshold is not our choice: it is applied before the data reach the file.
# Everything else is our choice, and each choice has a parameter to justify: for each one there is
# a sensitivity analysis ("Alternatives tried" table in §11).
#
# **Run time:** about 20 minutes, almost all of it in the FA cross-validations (§9.1, §9.5)
# and the GPFA cross-validation (§10.2).

# %% [markdown]
# ## 0 · Setup

# %%
import sys                                        # needed to add src/ to the paths where Python looks for modules
from pathlib import Path                          # file paths, independent of the operating system

import numpy as np                                # arrays and linear algebra
import pandas as pd                               # tables, only to print readable summaries
import matplotlib.pyplot as plt                   # figures
from matplotlib.patches import Patch              # coloured rectangles for the phase legends
from scipy.ndimage import gaussian_filter1d       # Gaussian smoothing along a single axis
from sklearn.decomposition import PCA, FactorAnalysis  # PCA (in §6 we also redo it by hand) and Factor Analysis (§9)
from sklearn.model_selection import KFold         # splitting the data into blocks for cross-validation (§9.1)
from joblib import Parallel, delayed              # to run several fits in parallel on the computer's cores
from sklearn.preprocessing import StandardScaler  # column-by-column z-score
from pynwb import NWBHDF5IO                       # reading NWB files

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()  # project root, whichever folder the notebook is started from
sys.path.insert(0, str(ROOT / "src"))             # makes the project package importable
from thesis_h1 import io                            # used only for colours/phases and to check our steps

FIGS = ROOT / "figures"                           # folder where we export the figures as PNG
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False})  # same plot style as the other notebooks

SESS = ROOT / "data/raw/sub-HumanPitt-held-in-calib/sub-HumanPitt-held-in-calib_ses-19250108T110520.nwb"  # session analysed

# %% [markdown]
# ## 1 · The threshold: where the `spike_times` come from
#
# ### What a *threshold crossing* is
#
# Each electrode of the array records the **extracellular voltage** at 30 kHz. When a neuron
# close to the tip fires an action potential, the voltage shows a fast negative deflection
# (shorter than 1 ms). The acquisition system (Blackrock NeuroPort):
#
# 1. **filters** the signal, removing the slow components (field potentials) and keeping the
#    band in which spikes fall;
# 2. estimates the **background noise** of each channel as the **RMS** (*root mean square*, the standard
#    deviation of the voltage) over a reference period at the start of the session;
# 3. records an event every time the voltage **drops below −4.5 × RMS**.
#
# Each event is a *threshold crossing*. **There is no spike sorting**: events from the same
# electrode may come from several nearby neurons. For this reason the 176 "units" of the NWB file are
# **multi-unit channels**, one per electrode, and not single neurons.
#
# ### Where the value −4.5 × RMS comes from
#
# **The FALCON paper does not report the numerical value for H1**, and the NWB file does not store it either.
# The paper only says that threshold crossings are extracted "as per standard practice" and, in the
# discussion of limitations, that thresholds are generally a multiple of the RMS but that the
# exact multiple and protocol vary from dataset to dataset.
#
# The value is derived from work **by the same lab on the same participant**. The
# participant is recognised by the number of channels: 176 = 2 arrays × **88 wired electrodes**.
# This is the configuration described in Flesher et al. 2021 (*Science*), one of the three works that FALCON
# cites as the source of H1: *"Two microelectrode arrays with 88 wired electrodes were implanted in the
# hand and arm region of motor cortex"*.
#
# | source | participant | threshold | filter | bin |
# |---|---|---|---|---|
# | Karpowicz et al. 2024 (FALCON, NeurIPS D&B), App. A.3.5 | H1 | **not reported** | not reported | 20 ms |
# | Downey et al. 2017, *Sci. Rep.* 7:16947 | "Subject 2": 2 arrays of 88 channels in M1 | **−4.5 × RMS** | band-pass 250–7500 Hz | 20 ms |
# | Quick et al. 2020, *IEEE EMBC*, 3355–3358 | 2 arrays of 88 electrodes in M1, trial NCT01894802 | **−4.5 × RMS**, RMS measured at session start | 0.3–7500 Hz, then 750 Hz high-pass | 20 ms |
#
# Two independent works on the same participant report the same value. In the thesis it should
# be written as: *"threshold at −4.5 × RMS, following the lab protocol (Downey et al. 2017;
# Quick et al. 2020); the FALCON paper does not report the value"*.
#
# Two notes:
#
# - **The threshold is recomputed at every session**, because the RMS is measured at the start of each one.
#   The same electrode may therefore have a slightly different threshold from one day to the next, and
#   this contributes to the between-session non-stationarity that the benchmark is meant to measure.
# - Figure 1 of the FALCON paper lists 172 channels for H1, but the NWB files contain 176.
#   We use the 176 in the file.
#
# ### What we can check ourselves in the data
#
# We do not have the raw voltage, but the event times carry two traces of the system that
# produced them. We check them below.

# %%
THRESH_RMS = -4.5    # threshold used upstream by the lab (table above). Documentation only: it enters no
                     # calculation, because the crossings were already extracted before reaching the file
FS_RAW = 30_000      # sampling rate of the raw voltage (Hz), from the FALCON paper

with NWBHDF5IO(str(SESS), "r") as f:                   # opens the file read-only; "with" closes it at the end of the block
    nwb = f.read()                                     # reads the file structure (the actual data are read when needed)
    n_ch = len(nwb.units.id)                           # number of channels = rows of the "units" table (176)
    spike_times = [np.asarray(nwb.units["spike_times"][u])  # for each channel u, the array of crossing times (in s)
                   for u in range(n_ch)]
    vel = nwb.acquisition["OpenLoopKinematicsVelocity"]  # velocity of the 7 DoF: used only to know how many bins the session has
    n_bins = vel.data.shape[0]                         # one kinematics sample every 20 ms = number of bins
    offset = float(vel.offset or 0.0)                  # time (s) of the first sample; here it is 0

n_events = np.array([len(st) for st in spike_times])   # number of crossings of each channel
print(f"{n_ch} channels, {n_events.sum():,} threshold crossings in total")

all_t = np.concatenate(spike_times)                    # the times of all channels in a single vector
grid_err = np.abs(all_t * FS_RAW - np.round(all_t * FS_RAW)).max()  # maximum distance from an exact multiple of 1/30000 s
print(f"1) Maximum deviation from the 30 kHz grid: {grid_err:.1e} samples "
      "-> every time is the index of a sample of the raw signal")

min_isi_ms = np.array([np.diff(np.sort(st)).min() * 1e3  # shortest interval between two consecutive crossings, in ms
                       for st in spike_times if len(st) > 1])  # (only for channels with at least two events)
print(f"2) Minimum interval between two crossings of the same channel: {np.median(min_isi_ms):.3f} ms "
      f"(median across channels) = {np.median(min_isi_ms) * FS_RAW / 1000:.0f} samples at 30 kHz")

# %% [markdown]
# **Reading.**
#
# 1. All times fall exactly on multiples of 1/30000 s: they are sample indices of the 30 kHz
#    signal, i.e. events detected on the raw signal.
# 2. On more than half of the channels the shortest interval between two events is **exactly 1.633 ms
#    (49 samples)**, and on no channel is it shorter. A minimum identical to the sample across so many
#    different channels points to a limit of the instrument, not of physiology. It is compatible with the
#    detector's dead time: after a crossing the processor saves a waveform snippet
#    (1.6 ms) and during that time cannot detect another one.

# %% [markdown]
# ## 2 · From event times to a matrix: binning
#
# `spike_times` is a list of 176 arrays, each with the event times of one channel, and the channels
# have different numbers of events. PCA instead needs a **matrix**: rows = time points,
# columns = channels, with the same number of rows for all.
#
# **Binning** performs this conversion: it divides time into consecutive 20 ms windows and, for
# each channel, **counts** how many events fall in each window.
#
# **Why 20 ms.** It is the native step of the benchmark: kinematics are sampled at 50 Hz, i.e. one
# sample every 20 ms. With 20 ms bins each row of the neural matrix corresponds exactly to one
# kinematics sample, and this alignment is what the decoder needs. With any other step
# the kinematics would have to be resampled.
#
# > **Binning is not smoothing.** Binning *counts*: each bin looks only at its own 20 ms and
# > knows nothing about neighbouring bins (the windows do not overlap). The smoothing of §4 will be the
# > **first and only** time we mix neighbouring bins: we are not smoothing twice.

# %%
BIN_S = 0.02                                            # bin width: 20 ms
edges = offset + np.arange(n_bins + 1) * BIN_S          # n_bins + 1 edges: 0, 0.02, 0.04, ... Bin i covers [edges[i], edges[i+1])
counts = np.zeros((n_bins, n_ch), dtype=np.int16)       # empty matrix: rows = time bins, columns = channels
for ch, st in enumerate(spike_times):                   # for each channel...
    counts[:, ch], _ = np.histogram(st, bins=edges)     # ...count how many crossings fall in each bin

print(f"Counts matrix: {counts.shape[0]} bins × {counts.shape[1]} channels "
      f"({n_bins * BIN_S:.1f} s of recording)")
print(f"Events outside the bin edges (before the first bin or after the last): {n_events.sum() - counts.sum()}")

s = io.load_session(SESS)                               # the same session loaded with the project function
print("Identical to the io.load_session matrix:", np.array_equal(counts, s.spikes))

# %% [markdown]
# ## 3 · Active-channel selection
#
# For each channel we compute the mean firing rate (total events / duration) and keep only those
# with **at least 0.5 events per second**. There are three reasons, and the third is clearly visible in §5:
#
# 1. **No task information.** A channel at 0.2 Hz has ~50 events in 4 minutes: too few
#    to tell whether its activity changes from one phase to another.
# 2. **Zero variance.** A channel with no events has zero variance, and the z-score of §5 would divide
#    by zero. In this session there is at least one such channel.
# 3. **Outliers after the z-score.** An almost silent channel has a tiny standard deviation. When
#    the z-score divides by that standard deviation, its rare events become huge values.
#    PCA, which looks for maximum variance, would end up chasing those few points.
#
# **0.5 Hz is our choice**, not a benchmark value. The table below shows how many
# channels different thresholds would exclude: 0.5 Hz removes the clearly silent channels without losing
# the weak but active ones.

# %%
duration_s = n_bins * BIN_S                             # session duration in seconds
rate_hz = counts.sum(axis=0) / duration_s               # mean firing rate of each channel = total events / duration
MIN_HZ = 0.5                                            # activity threshold (our choice)
ACTIVE = np.flatnonzero(rate_hz >= MIN_HZ)              # indices of the channels above the threshold
X_counts = counts[:, ACTIVE]                            # reduced matrix: only the columns of the active channels

print(f"Active channels: {len(ACTIVE)} of {n_ch}  (excluded {n_ch - len(ACTIVE)})")
print("Same selection as io.active_units:", np.array_equal(ACTIVE, io.active_units(s, min_hz=MIN_HZ)))
print(f"Channels with zero events over the whole session: {(rate_hz == 0).sum()}")
display(pd.DataFrame({"threshold (Hz)": [0.1, 0.5, 1, 2, 5],
                      "excluded channels": [int((rate_hz < th).sum()) for th in (0.1, 0.5, 1, 2, 5)]})
        .set_index("threshold (Hz)").T)

fig, ax = plt.subplots(figsize=(11, 3.4))
order = np.argsort(rate_hz)                             # channels sorted from least to most active
ax.bar(np.arange(n_ch), rate_hz[order], width=0.8,
       color=np.where(rate_hz[order] >= MIN_HZ, "C0", "0.7"))  # blue = kept, grey = excluded
ax.axhline(MIN_HZ, color="C3", ls="--", lw=1.2)
ax.text(2, MIN_HZ * 1.25, f"{MIN_HZ} Hz", color="C3", fontsize=9)
ax.set_yscale("symlog", linthresh=0.1)                  # logarithmic scale that also accepts zero: slow channels stay visible
ax.set(xlabel="channels, sorted by firing rate", ylabel="mean firing rate (Hz)",
       title=f"Mean firing rate per channel: {n_ch - len(ACTIVE)} excluded (grey), {len(ACTIVE)} kept (blue)",
       xlim=(-1, n_ch))
fig.tight_layout(); fig.savefig(FIGS / "02new_active_channels.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# **A note for when we use more sessions.** The selection must be made **once only** (for
# example on the training sessions) and then kept fixed. If each session had its own set
# of channels, the starting space of PCA would change from one session to another and the manifolds would
# not be comparable.

# %% [markdown]
# ## 4 · Smoothing: from noisy counts to firing rate
#
# ### Why it is needed
#
# Let us look at what a 20 ms bin really contains.

# %%
vals, freq = np.unique(X_counts, return_counts=True)    # distinct values present in the matrix and how many times they occur
frac = freq / X_counts.size                             # fraction of bins with 0, 1, 2, ... events
display(pd.DataFrame({"events in the bin": vals[:5], "fraction of bins (%)": (frac[:5] * 100).round(2)})
        .set_index("events in the bin").T)
print(f"Mean: {X_counts.mean():.2f} events per bin and per channel "
      f"(= {X_counts.mean() / BIN_S:.1f} Hz)")

# %% [markdown]
# More than three bins out of four are 0 and almost all the others are 1. The average channel fires at
# ~14 Hz, i.e. **0.28 events per bin**: in any single bin we see either 0 or 1, largely
# by chance. A single bin therefore gives a very poor estimate of the firing rate: it says "0 Hz" or
# "50 Hz", never 14 Hz.
#
# What we are interested in is the **underlying firing rate** λ(t): a quantity that varies
# slowly, on scales of hundreds of ms, following the movement. The counts are noisy samples
# noisy samples of it (approximately Poisson). To estimate λ(t) at a given instant we take a **weighted
# average of the neighbouring bins**: this is what smoothing is.
#
# ### How we do it
#
# We use a **Gaussian kernel** with standard deviation σ = 50 ms (2.5 bins), i.e. a weighted average
# in which the central bin weighs the most and the weight falls off gently with distance.
#
# - **How much noise it removes.** For noise independent from bin to bin, a Gaussian kernel is equivalent
#   to averaging about 2√π · σ ≈ 8.9 bins (about 180 ms): the noise variance is reduced by about
#   9 times.
# - **What it costs.** Variations faster than σ are attenuated. The movement-related modulations of
#   motor cortex last hundreds of ms, so with σ = 50 ms they are preserved.
# - **Why not simply use wider bins?** Because we would lose the 20 ms alignment with
#   the kinematics. Gaussian smoothing reduces noise **while keeping one value every 20 ms**, and
#   its weights fall off gradually instead of cutting off sharply at an edge.
# - **The kernel is symmetric**: it uses both the past and the future, so it introduces no delay. This is
#   fine for offline analysis; an online decoder has no access to the future and would use a causal filter.
#
# The filter acts **along time, channel by channel**: it never mixes different channels. For this reason
# the order of selection (§3) and smoothing does not matter. We apply it to the whole session
# in one pass, crossing the trial boundaries, because the recording is continuous.

# %%
SIGMA_MS = 50                                           # standard deviation of the Gaussian kernel, in ms
sigma_bins = (SIGMA_MS / 1000) / BIN_S                  # the same, expressed in bins: 0.050 s / 0.020 s = 2.5 bins
smoothed = gaussian_filter1d(X_counts.astype(float),    # conversion to float: after averaging the values are no longer integers
                             sigma=sigma_bins,          # kernel width, in bins
                             axis=0,                    # along time (the rows): each channel is filtered separately
                             mode="nearest")            # at the edges of the recording it repeats the first/last value
                                                        # instead of adding non-existent zeros
rates = smoothed / BIN_S                                # from "events per bin" to "events per second" (Hz): only a change of units

print(f"Firing rate: {rates.shape[0]} × {rates.shape[1]} matrix, sigma = {SIGMA_MS} ms = {sigma_bins} bins")
print(f"Equivalent kernel window: {2 * np.sqrt(np.pi) * sigma_bins:.1f} bins "
      f"= {2 * np.sqrt(np.pi) * SIGMA_MS:.0f} ms")
print("Identical to io.smooth_rates:",
      np.allclose(rates, io.smooth_rates(counts, sigma_ms=SIGMA_MS)[:, ACTIVE], atol=1e-4))

# %%
ch = int(np.argmin(np.abs(rate_hz[ACTIVE] - rate_hz[ACTIVE].mean())))  # a "typical" channel: the one whose firing rate is closest to the mean
a, b = io.trial_slices(s)[0]                            # start and end bins of the first trial
a, b = a + 100, a + 350                                 # we take 5 s (250 bins), from second 2 to second 7
t = np.arange(a, b) * BIN_S                             # time axis in seconds

fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.4), gridspec_kw={"width_ratios": [1, 3.2]})
k = np.arange(-8, 9)                                    # bin positions around the kernel centre
w = np.exp(-0.5 * (k / sigma_bins) ** 2); w /= w.sum()  # Gaussian weights, normalised to sum 1
ax[0].bar(k * BIN_S * 1000, w, width=14, color="C0")
ax[0].set(xlabel="distance from the central bin (ms)", ylabel="weight",
          title=f"Gaussian kernel, σ = {SIGMA_MS} ms")

# Background coloured by task phase (as in §7): it lets us see whether the peaks of the curve
# fall during movement, i.e. whether smoothing keeps the signal and does not merely "clean" the curve
phase = lambda tag: "Presentation" if tag.startswith("Presentation") else tag  # "Presentation3" -> "Presentation"
ep = s.epochs[(s.epochs.stop_time > t[0]) & (s.epochs.start_time < t[-1] + BIN_S)]  # phases falling within the displayed window
for _, e in ep.iterrows():
    ax[1].axvspan(max(e.start_time, t[0]), min(e.stop_time, t[-1] + BIN_S),  # clipped at the window edges
                  color=io.PHASE_COLORS.get(phase(e.tag), "0.9"), lw=0,
                  alpha=0.10 if phase(e.tag) in ("Presentation", "Intertrial") else 0.25)  # movement phases more prominent

ax[1].bar(t, X_counts[a:b, ch] / BIN_S, width=BIN_S * 0.9, color="0.6", label="bin count / 20 ms")
for sig, c in [(20, "#9ecae1"), (50, "#3182bd"), (100, "#08306b")]:   # three widths, from light (narrow) to dark (wide)
    r_ = gaussian_filter1d(X_counts[:, ch].astype(float), sig / 1000 / BIN_S, mode="nearest") / BIN_S
    ax[1].plot(t, r_[a:b], color=c, lw=2 if sig == SIGMA_MS else 1.3, label=f"smoothed, σ = {sig} ms")
ax[1].set(xlabel="time in the session (s)", ylabel="firing rate (Hz)", xlim=(t[0], t[-1] + BIN_S),
          title=f"Channel {ACTIVE[ch]} (mean {rate_hz[ACTIVE][ch]:.1f} Hz): raw bins are only multiples of 50 Hz")

leg = ax[1].legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.15), ncol=4)  # legend of the curves, below the plot
ax[1].add_artist(leg)                                   # we keep it, so that we can add a second legend
phases_views = list(dict.fromkeys(phase(x) for x in ep.tag))  # phases present in the window, in order of appearance
ax[1].legend(handles=[Patch(color=io.PHASE_COLORS.get(f, "0.9"), alpha=0.5, label=f) for f in phases_views],
             fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.27), ncol=len(phases_views), title="background = task phase",
             title_fontsize=8)
fig.tight_layout(); fig.savefig(FIGS / "02new_smoothing.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# **Reading.** The grey bars are the raw count: a bin with one event "says" 50 Hz, one with
# two events 100 Hz, an empty one 0 Hz. The curves are the firing-rate estimate with three kernel widths. With
# **σ = 20 ms** the curve is still dominated by single events; with **σ = 100 ms** it becomes very
# smooth but also flattens the true variations. **σ = 50 ms** is the compromise we adopt,
# and how much this choice affects the PCA result is measured in §6.
#
# **The coloured background is the task phase** (as in §7): it serves to check that the peaks of the
# curve are not just smoothed noise. The two highest peaks of the σ = 50 ms curve, at about
# 2.76 s and 4.84 s, fall during `Reach` and `Orient`, two movement phases; over this stretch the
# channel fires on average at about 20 Hz during movement and at about 11 Hz during
# `Presentation`. Smoothing therefore lets task-related variations through.
#
# Be careful not to ask too much of this figure: it is **one channel in one trial**, an illustration and
# not a proof. The real check, over the whole population and all repetitions, is in §7 and §8.
#
# The division by 0.02 s turns counts into Hz, but it is **only a change of units**: after the
# z-score of §5 it disappears (multiplying a column by a constant does not change its z-score).
# It only serves to make the figures readable.

# %% [markdown]
# ## 5 · Per-channel normalization (z-score)
#
# ### Why it is needed
#
# PCA looks for the directions of **maximum variance**. In count data, however, the variance
# grows with the mean (for a Poisson process they are equal). A channel firing at 50 Hz has
# much more variance than one at 5 Hz **only because it fires more**, not because it carries more
# information. Without normalization, the first component would end up describing mainly the
# few most active channels.
#
# The **z-score** transforms each channel as follows: z = (x − mean) / standard deviation. It does two things:
#
# 1. **subtracts the mean** (centring). PCA requires this anyway: it looks for directions of
#    variation *around the mean*;
# 2. **divides by the standard deviation**, so that each channel has variance 1 and counts equally.
#    PCA then operates on the **correlation** matrix between channels, rather than on the
#    covariance matrix.
#
# Below we check the two claims: variance tied to the mean, and PC1 dominated by the most
# active channels if we do not normalize.

# %%
mu = rates.mean(axis=0)                                 # temporal mean of each channel (Hz)
var = rates.var(axis=0)                                 # temporal variance of each channel
print(f"Correlation between channel mean and variance: r = {np.corrcoef(mu, var)[0, 1]:.2f}")

Z = StandardScaler().fit_transform(rates)               # for each column: subtracts the mean and divides by the standard deviation
print(f"After the z-score: means in [{Z.mean(0).min():.1e}, {Z.mean(0).max():.1e}], "
      f"standard deviations in [{Z.std(0).min():.3f}, {Z.std(0).max():.3f}]")

pc1_raw = PCA(n_components=1).fit(rates).components_[0]  # PC1 without normalization (sklearn centres the data anyway)
pc1_z = PCA(n_components=1).fit(Z).components_[0]        # PC1 after the z-score
top10 = np.argsort(mu)[::-1][:10]                        # the 10 channels with the highest firing rate
print(f"Weight of the 10 most active channels in PC1 (sum of squared weights, out of a total of 1): "
      f"{(pc1_raw[top10] ** 2).sum():.2f} without z-score, {(pc1_z[top10] ** 2).sum():.2f} with z-score")

fig, ax = plt.subplots(1, 2, figsize=(12, 4))
ax[0].scatter(mu, var, s=14, color="C0", alpha=0.7)
ax[0].set(xlabel="mean firing rate of the channel (Hz)", ylabel="variance over time (Hz²)",
          title="The more active a channel, the larger its variance")
ax[1].scatter(mu, pc1_z ** 2, s=16, color="C0", alpha=0.75, label="with z-score")
ax[1].scatter(mu, pc1_raw ** 2, s=16, color="C1", alpha=0.75, label="without z-score")  # drawn last, so they are not covered
ax[1].set(xlabel="mean firing rate of the channel (Hz)", ylabel="weight in PC1 (squared)",
          title="Without z-score, PC1 is made of the most active channels")
ax[1].legend()
fig.tight_layout(); fig.savefig(FIGS / "02new_normalization.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# ### The link with channel selection (§3)
#
# The z-score has a cost: it gives every channel the same weight, **including the almost silent ones**, which
# have very little signal. Had we not removed them, the z-score would have amplified them. We
# check this by repeating smoothing and z-score on **all** 176 channels and looking at the most
# extreme value each channel reaches.

# %%
rates_all = gaussian_filter1d(counts.astype(float), sigma_bins, axis=0, mode="nearest") / BIN_S  # smoothing of all 176 channels
sd_all = rates_all.std(axis=0)                          # standard deviation of each channel
silent = (rate_hz < MIN_HZ) & (sd_all > 0)              # excluded channels that nevertheless have at least one event
z_peak = (rates_all.max(0) - rates_all.mean(0)) / np.where(sd_all > 0, sd_all, np.nan)  # z-score peak per channel
print(f"Channels with zero variance (z-score impossible): {(sd_all == 0).sum()}")
print(f"Z-score peak, median: {np.nanmedian(z_peak[silent]):.1f} in almost silent channels, "
      f"{np.median(z_peak[ACTIVE]):.1f} in active channels")

# %% [markdown]
# In almost silent channels the rare events produce higher z-score peaks than in active channels
# (median 9.8 versus 6.3): a few isolated points, far from the mean, the kind of point that PCA
# tends to chase. One channel, moreover, cannot be normalized at all. Channel selection and
# z-score therefore go together.
#
# **Alternative.** The *soft normalization* of Churchland et al. (2012) divides by (channel
# range + 5 Hz): a middle way that reduces the weight of very active channels without amplifying
# weak ones. We keep the z-score for consistency with notebook 02.

# %% [markdown]
# ## 6 · PCA
#
# ### What it computes
#
# Call **Z** the T × N matrix of normalized data (T = bins, N = 160 channels).
#
# 1. **Covariance matrix** between channels: C = Zᵀ Z / (T − 1), of size N × N. The element
#    C[i, j] says how much channels i and j vary together. Since Z is z-scored, C is the
#    correlation matrix.
# 2. **Eigenvectors and eigenvalues** of C: C = W Λ Wᵀ.
#    - Each column of **W** is a **principal direction**: a combination of the 160 channels, i.e.
#      a "mode" of population co-activation.
#    - Each eigenvalue λₖ is the **variance** of the data along that direction. The components are
#      ordered from the largest variance to the smallest, and are orthogonal to each other.
# 3. **Projection**: Y = Z W. Row t of Y says where the population is, at time t,
#    along each direction. The first columns of Y are the **latent trajectory**.
#
# From λ we derive the dimensionality measures:
#
# - **explained variance** of component k: λₖ / Σλ;
# - **participation ratio**: PR = (Σλ)² / Σλ². It equals 1 if all the variance lies in one direction and
#   N if it is spread uniformly over all of them: an "effective number of dimensions".

# %%
pca = PCA()                                             # without n_components: computes all 160 components (full spectrum)
scores = pca.fit_transform(Z)                           # fit = estimates the directions; transform = projects each bin -> T × 160 matrix
W = pca.components_.T                                   # principal directions: one column per component, 160 weights each
lam = pca.explained_variance_                           # variance along each direction (the eigenvalues), in decreasing order
evr = pca.explained_variance_ratio_                     # the same variance, as a fraction of the total
cum = np.cumsum(evr)                                    # cumulative variance of the first k components

print(f"Scores: {scores.shape}   Directions W: {W.shape}")
print(f"Sum of eigenvalues = {lam.sum():.1f}  (= number of channels: after the z-score each carries variance 1)")

# %% [markdown]
# **Check: scikit-learn's PCA really is this algebra.** We redo the computation by hand with
# `numpy` and compare the results.

# %%
C = np.cov(Z, rowvar=False)                             # 160 × 160 covariance between the columns (the channels) of Z
eigval, eigvec = np.linalg.eigh(C)                      # eigenvalues and eigenvectors of a symmetric matrix (in increasing order)
order_ = np.argsort(eigval)[::-1]                       # indices to put them in decreasing order
eigval, eigvec = eigval[order_], eigvec[:, order_]      # reordering: first the direction with maximum variance
Y = Z @ eigvec[:, :3]                                   # manual projection onto the first 3 directions

print("Eigenvalues identical to pca.explained_variance_:", np.allclose(eigval, lam))
print("Correlation (absolute value) between manual projection and sklearn scores, PC1-3:",
      [round(abs(np.corrcoef(Y[:, k], scores[:, k])[0, 1]), 6) for k in range(3)])

# %% [markdown]
# They coincide. The correlation is taken in absolute value because an eigenvector is defined **up to
# sign**: w and −w describe the same direction. The axis of a PC may therefore come out "flipped" from
# one computation to the next, and the sign of PCs should not be interpreted.
#
# ### How many dimensions?

# %%
PR = lam.sum() ** 2 / (lam ** 2).sum()                  # participation ratio
n80 = int(np.searchsorted(cum, 0.80) + 1)               # first k with cumulative variance >= 80%
n90 = int(np.searchsorted(cum, 0.90) + 1)               # first k with cumulative variance >= 90%
print(f"Participation ratio: {PR:.1f}")
print(f"PC1: {evr[0] * 100:.1f}%   PC1-3: {cum[2] * 100:.1f}%   "
      f"components for 80%: {n80}   for 90%: {n90}   (out of {Z.shape[1]})")

fig, ax = plt.subplots(1, 2, figsize=(12, 3.8))
ax[0].plot(np.arange(1, 31), evr[:30] * 100, "o-", color="C0", ms=4, lw=1.5)
ax[0].set(xlabel="component", ylabel="explained variance (%)", title="Scree plot (first 30 components)")
ax[1].plot(np.arange(1, len(cum) + 1), cum * 100, color="C0", lw=2)
for q, n, ls in [(80, n80, "--"), (90, n90, ":")]:
    ax[1].axhline(q, color="0.6", ls=ls, lw=1)
    ax[1].annotate(f"{q}% with {n} PCs", (n, q), xytext=(n - 60, q + 4), fontsize=9,
                   arrowprops=dict(arrowstyle="-", color="0.5"))
ax[1].set(xlabel="number of components", ylabel="cumulative variance (%)",
          title="Cumulative variance", ylim=(0, 101), xlim=(0, Z.shape[1]))
fig.tight_layout(); fig.savefig(FIGS / "02new_pca_scree.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# **Reading.** The variance is **spread over a great many directions**: the first 3 components
# explain less than 9% of it and about 100 out of 160 are needed to reach 80%. The participation ratio
# (~106) is the same as in notebook 02.
#
# This does not mean that motor activity is 100-dimensional. At 20 ms, with 0.28 events per bin,
# each channel is dominated by **Poisson noise**, which is independent across channels and therefore
# spreads uniformly over all directions. PCA maximizes the *total* variance and cannot
# tell this noise from shared signal. There is still a signal, however: the first components
# explain more than average (1/160 ≈ 0.6% each).

# %% [markdown]
# ## 6.1 · Sensitivity analysis: how much does it depend on σ?
#
# σ is our choice, and the participation ratio is a number we will report in the thesis. Before
# trusting "PR ≈ 106" we must know **how much it would change with a different σ**. We therefore repeat
# the whole pipeline (smoothing → z-score → PCA) with six values of σ, from no smoothing to
# 200 ms, keeping everything else fixed: same session, same 160 channels, 20 ms bins.

# %%
SIGMAS = (0, 10, 20, 50, 100, 200)                      # sigmas tried, in ms; 0 = no smoothing (raw counts)
sens = {}                                               # for each sigma, the full spectrum of explained variance
for sig in SIGMAS:                                      # one full pass of the pipeline for each sigma
    X = X_counts.astype(float)                          # always start from the counts of the active channels
    if sig > 0:                                         # with sigma = 0 smoothing is skipped
        X = gaussian_filter1d(X, sig / 1000 / BIN_S, axis=0, mode="nearest")  # smoothing with the sigma of this pass
    sens[sig] = PCA().fit(StandardScaler().fit_transform(X)).explained_variance_ratio_  # z-score + PCA: variance fractions

def pr(v):                                              # participation ratio; unchanged if the variances are given as fractions
    return v.sum() ** 2 / (v ** 2).sum()

tab_sigma = pd.DataFrame({                              # one row per sigma, one column per measure
    "participation ratio": [pr(sens[g]) for g in SIGMAS],
    "var. PC1 (%)": [sens[g][0] * 100 for g in SIGMAS],
    "var. PC1-3 (%)": [sens[g][:3].sum() * 100 for g in SIGMAS],
    "PCs for 80%": [int(np.searchsorted(np.cumsum(sens[g]), 0.80) + 1) for g in SIGMAS],
    "PCs for 90%": [int(np.searchsorted(np.cumsum(sens[g]), 0.90) + 1) for g in SIGMAS],
}, index=pd.Index(SIGMAS, name="sigma (ms)")).round(1)
display(tab_sigma)

shades = plt.cm.Blues(np.linspace(0.35, 1.0, len(SIGMAS)))  # a single colour: light = small sigma, dark = large sigma
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.2))
for c, g in zip(shades, SIGMAS):                        # cumulative-variance curve for each sigma
    lab = f"σ = {g} ms" + (" (raw counts)" if g == 0 else "") + ("  ← chosen" if g == SIGMA_MS else "")
    ax[0].plot(np.arange(1, len(sens[g]) + 1), np.cumsum(sens[g]) * 100, color=c,
               lw=2.8 if g == SIGMA_MS else 1.4, label=lab)  # the chosen sigma is the thickest line
ax[0].axhline(80, color="0.6", ls="--", lw=1)
ax[0].set(xlabel="number of components", ylabel="cumulative variance (%)",
          title="Cumulative variance of PCA as σ varies", xlim=(0, Z.shape[1]), ylim=(0, 101))
ax[0].legend(fontsize=8, loc="lower right")

prs = tab_sigma["participation ratio"].to_numpy()       # the PRs of the table, in the order of SIGMAS
ax[1].plot(SIGMAS, prs, "o-", color="C0", lw=1.5, ms=6)
ax[1].plot(SIGMA_MS, pr(sens[SIGMA_MS]), "o", color="C3", ms=11, zorder=3, label=f"chosen σ ({SIGMA_MS} ms)")
for g, y in zip(SIGMAS, prs):
    if g in (0, SIGMA_MS, SIGMAS[-1]):                  # labels only at the extremes and at the chosen value
        ax[1].annotate(f"{y:.1f}", (g, y), xytext=(7, 7), textcoords="offset points", fontsize=9)
ax[1].set(xlabel="smoothing σ (ms)", ylabel="participation ratio",
          title="The participation ratio falls with σ, without levelling off", ylim=(0, 175))
ax[1].legend(loc="upper right")
fig.tight_layout(); fig.savefig(FIGS / "02new_sigma_sensitivity.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# **Reading.**
#
# - The participation ratio **falls continuously** as σ increases: from ~150 on the raw counts
#   to ~65 with σ = 200 ms, less than half. Smoothing removes noise that is independent across
#   channels and lets the common part emerge. σ = 10 ms gives almost the same result as the raw
#   counts: it is half a bin, and the kernel mixes neighbouring bins very little.
# - **There is no plateau**, i.e. no value of σ beyond which the dimensionality stops
#   changing and one could say "this is the true dimensionality". The PR therefore **is not useful for choosing
#   σ**. σ = 50 ms is chosen for another reason: the time scale of the signal (motor modulations
#   last hundreds of ms, §4). A criterion based on behaviour, i.e. how well the
#   components predict the kinematics as σ varies, will come with the decoder.
# - **The qualitative conclusion holds, however**: with any σ many tens of components are needed
#   to reach 80% of the variance. That single-trial activity is high-dimensional and dominated
#   by noise does not depend on σ; the exact number does.
#
# This is why notebook 02 moved on to FA and GPFA, which separate signal and noise with a statistical
# model rather than with a filter chosen by us.
#
# > **Rule for the thesis.** A dimensionality number obtained with PCA must always be reported
# > together with the preprocessing that produced it, for example: *"PR ≈ 106 (20 ms bins, Gaussian
# > smoothing σ = 50 ms, z-score, 160 channels, one session)"*. Without σ the number cannot be
# > compared with any other.

# %% [markdown]
# ## 7 · What the first components show
#
# We project one trial onto the first 3 PCs and colour the background with the task phases. We keep all the
# bins, including `Presentation` and `Intertrial`: `eval_mask` concerns the kinematics ("Neural
# data is not affected", says the file), whereas here we are analysing only neural activity.

# %%
def base_tag(tag):                                      # "Presentation3" -> "Presentation": a single colour for all presentations
    return "Presentation" if tag.startswith("Presentation") else tag

a, b = io.trial_slices(s)[0]                            # start and end bins of the first trial
t = np.arange(a, b) * BIN_S                             # time of each bin of the trial, in seconds
ep = s.epochs[(s.epochs.stop_time > t[0]) & (s.epochs.start_time < t[-1])]  # phases falling within the trial

fig, ax = plt.subplots(3, 1, figsize=(12, 6.2), sharex=True)
for k in range(3):                                      # one panel per component
    for _, e in ep.iterrows():                          # background coloured by phase
        ax[k].axvspan(e.start_time, e.stop_time, color=io.PHASE_COLORS[base_tag(e.tag)],
                      alpha=0.10 if base_tag(e.tag) in ("Presentation", "Intertrial") else 0.28, lw=0)
    ax[k].plot(t, scores[a:b, k], color="0.15", lw=1.2)  # the trajectory on PC k
    ax[k].set_ylabel(f"PC{k + 1}")
ax[-1].set_xlabel("time in the trial (s)")
ax[0].set_title("First trial projected onto the first 3 principal components (background = task phase)")
handles = [Patch(color=io.PHASE_COLORS[p], alpha=0.5, label=p) for p in io.MOVEMENT_PHASES + ("Presentation", "Intertrial")]
fig.legend(handles=handles, ncol=10, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(FIGS / "02new_pca_timecourse.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# **Reading.** Some variations follow the phases: PC3 has a clear excursion during `Orient` and
# `Grasp`, PC2 during `Carry` and `Orient2`. On top of this, however, there is a lot of bin-by-bin noise, the
# visual counterpart of the participation ratio ~106, and from a single trial one cannot tell how much these
# excursions repeat from one trial to the next. With the task phases as repeated
# conditions we can do better by averaging the repetitions (PSTH) or by using a model of the noise (FA)
# and of time (GPFA). These are the next steps.

# %% [markdown]
# ## 8 · Is there a signal? PCA on condition means
#
# In §6 PCA found ~106 dimensions, and we said that they are mostly noise. Here we
# check this.
#
# **The idea.** In every trial the participant repeats the same sequence of phases, so in one
# session each phase is repeated about 15 times. Each repetition contains:
#
# - the **signal**, i.e. what the brain does for that movement: it is (almost) the same every time;
# - the **noise**, which instead is different every time.
#
# If we take the **mean of the repetitions**, the noise cancels and the signal remains, as when
# one overlays 15 grainy photos of the same scene. We then run PCA on the means.
#
# For each movement phase we take the **1 s window starting at the onset of the phase**
# (50 bins). We use the data of §5 (smoothed at σ = 50 ms and z-scored), so that the comparison with §6 is
# direct.
#
# **Why 1 s.** The window has a fixed duration, the same for all phases and all repetitions,
# for two reasons:
#
# 1. **It must contain the whole neural response.** In notebook 01 (§7, population PSTH
#    aligned to phase onset) the response of the most modulated phases, the hand ones, rises
#    immediately, peaks at about 250 ms and returns towards baseline at about 0.7-0.8 s (`Shape`,
#    `Release`; `Grasp` stays high until about 0.6-0.7 s). A 1 s window contains it
#    entirely; a 0.5 s one would cut off its second half.
# 2. **To average the repetitions, the windows must have the same length** (all
#    50 bins). This explains why the duration is fixed, not why it is 1 s.
#
# **What the window contains beyond the phase.** Half of the phases last less than 1 s (`SnapTo` about
# 0.4 s, `Shape` 0.6 s, `Grasp` and `Release` 0.8 s), the other half more (`Reach`, `Orient`, `Carry`,
# `Orient2`, 1.1-1.5 s). After every movement there is always a `Presentation`, i.e. the cue of the
# next movement with the arm at rest: the window of a short phase spills over into that cue, not
# into the next movement. Since the sequence of phases is always the same, the extra stretch is also the
# same in every repetition.
#
# 1 s remains our choice nonetheless: in §10.7 we redo the main measures with windows of
# 0.5 and 0.75 s, and also without the repetitions in which the phase lasts a single bin.
#
# **There are 16 conditions, not 8: phase × task version.** The task alternates two versions between
# odd and even trials (we found this in notebook 03, §9.2): for example the degree of
# freedom g3 of the grasp goes in opposite directions, in `Shape` it falls in odd trials and rises in
# even ones. The roughly 15 repetitions of a phase **are therefore not replicates of the same movement**:
# they are about 7-8 repetitions of one version and 7-8 of the other. Averaging them all together mixes two
# different responses. For this reason a **condition** is a (phase, version) pair, with the version given
# by the number of the trial in which the window starts (odd or even): 8 phases × 2 versions = 16
# conditions, with about 7 repetitions each.

# %%
WIN_S = 1.0                                             # duration of the window after the onset of each phase, in s
L = int(round(WIN_S / BIN_S))                           # the same duration in bins: 1 s / 20 ms = 50
windows = io.phase_windows(s, win_s=WIN_S, align="start")  # one window for each repetition of each movement phase
VERSIONS = ("odd", "even")                          # the two task versions: odd trials and even trials

def cond_of(w):                                         # the condition of a window: (phase, version)
    return (w["tag"], "odd" if w["trial"] % 2 == 1 else "even")

CONDS = [(ph, v) for ph in io.MOVEMENT_PHASES for v in VERSIONS]  # the 16 conditions, phase by phase
reps = {c: np.stack([Z[w["start"]:w["stop"]]            # for each condition, its windows stacked on top of each other:
                     for w in windows if cond_of(w) == c])  # array (repetitions, 50 bins, 160 channels)
        for c in CONDS}
print("Repetitions per condition:", {f"{ph} {v}": r.shape[0] for (ph, v), r in reps.items()})

PSTH = np.concatenate([reps[c].mean(axis=0)             # mean over repetitions (axis 0): one "mean" window per condition
                       for c in CONDS])                 # the 16 conditions one after the other: matrix (16 × 50, 160) = (800, 160)
pca_avg = PCA().fit(PSTH)                               # PCA on the means
lam_avg = pca_avg.explained_variance_                   # the variances along its directions
print(f"PCA on condition means: {PSTH.shape[0]} × {PSTH.shape[1]} matrix, "
      f"participation ratio = {pr(lam_avg):.1f}, PC1-3 = {lam_avg[:3].sum() / lam_avg.sum() * 100:.0f}% of the variance")

# %% [markdown]
# ### Caution: the right comparison is not 106 versus 11
#
# The two PCAs do not have the same number of points: in §6 there were 12233 bins, here 800 (16 conditions ×
# 50 bins). With few points PCA cannot "fill" many dimensions, and the participation ratio
# drops even without any averaging. To separate the two effects we **always keep 800 points** and
# change only **how many repetitions we average**:
#
# - **phase-aligned windows**: for each condition, the mean of *n* randomly chosen repetitions;
# - **control with random windows**: same procedure, but with windows taken at random points of the
#   session, not aligned to any phase. Here there is no common signal to bring out.
#
# Each random draw is repeated 20 times and the results are averaged.

# %%
rng = np.random.default_rng(0)                          # fixed seed: the random draws are reproducible
N_REP = min(r.shape[0] for r in reps.values())          # repetitions available in all conditions (the one with the fewest)
N_DRAWS = 20                                            # how many random draws for each value of n
N_LIST = sorted({1, 2, 4, N_REP})                       # how many repetitions to average

def pr_aligned(n):                                      # mean PR when averaging n true repetitions of each condition
    out = []
    for _ in range(N_DRAWS):
        M = np.concatenate([reps[c][rng.choice(len(reps[c]), n, replace=False)].mean(axis=0)  # n random repetitions, averaged
                            for c in CONDS])                                                 # the 16 conditions in a row: (800, 160)
        out.append(pr(PCA().fit(M).explained_variance_))
    return np.mean(out)

def pr_random(n):                                       # same thing, but with windows taken at random points of the session
    out = []
    for _ in range(N_DRAWS):
        M = np.concatenate([np.mean([Z[a:a + L] for a in rng.integers(0, n_bins - L, n)], axis=0)  # n random windows, averaged
                            for _ in CONDS])                                                       # 16 "fake conditions": (800, 160)
        out.append(pr(PCA().fit(M).explained_variance_))
    return np.mean(out)

pr_al = [pr_aligned(n) for n in N_LIST]                 # one PR for each n, aligned windows
pr_rd = [pr_random(n) for n in N_LIST]                  # one PR for each n, random windows
display(pd.DataFrame({"phase-aligned windows": pr_al, "random windows (control)": pr_rd},
                     index=pd.Index(N_LIST, name="repetitions averaged")).round(1).T)

# %%
fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.4), gridspec_kw={"width_ratios": [1.15, 1, 1]})
ax[0].plot(N_LIST, pr_al, "o-", color="C0", lw=2, ms=7, label="phase-aligned windows")
ax[0].plot(N_LIST, pr_rd, "s--", color="0.55", lw=1.5, ms=7, label="random windows (control)")
ax[0].set(xlabel="repetitions averaged", ylabel="participation ratio (800 points)", xticks=N_LIST, ylim=(0, None),
          title="Averaging lowers the PR only if the windows are aligned")
ax[0].legend(loc="lower left")

VER_LS = {"odd": "-", "even": "--"}                 # solid line = odd trials, dashed = even trials
one = {c: pca_avg.transform(reps[c][0])[:, :2] for c in CONDS}        # the first repetition of each condition
avg = {c: pca_avg.transform(reps[c].mean(axis=0))[:, :2] for c in CONDS}  # the mean of each condition
for k, (D, ttl) in enumerate([(one, "A single repetition per condition"),
                              (avg, f"Mean of the repetitions ({N_REP}-{max(len(r) for r in reps.values())} per condition)")], start=1):
    for (ph, v) in CONDS:
        ax[k].plot(D[(ph, v)][:, 0], D[(ph, v)][:, 1], color=io.PHASE_COLORS[ph], ls=VER_LS[v], lw=1.6)  # trajectory in the PC1-PC2 plane
        ax[k].plot(D[(ph, v)][0, 0], D[(ph, v)][0, 1], "o", color=io.PHASE_COLORS[ph], ms=6)  # dot = start of the phase
    ax[k].set(xlabel="PC1", ylabel="PC2", title=ttl)
lims = np.array([[*ax[k].get_xlim(), *ax[k].get_ylim()] for k in (1, 2)])  # the limits each panel would choose on its own
for k in (1, 2):                                        # same scale in the two panels, wide enough for all the trajectories
    ax[k].set_xlim(lims[:, 0].min(), lims[:, 1].max()); ax[k].set_ylim(lims[:, 2].min(), lims[:, 3].max())
handles = ([plt.Line2D([], [], color=io.PHASE_COLORS[ph], lw=2, label=ph) for ph in io.MOVEMENT_PHASES]
           + [plt.Line2D([], [], color="0.3", ls=VER_LS[v], lw=1.6, label=f"trial {v}") for v in VERSIONS])
fig.legend(handles=handles, ncol=10, fontsize=8, loc="lower center", bbox_to_anchor=(0.62, -0.02))
fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(FIGS / "02new_pca_phase_averages.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# **Reading.**
#
# - **At equal number of points** (800), a single repetition per condition gives a participation ratio of
#   about 50. Averaging the 7 **aligned** repetitions available in every condition brings it down to about
#   21, and the curve has not yet flattened: with more repetitions it would fall further. ~20 is therefore an
#   upper bound, not the final value.
# - With **random windows** the PR stays around 50 whatever n is. Averaging pure noise still gives
#   noise, only smaller, spread over the same directions. The drop for aligned windows
#   therefore comes from what the repetitions **have in common**: the task signal.
# - **Why ~20 and not ~11, as when mixing the two versions.** With 8 conditions (the phases, without
#   distinguishing the version) 14-15 repetitions were averaged over 400 points; with 16 conditions 7-8
#   are averaged over 800 points. Fewer repetitions leave more noise, and more points give PCA
#   more room to count it. The higher number does not mean that the signal is more complex: it is
#   the price of not mixing two different movements. The drop with averaging is about 59% (§10.7).
# - In the figures: a single repetition gives tangled, overlapping trajectories; the means give
#   smooth trajectories, separated by condition (solid line = odd trials, dashed =
#   even trials). (The axes are those of the PCA on the means: they are fine for looking, not for
#   measuring how good a single trajectory is, because that PCA has already "seen" the conditions.)
#
# **The signal is there and it is simple, but in a single trial it is buried in noise.** In practice, though, averaging
# cannot be done: a brain-computer interface must understand the movement **while** the
# person attempts it, just once. We need a method that separates signal and noise **without
# averaging**. This is the job of Factor Analysis.

# %% [markdown]
# ## 9 · Factor Analysis: separating the shared signal from the private noise of each channel
#
# ### The idea
#
# FA describes each channel as the sum of two parts:
#
# - a **shared part**: a combination of a few *factors* common to the whole population. This is the
#   signal we are interested in;
# - a **private part**: the noise of the individual channel, with an amplitude that differs from channel to channel.
#
# As in a choir: FA separates the melody that everyone sings together from the off-key notes of each
# singer.
#
# In formulas, for each bin t:
#
# $$x(t) = W\,z(t) + \mu + \varepsilon(t), \qquad \varepsilon(t) \sim \mathcal{N}(0, \Psi)$$
#
# - $x(t)$: the 160 channels; $z(t)$: the K shared factors, with K much smaller than 160;
# - $W$ (160 × K): how much each channel "listens" to each factor; $\mu$: the mean of each channel;
# - $\varepsilon(t)$: the private noise. $\Psi$ is diagonal: **a different noise variance for
#   each channel**, and no noise in common between channels.
#
# **The difference from PCA lies entirely in Ψ.** PCA puts all the variance in the same
# pot. FA estimates for each channel how much of its variance is private noise and sets it
# aside: what remains, $W W^\top$, is the shared structure.
#
# ### Three choices, and why
#
# 1. **Raw counts at 20 ms, without smoothing.** The job of FA is to separate noise from
#    signal: if we smooth first, we remove part of the noise ourselves with a hand-chosen filter and
#    FA does not see it. GPFA, the next step, also works on raw counts. We measure the effect of
#    smoothing anyway in §9.5.
# 2. **The z-score is not necessary here.** FA estimates a different noise for each channel, so it
#    adapts by itself to the scale of each. We apply it only for numerical convenience, and in §9.2
#    we check that the results do not change.
# 3. **The number of factors K is chosen by cross-validation** (§9.1).
#
# > **Technical note.** scikit-learn's FA, with the default settings, uses an *approximate*
# > SVD: on these data it stops after very few iterations, before reaching the
# > solution. We use `svd_method="lapack"` (exact SVD), which really converges. We check this just
# > below with an arbitrary K (20), only for the comparison.

# %%
X_fa = StandardScaler().fit_transform(X_counts.astype(float))  # raw counts of the 160 active channels, z-scored

fa_default = FactorAnalysis(n_components=20, random_state=0).fit(X_fa)                      # default settings
fa_exact = FactorAnalysis(n_components=20, svd_method="lapack", random_state=0).fit(X_fa)   # exact SVD
print(f"default : {fa_default.n_iter_:3d} iterations, mean log-likelihood = {fa_default.score(X_fa):.3f}")
print(f"exact   : {fa_exact.n_iter_:3d} iterations, mean log-likelihood = {fa_exact.score(X_fa):.3f}")

# %% [markdown]
# The *log-likelihood* measures how well the model explains the data: **higher is better**. The exact
# version reaches a higher value because it keeps improving until it converges; the default one
# stops earlier. From here on we always use `svd_method="lapack"`. (The FA of notebook 02
# also used the default settings: its numbers should be taken with caution.)
#
# ### 9.1 · How many factors? Cross-validation
#
# With too few factors FA loses signal; with too many it starts to "memorize" the noise
# of the data on which it is fitted. To find the middle way:
#
# 1. we split the session into 5 blocks;
# 2. we fit FA on 4 blocks and measure the log-likelihood on the fifth, which the model has
#    never seen;
# 3. we repeat leaving out each block in turn, and take the mean.
#
# The K with the highest score is the one that **works best on new data**. The blocks are **contiguous**
# pieces of time, not randomly shuffled bins: two neighbouring bins resemble each other, and if one ended up
# in the fit and the other in the test, the test would be too easy.
#
# That makes 30 FA fits (6 values of K × 5 blocks): we run them in parallel on the computer's
# cores. It takes a few minutes.

# %%
K_GRID = [2, 5, 10, 15, 20, 30]                         # values of K to try
folds = list(KFold(n_splits=5, shuffle=False).split(X_fa))  # 5 contiguous blocks: shuffle=False = do not shuffle the bins

def cv_score(X, K, train, test):                        # one fit: FA on "train", score on "test"
    fa_ = FactorAnalysis(n_components=K, svd_method="lapack", random_state=0).fit(X[train])
    return fa_.score(X[test])                           # mean log-likelihood per bin on unseen data

scores_cv = Parallel(n_jobs=-1)(delayed(cv_score)(X_fa, K, tr, te)   # all (K, block) combinations in parallel
                                for K in K_GRID for tr, te in folds)
cv_ll = np.array(scores_cv).reshape(len(K_GRID), len(folds)).mean(axis=1)  # one row per K, one column per block -> mean over blocks
K_FA = K_GRID[int(np.argmax(cv_ll))]                    # the K with the highest score
print("Log-likelihood on unseen data:", dict(zip(K_GRID, cv_ll.round(3))))
print(f"-> K* = {K_FA} factors")

fig, ax = plt.subplots(figsize=(6.5, 3.8))
ax.plot(K_GRID, cv_ll, "o-", color="C0", lw=2, ms=7)
ax.axvline(K_FA, color="C3", ls="--", lw=1.2, label=f"K* = {K_FA}")
ax.set(xlabel="number of factors K", ylabel="log-likelihood on unseen data",
       title="FA: choosing K by cross-validation")
ax.legend()
fig.tight_layout(); fig.savefig(FIGS / "02new_fa_cv.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# **Reading.** The curve rises as long as the added factors capture signal, then falls when they
# start to chase the noise. Around the maximum it is flat: 10, 15 and 20 factors give almost the
# same score. K* should be read as "between 10 and 20", not as an exact number.
#
# ### 9.2 · How much variance is shared?
#
# We fit the final FA with K* factors on the whole session. For each channel:
#
# - **shared variance** = the sum of its squared weights in the matrix W;
# - **private variance** = its element of Ψ;
# - **shared fraction** = shared / (shared + private).

# %%
fa = FactorAnalysis(n_components=K_FA, svd_method="lapack", random_state=0).fit(X_fa)  # final FA, on the whole session
W_fa = fa.components_.T                                 # weights: one row per channel, one column per factor (160 × K)
psi = fa.noise_variance_                                # private variance of each channel (the diagonal of Ψ)
shared = (W_fa ** 2).sum(axis=1)                        # shared variance of each channel: sum of its squared weights
frac_shared = shared / (shared + psi)                   # shared fraction of each channel
print(f"Shared fraction per channel: median {np.median(frac_shared):.3f}, "
      f"5th-95th percentile {np.percentile(frac_shared, 5):.3f}-{np.percentile(frac_shared, 95):.3f}, "
      f"maximum {frac_shared.max():.2f}")

fa_noz = FactorAnalysis(n_components=K_FA, svd_method="lapack", random_state=0).fit(X_counts.astype(float))  # same FA, without z-score
sh_noz = (fa_noz.components_ ** 2).sum(axis=0)          # shared variance without z-score
frac_noz = sh_noz / (sh_noz + fa_noz.noise_variance_)   # shared fraction without z-score
print(f"With and without z-score, maximum difference in the shared fraction: {np.abs(frac_noz - frac_shared).max():.4f}")

fig, ax = plt.subplots(figsize=(7.5, 3.8))
ax.hist(frac_shared * 100, bins=30, color="C0")
ax.axvline(np.median(frac_shared) * 100, color="C3", ls="--", lw=1.2,
           label=f"median {np.median(frac_shared) * 100:.1f}%")
ax.set(xlabel="variance shared with the population (% of the channel variance)", ylabel="number of channels",
       title=f"FA with {K_FA} factors, raw counts at 20 ms")
ax.legend()
fig.tight_layout(); fig.savefig(FIGS / "02new_fa_shared.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# **Reading.** In a typical channel only a small part of the variance, a few percentage points,
# is shared with the rest of the population: **the rest is private noise**. This explains the
# very high participation ratio of PCA (§6): PCA counted this noise too, FA sets it
# aside. The shared fraction varies a lot from channel to channel: some channels are much more tied
# to the population than others.
#
# With and without z-score the shared fraction is the same: as expected, FA does not depend on the scale
# of the channels.
#
# ### 9.3 · How many dimensions does the shared part have?
#
# The K* factors do not all weigh the same. We look at the shared structure $W W^\top$: its
# eigenvalues say how much shared variance lies along each direction. To compare it with
# PCA we use **the same two measures for all methods**:
#
# - **participation ratio** (as in §6): the "effective number" of dimensions. It looks mostly at the
#   large directions;
# - **dimensions for 95%**: how many directions are needed to reach 95% of the variance. It also counts
#   the small directions, so it is more sensitive to noise. For FA, applied to the shared
#   part, it is the *shared dimensionality* of Williamson et al. (2016).

# %%
eig_sh = np.linalg.eigvalsh(W_fa @ W_fa.T)[::-1][:K_FA]  # variances of the shared part, from the largest (the others are 0)
cum_sh = np.cumsum(eig_sh) / eig_sh.sum()               # cumulative shared variance

def n95(v):                                             # how many directions are needed for 95% of a variance
    return int(np.searchsorted(np.cumsum(v) / v.sum(), 0.95) + 1)

d_shared = n95(eig_sh)                                  # shared dimensionality: directions for 95% of the shared part
pr_shared = pr(eig_sh)                                  # participation ratio of the shared part: the same measure as for PCA
print(f"Shared part: participation ratio {pr_shared:.1f}, dimensions for 95%: {d_shared}, "
      f"first 3 directions: {cum_sh[2] * 100:.0f}% of the shared variance")

display(pd.DataFrame(
    [["PCA on all bins (σ = 50 ms)", "signal + all the noise", PR, n95(lam)],
     ["PCA on condition means", "signal + noise left after averaging", pr(lam_avg), n95(lam_avg)],
     [f"FA on all bins, raw counts (K* = {K_FA})", "only the part shared across channels", pr_shared, d_shared]],
    columns=["method", "what it counts", "participation ratio", "dimensions for 95%"]).set_index("method").round(1))

# %% [markdown]
# **Reading.**
#
# - **With the same measure (participation ratio)** FA on single bins finds about 9.5 effective
#   dimensions, **without any averaging**: fewer than the condition means (~20) and far below
#   the ~106 of PCA. The means of 7-8 repetitions still contain a lot of noise (§8), which FA
#   sets aside.
# - **With the dimensions for 95%** the difference is even sharper: PCA on the means needs
#   78, FA 14. The noise left after averaging, although small, is spread over many
#   directions, and the last percentages of variance are almost all noise.
#
# Note: here FA uses the whole session (including the pauses between movements), the
# means only the first second of each phase. The like-for-like comparison, on the same instants, is in §9.6.
#
# ### 9.4 · Are single-trial trajectories cleaner?
#
# FA tells us **how much** shared signal there is. But are the trajectories of a single trial
# better than those of PCA? We use the 1 s windows of §8 and the first 3 latent dimensions,
# with two measures:
#
# - **roughness**: how much the trajectory jumps from one bin to the next. Lower =
#   smoother;
# - **consistency**: how similar the repetitions of the same condition are, i.e. the same phase
#   **and** the same task version (§8): mean correlation between pairs of repetitions, computed
#   within each of the 16 conditions and then averaged. Higher = better: repetitions of the same
#   movement should resemble each other. Repetitions of different versions are not compared, because the
#   movement is different.
#
# For a fair comparison, PCA and FA receive **the same data** (raw counts). We add PCA and
# FA on the smoothed data (σ = 50 ms) as a reference, to see how much smoothing matters.
#
# FA factors have no natural order: they can be rotated without changing the model. We
# rotate them so that they are **ordered like the PCA components**, from the most important (with the
# SVD of W, as GPFA will do).

# %%
def orth_latents(fa_, X):                               # FA factors rotated and ordered from the most important
    U_, S_, Vt_ = np.linalg.svd(fa_.components_.T, full_matrices=False)  # W = U · S · Vt
    return fa_.transform(X) @ Vt_.T * S_                # coordinates along the U directions, ordered by importance (S)

def roughness(trajs):                                   # mean roughness of a list of trajectories (bins × dimensions)
    return float(np.mean([(np.diff((T_ - T_.mean(0)) / (T_.std(0) + 1e-9), n=2, axis=0) ** 2).mean()
                          for T_ in trajs]))            # z-score of each dimension, then squared second difference

def consistency_w(trajs, win_list, per_phase=False):    # mean correlation between repetitions of the same condition (phase, version)
    conds_ = [cond_of(w) for w in win_list]
    out = {}
    for c in CONDS:
        R = np.stack([tr for tr, cw in zip(trajs, conds_) if cw == c])  # (repetitions, bins, 3 dimensions)
        out[c] = np.mean([np.corrcoef(R[:, :, j])[np.triu_indices(len(R), 1)].mean()  # all pairs of repetitions
                          for j in range(R.shape[2])])                              # for each dimension, then mean
    if per_phase:                                       # per phase: mean of the two versions
        return {ph: (out[(ph, "odd")] + out[(ph, "even")]) / 2 for ph in io.MOVEMENT_PHASES}
    return float(np.mean(list(out.values())))           # mean over the 16 conditions

def consistency(trajs):                                 # consistency on the 1 s windows of §8
    return consistency_w(trajs, windows)

fa_sm = FactorAnalysis(n_components=K_FA, svd_method="lapack", random_state=0).fit(Z)  # same FA on the smoothed data of §5
latents = {
    "PCA, raw counts": PCA(3).fit_transform(X_fa),
    f"FA, raw counts (K = {K_FA})": orth_latents(fa, X_fa)[:, :3],
    "PCA, smoothed σ = 50 ms": scores[:, :3],
    f"FA, smoothed σ = 50 ms (K = {K_FA})": orth_latents(fa_sm, Z)[:, :3],
}
rows = []
for name, lat in latents.items():
    trajs = [lat[w["start"]:w["stop"]] for w in windows]  # the 1 s windows, for each method
    rows.append({"method": name, "roughness (lower = better)": roughness(trajs),
                 "consistency (higher = better)": consistency(trajs)})
tab_traj = pd.DataFrame(rows).set_index("method").round(3)
display(tab_traj)

# %% [markdown]
# **Reading.** This is the most instructive result of the section.
#
# - **FA and PCA on raw counts behave almost the same**: equally rough trajectories and
#   repetitions that are equally dissimilar. Yet FA has just shown that it can
#   separate signal and noise.
# - **Smoothing changes everything**: roughness collapses and consistency rises from about 0.035 to
#   about 0.12-0.13, for both PCA and FA.
#
# The reason: FA separates the noise **across channels**, but treats each bin on its own, without knowing that the
# previous bin and the next one are close in time. The noise **over time** all remains. For
# single-trial trajectories we need a **model of time**: this is exactly what GPFA adds to
# FA. Modelling the noise tells us *how many* shared dimensions there are; modelling time lets us
# see *how they move* within a single trial.
#
# ### 9.5 · Sensitivity analysis: what if we smooth before FA?
#
# As for PCA (§6.1), we repeat FA, with its cross-validation, on data smoothed with different
# σ. Alongside we put the participation ratio of PCA with the same σ (from §6.1).
# It takes a few minutes.

# %%
K_GRID_S = [5, 10, 15, 20, 30]                          # grid of K for the smoothed data
SIG_FA = (20, 50, 100)                                  # sigmas tried (0 = raw counts, already done above)
data_s = {g: StandardScaler().fit_transform(            # data smoothed with each sigma, then z-scored
              gaussian_filter1d(X_counts.astype(float), g / 1000 / BIN_S, axis=0, mode="nearest"))
          for g in SIG_FA}
res_s = Parallel(n_jobs=-1)(delayed(cv_score)(data_s[g], K, tr, te)  # cross-validation for each (sigma, K, block)
                            for g in SIG_FA for K in K_GRID_S for tr, te in folds)
res_s = np.array(res_s).reshape(len(SIG_FA), len(K_GRID_S), len(folds)).mean(axis=2)  # mean over blocks -> (sigma, K)

def fa_summary(X, K):                                   # median shared fraction, PR and dimensions for 95% of an FA
    f_ = FactorAnalysis(n_components=K, svd_method="lapack", random_state=0).fit(X)
    W_ = f_.components_.T
    sh_ = (W_ ** 2).sum(axis=1)
    e_ = np.linalg.eigvalsh(W_ @ W_.T)[::-1][:K]
    return np.median(sh_ / (sh_ + f_.noise_variance_)), pr(e_), n95(e_)

rows = [{"sigma (ms)": 0, "FA: K*": K_FA, "FA: shared PR": pr_shared, "FA: dimensions for 95%": d_shared,
         "FA: median shared fraction (%)": np.median(frac_shared) * 100, "PCA: participation ratio": pr(sens[0])}]
for i, g in enumerate(SIG_FA):
    K_g = K_GRID_S[int(np.argmax(res_s[i]))]            # the best K for this sigma
    med_g, pr_g, d_g = fa_summary(data_s[g], K_g)
    rows.append({"sigma (ms)": g, "FA: K*": K_g, "FA: shared PR": pr_g, "FA: dimensions for 95%": d_g,
                 "FA: median shared fraction (%)": med_g * 100, "PCA: participation ratio": pr(sens[g])})
tab_fa_sigma = pd.DataFrame(rows).set_index("sigma (ms)").round(1)
display(tab_fa_sigma)

# %% [markdown]
# **Reading.**
#
# - **The shared fraction rises with σ.** Smoothing mainly removes the private noise, which changes
#   from one bin to the next, while the slower shared signal remains. The shared part therefore
#   becomes a larger slice of the total. No new signal has appeared: we are looking at slower time
#   scales. For this reason the shared fraction must always be reported with σ (or with "raw
#   counts at 20 ms").
# - **The dimensionality of FA changes little**: with any σ the shared PR stays between about
#   9.5 and 13, and the dimensions for 95% between 14 and 18. The participation ratio of PCA, on the same
#   data, goes instead from ~150 to ~85. FA does not count private noise as dimensions, and for
#   this reason its estimate depends much less on preprocessing. This is a good argument for preferring
#   FA to PCA when one wants to say "how many dimensions".

# %% [markdown]
# ### 9.6 · Like-for-like comparison: FA on the same instants as the means
#
# In §9.3 FA used the whole session, including the pauses between movements, whereas
# PCA on the means used only the first second of each phase. Here we redo FA **only on the bins that
# fall in the phase windows**, i.e. at the same instants as PCA on the means, but **without
# averaging**: single bins, raw counts. Alongside we put PCA on the same single bins.
#
# Some windows overlap: a short phase, such as `SnapTo`, lasts less than 1 s and its
# window spills over into the next phase. Bins in common are counted only once. We choose the number of
# factors again with cross-validation, on these bins (5 contiguous blocks, as in
# §9.1).

# %%
in_win = np.zeros(n_bins, dtype=bool)                   # a true/false for each bin of the session, all false at the start
for w in windows:                                       # mark as true the bins of each phase window
    in_win[w["start"]:w["stop"]] = True
X_win = X_fa[in_win]                                    # only the bins in the windows, in time order (raw counts, z-scored)
print(f"{len(windows)} windows × {L} bins = {len(windows) * L} bins, of which {in_win.sum()} distinct")

folds_win = list(KFold(n_splits=5, shuffle=False).split(X_win))  # 5 contiguous blocks of these bins
res_win = Parallel(n_jobs=-1)(delayed(cv_score)(X_win, K, tr, te)  # cross-validation, as in §9.1
                              for K in K_GRID for tr, te in folds_win)
cv_win = np.array(res_win).reshape(len(K_GRID), len(folds_win)).mean(axis=1)  # one score for each K
K_WIN = K_GRID[int(np.argmax(cv_win))]                  # the K with the highest score
print("Log-likelihood on unseen data:", dict(zip(K_GRID, cv_win.round(3))), f"-> K* = {K_WIN}")

fa_win = FactorAnalysis(n_components=K_WIN, svd_method="lapack", random_state=0).fit(X_win)  # final FA on these bins
W_win = fa_win.components_.T                            # weights: 160 channels × K factors
eig_win = np.linalg.eigvalsh(W_win @ W_win.T)[::-1][:K_WIN]  # variances of the shared part, from the largest

lam_win_sm = PCA().fit(Z[in_win]).explained_variance_   # PCA on the same single bins, smoothed (like the means)
lam_win_raw = PCA().fit(X_win).explained_variance_      # PCA on the same single bins, raw (like FA)
display(pd.DataFrame(
    [["PCA on condition means", "mean of 7-8 repetitions, smoothed", pr(lam_avg), n95(lam_avg)],
     ["PCA on single bins (smoothed)", "no averaging, smoothed σ = 50 ms", pr(lam_win_sm), n95(lam_win_sm)],
     ["PCA on single bins (raw)", "no averaging, raw counts", pr(lam_win_raw), n95(lam_win_raw)],
     [f"FA on single bins (K* = {K_WIN})", "no averaging, raw counts", pr(eig_win), n95(eig_win)]],
    columns=["method", "data (always the same instants)", "participation ratio", "dimensions for 95%"]
).set_index("method").round(1))

# %% [markdown]
# **Reading.** This is the cleanest comparison in the notebook: same instants, same measures.
#
# - **Without averaging, PCA sees a great many dimensions** (~92 on the smoothed data, ~146 on the raw):
#   the noise.
# - **With averaging by condition, PCA drops to ~20.**
# - **FA, without any averaging, drops much lower: about 7.** It gets from single bins
#   more than PCA gets by averaging 7-8 repetitions.
#
# Why does FA fall below the means? Noise remains in the means: the average of 7-8 repetitions does not
# cancel it completely (the curve of §8 had not yet flattened), and that noise adds dimensions.
# FA instead sets it aside. This is clearly visible in the 95% column: the means need 78
# directions, FA 9.
#
# Here cross-validation chooses fewer factors than on the whole session (K* = 10 instead of 15). In
# part because the data are about half as many, in part because these instants are less varied: only
# the start of the movement phases, no pauses. Here too the curve is flat around the maximum.
#
# **Summary of FA.** FA solves the problem of *how much*: it separates the shared signal from the
# private noise and, **without any averaging**, finds a structure of about 7-10 effective dimensions,
# more compact than that of the condition means (which retain some noise) and stable
# with respect to σ. It does not solve the problem of *how it evolves
# over time*: single-trial trajectories remain as noisy as those of PCA (§9.4). The next
# step, GPFA, adds precisely the model of time.

# %% [markdown]
# ## 10 · GPFA: adding the model of time
#
# ### The idea
#
# FA (§9) separates the shared signal from the noise of each channel, but looks at **each bin on its own**:
# it does not know that the current bin and the one 20 ms ago are close in time. For this reason its
# trajectories remain rough (§9.4).
#
# **GPFA** (*Gaussian Process Factor Analysis*, Yu et al. 2009) is an FA with one more idea:
# **each factor changes gradually over time**. For each bin the model is the same as FA,
#
# $$y(t) = C\,x(t) + d + \varepsilon(t), \qquad \varepsilon(t) \sim \mathcal{N}(0, R), \ R \text{ diagonal}$$
#
# but in addition each factor $x_i(t)$ follows a *Gaussian process*: two nearby instants have
# similar values, and how close they must be is given by a **time scale** $\tau_i$, different for each
# factor and **learned from the data**:
#
# $$\text{similarity between } x_i(t_1) \text{ and } x_i(t_2) = \exp\!\left(-\frac{(t_1 - t_2)^2}{2\tau_i^2}\right)$$
#
# For example, with τ = 200 ms two instants 200 ms apart are still 60% similar, at
# 600 ms hardly at all.
#
# **What changes in practice.** To estimate the factors at an instant, GPFA uses the 160 counts of
# that instant **and also those of the neighbouring instants**, weighted according to τ and to the noise of
# each channel. It is like doing FA and smoothing together, with one important difference: **how much to
# smooth is learned by GPFA**, for each factor, instead of being chosen by us like the σ of §4.
#
# ### The choices, and why
#
# 1. **The "trials" are the 119 windows of 1 s** from the onset of each phase (§8). GPFA needs
#    sequences that repeat; we use the same windows as PCA on the means and FA in §9.6,
#    so the comparisons are like for like.
# 2. **The same bins as in the rest of the notebook.** `elephant` has its own way of grouping spikes into
#    bins which, because of rounding in the times, does not coincide with ours (we check this below).
#    We therefore give GPFA our own counts directly, window by window, using the
#    computation functions of `elephant` (`gpfa_core`).
# 3. **The square root of the counts**, as in the original paper and as `elephant` does by
#    default. In counts the noise grows with the firing rate (§5); with the square root it becomes similar
#    for slow and fast channels. For a like-for-like comparison, in §10.4 we give the square root of the counts
#    to FA as well.
# 4. **A fixed starting point.** GPFA starts from an FA that has a random element: we fix the
#    seed, so that every run gives the same result.

# %%
import contextlib, logging, os, warnings                # tools to silence the messages from elephant
import quantities as pq                                 # units of measure (ms), required by elephant
from elephant.gpfa import gpfa_core                     # elephant's GPFA computation functions
from elephant.conversion import BinnedSpikeTrain        # elephant's binning, only for the comparison below
import elephant.utils

elephant.utils.logger.setLevel(logging.ERROR)           # elephant prints many harmless warnings: we hide them
warnings.filterwarnings("ignore")
BIN_MS = BIN_S * 1000                                   # bin width in ms (20), as elephant wants it

sts = io.windows_spiketrains(s, windows, units=ACTIVE)  # for each window, the 160 spike trains (times from the start of the window)
diff_bins = sum(int((BinnedSpikeTrain(st, bin_size=BIN_MS * pq.ms).to_array().T  # counts according to elephant...
                     != X_counts[w["start"]:w["stop"]]).sum())                      # ...compared with ours
                for st, w in zip(sts, windows))
print(f"Bins with different counts between elephant's binning and ours: {diff_bins} "
      f"out of {len(windows) * L * len(ACTIVE):,}")

def make_seqs(win_list):                                # the data in the format elephant wants: one row per window
    return np.array([(w["stop"] - w["start"],                     # T: number of bins of the window (50)
                      np.sqrt(X_counts[w["start"]:w["stop"]].T))  # y: square root of the counts, channels × bins
                     for w in win_list], dtype=[("T", int), ("y", "O")])

def fit_gpfa(win_list, d, iters=500, tau_init=100.0, whole=False):  # fits a GPFA with d factors on the given windows
    np.random.seed(0)                                   # fixed starting point (the initial FA has a random part)
    cut_orig = gpfa_core.gpfa_util.cut_trials           # the elephant function that splits the windows into pieces of 20 bins
    if whole:                                           # whole=True: no pieces, the fit uses the whole windows (§10.6)
        gpfa_core.gpfa_util.cut_trials = lambda seqs, seg_length=np.inf: cut_orig(seqs, seg_length=np.inf)
    try:
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):  # hides the progress messages
            params, info = gpfa_core.fit(make_seqs(win_list), x_dim=d, bin_width=BIN_MS,
                                         em_max_iters=iters, tau_init=tau_init)
    finally:
        gpfa_core.gpfa_util.cut_trials = cut_orig       # in any case we restore the original function
    return params, info

def gpfa_ll_per_bin(params, win_list):                  # mean log-likelihood per bin on unseen windows
    _, ll = gpfa_core.exact_inference_with_ll(make_seqs(win_list), params, get_ll=True)
    return ll / sum(w["stop"] - w["start"] for w in win_list)

# %% [markdown]
# The differences are few, but they exist: they are events that fall exactly on the border between two bins, and
# that the two methods assign to different bins. Using our counts, PCA, FA and GPFA work
# on exactly the same data.
#
# ### 10.1 · How many iterations are needed?
#
# GPFA is fitted by successive approximation (the EM algorithm): at each iteration the model improves a
# little and the score (log-likelihood) rises. It stops when it no longer improves or when the
# maximum number of iterations is reached. Notebook 02 stopped at 100-150; the default of
# `elephant` is 500. We check with 10 factors (an arbitrary value, only for the check) and 1000
# iterations. It takes about a minute.

# %%
_, info_conv = fit_gpfa(windows, d=10, iters=1000)      # a GPFA left to run for a long time
ll_conv = np.array(info_conv["log_likelihoods"]) / (len(windows) * L)  # score per bin at each iteration
it_ok = np.flatnonzero(~np.isnan(ll_conv))              # elephant computes the score only every 5 iterations
for k in (100, 500, 1000):
    j = it_ok[it_ok < k].max()                          # the last iteration computed before k
    print(f"after {j + 1:4d} iterations: log-likelihood per bin = {ll_conv[j]:.4f}")

fig, ax = plt.subplots(figsize=(7, 3.6))
ax.plot(it_ok + 1, ll_conv[it_ok], color="C0", lw=2)
for k, lab, c in [(100, "notebook 02 (100)", "0.5"), (500, "chosen here (500)", "C3")]:
    ax.axvline(k, color=c, ls="--", lw=1.2, label=lab)
ax.set(xlabel="iteration", ylabel="log-likelihood per bin", ylim=(ll_conv[it_ok][2], None),
       title="GPFA (10 factors): the score rises as long as the model improves")
ax.legend(loc="lower right")
fig.tight_layout(); fig.savefig(FIGS / "02new_gpfa_convergence.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# **Reading.** After 100 iterations the model is still improving; around 500 the curve is
# now flat, and going up to 1000 changes the score by a few thousandths. We use 500 iterations.
#
# > **A detail of `elephant`.** During the fit, to go faster, `elephant` does not use the
# > whole 1 s windows but **splits them into pieces of 20 bins (400 ms)**. The score in this
# > figure is computed on those pieces, and for this reason it is not comparable with the scores of §10.2, which
# > we compute on the whole windows.
#
# ### 10.2 · How many factors? Cross-validation
#
# As for FA: we split the 119 windows into 5 blocks **contiguous in time**, fit GPFA on
# 4 blocks and score it on the fifth, in turn.
#
# We try many values of d, up to 50, to see whether at some point the score **stops rising and
# starts falling**: this would be the sign of **overfitting**, i.e. of factors that memorize the
# noise of the training data. To see it clearly we give GPFA two scores:
#
# - **on the training data** (the windows used to fit it): it always rises, because more factors are
#   more "knobs" to adapt with;
# - **on unseen data**: it rises as long as the extra factors capture signal, then falls.
#
# Both scores are computed on the whole windows. That makes 45 GPFA fits, in parallel: it is the
# slowest part of the notebook (several minutes).

# %%
D_GRID = [2, 5, 10, 15, 20, 25, 30, 40, 50]             # numbers of factors to try
folds_gp = list(KFold(n_splits=5, shuffle=False).split(np.arange(len(windows))))  # contiguous blocks of windows

def cv_gpfa(d, train, test):                            # one fit: GPFA on the "train" windows, two scores
    params, _ = fit_gpfa([windows[i] for i in train], d)
    return (gpfa_ll_per_bin(params, [windows[i] for i in train]),   # score on the training data
            gpfa_ll_per_bin(params, [windows[i] for i in test]))    # score on unseen data

res_gp = Parallel(n_jobs=-1)(delayed(cv_gpfa)(d, tr, te)  # all (d, block) combinations, in parallel
                             for d in D_GRID for tr, te in folds_gp)
res_folds = np.array(res_gp).reshape(len(D_GRID), len(folds_gp), 2)  # all the scores: (d, block, 2 scores)
ll_study, cv_gp = res_folds.mean(axis=1).T              # mean over blocks: score on the training data, score on unseen data
test_folds = res_folds[:, :, 1]                         # scores on unseen data, block by block: (d, 5 blocks)
D_BEST = D_GRID[int(np.argmax(cv_gp))]                  # the d with the highest mean score on unseen data
best_wins = (test_folds[D_GRID.index(D_BEST)] > test_folds).sum(axis=1)  # in how many blocks D_BEST beats each d
D_GP = min(d for d, w in zip(D_GRID, best_wins)         # the smallest d that D_BEST does NOT beat in all blocks
           if d <= D_BEST and w < len(folds_gp))
display(pd.DataFrame({"score on training data": ll_study, "score on unseen data": cv_gp,
                      f"blocks in which d = {D_BEST} does better": [f"{w} of {len(folds_gp)}" for w in best_wins]},
                     index=pd.Index(D_GRID, name="factors d")).round(3).T)
print(f"Highest mean score: d = {D_BEST}.  Simplest model not distinguishable from the best: d = {D_GP}.")

fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ax[0].plot(D_GRID, ll_study - ll_study[0], "o-", color="0.5", lw=2, ms=6, label="training data")
ax[0].plot(D_GRID, cv_gp - cv_gp[0], "o-", color="C0", lw=2, ms=6, label="unseen data")
ax[0].set(xlabel="number of factors d", ylabel="gain relative to d = 2",
          title="Training data vs unseen data")
ax[0].legend(loc="upper left")
ax[1].plot(D_GRID, cv_gp, "o-", color="C0", lw=2, ms=7)
ax[1].axvline(D_BEST, color="C3", ls="--", lw=1.2, label=f"highest score: d = {D_BEST}")
if D_BEST != D_GP:
    ax[1].axvline(D_GP, color="C2", ls=":", lw=1.5, label=f"chosen: d = {D_GP}")
ax[1].set(xlabel="number of factors d", ylabel="log-likelihood per bin, unseen data",
          title="Unseen data only (zoomed in)")
ax[1].legend(loc="lower right")
fig.tight_layout(); fig.savefig(FIGS / "02new_gpfa_cv.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# **Reading.**
#
# - **The score on the training data always rises** (grey, left): with more factors GPFA
#   fits the data it has seen ever better.
# - **The score on unseen data rises up to 25-30 factors and then falls** at 40 and 50 (blue), and the
#   model with 30 factors beats those with 40 and 50 in all 5 blocks (last row of the table). Beyond
#   ~30 the extra factors memorize the noise of the training data and worsen the prediction
#   on new data: this is **overfitting**. It is also visible in the distance between the two curves, which
#   keeps widening.
# - **The maximum is flat**: 25 and 30 factors have practically the same mean score.
#
# ### Why we choose 25 factors
#
# The mean alone is not enough to choose: we look at the 5 blocks **one by one** (last row
# of the table) and ask in how many blocks the model with the highest mean score (30 factors)
# does better than the others.
#
# - **Against 20 factors it wins in all 5 blocks**: 20 is really worse, not by chance. For this reason
#   we do not use 20, even though it comes close.
# - **Against 25 factors it wins in only 3 blocks out of 5**, and on average the difference is zero: 25 and 30
#   are equivalent.
# - **Among equivalent models we choose the simplest**, the one with fewer factors: fewer parameters
#   to estimate and less risk of chasing the noise.
#
# The rule, written in the code, is: *the smallest number of factors that the best one does not beat in
# all blocks*. Here it gives **25 factors**, which we use in the following sections.
#
# **An interesting comparison with FA.** FA, with its cross-validation, chose 10 factors
# (§9.6); GPFA sustains 25-30 without overfitting. The model of time helps to tell the slow
# signal from the noise, so GPFA manages to use more factors usefully.
#
# ### 10.3 · The time scales learned by GPFA
#
# We fit the final GPFA with the 25 factors chosen in §10.2, on all the windows, and look at the
# time scale τ it has
# learned for each factor. This is GPFA's answer to the question of §4: *how much to smooth?*
#
# To see how reliable they are, we repeat the fit starting from different initial time scales
# (50 and 200 ms instead of 100). If a τ is well determined by the data, the starting point does not matter.

# %%
params_gp, _ = fit_gpfa(windows, D_GP)                  # final GPFA, on all the windows
tau_gp = np.sort(BIN_MS / np.sqrt(params_gp["gamma"]))  # time scale of each factor, in ms (from gamma, as in elephant)

alt = Parallel(n_jobs=-1)(delayed(fit_gpfa)(windows, D_GP, tau_init=t0)  # same fits with different starting points
                          for t0 in (50.0, 200.0))
tau_alt = {t0: np.sort(BIN_MS / np.sqrt(p["gamma"])) for t0, (p, _) in zip((50.0, 200.0), alt)}
display(pd.DataFrame({"start 50 ms": tau_alt[50.0], "start 100 ms (chosen)": tau_gp,
                      "start 200 ms": tau_alt[200.0]},
                     index=pd.Index(np.arange(1, D_GP + 1), name="factor (fastest first)")).round(0).T)

fig, ax = plt.subplots(figsize=(9, 3.8))
ax.bar(np.arange(1, D_GP + 1), tau_gp, color="C0", width=0.7)
ax.axhline(SIGMA_MS, color="C3", ls="--", lw=1.2, label=f"σ chosen by hand for PCA ({SIGMA_MS} ms)")
ax.axhline(WIN_S * 1000, color="0.5", ls=":", lw=1.2, label="duration of a window (1000 ms)")
ax.set(xlabel="GPFA factor, from fastest to slowest", ylabel="time scale τ (ms)",
       title="How much each factor smooths: GPFA decides", xticks=np.arange(1, D_GP + 1))
ax.legend(loc="upper left")
fig.tight_layout(); fig.savefig(FIGS / "02new_gpfa_timescales.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# **Reading.**
#
# - GPFA learned time scales ranging from **about 50 ms** (the fastest factor, similar
#   to our σ) to **several hundred ms**. There is no single right σ for all: some factors
#   change quickly, others slowly. This is what a hand-chosen smoothing cannot do.
# - **The fastest factor is stable**: τ ≈ 45-50 ms wherever one starts.
# - **The others shift with the starting point**: starting from 50 ms they come out shorter, starting
#   from 200 ms longer, and the slowest change the most (up to ~900 ms). One possible reason is that
#   during the fit `elephant` looks at 400 ms pieces (§10.1), within which a slow factor is
#   almost constant. In §10.6 we try fitting on the whole windows, and find that it is not enough: the slow
#   scales remain undetermined. The data clearly say that alongside
#   a fast factor there are slow factors, from ~100 ms to many hundreds of ms, but they do not pin down
#   the value of each precisely. The reliable message is the range, not the single number.
#
# ### 10.4 · How many dimensions does GPFA see?
#
# The shared part of GPFA is $C C^\top$, exactly like $W W^\top$ in FA: we can
# compute the same measures. To isolate the effect of time, we compare it with an **FA on the
# same data** (square root of counts, same bins of the windows) **and with the same number of
# factors**: the only difference between the two is the model of time. To see how much the result
# depends on the number of factors, we redo both FA and GPFA with 10, 15, 20, 25 and 30 factors.

# %%
def shared_stats(C_, psi_):                             # PR, dimensions for 95% and median shared fraction
    e_ = np.linalg.eigvalsh(C_ @ C_.T)[::-1][:C_.shape[1]]  # variances of the shared part, from the largest
    sh_ = (C_ ** 2).sum(axis=1)                         # shared variance of each channel
    return pr(e_), n95(e_), np.median(sh_ / (sh_ + psi_)) * 100

X_sqrt = np.sqrt(X_counts)                              # square root of the counts, all bins
D_TAB = [10, 15, 20, 25, 30]                            # numbers of factors for the table

other_d = [d for d in D_TAB if d != D_GP]               # we already have the GPFA with D_GP (§10.3)
gp_fits = dict(zip(other_d, Parallel(n_jobs=-1)(delayed(fit_gpfa)(windows, d) for d in other_d)))
gp_fits[D_GP] = (params_gp, None)
fa_fits = {d: FactorAnalysis(n_components=d, svd_method="lapack", random_state=0).fit(X_sqrt[in_win])
           for d in D_TAB}                              # FA on the same bins, with the same number of factors
fa_sq = fa_fits[D_GP]                                   # the FA with D_GP factors, also used in §10.5

rows = []
for d in D_TAB:
    g_pr, g_95, g_sh = shared_stats(gp_fits[d][0]["C"], np.diag(gp_fits[d][0]["R"]))  # GPFA measures
    f_pr, f_95, f_sh = shared_stats(fa_fits[d].components_.T, fa_fits[d].noise_variance_)  # FA measures
    rows.append({"factors": d, "GPFA: participation ratio": g_pr, "FA: participation ratio": f_pr,
                 "GPFA: dimensions for 95%": g_95, "FA: dimensions for 95%": f_95,
                 "GPFA: median shared fraction (%)": g_sh, "FA: median shared fraction (%)": f_sh})
display(pd.DataFrame(rows).set_index("factors").round(1))

# %% [markdown]
# **Reading.**
#
# - **On "how many dimensions" GPFA and FA agree**: at every number of factors the
#   participation ratio of the shared part is almost the same (GPFA is always a little lower).
#   Adding time does not change *how much* shared structure there is.
# - **The number grows with the factors, without stopping**: from about 6.5 with 10 factors to about 13.4
#   with 30. The extra factors are not empty: they carry a part of the shared variance. For this reason
#   the number of dimensions must always be stated together with the number of factors, like σ for PCA.
# - **With the 25 factors chosen in §10.2 the participation ratio is about 12** (and about 21
#   directions are needed for 95% of the shared part). With 20 factors it would have been about 10.5: the choice
#   of the number of factors shifts the result by about 16%, and this is why the choice must be
#   made with cross-validation and not by eye. In any case these are few dimensions compared with the
#   ~100 of PCA.
# - The shared fraction of GPFA is a little lower than that of FA: GPFA considers
#   "shared" only what is shared **and** changes gradually. Common but very fast
#   fluctuations end up in the private noise.
#
# ### 10.5 · Single-trial trajectories: finally clean?
#
# The decisive comparison, on the same windows and with the same two measures as in §9.4 (roughness and
# consistency between repetitions, first 3 dimensions). All methods are fitted **on the same bins
# of the windows**:
#
# - PCA on raw counts and PCA on data smoothed at σ = 50 ms;
# - FA on the square root of the counts, with the same number of factors as GPFA: it is "GPFA without
#   time";
# - GPFA.
#
# The GPFA trajectories are obtained by estimating the factors in every window and rotating them so
# that they are ordered by importance, like the PCA components (`orthonormalize` of `elephant`).

# %%
seq_lat, _ = gpfa_core.exact_inference_with_ll(make_seqs(windows), params_gp, get_ll=False)  # factors in every window
_, seq_orth = gpfa_core.orthonormalize(params_gp, seq_lat)  # factors rotated and ordered by importance
G = [x.T for x in seq_orth["latent_variable_orth"]]     # for each window: (50 bins, d factors)

pca_raw_w = PCA(3).fit(X_fa[in_win])                    # PCA on raw counts, fitted on the bins of the windows
pca_sm_w = PCA(3).fit(Z[in_win])                        # PCA on smoothed data, fitted on the bins of the windows
z_fa_sq = orth_latents(fa_sq, X_sqrt)                   # FA on the square root of the counts, factors ordered (§9.4)

traj = {
    "PCA, raw counts": [pca_raw_w.transform(X_fa[w["start"]:w["stop"]]) for w in windows],
    "PCA, smoothed σ = 50 ms": [pca_sm_w.transform(Z[w["start"]:w["stop"]]) for w in windows],
    f"FA, square-root counts (K = {D_GP})": [z_fa_sq[w["start"]:w["stop"], :3] for w in windows],
    f"GPFA, square-root counts (d = {D_GP})": [g[:, :3] for g in G],
}
rows = []
for name, tr in traj.items():
    rows.append({"method": name, "roughness (lower = better)": roughness(tr),
                 "consistency (higher = better)": consistency(tr)})
tab_final = pd.DataFrame(rows).set_index("method")
display(tab_final.round(4))

display(pd.DataFrame({name: consistency_w(tr, windows, per_phase=True)  # consistency phase by phase
                     for name, tr in traj.items()}).T.round(2)              # (mean of the two versions)
        .rename_axis(columns="consistency per phase (mean of the two versions)"))

# %%
PH_SHOW = ("Release", "odd")                        # a well-modulated condition, for the figure
idx_ph = [i for i, w in enumerate(windows) if cond_of(w) == PH_SHOW]  # its repetitions
t_win = np.arange(L) * BIN_S * 1000                     # time from the start of the phase, in ms

fig, ax = plt.subplots(1, 4, figsize=(16, 3.6))
for k, (name, tr) in enumerate(traj.items()):
    R = np.stack([tr[i][:, 0] for i in idx_ph])         # first dimension of each repetition: (repetitions, 50)
    R = (R - R.mean()) / R.std()                        # same scale for all methods
    for r in R:
        ax[k].plot(t_win, r, color=io.PHASE_COLORS[PH_SHOW[0]], lw=0.8, alpha=0.6)  # each repetition, thin
    ax[k].plot(t_win, R.mean(axis=0), color="0.1", lw=2.2)                        # their mean, thick
    ax[k].set(title=name, xlabel="ms from the start of the phase", ylim=(-4, 4))
ax[0].set_ylabel("first latent dimension (z)")
fig.suptitle(f"The {len(idx_ph)} repetitions of {PH_SHOW[0]} in {PH_SHOW[1]} trials, one per line (black = mean)", y=1.02)
fig.tight_layout(); fig.savefig(FIGS / "02new_gpfa_repetitions.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# **Reading.**
#
# - **GPFA wins on both measures**: it is the smoothest and the one in which repetitions of the
#   same condition resemble each other the most (0.137 versus 0.117 for smoothed PCA).
#   In the figure the `Release` repetitions in odd trials are smooth and largely follow the
#   same pattern (they rise in the first ~200 ms and then fall), the same one found by smoothed
#   PCA, albeit with some dispersion; with PCA and FA on raw counts they are a jagged
#   cloud.
# - Phase by phase, GPFA is better on `Grasp` and `Release`, on par with smoothed PCA on `Shape`, and a
#   little worse on `Reach`.
# - **FA on the square root of the counts behaves like raw PCA**, even though it has the same data
#   and the same number of factors as GPFA. The only difference between FA and GPFA is the model of
#   time: **it is time that cleans the trajectories**.
# - The biggest jump is from "no model of time" to "a model of time". Between hand-smoothed PCA
#   and GPFA the difference is smaller; the advantage of GPFA is that we did not have to
#   choose σ, and that each factor has its own time scale.
# - **Honestly**, consistency remains low in absolute terms for many phases (`Reach`, `Orient`,
#   `SnapTo`, `Carry`, `Orient2` stay below 0.1 with all methods). The most reproducible phases are the hand
#   ones (`Shape`, `Grasp`, `Release`), consistent with the position of the arrays in the hand area of
#   motor cortex.
#
# ### GPFA trajectories in the plane, as in §8
#
# In §8 the clean trajectories were visible only by averaging the repetitions. With GPFA
# we look at **a single repetition per condition**. The axes are the first two dimensions of
# GPFA, estimated **without knowing which condition each window belongs to**.

# %%
first = {c: next(i for i, w in enumerate(windows) if cond_of(w) == c) for c in CONDS}  # first repetition of each condition
fig, ax = plt.subplots(1, 2, figsize=(11.5, 4.6), sharex=True, sharey=True)
for (ph, v) in CONDS:
    one_gp = G[first[(ph, v)]][:, :2]                   # one repetition
    avg_gp = np.mean([G[i][:, :2] for i, w in enumerate(windows) if cond_of(w) == (ph, v)], axis=0)  # mean of the repetitions
    for k, D_ in enumerate((one_gp, avg_gp)):
        ax[k].plot(D_[:, 0], D_[:, 1], color=io.PHASE_COLORS[ph], ls=VER_LS[v], lw=1.6)
        ax[k].plot(D_[0, 0], D_[0, 1], "o", color=io.PHASE_COLORS[ph], ms=6)  # dot = start of the phase
ax[0].set(xlabel="GPFA 1", ylabel="GPFA 2", title="GPFA: a single repetition per condition")
ax[1].set(xlabel="GPFA 1", title="GPFA: mean of the repetitions")
handles = ([plt.Line2D([], [], color=io.PHASE_COLORS[ph], lw=2, label=ph) for ph in io.MOVEMENT_PHASES]
           + [plt.Line2D([], [], color="0.3", ls=VER_LS[v], lw=1.6, label=f"trial {v}") for v in VERSIONS])
fig.legend(handles=handles, ncol=10, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.03))
fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(FIGS / "02new_gpfa_plane.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# **Reading.** Compared with the central panel of the §8 figure (one repetition, PCA), the
# trajectories of a single repetition are **smooth**, not jagged. They remain messier
# than the mean: GPFA removes the fast noise, but a single repetition is never identical to the
# mean, also because the attempted movement changes a little each time.
#
# **Summary of GPFA.** GPFA combines the two ingredients that were needed: the noise model of
# FA (which channels are reliable) and a model of time (nearby instants
# resemble each other), with time scales learned from the data. Thanks to the model of time it sustains more factors
# than FA without overfitting (25 versus 10). On *how many dimensions*, at equal number of factors, it
# agrees with FA (about 12 effective dimensions with the 25 chosen factors; the number grows with
# the factors);
# on *single-trial trajectories* it is the best of the linear methods, with no need to average
# and without choosing σ by hand.

# %% [markdown]
# ### 10.6 · What if GPFA learns on the whole windows?
#
# In §10.1 we saw that `elephant`, to go faster, splits during the fit each
# 50-bin window into pieces of 20 bins (400 ms). In §10.3 the slowest time scales changed
# with the starting point, and one possible reason was exactly this: within a 400 ms piece a
# slow factor looks almost still.
#
# Here we test it: **we fit GPFA on the whole windows** (50 bins, 1 s) and see whether the
# slow time scales stabilize and whether anything else changes. Everything else is the same: same
# windows, 25 factors, 500 iterations. Only the length of the pieces changes.
#
# The length of the pieces is not a parameter of `gpfa_core.fit`: it is fixed inside an
# internal function (`cut_trials`). With `whole=True` our `fit_gpfa` replaces it, only for the duration
# of the fit, with the version that does not cut.
#
# We run 8 fits in parallel: 3 on all the windows, starting from 50, 100 and 200 ms as in
# §10.3, and 5 for cross-validation. Each fit costs about 7 times more than with 20-bin
# pieces: a few minutes in all.

# %%
def seg_job(kind, arg):                                 # a fit on the whole windows
    if kind == "tau":                                   # for the time scales: all the windows, starting point arg (ms)
        return fit_gpfa(windows, D_GP, tau_init=arg, whole=True)
    train, test = folds_gp[arg]                         # for the score: block number arg serves as the test
    params, _ = fit_gpfa([windows[i] for i in train], D_GP, whole=True)
    return gpfa_ll_per_bin(params, [windows[i] for i in test])

TAU_STARTS = (50.0, 100.0, 200.0)                       # starting points of the time scales, in ms
jobs = [("tau", t0) for t0 in TAU_STARTS] + [("cv", k) for k in range(len(folds_gp))]
out = Parallel(n_jobs=-1)(delayed(seg_job)(kind, arg) for kind, arg in jobs)  # the 8 fits, in parallel
whole_fits = dict(zip(TAU_STARTS, out[:len(TAU_STARTS)]))  # for each start: (parameters, information about the fit)
cv_whole = np.array(out[len(TAU_STARTS):])              # score on unseen data, one value per block
params_whole = whole_fits[100.0][0]                     # the "whole windows" model with the standard starting point

tau_20 = {50.0: tau_alt[50.0], 100.0: tau_gp, 200.0: tau_alt[200.0]}  # time scales with 20-bin pieces (from §10.3)
tau_50 = {t0: np.sort(BIN_MS / np.sqrt(p["gamma"])) for t0, (p, _) in whole_fits.items()}  # with the whole windows

for t0, (_, info) in whole_fits.items():                # did the score reach the top after 500 iterations?
    ll_ = np.array(info["log_likelihoods"]) / (len(windows) * L)
    ok_ = np.flatnonzero(~np.isnan(ll_))
    print(f"whole windows, start {t0:3.0f} ms: score per bin after 100 / 300 / 500 iterations = "
          + " / ".join(f"{ll_[ok_[ok_ < k].max()]:.3f}" for k in (100, 300, 500)))

def tau_table(tau_by_start, label):                     # summary of the time scales for one way of fitting
    A = np.stack([tau_by_start[t0] for t0 in TAU_STARTS])  # (3 starts, 25 factors), from the fastest to the slowest
    ratio = A.max(axis=0) / A.min(axis=0)               # for each factor: how much it changes across the three starts
    rng_ = lambda v: f"{v.min():.0f}-{v.max():.0f}"     # range across the three starts
    return {"fit on": label, "fastest factor (ms)": rng_(A[:, 0]), "middle factor (ms)": rng_(A[:, D_GP // 2]),
            "slowest factor (ms)": rng_(A[:, -1]),
            "stable factors (change by less than 30%)": f"{int((ratio < 1.3).sum())} of {D_GP}"}

display(pd.DataFrame([tau_table(tau_20, "20-bin pieces (rest of the notebook)"),
                      tau_table(tau_50, "whole windows (50 bins)")]).set_index("fit on"))

fig, ax = plt.subplots(1, 2, figsize=(13, 4.2))
rank = np.arange(1, D_GP + 1)
for tau_by_start, c, lab in [(tau_20, "0.55", "20-bin pieces"), (tau_50, "C0", "whole windows")]:
    A = np.stack([tau_by_start[t0] for t0 in TAU_STARTS])
    for i, row in enumerate(A):                         # one line for each starting point
        ax[0].plot(rank, row, "o-", color=c, lw=1.2, ms=3.5, alpha=0.8, label=lab if i == 0 else None)
    ax[1].plot(rank, A.max(axis=0) / A.min(axis=0), "o-", color=c, lw=1.8, ms=5, label=lab)
ax[0].axhline(400, color="0.3", ls=":", lw=1); ax[0].text(1, 430, "one 20-bin piece (400 ms)", fontsize=8)
ax[0].axhline(1000, color="0.3", ls="--", lw=1); ax[0].text(1, 1080, "one window (1000 ms)", fontsize=8)
ax[0].set_yscale("log")                                 # logarithmic scale: the slow scales are much larger than the fast ones
ax[0].set(xlabel="factor, from fastest to slowest", ylabel="time scale τ (ms)",
          title="Time scales: three starting points for each way")
ax[0].legend(loc="lower right")
ax[1].axhline(1.3, color="C3", ls="--", lw=1.2, label="threshold: changes by 30%")
ax[1].set(xlabel="factor, from fastest to slowest", ylabel="largest τ / smallest τ across the three starts",
          title="How much it depends on the starting point")
ax[1].legend(loc="upper left")
fig.tight_layout(); fig.savefig(FIGS / "02new_gpfa_segments.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# **Reading.**
#
# - **The fastest factor remains stable**: about 45-55 ms in both ways.
# - **The slow time scales do not stabilize.** Fitting on the whole windows they actually become
#   longer (up to about 2.5 s, versus 0.9 s) and depend on the starting point even more:
#   in the right-hand panel, from factor 17 onwards, the blue line lies above the grey one.
# - **The intermediate scales also depend on the starting point**, in both ways: across the three
#   starts they change by a factor of 1.4-2, and only 1-2 factors out of 25 change by less than 30%.
# - **At 500 iterations the fit on the whole windows has not finished**: the score is still
#   rising and the three starts reach different scores (printout above). The time scales
#   need many more iterations than suffice for the score.
#
# > **Separate test**, not included here because it takes about 17 minutes per fit. With 2000
# > iterations on the whole windows the scales up to about 0.5 s come closer across the three starts
# > (differences of 10-30%), while the slow ones keep growing, up to tens of seconds.
# > A scale of 10 s within a 1 s window merely means "this factor is practically
# > still": the data say that it is slower than the window, not by how much.
#
# **Conclusion on the time scales.** The problem was not the length of the pieces: with 1 s
# windows the slow scales cannot be determined, however one fits. The reliable message remains
# that of §10.3: a fast factor around 50 ms and other, slower factors, from about 100 ms upwards.
#
# Time scales are one thing; but is the model, overall, better or worse? We compare
# the two ways of fitting on everything else: score on unseen data, number of dimensions, trajectories.

# %%
cv_20 = test_folds[D_GRID.index(D_GP)]                  # scores per block with 20-bin pieces (from §10.2)
seq_w, _ = gpfa_core.exact_inference_with_ll(make_seqs(windows), params_whole, get_ll=False)  # factors in every window
_, orth_w = gpfa_core.orthonormalize(params_whole, seq_w)  # rotated and ordered by importance
G_whole = [x.T[:, :3] for x in orth_w["latent_variable_orth"]]  # the first 3 dimensions in every window

gp_name = f"GPFA, square-root counts (d = {D_GP})"      # the name of the GPFA in the dictionary of §10.5
rows = []
for label, p_, cv_, tr_ in [("20-bin pieces (rest of the notebook)", params_gp, cv_20, traj[gp_name]),
                            ("whole windows (50 bins)", params_whole, cv_whole, G_whole)]:
    pr_, n95_, sh_ = shared_stats(p_["C"], np.diag(p_["R"]))  # dimensions and shared fraction (§10.4)
    rows.append({"fit on": label, "score on unseen data": cv_.mean(), "participation ratio": pr_,
                 "dimensions for 95%": n95_, "median shared fraction (%)": sh_,
                 "seghettatura": roughness(tr_), "consistency": consistency(tr_)})
display(pd.DataFrame(rows).set_index("fit on").round(3))
print(f"Blocks in which the fit on the whole windows has the highest score: {int((cv_whole > cv_20).sum())} of {len(cv_20)}"
      f"   (mean difference {np.mean(cv_whole - cv_20):+.3f} per bin)")

# %% [markdown]
# **Reading.**
#
# - **On unseen data the model fitted on the whole windows is better in all 5
#   blocks** (+0.09 per bin). This is not small: going from 20 to 25 factors gave +0.05 (§10.2).
# - **The number of dimensions does not change**: participation ratio 12.6 versus 12.2, and 21 dimensions
#   for 95% in both cases.
# - **The trajectories are very similar**: a little smoother (0.009 versus 0.011) and a little less
#   consistent (0.128 versus 0.137).
# - **The cost:** each fit takes about 7 times longer, and 500 iterations are not enough.
#
# **Choice.** In the rest of the notebook we keep the 20-bin pieces, the `elephant` default: no
# conclusion changes, and redoing the choice of the number of factors (§10.2) on the whole windows
# would take hours of computation. If in future GPFA is used to decode movement, where
# the score on unseen data is what matters, it will be worth fitting it on the whole windows and with more
# iterations.

# %% [markdown]
# ### 10.7 · Two checks on the windows
#
# All the work from §8 onwards uses 1 s windows from the onset of each phase, with all the
# repetitions. These are two choices of ours, and we check them **one at a time**, keeping everything else
# fixed (σ = 50 ms, 25 factors, same methods, no new cross-validation):
#
# - **A. The window duration**: 0.5 s and 0.75 s alongside 1 s. According to the PSTH of notebook 01
#   the response lasts about 0.7-0.8 s, so 0.75 s is the "natural" duration, and 0.5 s cuts off half of it.
# - **B. Repetitions that are too short**: in some trials a phase lasts a single bin (20 ms), i.e. the
#   movement is not there. We redo the measures excluding repetitions shorter than 100 ms.
#
# For each check we redo the two main measures:
#
# 1. **the drop in the participation ratio** from a single repetition to the mean of the repetitions,
#    with the random-window control (§8). We compare the drops and not the absolute PRs, because the
#    PR depends on the number of points, which changes with the window duration;
# 2. **roughness and consistency** of the PCA, FA and GPFA trajectories (§10.5).
#
# #### A. The window duration
#
# For 0.5 and 0.75 s a new GPFA is needed (same 25 factors), fitted on the shorter windows; for
# 1 s we reuse the one from §10.3. The random draws for the PR are new, so the values for 1 s
# may differ slightly from those of §8.

# %%
def pr_avg(win_list, n, aligned=True, n_draws=20, seed=0):  # mean PR when averaging n repetitions (or n random windows)
    rng_ = np.random.default_rng(seed)                  # fixed seed: reproducible draws
    L_ = win_list[0]["stop"] - win_list[0]["start"]     # bins per window
    reps_ = {c: np.stack([Z[w["start"]:w["stop"]] for w in win_list if cond_of(w) == c])  # (repetitions, bins, channels)
             for c in CONDS}
    out = []
    for _ in range(n_draws):
        if aligned:                                     # n true repetitions of each condition, averaged
            M = np.concatenate([reps_[c][rng_.choice(len(reps_[c]), n, replace=False)].mean(axis=0)
                                for c in CONDS])
        else:                                           # control: n windows at random points, averaged
            M = np.concatenate([np.mean([Z[a:a + L_] for a in rng_.integers(0, n_bins - L_, n)], axis=0)
                                for _ in CONDS])
        out.append(pr(PCA().fit(M).explained_variance_))
    return np.mean(out)

def window_trajs(win_list, gp_params):                  # the first 3 dimensions of PCA, FA and GPFA in every window
    mask = np.zeros(n_bins, dtype=bool)                 # the bins of the windows, counted only once
    for w in win_list:
        mask[w["start"]:w["stop"]] = True
    pca_r = PCA(3).fit(X_fa[mask])                      # PCA on raw counts, fitted on these bins
    pca_s = PCA(3).fit(Z[mask])                         # PCA on smoothed data, fitted on these bins
    fa_ = FactorAnalysis(n_components=D_GP, svd_method="lapack", random_state=0).fit(X_sqrt[mask])  # FA, same factors as GPFA
    z_fa_ = orth_latents(fa_, X_sqrt)
    lat_, _ = gpfa_core.exact_inference_with_ll(make_seqs(win_list), gp_params, get_ll=False)  # GPFA factors in every window
    _, orth_ = gpfa_core.orthonormalize(gp_params, lat_)
    return {"PCA, raw counts": [pca_r.transform(X_fa[w["start"]:w["stop"]]) for w in win_list],
            "PCA, smoothed σ = 50 ms": [pca_s.transform(Z[w["start"]:w["stop"]]) for w in win_list],
            f"FA, square-root counts (K = {D_GP})": [z_fa_[w["start"]:w["stop"], :3] for w in win_list],
            f"GPFA, square-root counts (d = {D_GP})": [x.T[:, :3] for x in orth_["latent_variable_orth"]]}

WIN_LIST = [0.5, 0.75, 1.0]                             # durations to compare, in s
win_by_len = {ws: io.phase_windows(s, win_s=ws, align="start") for ws in WIN_LIST}  # the windows for each duration
new_gp = Parallel(n_jobs=-1)(delayed(fit_gpfa)(win_by_len[ws], D_GP) for ws in WIN_LIST if ws != WIN_S)  # new GPFAs
gp_by_len = dict(zip([ws for ws in WIN_LIST if ws != WIN_S], [p for p, _ in new_gp]))
gp_by_len[WIN_S] = params_gp                            # for 1 s we reuse the GPFA of §10.3

rows_pr, rows_tr = [], []
for ws in WIN_LIST:
    wl = win_by_len[ws]
    n_rep = min(sum(cond_of(w) == c for w in wl) for c in CONDS)  # repetitions available in all conditions
    single, avg, rand = pr_avg(wl, 1), pr_avg(wl, n_rep), pr_avg(wl, n_rep, aligned=False)
    rows_pr.append({"window duration (s)": ws, "bins per window": wl[0]["stop"] - wl[0]["start"],
                    "PR, single repetition": single, f"PR, mean of the repetitions": avg,
                    "PR, random windows averaged": rand, "drop with averaging (%)": (1 - avg / single) * 100})
    for name, tr in window_trajs(wl, gp_by_len[ws]).items():
        rows_tr.append({"window duration (s)": ws, "method": name,
                        "seghettatura": roughness(tr), "consistency": consistency_w(tr, wl)})
tab_len_pr = pd.DataFrame(rows_pr).set_index("window duration (s)")
tab_len_tr = pd.DataFrame(rows_tr)
display(tab_len_pr.round(1))
display(tab_len_tr.pivot(index="method", columns="window duration (s)", values="consistency")
        .round(3).rename_axis(columns="consistency, windows of (s)"))
display(tab_len_tr.pivot(index="method", columns="window duration (s)", values="seghettatura")
        .round(4).rename_axis(columns="roughness, windows of (s)"))

# %%
METHOD_COLORS = dict(zip(tab_len_tr["method"].unique(), ["0.6", "C0", "C1", "C2"]))  # a fixed colour per method
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
ax[0].plot(WIN_LIST, tab_len_pr["drop with averaging (%)"], "o-", color="C0", lw=2, ms=7, label="phase-aligned windows")
ax[0].plot(WIN_LIST, (1 - tab_len_pr["PR, random windows averaged"] / tab_len_pr["PR, single repetition"]) * 100,
           "s--", color="0.55", lw=1.5, ms=7, label="random windows (control)")
ax[0].axhline(0, color="0.3", lw=0.8)
ax[0].set(xlabel="window duration (s)", ylabel="drop in PR with averaging (%)", xticks=WIN_LIST,
          title="How much averaging lowers the PR")
ax[0].legend(loc="center right")
for name, g in tab_len_tr.groupby("method", sort=False):
    ax[1].plot(g["window duration (s)"], g["consistency"], "o-", color=METHOD_COLORS[name], lw=2, ms=7, label=name)
ax[1].set(xlabel="window duration (s)", ylabel="consistency between repetitions", xticks=WIN_LIST,
          title="Consistency of single-trial trajectories")
ax[1].legend(fontsize=8, loc="upper left")
fig.tight_layout(); fig.savefig(FIGS / "02new_window_length.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# **Reading.**
#
# - **The conclusion of §8 does not depend on the duration.** With 0.5, 0.75 and 1 s the average of the
#   aligned repetitions lowers the participation ratio by 59-63%, whereas averaging
#   random windows does not lower it (it even raises it slightly). The absolute PR values change
#   with the duration, because with shorter windows there are fewer points, but the drop is the same.
# - **The ranking of the methods does not change.** At every duration GPFA is the smoothest and the most
#   consistent, smoothed PCA follows closely, raw PCA and FA remain far behind.
# - **Consistency grows with the duration, for all methods** (GPFA: 0.045 → 0.068 → 0.137). A longer
#   window contains more of the response that repeats identically, i.e. the rise and fall
#   of the PSTH of notebook 01; with 0.5 s we see only the first half. For this reason consistency values
#   must be compared between methods at the same duration, not across different durations.
#
# **In short:** with 0.5 or 0.75 s no conclusion changes, and 1 s contains the whole response.
# We keep 1 s.
#
# #### B. Repetitions that are too short
#
# We return to the 1 s windows and remove the repetitions in which the phase lasts less than 100 ms. The
# models remain those fitted before: only which repetitions enter the measures changes. For the
# PR comparison we average the same number of repetitions in both cases, i.e. as many as remain
# in the condition that has the fewest.

# %%
MIN_PHASE_S = 0.1                                       # minimum duration of a phase, in s
dur_at = {int(round(e.start_time / BIN_S)): e.dur       # duration of each movement phase, indexed by the start bin
          for _, e in s.epochs.iterrows() if e.tag in io.MOVEMENT_PHASES}
keep = [dur_at[w["start"]] >= MIN_PHASE_S for w in windows]  # true = repetition long enough
windows_long = [w for w, k in zip(windows, keep) if k]  # the windows that remain
print("Repetitions excluded per phase:",
      pd.Series([w["tag"] for w, k in zip(windows, keep) if not k]).value_counts().to_dict())

n_same = min(sum(cond_of(w) == c for w in windows_long) for c in CONDS)  # repetitions averaged in both cases
rows = []
for label, wl in [("all repetitions", windows), (f"without phases < {MIN_PHASE_S * 1000:.0f} ms", windows_long)]:
    single, avg = pr_avg(wl, 1), pr_avg(wl, n_same)
    rows.append({"repetitions": label, "PR, single repetition": single, f"PR, mean of {n_same} repetitions": avg,
                 "drop with averaging (%)": (1 - avg / single) * 100})
display(pd.DataFrame(rows).set_index("repetitions").round(1))

rows = []
for name, tr in traj.items():                           # the trajectories of §10.5, with and without the short repetitions
    for label, idx in [("all", range(len(windows))), ("without the short ones", [i for i, k in enumerate(keep) if k])]:
        wl = [windows[i] for i in idx]
        rows.append({"method": name, "repetitions": label,
                     **consistency_w([tr[i] for i in idx], wl, per_phase=True),
                     "mean over phases": consistency_w([tr[i] for i in idx], wl)})
display(pd.DataFrame(rows).set_index(["method", "repetitions"]).round(2))

# %% [markdown]
# **Reading.**
#
# - The excluded repetitions are 12 out of 119: 5 of `Orient`, 3 of `Orient2`, 2 of `Reach` and 2 of
#   `Carry`.
#   Split by version, the condition that loses the most is `Orient` in even trials (3 out of 7): 4
#   remain, which is why the PR comparison averages 4 repetitions.
# - **The PR does not change**: the drop is the same (about 40% with 4 repetitions).
# - **Consistency does not change for 6 phases out of 8** (at most ±0.02). `Orient` changes (GPFA from 0.08
#   to 0.01, smoothed PCA from 0.06 to −0.01) and `Carry` a little (smoothed PCA +0.03): these are the phases that
#   lose the most repetitions, and with 4-6 repetitions per condition the measure is noisy. The mean
#   over phases changes by at most 0.01.
# - Therefore the low consistency of `Reach`, `Orient`, `Carry` and `Orient2` **does not depend on these
#   repetitions**: it stays low without them too. One possible explanation, to be verified, is that the
#   arrays are in the hand area and these phases mainly move the arm and wrist.
#
# We keep all the repetitions: excluding them does not change the results.

# %% [markdown]
# ## 11 · Summary
#
# | step | parameter | why | if we changed it |
# |---|---|---|---|
# | threshold (upstream, not ours) | −4.5 × RMS, recomputed at every session | separates spikes from background noise; lab value (Downey 2017, Quick 2020), not reported by FALCON | cannot be changed: we do not have the raw voltage |
# | 1 · binning | 20 ms | native step of the benchmark, aligned 1:1 with the kinematics | the kinematics would have to be resampled |
# | 2 · active channels | ≥ 0.5 Hz → 160 of 176 | almost silent channels carry no information, have variance ~0 and become outliers after the z-score | 0.1 Hz excludes 1, 2 Hz excludes 46 |
# | 3 · smoothing | Gaussian, σ = 50 ms | a 20 ms bin contains 0 or 1 events: a weighted average of neighbouring bins is needed to estimate the firing rate | the participation ratio goes from ~150 (no smoothing) to ~65 (σ = 200 ms), with no plateau (§6.1) |
# | 4 · z-score | mean 0, variance 1 per channel | variance grows with firing rate: without z-score PC1 describes the most active channels | alternative: *soft normalization* (Churchland 2012) |
# | 5 · PCA | all components | directions of maximum variance of the population | — |
# | PCA on the means | 1 s windows from the onset of each phase; 16 conditions phase × task version, 7-8 repetitions | averaging cancels the noise and leaves the task signal; the two versions (odd and even trials) are different movements and are not mixed | the comparison must be made at equal number of points (§8) |
# | FA | raw counts 20 ms, K chosen by cross-validation, exact SVD | separates the shared signal from the private noise of each channel | with smoothing the shared fraction rises, the dimensionality changes little (§9.5) |
# | GPFA | square root of counts, 1 s windows, 500 iterations, d by cross-validation | FA + a time scale learned for each factor: smooth single-trial trajectories | at equal factors same dimensions as FA; better trajectories (§10.5) |
#
# ### Alternatives tried in this notebook
#
# Each choice was compared with alternatives, changing only one parameter at a time.
#
# | what | values tried | effect | where |
# |---|---|---|---|
# | detection threshold (synthetic signal) | −3, −4.5, −6 × RMS | −3: false events from noise; −4.5: both neurons; −6: loses spikes of the distant neuron | §1 |
# | channel activity threshold | 0.1, 0.5, 1, 2, 5 Hz | excludes 1, 16, 28, 46, 72 channels out of 176 | §3 |
# | smoothing width σ | 20, 50, 100 ms (figure); 0, 10, 20, 50, 100, 200 ms (PCA) | PR from ~150 to ~65, no plateau: the number depends on σ, the qualitative conclusion does not | §4, §6.1 |
# | normalization | none vs z-score | weight of the 10 most active channels in PC1: 64% vs 21% | §5 |
# | repetitions averaged | 1, 2, 4, 7; aligned vs random windows | the PR (800 points) falls from ~50 to ~21 only with phase-aligned windows | §8 |
# | FA algorithm | approximate SVD (default) vs exact | the default version stops before the solution | §9 |
# | number of factors K | 2, 5, 10, 15, 20, 30 (cross-validation) | K* = 15, flat curve between 10 and 20 | §9.1 |
# | z-score before FA | with / without | identical shared fraction: FA does not depend on the scale | §9.2 |
# | method for the trajectories | PCA vs FA, raw vs smoothed counts | FA does not improve single-trial trajectories; smoothing does | §9.4 |
# | smoothing before FA | σ = 0, 20, 50, 100 ms | shared fraction ~4% → ~19%; FA dimensionality stable (PR 9.5-13) | §9.5 |
# | instants used by FA | whole session vs only the phase windows | shared PR 9.5 → 7; K* 15 → 10 | §9.6 |
# | window duration | 0.5, 0.75, 1 s | similar drop in PR with averaging (59-63%); same ranking of methods; higher consistency with 1 s | §10.7 |
# | repetitions that are too short | all vs without phases < 100 ms (12 of 119) | same drop in PR; mean consistency ±0.01 (only `Orient` changes, left with 4 repetitions) | §10.7 |
# | binning for GPFA | elephant's vs ours | few different bins (events on the border): we use ours | §10 |
# | GPFA iterations | 100 (notebook 02), 500, 1000 | at 100 it has not finished; 500 ≈ 1000 | §10.1 |
# | number of factors d | 2, 5, 10, 15, 20, 25, 30, 40, 50 (cross-validation) | maximum score at 25-30, then it falls (overfitting); 25 chosen: beats 20 in 5 blocks out of 5, ties with 30 | §10.2 |
# | initial time scale | 50, 100, 200 ms | the fastest τ (~45-50 ms) is stable, the others shift: the range is reliable, not the single value | §10.3 |
# | pieces used to fit GPFA | 20 bins (elephant default) vs whole windows (50 bins) | slow scales do not stabilize; score on unseen data +0.09 (5 blocks out of 5); same dimensions; 7 times slower | §10.6 |
# | factors for FA and GPFA | 10, 15, 20, 25, 30 | shared PR from ~6.5 to ~13.4: grows with the factors; FA and GPFA agree at every value | §10.4 |
# | FA vs GPFA at equal data and factors | with / without model of time | same dimensions; only GPFA cleans the trajectories | §10.4, §10.5 |
#
# **The pipeline in one sentence:** we take the `spike_times` (already thresholded at −4.5 × RMS), count them
# in 20 ms bins, keep the active channels, smooth along time, bring every channel
# to the same scale and run PCA.
#
# **What we learned about PCA and FA.**
#
# 1. Single-trial PCA sees ~106 dimensions, but they are mostly noise: averaging the 7-8
#    repetitions of each of the 16 conditions (phase × task version) it drops to ~20 (at equal
#    number of points: from ~50 to ~21), and would fall further with more repetitions.
# 2. FA separates the shared signal from the private noise **without averaging**: on the same
#    instants as the means it finds ~7 effective dimensions (the means ~20), and this estimate depends little
#    on smoothing, unlike that of PCA.
# 3. Neither PCA nor FA produces clean single-trial trajectories: they treat each bin on its own.
# 4. GPFA adds the model of time, with a time scale learned for each factor (from
#    ~50 ms to hundreds of ms). It sustains more factors than FA without overfitting (25 versus 10). On
#    "how many dimensions", at equal factors, it agrees with FA (~12 with 25 factors); on
#    single-trial trajectories it is the best linear method, with no averaging and without choosing σ by hand.
# 5. These conclusions do not depend on the window duration (0.5, 0.75 or 1 s) nor on the
#    repetitions in which the phase lasts a single bin (§10.7).
#
# ### References
#
# - Karpowicz, B. M., Ye, J., et al. (2024). *Few-shot Algorithms for Consistent Neural Decoding
#   (FALCON) Benchmark*. NeurIPS Datasets & Benchmarks Track.
# - Downey, J. E., Brane, L., Gaunt, R. A., Tyler-Kabara, E. C., Boninger, M. L., Collinger, J. L.
#   (2017). Motor cortical activity changes during neuroprosthetic-controlled object interaction.
#   *Scientific Reports*, 7, 16947. doi:10.1038/s41598-017-17222-3
# - Quick, K. M., Weiss, J. M., Clemente, F., Gaunt, R. A., Collinger, J. L. (2020). Intracortical
#   microstimulation feedback improves grasp force accuracy in a human using a brain-computer
#   interface. *IEEE EMBC 2020*, 3355–3358. doi:10.1109/EMBC44109.2020.9175926
# - Flesher, S. N., et al. (2021). A brain-computer interface that evokes tactile sensations
#   improves robotic arm control. *Science*, 372(6544), 831–836. doi:10.1126/science.abd0380
# - Churchland, M. M., et al. (2012). Neural population dynamics during reaching. *Nature*, 487,
#   51–56.
# - Yu, B. M., Cunningham, J. P., Santhanam, G., Ryu, S. I., Shenoy, K. V., Sahani, M. (2009).
#   Gaussian-process factor analysis for low-dimensional single-trial analysis of neural population
#   activity. *Journal of Neurophysiology*, 102, 614–635.
# - Williamson, R. C., Cowley, B. R., Litwin-Kumar, A., Doiron, B., Kohn, A., Smith, M. A., Yu, B. M.
#   (2016). Scaling properties of dimensionality reduction for neural populations and network models.
#   *PLoS Computational Biology*, 12(12), e1005141.
