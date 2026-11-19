# %% [markdown]
# # 04 · "Deep" methods: CEBRA and LFADS · FALCON H1
#
# **Goal.** In notebooks 02 and 03 every method followed a fixed recipe to go from the data to the
# coordinates: linear projections (PCA, FA, GPFA) or constructions on the neighbor graph (Isomap,
# UMAP). Here the recipe is learned by a **neural network**, that is, a nonlinear function with many
# parameters fitted to the data (review, §4.2).
#
# - **Part A, CEBRA** (Schneider, Lee & Mathis 2023), this notebook.
# - **Part B, LFADS** (Pandarinath et al. 2018), §9-§15, after part A.
#
# **CEBRA in one sentence.** A network learns to place time points that "go together" **close** in the
# map and randomly chosen ones **far apart**. What "go together" means is our choice:
#
# - **CEBRA-Time**: time points that are close in time. It uses no labels;
# - **CEBRA-Behavior**: time points with similar behavior. It uses labels.
#
# Two points to keep in mind throughout the notebook:
#
# - **CEBRA is not a generative model.** It has only the half that goes from the data to the map (the
#   *encoder*), not the half that reconstructs the data from the map. For this reason we call it "deep"
#   or "contrastive", never "generative".
# - **CEBRA-Time brings together time points that are close in time**: this is exactly the "time chain"
#   that we had to avoid in 03 (§2). We will have to show that the map contains more than that.
#
# The questions:
#
# 1. With 2-3 coordinates, does CEBRA-Time decode the task phases better than PCA, Isomap and UMAP? And
#    compared with a "ceiling" built on equal terms, with the same input and the same time window?
# 2. What does it learn: the structure of the task or only the time chain?
# 3. With labels (CEBRA-Behavior), how much is gained compared with a linear method that uses the
#    same labels?
#
# The pipeline:
#
# ```
# spike_times ──(as in 02/03: 20 ms bins, 160 active channels, σ = 50 ms, z-score)──► Z    12233 × 160
#
# Z ──(§1: 25-component PCA estimated on the training trials only, applied to the whole session)──► CEBRA input
#
# input ──(CEBRA: network that reads a time window)──► low-dimensional map
#
# evaluation: the same 1106 points as in 03 (movement phases, one bin every 100 ms), on unseen trials
# ```
#
# The notebook is built **one step at a time**: §1 data and common yardstick · §2 the idea on a
# synthetic example · §3 CEBRA-Time, one training run · §4 evaluation with the common yardstick ·
# §5 controls · §6 CEBRA-Behavior · §7 sensitivity and alternatives · §8 summary.
# **Current status: complete. Part A (CEBRA), §0-§8; part B (LFADS), §9-§15.**
#
# Same session as all the notebooks (`held-in-calib`, `19250108T110520`).
#
# **Bins from before 30 Oct 2026 (`bins="legacy"`).** This notebook loads the data with
# `io.load_session(..., bins="legacy")`: bin k counts the spikes in the 20 ms *after* kinematics sample k,
# the convention of `io.py` up to 30 Oct 2026 (row k is row k+1 of FALCON). It stays this way
# because the models saved in `cache_04` depend on the counts, and they are not retrained: it is the record
# of how the hyperparameters were chosen. **The thesis numbers are those of the replication on the
# same session with the FALCON bins and the same hyperparameters** (`rep/19250108T110520/04_rep.out.ipynb`,
# table in `rep/tabelle/19250108T110520.md`; the cells not rerun in the replication are listed in
# `rep/cfg04.py`).
#
# **Running time:** about 6-7 minutes if the trainings are already saved on disk (§3.1);
# about 2 hours the first time (much more with the computer under load: §7 alone took
# almost 2 hours). With the computer under load it can take much longer (one run
# lasted 39 minutes).

# %% [markdown]
# ## 0 · Setup

# %%
import os                                         # to set environment variables before the other imports
os.environ.setdefault("KMP_WARNINGS", "off")      # hides a harmless OpenMP message that would appear with UMAP
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")  # numba (used by UMAP) does not load a second copy of OpenMP: together with the one from scikit-learn the kernel sometimes hangs (as in 03)
import sys                                        # to add src/ to the paths where Python looks for modules
import warnings                                   # to hide harmless warnings
warnings.filterwarnings("ignore")                 # scikit-learn and UMAP print harmless warnings (also when importing UMAP)
from pathlib import Path                          # file paths, independent of the operating system

import numpy as np                                # arrays and linear algebra
import pandas as pd                               # tables for the summaries
import matplotlib.pyplot as plt                   # figures
from joblib import Parallel, delayed              # to run the repetitions with different seeds in parallel
from sklearn.decomposition import PCA             # PCA (input of the methods and comparison method)
from sklearn.preprocessing import StandardScaler  # column-by-column z-score
from sklearn.manifold import Isomap               # Isomap, comparison method (notebook 03)
from sklearn.neighbors import KNeighborsClassifier  # the "neighbors" classifier of the common yardstick
from sklearn.model_selection import GroupKFold    # cross-validation that keeps the bins of the same trial together
from sklearn.metrics import balanced_accuracy_score  # accuracy that weights all phases equally
from umap import UMAP                             # UMAP, comparison method (notebook 03, §9)

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()  # project root, whichever folder the notebook starts from
sys.path.insert(0, str(ROOT / "src"))             # makes the project package importable
from thesis_h1 import io                            # data loading, phases, colors (checked step by step in 02_new)

FIGS = ROOT / "figures"                           # folder of the PNG figures
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False})  # same plot style as the other notebooks

SESS = ROOT / "data/raw/sub-HumanPitt-held-in-calib/sub-HumanPitt-held-in-calib_ses-19250108T110520.nwb"  # analyzed session

# %% [markdown]
# ## 1 · The data and the common yardstick
#
# Before introducing CEBRA we redo the starting point of 03: same data, same evaluation points, same
# cross-validation, same numbers. Then we make the three data decisions that
# concern only CEBRA (§1.3-§1.5).
#
# ### 1.1 · The same data as notebook 03
#
# 20 ms bins, 160 active channels (≥ 0.5 Hz), Gaussian smoothing with σ = 50 ms, per-channel
# z-score: this is the pipeline rebuilt by hand in 02_new and used in 03 (§1). To each bin we attach two
# labels that do **not** enter CEBRA-Time but serve to evaluate it: the **task phase** and the
# **trial**.
#
# The **evaluation points** are those of 03 (§2): the bins of the 8 movement phases, one every 5
# (one every 100 ms), so as not to measure the time chain. The **cross-validation** is the same:
# 5 blocks of 3 trials (`GroupKFold`), so each trial ends up in the test set exactly once.

# %%
s = io.load_session(SESS, bins="legacy")                    # counts at 20 ms (bins from before 30/10, see the header), kinematics, phases
ACTIVE = io.active_units(s, min_hz=0.5)                     # the 160 active channels
Z = StandardScaler().fit_transform(                         # per-channel z-score...
    io.smooth_rates(s.spikes, sigma_ms=50)[:, ACTIVE].astype(float))  # ...of the smoothed firing rates with σ = 50 ms
n_bins = s.n_bins                                           # number of bins in the session

def base_tag(tag):                                          # "Presentation3" -> "Presentation": one label for all presentations
    return "Presentation" if tag.startswith("Presentation") else tag

phase = np.array([base_tag(p) for p in io.phase_per_bin(s)], dtype=object)  # phase of each bin ("" if no phase covers it)
trial = s.trial_num                                         # trial of each bin (from 1 to 15)
MOV = np.isin(phase, io.MOVEMENT_PHASES)                    # true for the bins that fall in a movement phase

STEP = 5                                                    # one bin every 5 = one every 100 ms (03, §2)
K_PCA = 25                                                  # dimensions of the initial PCA (03, §2: the number of factors chosen for GPFA)
K_NN = 10                                                   # neighbors of the classifier and of Isomap/UMAP (03)
bins_all = np.flatnonzero(MOV)                              # all the bins of the movement phases
bins = bins_all[bins_all % STEP == 0]                       # the evaluation points: one every 100 ms
FOLDS = list(GroupKFold(n_splits=5).split(bins, groups=trial[bins]))  # the 5 blocks of 03: (training indices, test indices) among the points
TEST_TRIALS = [np.unique(trial[bins[tst]]) for _, tst in FOLDS]       # the 3 test trials of each block

print(f"Z: {Z.shape[0]} bins × {Z.shape[1]} channels, {len(np.unique(trial))} trials, {n_bins * io.BIN_S / 60:.1f} minutes")
print(f"Evaluation points: {len(bins)} (movement bins: {len(bins_all)})")
print("Test trials per block:", {k: t.tolist() for k, t in enumerate(TEST_TRIALS)})

# %% [markdown]
# **The kinematics.** In §6 (CEBRA-Behavior) we will be able to use the kinematics as a label. The file
# contains two versions of it, both with 7 degrees of freedom (translation x, y, z, orientation, three
# grasp degrees): the **position** (`s.kin`) and the **velocity** (`s.vel`, the official target
# of the FALCON benchmark). We check that the second is really the variation of the first.

# %%
dkin = np.diff(s.kin, axis=0) / io.BIN_S                    # change in position from one bin to the next, per second
r_vel = [np.corrcoef(dkin[:, j], s.vel[1:, j])[0, 1] for j in range(7)]  # correlation with the velocity in the file, degree of freedom by degree of freedom
print(f"Position {s.kin.shape}, velocity {s.vel.shape}")
print("Correlation between change in position and velocity:", dict(zip(io.KIN_LABELS, np.round(r_vel, 2))))

# %% [markdown]
# **Reading.** The correlation is high for all 7 degrees of freedom (from 0.73 to 0.91): `s.vel` is
# the velocity of the position `s.kin`. It is not 1, probably because the file computes the velocity
# with a filter or a lag different from the plain difference between consecutive bins; we did not
# check this, and for §6 it changes nothing.
#
# ### 1.2 · The common yardstick: redoing the numbers of 03
#
# To compare CEBRA with everything else we use **a single measure**, the one of 03 (§6 and §9.1):
#
# - **phase decoding**: a 10-neighbor classifier looks at the coordinates of a point and
#   says which task phase it is in. **Balanced accuracy**: the mean, over the 8 phases, of the
#   fraction of points correctly recognized; at chance it is 1/8 = 12.5%;
# - **clean estimate**: in each block the method (initial PCA included) is estimated **only on the
#   training trials**; the points of the test trials are projected afterwards, with `transform`. The "shortcut" of
#   03 (§7), that is, estimating the method on all trials and cross-validating only the
#   classifier, inflates the result and is forbidden;
# - the **ceiling**: the same classifier with no map at all, on the 25 dimensions.
#
# The functions are rewritten from 03 (§9.1). We run them with 2 and 3 dimensions: §4 needs
# both, because CEBRA's "3 dimensions" are not equivalent to 3 PCA components (we will see this).
# The numbers must match those of 03.

# %%
PCA_F = [PCA(K_PCA).fit(Z[bins_all[~np.isin(trial[bins_all], tt)]])  # in each block, the 25-component PCA sees only the movement bins of the training trials (03, §9.1)
         for tt in TEST_TRIALS]
X25 = [p.transform(Z) for p in PCA_F]                       # for each block, the whole session projected onto its 25 directions (12233 × 25)
INP_PC = (bins, [(X25[k][bins[trn]], X25[k][bins[tst]], trn, tst)  # the input of the common yardstick: points, and for each block (training, test, indices)
                 for k, (trn, tst) in enumerate(FOLDS)])

def decode_map(inp, embed):                                 # balanced accuracy (%) with any map, clean estimate (03, §9.1)
    b, folds = inp
    pred = np.empty(len(b), dtype=object)                   # the predicted phase for each point, when it is in the test set
    for Xtr, Xte, trn, tst in folds:
        Etr, Ete = embed(Xtr, Xte)                          # the map is estimated on the training set; the test points are projected
        pred[tst] = KNeighborsClassifier(n_neighbors=K_NN).fit(Etr, phase[b[trn]]).predict(Ete)
    return balanced_accuracy_score(phase[b], pred) * 100

def emb_pca(d):                                             # PCA with d dimensions on the 25 training PCs (03, §9.1)
    def f(Xtr, Xte):
        p_ = PCA(d).fit(Xtr)
        return p_.transform(Xtr), p_.transform(Xte)
    return f

def emb_isomap(d):                                          # Isomap with k = 10 (03, §6)
    def f(Xtr, Xte):
        i_ = Isomap(n_neighbors=K_NN, n_components=d).fit(Xtr)
        return i_.embedding_, i_.transform(Xte)
    return f

def emb_umap(d, seed=0):                                    # UMAP with n_neighbors = 10, min_dist = 0.1 (03, §9.1)
    def f(Xtr, Xte):
        u = UMAP(n_neighbors=K_NN, n_components=d, min_dist=0.1, random_state=seed, n_jobs=1).fit(Xtr)
        return u.embedding_, u.transform(Xte)
    return f

def emb_none(Xtr, Xte):                                     # no map: the classifier works on the 25 dimensions (the ceiling)
    return Xtr, Xte

SEEDS = range(5)                                            # the 5 seeds of the measures that depend on chance

def mean_range(a):                                          # "mean (min-max)" of a series of repetitions
    return f"{np.mean(a):.1f} ({np.min(a):.1f}-{np.max(a):.1f})"

REF_03 = {"PCA, d = 2": "26.0", "PCA, d = 3": "32.4", "Isomap, d = 2": "31.8", "Isomap, d = 3": "36.0",  # the numbers of 03 (§6 and §9.1), to be reproduced
          "UMAP, d = 2": "41.2 (40.4-42.3)", "UMAP, d = 3": "42.6 (41.9-43.2)", "tetto, 25 dimensioni": "42.2"}

ACC = {}                                                    # the accuracies of this notebook: name -> value (or list of 5 values)
for d in (2, 3):
    ACC[f"PCA, d = {d}"] = decode_map(INP_PC, emb_pca(d))
    ACC[f"Isomap, d = {d}"] = decode_map(INP_PC, emb_isomap(d))
    ACC[f"UMAP, d = {d}"] = Parallel(n_jobs=5)(delayed(decode_map)(INP_PC, emb_umap(d, sd)) for sd in SEEDS)  # 5 seeds in parallel
ACC["tetto, 25 dimensioni"] = decode_map(INP_PC, emb_none)

tab_ref = pd.DataFrame({"here": {k: mean_range(v) if isinstance(v, list) else f"{v:.1f}" for k, v in ACC.items()},
                        "notebook 03": REF_03})
display(tab_ref)

# %% [markdown]
# **Reading.** All the numbers match those of 03 to the decimal, including the minimum and maximum over the
# 5 UMAP seeds: the common yardstick is the same. The starting point for CEBRA is therefore: with 3
# dimensions, PCA 32.4%, Isomap 36.0%, UMAP 42.6%; with no map, on the 25 dimensions, 42.2%.
#
# ### 1.3 · Choice A: what CEBRA receives
#
# Three possibilities:
#
# | input | pros | cons |
# |---|---|---|
# | **25 PCs of the smoothed firing rates** (PCA estimated on the training trials) | it is the input of PCA, Isomap and UMAP in the table of 03: only the method changes | the PCA might remove information that a network could use |
# | 160 smoothed channels | no pre-reduction: the network chooses by itself which combinations to use | input different from that of the table of 03 |
# | 160 channels, raw counts | it is the "rawest" input, often used with CEBRA: the network's time window can act as smoothing | in 03 the raw counts gave 17% (§9.1) |
#
# **We choose the 25 PCs**: it is the only way to attribute to CEBRA, and not to the input, a difference
# from the table of 03. The other two possibilities are among the alternatives of §7.
#
# A difference from 03: there the methods received only the evaluation points, here CEBRA
# reads a **continuous sequence** in time (§1.4). The PCA of each block is still estimated on the
# movement bins of the training trials only, as in §1.2, but we apply it to the **whole session**.
# They are therefore exactly the same 25 PCs as in the common yardstick.

# %%
same = all(np.array_equal(X25[k][bins[trn]], Xtr) and np.array_equal(X25[k][bins[tst]], Xte)  # the evaluation points taken from the continuous sequence...
           for k, (Xtr, Xte, trn, tst) in enumerate(INP_PC[1]))                               # ...coincide with those of the common yardstick?
print(f"CEBRA input: 5 sequences (one per block) of {X25[0].shape[0]} bins × {X25[0].shape[1]} PCs; "
      f"evaluation points identical to those of §1.2: {same}")
print("Variance of the movement bins retained by the 25 components, per block (%):",
      np.round([p.explained_variance_ratio_.sum() * 100 for p in PCA_F], 1))

# %% [markdown]
# **Reading.** The evaluation points taken from the continuous sequence are bit-for-bit identical to those
# used by the common yardstick: CEBRA and the other methods are judged on the same numbers. In every
# block the 25 components retain about 40% of the variance of the movement bins.

# %% [markdown]
# ### 1.4 · Choice B: which bins CEBRA learns from
#
# CEBRA-Time learns from pairs of time points that are close in time, so it needs a continuous "film".
# Two possibilities:
#
# - **all the bins of the training trials**, pauses included (`Presentation`, `Intertrial`): the film
#   is whole;
# - only the movement bins: the film would be cut into pieces, with a jump at every phase change.
#
# **We choose all the bins of the training trials.** The movement bins alone are the alternative of
# §7. In both cases the evaluation stays on the same movement points as in §1.2, and the
# classifier learns only from the points of the training trials.

# %%
TRAIN_BINS = [np.flatnonzero(~np.isin(trial, tt)) for tt in TEST_TRIALS]  # for each block, all the bins of the 12 training trials, in time order
tab_bins = pd.DataFrame([{"test trials": ", ".join(map(str, tt)),
                          "training bins (all)": len(tb),
                          "of which movement": int(MOV[tb].sum()),
                          "of which pause": int((~MOV[tb]).sum()),
                          "evaluation points, training": len(trn),
                          "evaluation points, test": len(tst)}
                         for tt, tb, (trn, tst) in zip(TEST_TRIALS, TRAIN_BINS, FOLDS)],
                        index=pd.Index(range(5), name="block"))
display(tab_bins)

# %% [markdown]
# **Reading.** In every block CEBRA learns from about 9700-9900 bins (a little over 3 minutes), and **only
# 45% of them are movement**: more than half of the film is pauses, mostly `Presentation`, when the
# participant looks at the cue with the arm still. This is worth keeping in mind: part of what
# CEBRA learns will serve to describe the pauses, which we do not evaluate. The classifier learns from 879-892
# points and is judged on 214-227 points per block.
#
# ### 1.5 · Choice C: continuous time and the "seams"
#
# When we remove the 3 test trials and glue the 12 training trials together, where a test trial used to be two pieces
# of session far apart in time end up one after the other: this is a **seam**, as in a film strip
# cut and glued back together. CEBRA can "look across" a seam in two ways:
#
# - the **positive pair** of CEBRA-Time (a time point and the one Δ bins later): if there is a
#   seam between the two, the network believes two time points that are not close in time to be close;
# - the network's **time window**: for each point the network reads a few bins before and after, and
#   near a seam the window mixes two pieces of session.
#
# There are two possibilities: accept the seams, or cut the sequence at every trial and forbid
# pairs and windows that straddle a cut. The second is not provided by CEBRA's simple interface (it would need
# the low-level one). Before choosing we measure how much the seams matter, with the
# standard values that we will use in §3 (Δ = 10 bins, window of 10 bins) and that we will vary in §7.
#
# **The test trials** do not have this problem: we project them **one at a time**, so the network's window
# never sees bins of other trials. At the trial edges the library repeats the first and
# last bin (`pad_before_transform`).

# %%
DELTA, WIN = 10, 10                                         # positive pair at 10 bins (200 ms) and window of 10 bins: the standard values of §3

def seam_stats(tb):                                         # seams of a training sequence and how much they matter
    seams = np.flatnonzero(np.diff(tb) > 1) + 1             # positions in the sequence where the previous bin is not the one 20 ms earlier
    i = np.arange(len(tb) - DELTA)                          # the possible first elements of a pair (time point, time point + Δ)
    pair_cross = np.isin(i[:, None] + np.arange(1, DELTA + 1)[None, :], seams).any(1)  # does the pair straddle a seam?
    j = np.arange(WIN // 2, len(tb) - WIN // 2)             # the points whose window (5 bins before, 5 after) lies entirely within the sequence
    win_cross = np.isin(j[:, None] + np.arange(-WIN // 2 + 1, WIN // 2)[None, :], seams).any(1)  # does the window contain a seam?
    return {"joints": len(seams), "pairs straddling (%)": pair_cross.mean() * 100,
            "windows straddling (%)": win_cross.mean() * 100}

tab_seams = pd.DataFrame([seam_stats(tb) for tb in TRAIN_BINS], index=pd.Index(range(5), name="block"))
display(tab_seams.round(2))
sides = [(phase[tb[sp - 1]], phase[tb[sp]]) for tb in TRAIN_BINS for sp in np.flatnonzero(np.diff(tb) > 1) + 1]  # phase just before and just after each seam
print("Phase on the two sides of the seams (before, after):", pd.Series(sides).value_counts().to_dict())

# %% [markdown]
# **Reading.** The seams are 1 to 3 per block (in block 4 only one: trials 1 and 2 are
# at the start of the session, and removing them creates no seams) and they affect **at most 0.3%** of the
# pairs and windows: about 3 in 1000. They all fall inside `Release` (for the reason we see
# below): a straddling pair joins the end of one `Release` to the end of another `Release`, two similar
# moments of the task but far apart in time. **We accept the seams** and state it: the effect
# is too small to justify the low-level interface.
#
# **The boundary between two trials.** Looking at where the trial number changes we found something
# that had not emerged in the previous notebooks: the boundary **falls inside the `Release` phase** of the
# previous trial, a few bins before its end. The last bins of each `Release` therefore carry the
# number of the next trial. We measure how many of the evaluation points are affected, and how many
# are so close to a boundary that the smoothing (σ = 50 ms, reaching about 150 ms) mixes the
# counts of the two trials.

# %%
starts = np.flatnonzero(np.diff(trial) != 0) + 1            # first bin of each trial (from the second on)
tail = [int(np.argmax(phase[a:] != "Release")) for a in starts]  # how many Release bins there are after the boundary, in the new trial
print(f"The boundary falls in Release in {np.mean(phase[starts] == 'Release') * 100:.0f}% of cases; "
      f"Release bins after the boundary: from {min(tail)} to {max(tail)} (from {min(tail) * 20} to {max(tail) * 20} ms)")

run_start = np.r_[0, np.flatnonzero(phase[1:] != phase[:-1]) + 1]  # start of each stretch of constant phase
first_of_run = run_start[np.searchsorted(run_start, np.arange(n_bins), side="right") - 1]  # for each bin, the first bin of its stretch of phase
split_pts = trial[bins] != trial[first_of_run[bins]]        # points whose phase began in the previous trial
near = np.min(np.abs(bins[:, None] - starts[None, :]), axis=1) <= 7  # points no more than 7 bins (140 ms) from a boundary
print(f"Evaluation points in a Release 'split' by the boundary: {split_pts.sum()} of {len(bins)}")
print(f"Evaluation points within 150 ms of a trial boundary: {near.sum()} of {len(bins)} "
      f"({pd.Series(phase[bins][near]).value_counts().to_dict()})")

# %%
fig = plt.figure(figsize=(14, 6.2))
gs = fig.add_gridspec(2, 1, height_ratios=(1.5, 1))
ax = fig.add_subplot(gs[0])                                 # top: the 5 blocks, trial by trial
t_min = np.arange(n_bins) * io.BIN_S / 60                   # time of each bin, in minutes
for k, tt in enumerate(TEST_TRIALS):
    for t_ in np.unique(trial):                             # one rectangle per trial: gray = training, red = test
        m = trial == t_
        ax.barh(k, t_min[m][-1] - t_min[m][0] + io.BIN_S / 60, left=t_min[m][0], height=0.6,
                color="C3" if t_ in tt else "0.8", edgecolor="white", lw=0.5)
        if t_ in tt:
            ax.text(t_min[m].mean(), k, str(t_), ha="center", va="center", fontsize=8, color="white")
    tb = TRAIN_BINS[k]
    for sp in np.flatnonzero(np.diff(tb) > 1) + 1:          # the seams: where the training sequence skips a test trial
        ax.plot(t_min[tb[sp]], k, marker="v", color="k", ms=7)
ax.set(yticks=range(5), yticklabels=[f"block {k}" for k in range(5)], xlabel="time (minutes)",
       title="Training (gray) and test (red, with the number) trials; ▼ = seam in the training sequence")
ax.invert_yaxis(); ax.grid(False)

ax2 = fig.add_subplot(gs[1])                                # bottom: zoom around the first boundary
a0 = starts[0]
win_b = np.arange(a0 - 75, a0 + 25)                         # 1.5 s before and 0.5 s after the boundary
for b_ in win_b:
    ax2.axvspan(b_ * io.BIN_S, (b_ + 1) * io.BIN_S, ymin=0.55, ymax=1, color=io.PHASE_COLORS.get(phase[b_], "white"), lw=0)
ax2.step(win_b * io.BIN_S, trial[win_b], where="post", color="k", lw=1.5)  # the trial number
ax2.axvline(a0 * io.BIN_S, color="k", ls="--", lw=1)
ax2.set(ylim=(trial[a0] - 1.6, trial[a0] + 1.2), yticks=[trial[a0] - 1, trial[a0]], xlabel="time (s)", ylabel="trial",
        title=f"Around the first boundary: colored band = phase ({', '.join(sorted(set(phase[win_b]) - {''}))}); line = trial number")
handles = [plt.Rectangle((0, 0), 1, 1, color=io.PHASE_COLORS[p]) for p in sorted(set(phase[win_b]) - {""})]
ax2.legend(handles, sorted(set(phase[win_b]) - {""}), loc="lower left", fontsize=8, ncol=3)
fig.tight_layout(); fig.savefig(FIGS / "04_folds.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# **Reading.**
#
# - The boundary between two trials **always** falls inside `Release`, 1 to 4 bins (20-80 ms) before
#   its end: in the file, every trial begins with the tail of the previous `Release` (bottom figure).
# - **6 points out of 1106** belong to a `Release` that began in the previous trial. If one of the two
#   trials is in the test set and the other in training, that repetition of `Release` is split between training and
#   test. **27 points** (all of `Release`) are close enough to a boundary for the smoothing
#   to mix the counts of the two trials.
# - It is few (at most 2.4% of the points) and it held identically in 03, which used the same blocks:
#   it does not change the comparison. We leave it as it is so as not to change the common yardstick. A possible
#   correction would be to move the boundary to the end of `Release`; we did not do it.
# - Nothing is added for CEBRA: the test trials are projected on their own, so the network's window
#   never links them to the training trials.

# %% [markdown]
# ## 2 · The idea of CEBRA on a synthetic example
#
# ### What CEBRA does
#
# CEBRA (Schneider, Lee & Mathis 2023) is made of two pieces:
#
# 1. **A network** (the *encoder*) that reads a short **time window** around each time point,
#    here 10 bins (200 ms: 100 ms before and 100 ms after), and returns a point in a few dimensions.
#    With the **cosine** distance, the standard choice, the point is "normalized": it always lies on
#    a **sphere** of radius 1. A sphere in 3 dimensions has only 2 free coordinates, like a
#    globe (latitude and longitude).
# 2. **A game** with which the network trains. At each step we take:
#    - a **reference** time point;
#    - a **positive** time point, which "goes together" with the reference: in CEBRA-Time it is the time point
#      **Δ bins later** (here Δ = 10, that is 200 ms);
#    - many **negative** time points, taken at random in the session.
#
#    The network wins if, in the map, the reference is **closer to the positive than to the negatives**. It is
#    like recognizing a friend in a crowd from a photo taken 200 ms ago: you learn to look at what
#    stays the same (the face), not at what changes at every moment (the expression).
#
# **Why it works on noisy neural data.** The signal changes slowly, so it is almost the same
# at t and at t + 200 ms; the noise changes at every bin, so it does not help to recognize the positive. To
# win, the network learns to keep what is slow and to discard the rest (the idea of *time-contrastive
# learning*, Hyvärinen & Morioka 2016). **Caution**: "what is slow" is not necessarily the task.
# "Where in the trial we are" also changes slowly: it is the time chain, which we will check in
# §5.
#
# ### The cost function: InfoNCE
#
# For a reference x, its positive y⁺ and N negatives y₁…y_N, the **similarity** between two points
# of the map is the cosine of the angle between them, s ∈ [−1, 1], divided by a **temperature** τ.
# CEBRA minimizes, on average over the group (*batch*) of references:
#
# $$L = -\frac{s(x, y^+)}{\tau} + \log \sum_{i=1}^{N} e^{\,s(x, y_i)/\tau}$$
#
# - the first term rewards closeness to the positive (the library calls it *align*);
# - the second penalizes closeness to the negatives and pushes the points to **spread over the whole sphere**
#   (*uniform*; the two names come from Wang & Isola 2020).
#
# It is the **InfoNCE** function (van den Oord et al. 2018), with a difference that we will check in the
# library code: in the original version the sum of the second term also includes the
# positive, in CEBRA only the negatives.
#
# Two useful numbers, which we derive from the formula:
#
# - if the map "collapses" (all points in the same place), all similarities are equal and
#   L = log N: this is the **chance level**. CEBRA reports the **goodness of fit** in bits, that is how far
#   it falls below this level: (log N − L) · log₂ e;
# - with the cosine distance and τ = 1 there is also a **minimum**: in the best case s = 1 with the positive and
#   s = −1 with all the negatives, so L ≥ log N − 2, that is at most 2/ln 2 ≈ 2.9 bits.
#
# ### Why a synthetic example
#
# Like the rolled sheet of 03 (§3): before the real data we try CEBRA on data for which we
# **know the answer**.
#
# ### 2.1 · The synthetic data: a point wandering on the rolled sheet
#
# - A point moves **slowly** on a square sheet: two coordinates, u (the length of the
#   sheet) and v (the width), between 0 and 1. The velocity changes direction gradually, and at the edges
#   the point bounces.
# - The sheet is **rolled up** in 3D space, as in 03.
# - We read it through **30 "channels"**: each channel is a random combination of the 3 coordinates
#   of the roll plus a **noise** that is different at every bin, stronger than the signal (double amplitude).
# - 10 000 bins of 20 ms (200 s): roughly the length of the training film of the real data (§1.4).
#   CEBRA trains on the first 80% of the time; the last 20% is the test.
# %%
from scipy.ndimage import gaussian_filter1d, uniform_filter1d  # Gaussian smoothing and moving average along time

T_SYN, N_CH, NOISE = 10_000, 30, 2.0                        # bins, channels, noise amplitude (the signal of each channel has amplitude about 1)
rng_syn = np.random.default_rng(0)                          # fixed seed: same synthetic data at every run
vel = gaussian_filter1d(rng_syn.normal(size=(T_SYN, 2)), sigma=25, axis=0)  # velocity that changes direction gradually (over about 0.5 s)
vel = vel / vel.std(0) * 0.004                              # typical step: 0.4% of the sheet every 20 ms
free = 0.5 + np.cumsum(vel, axis=0)                         # the path the point would take on a sheet without edges, starting from the center
UV = 1 - np.abs((free % 2) - 1)                             # "folded" into [0, 1]: it is the path that bounces off the edges (coordinates u, v)
t_roll = 1.5 * np.pi * (1 + 2 * UV[:, 0])                   # from u to the roll angle, as in make_swiss_roll (03, §3)
ROLL = np.c_[t_roll * np.cos(t_roll), 21 * UV[:, 1], t_roll * np.sin(t_roll)]  # the point on the roll, in 3D
ROLL = (ROLL - ROLL.mean(0)) / ROLL.std(0)                  # each coordinate with variance 1
W_syn = rng_syn.normal(size=(3, N_CH)) / np.sqrt(3)         # how each channel combines the 3 coordinates (at random)
SIG = ROLL @ W_syn                                          # the signal of each channel, without noise
X_raw = SIG + NOISE * rng_syn.normal(size=(T_SYN, N_CH))    # plus an independent noise at every bin and in every channel
MU_SYN, SD_SYN = X_raw.mean(0), X_raw.std(0)                # mean and standard deviation of each channel
X_syn = ((X_raw - MU_SYN) / SD_SYN).astype(np.float32)      # z-score; float32 is the format that networks use
N_TR = int(0.8 * T_SYN)                                     # the first 80% of the time is training, the rest is test
print(f"Synthetic data: {T_SYN} bins × {N_CH} channels; training {N_TR} bins, test {T_SYN - N_TR} bins")
print(f"Signal-to-noise ratio per channel (variances): {SIG.var(0).mean() / NOISE ** 2:.2f}")
print(f"Does the point cover the whole sheet in the test too? u from {UV[N_TR:, 0].min():.2f} to {UV[N_TR:, 0].max():.2f}, "
      f"v from {UV[N_TR:, 1].min():.2f} to {UV[N_TR:, 1].max():.2f}")

# %% [markdown]
# The noise has about 4 times the variance of the signal in every channel: a single bin is almost all
# noise, as in the real data. The last 20% of the time, the test, also covers the whole sheet:
# the R² of §2.4 is not distorted by a test that stays confined to one corner.

# %% [markdown]
# ### 2.2 · The pair game, on an example
#
# On the left, one channel over time: the black curve is the signal without noise, the gray one what
# CEBRA sees. On the right, the path on the sheet: a reference time point, its positive 200 ms
# later and a few negatives taken at random. The positive is close on the sheet, the negatives are not: this is the
# difference that the network must learn to see through the noise.

# %%
DELTA_SYN = 10                                              # the positive of CEBRA-Time: 10 bins (200 ms) after the reference
rng_pair = np.random.default_rng(1)                         # fixed seed for the example
ref0 = 2_000                                                # an arbitrary reference time point
negs0 = rng_pair.integers(0, N_TR - DELTA_SYN, 15)          # 15 negatives, taken at random as CEBRA does (in the whole training set)
t_ax = np.arange(T_SYN) * io.BIN_S                          # time of each bin, in s

fig, ax = plt.subplots(1, 2, figsize=(13, 4.2), gridspec_kw={"width_ratios": (1.6, 1)})
w_ = slice(ref0 - 100, ref0 + 100)                          # 4 s around the reference
ax[0].plot(t_ax[w_], X_syn[w_, 0], color="0.65", lw=0.8, label="what CEBRA sees (signal + noise)")
ax[0].plot(t_ax[w_], (SIG[w_, 0] - MU_SYN[0]) / SD_SYN[0],  # the signal without noise, with the same z-score as the channel
           color="k", lw=2, label="signal without noise (same scale)")
ax[0].axvline(t_ax[ref0], color="C0", lw=2, label="reference")
ax[0].axvline(t_ax[ref0 + DELTA_SYN], color="C2", lw=2, label="positive (+200 ms)")
ax[0].set(xlabel="time (s)", ylabel="channel 1 (z)", title="One channel: the signal is slow, the noise changes at every bin")
ax[0].legend(fontsize=8, loc="lower left", ncol=2)
ax[1].plot(UV[:N_TR, 0], UV[:N_TR, 1], color="0.85", lw=0.4)  # the whole training path on the sheet
ax[1].plot(UV[ref0 - 100:ref0 + 100, 0], UV[ref0 - 100:ref0 + 100, 1], color="0.3", lw=1.2)  # the 4 s of the left panel
ax[1].scatter(*UV[negs0].T, color="C3", s=30, zorder=3, label="negatives (random)")
ax[1].scatter(*UV[ref0], color="C0", s=80, zorder=4, label="reference")
ax[1].scatter(*UV[ref0 + DELTA_SYN], color="C2", s=80, marker="s", zorder=4, label="positive (+200 ms)")
ax[1].set(xlabel="u (length of the sheet)", ylabel="v (width)", title="On the sheet (not rolled up)", aspect="equal")
ax[1].legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1))
fig.tight_layout(); fig.savefig(FIGS / "04_contrastive_pairs.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# ### 2.3 · How we train CEBRA in this notebook
#
# Three precautions, the same for all the trainings of the notebook:
#
# - **A separate process.** CEBRA's networks use the `torch` library, which brings its own
#   copy of OpenMP, the system that splits computations across cores; scikit-learn has another. Two
#   copies in the same process can hang it, as happened with UMAP in 03. For this reason
#   every training runs in a **separate process** (`ProcessPoolExecutor`), and the notebook never
#   imports `torch`. The training code stays written below: it is just executed
#   elsewhere. (The separate process still contains both copies, because CEBRA in turn imports
#   scikit-learn; but there only `torch` does the work.)
# - **Two trainings at a time, on the Mac's GPU** (MPS). Measured before starting, on dummy data of the same
#   shape: about 10 times faster than the CPU; two trainings together each run at the same speed as a single one.
# - **The seed really fixed.** CEBRA 0.6.0 ignores the seed we give it: we see this, and we
#   fix it, in §2.7.
#
# **The parameters** are those of CEBRA's official examples for hippocampus data: network
# `offset10-model` (window of 10 bins), batches of 512 references, learning rate 3·10⁻⁴,
# τ = 1, Δ = 10 bins, cosine distance, 3-dimensional map. In §3 we compare them with those
# reported in the paper. Here we run 5000 iterations.

# %%
from joblib.externals.loky import ProcessPoolExecutor      # a pool of separate processes, only for CEBRA (different from the one joblib uses for UMAP)

POOL = ProcessPoolExecutor(max_workers=2)                   # two trainings at a time
CEBRA_STD = dict(model_architecture="offset10-model",       # convolutional network with a window of 10 bins (5 before, 5 after)
                 batch_size=512,                            # references per step (and as many negatives)
                 learning_rate=3e-4,                        # how much the network weights are corrected at each step
                 temperature=1.0,                           # τ of the cost function
                 time_offsets=10,                           # Δ: the positive is 10 bins (200 ms) after the reference
                 distance="cosine",                         # similarity = cosine; the points lie on a sphere
                 output_dimension=3,                        # dimensions of the map
                 max_iterations=5000)                       # training steps (for the real data it is decided in §3)

def cebra_fit(X_train, X_apply, cfg, seed, fix_seed=True, monitor=None, every=None, model_path=None):
    """Trains CEBRA-Time on X_train; returns the map of each array in X_apply, the cost function
    at each step and the duration. Runs in a separate process (see above).
    From §3: monitor(step, solver) is called every `every` steps; model_path = where to save the model."""
    import time                                             # to measure the duration
    import numpy as np                                      # arrays
    import torch                                            # the neural network library used by CEBRA
    import cebra                                            # CEBRA
    import cebra.distributions.base as cdb                  # the class that creates CEBRA's random number generators (§2.7)
    torch.set_num_threads(1)                                # the heavy computations are on the GPU: a single CPU thread, less OpenMP at work
    if not hasattr(cdb.HasGenerator, "orig_init"):          # the seed fix (§2.7), installed once per process
        cdb.HasGenerator.orig_init = cdb.HasGenerator.__init__  # the original method, which picks a random seed
        def seeded_init(self, device, seed):                # the corrected method
            cdb.HasGenerator.orig_init(self, device=device, seed=seed)  # does everything it did before...
            if cdb.HasGenerator.fixed_seed is not None:     # ...then, if we have fixed a seed, replaces the generator
                self._seed = cdb.HasGenerator.fixed_seed * 1000 + cdb.HasGenerator.counter  # a different seed for each generator, but always the same
                cdb.HasGenerator.counter += 1               # the next generator will get the next seed
                self._generator = torch.Generator(device=self.device).manual_seed(self._seed)
        cdb.HasGenerator.__init__ = seeded_init
    cdb.HasGenerator.fixed_seed = seed if fix_seed else None  # fix_seed=False: the library behaves as originally
    cdb.HasGenerator.counter = 0                            # the generators of this training are numbered from 0
    torch.manual_seed(seed); np.random.seed(seed)           # seeds of the network's initial weights and of numpy
    device = "mps" if torch.backends.mps.is_available() else "cpu"  # the Mac's GPU, if available
    model = cebra.CEBRA(**cfg, device=device, verbose=False)  # the model with the chosen parameters
    t0 = time.time()
    if monitor is None:
        model.fit(X_train)                                  # the training: without labels, hence CEBRA-Time
    else:
        model.fit(X_train, callback=monitor, callback_frequency=every)  # the same, calling monitor(step, solver) every `every` steps (§3)
        monitor(cfg["max_iterations"], model.solver_)       # and one last time at the end
    seconds = time.time() - t0
    if model_path is not None:
        model.save(str(model_path))                         # the trained model, on disk (§3.1)
    out = {"maps": [model.transform(x) for x in X_apply],   # each sequence projected on its own (edges: the first/last bin is repeated)
           "loss": np.asarray(model.state_dict_["loss"]),   # the cost function at every training step
           "seconds": seconds, "device": device}
    if monitor is not None:
        out["monitor"] = monitor.history                    # the measures collected during training (§3)
    return out

def run_cebra(jobs, fn=None):                               # runs a list of trainings (dictionaries of arguments), two at a time
    fn = cebra_fit if fn is None else fn                    # the training function (from §6 also the one with labels)
    futures = [POOL.submit(fn, **j) for j in jobs]          # sends them all to the queue...
    return [f.result() for f in futures]                    # ...and waits for the results, in the same order

# %% [markdown]
# ### 2.4 · What CEBRA-Time learns on the sheet
#
# We train CEBRA-Time on the first 80% of the time and project the last 20% separately. Then we ask:
# **can the position on the sheet be recovered from the map?** For each test point a *10-neighbor
# regressor* (like the classifier of the common yardstick, but for a number) estimates u and v from the mean of the 10
# closest training points in the map. We measure the R², the fraction of the variability of u and v
# explained (1 = perfect, 0 = as good as always predicting the mean).
#
# Terms of comparison:
#
# - **3-component PCA** on the 30 channels, bin by bin;
# - **3-component PCA after a 200 ms moving average**: the same window that CEBRA reads, but
#   without any learning. It separates how much of CEBRA's result comes from the **mere fact of
#   looking at 200 ms** (this is the idea of the "equal-terms ceiling" of §4);
# - **no map**: the regressor directly on the 30 channels, with and without moving average.
#
# And three versions of CEBRA, to see the sphere problem right away:
#
# - **cosine, 3 dimensions** (standard): sphere, 2 free coordinates;
# - **cosine, 4 dimensions**: sphere in 4D, 3 free coordinates;
# - **Euclidean, 3 dimensions**: no sphere, 3 free coordinates (network `offset10-model-mse`, which
#   does not normalize).
#
# Three trainings, two at a time: about 3 minutes.

# %%
from sklearn.neighbors import KNeighborsRegressor           # neighbor regressor: estimates a number from the mean of the neighbors
from sklearn.metrics import r2_score                        # R²: fraction of the variability explained

SYN_CFG = {                                                 # the three versions of CEBRA-Time (only the distance and the dimensions change)
    "CEBRA-Time, cosine, 3 dimensions (sphere: 2 free coordinates)": dict(CEBRA_STD),
    "CEBRA-Time, cosine, 4 dimensions (sphere: 3 free coordinates)": dict(CEBRA_STD, output_dimension=4),
    "CEBRA-Time, Euclidean, 3 dimensions (3 free coordinates)": dict(CEBRA_STD, distance="euclidean",
                                                                    model_architecture="offset10-model-mse"),
}
std_name = next(iter(SYN_CFG))                              # the standard version (cosine, 3 dimensions)
res_syn = dict(zip(SYN_CFG, run_cebra([dict(X_train=X_syn[:N_TR], X_apply=[X_syn[:N_TR], X_syn[N_TR:]], cfg=c, seed=0)
                                       for c in SYN_CFG.values()])))  # training and test projected separately

def r2_uv(E_tr, E_te):                                      # R² of u and v estimated with 10 neighbors in the map (training -> test)
    pred = KNeighborsRegressor(n_neighbors=10).fit(E_tr, UV[:N_TR]).predict(E_te)
    return {"R² of u (length)": r2_score(UV[N_TR:, 0], pred[:, 0]), "R² of v (width)": r2_score(UV[N_TR:, 1], pred[:, 1])}

Xs_tr = uniform_filter1d(X_syn[:N_TR], size=10, axis=0, mode="nearest")  # moving average of 10 bins (200 ms), training...
Xs_te = uniform_filter1d(X_syn[N_TR:], size=10, axis=0, mode="nearest")  # ...and test, separately (no test bin enters the training)
pca_syn, pca_syn_ma = PCA(3).fit(X_syn[:N_TR]), PCA(3).fit(Xs_tr)  # the two PCAs, estimated on the training set
MAPS_SYN = {                                                # all the maps: (training, test)
    "PCA, 3 components (one bin)": (pca_syn.transform(X_syn[:N_TR]), pca_syn.transform(X_syn[N_TR:])),
    "PCA, 3 components after a 200 ms moving average": (pca_syn_ma.transform(Xs_tr), pca_syn_ma.transform(Xs_te)),
    **{name: tuple(r["maps"]) for name, r in res_syn.items()},
    "no map: 30 channels (one bin)": (X_syn[:N_TR], X_syn[N_TR:]),
    "no map: 30 channels after the moving average": (Xs_tr, Xs_te),
}
tab_syn = pd.DataFrame({name: r2_uv(*E) for name, E in MAPS_SYN.items()}).T
display(tab_syn.round(2))
print("Duration of the three trainings (s, in table order):", [round(r["seconds"]) for r in res_syn.values()],
      "| device:", res_syn[std_name]["device"])
radius = np.linalg.norm(res_syn[std_name]["maps"][0], axis=1)  # distance of each point from the center, in the standard map
print(f"Distance from the center of the points of the standard map (cosine, 3 dimensions): from {radius.min():.4f} to {radius.max():.4f}")

# %%
fig = plt.figure(figsize=(17, 7.6))
panels = [("The true roll (without noise)", ROLL[:N_TR]),
          ("PCA, one bin", MAPS_SYN["PCA, 3 components (one bin)"][0]),
          ("PCA after the moving average", MAPS_SYN["PCA, 3 components after a 200 ms moving average"][0]),
          ("CEBRA-Time, cosine, 3D (sphere)", res_syn[std_name]["maps"][0]),
          ("CEBRA-Time, Euclidean, 3D", res_syn[list(SYN_CFG)[2]]["maps"][0])]
for row, (j, lab) in enumerate([(0, "u (length)"), (1, "v (width)")]):
    for k, (ttl, E) in enumerate(panels):
        a_ = fig.add_subplot(2, 6, row * 6 + k + 1, projection="3d")
        a_.scatter(*E[::4, :3].T, c=UV[:N_TR:4, j], cmap="viridis", s=2)  # one point every 4, colored with the position on the sheet
        a_.set_title(f"{ttl}\ncolor = {lab}", fontsize=9)
        a_.set_xticklabels([]); a_.set_yticklabels([]); a_.set_zticklabels([])
a_ = fig.add_subplot(1, 6, 6)                               # right: the cost function during training (standard version)
loss_std = res_syn[std_name]["loss"]
a_.plot(np.arange(1, len(loss_std) + 1), loss_std, color="0.75", lw=0.5, label="at every step")
a_.plot(np.arange(1, len(loss_std) + 1), uniform_filter1d(loss_std, 100), color="C0", lw=2, label="mean over 100 steps")
a_.axhline(np.log(512), color="0.4", ls="--", lw=1, label="chance level: log N")
a_.axhline(np.log(512) - 2, color="C3", ls=":", lw=1.5, label="lowest possible (τ = 1)")
a_.set(xlabel="training step", ylabel="InfoNCE", title="Cost function\n(cosine, 3D)")
a_.legend(fontsize=7, loc="center right")
fig.tight_layout(); fig.savefig(FIGS / "04_synthetic_maps.png", bbox_inches="tight"); plt.show()
print("Cost function (cosine, 3D), mean over 500 steps:",
      {f"steps {a + 1}-{a + 500}": round(float(loss_std[a:a + 500].mean()), 3) for a in (0, 1000, 2000, 3000, 4500)})

# %% [markdown]
# **Reading.**
#
# - **Bin by bin, the noise wins**: PCA recovers little of the position (R² 0.18 for u, 0.45 for v),
#   and the regressor on the 30 channels even less.
# - **The 200 ms moving average is enough to clean the data**: PCA after the average reaches 0.93 and 0.91,
#   the best result of all. This is no accident: we built these data ourselves as a slow
#   signal mixed **linearly** plus a noise **independent at every bin**, and for such data
#   "average over time + PCA" is the ideal recipe. CEBRA does not beat it here. It is a warning for the real
#   data: the right comparison for CEBRA is with **the same time window without learning** (the
#   "equal-terms ceiling" of §4), not with bin-by-bin PCA.
# - **The sphere matters.** With the cosine distance and 3 dimensions (the standard version) v is recovered well
#   (0.77), u not at all (−0.21, worse than always predicting the mean). In the figure, colored by u, the
#   ends of the roll (purple and yellow) end up next to each other on the sphere. The most
#   likely explanation: the *uniform* term pushes the points to cover **the whole** sphere, and a sheet that
#   covers a ball must close on itself, like the paper around a ball, and the edges
#   touch. The figure suggests this, but we did not measure it directly. With **3 free
#   coordinates** (cosine with 4 dimensions, or Euclidean distance with 3) u becomes readable again: 0.58 and 0.61.
# - Hence the **rule for §4**: compare CEBRA with the other methods at equal numbers of free coordinates
#   (cosine CEBRA with 3 dimensions against d = 2, with 4 dimensions against d = 3) and always with the same
#   window without learning.
# - **The cost function** (right) falls quickly in the first 200 steps, then very slowly:
#   between steps 2001-2500 and 4501-5000 it still drops by 0.016 (from 5.460 to 5.444). It stays far from the lowest possible value:
#   we explain this in §2.5.
#
# ### 2.5 · The cost function computed by hand
#
# We take the standard map (cosine, 3 dimensions) of the training points and redo **one step
# of the game** as CEBRA-Time does it: 512 random references, each with its positive 10 bins
# later, and 512 random negatives, the same for all references (this is what the library does). We compute
# InfoNCE in three ways: by hand with CEBRA's formula, by hand with the original formula (positive
# also in the sum), and with the library function (`FixedCosineInfoNCE`, run in the separate
# process because it requires `torch`).

# %%
E_syn = res_syn[std_name]["maps"][0]                        # the standard map of the training points (8000 × 3), on the sphere
N_B, TAU = 512, 1.0                                         # references per batch and temperature, as in training
rng_b = np.random.default_rng(2)                            # fixed seed for the batch
ref = rng_b.integers(0, N_TR - DELTA_SYN, N_B)              # references: random time points (CEBRA draws them uniformly in [0, T − Δ))
neg = rng_b.integers(0, N_TR - DELTA_SYN, N_B)              # negatives: other random time points, from the same distribution
pos = ref + DELTA_SYN                                       # positives: 10 bins after each reference

def logsumexp(a, axis=1):                                   # log Σ exp(a), computed without exceeding the number limits (the maximum is subtracted)
    m = a.max(axis=axis, keepdims=True)
    return (m + np.log(np.exp(a - m).sum(axis=axis, keepdims=True))).squeeze(axis)

s_pos = (E_syn[ref] * E_syn[pos]).sum(1) / TAU              # similarity (cosine) of each reference with its positive, divided by τ
s_neg = E_syn[ref] @ E_syn[neg].T / TAU                     # similarity of each reference with each of the 512 negatives (512 × 512)
L_cebra = np.mean(-s_pos + logsumexp(s_neg))                # InfoNCE as in CEBRA: only the negatives in the sum
L_orig = np.mean(-s_pos + logsumexp(np.c_[s_pos, s_neg]))   # original InfoNCE: the positive also in the sum

def infonce_library(ref_e, pos_e, neg_e, tau):              # the library function (runs in the separate process)
    import torch, cebra
    crit = cebra.models.criterions.FixedCosineInfoNCE(temperature=tau)
    loss, align, uniform = crit(torch.from_numpy(ref_e), torch.from_numpy(pos_e), torch.from_numpy(neg_e))
    return float(loss), float(align), float(uniform)

L_lib, align_lib, unif_lib = POOL.submit(infonce_library, E_syn[ref], E_syn[pos], E_syn[neg], TAU).result()

def bits(L, n=N_B):                                         # goodness of fit: how far below the chance level, in bits
    return (np.log(n) - L) * np.log2(np.e)

display(pd.DataFrame({"InfoNCE": {"by hand, CEBRA formula": L_cebra, "library (FixedCosineInfoNCE)": L_lib,
                                  "by hand, original formula (positive in the sum)": L_orig,
                                  "training: mean of the last 500 steps": loss_std[-500:].mean(),
                                  "chance level (collapsed map): log 512": np.log(N_B),
                                  "lowest possible with τ = 1: log 512 − 2": np.log(N_B) - 2}}).assign(
    **{"goodness of fit (bit)": lambda d: bits(d["InfoNCE"])}).round(4))
print(f"Difference by hand − library: {L_cebra - L_lib:.2e}")
print(f"The two terms of the library: align = {align_lib:.4f} (by hand {np.mean(-s_pos):.4f}), "
      f"uniform = {unif_lib:.4f} (by hand {np.mean(logsumexp(s_neg)):.4f})")

# %% [markdown]
# **Reading.**
#
# - **The calculation by hand matches the library**: 5.4448 in both cases, difference 5·10⁻⁷
#   (32-bit number rounding). The two terms also match: *align* = −0.96, that is the mean
#   cosine between reference and positive is 0.96 (they are almost at the same point of the sphere), and
#   *uniform* = 6.40.
# - **The original formula**, with the positive in the sum, gives 5.4492: with 512 negatives adding one
#   term changes very little (+0.004). The difference between the two versions is therefore a detail.
# - A single batch of 512 is enough to represent the training: 5.4448 against 5.4437 for the mean of the
#   last 500 steps.
# - **Goodness of fit: 1.14 bits, out of a maximum of 2.89.** Why so far from the maximum, if the
#   positive almost coincides with the reference? Because the maximum would require all the negatives
#   to be **on the opposite side** of the sphere. But the negatives are randomly chosen time points, and the map covers
#   the whole sphere: some of the negatives always fall near the reference. The maximum is
#   unreachable by construction, and the goodness of fit must be read as a comparison between maps trained
#   in the same way, not as an absolute grade.
#
# ### 2.6 · The temperature
#
# The temperature τ decides **how strict the judge** of the game is. With the similarities of a batch
# we can compute the probability with which the judge "chooses" the positive among the 513 candidates
# (positive + 512 negatives): exp(s⁺/τ) / [exp(s⁺/τ) + Σ exp(sᵢ/τ)].
#
# - **small τ**: small differences in similarity become huge differences in probability. The
#   judge looks almost only at the negatives **most similar** to the reference (the best
#   impostors);
# - **large τ**: all the similarities flatten out, and all the candidates weigh almost the same.
#
# On the same map (trained with τ = 1) we change only τ in the calculation, and look at two things: the
# probability of choosing the positive and how much of the negatives' "weight" goes to the 10 most
# similar out of 512. We are **not** training with different τ: this shows how τ judges, not what
# map it would produce. We measure the effect on training on the real data in §7.

# %%
TAUS = np.array([0.05, 0.1, 0.2, 0.5, 1, 2, 5])            # temperatures tried
rows_t = []
for tau in TAUS:
    sp, sn = s_pos * TAU / tau, s_neg * TAU / tau           # the same similarities, divided by the new τ
    p_pos = np.exp(sp - logsumexp(np.c_[sp, sn]))           # probability that the judge chooses the positive
    w_neg = np.exp(sn - logsumexp(sn)[:, None])             # weight of each negative (for each reference the weights sum to 1)
    top10 = np.sort(w_neg, axis=1)[:, -10:].sum(1)          # weight of the 10 most similar negatives
    rows_t.append({"τ": tau, "probability of choosing the positive": p_pos.mean(),
                   "weight of the 10 most similar negatives (out of 512)": top10.mean(),
                   "maximum possible goodness of fit (bits)": 2 / tau * np.log2(np.e)})
tab_tau = pd.DataFrame(rows_t).set_index("τ")
display(tab_tau.round(3))
print(f"At chance: probability 1/513 = {1 / 513:.4f}; weight of the 10 negatives if all equal: 10/512 = {10 / 512:.3f}")

# %% [markdown]
# **Reading.**
#
# - **With τ = 1 the judge is unsure**: it chooses the positive with probability 0.4%, barely double
#   the chance level (0.19%). With the cosine distance the similarities range from −1 to 1, so with τ = 1 the
#   exponentials only range from e⁻¹ to e¹: no candidate can weigh much more than the others. This is the
#   same reason why the goodness of fit cannot exceed 2.9 bits.
# - **Lowering τ the judge becomes strict**: with τ = 0.05 the probability rises to 3.5%, and the 10
#   most similar negatives (out of 512) take 46% of the weight, against 4% with τ = 1 and 2% if
#   they all weighed the same.
# - **What this means for training**: with small τ the network works mainly to separate the
#   reference from the closest impostors (local detail); with large τ it treats all the
#   negatives in the same way (the overall arrangement). Which is better for our data cannot be
#   said a priori: it is among the tests of §7.
# - **The scale of the goodness of fit depends on τ** (maximum 2/τ · log₂ e): maps trained with different
#   τ cannot be compared with this number.
#
# ### 2.7 · The seed: a library problem and the fix
#
# CEBRA depends on chance: the network's initial weights and the references, positives and negatives drawn at
# every step. To be able to repeat a result, the **seed** of the random numbers is fixed. But in CEBRA
# 0.6.0 the class that draws the time points (`HasGenerator`) creates a generator **with a seed chosen at
# random by the system** and ignores the one we give it. It is like asking a friend to shuffle a deck
# "always the same way" and finding that each time they pick up a new deck.
#
# The fix (in `cebra_fit`, §2.3) replaces the generator with one that starts from a known seed.
# Each generator receives a **different** seed (seed × 1000 + a counter): if two generators
# started from the same seed they would draw the same numbers, and for example references and positives
# would not be independent.
#
# The check: two trainings with the same seed without the fix, and two with the fix
# (500 steps each, standard version).

# %%
res_seed = run_cebra([dict(X_train=X_syn[:N_TR], X_apply=[X_syn[:N_TR]], cfg=dict(CEBRA_STD, max_iterations=500),
                           seed=0, fix_seed=fix) for fix in (False, False, True, True)])  # same seed (0) for all four
diff_off = np.abs(res_seed[0]["maps"][0] - res_seed[1]["maps"][0]).max()  # maximum difference between the two maps without the fix
diff_on = np.abs(res_seed[2]["maps"][0] - res_seed[3]["maps"][0]).max()   # ...and with the fix
print(f"Same seed, without the fix: maximum difference between the two maps {diff_off:.3f} "
      f"(final cost function {res_seed[0]['loss'][-1]:.4f} and {res_seed[1]['loss'][-1]:.4f})")
print(f"Same seed, with the fix:  maximum difference between the two maps {diff_on:.3f} "
      f"(final cost function {res_seed[2]['loss'][-1]:.4f} and {res_seed[3]['loss'][-1]:.4f})")
print("With the fix the two maps are identical bit for bit:", np.array_equal(res_seed[2]["maps"][0], res_seed[3]["maps"][0]),
      "| and also the whole cost function:", np.array_equal(res_seed[2]["loss"], res_seed[3]["loss"]))

# %% [markdown]
# **Reading.**
#
# - **Without the fix, the same seed gives different maps**: a point moves by a few tenths,
#   on a sphere of radius 1, and the final cost function is also different. The exact numbers change
#   at every run of the notebook (in the runs made so far the maximum difference ranged from
#   0.13 to 0.74), and this is exactly the problem.
# - **With the fix the two trainings are identical**, bit for bit: same map and same cost function
#   at every step.
# - **Also verified on the whole notebook**: running it twice in a row, all the CEBRA
#   results obtained with the fix (R² of §2.4, cost function, InfoNCE of §2.5) are
#   identical; only the example without the fix above changes.
# - From here on **all the trainings use the fix**. A result can therefore be redone identically
#   on this computer with these versions (cebra 0.6.0, torch 2.7.1, the Mac's GPU). On another
#   machine or with other versions it is not guaranteed: for this reason, from §3 on, the trained models are saved
#   on disk.
# - The fix does **not** remove the dependence on chance: with different seeds the maps remain different.
#   We measure how much this matters with 5 seeds.

# %% [markdown]
# ## 3 · CEBRA-Time on the real data: a first training run
#
# One main training run, plus a few controls, to answer two questions before measuring
# anything:
#
# 1. **How many iterations are needed?** We decide this with **validation** trials, never with the
#    test ones (§3.2).
# 2. **What does the map look like?** (§3.3)
#
# ### The parameters, and where they come from
#
# We checked them in the paper (Schneider, Lee & Mathis 2023, full text on PubMed
# Central, PMC10172131):
#
# | parameter | value here | what the paper says |
# |---|---|---|
# | network | `offset10-model`: window of 10 bins (200 ms), **centered** (100 ms before and 100 after) | Methods, *CEBRA model parameters used*: `offset10-model` for the neural data of rat hippocampus, primate S1 and Allen (2-photon) |
# | Δ (`time_offsets`) | 10 bins (200 ms) | same section: `time_offsets=10` for the same data (there with 25 ms bins in the hippocampus, hence 250 ms) |
# | batch (`batch_size`) | 512 | same section: `batch_size=512` |
# | temperature τ | 1 | Extended Data Fig. 3 (hippocampus): τ = 1 for CEBRA-Behaviour, **τ = 2.25 for CEBRA-Time**. We keep 1, the library default, so as to use the same τ with CEBRA-Behavior in §6 as well; 2.25 is among the tests of §7 |
# | learning rate | 3·10⁻⁴ | library default; in the parts of the paper we read it is not reported for these data |
# | iterations | **to be decided here** | Extended Data Fig. 3 (hippocampus): 10 000 for CEBRA-Behaviour, 4000 for CEBRA-Time. In the Methods the number of steps is the only parameter **not** chosen with validation |
# | dimensions | 3 | Extended Data Fig. 3: *"latent dimension three (the minimum)"* |
# | distance | cosine | Methods: dot product between normalized vectors |
# | optimizer | Adam | library default; the paper does not name it in the parts we read |
#
# **The centered window uses 100 ms of future.** This is fine for analyzing data; a real-time
# brain-computer interface could not do it.
#
# ### 3.1 · Saving the models to disk
#
# From §4 on, many trainings are needed (5 blocks × 5 seeds for each configuration). Every trained
# model is saved to disk, and the next time it is read back instead of retraining it, like a dish
# already cooked and put in the freezer:
#
# - the **file name** is a fingerprint (*hash*) of everything that determines the result: the code of
#   `cebra_fit`, the parameters, the seed, and the data (training, sequences to project, validation
#   trials). If even a single number of the data changes, the name changes, and the model is retrained;
# - both the model (`.pt`, so that it can be reused) and the results (`.npz`: maps, cost
#   function, validation measures, duration) are saved;
# - the folder is **outside the Desktop** (`~/thesis_h1_work/cache_04`): the Desktop is synchronized with
#   iCloud, which can remove files from disk;
# - `RETRAIN = True` redoes all the trainings from scratch. With the seed fix (§2.7) they must
#   give the same results, at least on this computer and with these versions.
# %%
import hashlib, inspect, json                               # fingerprints of the data, source text of the code, saving the measures

CACHE = Path.home() / "thesis_h1_work" / "cache_04"           # where the models are saved: outside the Desktop (iCloud)
CACHE.mkdir(parents=True, exist_ok=True)
RETRAIN = False                                             # True = redoes all the trainings instead of reading them back

def job_key(job, fn=None):                                  # fingerprint of a training: changes if anything that determines it changes
    h = hashlib.sha1()
    for f in (cebra_fit, ValMonitor.__init__, ValMonitor.__call__, ValMonitor.infonce):
        h.update(inspect.getsource(f).encode())             # the code of the training and of the measures
    if fn is not None and fn is not cebra_fit:
        h.update(inspect.getsource(fn).encode())            # from §6: also the code of the function with labels
    h.update(json.dumps({k: job.get(k) for k in ("cfg", "seed", "fix_seed", "every")},
                        sort_keys=True, default=str).encode())  # parameters, seed, frequency of the measures
    arrays = [job["X_train"], *job["X_apply"]]              # the data...
    if job.get("monitor") is not None:
        h.update(job["monitor"].key.encode())               # ...the settings of the validation measures...
        arrays += [s_ for seqs in job["monitor"].sets.values() for s_ in seqs]  # ...and their trials
    if job.get("y_train") is not None:
        arrays.append(job["y_train"])                       # from §6: the labels...
    for name, (Xv, yv) in sorted((job.get("val") or {}).items()):
        h.update(name.encode()); arrays += [Xv, yv]         # ...and the validation trials with their labels
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())         # every number of the data enters the fingerprint
    return h.hexdigest()[:16]

def run_cebra_cached(jobs, fn=None):                        # like run_cebra (§2.3), but reads back from disk the trainings already done
    keys = [job_key(j, fn) for j in jobs]
    todo = [i for i, k in enumerate(keys) if RETRAIN or not (CACHE / f"{k}.npz").exists()]  # the ones that really have to be run
    fresh = run_cebra([dict(jobs[i], model_path=CACHE / f"{keys[i]}.pt") for i in todo], fn)  # two at a time, in the separate process
    for i, r in zip(todo, fresh):                           # the new results are saved right away
        np.savez(CACHE / f"{keys[i]}.npz", loss=r["loss"], seconds=r["seconds"], device=r["device"],
                 monitor=json.dumps(r.get("monitor")), extra=json.dumps(r.get("extra")),
                 **{f"map_{j}": m for j, m in enumerate(r["maps"])})
    out = []
    for k in keys:                                          # all the results, in the order of the jobs, read from disk
        z = np.load(CACHE / f"{k}.npz")
        out.append({"maps": [z[f"map_{j}"] for j in range(sum(n.startswith("map_") for n in z.files))],
                    "loss": z["loss"], "seconds": float(z["seconds"]), "device": str(z["device"]),
                    "monitor": json.loads(str(z["monitor"])), "key": k,
                    "extra": json.loads(str(z["extra"])) if "extra" in z.files else None})  # extra measures (from §6)
    print(f"Trainings: {len(todo)} run now, {len(jobs) - len(todo)} read back from disk")
    return out

# %% [markdown]
# ### 3.2 · How many iterations? Validation
#
# **The scheme.** We use block 0 (test trials 3, 8 and 14, which are **not** touched here). Of its
# 12 training trials we set aside 2 for **validation**, 12 and 15: one even and one
# odd, for the two versions of the task (03, §9.2). CEBRA trains on the other 10.
#
# **The initial PCA must not see the validation trials either.** The 25 PCs of block 0 (§1.3)
# come from a PCA estimated on the movement bins of all 12 training trials of the block, hence
# also the 2 validation ones. PCA uses no labels, but it "sees" the data: this is the shortcut of 03
# (§7). For validation we therefore estimate a PCA on only the 10 trials CEBRA trains on. How much
# does it matter? We measure how much variance of the validation trials the two PCAs retain.

# %%
K0 = 0                                                      # the block used in this section
VAL_TRIALS, VAL_TRIALS_B, CHECK_TRIALS = (12, 15), (6, 7), (4, 5)  # two pairs of validation trials and one pair of training trials on which we measure (one even and one odd)
TRIALS = np.unique(trial)                                   # the 15 trials
fit_trials = [t for t in TRIALS if t not in TEST_TRIALS[K0] and t not in VAL_TRIALS]      # the 10 trials CEBRA trains on (validation 12, 15)
fit_trials_b = [t for t in TRIALS if t not in TEST_TRIALS[K0] and t not in VAL_TRIALS_B]  # the 10 trials CEBRA trains on (validation 6, 7)

def pca_seqs(fit_tr):                                       # 25-component PCA estimated on the movement bins of the fit_tr trials only
    p_ = PCA(K_PCA).fit(Z[bins_all[np.isin(trial[bins_all], fit_tr)]])
    X_ = p_.transform(Z)                                    # the whole session on its 25 directions
    return p_, X_, {t: X_[trial == t].astype(np.float32) for t in TRIALS}  # and each trial separately, in the format of the networks

def var_kept(p_, Xv):                                       # fraction (%) of the variance of Xv that lies in the 25 directions of the PCA p_
    Xc = Xv - p_.mean_
    return ((Xc @ p_.components_.T) ** 2).sum() / (Xc ** 2).sum() * 100

PCA_A, XA, SEQ_A = pca_seqs(fit_trials)                     # the PCA for validation with trials 12 and 15
PCA_B, XB, SEQ_B = pca_seqs(fit_trials_b)                   # the PCA for validation with trials 6 and 7

def mov(trs):                                               # the movement bins of some trials
    return Z[bins_all[np.isin(trial[bins_all], trs)]]

display(pd.DataFrame({str(vt): {"PCA of §1.3 (12 trials: it has seen them)": var_kept(PCA_F[K0], mov(vt)),
                                "PCA on the 10 training trials only": var_kept(p_, mov(vt))}
                      for vt, p_ in ((VAL_TRIALS, PCA_A), (VAL_TRIALS_B, PCA_B))}).T.rename_axis("validation trials").round(1))
print(f"For comparison: on the 10 trials it is estimated on, the PCA retains {var_kept(PCA_A, mov(fit_trials)):.1f}% of the variance; "
      f"on the test trials of block 0, the PCA of §1.3 retains {var_kept(PCA_F[K0], mov(TEST_TRIALS[K0])):.1f}%")

# %% [markdown]
# **Reading.** It is a large difference, and we found it by chance, while checking a first
# version of this section.
#
# - The 25-component PCA retains **40-42%** of the variance of the validation trials if it has seen them, only
#   **29%** if it has not. On the trials it is estimated on it retains 40.7%; on the test trials of
#   block 0, which it never sees, 27.7%. The "clean" validation trials therefore behave like
#   the test ones, which is what we want.
# - About **one third** (28-30%) of what the 25 directions capture in the training trials is not
#   found again in the new trials: part of the 25 directions describes fluctuations specific to the trials
#   the PCA is estimated on. A possible explanation: the smoothed data change slowly, so
#   the roughly 3700 movement bins are worth many fewer independent points, and 25 directions are enough to
#   fit even the noise of those trials. We did not verify this.
# - For validation we therefore use the **PCA estimated on the 10 trials only** that CEBRA trains on.
#   In the common yardstick (§1.2), as in 03, the PCA never sees the test trials: those numbers are
#   clean.
#
# **What we measure during training.** Every 500 steps we redo the measure of §2.5,
# InfoNCE with the same formula as the library, on batches of time points fixed in advance (20 batches
# of 512 references, with their positives 10 bins later in the same trial and 512 negatives), on two
# sets of trials projected one at a time:
#
# - **2 validation trials** (12 and 15): CEBRA has never seen them, and neither has the PCA;
# - **2 training trials** (4 and 5): the same measure, but on trials that CEBRA has seen.
#
# The two sets have the same shape (2 trials, one even and one odd), so the two numbers can be
# compared. If the training curve keeps falling while the validation one rises again, the
# network is memorizing the trials it sees (*overfitting*).
#
# **The rule, fixed before looking**: the number of steps is the smallest one, rounded up to the
# thousands, at which the validation cost function has made **98% of its
# descent** (from the initial value to the minimum). We train up to 10 000 steps, the maximum used
# in the paper. About 2 minutes.

# %%
class ValMonitor:
    """During training, measures InfoNCE (as in §2.5) on batches of time points fixed in advance, on
    some sets of trials projected one at a time. It runs in the separate process."""

    def __init__(self, sets, delta=10, tau=1.0, n_batches=20, batch=512, seed=0):
        rng_m = np.random.default_rng(seed)                 # fixed seed: the same batches at every measure
        self.sets, self.delta, self.tau, self.history = sets, delta, tau, []
        self.idx = {}                                       # for each set: 20 pairs (references, negatives)
        for name, seqs in sets.items():
            lens = [len(s_) for s_ in seqs]                 # length of each trial
            starts = np.cumsum([0] + lens[:-1])             # where each trial begins, once they are lined up
            refs = np.concatenate([a + np.arange(n - delta) for a, n in zip(starts, lens)])  # references with the positive in the same trial
            self.idx[name] = [(rng_m.choice(refs, batch), rng_m.integers(0, sum(lens), batch)) for _ in range(n_batches)]
        self.key = f"ValMonitor({ {k: [len(s_) for s_ in v] for k, v in sets.items()} }, {delta}, {tau}, {n_batches}, {batch}, {seed})"

    def __call__(self, num_steps, solver):                  # called by CEBRA every `every` steps
        import torch                                        # (we are in the separate process)
        was_training = solver.model.training                # projecting puts the network in "use" mode: afterwards we put it back as it was
        row = {"step": int(num_steps)}
        for name, seqs in self.sets.items():
            E = np.concatenate([solver.transform(torch.from_numpy(s_)).cpu().numpy() for s_ in seqs])  # each trial projected on its own
            row[name] = float(np.mean([self.infonce(E[r], E[r + self.delta], E[n]) for r, n in self.idx[name]]))
        solver.model.train(was_training)
        self.history.append(row)

    def infonce(self, ref, pos, neg):                       # the formula of §2.5 (cosine: the points are already on the sphere)
        s_pos = (ref * pos).sum(1) / self.tau
        s_neg = ref @ neg.T / self.tau
        m = s_neg.max(1, keepdims=True)                     # the maximum is subtracted so that the exponentials do not "explode"
        return float(np.mean(-s_pos + m[:, 0] + np.log(np.exp(s_neg - m).sum(1))))

def rule98(v):                                              # the rule: first step with 98% of the descent, rounded to the thousands
    reached = v.values <= v.min() + 0.02 * (v.iloc[0] - v.min())  # where the curve has already made 98% of the descent to the minimum
    return int(np.ceil(v.index[np.argmax(reached)] / 1000) * 1000)

STEPS_SHOW = [0, 500, 1000, 1500, 2000, 3000, 4000, 6000, 8000, 10_000]  # the rows shown in the tables

monitor0 = ValMonitor({"2 training trials (4, 5)": [SEQ_A[t] for t in CHECK_TRIALS],
                       "2 validation trials (12, 15)": [SEQ_A[t] for t in VAL_TRIALS]})
res3 = run_cebra_cached([dict(X_train=np.concatenate([SEQ_A[t] for t in fit_trials]),  # the 10 trials in a row: the training film
                              X_apply=[SEQ_A[t] for t in TRIALS],                       # then all 15 trials, projected one at a time
                              cfg=dict(CEBRA_STD, max_iterations=10_000), seed=0,
                              monitor=monitor0, every=500)])[0]
hist3 = pd.DataFrame(res3["monitor"]).set_index("step")    # the measures during training: one row every 500 steps
print(f"Training trials: {[int(t) for t in fit_trials]} ({sum(len(SEQ_A[t]) for t in fit_trials)} bins); "
      f"validation: {VAL_TRIALS}; test (not used): {TEST_TRIALS[K0].tolist()}")
print(f"Duration: {res3['seconds']:.0f} s for {len(res3['loss'])} steps ({res3['seconds'] / len(res3['loss']) * 1000:.1f} ms per step, "
      f"measures included), device {res3['device']}")
v3 = hist3["2 validation trials (12, 15)"]               # the validation curve
N_ITER_A = rule98(v3)
display(hist3.loc[STEPS_SHOW].assign(
    **{"goodness of fit, validation (bits)": lambda d: bits(d["2 validation trials (12, 15)"])}).round(3))
print(f"Validation minimum: {v3.min():.4f} at step {v3.idxmin()}; 98% rule -> {N_ITER_A} steps")

# %% [markdown]
# **A second check, added after seeing the first result.** In a first version of
# this section (with the PCA of §1.3, which had seen the validation trials) the validation curve
# reached its minimum very early and then rose again, while that of the training trials
# stayed low: the sign of overfitting. But trials 12 and 15 are at the end of the session, and if the
# signals change slowly over the course of the session (for example because the electrodes shift
# slightly) this too could make the validation trials look "different". We therefore repeat the
# same measure with two other validation trials, **6 and 7** (mid-session, one even and one
# odd), with their clean PCA and with another seed. We apply the same rule and, between the two
# values, **keep the larger one** (more iterations: the cautious choice). About 2 minutes.

# %%
monitor_b = ValMonitor({"2 training trials (4, 5)": [SEQ_B[t] for t in CHECK_TRIALS],
                        "2 validation trials (6, 7)": [SEQ_B[t] for t in VAL_TRIALS_B]})
res3b = run_cebra_cached([dict(X_train=np.concatenate([SEQ_B[t] for t in fit_trials_b]), X_apply=[SEQ_B[t] for t in TRIALS],
                               cfg=dict(CEBRA_STD, max_iterations=10_000), seed=1, monitor=monitor_b, every=500)])[0]
hist3b = pd.DataFrame(res3b["monitor"]).set_index("step")
v3b = hist3b["2 validation trials (6, 7)"]
N_ITER_B = rule98(v3b)
N_ITER = max(N_ITER_A, N_ITER_B)                            # the number of steps used from here on
display(hist3b.loc[STEPS_SHOW].assign(
    **{"goodness of fit, validation (bits)": lambda d: bits(d["2 validation trials (6, 7)"])}).round(3))
print(f"Validation minimum: {v3b.min():.4f} at step {v3b.idxmin()}; 98% rule -> {N_ITER_B} steps")
print(f"Number of steps chosen (the larger of the two): N_ITER = {N_ITER}")

# %%
fig, ax = plt.subplots(1, 2, figsize=(14, 4.3))
loss3 = res3["loss"]                                        # the cost function on each training batch, step by step
ax[0].plot(np.arange(1, len(loss3) + 1), loss3, color="0.8", lw=0.4, label="at every step")
ax[0].plot(np.arange(1, len(loss3) + 1), uniform_filter1d(loss3, 200), color="C0", lw=2, label="mean over 200 steps")
ax[0].axhline(np.log(512), color="0.4", ls="--", lw=1, label="chance level: log 512")
ax[0].set(xlabel="step", ylabel="InfoNCE", title="During training (batches drawn from the 10 trials, seed 0)")
ax[0].legend(fontsize=8)
for h_, run, ls in ((hist3, "seed 0", "-"), (hist3b, "seed 1", "--")):  # the curves measured every 500 steps, in the two trainings
    for col in h_.columns:
        c = "C0" if "training" in col else ("C3" if "12" in col else "C1")
        ax[1].plot(h_.index[1:], h_[col].iloc[1:], ls, marker="o", color=c, ms=3, lw=1.5, label=f"{col}, {run}")
ax[1].axvline(N_ITER, color="k", lw=1.2, label=f"chosen: {N_ITER} steps (98% rule)")
ax[1].axvline(4000, color="0.5", ls=":", lw=1.2, label="paper, CEBRA-Time: 4000")
ax[1].axvline(10_000, color="0.5", ls="--", lw=1.2, label="paper, CEBRA-Behaviour: 10 000")
ax[1].set(xlabel="step", ylabel="InfoNCE (same batches at every measure)",
          title="Seen trials versus never-seen trials (one measure every 500 steps)")
ax[1].legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
fig.tight_layout(); fig.savefig(FIGS / "04_convergence.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# **Reading.**
#
# - **During training** (left) the cost function on the training batches falls for
#   all 10 000 steps, ever more slowly (from 5.83 in the first 200 steps to 5.42 in the last
#   200).
# - **On never-seen trials it does not.** With both validation pairs the minimum is reached at **1000
#   steps** (5.802 for trials 12-15, 5.769 for trials 6-7); then the curve slowly rises up to
#   10 000 (5.861 and 5.865). On training trials 4-5, over the same time, it stays low or keeps
#   falling. This is **overfitting**: after about 1000 steps the network learns things about the trials it sees that
#   do not hold for new ones. It happens in both pairs, at the end of the session (12-15) and mid-session
#   (6-7), so it does not depend on a slow change over the course of the session.
# - **Even at the best point the gap is large**: at 1000 steps the goodness of fit is about 1.0
#   bit on the training trials and 0.63-0.68 bit on the validation ones. An important part of
#   what CEBRA learns is specific to the trials it sees, as for the PCA above.
# - **The rule gives 1000 steps for both validations: N_ITER = 1000.** This is much less than the 4000
#   (CEBRA-Time) and 10 000 (CEBRA-Behaviour) of the paper, which however concern other data. A practical
#   consequence: a training run lasts about 11 seconds.
# - In the first version of this section, with the PCA that had seen the validation trials, the
#   curves had the same shape (minimum at 500-1000 steps, then a rise) and the rule gave the same
#   number of steps; the rise was steeper.
# - **Limitations**: a single block, two pairs of validation trials, one seed per pair; the measure is
#   on fixed batches of time points, so it has some noise (the oscillations from one point to the next).
#
# ### Phase decoding on the validation trials
#
# The rule chooses with CEBRA's cost function, which knows nothing about the phases. But what we will
# measure in §4 is phase decoding. We therefore check that the two agree,
# **always and only on the validation trials** (12 and 15): we train CEBRA on the usual 10
# trials with 500, 1000, 2000 and 5000 steps (plus the 10 000-step run just done), and for each we measure
# phase decoding with the common yardstick (10-neighbor classifier trained on the points
# of the 10 training trials, judged on the points of the 2 validation trials). There are only 2 trials,
# so the numbers are noisy: we look at the trend, not the single value. It is not used to
# choose, but to check the choice. For reference, with the same scheme and the same clean PCA:
# PCA with d = 2 (the free coordinates of 3-dimensional CEBRA) and with d = 3, and the
# classifier on the 25 dimensions. About 1-2 minutes.

# %%
ITER_LIST = sorted({500, 1000, 2000, 5000, N_ITER} - {10_000})  # the numbers of steps tried (10 000 is already done)
res_it = dict(zip(ITER_LIST, run_cebra_cached([dict(X_train=np.concatenate([SEQ_A[t] for t in fit_trials]),
                                                     X_apply=[SEQ_A[t] for t in TRIALS],
                                                     cfg=dict(CEBRA_STD, max_iterations=n_it), seed=0) for n_it in ITER_LIST])))
res_it[10_000] = res3                                       # the 10 000-step training run above (same seed, same trials)
ev_fit = bins[np.isin(trial[bins], fit_trials)]             # evaluation points of the 10 training trials
ev_val = bins[np.isin(trial[bins], VAL_TRIALS)]             # evaluation points of the 2 validation trials

def val_acc(E):                                             # common yardstick, but training = 10 trials, judging = 2 validation trials
    clf = KNeighborsClassifier(n_neighbors=K_NN).fit(E[ev_fit], phase[ev_fit])
    return balanced_accuracy_score(phase[ev_val], clf.predict(E[ev_val])) * 100

acc_it = {n_it: val_acc(np.concatenate(r["maps"])) for n_it, r in sorted(res_it.items())}
acc_ref = {f"PCA, d = {d}": val_acc(PCA(d).fit(XA[ev_fit]).transform(XA)) for d in (2, 3)}
acc_ref["no map, 25 dimensions"] = val_acc(XA)
print(f"Points: {len(ev_fit)} training, {len(ev_val)} validation")
display(pd.Series({f"CEBRA-Time, {n_it} steps": a for n_it, a in acc_it.items()} | acc_ref,
                  name="phase decoded on trials 12 and 15 (%)").round(1).to_frame())

# %% [markdown]
# **Reading.**
#
# - **Phase decoding agrees with the rule**: on the validation trials it is highest
#   with 500-1000 steps (44.0% and 41.4%) and drops to 35-37% from 2000 steps on. Training more does
#   worse even for what we will measure.
# - With the same scheme PCA reaches 27.4% with d = 2 and 31.0% with d = 3; the classifier
#   on the 25 dimensions 47.0%. With 1000 steps CEBRA (2 free coordinates) is far above PCA
#   and a little below the classifier on the 25 dimensions.
# - **Caution**: there are only 2 trials (153 points), and differences of a few points are noise (for
#   example 500 versus 1000 steps). This table was not used to choose: had we used it
#   we would perhaps have taken 500 steps. We keep the 1000 of the rule. The real measure, on 5 blocks, 5 seeds
#   and the test trials, is in §4.
#
# ### 3.3 · What the map looks like
#
# The map is a sphere. To see all of it at once we open it up as a **world map** (Mollweide
# projection, the one of the globes hanging in classrooms): longitude and latitude of each point.
# The orientation of the sphere is arbitrary, so the edges of the world map cut it at an
# arbitrary point: two points at opposite edges can be close on the sphere.
#
# Two models, both trained on the 10 trials: the one with the chosen number of steps (top) and the one
# with 10 000 steps (bottom). For each, three panels:
#
# - the movement points of the **training trials** (one every 100 ms, as in the common yardstick);
# - the movement points of the **never-seen trials**: validation (12, 15) and test of block 0 (3, 8, 14).
#   We look at them only after choosing the number of steps;
# - **all the bins** of all the trials, with the pauses (`Presentation`, `Intertrial`) in gray: more than
#   half of the training film is pauses (§1.4).

# %%
seen = np.isin(trial, fit_trials)                           # bins of the trials CEBRA trained on
every2 = np.arange(0, n_bins, 2)                            # one bin every 2, so as not to cover everything
pause = every2[~MOV[every2]]

def to_lonlat(E):                                           # longitude and latitude of each point of the sphere
    return np.arctan2(E[:, 1], E[:, 0]), np.arcsin(np.clip(E[:, 2], -1, 1))

fig = plt.figure(figsize=(17, 8.6))
for row, n_it in enumerate((N_ITER, 10_000)):
    lon, lat = to_lonlat(np.concatenate(res_it[n_it]["maps"]))  # the map of all 12233 bins (the trials are in time order)
    for k, (ttl, b_) in enumerate([("training trials (10): movement points", bins[seen[bins]]),
                                   ("never-seen trials (12, 15 and 3, 8, 14): movement points", bins[~seen[bins]])]):
        a_ = fig.add_subplot(2, 3, row * 3 + k + 1, projection="mollweide")
        for p in io.MOVEMENT_PHASES:                        # one color per phase
            m = phase[b_] == p
            a_.scatter(lon[b_][m], lat[b_][m], s=5, alpha=0.7, color=io.PHASE_COLORS[p])
        a_.set_title(f"{n_it} steps: {ttl}", fontsize=10); a_.set_xticklabels([]); a_.set_yticklabels([])
    a_ = fig.add_subplot(2, 3, row * 3 + 3, projection="mollweide")
    a_.scatter(lon[pause], lat[pause], s=2, color="0.75", alpha=0.5)  # pauses in gray
    for p in io.MOVEMENT_PHASES:
        m = every2[phase[every2] == p]
        a_.scatter(lon[m], lat[m], s=2, alpha=0.6, color=io.PHASE_COLORS[p])
    a_.set_title(f"{n_it} steps: all bins, all trials (pauses in gray)", fontsize=10)
    a_.set_xticklabels([]); a_.set_yticklabels([])
handles = [plt.Line2D([], [], color=io.PHASE_COLORS[p], marker="o", ls="", ms=5, label=p) for p in io.MOVEMENT_PHASES]
handles.append(plt.Line2D([], [], color="0.75", marker="o", ls="", ms=5, label="pause"))
fig.legend(handles=handles, ncol=9, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.04, 1, 1)); fig.savefig(FIGS / "04_cebra_time_map.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# **Reading.**
#
# - **With 1000 steps** (top), in the training trials the phases occupy different zones of the sphere: on
#   one side the hand phases (`Release` in dark gray, `Shape` in red, `Grasp` in purple),
#   on the other those of the arm (`Reach`, `Orient`, `SnapTo`, `Carry`, `Orient2`), more mixed
#   with each other. It is the same division seen in 03 (§4 and §9.2). In the never-seen trials the same zones
#   can be recognized, more scattered: `Release` and `Shape` remain grouped.
# - **With 10 000 steps** (bottom) the clusters of the training trials are tighter and more detailed, but the
#   never-seen trials scatter. This is the overfitting of the curves, made visible: had we looked
#   only at the training trials (or estimated the map on all trials, the shortcut of 03), the model
#   trained for longer would have seemed better.
# - **The pauses** (light gray, right) occupy a large part of the sphere: CEBRA uses part of its
#   "space" to describe moments that we do not evaluate. It is the consequence of choice B (§1.4);
#   the alternative, movement bins only, is among the tests of §7.
# - **What the figure does not say**: the orientation of the sphere is arbitrary and the world map distorts
#   the zones near the edges; it is a description, not a measure. The measures are in §4.

# %% [markdown]
# ## 4 · CEBRA-Time with the common yardstick
#
# Now the real measure, the same for all methods (§1.2): phase decoding with the
# 10-neighbor classifier, balanced accuracy, the same 1106 points and the same 5 blocks
# of 03. In each block **CEBRA trains only on the 12 training trials** (the film of §1.4, with its
# seams) and the test trials are projected one at a time. Parameters of §3, with **N_ITER = 1000**
# steps. **5 seeds** for each measure: mean and min-max.
#
# Three issues to settle before comparing the numbers.
#
# **1. The sphere: comparison at equal free coordinates.** With the cosine distance CEBRA's points
# lie on a sphere: 3 dimensions are 2 free coordinates, like latitude and longitude. §2.4
# showed that this is not a detail: on the synthetic sheet the 3-dimensional sphere lost one of the
# two directions of the sheet. The rule: **3-dimensional CEBRA is compared with PCA, Isomap and UMAP at
# d = 2; 4-dimensional CEBRA (3 free coordinates) with d = 3.** We train both versions.
#
# **2. The classifier on the sphere.** The CEBRA paper decodes with a neighbor classifier
# that uses the cosine distance (Methods, *Decoding analysis*); our common yardstick uses the straight-line
# distance. For points on the sphere it is the same thing: the squared straight-line distance
# equals 2 − 2 · cosine, so the order of the neighbors is identical. We verify this.
#
# **3. An equal-terms ceiling.** The ceiling of 03 (the classifier on the 25 dimensions of a single bin)
# holds for methods that see one time point at a time. CEBRA sees a **200 ms window** (5 bins
# before and 5 after, from bin b − 5 to bin b + 4): it can legitimately exceed that ceiling, just because
# it looks at more. We therefore build two ceilings with the same window, with no learning at all
# (the idea of the moving average of §2.4):
#
# - **stacked window**: for each point, the 10 bins of the window side by side, 10 × 25 =
#   250 numbers. It contains everything CEBRA sees;
# - **window mean**: the mean of the 10 bins, 25 numbers.
#
# As for CEBRA, the window does not leave the trial: at the edges the first or last bin is repeated. It must be
# remembered that the smoothing (σ = 50 ms) already gives **all** methods a window of about ±150 ms:
# CEBRA's is added to this.
#
# ### 4.1 · Phase decoding
#
# 50 trainings (2 versions × 5 seeds × 5 blocks) of 1000 steps, two at a time: about 5 minutes the
# first time, then they are read back from disk.

# %%
from itertools import product                               # all combinations of several lists

CEBRA_T = dict(CEBRA_STD, max_iterations=N_ITER)            # CEBRA-Time with the parameters of §3
OUT_DIMS = (3, 4)                                           # 3 dimensions = 2 free coordinates; 4 dimensions = 3 free coordinates
COMBOS = list(product(OUT_DIMS, SEEDS, range(5)))           # (dimensions, seed, block): 50 trainings
res4 = run_cebra_cached([dict(X_train=X25[k][TRAIN_BINS[k]].astype(np.float32),  # the training film of block k (§1.4)
                              X_apply=[X25[k][trial == t].astype(np.float32) for t in TRIALS],  # all 15 trials, one at a time
                              cfg=dict(CEBRA_T, output_dimension=od), seed=sd)
                         for od, sd, k in COMBOS])
MAPS4 = {c: np.concatenate(r["maps"]) for c, r in zip(COMBOS, res4)}  # for each combination, the map of all 12233 bins
print(f"Mean duration of a training run: {np.mean([r['seconds'] for r in res4]):.0f} s")

def decode_cebra(od, sd):                                   # common yardstick: in each block, training and test points from the map of that block
    inp = (bins, [(MAPS4[(od, sd, k)][bins[trn]], MAPS4[(od, sd, k)][bins[tst]], trn, tst)
                  for k, (trn, tst) in enumerate(FOLDS)])
    return decode_map(inp, emb_none)                        # the map is already made: the classifier works directly on its coordinates

for od in OUT_DIMS:
    ACC[f"CEBRA-Time, {od} dimensioni"] = [decode_cebra(od, sd) for sd in SEEDS]  # 5 seeds

E0 = MAPS4[(3, 0, 0)]; trn0, tst0 = FOLDS[0]                # check of issue 2, on the 3-dimensional map (seed 0, block 0)
pred_euc = KNeighborsClassifier(n_neighbors=K_NN).fit(E0[bins[trn0]], phase[bins[trn0]]).predict(E0[bins[tst0]])
pred_cos = KNeighborsClassifier(n_neighbors=K_NN, metric="cosine").fit(E0[bins[trn0]], phase[bins[trn0]]).predict(E0[bins[tst0]])
print(f"Straight-line distance versus cosine: identical predictions for {np.mean(pred_euc == pred_cos) * 100:.1f}% of the test points")

# %%
HALF = 5                                                    # the window of offset10-model: from bin b − 5 to bin b + 4
first_last = {t: np.flatnonzero(trial == t)[[0, -1]] for t in TRIALS}  # first and last bin of each trial

def window_idx(b):                                          # for each point, the 10 bins of its window, without leaving the trial
    lo = np.array([first_last[trial[x]][0] for x in b])     # first bin of the trial of each point
    hi = np.array([first_last[trial[x]][1] for x in b])     # last bin
    return np.clip(b[:, None] + np.arange(-HALF, HALF)[None, :], lo[:, None], hi[:, None])  # at the edges the edge bin is repeated

W_IDX = window_idx(bins)                                    # (1106 points, 10 bins)
INP_STACK = (bins, [(X25[k][W_IDX[trn]].reshape(len(trn), -1), X25[k][W_IDX[tst]].reshape(len(tst), -1), trn, tst)
                    for k, (trn, tst) in enumerate(FOLDS)])  # the 10 bins in a row: 250 numbers per point
INP_MEAN = (bins, [(X25[k][W_IDX[trn]].mean(1), X25[k][W_IDX[tst]].mean(1), trn, tst)
                   for k, (trn, tst) in enumerate(FOLDS)])   # the mean of the 10 bins: 25 numbers per point
ACC["tetto alla pari: finestra in fila (250 numeri)"] = decode_map(INP_STACK, emb_none)
ACC["tetto alla pari: media sulla finestra (25 numeri)"] = decode_map(INP_MEAN, emb_none)

ROWS_4 = [("PCA, d = 2", 2), ("Isomap, d = 2", 2), ("UMAP, d = 2", 2), ("CEBRA-Time, 3 dimensioni", 2),
          ("PCA, d = 3", 3), ("Isomap, d = 3", 3), ("UMAP, d = 3", 3), ("CEBRA-Time, 4 dimensioni", 3),
          ("tetto, 25 dimensioni", 25), ("tetto alla pari: media sulla finestra (25 numeri)", 25),
          ("tetto alla pari: finestra in fila (250 numeri)", 250)]
tab_4 = pd.DataFrame([{"method": name, "free coordinates": free,
                       "phase decoding (%)": mean_range(ACC[name]) if isinstance(ACC[name], list) else f"{ACC[name]:.1f}"}
                      for name, free in ROWS_4]).set_index("method")
display(tab_4)

# %%
fig, ax = plt.subplots(figsize=(10, 5.2))
names_plot = [n for n, _ in ROWS_4][::-1]                   # from top to bottom in the order of the table
for yi, name in enumerate(names_plot):
    v = np.atleast_1d(ACC[name])                            # 1 value, or 5 (one per seed)
    col = "C3" if name.startswith("CEBRA") else ("C2" if name.startswith("UMAP") else
                                                 ("0.35" if name.startswith("ceiling") else "C0" if name.startswith("Isomap") else "C1"))
    ax.scatter(v, np.full(len(v), yi), color=col, s=18, alpha=0.6, zorder=2)  # the seeds, one per point
    ax.scatter(v.mean(), yi, color=col, s=80, marker="D" if len(v) > 1 else "o", edgecolor="k", zorder=3)  # the mean
ax.axvline(12.5, color="0.5", ls="--", lw=1, label="chance (12.5%)")
ax.axhline(2.5, color="0.8", lw=0.8); ax.axhline(6.5, color="0.8", lw=0.8)  # separates the three groups
ax.set(yticks=range(len(names_plot)), yticklabels=names_plot, xlim=(10, 60), xlabel="phase decoding, balanced accuracy (%)",
       title="Phase decoding, never-seen trials (small dot = one seed, large = mean)")
ax.text(59, 8.5, "2 free\ncoordinates", ha="right", va="center", fontsize=8, color="0.4")
ax.text(59, 4.5, "3 free\ncoordinates", ha="right", va="center", fontsize=8, color="0.4")
ax.text(59, 1, "no map", ha="right", va="center", fontsize=8, color="0.4")
ax.legend(loc="lower left", fontsize=8)
fig.tight_layout(); fig.savefig(FIGS / "04_phase_decoding.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# **Lettura.**
#
# - **The classifier on the sphere** (issue 2): with the cosine distance and with the straight-line
#   distance the predictions are identical for 100% of the points, as expected.
# - **With 2 free coordinates CEBRA-Time reaches 37.7%** (35.3-40.6 across seeds): much better than
#   PCA (26.0%) and Isomap (31.8%), **worse than UMAP** (41.2%, 40.4-42.3). CEBRA's best seed
#   barely reaches UMAP's worst.
# - **With 3 free coordinates: 40.4%** (38.5-42.7), against 42.6% for UMAP (41.9-43.2), 36.0% for
#   Isomap and 32.4% for PCA. Still below UMAP on average; here the intervals overlap.
# - **The sphere costs**: going from 3 to 4 dimensions (from 2 to 3 free coordinates) gains 2.7
#   points, as in the synthetic sheet of §2.4.
# - **CEBRA depends on the seed more than UMAP**: 5 points between the best and the worst seed, against 1-2.
# - **The equal-terms ceilings.** Looking at 200 ms instead of one bin raises the ceiling from 42.2% to **44.9%** (mean
#   over the window) and to 43.9% (stacked window). The window therefore adds a little
#   information, 2-3 points, but CEBRA does not exploit it: it stays 4.5 points below its equal-terms ceiling
#   with 3 free coordinates. The stacked window, which contains more information, does slightly worse
#   than the mean: with 250 numbers per point the neighbor classifier struggles to find the right neighbors.
#
# **In summary: on this session, with the common yardstick, CEBRA-Time does not beat UMAP.** It does better than
# PCA and Isomap, but it does not retain in 2-3 coordinates all the phase information that UMAP retains,
# and it does not use the extra information in its time window. It is a valid result, with two caveats:
# CEBRA's parameters are the standard ones (only the number of steps was chosen on our
# data), and in §7 we will try others; and we do not yet know what the map contains beyond the
# time chain (§5).
#
# ### 4.2 · Single-trial trajectories
#
# The measure of 03 (§6.1 and §9.4), on the 119 windows of 1 s from the start of each phase: **roughness**
# (how much the trajectory jumps from one bin to the next: lower = better) and **consistency** (how much
# the repetitions of the same phase resemble each other: higher = better).
#
# - **Models estimated on all 15 trials**, like Isomap and UMAP in 03: the 5 block maps cannot
#   be mixed, because each has its own orientation on the sphere, and the consistency
#   would compare repetitions projected in different maps. This measure therefore **describes** the
#   maps, it does not say how well they generalize to new trials (for that there is §4.1). Input: the 25 PCs
#   estimated on all trials, as in 03. 5 seeds for each version, 1000 steps: about 1 minute.
# - **All the coordinates of the map**: 3 for 3-dimensional CEBRA, 4 for 4-dimensional CEBRA; the
#   first 3 for PCA, Isomap and UMAP (03). The two measures are means over the dimensions, so the number
#   of dimensions does not favor them.
# - **The two versions of the task.** In 03 (§9.2) we discovered that the task alternates two versions
#   between odd and even trials: the 15 repetitions of a phase are not replicates. We therefore compute the
#   consistency also **within** each version (only the repetitions of the odd trials, only
#   those of the even ones) and average them. It is the same measure that 02_new and 03 have used since 29/10
#   (16 conditions, phase × version). We apply it here also to PCA, Isomap and UMAP, recomputed with
#   the same code as 03. We do not recompute GPFA (02_new, §10.5), because it would require
#   re-estimating it: we quote it from 02_new, 0.11 mixing the versions (computed before the fix)
#   and 0.137 per version.
# %%
windows = io.phase_windows(s, win_s=1.0, align="start")     # the 119 windows of 1 s of 02 and 03
WIN_TAG = np.array([w["tag"] for w in windows])             # the phase of each window
WIN_ODD = np.array([w["trial"] % 2 == 1 for w in windows])  # true for the windows of the odd trials (one of the two versions)

def roughness(trajs):                                       # mean roughness (03, §6.1)
    return float(np.mean([(np.diff((T_ - T_.mean(0)) / (T_.std(0) + 1e-9), n=2, axis=0) ** 2).mean()
                          for T_ in trajs]))                # z-score of each dimension, then the squared second difference

def consistency(trajs, keep=None):                          # mean correlation between repetitions of the same phase (03, §6.1); keep = which windows to use
    keep = np.ones(len(windows), bool) if keep is None else keep
    out = []
    for p in io.MOVEMENT_PHASES:
        R = np.stack([trajs[i] for i in np.flatnonzero((WIN_TAG == p) & keep)])  # (repetitions, 50 bins, dimensions)
        out.append(np.mean([np.corrcoef(R[:, :, j])[np.triu_indices(len(R), 1)].mean() for j in range(R.shape[2])]))
    return float(np.mean(out))

def traj_stats(trajs):                                      # the measures of a list of trajectories
    odd, even = consistency(trajs, WIN_ODD), consistency(trajs, ~WIN_ODD)
    return {"seghettatura": roughness(trajs), "consistenza, tutte le ripetizioni": consistency(trajs),
            "consistenza, solo trial dispari": odd, "consistenza, solo trial pari": even,
            "consistenza per versione (media delle due)": (odd + even) / 2}

def cut(E):                                                 # from the map of the whole session, the 119 windows
    return [E[w["start"]:w["stop"]] for w in windows]

pca_all = PCA(n_components=K_PCA).fit(Z[bins_all])          # the 25-component PCA on all trials (03, §2)
Z25 = pca_all.transform(Z)                                  # the whole session on the 25 directions
X_all = Z25[bins]                                           # the 1106 points (03, §2)
win_X = cut(Z25)                                            # the windows in the 25-dimensional space
iso_all = Isomap(n_neighbors=K_NN, n_components=15).fit(X_all)  # Isomap as in 03 (§4: 15 coordinates, we use 3)
TRAJ = {"PCA": [traj_stats([w_[:, :3] for w_ in cut(PCA(3).fit(X_all).transform(Z25))])],
        "Isomap": [traj_stats([iso_all.transform(x_)[:, :3] for x_ in win_X])],
        "UMAP": [traj_stats(np.split(UMAP(n_neighbors=K_NN, n_components=3, random_state=sd, n_jobs=1).fit(X_all)
                                     .transform(np.concatenate(win_X)), len(win_X))) for sd in SEEDS]}
res4t = run_cebra_cached([dict(X_train=Z25.astype(np.float32), X_apply=[Z25.astype(np.float32)],  # the whole session, a single sequence
                               cfg=dict(CEBRA_T, output_dimension=od), seed=sd) for od in OUT_DIMS for sd in SEEDS])
MAPS4T = dict(zip(product(OUT_DIMS, SEEDS), [r["maps"][0] for r in res4t]))  # (dimensions, seed) -> map of the whole session
for od in OUT_DIMS:
    TRAJ[f"CEBRA-Time, {od} dimensioni"] = [traj_stats(cut(MAPS4T[(od, sd)])) for sd in SEEDS]

rows_t = {}
for name, runs in TRAJ.items():
    d_ = pd.DataFrame(runs)                                 # one row per seed (only one for PCA and Isomap)
    rows_t[name] = {c: (f"{d_[c].mean():.3f} ({d_[c].min():.3f}-{d_[c].max():.3f})" if len(d_) > 1 else f"{d_[c].iloc[0]:.3f}")
                    for c in d_.columns}
rows_t["GPFA (from 03, §6.1, and from 02_new, §10.5)"] = {"seghettatura": "0.011", "consistenza, tutte le ripetizioni": "0.11"}
tab_traj4 = pd.DataFrame(rows_t).T.fillna("not computed")
display(tab_traj4)
print("In 03, with a single UMAP seed: roughness PCA 0.013, Isomap 0.27, UMAP 0.59; consistency per version 0.109, 0.122, 0.144"
      " (mixing the versions, before the 29/10 fix: 0.088, 0.095, 0.13)")

# %%
PH_SHOW = "Release"                                         # the most reproducible phase in 02 and 03
idx_ph = np.flatnonzero(WIN_TAG == PH_SHOW)                 # its repetitions
t_win = np.arange(50) * 20                                  # ms from the start of the phase
show = {"PCA": cut(PCA(3).fit(X_all).transform(Z25)),
        "UMAP (seed 0)": np.split(UMAP(n_neighbors=K_NN, n_components=3, random_state=0, n_jobs=1).fit(X_all)
                                  .transform(np.concatenate(win_X)), len(win_X)),
        "CEBRA-Time, 3 dimensions (seed 0)": cut(MAPS4T[(3, 0)])}
fig, ax = plt.subplots(1, 3, figsize=(15, 3.8), sharey=True)
for a_, (name, tr_) in zip(ax, show.items()):
    R = np.stack([tr_[i][:, 0] for i in idx_ph])            # the first coordinate of each repetition
    R = (R - R.mean()) / R.std()                            # same scale for all methods
    for odd, c, lab in ((True, "C1", "odd trials"), (False, "C0", "even trials")):
        sel = WIN_ODD[idx_ph] == odd
        for r_ in R[sel]:
            a_.plot(t_win, r_, color=c, lw=0.7, alpha=0.45)
        a_.plot(t_win, R[sel].mean(0), color=c, lw=2.5, label=f"{lab} (mean)")  # mean of each version
    a_.set(title=f"{name}: first coordinate", xlabel="ms from the start of the phase")
ax[0].set_ylabel("z"); ax[0].legend(fontsize=8)
fig.suptitle(f"The {len(idx_ph)} repetitions of {PH_SHOW}, colored by task version", y=1.02)
fig.tight_layout(); fig.savefig(FIGS / "04_trajectories.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# **Lettura.**
# **Reading.**
#
# - **The numbers of 03 are reproduced**: roughness and per-version consistency PCA 0.013 and 0.109,
#   Isomap 0.275 and 0.122, identical to 03; UMAP with 5 seeds 0.596 (0.571-0.628) and 0.144 (0.137-0.152),
#   in line with the single seed of 03 (0.59 and 0.144). Mixing the versions: 0.088, 0.095 and 0.125,
#   the values of 03 before the fix.
# - **CEBRA's trajectories are smooth, but not as much as those of the linear methods.** Roughness
#   0.080 (3 dimensions) and 0.074 (4 dimensions): 7-8 times less than UMAP and 3-4 times less than Isomap,
#   but 6 times more than PCA (0.013) and GPFA (0.011). The reason lies in how they are built:
#   CEBRA applies the same network to overlapping windows (two neighboring bins share 9 of the 10
#   input bins), while Isomap and UMAP place each point by looking at its neighbors, and jump
#   when the neighbors change. PCA is a simple projection of the smoothed data.
# - **Consistency**: 0.089 for 3-dimensional CEBRA (like PCA, 0.088) and 0.106 for 4 dimensions
#   (like GPFA, 0.11), below UMAP (0.125). It varies a lot across seeds (for example from 0.086 to 0.127
#   with 4 dimensions).
# - **The two task versions.** For all methods the consistency within a single version is higher
#   than when mixing them: PCA from 0.088 to 0.109, UMAP from 0.125 to 0.144, 4-dimensional CEBRA from
#   0.106 to 0.122. Mixing the versions therefore lowered the consistency in 02 and 03, as
#   suspected (03, §9.2); since 29/10 both compute it per version. The ranking of the methods does not change, though. The even trials are more consistent than the
#   odd ones with all methods (for example UMAP 0.187 against 0.102): we do not know why, and we did not
#   investigate.
# - In the figure, the repetitions of `Release` along the first coordinate: smooth with PCA, jumpy with
#   UMAP, smooth but flatter with CEBRA. CEBRA's first coordinate is not "the most important" as
#   in PCA: on the sphere the orientation is arbitrary, so the figure serves to see the
#   roughness, not the shape.
# - **Limitations**: these maps are estimated on all trials, so they describe and do not measure
#   generalization; with about 7 repetitions per version the per-version consistency is noisy;
#   for GPFA we did not recompute it here (02_new, §10.5: 0.137).

# %% [markdown]
# ## 5 · Controls: what is in CEBRA-Time's map
#
# Three questions:
#
# 1. **With no task-related structure, does CEBRA still find something?** We try it on the random-phase
#    twin of 03 (§5.1).
# 2. **How much does it depend on the seed?** With the between-training consistency measure used in the CEBRA
#    paper (§5.2).
# 3. **Does CEBRA-Time learn only the time chain?** This is the most important question, because its game
#    brings together exactly time points that are close in time within the same trial (§5.3).
#
# All the controls use **3-dimensional** CEBRA (the sphere with 2 free coordinates): it is the
# standard version, and the only one that can be drawn. In §4 the two versions behaved
# similarly (37.7% and 40.4%). Parameters of §3 (Δ = 10 bins, 1000 steps), except where we change them
# on purpose.
#
# ### 5.1 · The random-phase twin
#
# The twin of 03 (§5.2): same Fourier amplitudes as the data, phases randomly shuffled (the same for
# all 25 dimensions). It has the same covariance and the same speed of variation over time,
# hence **the same time chain**, but no link to the task phases. We build it, for
# each block, on the 25 PCs of the whole session (§1.3), and on it we train CEBRA exactly as in
# §4. The phase labels are the real ones, but on the twin they have nothing to do with anything: **the decoding
# must drop to chance** (12.5%). If it did not, it would mean that something passes from training to
# test by other routes. For comparison, the same control with the classifier on the 25 dimensions
# of the twin.
#
# 5 twins (seeds 0-4) × 5 blocks, one CEBRA seed per twin: 25 trainings, about 3 minutes.

# %%
def phase_randomize(Xt, seed):                              # twin: same Fourier amplitudes, random phases equal for all columns (03, §5.2)
    rng_ = np.random.default_rng(seed)
    F = np.fft.rfft(Xt - Xt.mean(0), axis=0)                # spectrum of each column
    phases = np.exp(2j * np.pi * rng_.uniform(size=F.shape[0]))  # one random phase for each frequency
    phases[0] = 1                                           # the constant component is left untouched
    if len(Xt) % 2 == 0:
        phases[-1] = 1                                      # nor is the one at the maximum frequency
    return np.fft.irfft(F * phases[:, None], n=len(Xt), axis=0) + Xt.mean(0)

TWINS = {(k, sd): phase_randomize(X25[k], sd) for k in range(5) for sd in SEEDS}  # (block, twin seed) -> twin of the whole session
res_tw = run_cebra_cached([dict(X_train=TWINS[(k, sd)][TRAIN_BINS[k]].astype(np.float32),
                                X_apply=[TWINS[(k, sd)][trial == t].astype(np.float32) for t in TRIALS],
                                cfg=CEBRA_T, seed=sd) for sd in SEEDS for k in range(5)])
MAPS_TW = {(sd, k): np.concatenate(r["maps"]) for (sd, k), r in zip(product(SEEDS, range(5)), res_tw)}

def fold_inp(get):                                          # input of the common yardstick from a function block -> data of all bins
    return (bins, [(get(k)[bins[trn]], get(k)[bins[tst]], trn, tst) for k, (trn, tst) in enumerate(FOLDS)])

acc_tw = [decode_map(fold_inp(lambda k: MAPS_TW[(sd, k)]), emb_none) for sd in SEEDS]       # CEBRA on the twin
acc_tw_raw = [decode_map(fold_inp(lambda k: TWINS[(k, sd)]), emb_none) for sd in SEEDS]     # classifier on the 25 dimensions of the twin
display(pd.Series({"CEBRA-Time, 3 dimensions, real data (§4)": mean_range(ACC["CEBRA-Time, 3 dimensioni"]),
                   "CEBRA-Time, 3 dimensions, twins (5)": mean_range(acc_tw),
                   "no map, 25 dimensions, real data (§1.2)": f"{ACC['tetto, 25 dimensioni']:.1f}",
                   "no map, 25 dimensions, twins (5)": mean_range(acc_tw_raw),
                   "at chance": "12.5"}, name="phase decoding (%)").to_frame())

# %%
fig = plt.figure(figsize=(12, 4.4))
for j, (ttl, E) in enumerate([("Real data (block 0, seed 0)", MAPS4[(3, 0, 0)]), ("Twin 0 (block 0)", MAPS_TW[(0, 0)])]):
    lon_, lat_ = to_lonlat(E)
    a_ = fig.add_subplot(1, 2, j + 1, projection="mollweide")
    for p in io.MOVEMENT_PHASES:                            # the evaluation points of all trials, one color per phase
        m = bins[phase[bins] == p]
        a_.scatter(lon_[m], lat_[m], s=5, alpha=0.7, color=io.PHASE_COLORS[p])
    a_.set_title(ttl, fontsize=10); a_.set_xticklabels([]); a_.set_yticklabels([])
handles = [plt.Line2D([], [], color=io.PHASE_COLORS[p], marker="o", ls="", ms=5, label=p) for p in io.MOVEMENT_PHASES]
fig.legend(handles=handles, ncol=8, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.02))
fig.suptitle("CEBRA-Time, 3 dimensions: the 1106 evaluation points (in the twin the colors are only a reference)", y=1.0)
fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(FIGS / "04_twin_maps.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# **Reading.**
#
# - **On the twins CEBRA-Time drops exactly to chance**: 12.5% (10.7-13.9 across the 5 twins), like
#   the classifier on the 25 dimensions of the twin (12.8%). Nothing passes from training to test by
#   other routes: neither through the seams (§1.5), nor through the smoothing at trial boundaries, nor
#   through the time chain, which the twin preserves. The 37.7% on the real data comes from the link
#   between activity and task.
# - **The maps**: on the real data the phases occupy different zones of the sphere (those of the hand on one
#   side, those of the arm on the other, as in §3.3); on the twin the points cover the whole sphere
#   and the colors are mixed. Covariance and time chain alone are not enough to produce the
#   structure, as for UMAP in 03 (§9.2).
# - **Limitation**: the twin is Gaussian (03, §5.2), so the control rules out a linear,
#   Gaussian cloud with the same time chain, not every possible artifact.
#
# ### 5.2 · How much the seed matters: consistency between trainings
#
# In §4 the 5 CEBRA seeds gave accuracies up to 5 points apart. But do they also give **different
# maps**? The CEBRA paper measures the *consistency* between two maps of the same points as follows: we
# look for the linear transformation (rotation, stretching, shift) that brings the first map
# as close as possible to the second, and we measure the R², that is how much of the second is
# explained. R² = 1 means that the two maps are the same up to a linear transformation;
# the arbitrary orientation of the sphere therefore does not matter. It is done for every ordered pair of
# trainings (with 5 seeds, 20 pairs).
#
# We redo it by hand (scikit-learn's `LinearRegression`, as in the library) and compare it
# with `cebra.sklearn.metrics.consistency_score`, on the 1106 evaluation points:
#
# - the maps of the **models estimated on all trials** of §4.2 (5 seeds), 3- and 4-dimensional CEBRA;
# - **UMAP** with 2 and 3 dimensions, estimated on the same points with 5 seeds;
# - the maps of the **models of each block** of §4.1 (estimated on 12 trials), mean over blocks.
#
# PCA and Isomap do not depend on chance: their between-training consistency is 1 by construction.

# %%
from sklearn.linear_model import LinearRegression           # linear regression, as in the library function

def umap_fit(Xp, d, seed=0):                                # UMAP estimated on all the given points (03, §9.2)
    return UMAP(n_neighbors=K_NN, n_components=d, random_state=seed, n_jobs=1).fit_transform(Xp)

def run_consistency(maps):                                  # R² of the linear regression from each map to every other (ordered pairs)
    return np.array([LinearRegression().fit(A, B).score(A, B) for i, A in enumerate(maps)
                     for j, B in enumerate(maps) if i != j])

def consistency_library(maps):                              # the library function (in the separate process, because it imports torch)
    import cebra
    scores, pairs, ids = cebra.sklearn.metrics.consistency_score(maps, between="runs")
    return np.asarray(scores)

cons_maps = {f"CEBRA-Time, {od} dimensioni (stimato su tutti i trial)": [MAPS4T[(od, sd)][bins] for sd in SEEDS]
             for od in OUT_DIMS}
for d in (2, 3):                                            # UMAP on the same 1106 points, 5 seeds
    cons_maps[f"UMAP, d = {d}"] = Parallel(n_jobs=5)(delayed(umap_fit)(X_all, d, sd) for sd in SEEDS)
cons = {name: run_consistency(m_) for name, m_ in cons_maps.items()}
cons["CEBRA-Time, 3 dimensions (block models, mean over the 5 blocks)"] = np.array(
    [run_consistency([MAPS4[(3, sd, k)][bins] for sd in SEEDS]).mean() for k in range(5)])  # one mean per block
lib = POOL.submit(consistency_library, [m_.astype(np.float32) for m_ in cons_maps["CEBRA-Time, 3 dimensioni (stimato su tutti i trial)"]]).result()
mine = run_consistency([m_.astype(np.float32) for m_ in cons_maps["CEBRA-Time, 3 dimensioni (stimato su tutti i trial)"]])
print(f"By hand versus library (3-dimensional CEBRA, 20 pairs): maximum difference {np.abs(np.sort(lib) - np.sort(mine)).max():.1e}")
display(pd.Series({name: mean_range(v * 100) for name, v in cons.items()} | {"PCA, Isomap": "100 (by construction)"},
                  name="between-training consistency: R² (%), mean (min-max)").to_frame())

# %% [markdown]
# **Reading.**
#
# - **The calculation by hand matches the library** (difference 0 over the 20 pairs).
# - **CEBRA's maps change much more than UMAP's from one seed to another**: R² 63% with 3
#   dimensions and 75% with 4, against 87% and 93% for UMAP. Some CEBRA seed pairs are far apart
#   (minimum 47% with 3 dimensions): same data, same parameters, and two maps that are not linear
#   transformations of each other. With the models of the single blocks (12 trials) it is the same: 66%.
# - It is consistent with §4, where across seeds there were 5 points of difference in phase
#   decoding. We did not verify the reason; one hypothesis is that with little data and 1000 steps the network
#   settles into different solutions depending on the starting point.
# - **Consequence for the thesis**: a single CEBRA map must not be interpreted alone; several
#   seeds must be reported, and the shape seen in one map must be checked in the others.
#
# ### 5.3 · Does CEBRA-Time learn only the time chain?
#
# In 03 (§2) the "time chain" was this: on the smoothed data the neighbor of a time point is almost
# always "itself a moment before or after", in the same trial. CEBRA-Time does exactly this, by
# construction: it brings each time point close to the one 200 ms later. A map made **only** of the chain
# would put each trial on its own track, separate from the others, and would not serve to decode
# phases in new trials. That the decoding of §4 is far above chance already says that the map
# contains more. But another, subtler possibility remains: **the clock**. The phases always come
# in the same order, so "how long since the trial began" already says a lot about the phase. A
# map that encoded only the elapsed time, aligned across trials, would decode the phases without
# looking at movement-related activity.
#
# Four tests:
#
# - **(a) Who are the neighbors in the map?** The measure of 03 (§2): of the 10 neighbors of each point, how many
#   are time chain (same trial, within 200 ms), how many "same phase, other trial", how many
#   other. On the models estimated on all trials, compared with the input, PCA and UMAP.
# - **(b) The clock alone**: the common yardstick with a single number per point, the time elapsed
#   since the start of the trial.
# - **(c) How much clock is in each map**: with the same scheme (never-seen trials), how well a
#   10-neighbor regressor recovers the elapsed time from the map (R²).
# - **(d) Changing Δ**: positive pairs at 20 ms (Δ = 1) and at 1 s (Δ = 50) instead of 200 ms. With
#   Δ = 1 the game is trivial (the next frame of the film, almost identical thanks to the smoothing):
#   only the chain should remain. With Δ = 50 the two time points are often in different phases.
#   Caution: the 1000 steps were chosen for Δ = 10; for the other Δ we did not
#   recheck them. 2 seeds for each Δ, both per block (20 trainings) and on all trials (4).
#   About 3 minutes.

# %%
from scipy.spatial.distance import pdist, squareform        # distances between all pairs of points

K_MIX, CHAIN_BINS = 10, 10                                  # neighbors counted and time chain (same trial, within 10 bins = 200 ms), as in 03

def knn_index(Xp, k=K_MIX):                                 # indices of the k neighbors of each point, excluding the point itself (03, §2)
    D = squareform(pdist(Xp)); np.fill_diagonal(D, np.inf)
    return np.argpartition(D, k, axis=1)[:, :k]

def neighbor_mix(Y, b=bins):                                # composition of the neighbors of each point in a map Y (03, §2)
    nb = knn_index(Y)
    tr_, ph_ = trial[b], phase[b]
    chain = (tr_[nb] == tr_[:, None]) & (np.abs(b[nb] - b[:, None]) <= CHAIN_BINS)  # same trial, within 200 ms
    good = ~chain & (ph_[nb] == ph_[:, None]) & (tr_[nb] != tr_[:, None])           # same phase, other trial
    return {"time chain (%)": chain.mean() * 100, "same phase, other trial (%)": good.mean() * 100,
            "other (%)": (1 - chain.mean() - good.mean()) * 100}

DELTAS = (1, 50)                                            # Δ tried besides 10
res_from = run_cebra_cached([dict(X_train=Z25.astype(np.float32), X_apply=[Z25.astype(np.float32)],  # on all trials, as in §4.2
                                  cfg=dict(CEBRA_T, time_offsets=dl), seed=sd) for dl in DELTAS for sd in (0, 1)])
MAPS_FROM = dict(zip(product(DELTAS, (0, 1)), [r["maps"][0] for r in res_from]))

mix_rows = {"input: 25 PC": [neighbor_mix(X_all)],
            "PCA, d = 2": [neighbor_mix(PCA(2).fit_transform(X_all))],
            "UMAP, d = 2 (5 seeds)": [neighbor_mix(Y_) for Y_ in cons_maps["UMAP, d = 2"]],
            "CEBRA-Time, Δ = 10 (5 seeds)": [neighbor_mix(MAPS4T[(3, sd)][bins]) for sd in SEEDS],
            **{f"CEBRA-Time, Δ = {dl} (2 seeds)": [neighbor_mix(MAPS_FROM[(dl, sd)][bins]) for sd in (0, 1)] for dl in DELTAS}}
tab_mix = pd.DataFrame({name: pd.DataFrame(r_).mean() for name, r_ in mix_rows.items()}).T
ph_b, tr_b = phase[bins], trial[bins]
tab_mix.loc["at chance", "same phase, other trial (%)"] = np.mean([np.mean((ph_b == ph_b[i]) & (tr_b != tr_b[i])) for i in range(len(bins))]) * 100
display(tab_mix.round(1))

# %%
clock = np.zeros(n_bins)                                    # the clock: seconds elapsed since the start of the trial, for each bin
for t in TRIALS:
    clock[trial == t] = (np.arange((trial == t).sum())) * io.BIN_S
ACC["l'orologio da solo (1 numero)"] = decode_map(fold_inp(lambda k: clock[:, None]), emb_none)

def regress_map(inp, embed, target):                        # R² with which a 10-neighbor regressor recovers `target` from the map (clean estimate, as decode_map)
    b, folds = inp
    pred = np.empty(len(b))
    for Xtr, Xte, trn, tst in folds:
        Etr, Ete = embed(Xtr, Xte)
        pred[tst] = KNeighborsRegressor(n_neighbors=K_NN).fit(Etr, target[b[trn]]).predict(Ete)
    return r2_score(target[b], pred)

clock_r2 = {"PCA, d = 2": [regress_map(INP_PC, emb_pca(2), clock)],
            "Isomap, d = 2": [regress_map(INP_PC, emb_isomap(2), clock)],
            "UMAP, d = 2": Parallel(n_jobs=5)(delayed(regress_map)(INP_PC, emb_umap(2, sd), clock) for sd in SEEDS),
            "CEBRA-Time, 3 dimensioni": [regress_map(fold_inp(lambda k: MAPS4[(3, sd, k)]), emb_none, clock) for sd in SEEDS],
            "no map, 25 dimensions": [regress_map(INP_PC, emb_none, clock)]}
ACC_OF = {"no map, 25 dimensions": "tetto, 25 dimensioni"}  # where to find in ACC the accuracy of each row

def fmt(v, scale=1):                                        # "mean (min-max)" if there are several seeds, otherwise the number
    v = np.atleast_1d(v) * scale
    return mean_range(v) if len(v) > 1 else f"{v[0]:.1f}"

display(pd.DataFrame({name: {"clock recovered from the map (R², %)": fmt(v, 100),
                             "phase decoding (%)": fmt(ACC[ACC_OF.get(name, name)])}
                      for name, v in clock_r2.items()}).T)
acc_clock = ACC["l'orologio da solo (1 numero)"]
print(f"The clock alone decodes the phase at {acc_clock:.1f}% (chance 12.5%)")

# %% [markdown]
# **Do map and clock guess the same things?** The R² of the clock recovered from the maps is close to
# zero or negative, but on its own it is not enough to say that the maps contain nothing of the clock:
# swapping `Reach` (at the start of the trial) with `Orient2` (towards the end) costs many seconds of
# error, and the R² heavily penalizes large errors. A more direct, point-by-point check: for each
# test point we look at whether the map guessed it, the clock, both or neither (fractions of the
# 1106 points, without balancing the phases). If the map guessed almost only where the
# clock also guesses, it would be suspect. And phase by phase: which phase each one recognizes (the fraction of the points
# of that phase correctly recognized).

# %%
def predict_map(inp, embed):                                # the phase predictions of each point, with the clean estimate of decode_map
    b, folds = inp
    pred = np.empty(len(b), dtype=object)
    for Xtr, Xte, trn, tst in folds:
        Etr, Ete = embed(Xtr, Xte)
        pred[tst] = KNeighborsClassifier(n_neighbors=K_NN).fit(Etr, phase[b[trn]]).predict(Ete)
    return pred

ph_true = phase[bins]                                       # the true phase of each point
pred_clock = predict_map(fold_inp(lambda k: clock[:, None]), emb_none)  # the clock's predictions
PREDS = {"CEBRA-Time, 3 dimensions (5 seeds)": [predict_map(fold_inp(lambda k: MAPS4[(3, sd, k)]), emb_none) for sd in SEEDS],
         "UMAP, d = 2 (5 seeds)": Parallel(n_jobs=5)(delayed(predict_map)(INP_PC, emb_umap(2, sd)) for sd in SEEDS),
         "no map, 25 dimensions": [predict_map(INP_PC, emb_none)]}
ok_clock = pred_clock == ph_true                            # where the clock guesses right
tab_cc = pd.DataFrame({name: pd.DataFrame([{"correct for both": np.mean(ok & ok_clock) * 100,
                                            "map only": np.mean(ok & ~ok_clock) * 100,
                                            "clock only": np.mean(~ok & ok_clock) * 100,
                                            "wrong for both": np.mean(~ok & ~ok_clock) * 100}
                                           for ok in [p_ == ph_true for p_ in preds]]).mean()
                       for name, preds in PREDS.items()}).T
display(tab_cc.round(1))

recall = {"l'orologio": {p: np.mean(pred_clock[ph_true == p] == p) * 100 for p in io.MOVEMENT_PHASES}}
for name, preds in PREDS.items():                           # phase by phase, mean over seeds
    recall[name] = {p: np.mean([np.mean(p_[ph_true == p] == p) for p_ in preds]) * 100 for p in io.MOVEMENT_PHASES}
display(pd.DataFrame(recall).T.round(0))

# %%
res_dfold = run_cebra_cached([dict(X_train=X25[k][TRAIN_BINS[k]].astype(np.float32),  # as in §4.1, changing only Δ
                                   X_apply=[X25[k][trial == t].astype(np.float32) for t in TRIALS],
                                   cfg=dict(CEBRA_T, time_offsets=dl), seed=sd)
                              for dl in DELTAS for sd in (0, 1) for k in range(5)])
MAPS_DF = {c: np.concatenate(r["maps"]) for c, r in zip(product(DELTAS, (0, 1), range(5)), res_dfold)}
acc_delta = {10: ACC["CEBRA-Time, 3 dimensioni"][:2]}       # Δ = 10: the same two seeds (0 and 1) of §4.1
for dl in DELTAS:
    acc_delta[dl] = [decode_map(fold_inp(lambda k: MAPS_DF[(dl, sd, k)]), emb_none) for sd in (0, 1)]
display(pd.DataFrame({f"Δ = {dl} bins ({dl * 20} ms)": {
    "phase decoding (%), seeds 0 and 1": " and ".join(f"{a:.1f}" for a in acc_delta[dl]),
    "time chain among neighbors (%)": f"{tab_mix.loc[f'CEBRA-Time, Δ = {dl} ' + ('(5 seeds)' if dl == 10 else '(2 seeds)'), 'time chain (%)']:.1f}",
    "same phase, other trial (%)": f"{tab_mix.loc[f'CEBRA-Time, Δ = {dl} ' + ('(5 seeds)' if dl == 10 else '(2 seeds)'), 'same phase, other trial (%)']:.1f}"}
    for dl in sorted(acc_delta)}).T)

# %%
fig, ax = plt.subplots(1, 2, figsize=(14, 4.2), gridspec_kw={"width_ratios": (1.25, 1)})
cats, cols = ["time chain (%)", "same phase, other trial (%)", "other (%)"], ["C3", "C2", "0.82"]
rows_plot = [r_ for r_ in tab_mix.index if r_ != "at chance"]
for i, name in enumerate(rows_plot):                        # one bar per map, the three groups one after the other
    left = 0
    for c, col in zip(cats, cols):
        v = tab_mix.loc[name, c]
        ax[0].barh(i, v, left=left, color=col, label=c.replace(" (%)", "") if i == 0 else None)
        if v > 5:
            ax[0].text(left + v / 2, i, f"{v:.0f}%", ha="center", va="center", fontsize=8)
        left += v
ax[0].set_yticks(range(len(rows_plot))); ax[0].set_yticklabels(rows_plot); ax[0].invert_yaxis()
ax[0].set(xlim=(0, 100), xlabel="% of the 10 neighbors", title="Who are the 10 neighbors of each point? (maps estimated on all trials)")
ax[0].legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.17), fontsize=8)
names_c = list(clock_r2)
xr = [np.mean(clock_r2[n]) * 100 for n in names_c]          # how much clock there is in the map
ya = [np.mean(ACC[ACC_OF.get(n, n)]) for n in names_c]    # how well it decodes the phase
for n, x_, y_ in zip(names_c, xr, ya):
    ax[1].scatter(x_, y_, s=60, color="C3" if n.startswith("CEBRA") else ("C2" if n.startswith("UMAP") else
                                                                         "C0" if n.startswith("Isomap") else "C1" if n.startswith("PCA") else "0.35"))
    ax[1].annotate(n, (x_, y_), textcoords="offset points", xytext=(5, 4), fontsize=8)
ax[1].axhline(ACC["l'orologio da solo (1 numero)"], color="k", ls=":", lw=1.2, label="the clock alone")
ax[1].axhline(12.5, color="0.5", ls="--", lw=1, label="at chance")
ax[1].set(xlabel="clock recovered from the map (R², %)", ylabel="phase decoding (%)", ylim=(0, 60),
          title="Phase and clock, never-seen trials")
ax[1].legend(fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(FIGS / "04_time_chain.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# **Reading.**
#
# **(a) The time chain does not dominate the map.** In CEBRA-Time's map (Δ = 10) only 7.9% of the
# neighbors are time chain: less than in the input (20.4%) and in UMAP (16.1%). The "same phase,
# other trial" neighbors are 33.1%, as in UMAP (33.9%), more than in the input (27.1%) and much more than chance
# (12.7%). Although its game brings each time point close to the one 200 ms later, at the scale of the evaluation points
# (one every 100 ms) a point's neighbors come mostly from other trials.
#
# **(b) The clock alone decodes the phase at 48.5%**: more than all the maps and more than the ceilings
# (42.2% and 44.9%). Knowing only how long ago the trial began is enough to guess the phase better than
# with neural activity, with any method. It is not an error of the methods, which do not
# see the clock, but it changes how to read the absolute numbers of the common yardstick, those of 03 too: the
# comparison between methods stays on equal terms, but "40%" must be read next to what the structure of the task
# gives for free.
#
# **(c) The maps are not a clock.**
#
# - From the elapsed time the maps recover almost nothing: R² between −11% and +2% for all, CEBRA
#   included (−10.6%).
# - Point by point, map and clock guess **different things**: of the points that CEBRA recognizes
#   (20.5% + 17.5% = 38%), almost half (17.5%) are wrong for the clock. The same holds for
#   UMAP and for the classifier on the 25 dimensions.
# - Phase by phase the split is sharp. The clock recognizes the arm phases, which follow one another
#   at almost fixed times after the start of the trial: `Reach` 99%, `Orient` 81%, `Carry` 73%. It fails on
#   `Shape` (4%) and `SnapTo` (12%). The neural maps instead recognize the hand phases: `Shape`
#   66-75%, `Release` 60-68%, `Grasp` 45-59%, consistent with the position of the arrays in the hand
#   area (02 and 03). The arm phases, in the maps, get confused with each other (22-50%).
#
# **(d) Δ = 10 is the best value** of the three: 35.3% and 35.9% (seeds 0 and 1), against 30.5% and 27.8% with
# Δ = 1 and 21.8% and 23.9% with Δ = 50. With Δ = 1 the game is too easy; with Δ = 50 the positive often falls
# in another phase. The hypothesis "with Δ = 1 only the chain remains" is not confirmed: the chain among
# the neighbors drops (3.7%) together with everything else ("other" at 76%). The network simply learns
# less. Limitation: the 1000 steps were chosen for Δ = 10.
#
# **In summary**: CEBRA-Time with Δ = 10 does not learn only the time chain, nor a clock.
# It mostly learns the activity linked to the hand phases, like UMAP as far as neighbors are concerned,
# but with maps less stable from one seed to another (§5.2). And phase decoding, the common
# measure of all the notebooks, has a very high "no neurons" term of comparison: the clock.

# %% [markdown]
# ## 6 · CEBRA-Behavior: the map guided by the kinematics
#
# CEBRA-Time decided who "goes together" with time. **CEBRA-Behavior** decides it with a behavior
# label: two time points go together if the behavior is similar. It is therefore a
# **supervised** method (it uses labels), and it cannot be compared on equal terms with the methods without labels
# of §1-§5.
#
# **The labels: position and velocity together** (user's choice). For each bin, the 7 degrees of
# freedom of the position (`s.kin`) and the 7 of the velocity (`s.vel`, the FALCON target): 14 numbers.
# It is the **reference** kinematics shown by the virtual arm, not an executed movement
# (the participant attempts it: §1.1).
#
# **How CEBRA chooses the positive** (read in the code, `TimedeltaDistribution`): to the kinematics
# of the reference it adds a 200 ms variation (Δ = 10 bins) observed at a random time point of the
# session, and takes as positive the time point, **of any trial**, with the kinematics closest to
# the one obtained. In practice: "the time point, even in another trial, at which the arm is where
# this one will be shortly". The negatives remain random time points.
#
# **The labels must be put on the same scale.** The closest kinematics is searched with the
# straight-line distance; without correction the large numbers would weigh more (the velocities, in units per
# second) and the positions almost nothing. The library documentation says this explicitly.
# We z-score each of the 14 numbers, with the mean and standard deviation of the training trials only
# of each block.
#
# **What the comparison can and cannot say.**
#
# - The evaluation stays phase decoding, which is not the label: the test is not
#   circular. But the reference kinematics is almost the same in every repetition, so it carries
#   a lot of information about the phase. It also contains the **task version**: the grasp degree of freedom g3
#   goes in opposite directions in the odd and even trials (03, §9.2). We expect this, and we
#   measure it.
# - The right term of comparison is a **linear method with the same labels**: **PLS**
#   (*partial least squares*), which looks for the directions of the 25 PCs that best predict the kinematics,
#   estimated on the same training bins as CEBRA. With 2 components it is compared with 3-dimensional CEBRA-Behavior,
#   with 3 components with the 4-dimensional one.
# - **The paper's control**: shuffled labels. We make two versions (§6.4).
#
# First, though, the number of steps: with labels the overfitting may change, so we redo the
# check of §3 (§6.2).

# %%
from sklearn.cross_decomposition import PLSRegression       # PLS: linear projection guided by the labels

Y_KIN = np.c_[s.kin, s.vel]                                 # the labels: 7 positions + 7 velocities for each bin (12233 × 14)
KIN_NAMES = [f"position {n}" for n in io.KIN_LABELS] + [f"velocity {n}" for n in io.KIN_LABELS]

def label_z(train_trials):                                  # z-score of the 14 labels with mean and deviation of the training trials only
    m = np.isin(trial, train_trials)
    return ((Y_KIN - Y_KIN[m].mean(0)) / Y_KIN[m].std(0)).astype(np.float32)

def cebra_fit_beh(X_train, y_train, X_apply, cfg, seed, val=None, model_path=None):
    """Like cebra_fit (§2.3), but with labels: CEBRA-Behavior. val = {name: (X, y)}: sets on which to measure,
    at the end, the cost function with the same choice of positives. Runs in the separate process."""
    import time                                             # to measure the duration
    import numpy as np                                      # arrays
    import torch                                            # the neural network library used by CEBRA
    import cebra                                            # CEBRA
    import cebra.distributions.base as cdb                  # for the seed fix (§2.7)
    torch.set_num_threads(1)                                # the heavy computations are on the GPU
    if not hasattr(cdb.HasGenerator, "orig_init"):          # the same seed fix as cebra_fit (§2.7), identical code
        cdb.HasGenerator.orig_init = cdb.HasGenerator.__init__
        def seeded_init(self, device, seed):
            cdb.HasGenerator.orig_init(self, device=device, seed=seed)
            if cdb.HasGenerator.fixed_seed is not None:
                self._seed = cdb.HasGenerator.fixed_seed * 1000 + cdb.HasGenerator.counter
                cdb.HasGenerator.counter += 1
                self._generator = torch.Generator(device=self.device).manual_seed(self._seed)
        cdb.HasGenerator.__init__ = seeded_init
    cdb.HasGenerator.fixed_seed = seed                      # always with the fix
    cdb.HasGenerator.counter = 0
    torch.manual_seed(seed); np.random.seed(seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = cebra.CEBRA(**cfg, device=device, verbose=False)
    t0 = time.time()
    model.fit(X_train, y_train)                             # with a continuous label: CEBRA-Behavior (positives chosen with "time_delta")
    seconds = time.time() - t0
    if model_path is not None:
        model.save(str(model_path))
    extra = {name: float(cebra.sklearn.metrics.infonce_loss(model, Xv, yv, num_batches=200))  # cost function on given data, 200 batches
             for name, (Xv, yv) in (val or {}).items()}
    return {"maps": [model.transform(x) for x in X_apply], "loss": np.asarray(model.state_dict_["loss"]),
            "seconds": seconds, "device": device, "extra": extra}

print(f"Labels: {Y_KIN.shape[1]} numbers per bin; standard deviation before the z-score: "
      f"positions from {Y_KIN[:, :7].std(0).min():.2f} to {Y_KIN[:, :7].std(0).max():.2f}, "
      f"velocities from {Y_KIN[:, 7:].std(0).min():.2f} to {Y_KIN[:, 7:].std(0).max():.2f}")

# %% [markdown]
# ### 6.2 · How many steps with labels?
#
# The scheme of §3.2: block 0, two pairs of validation trials (12-15 and 6-7), PCA and labels
# estimated on the 10 training trials only. Here we measure CEBRA-Behavior's cost function (the
# same choice of positives, with the labels of the measured trials) at the **end** of trainings of
# 1, 500, 1000, 2000, 5000 and 10 000 steps (with 1 step the network is practically the initial one), on the 2
# validation trials and, for comparison, on the 2 training trials 4 and 5. Same rule as §3.2:
# the smallest number of steps with 98% of the descent, rounded to the thousands; between the two
# pairs, the larger. As a check, phase decoding on trials 12 and 15. About 4
# minutes.

# %%
ITER_BEH = (1, 500, 1000, 2000, 5000, 10_000)               # numbers of steps tried
PAIRS = {"A": (VAL_TRIALS, fit_trials, SEQ_A, 0), "B": (VAL_TRIALS_B, fit_trials_b, SEQ_B, 1)}  # pair: (validation, training, input, seed)
jobs_it, keys_it = [], []
for name, (vt, ft, seq, sd) in PAIRS.items():
    Yz = label_z(ft)                                        # labels z-scored with the 10 training trials only
    ylab = {t: Yz[trial == t] for t in TRIALS}              # the labels of each trial
    val_sets = {f"2 training trials {CHECK_TRIALS}": (np.concatenate([seq[t] for t in CHECK_TRIALS]), np.concatenate([ylab[t] for t in CHECK_TRIALS])),
                f"2 validation trials {vt}": (np.concatenate([seq[t] for t in vt]), np.concatenate([ylab[t] for t in vt]))}
    for n_it in ITER_BEH:
        jobs_it.append(dict(X_train=np.concatenate([seq[t] for t in ft]), y_train=np.concatenate([ylab[t] for t in ft]),
                            X_apply=[seq[t] for t in TRIALS], cfg=dict(CEBRA_STD, max_iterations=n_it), seed=sd, val=val_sets))
        keys_it.append((name, n_it))
res_it_b = dict(zip(keys_it, run_cebra_cached(jobs_it, fn=cebra_fit_beh)))

N_ITER_BEH = 0
for name, (vt, ft, seq, sd) in PAIRS.items():
    h_ = pd.DataFrame({n_it: res_it_b[(name, n_it)]["extra"] for n_it in ITER_BEH}).T.rename_axis("steps")
    v_ = h_[f"2 validation trials {vt}"]
    n_ = rule98(v_)                                         # the rule of §3.2
    N_ITER_BEH = max(N_ITER_BEH, n_)
    print(f"Pair {name} (validation {vt}, seed {sd}): minimum {v_.min():.4f} at {v_.idxmin()} steps; 98% rule -> {n_} steps")
    display(h_.assign(**{"goodness of fit, validation (bits)": bits(v_)}).round(3))
acc_it_b = {n_it: val_acc(np.concatenate(res_it_b[("A", n_it)]["maps"])) for n_it in ITER_BEH}  # phase on trials 12 and 15 (§3.2)
print("Phase decoded on trials 12 and 15 (%):", {n_it: round(a, 1) for n_it, a in acc_it_b.items()})
print(f"Number of steps chosen for CEBRA-Behavior: N_ITER_BEH = {N_ITER_BEH}")

# %% [markdown]
# **Reading.**
#
# - **As for CEBRA-Time, overfitting sets in early.** On the validation trials the loss is already at
#   its minimum at 500 steps (5.698 for trials 12-15, 5.681 for trials 6-7), then rises up to
#   10 000 (5.811 and 5.759), while on training trials 4-5 it keeps decreasing. The rule gives
#   1000 steps for both pairs: **N_ITER_BEH = 1000**.
# - The validation goodness of fit reaches 0.78-0.80 bits, against 0.63-0.68 for CEBRA-Time: the
#   numbers are not comparable, because the positives are chosen differently.
# - **First signal to keep in mind**: on trials 12 and 15 the phase is decoded at 31.2% with 500-1000
#   steps, and less afterwards; CEBRA-Time, at the same point, was at 41-44% (§3.2).
#
# ### 6.3 · The common yardstick
#
# CEBRA-Behavior with 3 and 4 dimensions, 5 seeds × 5 blocks, with the labels of the training trials only
# (test trials are projected without labels, as always). Alongside: CEBRA-Time (§4), PLS with
# 2 and 3 components, PCA and UMAP, the ceilings and the clock (§5.3). Two extra measures, to see what
# the labels put into the map:
#
# - **the task version**: from the map, with the same 10-neighbor classifier, can we tell whether
#   a point comes from an odd or an even trial? (balanced accuracy; chance 50%);
# - **the kinematics**: with a 10-neighbor regressor, how well can position and velocity be recovered (mean R²
#   over the 7 degrees of freedom), again on unseen trials.
#
# 50 trainings: about 6 minutes.

# %%
CEBRA_B = dict(CEBRA_STD, max_iterations=N_ITER_BEH)        # CEBRA-Behavior with the number of steps from §6.2
YZ_F = [label_z(np.unique(trial[TRAIN_BINS[k]])) for k in range(5)]  # z-scored labels per block (statistics of the training trials)
res6 = run_cebra_cached([dict(X_train=X25[k][TRAIN_BINS[k]].astype(np.float32), y_train=YZ_F[k][TRAIN_BINS[k]],
                              X_apply=[X25[k][trial == t].astype(np.float32) for t in TRIALS],
                              cfg=dict(CEBRA_B, output_dimension=od), seed=sd) for od, sd, k in COMBOS], fn=cebra_fit_beh)
MAPS6 = {c: np.concatenate(r["maps"]) for c, r in zip(COMBOS, res6)}
for od in OUT_DIMS:
    ACC[f"CEBRA-Behavior, {od} dimensioni"] = [decode_map(fold_inp(lambda k: MAPS6[(od, sd, k)]), emb_none) for sd in SEEDS]

PLS_F = {nc: [PLSRegression(n_components=nc).fit(X25[k][TRAIN_BINS[k]], YZ_F[k][TRAIN_BINS[k]]) for k in range(5)]
         for nc in (2, 3)}                                  # PLS on the same training bins and with the same labels as CEBRA
for nc in (2, 3):
    ACC[f"PLS, {nc} componenti"] = decode_map(fold_inp(lambda k: PLS_F[nc][k].transform(X25[k])), emb_none)

VERSION = np.where(trial % 2 == 1, "odd", "even")       # the task version of each bin

def decode_label(inp, embed, lab):                          # like decode_map, but with a different label to decode
    b, folds = inp
    pred = np.empty(len(b), dtype=object)
    for Xtr, Xte, trn, tst in folds:
        Etr, Ete = embed(Xtr, Xte)
        pred[tst] = KNeighborsClassifier(n_neighbors=K_NN).fit(Etr, lab[b[trn]]).predict(Ete)
    return balanced_accuracy_score(lab[b], pred) * 100

def kin_r2(inp, embed, cols):                               # mean R² with which a 10-neighbor regressor recovers some kinematics columns
    b, folds = inp
    pred = np.empty((len(b), len(cols)))
    for Xtr, Xte, trn, tst in folds:
        Etr, Ete = embed(Xtr, Xte)
        pred[tst] = KNeighborsRegressor(n_neighbors=K_NN).fit(Etr, Y_KIN[b[trn]][:, cols]).predict(Ete)
    return r2_score(Y_KIN[b][:, cols], pred) * 100          # mean of the 7 R²

POS, VEL = list(range(7)), list(range(7, 14))

def umap_fold(d, sd, k):                                    # UMAP of block k as in emb_umap (§1.2): (training points, test points)
    Xtr, Xte, trn, tst = INP_PC[1][k]
    u = UMAP(n_neighbors=K_NN, n_components=d, min_dist=0.1, random_state=sd, n_jobs=1).fit(Xtr)
    return u.embedding_, u.transform(Xte)

UMAP_F = dict(zip(product((2, 3), SEEDS, range(5)),         # estimated once and reused for the three measures
                  Parallel(n_jobs=5)(delayed(umap_fold)(d, sd, k) for d, sd, k in product((2, 3), SEEDS, range(5)))))

def umap_inp(d, sd):                                        # the common-yardstick input with the UMAP maps already estimated
    return (bins, [(*UMAP_F[(d, sd, k)], trn, tst) for k, (trn, tst) in enumerate(FOLDS)])

EXTRA_MAPS = {                                              # the maps on which to measure version and kinematics: [(input, function)], one per seed
    "PCA, d = 2": [(INP_PC, emb_pca(2))], "PCA, d = 3": [(INP_PC, emb_pca(3))],
    "UMAP, d = 2": [(umap_inp(2, sd), emb_none) for sd in SEEDS], "UMAP, d = 3": [(umap_inp(3, sd), emb_none) for sd in SEEDS],
    **{f"CEBRA-Time, {od} dimensioni": [(fold_inp(lambda k, od=od, sd=sd: MAPS4[(od, sd, k)]), emb_none) for sd in SEEDS] for od in OUT_DIMS},
    **{f"CEBRA-Behavior, {od} dimensioni": [(fold_inp(lambda k, od=od, sd=sd: MAPS6[(od, sd, k)]), emb_none) for sd in SEEDS] for od in OUT_DIMS},
    **{f"PLS, {nc} componenti": [(fold_inp(lambda k, nc=nc: PLS_F[nc][k].transform(X25[k])), emb_none)] for nc in (2, 3)},
    "tetto, 25 dimensioni": [(INP_PC, emb_none)],
}
EXTRA = {name: {"versione riconosciuta (%)": [decode_label(i_, e_, VERSION) for i_, e_ in lst],
                "posizione ricavata (R², %)": [kin_r2(i_, e_, POS) for i_, e_ in lst],
                "velocità ricavata (R², %)": [kin_r2(i_, e_, VEL) for i_, e_ in lst]} for name, lst in EXTRA_MAPS.items()}

ROWS_6 = [("PCA, d = 2", 2, "no"), ("UMAP, d = 2", 2, "no"), ("CEBRA-Time, 3 dimensioni", 2, "no"),
          ("PLS, 2 componenti", 2, "sì"), ("CEBRA-Behavior, 3 dimensioni", 2, "sì"),
          ("PCA, d = 3", 3, "no"), ("UMAP, d = 3", 3, "no"), ("CEBRA-Time, 4 dimensioni", 3, "no"),
          ("PLS, 3 componenti", 3, "sì"), ("CEBRA-Behavior, 4 dimensioni", 3, "sì"),
          ("tetto, 25 dimensioni", 25, "no"), ("tetto alla pari: media sulla finestra (25 numeri)", 25, "no"),
          ("l'orologio da solo (1 numero)", 1, "(the time)")]
tab_6 = pd.DataFrame([{"method": name, "free coordinates": free, "uses kinematics": lab,
                       "phase decoded (%)": fmt(ACC[name]),
                       **{c: fmt(v) for c, v in EXTRA.get(name, {}).items()}} for name, free, lab in ROWS_6]).set_index("method").fillna("")
display(tab_6)

# %% [markdown]
# **Reading.** The result is negative, and clear-cut.
#
# - **CEBRA-Behavior decodes the phases worse than CEBRA-Time**: 25.6% (22.4-27.9) with 2 free
#   coordinates, against 37.7%; 28.7% (27.3-30.1) with 3, against 40.4%. It is below PCA with d = 3 (32.4%) and
#   roughly on par with PCA with d = 2 (26.0%). The labels, here, make the map worse.
# - **PLS, with the same labels, does the same**: 27.3% and 28.3%. So this is not a problem of the
#   nonlinear method: the directions of activity that follow these labels are not the ones that
#   distinguish the phases. CEBRA-Behavior does no better than its linear comparison term.
# - **The task version is not decoded** from any map: 47-55%, with chance at 50%. Not even
#   from CEBRA-Behavior (49.2-49.9%), which did have g3 among the labels. The two `Shape` islands in 03
#   concerned a single phase.
# - **The kinematics, on unseen trials, cannot be recovered from the maps**: for position the R² is
#   negative with all methods, even with the classifier on the 25 dimensions (−16%); for
#   velocity the maps lie between −6% and +7%, the 25 dimensions at 19%. The labels entered
#   CEBRA-Behavior during training, but do not come out on new trials. The reason is in §6.5.
#
# ### 6.4 · The control with shuffled labels
#
# The CEBRA paper also trains the model with shuffled labels, "permuting all samples
# in time" (Extended Data Fig. 5): if the map stays good, it does not come from the link between activity and
# labels. We make two versions, on the training trials of each block:
#
# - **shuffled in time** (as in the paper): each bin receives the kinematics of a random bin. The
#   labels have neither shape nor continuity any more: we expect chance, or close to it;
# - **swapped between trials**: each training trial receives the kinematics of an **other** training
#   trial, aligned at the start (if the other is shorter, its last value is repeated). This is a
#   stricter control: the labels remain realistic movements, in the same order of phases, but
#   are no longer aligned with the activity of that trial. Given the clock of §5.3, part
#   of the information might remain even so.
#
# CEBRA-Behavior with 3 dimensions, 5 seeds × 5 blocks for each control; for comparison, PLS with
# 2 components with the same shuffled labels. 50 trainings: about 6 minutes.

# %%
def shuffled_labels(k, how, seed):                          # the labels of the training trials of block k, shuffled
    rng_s = np.random.default_rng(seed)
    tb = TRAIN_BINS[k]
    y_ = YZ_F[k][tb]
    if how == "time":                                      # each bin receives the label of a random bin
        return y_[rng_s.permutation(len(tb))]
    trs = list(np.unique(trial[tb]))                        # the training trials
    while True:                                             # a shuffle in which no trial keeps its own kinematics
        perm = rng_s.permutation(trs)
        if not np.any(perm == np.array(trs)):
            break
    out = []
    for t, t_other in zip(trs, perm):                       # each trial receives the kinematics of another, aligned at the start
        n_t, y_o = (trial == t).sum(), YZ_F[k][trial == t_other]
        out.append(y_o[:n_t] if len(y_o) >= n_t else np.r_[y_o, np.repeat(y_o[-1:], n_t - len(y_o), axis=0)])
    return np.concatenate(out).astype(np.float32)

CONTROLS = ("time", "trial")
res_sh = run_cebra_cached([dict(X_train=X25[k][TRAIN_BINS[k]].astype(np.float32), y_train=shuffled_labels(k, how, sd),
                                X_apply=[X25[k][trial == t].astype(np.float32) for t in TRIALS],
                                cfg=dict(CEBRA_B, output_dimension=3), seed=sd)
                           for how in CONTROLS for sd in SEEDS for k in range(5)], fn=cebra_fit_beh)
MAPS_SH = {c: np.concatenate(r["maps"]) for c, r in zip(product(CONTROLS, SEEDS, range(5)), res_sh)}
rows_sh = {"true labels": {"CEBRA, 3 dimensions": fmt(ACC["CEBRA-Behavior, 3 dimensioni"]),
                           "PLS, 2 components": fmt(ACC["PLS, 2 componenti"])}}
for how, lab in (("time", "shuffled in time (as in the paper)"), ("trial", "swapped between trials")):
    pls_sh = [decode_map(fold_inp(lambda k, sd=sd: PLSRegression(n_components=2).fit(
                  X25[k][TRAIN_BINS[k]], shuffled_labels(k, how, sd)).transform(X25[k])), emb_none) for sd in SEEDS]
    rows_sh[lab] = {"CEBRA, 3 dimensions": fmt([decode_map(fold_inp(lambda k, sd=sd: MAPS_SH[(how, sd, k)]), emb_none)
                                                          for sd in SEEDS]),
                    "PLS, 2 components": fmt(pls_sh)}
rows_sh["no labels (CEBRA-Time, §4)"] = {"CEBRA, 3 dimensions": fmt(ACC["CEBRA-Time, 3 dimensioni"])}
display(pd.DataFrame(rows_sh).T.fillna("").rename_axis("labels").rename(columns=lambda c: f"phase decoded (%): {c}"))

# %%
fig = plt.figure(figsize=(17, 3.6))
panels6 = [("CEBRA-Time (block 0, seed 0)", MAPS4[(3, 0, 0)], phase, io.PHASE_COLORS),
           ("CEBRA-Behavior (block 0, seed 0): color = phase", MAPS6[(3, 0, 0)], phase, io.PHASE_COLORS),
           ("CEBRA-Behavior (block 0, seed 0): color = version", MAPS6[(3, 0, 0)], VERSION, {"odd": "#e6ab02", "even": "0.15"}),
           ("labels swapped between trials (block 0, seed 0)", MAPS_SH[("trial", 0, 0)], phase, io.PHASE_COLORS)]
for j, (ttl, E, lab, colors) in enumerate(panels6):
    lon_, lat_ = to_lonlat(E)
    a_ = fig.add_subplot(1, 4, j + 1, projection="mollweide")
    for v_, c_ in colors.items():
        m = bins[lab[bins] == v_]
        a_.scatter(lon_[m], lat_[m], s=4, alpha=0.7, color=c_)
    a_.set_title(ttl, fontsize=9); a_.set_xticklabels([]); a_.set_yticklabels([])
handles = [plt.Line2D([], [], color=io.PHASE_COLORS[p], marker="o", ls="", ms=5, label=p) for p in io.MOVEMENT_PHASES]
handles += [plt.Line2D([], [], color=c_, marker="o", ls="", ms=5, label=f"trial {v_}") for v_, c_ in (("odd", "#e6ab02"), ("even", "0.15"))]
fig.legend(handles=handles, ncol=10, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.04))
fig.tight_layout(rect=(0, 0.07, 1, 1)); fig.savefig(FIGS / "04_behavior_maps.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# **Reading.**
#
# - **Labels shuffled in time** (the paper's control): 15.1% (12.9-17.3), almost chance;
#   PLS 14.8%. With unstructured labels CEBRA-Behavior learns almost nothing, as
#   in the paper. So the labels matter.
# - **Labels swapped between trials**: CEBRA reaches **29.8%** (28.1-31.3), **more than with the
#   true labels** (25.6%), while PLS drops to chance (16.4%). This is the alarm bell we were
#   looking for. An explanation consistent with §6.5, which we have not verified directly:
#   the swapped labels are smooth in time but unrelated to that trial, so the nearest
#   kinematics is usually that of a nearby instant **of the same trial**, and
#   CEBRA-Behavior ends up behaving like CEBRA-Time (positives close in time), approaching
#   its result (37.7%). With the true labels, instead, some of the positives are taken from
#   other trials, at moments with similar kinematics but in a different phase, and the phases get mixed.
#   This could be checked by counting how many positives fall in the same trial; we did not do it.
# - **In the maps** (figure): with CEBRA-Time the phases occupy distinct regions; with CEBRA-Behavior they are
#   mixed, and so are the two versions; with the swapped labels some regions reappear
#   (for example `Shape`, in red).
# - The paper's control says that "the labels matter"; here they matter, but in the wrong
#   direction for phase decoding.
#
# ### 6.5 · Why these labels do not help to decode the phases
#
# Two questions about the kinematics, independent of CEBRA:
#
# 1. **Does the kinematics repeat from one trial to the next?** For each degree of freedom of the position
#    we compare the total variability (over the 1106 points) with the variability between repetitions of the
#    same phase (the mean position of each repetition, then the standard deviation across
#    repetitions, averaged over phases). If the second is small relative to the first, each phase has its
#    own position; if it is equal, the position changes from one trial to the next more than from one phase
#    to the next.
# 2. **How much of the kinematics can be recovered from the activity**, degree by degree, on unseen trials: with
#    the 10-neighbor regressor of the common yardstick and with a linear regression (*ridge*), both on the
#    25 PCs of a single bin. This is not the FALCON benchmark (which uses all bins and a history of
#    previous bins): it is a control with the same points and the same blocks as the rest of the notebook.

# %%
from sklearn.linear_model import Ridge                      # linear regression with a small penalty

rows_k = {}
for j, n in enumerate(io.KIN_LABELS):                       # the 7 degrees of freedom of the position
    y_ = pd.Series(Y_KIN[bins, j])
    per_rep = y_.groupby([phase[bins], trial[bins]]).mean() # mean position of each repetition of each phase
    rows_k[f"position {n}"] = {"total variability (std. dev.)": y_.std(),
                               "variability between repetitions of the same phase": per_rep.groupby(level=0).std().mean()}
display(pd.DataFrame(rows_k).T.round(3))

def kin_pred_r2(make_model):                                # R² of each degree of freedom, with a model fitted on the training points of each block
    pred = np.empty((len(bins), Y_KIN.shape[1]))
    for Xtr, Xte, trn, tst in INP_PC[1]:
        pred[tst] = make_model().fit(Xtr, Y_KIN[bins[trn]]).predict(Xte)
    return r2_score(Y_KIN[bins], pred, multioutput="raw_values")

display(pd.DataFrame({"10 neighbors, 25 PC": kin_pred_r2(lambda: KNeighborsRegressor(n_neighbors=K_NN)),
                      "linear (ridge), 25 PC": kin_pred_r2(lambda: Ridge(alpha=1.0))}, index=KIN_NAMES).T.round(2))

# %% [markdown]
# **Reading.**
#
# - **The position changes from one trial to the next more than from one phase to the next.** For the translation
#   (tx, ty, tz), the orientation (rx) and g3, the variability between repetitions of the same phase is as large
#   as the total one (for example 0.514 against 0.525 for rx). The reference arm, that is,
#   does not go to the same place at each repetition, and g3 changes with the version. Only g1 and g2, two
#   grasp degrees, repeat phase by phase (0.15 against 0.37-0.39). We did not check in the
#   dataset paper why the positions change (different objects or targets from one trial
#   to the next?).
# - **From the activity, the position cannot be recovered** on new trials: negative R² for all 7 degrees,
#   even with the linear regression. **The velocities of the hand and wrist can be recovered instead**: rx
#   0.38-0.39, g1 0.20-0.39, g2 0.15-0.25, g3 0.22-0.54. Those of the translation cannot (between −0.14 and 0.03). This is
#   consistent with the position of the arrays, in the hand area.
# - So the chosen labels mix what the activity encodes (the velocities of hand and wrist)
#   with what it does not encode and which does not follow the phases (the positions, the translation velocities).
#   CEBRA-Behavior is pushed to bring together moments with similar kinematics, which for the activity
#   and for the phases have nothing in common: the map gets worse.
# - **The lesson**: with CEBRA-Behavior the choice of labels decides almost everything, and a label
#   that is "natural" for a BCI (position and velocity) can be the wrong one to describe the
#   manifold. Better-suited labels (velocity only, hand and wrist only, or the phases) are among the
#   tests of §7.

# %% [markdown]
# ## 7 · Sensitivity analysis and alternatives
#
# We change **one choice at a time** with respect to the main CEBRA-Time configuration (§4): 3
# dimensions on the sphere, cosine distance, τ = 1, `offset10-model` network (200 ms window), Δ = 10,
# 1000 steps, 25 PCs of the smoothed firing rates, all bins of the training trials. For each variant:
# phase decoding with the common yardstick, 5 blocks.
#
# **How many seeds.** In §4 the CEBRA seeds gave results up to 5 points apart: with 1 or 2 seeds many
# differences between variants would be just noise. We therefore use **3 seeds** (0, 1, 2), and as the
# comparison term the main configuration **with the same 3 seeds** (not with the mean of the 5 of §4).
# Differences of 2-3 points remain within the noise.
#
# Three groups:
#
# - **7a · the model**: dimensions, temperature, network window, number of steps, distance;
# - **7b · the input**: 160 channels without PCA, raw counts, GPFA factors, movement bins only;
# - **7c · the CEBRA-Behavior labels**: velocity only, hand and wrist velocity only, phases.
#
# ### 7a · The model
#
# | variant | what changes | why try it |
# |---|---|---|
# | 8 and 16 dimensions | more coordinates on the sphere (the classifier uses them all) | 3 dimensions is the paper's minimum; with more room the sphere weighs less (§2.4) |
# | τ = 0.1, τ = 2.25, learned τ | the temperature | 2.25 is the paper's value for CEBRA-Time (Extended Data Fig. 3); 0.1 the harsh judge of §2.6; "learned" lets the network decide (`temperature_mode="auto"`, minimum 0.1) |
# | `offset1-model`, `offset5-model`, `offset36-model` | the network window: 1 bin (no window), 5 bins (100 ms), 36 bins (720 ms, deeper network) | how much looking at time matters |
# | 500, 2000, 5000 steps | the trainings | the choice of §3 on the test trials (2000-5000 is the paper's range) |
# | Euclidean distance | no sphere (`offset10-model-mse`) | 3 free coordinates instead of 2 (§2.4) |
#
# 12 variants × 3 seeds × 5 blocks = 180 trainings: about 25-30 minutes the first time.

# %%
SEEDS3 = (0, 1, 2)                                          # the seeds of the sensitivity analyses

def variant_jobs(cfg, X_full=None, train_bins=None, labels=None):  # the trainings of one variant: 3 seeds × 5 blocks
    jobs = []
    for sd in SEEDS3:
        for k in range(5):
            Xk = (X25[k] if X_full is None else X_full[k]).astype(np.float32)  # the whole-session input for block k
            tb = TRAIN_BINS[k] if train_bins is None else train_bins[k]       # the bins on which we train
            j = dict(X_train=Xk[tb], X_apply=[Xk[trial == t] for t in TRIALS], cfg=cfg, seed=sd)
            if labels is not None:
                j["y_train"] = labels[k][tb]                # the labels (only for CEBRA-Behavior, §7c)
            jobs.append(j)
    return jobs

def run_variants(variants, fn=None):                        # all the variants in a single queue; returns the 3 accuracies of each
    names, jobs = [], []
    for name, v in variants.items():
        js = variant_jobs(**v)
        jobs += js; names += [name] * len(js)
    res = run_cebra_cached(jobs, fn=fn)
    out, maps_out = {}, {}
    for name in variants:
        rs = [r for n_, r in zip(names, res) if n_ == name] # the 15 results of the variant: outer seed, inner block
        maps = {(sd, k): np.concatenate(rs[i * 5 + k]["maps"]) for i, sd in enumerate(SEEDS3) for k in range(5)}
        out[name] = [decode_map(fold_inp(lambda k, sd=sd: maps[(sd, k)]), emb_none) for sd in SEEDS3]
        maps_out[name] = maps
    return out, maps_out

MAIN = "principale (3 dimensioni, τ = 1, offset10, 1000 passi, coseno)"
VAR_A = {MAIN: dict(cfg=CEBRA_T),                           # the same trainings as in §4 (seeds 0-2): they are read back from disk
         "8 dimensioni": dict(cfg=dict(CEBRA_T, output_dimension=8)),
         "16 dimensioni": dict(cfg=dict(CEBRA_T, output_dimension=16)),
         "τ = 0.1": dict(cfg=dict(CEBRA_T, temperature=0.1)),
         "τ = 2.25 (articolo, CEBRA-Time)": dict(cfg=dict(CEBRA_T, temperature=2.25)),
         "τ imparata (partenza 1, minimo 0.1)": dict(cfg=dict(CEBRA_T, temperature_mode="auto", min_temperature=0.1)),
         "rete senza finestra (offset1-model)": dict(cfg=dict(CEBRA_T, model_architecture="offset1-model")),
         "finestra di 100 ms (offset5-model)": dict(cfg=dict(CEBRA_T, model_architecture="offset5-model")),
         "finestra di 720 ms (offset36-model)": dict(cfg=dict(CEBRA_T, model_architecture="offset36-model")),
         "500 passi": dict(cfg=dict(CEBRA_T, max_iterations=500)),
         "2000 passi": dict(cfg=dict(CEBRA_T, max_iterations=2000)),
         "5000 passi": dict(cfg=dict(CEBRA_T, max_iterations=5000)),
         "distanza euclidea (offset10-model-mse)": dict(cfg=dict(CEBRA_T, distance="euclidean", model_architecture="offset10-model-mse"))}
ACC_A, MAPS_A = run_variants(VAR_A)
print("Check: the main configuration with seeds 0-2 matches §4:",
      np.allclose(ACC_A[MAIN], ACC["CEBRA-Time, 3 dimensioni"][:3]))

def sens_table(acc, ref_name):                              # table: mean (min-max) and difference from the main one, seed by seed
    ref = np.array(acc[ref_name])
    return pd.DataFrame({name: {"phase decoded (%), 3 seeds": mean_range(v),
                                "difference from the main one (points), mean": f"{np.mean(np.array(v) - ref):+.1f}",
                                "difference, seed by seed": " ".join(f"{d_:+.1f}" for d_ in np.array(v) - ref)}
                         for name, v in acc.items()}).T

display(sens_table(ACC_A, MAIN))

# %% [markdown]
# **A comparison at equal free coordinates also for the higher dimensions.** A sphere in 8
# dimensions has 7 free coordinates, one in 16 has 15: the rule of §4 asks to compare them with
# PCA and UMAP at d = 7 and d = 15, which we did not have so far. With the same blocks of the common yardstick and,
# for UMAP, the same 3 seeds.

# %%
HI_ACC = {}                                                 # (method, free coordinates) -> accuracy (or list of 3, one per seed)
for d, od in ((7, 8), (15, 16)):
    HI_ACC[("PCA", d)] = decode_map(INP_PC, emb_pca(d))
    HI_ACC[("UMAP", d)] = Parallel(n_jobs=3)(delayed(decode_map)(INP_PC, emb_umap(d, sd)) for sd in SEEDS3)
    HI_ACC[("CEBRA-Time", d)] = ACC_A[f"{od} dimensioni"]
display(pd.DataFrame({f"{d} free coordinates": {m: fmt(HI_ACC[(m, d)]) for m in ("PCA", "UMAP", "CEBRA-Time")}
                      for d in (7, 15)}).T.rename(columns={"UMAP": "UMAP (3 seeds)", "CEBRA-Time": "CEBRA-Time (3 seeds)"}))

# %% [markdown]
# **Reading.** The main configuration, with seeds 0-2, scores 36.1% (35.3-37.0): less than the mean
# of the 5 seeds of §4 (37.7%), because these three are the lowest seeds. This is why we compare
# the variants seed by seed.
#
# - **The number of dimensions is the choice that matters most.** With 8 dimensions 43.9% (+7.8; from +5.4 to
#   +10.3 depending on the seed), with 16 42.6% (+6.5). At equal free coordinates CEBRA-Time is worth
#   UMAP (43.9% against 43.3% with 7 coordinates; 42.6% against 43.6% with 15) and beats PCA (39.0% and
#   40.7%). The disadvantage with respect to UMAP in §4 therefore comes from the **few dimensions**: with 2-3
#   free coordinates the CEBRA sphere, which pushes the points to cover all of it, compresses worse than
#   UMAP. With more room the two methods are equivalent and both reach the ceiling of the 25 dimensions
#   (42.2%), close to the like-for-like ceiling (44.9%).
# - **Euclidean distance: 40.4%** (+4.3; from +3.9 to +5.0): 3 free coordinates instead of 2, and indeed
#   the same value as CEBRA with 4 dimensions and cosine (40.4%, §4). The free-coordinates
#   rule holds.
# - **The temperature does not matter**, within the noise: τ = 0.1 +1.5 (one seed +4.6, the others 0), τ = 2.25
#   +0.1, learned τ +0.7. The paper's value for CEBRA-Time (2.25) changes nothing here.
# - **The network window matters little.** Without a window (a single bin) −1.0; 100 ms +1.2; 720 ms
#   +2.5, positive with all three seeds. CEBRA without a window is almost as good as with the window:
#   its advantage over PCA (26-32% with 2-3 coordinates) does not come from looking at 200 ms, consistent with
#   §4 (CEBRA does not exploit the extra information in the window).
# - **The steps**: 500 +0.7, 2000 +0.3, 5000 −2.0 (negative with all three seeds). On the test trials the
#   choice of §3 is found again: training longer makes things worse.

# %% [markdown]
# ### 7b · The input
#
# | variant | what CEBRA receives | why try it |
# |---|---|---|
# | 160 smoothed channels | the smoothed and z-scored firing rates, **without PCA** ("everything in input") | the network can choose the channel combinations by itself; PCA might remove useful information |
# | 160 channels, raw counts | the 20 ms counts, z-scored, with neither smoothing nor PCA | the rawest input: the network window can act as smoothing |
# | 25 GPFA factors | the GPFA factors (02_new, §10), estimated per block on the training trials only | in 03 it was the best input for Isomap and UMAP (§7.2 and §9.1) |
# | movement bins only | the 25 PCs, but CEBRA trains only on the movement bins of the training trials | choice B of §1.4: the pauses took up half of the movie |
#
# For each input the comparison term changes too: alongside CEBRA we put the classifier
# without a map and UMAP with d = 2 (seeds 0-2) **on the same input** and on the same 1106 points.
#
# **GPFA factors on the whole session.** In 03 the factors existed only in the 1 s windows
# from the start of the phases (872 points out of 1106). CEBRA needs a continuous signal. We estimate the
# GPFA as in 03 (square root of the counts, 25 factors, 500 iterations, on the 1 s windows of the training
# trials only of each block), then compute the factors on the **whole session**, trial by
# trial, in overlapping 2 s segments: from each segment we keep the central second, so each
# bin comes from a segment in which it is far from the edges. The factors are rotated and ordered as in 02_new
# (`orthonormalize`). In this way they are evaluated on the same 1106 points as the whole notebook.
#
# 4 variants × 15 trainings, plus 5 GPFA estimates in parallel: about 12-15 minutes the first time.

# %%
import contextlib, logging                                  # to silence the progress messages of elephant
from elephant.gpfa import gpfa_core                         # the GPFA computation functions (02_new, §10)
import elephant.utils
elephant.utils.logger.setLevel(logging.ERROR)

Zc = StandardScaler().fit_transform(s.spikes[:, ACTIVE].astype(float))  # z-scored raw counts, 160 channels (03, §2)
Y_SQRT = np.sqrt(s.spikes[:, ACTIVE].astype(float))         # the square root of the counts: the GPFA input (02_new, §10)

def make_seqs(segs):                                        # the data in elephant's format: one segment (start, end) per row
    return np.array([(b_ - a_, Y_SQRT[a_:b_].T) for a_, b_ in segs], dtype=[("T", int), ("y", "O")])

def fit_gpfa_fold(k):                                       # 25-factor GPFA on the 1 s windows of the training trials of block k (03, §7.2)
    train_tr = np.unique(trial[TRAIN_BINS[k]])
    np.random.seed(0)                                       # fixed starting point
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        params, _ = gpfa_core.fit(make_seqs([(w["start"], w["stop"]) for w in windows if w["trial"] in train_tr]),
                                  x_dim=K_PCA, bin_width=20.0, em_max_iters=500, tau_init=100.0)
    return params

SEG, KEEP = 100, 50                                         # 2 s segments (100 bins), every 1 s (50 bins)

def gpfa_continuous(params):                                # the 25 ordered factors of each bin of the session, in overlapping segments within each trial
    segs, owner = [], []
    for t in TRIALS:
        a0, a1 = first_last[t][0], first_last[t][1] + 1     # start and end of the trial
        starts = list(range(a0, a1 - SEG + 1, KEEP))
        if starts[-1] != a1 - SEG:
            starts.append(a1 - SEG)                         # the last segment ends exactly at the end of the trial
        centers = np.array(starts) + SEG / 2
        for x in range(a0, a1):                             # each bin comes from the segment in which it is most central
            owner.append((x, len(segs) + int(np.argmin(np.abs(centers - x)))))
        segs += [(a_, a_ + SEG) for a_ in starts]
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        lat, _ = gpfa_core.exact_inference_with_ll(make_seqs(segs), params, get_ll=False)
        _, orth = gpfa_core.orthonormalize(params, lat)     # factors rotated and ordered by importance
    out = np.empty((n_bins, K_PCA))
    for x, i in owner:
        out[x] = orth["latent_variable_orth"][i][:, x - segs[i][0]]
    return out

GP_PARAMS = Parallel(n_jobs=5)(delayed(fit_gpfa_fold)(k) for k in range(5))  # the 5 GPFAs in parallel
X_GP = [gpfa_continuous(p_) for p_ in GP_PARAMS]            # for each block, the factors of the whole session (12233 × 25)
print(f"GPFA factors: {X_GP[0].shape}; bins with no value: {int(np.isnan(np.stack(X_GP)).sum())}")

TRAIN_MOV = [tb[MOV[tb]] for tb in TRAIN_BINS]              # only the movement bins of the training trials
VAR_B = {"160 canali smussati (nessuna PCA)": dict(cfg=CEBRA_T, X_full=[Z] * 5),
         "160 canali, conteggi grezzi": dict(cfg=CEBRA_T, X_full=[Zc] * 5),
         "25 fattori GPFA": dict(cfg=CEBRA_T, X_full=X_GP),
         "solo bin di movimento (25 PC)": dict(cfg=CEBRA_T, train_bins=TRAIN_MOV)}
ACC_B, MAPS_B = run_variants(VAR_B)

REF_B = {"25 PC (principale)": INP_PC,                      # the same inputs for the classifier without a map and for UMAP
         "160 canali smussati (nessuna PCA)": fold_inp(lambda k: Z),
         "160 canali, conteggi grezzi": fold_inp(lambda k: Zc),
         "25 fattori GPFA": fold_inp(lambda k: X_GP[k])}
rows_b = {}
for name, inp in REF_B.items():
    cebra_v = ACC_A[MAIN] if name.startswith("25 PC") else ACC_B[name]
    rows_b[name] = {"CEBRA-Time, 3 dimensioni (3 semi)": mean_range(cebra_v),
                    "UMAP, d = 2 (3 semi)": mean_range([decode_map(inp, emb_umap(2, sd)) for sd in SEEDS3]),
                    "nessuna mappa (tutte le dimensioni)": f"{decode_map(inp, emb_none):.1f}"}
rows_b["solo bin di movimento (25 PC)"] = {"CEBRA-Time, 3 dimensioni (3 semi)": mean_range(ACC_B["solo bin di movimento (25 PC)"])}
display(pd.DataFrame(rows_b).T.fillna("same as the 25 PC row").rename_axis("input"))
display(sens_table({MAIN: ACC_A[MAIN], **ACC_B}, MAIN))

# %% [markdown]
# **Reading.**
#
# - **160 channels without PCA** ("everything in input"): 34.7% (−1.3, within the noise). The 25-PC PCA does not
#   remove useful information from CEBRA; UMAP and the classifier also drop a little (38.0% and
#   40.7%).
# - **Raw counts**: CEBRA reaches 34.0% (−2.1), almost as with the smoothed data, while UMAP
#   (15.4%) and the classifier (16.8%) drop to chance level. This is the real strength of
#   CEBRA on these data: its window and the time game do by themselves the smoothing that UMAP
#   has to be given ready-made, as in the synthetic sheet of §2.4. Note: the classifier on the raw
#   counts looks at a single bin; a like-for-like ceiling with the window was not computed.
# - **GPFA factors**: 43.1% (+7.0; from +4.5 to +8.6), the best input for CEBRA too. But on the same
#   input UMAP reaches 51.0% and the classifier without a map 50.7%: CEBRA stays 8 points below
#   what the GPFA factors already contain. The GPFA computed on the whole session recovers the numbers
#   of 03 (UMAP 50.5% and ceiling 50.8% on the 872 window points), now on all 1106 points.
# - **Movement bins only**: 29.4% (−6.7; from −5.0 to −7.7). The pauses help. Removing them, the movie
#   has a jump at every phase change, and the training bins are fewer than half (about 4400 instead of
#   about 9800): we did not separate the two causes.
#
# ### 7c · The CEBRA-Behavior labels
#
# §6.5 showed that the hand and wrist velocities can be recovered from the activity, not the positions.
# We therefore try three labels different from "position + velocity" (§6), each with its own linear
# comparison term. Same parameters as §6 (1000 steps), 3 dimensions, 3 seeds.
#
# | labels | numbers per bin | linear comparison |
# |---|---|---|
# | velocity only | 7 | PLS with 2 components |
# | hand and wrist velocity only (rx, g1, g2, g3) | 4 | PLS with 2 components |
# | task phases | 1 (one class per bin: the 8 movement phases, plus a "pause" class for `Presentation` and `Intertrial`) | LDA with 2 components |
#
# **With the phases as label the test becomes almost circular**: the network learns precisely to separate
# the phases, and phase decoding on the test trials only says how well this separation
# holds for unseen trials. It is the most favorable case possible, and its right comparison is LDA
# (*linear discriminant analysis*), the linear projection that best separates the classes.
#
# 3 variants × 15 trainings: about 5 minutes.

# %%
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis  # the linear projection that best separates the classes

HAND = [10, 11, 12, 13]                                     # within the 14 labels: the velocities of rx, g1, g2, g3
PH_LAB = np.where(MOV, phase, "pause")                      # the 8 movement phases; Presentation, Intertrial (and the only bin with no phase) become "pause"
CLASSES = sorted(set(PH_LAB))                               # 9 classes, present in the training trials of every block
PH_INT = np.array([CLASSES.index(p) for p in PH_LAB], dtype=np.int64)  # the class of each bin as an integer: the discrete label for CEBRA
VAR_C = {"solo velocità (7)": dict(cfg=CEBRA_B, labels=[YZ_F[k][:, VEL] for k in range(5)]),
         "solo velocità di mano e polso (4)": dict(cfg=CEBRA_B, labels=[YZ_F[k][:, HAND] for k in range(5)]),
         "fasi del task (classi)": dict(cfg=CEBRA_B, labels=[PH_INT] * 5)}
ACC_C, MAPS_C = run_variants(VAR_C, fn=cebra_fit_beh)

lin_c = {"solo velocità (7)": [decode_map(fold_inp(lambda k: PLSRegression(n_components=2).fit(
             X25[k][TRAIN_BINS[k]], YZ_F[k][TRAIN_BINS[k]][:, VEL]).transform(X25[k])), emb_none)],
         "solo velocità di mano e polso (4)": [decode_map(fold_inp(lambda k: PLSRegression(n_components=2).fit(
             X25[k][TRAIN_BINS[k]], YZ_F[k][TRAIN_BINS[k]][:, HAND]).transform(X25[k])), emb_none)],
         "fasi del task (classi)": [decode_map(fold_inp(lambda k: LinearDiscriminantAnalysis(n_components=2).fit(
             X25[k][TRAIN_BINS[k]], PH_INT[TRAIN_BINS[k]]).transform(X25[k])), emb_none)]}
rows_c = {"position + velocity (14), §6": {"CEBRA-Behavior, 3 dimensions (3 seeds)": mean_range(ACC["CEBRA-Behavior, 3 dimensioni"][:3]),
                                            "linear comparison, 2 components": f"{ACC['PLS, 2 componenti']:.1f} (PLS)"}}
for name in VAR_C:
    rows_c[name] = {"CEBRA-Behavior, 3 dimensions (3 seeds)": mean_range(ACC_C[name]),
                    "linear comparison, 2 components": f"{lin_c[name][0]:.1f} ({'LDA' if 'phases' in name else 'PLS'})"}
rows_c["no labels: CEBRA-Time (main, §7a)"] = {"CEBRA-Behavior, 3 dimensions (3 seeds)": mean_range(ACC_A[MAIN]),
                                                            "linear comparison, 2 components": f"{ACC['PCA, d = 2']:.1f} (PCA)"}
display(pd.DataFrame(rows_c).T.rename_axis("labels"))

# %%
ALL_SENS = [("7a · model", n_, v_) for n_, v_ in ACC_A.items() if n_ != MAIN] + \
           [("7b · input", n_, v_) for n_, v_ in ACC_B.items()] + \
           [("7c · labels (CEBRA-Behavior)", n_, v_) for n_, v_ in ACC_C.items()] + \
           [("7c · labels (CEBRA-Behavior)", "position + velocity (§6)", ACC["CEBRA-Behavior, 3 dimensioni"][:3])]
ref = np.array(ACC_A[MAIN])
fig, ax = plt.subplots(figsize=(10, 8.5))
for yi, (grp, name, v) in enumerate(ALL_SENS[::-1]):
    v = np.array(v)
    col = {"7a · model": "C0", "7b · input": "C2", "7c · labels (CEBRA-Behavior)": "C3"}[grp]
    ax.plot([v.min(), v.max()], [yi, yi], color=col, lw=2, alpha=0.5)  # min-max across the 3 seeds
    ax.scatter(v.mean(), yi, color=col, s=50, zorder=3)
ax.axvspan(ref.min(), ref.max(), color="0.85", zorder=0, label="main: min-max of the 3 seeds")
ax.axvline(ref.mean(), color="k", lw=1.2, label=f"main: mean ({ref.mean():.1f}%)")
ax.axvline(np.mean(ACC["UMAP, d = 2"]), color="C2", ls="--", lw=1,
           label="UMAP, d = 2 on the 25 PC (§4)")
ax.axvline(ACC["l'orologio da solo (1 numero)"], color="k", ls=":", lw=1, label="the clock alone (§5.3)")
ax.axvline(12.5, color="0.5", ls="--", lw=0.8, label="chance level")
ax.set(yticks=range(len(ALL_SENS)), yticklabels=[f"{n_}" for _, n_, _ in ALL_SENS[::-1]], xlim=(10, 70),
       xlabel="phase decoded (%), 3 seeds (dot = mean, bar = min-max)",
       title="Sensitivity analysis: one choice at a time (blue: model; green: input; red: labels)")
ax.legend(fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(FIGS / "04_sensitivity.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# **Reading.**
#
# - **None of the labels tried does better than CEBRA-Time without labels** (36.1%):
#   CEBRA-Behavior with position + velocity 25.3%, with velocities only 31.5%, with hand and
#   wrist velocities 30.2%, with the phases 32.7%.
# - **Removing the positions gains 6 points** (from 25.3% to 31.5%), as expected from
#   §6.5: for the phases the positions are noise. Keeping only hand and wrist adds nothing (30.2%).
#   Velocities alone are not enough to distinguish the phases: several phases have similar velocities (for
#   example those in which the hand is still).
# - **With the phases as label, CEBRA-Behavior reaches only 32.7%**: less than without labels,
#   even though the label is exactly the one to decode. LDA, with the same labels and a
#   simple linear projection, reaches **43.2%**: 10 points more, and more than UMAP with 2 coordinates
#   (41.2%). A probable explanation, not verified: with the classes CEBRA brings together all instants
#   of the same phase in the training trials and also learns what is specific to those trials
#   (the overfitting of §3 and §6.2), while LDA, with few parameters, generalizes better. The 1000
#   steps, moreover, had been chosen with the continuous labels: for the classes we did not
#   recheck them.
# - PLS with the velocities (25.4% and 26.0%) is worth as much as PCA with 2 components (26.0%).
#
# **In the figure** the three families of tests at a glance: among the model choices only the
# dimensions (and the Euclidean distance, which is a way of having more free coordinates) really shift
# the result; among the inputs, the GPFA factors up and movement only down; all the labels
# down. No variant beats the clock (48.5%).

# %% [markdown]
# ## 8 · CEBRA summary
#
# ### The choices
#
# | step | choice | why | alternatives (where) |
# |---|---|---|---|
# | data and evaluation | the same as 03: 1106 points (movement phases, one every 100 ms), 5 blocks of 3 trials, 10-neighbor classifier, balanced accuracy | compare CEBRA with everything else with the same measure; the numbers of 03 are reproduced to the decimal | — (§1.2) |
# | input | 25 PCs of the smoothed firing rates (σ = 50 ms), PCA estimated on the training trials only | the same input as PCA, Isomap and UMAP in 03 | 160 channels, raw counts, GPFA factors (§7b) |
# | training bins | all bins of the training trials, pauses included | CEBRA-Time needs a continuous "movie" | movement bins only (§7b) |
# | joins between non-consecutive trials | accepted | they affect at most 0.3% of the pairs | — (§1.5) |
# | network | `offset10-model`: centered 200 ms window | paper (hippocampus, S1, Allen 2P) | no window, 100 ms, 720 ms (§7a) |
# | Δ (positive pairs) | 10 bins, 200 ms | paper | 20 ms, 1 s (§5.3) |
# | temperature | τ = 1 | default value; the same for CEBRA-Time and CEBRA-Behavior | 0.1, 2.25 (paper, CEBRA-Time), learned (§7a) |
# | steps | 1000 | validation on two pairs of trials (12-15 and 6-7), rule fixed before looking | 500-20 000 (§3.2, §6.2, §7a) |
# | dimensions | 3 and 4 on the sphere, i.e. 2 and 3 free coordinates | comparison at equal free coordinates with 03 | 8 and 16 (§7a) |
# | distance | cosine | paper | Euclidean (§2.4, §7a) |
# | seed | fixed in the library (`HasGenerator`) | CEBRA 0.6.0 ignores the seed it is given | — (§2.7) |
# | repetitions | 5 seeds (3 in the sensitivity analyses) | the maps change a lot from one seed to another | — (§5.2) |
# | CEBRA-Behavior labels | position + velocity, 14 z-scored numbers (user's choice) | the variable that a BCI must recover | velocity only, hand and wrist, phases (§7c) |
# | linear comparison with the labels | PLS (kinematics), LDA (phases) | same input and same labels as CEBRA-Behavior | — (§6.3, §7c) |
#
# ### Alternatives tried in this notebook
#
# | what | values tried | effect | where |
# |---|---|---|---|
# | hand-written loss function | InfoNCE with the CEBRA formula and with the original one, against the library | identical to the library (difference 5·10⁻⁷); the original formula changes by 0.004 | §2.5 |
# | sphere, on the synthetic sheet | cosine at 3 and 4 dimensions, Euclidean at 3 | at 3 dimensions the sphere closes the sheet back on itself (R² of the length −0.21); with 3 free coordinates 0.58-0.61; moving average + PCA 0.93 | §2.4 |
# | seed | with and without the fix | without: different maps at every run; with: bit-for-bit identical | §2.7 |
# | PCA in the validation | seen or not seen by the validation trials | 40-42% against 28-29% of the validation-trial variance: the shortcut was there, fixed | §3.2 |
# | steps (validation) | 1-20 000, two pairs of trials, two seeds | minimum at 500-1000 steps, then overfitting | §3.2, §6.2 |
# | classifier on the sphere | straight-line distance against cosine | identical predictions | §4.1 |
# | like-for-like ceiling | 200 ms window in a row (250 numbers), mean over the window (25) | 43.9% and 44.9%, against 42.2% without a window | §4.1 |
# | consistency by task version | odd and even separately | higher for all methods; the order does not change | §4.2 |
# | random-phase twin | 5 twins × 5 blocks | CEBRA at chance (12.5%) | §5.1 |
# | consistency across seeds | R² after linear alignment | CEBRA 63-75%, UMAP 87-93% | §5.2 |
# | time chain | composition of the neighbors; Δ = 1, 10, 50 | chain 8% (input 20%); Δ = 10 the best | §5.3 |
# | clock | time from the start of the trial, alone and inside the maps | alone 48.5%; the maps do not contain it; it decodes the arm phases, the maps those of the hand | §5.3 |
# | CEBRA-Behavior, controls | labels shuffled in time, swapped between trials | 15.1% and 29.8% (true: 25.6%) | §6.4 |
# | kinematics | repeatability of the positions; kinematics recovered from the activity | positions differ from trial to trial; only the hand and wrist velocities can be recovered | §6.5 |
# | dimensions | 8, 16 | +7.8 and +6.5 points; at equal free coordinates CEBRA = UMAP | §7a |
# | temperature, window, steps | τ 0.1, 2.25, learned; 1, 5, 36 bins; 500, 2000, 5000 steps | within the noise, except 5000 steps (−2.0) and 36 bins (+2.5) | §7a |
# | Euclidean distance | `offset10-model-mse` | +4.3 (3 free coordinates instead of 2) | §7a |
# | input | 160 channels, raw counts, GPFA factors, movement only | −1.3, −2.1 (UMAP at chance), +7.0 (UMAP 51.0%), −6.7 | §7b |
# | CEBRA-Behavior labels | velocity only, hand and wrist, phases | 31.5%, 30.2%, 32.7%: all below CEBRA-Time; LDA with the phases 43.2% | §7c |
#
# ### The final table: all methods, the same measure
#
# Built by the code with the numbers computed in this notebook. For the GPFA we use its first
# factors (ordered by importance, like the PCA components), estimated per block on the training trials only
# and computed on the whole session (§7b); the measures of its trajectories come from 03
# (§6.1) and from 02_new (§10.5). Where there is an interval: mean (min-max) across the seeds.

# %%
GP_ACC = {d: decode_map(fold_inp(lambda k, d=d: X_GP[k][:, :d]), emb_none) for d in (2, 3, 7)}  # the GPFA with its first d factors
LDA3 = decode_map(fold_inp(lambda k: LinearDiscriminantAnalysis(n_components=3).fit(
    X25[k][TRAIN_BINS[k]], PH_INT[TRAIN_BINS[k]]).transform(X25[k])), emb_none)  # the LDA with 3 components

def row(name, lab, a2=None, a3=None, a7=None):              # one row of the final table
    return {"method": name, "labels": lab, "2 free coordinates": fmt(a2) if a2 is not None else "",
            "3 free coordinates": fmt(a3) if a3 is not None else "", "7 free coordinates": fmt(a7) if a7 is not None else ""}

final_dec = pd.DataFrame([
    row("PCA", "no", ACC["PCA, d = 2"], ACC["PCA, d = 3"], HI_ACC[("PCA", 7)]),
    row("Isomap", "no", ACC["Isomap, d = 2"], ACC["Isomap, d = 3"]),
    row("UMAP", "no", ACC["UMAP, d = 2"], ACC["UMAP, d = 3"], HI_ACC[("UMAP", 7)]),
    row("GPFA (primi fattori)", "no", GP_ACC[2], GP_ACC[3], GP_ACC[7]),
    row("CEBRA-Time (3, 4 e 8 dimensioni)", "no", ACC["CEBRA-Time, 3 dimensioni"], ACC["CEBRA-Time, 4 dimensioni"], HI_ACC[("CEBRA-Time", 7)]),
    row("CEBRA-Behavior, posizione + velocità", "sì", ACC["CEBRA-Behavior, 3 dimensioni"], ACC["CEBRA-Behavior, 4 dimensioni"]),
    row("PLS, posizione + velocità", "sì", ACC["PLS, 2 componenti"], ACC["PLS, 3 componenti"]),
    row("CEBRA-Behavior, fasi (3 semi)", "yes (the phases)", ACC_C["fasi del task (classi)"]),
    row("LDA, fasi", "yes (the phases)", lin_c["fasi del task (classi)"][0], LDA3),
]).set_index("method")
display(final_dec)

refs = pd.Series({"chance level": "12.5",
                  "no map: 25 PC of one bin (the ceiling from 03)": f"{ACC['tetto, 25 dimensioni']:.1f}",
                  "no map: 25 PC, mean over 200 ms (like-for-like ceiling)": f"{ACC['tetto alla pari: media sulla finestra (25 numeri)']:.1f}",
                  "no map: 25 GPFA factors": rows_b["25 fattori GPFA"]["nessuna mappa (tutte le dimensioni)"],
                  "UMAP, d = 2, on the GPFA factors (3 seeds)": rows_b["25 fattori GPFA"]["UMAP, d = 2 (3 semi)"],
                  "CEBRA-Time, 3 dimensions, on the GPFA factors (3 seeds)": mean_range(ACC_B["25 fattori GPFA"]),
                  "the clock alone (no neural data)": f"{acc_clock:.1f}"},
                 name="phase decoded (%)")
display(refs.to_frame())

def tstat(name, col):                                       # one trajectory measure (§4.2): mean (min-max) across seeds
    v = np.array([r_[col] for r_ in TRAJ[name]])
    return f"{v.mean():.3f}" if len(v) == 1 else f"{v.mean():.3f} ({v.min():.3f}-{v.max():.3f})"

final_traj = pd.DataFrame({
    name: {"seghettatura": tstat(name, "seghettatura"), "consistency": tstat(name, "consistenza, tutte le ripetizioni"),
           "consistency by version": tstat(name, "consistenza per versione (media delle due)"),
           "consistency across seeds (R², %)": c_}
    for name, c_ in (("PCA", "100 (by construction)"), ("Isomap", "100 (by construction)"),
                     ("UMAP", mean_range(cons["UMAP, d = 3"] * 100) + " (d = 3)"),
                     ("CEBRA-Time, 3 dimensioni", mean_range(cons["CEBRA-Time, 3 dimensioni (stimato su tutti i trial)"] * 100)),
                     ("CEBRA-Time, 4 dimensioni", mean_range(cons["CEBRA-Time, 4 dimensioni (stimato su tutti i trial)"] * 100)))}).T
final_traj.loc["GPFA (from 03 and 02_new)"] = ["0.011", "0.11", "not computed", "not computed"]
display(final_traj)

# %%
METH = ["PCA", "Isomap", "UMAP", "GPFA (primi fattori)", "CEBRA-Time (3, 4 e 8 dimensioni)",
        "CEBRA-Behavior, posizione + velocità", "PLS, posizione + velocità"]
VALS = {"PCA": [ACC["PCA, d = 2"], ACC["PCA, d = 3"]], "Isomap": [ACC["Isomap, d = 2"], ACC["Isomap, d = 3"]],
        "UMAP": [ACC["UMAP, d = 2"], ACC["UMAP, d = 3"]], "GPFA (primi fattori)": [GP_ACC[2], GP_ACC[3]],
        "CEBRA-Time (3, 4 e 8 dimensioni)": [ACC["CEBRA-Time, 3 dimensioni"], ACC["CEBRA-Time, 4 dimensioni"]],
        "CEBRA-Behavior, posizione + velocità": [ACC["CEBRA-Behavior, 3 dimensioni"], ACC["CEBRA-Behavior, 4 dimensioni"]],
        "PLS, posizione + velocità": [ACC["PLS, 2 componenti"], ACC["PLS, 3 componenti"]]}
COLS8 = {"PCA": "C1", "Isomap": "C0", "UMAP": "C2", "GPFA (primi fattori)": "C4", "CEBRA-Time (3, 4 e 8 dimensioni)": "C3",
         "CEBRA-Behavior, posizione + velocità": "#f4a6a6", "PLS, posizione + velocità": "0.6"}
fig, ax = plt.subplots(figsize=(12, 4.8))
w = 0.11
for i, m in enumerate(METH):
    for j, v in enumerate(VALS[m]):                         # j = 0: 2 free coordinates; j = 1: 3 free coordinates
        v = np.atleast_1d(v)
        x = j + (i - (len(METH) - 1) / 2) * w
        ax.bar(x, v.mean(), width=w * 0.92, color=COLS8[m], label=(m.replace(" (3, 4 e 8 dimensioni)", " (3 and 4 dimensions)")
                                                                  if j == 0 else None))
        if len(v) > 1:
            ax.plot([x, x], [v.min(), v.max()], color="k", lw=1)  # min-max across seeds
for yv, lab, ls in ((ACC["tetto, 25 dimensioni"], "no map, 25 PC (42.2%)", "--"),
                    (ACC["tetto alla pari: media sulla finestra (25 numeri)"], "like-for-like ceiling, 200 ms (44.9%)", "-."),
                    (ACC["l'orologio da solo (1 numero)"], "the clock alone (48.5%)", ":"), (12.5, "chance level", "-")):
    ax.axhline(yv, color="0.3", ls=ls, lw=1, label=lab)
ax.set(xticks=[0, 1], xticklabels=["2 free coordinates", "3 free coordinates"], ylim=(0, 55),
       ylabel="phase decoded (%)", title="Phase decoding on unseen trials: all methods")
ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
fig.tight_layout(); fig.savefig(FIGS / "04_summary.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# **Reading.**
#
# - **At equal free coordinates, among the methods without labels** UMAP is the best with 2-3
#   coordinates (41.2% and 42.6%), CEBRA-Time second (37.7% and 40.4%), then Isomap (31.8% and 36.0%), the
#   first GPFA factors (31.1% and 33.8%) and PCA (26.0% and 32.4%). With 7 coordinates the GPFA rises
#   to 45.3%, above UMAP (43.3%) and CEBRA (43.9%): the GPFA factors spread the information
#   about the phases over more dimensions. PCA, Isomap, UMAP and CEBRA receive the 25 PCs; the GPFA starts from the
#   square root of the counts.
# - **With the labels**: with the kinematics, CEBRA-Behavior and PLS (25.6-28.7%) are below all the
#   methods without labels, except PCA with 2 components. With the phases **LDA** is the best in the
#   table: 43.2% with 2 components and 45.8% with 3, at the level of the like-for-like ceiling (44.9%);
#   CEBRA-Behavior, with the same labels, 32.7%.
# - **The comparison terms**: on the 25 PCs no method without labels exceeds the like-for-like ceiling
#   (44.9%). The best input remains that of the GPFA factors (classifier 50.7%, UMAP 51.0%).
#   The clock, with no neural data at all, stands at 48.5%, above every method on the 25 PCs.
# - **Trajectories and stability**: PCA and GPFA are the smoothest (0.01), CEBRA is in the middle (0.07-0.08),
#   Isomap and UMAP jump (0.3-0.6). The consistency across repetitions is highest for UMAP (0.125; 0.144
#   by version); CEBRA with 4 dimensions, GPFA and Isomap lie between 0.095 and 0.11, PCA and CEBRA with 3
#   dimensions around 0.09. By version the 02_new GPFA reaches 0.137, close to UMAP. CEBRA is the least stable from one seed to another (63-75%, against 87-93%
#   for UMAP).
# - **In one sentence**: on this session CEBRA-Time is a good method without labels, between Isomap and
#   UMAP with few dimensions and on par with UMAP with more dimensions, with smoother trajectories than UMAP but
#   less stable maps; CEBRA-Behavior, with the labels tried, does not beat its linear
#   comparison terms.
#
# ### What we learned about CEBRA
#
# 1. **With 2-3 free coordinates CEBRA-Time lies between Isomap and UMAP**: 37.7% and 40.4% against 41.2% and
#    42.6% for UMAP (§4). The disadvantage comes from the sphere and the few dimensions: with 7 free
#    coordinates CEBRA-Time is worth UMAP (43.9% against 43.3%) and reaches the ceiling of the 25 dimensions (§7a).
#    It adds no information: it does not even use the extra information in its 200 ms window (like-for-like
#    ceiling 44.9%, §4.1; without a window almost the same result, §7a).
# 2. **It really learns the task-related activity**: on the random-phase twin it drops to chance (§5.1),
#    among the neighbors in its map the time chain weighs less than in the data (8% against 20%), and the
#    map does not contain the clock (§5.3). Like the other methods, it mainly decodes the phases of the
#    hand.
# 3. **With little data it overfits immediately**: 1000 steps, against the paper's 4000-10 000, and
#    training longer makes things worse even on the test trials (§3.2, §6.2, §7a). Validation must be done on
#    separate trials and with an input that has not seen them: even the 25-PC PCA adapts to the trials on which it
#    is estimated (§3.2).
# 4. **It is unstable**: from one seed to another the accuracy changes by up to 5 points and the maps
#    resemble each other much less than those of UMAP (R² 63-75% against 87-93%, §5.2). A CEBRA map cannot
#    be interpreted on its own.
# 5. **The trajectories are smooth, but not as smooth as those of the linear methods** (roughness 0.07-0.08
#    against 0.01 for PCA and GPFA, 0.6 for UMAP); the consistency is like that of PCA and GPFA, below UMAP
#    (§4.2).
# 6. **Its strength on these data is the raw counts**: without smoothing CEBRA loses 2
#    points, UMAP collapses to chance (§7b). The weak point: on the best input, the GPFA factors, it stays 8
#    points below the simple classifier on those factors (43.1% against 50.7%).
# 7. **CEBRA-Behavior depends entirely on the labels, and here none helps.** With position + velocity
#    it gets worse (25.6%): the reference positions change from one trial to the next and the activity does not
#    encode them (§6.5). With velocities only, with hand and wrist, or with the phases it stays below CEBRA-Time; with
#    the phases the linear LDA does 10 points better (§7c). The right comparison for CEBRA-Behavior is the
#    linear method with the same labels, and on this session CEBRA-Behavior does not beat it.
# 8. **Phase decoding has a very high "no neurons" comparison term**: knowing
#    how long ago the trial started is enough for 48.5% (§5.3), more than any map. The maps
#    decode the phases of the hand, the clock those of the arm.
#
# ### What we did not verify
#
# - **A single session**, and parameters changed one at a time, never in combination (for example
#   GPFA factors + 8 dimensions).
# - The paper's parameters were read in the accessible parts of the full text: learning
#   rate and optimizer do not appear in it for these data.
# - Exact reproducibility holds on this computer, with these versions (cebra 0.6.0, torch
#   2.7.1, Mac GPU); on other machines the saved models are authoritative.
# - Why the CEBRA maps change so much between seeds (§5.2), why CEBRA-Behavior with the phases
#   generalizes badly (§7c; the 1000 steps were not rechecked for the classes), why the
#   labels swapped between trials give a better result than the true ones (§6.4; the positives were not
#   counted), why the positions change from one trial to the next (§6.5).
# - The twin is Gaussian (§5.1); the like-for-like ceiling on the raw counts was not computed
#   (§7b); the FALCON benchmark (velocity decoding) was not redone.
#
# > **Rule for the thesis.** A CEBRA result should be reported with: the version (Time or Behavior, and
# > with which labels), the number of dimensions **and of free coordinates**, how the
# > number of steps was chosen, the seeds (mean and min-max) and the consistency across seeds, the comparison at equal
# > free coordinates (PCA, UMAP) and, for CEBRA-Behavior, the linear method with the same
# > labels. For phase decoding, the clock as well. For example: *"CEBRA-Time
# > (offset10-model, Δ = 200 ms, τ = 1, 1000 steps chosen on validation trials, cosine distance)
# > on 25 PCs of the smoothed firing rates (σ = 50 ms): phase decoding on unseen trials 37.7% with 3
# > dimensions, i.e. 2 free coordinates (35.3-40.6 over 5 seeds), against 41.2% for UMAP with d = 2;
# > consistency across seeds R² = 63%; on the random-phase twin 12.5%; the clock alone 48.5%"*.
#
# ### References
#
# - Schneider, S., Lee, J. H., Mathis, M. W. (2023). Learnable latent embeddings for joint
#   behavioural and neural analysis. *Nature*, 617, 360–368. (Full text: PubMed Central,
#   PMC10172131.)
# - van den Oord, A., Li, Y., Vinyals, O. (2018). Representation learning with contrastive
#   predictive coding. *arXiv*:1807.03748.
# - Hyvärinen, A., Morioka, H. (2016). Unsupervised feature extraction by time-contrastive learning
#   and nonlinear ICA. *Advances in Neural Information Processing Systems*, 29.
# - Wang, T., Isola, P. (2020). Understanding contrastive representation learning through alignment
#   and uniformity on the hypersphere. *Proceedings of the 37th International Conference on Machine
#   Learning* (PMLR 119).
# - McInnes, L., Healy, J., Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection
#   for dimension reduction. *arXiv*:1802.03426.
# - Tenenbaum, J. B., de Silva, V., Langford, J. C. (2000). A global geometric framework for
#   nonlinear dimensionality reduction. *Science*, 290(5500), 2319–2323.
# - Yu, B. M., et al. (2009). Gaussian-process factor analysis for low-dimensional single-trial
#   analysis of neural population activity. *Journal of Neurophysiology*, 102, 614–635.
# - Theiler, J., Eubank, S., Longtin, A., Galdrikian, B., Farmer, J. D. (1992). Testing for
#   nonlinearity in time series: the method of surrogate data. *Physica D*, 58, 77–94.
# - Prichard, D., Theiler, J. (1994). Generating surrogate data for time series with several
#   simultaneously measured variables. *Physical Review Letters*, 73(7), 951–954.
# - Wold, S., Sjöström, M., Eriksson, L. (2001). PLS-regression: a basic tool of chemometrics.
#   *Chemometrics and Intelligent Laboratory Systems*, 58, 109–130.
# - Fisher, R. A. (1936). The use of multiple measurements in taxonomic problems. *Annals of
#   Eugenics*, 7, 179–188.

# %% [markdown]
# ---
# # Part B · LFADS
#
# **LFADS in a sentence.** *Latent Factor Analysis via Dynamical Systems* (Pandarinath et al. 2018) is
# a model that, like the GPFA (02_new, §10), starts from the **counts** of each trial and returns
# **smooth trajectories** of a few factors. The difference: the GPFA only knows that the factors "do not make
# jumps" (a generic rule, the same for all moments), LFADS **learns the rules of motion**: from
# where the state is now, where it will go next. It is the difference between knowing that a car does not
# teleport and knowing the road it is driving on. In the review it is the dynamical model of §2.6
# (equation 3) made concrete with a recurrent network (§4.2); in §5.5 it is cited for decoding in
# brain-computer interfaces.
#
# LFADS is the **"nonlinear GPFA"**: the rules of motion are a neural network (nonlinear), and the
# link between factors and firing rates goes through an exponential. Likewise, a network that
# looks at one bin at a time, without rules of motion, would be the "nonlinear FA": we try it in §14.
#
# **Unlike CEBRA, LFADS is a true generative model.** It has both halves: from data to
# factors (the *encoder*) and from factors to data (the *generator* with its output), and it assigns a
# probability to every possible sequence of counts. For this reason it can produce fake trials (§11) and it
# can be judged on how well it **predicts unseen data** (§12), which makes no sense for CEBRA.
#
# **The question of part B.** On these data, do the nonlinearity and the learned dynamics improve the
# single-trial trajectories relative to the GPFA? The main comparison is **LFADS against GPFA on
# equal terms**: same counts, same recording segments, same number of factors (25).
#
# **The choices agreed before starting:**
#
# | choice | decision | alternative (where) |
# |---|---|---|
# | code | a small hand-written LFADS to understand (§10), the `lfads-torch` library (Sedler & Pandarinath 2023) for the real trainings, with a training loop written here | the library's Lightning `Trainer` (requires a fix, §10) |
# | data | the reference session only, as in all the notebooks | the other two sessions of the same day (§14) |
# | cutting | 600 ms segments overlapping by 200 ms, within each trial, stitched with graded weights (§9.2) | 400 ms and 1 s (§14) |
# | external inputs (*controller*) | no | yes (§14) |
#
# **Two limits to declare right away.**
#
# - **Little data.** A recurrent network has many parameters, and the risk is that it memorizes
#   instead of understanding. With spikes there is a particular form of this risk: the network can learn to
#   **copy** the input, and the usual validation does not notice (Keshtkaran & Pandarinath
#   2019). With fixed parameters the performance of LFADS gets much worse as the number of trials decreases
#   (same paper; Keshtkaran et al. 2022). The defenses: a small model, *coordinated
#   dropout* (§10), validation on unseen trials.
# - **No AutoLFADS.** AutoLFADS (Keshtkaran et al. 2022) chooses the parameters by itself, training
#   dozens of models in parallel that exchange the best parameters (40 for the
#   maze data). On a laptop this is not feasible: **we choose the parameters by hand, with a small
#   grid evaluated on the validation trials** (§11).
#
# The pipeline:
#
# ```
# spike_times ──(20 ms bins, 160 active channels, NO smoothing)──► counts   12233 × 160
#
# counts ──(§9: 600 ms segments within each trial, overlapping by 200 ms)──► sequences   30 bins × 160
#
# training-trial sequences ──(LFADS)──► trained model
#
# each trial ──(already trained model, without retraining)──► factors of each segment ──(stitching)──► 25 factors per bin
#
# evaluation: common yardstick (1106 points, 5 blocks), trajectories, hidden channels
# ```
#
# The steps: §9 the data · §10 the idea on a synthetic example · §11 a first training · §12 common yardstick
# and comparison with the GPFA · §13 controls · §14 sensitivity and alternatives · §15 summary and final
# table. **Current status: complete, §9-§15.**
#
# **How to run the notebook.** On the evening of 27 October, when rerunning the whole notebook, the kernel
# shut down by itself three times in a row during §7c of part A, always at the same point: corrupted
# memory inside numpy, with two copies of OpenMP loaded and dozens of threads calling OpenBLAS (which
# reported it with a warning: *precompiled NUM_THREADS exceeded*). Launching Jupyter with
# `OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1` the run completes, with the same results: only
# the last binary digits of some computations change. For this reason 15 CEBRA models of §7b, whose input is the
# GPFA factors, were retrained, with the same accuracies. Why the problem appeared only
# that evening we did not understand.

# %% [markdown]
# ## 9 · The data for LFADS
#
# Four decisions, before any network:
#
# - **9.1** what LFADS receives: the **raw counts**;
# - **9.2** how the recording is cut into **fixed-length segments**, and how they are stitched back together;
# - **9.3** which trials are used to train, which to **validate** and which for the **test**;
# - **9.4** which channels to **hide** for the measure specific to generative models (§12);
#
# plus two caveats about the task (§9.5): the `SnapTo` phase and the two versions of the task.
#
# ### 9.1 · The raw counts
#
# Up to here (part A, 03) the methods received the **smoothed** firing rates (σ = 50 ms). LFADS does not: it receives
# the **counts** of each 20 ms bin, small integers, and has its own noise model. It assumes
# that, given the "true" rate of a channel at an instant, the number of spikes is **random according to
# Poisson**: like the number of cars passing by your house in a minute, which fluctuates around an
# average. The task of LFADS is precisely to recover that average, instant by instant. If we smoothed
# first, we would be doing ourselves, with a hand-chosen filter, part of the work that LFADS has to do, and the data
# would no longer be counts (Poisson only makes sense for integers). It is the same choice as for FA and
# GPFA in 02_new (§9 and §10), which started from the counts.
#
# We check that they are exactly the counts already used in part A: those from which the GPFA starts
# (their square root, §7b) and those given to CEBRA in the "raw counts" alternative (§7b).

# %%
COUNTS = s.spikes[:, ACTIVE].astype(np.float32)             # the raw 20 ms counts of the 160 active channels: the LFADS input (float32, the networks' format)
N_CH = COUNTS.shape[1]                                      # number of channels (160)
print(f"Counts: {COUNTS.shape[0]} bins × {N_CH} channels; all integers: {np.array_equal(COUNTS, np.round(COUNTS))}; "
      f"maximum in a bin: {int(COUNTS.max())}")
print("Same counts as the GPFA (their square root is Y_SQRT, §7b):", np.array_equal(np.sqrt(COUNTS.astype(float)), Y_SQRT))
print("Same counts given to CEBRA in §7b (z-scored):", np.allclose(StandardScaler().fit_transform(COUNTS.astype(float)), Zc))

vals = pd.Series({"0 spike": np.mean(COUNTS == 0), "1 spike": np.mean(COUNTS == 1), "2 spike": np.mean(COUNTS == 2),
                  "3 spike": np.mean(COUNTS == 3), "4 or more": np.mean(COUNTS >= 4)}) * 100  # how often a channel sees k spikes in a bin
rate_hz = COUNTS.mean(0) / io.BIN_S                         # mean rate of each channel, in Hz
display(vals.round(2).rename("% of bins (all channels)").to_frame().T)
print(f"Mean rate per channel: median {np.median(rate_hz):.1f} Hz (from {rate_hz.min():.1f} to {rate_hz.max():.1f} Hz); "
      f"expected spikes in a bin at the median rate: {np.median(rate_hz) * io.BIN_S:.2f}")

# %% [markdown]
# **Reading.**
#
# - They are the same counts as in part A, bit for bit: those from which the GPFA starts and those given to
#   CEBRA in the alternative of §7b. LFADS and GPFA therefore start from the same numbers.
# - **In a bin a channel almost always sees 0 or 1 spikes**: 0 in 77% of cases, 1 in 18%, 2 in 4%,
#   3 or more in less than 1% (at most 8). With the median rate, about 10 Hz, in 20 ms one
#   expects 0.2 spikes. This is the realm of small integer numbers, where the Poisson model is the
#   natural one; the GPFA instead treats them with the square root and Gaussian noise, an approximation.
# - A single bin of a single channel therefore says very little: to recover the "true" rate
#   LFADS has to put together many bins and many channels, like the GPFA.

# %% [markdown]
# ### 9.2 · Cutting into segments and stitching back
#
# LFADS works on **fixed-length sequences**: for each one the encoder reads the whole sequence, derives
# an **initial state** from it, and the generator, starting from there, "unrolls" the trajectory bin by bin
# (we will see this in §10). Our trials instead last 13-19 s, all different. The options:
#
# | cutting | pros | cons |
# |---|---|---|
# | **600 ms segments, overlapping by 200 ms** (chosen) | the AutoLFADS scheme for a continuous recording (Keshtkaran et al. 2022) and of the Neural Latents Benchmark (Pei et al. 2021), which cuts the 15 continuous minutes of the MC_RTT dataset into 600 ms "trials"; without external inputs the generator has to guess by itself, from the initial state, everything that happens in the segment: the shorter it is, the less it has to guess | each segment sees little context |
# | 1 s segments | same duration as the windows of the common yardstick (§4.2) | the generator has to unroll 1 s by itself, often straddling two phases |
# | 2 s segments, keeping the central second | the scheme used for the continuous GPFA of §7b | even more to unroll without inputs; one bin out of two is thrown away |
#
# **How we cut.** Within **each trial** separately: one segment every 20 bins (400 ms), so two
# consecutive segments share 10 bins (200 ms); the last segment ends exactly with the trial,
# as for the continuous GPFA of §7b. No segment crosses the boundary between two trials: unlike
# CEBRA (§1.5) there are no "joins".
#
# **How we stitch.** After the model, each bin has a value in every segment that contains it, and the two
# values do not coincide: at the edges of a segment the model knows less than at the center. We take a **weighted
# average**: in the shared zone the segment that ends gradually hands over to the one that begins,
# like two voices in a choir taking over from each other without a break. The weights are those of AutoLFADS (Methods,
# RTT dataset): with x going from 0 to 1 along the overlap, the segment that ends weighs 1 − x², the one
# that begins x². The two curves always sum to 1. We measure x at the center of each bin (from 0.05 to
# 0.95 instead of from 0 to 1), so no weight is exactly 0. Where more than two segments overlap
# or the overlap is longer (the last segment of each trial), we divide by the sum of the weights.
#
# **The manual check.** If we give the stitching the segments of the **counts themselves**, each bin receives
# identical values from all the segments that contain it, and the weighted average must return exactly the
# original counts.

# %%
SEG_LEN, SEG_OVL = 30, 10                                   # segments of 30 bins (600 ms), overlapping by 10 bins (200 ms)
SEG_STEP = SEG_LEN - SEG_OVL                                # a new segment every 20 bins (400 ms)
TRIAL_SPAN = {t: (int(np.flatnonzero(trial == t)[0]), int(np.flatnonzero(trial == t)[-1]) + 1)  # for each trial: first bin and bin after the last
              for t in TRIALS}

def segment_starts(a0, a1):                                 # where the segments of a trial running from bin a0 to bin a1 (excluded) start
    st = list(range(a0, a1 - SEG_LEN + 1, SEG_STEP))        # one segment every 20 bins...
    if st[-1] != a1 - SEG_LEN:
        st.append(a1 - SEG_LEN)                             # ...plus a last segment that ends exactly with the trial
    return np.array(st)

SEG_START = {t: segment_starts(*TRIAL_SPAN[t]) for t in TRIALS}  # for each trial, the first bin of each of its segments

def cut_segments(X, t):                                     # the segments of trial t of a signal X (bins × columns): (segments, 30 bins, columns)
    return np.stack([X[a:a + SEG_LEN] for a in SEG_START[t]])

x_ovl = (np.arange(SEG_OVL) + 0.5) / SEG_OVL                # position along the overlap, from 0 to 1 (at the center of each bin)
RAMP_IN, RAMP_OUT = x_ovl ** 2, 1 - x_ovl ** 2              # weight of the segment that begins (rises) and of the one that ends (falls): AutoLFADS

def segment_weights(n_seg):                                 # the weights of the 30 bins of each of the n_seg segments of a trial
    W = np.ones((n_seg, SEG_LEN))                           # 1 where the segment is alone
    W[1:, :SEG_OVL] = RAMP_IN                               # all segments except the first of the trial begin with the rising ramp
    W[:-1, -SEG_OVL:] = RAMP_OUT                            # all except the last end with the falling ramp
    return W

def stitch(Y_seg, t):                                       # stitches the segments of trial t (segments × 30 bins × columns) into a continuous signal (trial bins × columns)
    a0, a1 = TRIAL_SPAN[t]
    num, den = np.zeros((a1 - a0, Y_seg.shape[2])), np.zeros((a1 - a0, 1))
    for a, y, w in zip(SEG_START[t] - a0, Y_seg, segment_weights(len(Y_seg))):
        num[a:a + SEG_LEN] += w[:, None] * y                # sum of the weighted values...
        den[a:a + SEG_LEN] += w[:, None]                    # ...and of the weights
    return num / den                                        # weighted average in each bin

SEGS = {t: cut_segments(COUNTS, t) for t in TRIALS}         # the segments of the counts, trial by trial: the LFADS sequences
back = np.concatenate([stitch(SEGS[t], t) for t in TRIALS]) # stitched, one after the other (the trials are in time order and cover the whole session)
print(f"Segments per trial: from {min(len(v) for v in SEGS.values())} to {max(len(v) for v in SEGS.values())}; "
      f"in total {sum(len(v) for v in SEGS.values())} segments of {SEG_LEN} bins")
print("Each segment lies within its trial:", all(np.all(trial[a:a + SEG_LEN] == t) for t in TRIALS for a in SEG_START[t]))
print(f"Stitching the counts themselves: maximum difference from the original counts {np.abs(back - COUNTS).max():.1e}; "
      f"identical after rounding: {np.array_equal(np.round(back), COUNTS)}")
print("Weights in the regular overlaps (two segments, 10 bins): sum to 1:", np.allclose(RAMP_IN + RAMP_OUT, 1))
cover = np.zeros(n_bins, int)                               # how many segments contain each bin
for t in TRIALS:
    for a in SEG_START[t]:
        cover[a:a + SEG_LEN] += 1
print("Bins contained in 1, 2 or 3 segments (%):", {k: round(np.mean(cover == k) * 100, 1) for k in (1, 2, 3)})

# %% [markdown]
# **Reading.**
#
# - **610 segments** in total, from 33 to 48 per trial, and none crosses the boundary between two trials.
# - **The manual check works**: stitching the segments of the counts themselves recovers the original
#   counts; the maximum difference, 9·10⁻¹⁶, is the computer's rounding. In the
#   regular overlaps the weights sum to 1, as in AutoLFADS.
# - **Half of the bins are in two segments** (49.3%), the other half in just one (50.5%): of every 20 new bins,
#   10 are shared with the previous segment. Only 0.1% are in three segments, at the end of some
#   trials, where the last segment is aligned to the end.
# - **What the check does not say**: with identical values any weighted average returns the
#   same number. It says the stitching is a correct average, not that the LFADS values at the edges of the
#   segments are good. Whether the seams leave "steps" in the factors we will see on the real outputs
#   of the model (§11), and it matters for the roughness (§12).
#
# ### 9.3 · Training, validation and test
#
# **The test** is the same as in the whole notebook: the 5 blocks of 03, each with 3 test trials. In
# each block LFADS trains **only** on the training trials; for the test trials the
# factors are computed with the already trained model, without retraining (§12). The "shortcut" of 03 (§7), i.e.
# training on all trials, is forbidden.
#
# **The validation** is used during training: to decide when to stop (before the network starts
# to memorize) and which version of the model to keep. It cannot use the test trials. In each
# block we therefore set aside **2 of the 12 training trials**: LFADS learns its weights on the other
# 10, and looks at the 2 validation trials only to decide when to stop. The rule, fixed before
# looking: **the last even trial and the last odd trial** among the block's training trials.
#
# - One even and one odd: the two versions of the task (03, §9.2) are both represented.
# - In block 0 the rule gives exactly the pair 12 and 15 used for CEBRA in §3.2; in §11 we will repeat the
#   checks also with the second pair of §3.2 (6 and 7), in the middle of the session.
# - The common-yardstick classifier instead learns from the points of **all** 12 training trials,
#   as for every other method: the validation trials are not thrown away, they are only excluded from the network
#   weights.

# %%
VAL_LF = []                                                 # for each block, the 2 LFADS validation trials
for tt in TEST_TRIALS:
    tr_ = [int(t) for t in TRIALS if t not in tt]           # the 12 training trials of the block
    VAL_LF.append((max(t for t in tr_ if t % 2 == 0), max(t for t in tr_ if t % 2 == 1)))  # the last even and the last odd
FIT_LF = [[int(t) for t in TRIALS if t not in tt and t not in vt]  # the 10 trials on which LFADS learns its weights
          for tt, vt in zip(TEST_TRIALS, VAL_LF)]

def n_seg(trs):                                             # number of segments of a group of trials
    return sum(len(SEGS[t]) for t in trs)

tab_split = pd.DataFrame([{"test": ", ".join(map(str, tt)), "validation": ", ".join(map(str, sorted(vt))),
                           "trials used for learning": len(ft),
                           "segments: learn / validation / test": f"{n_seg(ft)} / {n_seg(vt)} / {n_seg(tt)}",
                           "seconds used for learning": round(sum(TRIAL_SPAN[t][1] - TRIAL_SPAN[t][0] for t in ft) * io.BIN_S),
                           "of which movement (%)": round(np.mean(MOV[np.isin(trial, ft)]) * 100),
                           "test: odd / even": f"{sum(t % 2 for t in tt)} / {sum(1 - t % 2 for t in tt)}"}
                          for tt, vt, ft in zip(TEST_TRIALS, VAL_LF, FIT_LF)], index=pd.Index(range(5), name="block"))
display(tab_split)
print("Block 0: validation equal to that of CEBRA (§3.2, trials 12 and 15):", tuple(sorted(VAL_LF[0])) == tuple(sorted(VAL_TRIALS)))

# %% [markdown]
# **Reading.**
#
# - In each block LFADS learns the weights on **about 400 segments** (10 trials, about 160 s, half of which
#   movement), stops by looking at about 90 validation segments, and is evaluated on about 120 test
#   segments. The segments overlap, so they are worth less than 400 independent examples. For comparison, the
#   smallest dataset of the Neural Latents Benchmark, MC_Maze-S, has 100 training trials (Pei et al.
#   2021).
# - LFADS learns on 10 trials, CEBRA on 12 (§1.4): the 2 validation trials cost a sixth of the data.
# - **The rule always puts the validation at the end of the session**: 12 and 15 in block 0 (the pair
#   of §3.2), 14 and 15 in blocks 1, 2 and 4, 13 and 14 in block 3. If the signals changed slowly
#   during the session, the validation would be a bit different from the test trials, spread over the whole
#   session. For CEBRA it made no difference (§3.2, pairs 12-15 and 6-7); for LFADS we recheck it
#   in §11.
# - In the test trials of each block there are always trials of both versions (1 or 2 odd, 1 or 2
#   even).
#
# ### 9.4 · The hidden channels
#
# In §12 we will measure something that only generative models can do: **predict unseen data**. The
# measure is that of the Neural Latents Benchmark (Pei et al. 2021), *co-smoothing*: in the test trials
# a part of the channels is hidden, the model sees only the others, and has to guess the activity of
# the hidden ones. It is like working out what a chorister you cannot hear is singing, by listening to the rest
# of the choir: it works only if you have understood the common melody. The score is in **bits per spike**: 0 means
# "like always saying the mean rate of the channel", more means better.
#
# Why not simply use the likelihood of the test trials: GPFA and LFADS have different noise
# models (square root of the counts with Gaussian noise, against counts with Poisson noise), and their
# likelihoods are not comparable; and on a test trial the model would read the same spikes
# it has to reconstruct, the situation in which "copying" goes unnoticed (Keshtkaran & Pandarinath
# 2019). The hidden channels avoid both problems.
#
# **Which channels.** The benchmark hides 25% of the units, chosen at random. We hide 40 channels
# out of 160, once and for all (fixed seed), **10 for each rate band** (channels
# sorted by mean rate and divided into 4 groups of 40): with a plain random draw, 40
# channels might end up being almost all slow or almost all fast.
#
# **A caveat specific to our data.** The benchmark uses isolated neurons (*sorted units*); we
# have **channels**, i.e. electrodes, and two nearby electrodes may partly see the same neurons.
# If a hidden channel had an almost identical "twin" among the visible ones, predicting it would be
# trivial. We check this: for each hidden channel, the highest correlation with a visible channel.
#
# The hidden channels are used **only** by the §12 models dedicated to this measure; all the other
# models see the 160 channels.

# %%
rng_h = np.random.default_rng(0)                            # fixed seed: the same hidden channels at every run
bands = np.array_split(np.argsort(rate_hz), 4)              # channels sorted by mean rate and divided into 4 bands of 40
HIDDEN = np.sort(np.concatenate([rng_h.choice(b_, 10, replace=False) for b_ in bands]))  # 10 hidden channels per band
HELD_IN = np.setdiff1d(np.arange(N_CH), HIDDEN)             # the 120 channels the model sees in the test trials

def describe(ch):                                           # rates and share of spikes of a group of channels
    return {"channels": len(ch), "median rate (Hz)": np.median(rate_hz[ch]),
            "from (Hz)": rate_hz[ch].min(), "to (Hz)": rate_hz[ch].max(),
            "share of all spikes (%)": COUNTS[:, ch].sum() / COUNTS.sum() * 100}

display(pd.DataFrame({"visible": describe(HELD_IN), "hidden": describe(HIDDEN)}).T.round(1))

R_ch = np.corrcoef(io.smooth_rates(s.spikes, sigma_ms=50)[:, ACTIVE].T)  # correlations between the smoothed firing rates (σ = 50 ms) of all pairs of channels
best_vis = R_ch[np.ix_(HIDDEN, HELD_IN)].max(1)             # for each hidden channel, the highest correlation with a visible channel
R_off = R_ch[np.triu_indices(N_CH, 1)]                      # all pairs of distinct channels
print(f"Hidden channels: highest correlation with a visible channel, median {np.median(best_vis):.2f} "
      f"(from {best_vis.min():.2f} to {best_vis.max():.2f}); across all pairs of channels: median {np.median(R_off):.2f}, "
      f"99th percentile {np.percentile(R_off, 99):.2f}")
print(f"Pairs of channels (among all 160) with correlation above 0.9: {int((R_off > 0.9).sum())}")

# %% [markdown]
# **Reading.**
#
# - **The hidden channels resemble the visible ones**: median rate 8.9 Hz against 9.9 Hz, almost the same
#   range, and they carry 25.9% of the spikes with 25% of the channels. The split into bands did its
#   job.
# - **No channel has a "twin"**: no pair, among all 160 channels, has a correlation
#   above 0.9. For a typical hidden channel the most similar visible channel has a correlation of 0.18;
#   the most favorable case reaches 0.61. Predicting a hidden channel therefore does not reduce to copying
#   a nearby one: one has to combine the information of many channels, i.e. use the common structure of the
#   population. This is exactly what we want to measure.
# - **Limitation**: we looked at the correlations of the smoothed firing rates (σ = 50 ms), not the
#   spike coincidences in the same bin; and a single draw of hidden channels. In §12 the score will be
#   valid for these 40 channels.
#
# ### 9.5 · Two caveats about the task: `SnapTo` and the two versions
#
# **`SnapTo`.** This is the automatic repositioning of the virtual arm: the participant does not attempt a
# movement, and the FALCON benchmark excludes it from the decoding evaluation (`eval_mask`). For
# LFADS we keep it, both in training and in the common yardstick:
#
# - **in training**, because LFADS describes the **neural activity**, not the movement: during
#   `SnapTo` the neurons fire anyway, and removing those bins would cut the sequences to pieces;
#   `eval_mask` concerns the kinematics, which LFADS does not use;
# - **in the common yardstick**, because `SnapTo` has been one of the 8 phases of the 1106 points since 03: removing it
#   would change the chance level (from 1/8 to 1/7) and the numbers would no longer be comparable. In §12
#   we will add the result without `SnapTo`, as a control.
#
# **The two versions of the task.** The odd and even trials are two different versions (03, §9.2): the
# repetitions of a phase are not replications. We took this into account in the validation (one trial per
# version, §9.3); for the trajectories we will compute the consistency also within each version, as for
# CEBRA (§4.2).

# %%
em = s.eval_mask                                            # true for the bins evaluated by the FALCON benchmark
display(pd.DataFrame({"session bins": pd.Series(phase).value_counts(),
                      "excluded by eval_mask": pd.Series(phase[~em]).value_counts(),
                      "evaluation points (1106)": pd.Series(phase[bins]).value_counts()}).fillna(0).astype(int)
        .reindex(list(io.MOVEMENT_PHASES) + ["Presentation", "Intertrial", ""]).dropna().rename(index={"": "(no phase)"}))
print(f"Bins excluded by eval_mask: {int((~em).sum())}, of which in SnapTo {int(((~em) & (phase == 'SnapTo')).sum())} "
      f"(SnapTo has {int((phase == 'SnapTo').sum())} bins)")

# %%
fig = plt.figure(figsize=(14, 8.4))
gs = fig.add_gridspec(3, 1, height_ratios=(1.4, 1.1, 0.8))
ax = fig.add_subplot(gs[0])                                 # top: who does what in each block
t_min = np.arange(n_bins) * io.BIN_S / 60                   # time of each bin, in minutes
for k, (tt, vt) in enumerate(zip(TEST_TRIALS, VAL_LF)):
    for t_ in TRIALS:                                       # one rectangle per trial: gray = learn, blue = validation, red = test
        a0, a1 = TRIAL_SPAN[t_]
        col = "C3" if t_ in tt else ("C0" if t_ in vt else "0.8")
        ax.barh(k, (a1 - a0) * io.BIN_S / 60, left=t_min[a0], height=0.6, color=col, edgecolor="white", lw=0.5)
        if t_ in tt or t_ in vt:
            ax.text(t_min[a0] + (a1 - a0) * io.BIN_S / 120, k, str(t_), ha="center", va="center", fontsize=8, color="white")
ax.set(yticks=range(5), yticklabels=[f"block {k}" for k in range(5)], xlabel="time (minutes)",
       title="LFADS learns its weights on the gray trials, stops by looking at the blue ones, is evaluated on the red ones")
ax.invert_yaxis(); ax.grid(False)

t0_ = TRIALS[0]; a0_ = TRIAL_SPAN[t0_][0]                   # middle: the first 6 s of the first trial, with its segments
show_n = 300                                                # 300 bin = 6 s
ax2 = fig.add_subplot(gs[1])
for b_ in range(a0_, a0_ + show_n):                         # the phase of each bin, as a colored band at the top
    ax2.axvspan(b_ * io.BIN_S, (b_ + 1) * io.BIN_S, ymin=0.86, ymax=1, color=io.PHASE_COLORS.get(phase[b_], "white"), lw=0)
SEG_COL = ("k", "0.55")                                     # segments in black and gray, alternating (the colors are already used by the phases)
for i, a in enumerate(SEG_START[t0_]):                      # one horizontal bar per segment, on two alternating rows
    if a < a0_ + show_n:
        ax2.plot([a * io.BIN_S, (a + SEG_LEN) * io.BIN_S], [i % 2, i % 2], color=SEG_COL[i % 2], lw=6, solid_capstyle="butt")
ax2.set(ylim=(-0.6, 2.6), yticks=[0, 1], yticklabels=["segments 1, 3, 5, ...", "segments 2, 4, 6, ..."],
        xlim=(a0_ * io.BIN_S, (a0_ + show_n) * io.BIN_S),
        title=f"Trial {t0_}, first 6 s: a 600 ms segment every 400 ms (the band at the top is the phase)")
ph_show = [p for p in ["Intertrial", "Presentation", *io.MOVEMENT_PHASES] if p in set(phase[a0_:a0_ + show_n])]  # the phases present, in task order
handles = [plt.Rectangle((0, 0), 1, 1, color=io.PHASE_COLORS[p]) for p in ph_show]
ax2.legend(handles, ph_show, loc="upper center", bbox_to_anchor=(0.5, 0.86), fontsize=8, ncol=len(ph_show), frameon=False)
ax2.grid(False)

ax3 = fig.add_subplot(gs[2], sharex=ax2)                    # bottom: the weight of each segment in the stitching
W_ = segment_weights(len(SEG_START[t0_]))
for i, (a, w) in enumerate(zip(SEG_START[t0_], W_)):
    if a < a0_ + show_n:
        ax3.plot((a + np.arange(SEG_LEN) + 0.5) * io.BIN_S, w, color=SEG_COL[i % 2], lw=1.5)
ax3.set(xlabel="time from the start of the session (s)", ylabel="weight", ylim=(-0.05, 1.1),
        title="Stitching weights: in the shared zone the segment that ends (1 − x²) hands over to the one that begins (x²)")
fig.tight_layout(); fig.savefig(FIGS / "04_lfads_data.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# **Reading.**
#
# - **`eval_mask` excludes almost only `SnapTo`**: 290 of its 291 bins, plus 17 pause bins. Of the
#   1106 evaluation points, 58 (5%) are `SnapTo`.
# - **More than half of the session is pause**: 6672 `Presentation` bins out of 12233. Like CEBRA (§1.4),
#   LFADS will have to describe the pauses too, which we do not evaluate.
# - **The figure.** Top: in each block LFADS learns on the gray trials, stops by looking at the two blue ones
#   (always towards the end of the session) and is evaluated on the three red ones. Middle: a 600 ms segment every
#   400 ms, on two rows to see them separately; a segment often contains a phase change. Bottom: the
#   stitching weights, which hand over gradually within the shared 200 ms; the first segment
#   of the trial starts with full weight, because there is no segment before it.
#
# ### Summary of §9
#
# | decision | choice | why | alternatives (where) |
# |---|---|---|---|
# | input | raw 20 ms counts, 160 channels (the same as the GPFA) | LFADS has its own noise model (Poisson) | — |
# | cutting | 600 ms segments every 400 ms, within each trial; 610 segments | the scheme of AutoLFADS and of the benchmark for continuous recordings | 400 ms, 1 s (§14) |
# | stitching | weighted average with the AutoLFADS ramps (1 − x², x²) | gradual handover between segments; verified by hand on the counts | — |
# | split | the 5 blocks of 03; in each block 10 trials for the weights, 2 for validation (the last even and the last odd), 3 for test | same test trials as the whole notebook; validation with both versions | second validation pair (§11) |
# | hidden channels | 40 out of 160, 10 per rate band, fixed seed | generative measure of §12, as in the benchmark | — |
# | `SnapTo` | kept in training and in the common yardstick | LFADS describes the activity, not the movement; comparability with 03 | common yardstick without `SnapTo` (§12) |
#
# **What we did not verify**: whether the validation at the end of the session is representative
# (§11); how the LFADS outputs behave at the seams (§11); whether 600 ms is the best length
# for these data (§14).

# %% [markdown]
# ## 10 · The idea of LFADS on a synthetic example
#
# As for CEBRA (§2) and Isomap (03, §3), before the real data we try LFADS on data whose
# **answer we know**. The example is that of the original paper (Sussillo et al. 2016): the
# **Lorenz system**, three numbers that change in time following fixed rules. It is the famous
# "butterfly attractor": the trajectory circles around two "wings" and jumps from one to the other in a way
# that is unpredictable in the long term, but at each instant the next step is **determined** by where it
# is. This is exactly the kind of rule LFADS has to learn: given the state now, where it goes next.
#
# The Lorenz system is hidden: we only see the spikes of 30 fake neurons, whose rate depends
# on the state. If LFADS works, from the spikes alone it must recover the state.
#
# §10 is divided into two parts:
#
# - **§10a (this one)**: the synthetic data; LFADS written by hand, one block at a time; the manual computation
#   compared with the library, for the outputs and for the loss function. No training;
# - **§10b**: training on the synthetic data and comparison with the ground truth and with the GPFA.
#
# ### 10.1 · The synthetic data: spikes driven by the Lorenz system
#
# We follow the recipe of Sussillo et al. (2016, §4.1), adapted to the format of our data:
#
# | | Sussillo et al. 2016 | here |
# |---|---|---|
# | system | Lorenz with σ = 10, ρ = 28, β = 8/3, integrated with the Euler method | same |
# | neurons | 30 | 30 |
# | rates | linear combination (random weights) of the 3 variables, then exponential; baseline level 5 spikes/s | same, with mean level **10 spikes/s** (the median of our channels, §9.1) |
# | spikes | Poisson | Poisson, in **20 ms** bins |
# | structure | 65 conditions × 20 trials of 1 s, each from a random initial state | **a single continuous recording of 200 s**, cut into 600 ms segments like the real data (§9.2) |
# | split | 80% training, 20% evaluation | first 160 s for the weights (like a real block, §9.3), then 20 s of validation and 20 s of test |
#
# **The speed of the system.** The Lorenz rules have no time unit: we choose one.
# Each 20 ms bin advances the system by 0.015 units, in 10 small Euler steps. So one turn
# around a wing lasts a little under a second, and a 600 ms segment contains about two thirds of one.

# %%
SIGMA_L, RHO_L, BETA_L = 10.0, 28.0, 8 / 3                  # the standard parameters of the Lorenz system (Sussillo et al. 2016)
DT_BIN, N_SUB = 0.015, 10                                   # each 20 ms bin advances the system by 0.015 time units, in 10 Euler steps
T_LZ, N_NEU = 10_000, 30                                    # 10 000 bins (200 s) and 30 fake neurons

def lorenz_step(v, dt):                                     # one Euler step: new state = state + dt × (velocity given by the Lorenz rules)
    x, y, z = v
    return v + dt * np.array([SIGMA_L * (y - x), x * (RHO_L - z) - y, x * y - BETA_L * z])

v_lz = np.array([1.0, 1.0, 1.0])                            # an arbitrary starting state
for _ in range(2_000):                                      # 2000 idle steps: the system settles onto the attractor
    v_lz = lorenz_step(v_lz, DT_BIN / N_SUB)
LZ = np.empty((T_LZ, 3))                                    # the Lorenz state in each bin
for t in range(T_LZ):
    for _ in range(N_SUB):                                  # 10 small steps per bin
        v_lz = lorenz_step(v_lz, DT_BIN / N_SUB)
    LZ[t] = v_lz
LZ_Z = (LZ - LZ.mean(0)) / LZ.std(0)                        # each variable with mean 0 and variance 1

rng_lz = np.random.default_rng(0)                           # fixed seed: same neurons and same spikes at every run
W_LZ = rng_lz.normal(0, 0.35, size=(3, N_NEU))              # how much each neuron "listens" to each of the 3 variables (random weights)
logr_lz = LZ_Z @ W_LZ                                       # the linear combination of the state, for each neuron
B_LZ = np.log(0.2) - np.log(np.exp(logr_lz).mean(0))        # the baseline level that gives on average 0.2 spikes per bin (10 spikes/s) for each neuron
RATE_LZ = np.exp(logr_lz + B_LZ)                            # the true rate: expected spikes in each bin (after the exponential)
SPK_LZ = rng_lz.poisson(RATE_LZ).astype(np.float32)         # the observed spikes: drawn at random according to Poisson

LZ_PARTS = {"learn": (0, 8_000), "validation": (8_000, 9_000), "test": (9_000, 10_000)}  # 160 s + 20 s + 20 s
LZ_START = {k: segment_starts(a0, a1) for k, (a0, a1) in LZ_PARTS.items()}  # the segment starts, with the rule of §9.2
LZ_SEGS = {k: np.stack([SPK_LZ[a:a + SEG_LEN] for a in st]) for k, st in LZ_START.items()}  # the segments: (segments, 30 bins, 30 neurons)

print(f"Spikes: {T_LZ} bins × {N_NEU} neurons; mean rate {SPK_LZ.mean() / io.BIN_S:.1f} spike/s "
      f"(true rate from {RATE_LZ.min() / io.BIN_S:.1f} to {RATE_LZ.max() / io.BIN_S:.1f} spikes/s); bins with 0 spikes: {np.mean(SPK_LZ == 0) * 100:.0f}%")
def nll_sum(k, lam):                                        # total Poisson error, without log k! (it is the same for all models and cancels in comparisons)
    return np.sum(lam - k * np.log(lam))
bps_true = (nll_sum(SPK_LZ, np.broadcast_to(SPK_LZ.mean(0), SPK_LZ.shape)) - nll_sum(SPK_LZ, RATE_LZ)) / SPK_LZ.sum() / np.log(2)
print(f"How much signal there is: with the true rates we gain {bps_true:.3f} bits per spike over the mean rate of each neuron alone")
print("600 ms segments:", {k: v.shape[0] for k, v in LZ_SEGS.items()})
cross = np.flatnonzero(np.diff(np.sign(LZ[:, 0])) != 0)     # the bins in which the first variable changes sign: the jump from one wing to the other
print(f"Jumps from one wing to the other: {len(cross)} in 200 s (one every {T_LZ / len(cross) * io.BIN_S:.2f} s on average)")

# %%
jump = cross[cross > 150][0]                                # the first jump from one wing to the other after the first 3 s
show = slice(jump - 150, jump + 150)                        # 6 s around that jump
t_s = np.arange(300) * io.BIN_S
fig = plt.figure(figsize=(16, 5.2))
gs = fig.add_gridspec(2, 3, height_ratios=(1.4, 1))
a_ = fig.add_subplot(gs[:, 0], projection="3d")             # left: the attractor (the first 160 s), and in red the 6 s shown on the right
a_.plot(*LZ[:8_000:2].T, color="0.7", lw=0.3)
a_.plot(*LZ[show].T, color="C3", lw=1.5)
a_.set(title="The Lorenz system (hidden)", xlabel="x", ylabel="y", zlabel="z")
a_.set_xticklabels([]); a_.set_yticklabels([]); a_.set_zticklabels([])
a_ = fig.add_subplot(gs[:, 1])                              # middle: the 3 variables over time
for j, (lab, c) in enumerate(zip("xyz", ("C0", "C1", "C2"))):
    a_.plot(t_s, LZ_Z[show, j] + 3.5 * (2 - j), color=c, lw=1.5, label=lab)
a_.axvline(150 * io.BIN_S, color="0.5", ls=":", lw=1)
a_.set(xlabel="time (s)", yticks=[], title="The 3 variables (z-score, shifted)\ndotted line: a wing jump")
a_.legend(fontsize=8, loc="upper left")
a_ = fig.add_subplot(gs[0, 2])                              # top right: the spikes of the 30 neurons
for n in range(N_NEU):
    tt_ = np.flatnonzero(SPK_LZ[show, n] > 0)
    a_.scatter(t_s[tt_], np.full(len(tt_), n), s=5, color="k", marker="|")
a_.set(ylabel="neuron", title="What LFADS sees: only the spikes")
a_.tick_params(labelbottom=False)                           # the time numbers only in the panel below
a_2 = fig.add_subplot(gs[1, 2], sharex=a_)                  # bottom right: the true rate of 3 neurons (which LFADS does not see)
for n, c in zip((0, 1, 2), ("C3", "C4", "C5")):
    a_2.plot(t_s, RATE_LZ[show, n] / io.BIN_S, color=c, lw=1.2, label=f"neuron {n}")
a_2.set(xlabel="time (s)", ylabel="spike/s", title="The true rate of 3 neurons (to be recovered)")
a_2.legend(fontsize=7, loc="upper left", ncol=3)
with warnings.catch_warnings():                            # the 3D panel does not allow all margins to be arranged: matplotlib warns, but the figure is fine
    warnings.simplefilter("ignore")
    fig.tight_layout()
fig.savefig(FIGS / "04_lfads_synthetic_data.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# **Reading.**
#
# - **The synthetic spikes resemble ours**: 10 spikes/s on average, 82% of bins empty (77% in the real
#   data, §9.1); the true rate of a neuron goes from less than 1 to about 150 spikes/s.
# - **The signal is weak but realistic**: knowing the true rates we gain 0.22 bits per spike
#   over the mean rate of each neuron alone. This is the same order of magnitude as the values
#   the Neural Latents Benchmark reports on real data (0.2-0.37 bits per spike on the maze, Pei et
#   al. 2021), although there hidden neurons are measured.
# - **The segments**: 400 to learn (as in a block of the real data, §9.3), 50 for validation, 50 for
#   test. The system jumps from one wing to the other 78 times in 200 s, on average every 2.6 s: a 600
#   ms segment contains a jump only now and then.
# - **The figure** shows the difficulty: in the 3 variables the wing jump (the dotted line) is
#   evident, and the true rates of the 3 neurons oscillate between about 3 and 40 spikes/s; in the spikes, the only thing
#   LFADS sees, nothing can be recognized by eye.
#
# ### 10.2 · LFADS, one block at a time
#
# LFADS is a *sequential variational autoencoder*: a network that **summarizes** a segment of data into a little
# information and then **reconstructs** it starting from that alone. The blocks, for a 30-bin segment:
#
# ```
# segment counts (30 bins × 30 neurons)
#    │  1. encoder: reads the segment forwards and backwards
#    ▼
# 2. initial condition g₀: a mean and an uncertainty (64 numbers each)   ← the only information that passes
#    │  3. generator: applies its rules of motion, bin after bin, without looking at the data
#    ▼
# g₁, g₂, …, g₃₀   (the generator state: 64 numbers per bin)
#    │  4. factors: a linear projection of the state (10 numbers per bin)
#    ▼
# f₁, …, f₃₀
#    │  5. rates: linear combination of the factors, then exponential (30 neurons per bin)
#    ▼
# λ₁, …, λ₃₀  →  how likely the observed counts are (Poisson)
# ```
#
# 1. **The encoder** is made of two **GRUs**, one that reads the segment from the first bin to the last and one
#    from the last to the first. A GRU is a small recurrent network: at each bin it takes its previous state
#    and the input of the bin, and computes the new state as **an average between the old state and a
#    new proposal**. How much each weighs is decided by a "gate" (z), itself computed from the data. It is
#    like updating an opinion by reading a book page by page: at each page part of it is kept
#    and part is changed. At the end, the two final states (forward and backward) summarize the whole segment.
# 2. **The initial condition.** From the summary, a linear transformation gives the **mean and uncertainty** of the
#    state from which the generator starts: "the system at the start of the segment is more or less here". This is the
#    *variational* part: during training a random point is drawn within the uncertainty; in evaluation
#    the mean is used.
# 3. **The generator** is another GRU, **with no inputs**: from bin 1 to bin 30 it only applies its rules
#    to its own state. Given the initial condition, the whole trajectory is determined. This is the key point
#    of LFADS: for each segment only the initial condition passes (64 numbers, to describe 900
#    counts), so the network is forced to put its knowledge of the system **in the rules**, which are
#    the same for all segments.
# 4. **The factors** are a linear projection of the generator state, with few numbers: they are
#    the equivalent of the GPFA factors, and the ones we will use for all the measures. The library
#    normalizes the directions of the projection (length 1): the scale is decided by the next step.
# 5. **The rates**: a linear combination of the factors plus a level per neuron, then the
#    **exponential**, which makes everything positive. This is the nonlinear link between factors and rates; the
#    GPFA instead has a linear link, on the square root of the counts.
#
# **The controller** (the inferred external inputs) we leave out, as agreed: with the controller
# the encoder could "push" the generator also during the segment, not only at the start.
#
# Below, LFADS written by hand with `numpy`, with the same formulas as the `lfads-torch` library
# (read in its code: `modules/recurrent.py`, `encoder.py`, `decoder.py`). Two details of the library
# to respect:
#
# - in the GRU the order of the gates is **z, r, n**, not r, z, n as in the standard PyTorch GRU, and the
#   gate r multiplies the state **before** the multiplication by the weights;
# - the GRU state is clipped at ±5 (`cell_clip`), a protection against exploding values.
#
# We write it to **understand**, not to train: training requires the derivatives of everything with respect to
# all the weights, which `torch` computes automatically. We leave that to the library (§10b).

# %%
def sigmoid(x):                                             # the "S" function between 0 and 1, used by the gates
    return 1 / (1 + np.exp(-x))

def gru_step(x, h, w, name, clip):                          # one GRU step of lfads-torch (ClippedGRUCell): input x, state h, weights w[name + ...]
    H = h.shape[1]                                          # state dimension
    W_ih, b_ih = w[name + "weight_ih"], w[name + "bias_ih"] # input weights (for the 3 gates, one on top of the other) and their offsets
    W_hh, b_hh = w[name + "weight_hh"], w[name + "bias_hh"] # weights of the previous state, and their offsets
    x_z, x_r, x_n = np.split(x @ W_ih.T + b_ih, 3, axis=1)  # the contribution of the input to gates z, r and to the proposal n (in this order)
    h_z, h_r = np.split(h @ W_hh[:2 * H].T + b_hh[:2 * H], 2, axis=1)  # the contribution of the previous state to gates z and r
    z = sigmoid(x_z + h_z)                                  # update gate: how much of the old state to keep (from 0 to 1)
    r = sigmoid(x_r + h_r)                                  # reset gate: how much of the old state to use to make the proposal
    n = np.tanh(x_n + (r * h) @ W_hh[2 * H:].T + b_hh[2 * H:])  # the proposal for the new state (r multiplies the state BEFORE the weights)
    return np.clip(z * h + (1 - z) * n, -clip, clip)        # the new state: average of old and proposal, clipped at ±clip

def lfads_forward_np(data, w, hp):
    """Hand-written LFADS: from the counts (segments × bins × neurons) to initial condition, states, factors and log-rates.
    Evaluation version: the mean of the initial condition is used and no dropout."""
    B, T, _ = data.shape                                    # segments, bins per segment, neurons
    clip = hp["cell_clip"]
    h_f = np.tile(w["encoder.ic_enc_h0"][0], (B, 1))        # 1. encoder: starting state of the GRU that reads forwards (a learned parameter)...
    h_b = np.tile(w["encoder.ic_enc_h0"][1], (B, 1))        # ...and of the one that reads backwards
    for t in range(T):                                      # from the first bin to the last
        h_f = gru_step(data[:, t], h_f, w, "encoder.ic_enc.fwd_gru.cell.", clip)
    for t in reversed(range(T)):                            # from the last bin to the first
        h_b = gru_step(data[:, t], h_b, w, "encoder.ic_enc.bwd_gru.cell.", clip)
    summary = np.concatenate([h_f, h_b], axis=1)            # the summary of the segment: the two final states, side by side
    ic_params = summary @ w["encoder.ic_linear.weight"].T + w["encoder.ic_linear.bias"]  # 2. initial condition: a linear transformation...
    ic_mean, ic_logvar = np.split(ic_params, 2, axis=1)     # ...that gives the mean and the logarithm of the variance
    ic_std = np.sqrt(np.exp(ic_logvar) + hp["ic_post_var_min"])  # the uncertainty (standard deviation), never below a minimum
    g = ic_mean @ w["decoder.ic_to_g0.weight"].T + w["decoder.ic_to_g0.bias"]  # 3. generator: the starting state g₀, from the mean of the initial condition
    W_fac = w["decoder.rnn.cell.fac_linear.weight"]         # 4. the factor directions in the generator state...
    W_fac = W_fac / np.linalg.norm(W_fac, axis=1, keepdims=True)  # ...each normalized to length 1 (KernelNormalizedLinear)
    no_input = np.zeros((B, 0))                             # the generator has no inputs: an "empty" input (0 columns)
    gen, fac = [], []
    for t in range(T):                                      # bin after bin, only the generator's rules
        g = gru_step(no_input, g, w, "decoder.rnn.cell.gen_cell.", clip)  # with no input, x @ W_ih is 0: only the offsets b_ih remain
        gen.append(g)
        fac.append(g @ W_fac.T)                             # the factors of the bin
    gen, fac = np.stack(gen, 1), np.stack(fac, 1)           # (segments, bins, 64) and (segments, bins, 10)
    log_rates = fac @ w["readout.0.weight"].T + w["readout.0.bias"]  # 5. log-rates: linear combination of the factors plus an offset per neuron
    return {"ic_mean": ic_mean, "ic_std": ic_std, "gen_states": gen, "factors": fac, "log_rates": log_rates}

# %% [markdown]
# ### 10.3 · The library, and the comparison with the manual computation
#
# **How we use `lfads-torch`.** The library (Sedler & Pandarinath 2023) is installed from GitHub (commit
# `007a555`), with three limits for this version of `pytorch_lightning` (2.6.4):
#
# - the configuration files are not installed: **we build the model directly in Python**
#   (`make_lfads`, below), with the parameters written out explicitly;
# - the module that loads the data (`lfads_torch.datamodules`) cannot be imported with this version of
#   Lightning: we pass the data ourselves;
# - the Lightning trainer breaks on a function that changed shape: we will write our own training
#   loop (§10b).
#
# **As for CEBRA, a separate process** (§2.3): the notebook never imports `torch`. In addition, a
# detail discovered by trying the library: if in the process `torch` is imported **before** `numpy`,
# Python shuts down instantly (the two copies of OpenMP, the one of `torch` and the one of conda,
# conflict); importing `numpy` first works. For this reason all the functions that run in the separate
# process start with `import numpy`. **The LFADS trainings will use the CPU, not the Mac's
# GPU**: in a test on fake data of the same shape the CPU was about 4.5 times faster (the GPU
# pays off for large computations; LFADS does many small computations, one bin at a time). We remeasure it in
# §10b.
#
# **The parameters of the synthetic example**, and where they come from:
#
# | parameter | value | source |
# |---|---|---|
# | encoder, initial condition, generator | 64 numbers each | our choice: small, for little data (the library configuration for MC_Maze-S uses 100) |
# | factors | 10 | our choice: the system has 3, we give more without saying how many are needed |
# | prior variance of the initial condition | 0.1 | Sussillo et al. 2016 (appendix); library configurations |
# | minimum variance of the initial condition | 10⁻⁴ | library configurations |
# | state clipping (`cell_clip`) | 5 | library configurations |
# | dropout | 0.05 | our choice; the library starts from 0.02 and searches it between 0 and 0.6 |
# | *coordinated dropout* | 0.3 | library configuration for MC_Maze-S |
# | weight of the KL term | 1 / (30 bins × 30 neurons) | the "true" likelihood (§10.4); the library searches it between 10⁻⁶ and 10⁻³ |
# | weight of the generator penalty (L2) | 10⁻² | the center (on a log scale) of the library's search interval, 10⁻⁴-1 |
# | KL and L2 ramps | from 0 to the full value in 80 epochs | library configurations |
# | learning rate | 4·10⁻³ | library configurations |
#
# The library values come from its repository: `configs/model/nlb_mc_maze_small.yaml` and
# `scripts/run_pbt.py` (the AutoLFADS search intervals). For the real data we will choose them with a
# small grid (§11).
#
# **The check.** We build a library model with random initial weights (seed 0), give it
# 8 segments of synthetic spikes, and read all its outputs. Then we copy its weights into our hand-written
# LFADS and redo the computation. If the formulas are the same, the outputs must coincide up to
# rounding: the library computes with 32-bit numbers, we with 64.

# %%
from joblib.externals.loky import ProcessPoolExecutor      # (already imported in §2.3) a pool of separate processes

POOL_LF = ProcessPoolExecutor(max_workers=4)                # the processes for LFADS: 4 at a time, on the CPU

SYN_HP = dict(n_in=N_NEU, n_out=N_NEU, seg_len=SEG_LEN,    # input and output neurons (in §12 they will differ), bins per segment
              enc_dim=64, ic_dim=64, gen_dim=64, fac_dim=10,  # sizes of encoder, initial condition, generator, factors
              ic_prior_var=0.1, ic_post_var_min=1e-4, cell_clip=5.0,  # prior and minimum variance of the initial condition; state clipping
              dropout=0.05, cd_rate=0.3,                    # dropout and coordinated dropout
              kl_ic_scale=1 / (SEG_LEN * N_NEU), l2_gen_scale=1e-2, l2_ic_enc_scale=0.0,  # weights of the KL term and of the L2 penalties
              ramp_epochs=80, lr=4e-3, loss_scale=1.0)      # ramps, learning rate, cost scale (the library uses 10⁴; here 1, to read the numbers)

def make_lfads(hp):
    """An lfads-torch LFADS built from the parameters hp, without controller. Called in the separate process."""
    import numpy as np                                      # numpy BEFORE torch: otherwise Python shuts down (§10.3)
    import torch
    from torch import nn
    from lfads_torch.model import LFADS                     # the library's model
    from lfads_torch.modules import augmentations           # the coordinated dropout
    from lfads_torch.modules.priors import MultivariateNormal, Null  # the prior distribution of the initial condition; "none" for the controller
    from lfads_torch.modules.recons import Poisson          # the noise model: Poisson
    from lfads_torch.modules.readin_readout import FanInLinear  # the linear readout from factors to log-rates
    cd = augmentations.CoordinatedDropout(cd_rate=hp["cd_rate"], cd_pass_rate=0.0, ic_enc_seq_len=0)
    return LFADS(encod_data_dim=hp["n_in"], encod_seq_len=hp["seg_len"], recon_seq_len=hp["seg_len"],  # shape of the data
                 ext_input_dim=0, ic_enc_seq_len=0,         # no external input; the encoder reads the whole segment
                 ic_enc_dim=hp["enc_dim"], ci_enc_dim=0, ci_lag=1, con_dim=0, co_dim=0,  # encoder; controller off (dimensions 0)
                 ic_dim=hp["ic_dim"], gen_dim=hp["gen_dim"], fac_dim=hp["fac_dim"], dropout_rate=hp["dropout"],
                 reconstruction=nn.ModuleList([Poisson()]), variational=True, co_prior=Null(),
                 ic_prior=MultivariateNormal(mean=0.0, variance=hp["ic_prior_var"], shape=hp["ic_dim"]),
                 ic_post_var_min=hp["ic_post_var_min"], cell_clip=hp["cell_clip"],
                 train_aug_stack=augmentations.AugmentationStack([cd], batch_order=[0], loss_order=[0]),  # coordinated dropout only in training
                 infer_aug_stack=augmentations.AugmentationStack(),
                 readin=nn.ModuleList([nn.Identity()]), readout=nn.ModuleList([FanInLinear(hp["fac_dim"], hp["n_out"])]),
                 loss_scale=hp["loss_scale"], recon_reduce_mean=True,  # reconstruction error as a mean over segments, bins and neurons
                 lr_scheduler=False, lr_init=hp["lr"], lr_stop=1e-5, lr_decay=0.95, lr_patience=6,
                 lr_adam_beta1=0.9, lr_adam_beta2=0.999, lr_adam_epsilon=1e-8, weight_decay=0.0,
                 l2_start_epoch=0, l2_increase_epoch=hp["ramp_epochs"], l2_ic_enc_scale=hp["l2_ic_enc_scale"],
                 l2_ci_enc_scale=0.0, l2_gen_scale=hp["l2_gen_scale"], l2_con_scale=0.0,
                 kl_start_epoch=0, kl_increase_epoch=hp["ramp_epochs"], kl_ic_scale=hp["kl_ic_scale"], kl_co_scale=0.0)

def lfads_library_check(hp, seed, data):
    """In the separate process: a library model with the initial weights (seed `seed`), its outputs and its cost on `data`."""
    import numpy as np                                      # numpy BEFORE torch
    import torch
    from lfads_torch.tuples import SessionBatch             # the library's data format
    from lfads_torch.modules.l2 import compute_l2_penalty   # the library's L2 penalty
    torch.set_num_threads(1)                                # a single CPU thread (§10b: more is not needed)
    torch.manual_seed(seed)                                 # seed of the initial weights
    model = make_lfads(hp)
    model.eval()                                            # evaluation mode: no dropout
    model.log = model.log_dict = lambda *a, **k: None       # outside the Lightning trainer there is nowhere to write the measures
    X = torch.tensor(data, dtype=torch.float32)
    batch = SessionBatch(encod_data=X, recon_data=X, ext_input=torch.zeros(len(X), X.shape[1], 0),
                         truth=torch.zeros(0), sv_mask=torch.ones_like(X))  # input = data to reconstruct; no external input
    with torch.no_grad():                                   # no derivatives: forward computation only
        out = model({0: batch}, sample_posteriors=False, output_means=False)[0]  # all the outputs, with the mean of the initial condition
        recon = model.recon[0].compute_loss(X, out.output_params).mean()  # the library's reconstruction error (Poisson)
        kl = model.ic_prior(out.ic_mean, out.ic_std)        # the library's KL term
        l2 = compute_l2_penalty(model, model.hparams)       # the library's L2 penalty
        model.hparams.variational = False                   # for the check: the cost with the mean, without random draws
        total = model._shared_step({0: (batch, ())}, 0, "valid")  # the total cost, computed by the function the library uses in training
    w = {k: v.detach().numpy().astype(np.float64) for k, v in model.state_dict().items()}  # all the weights, as numpy arrays
    return {"weights": w, "ic_mean": out.ic_mean.numpy(), "ic_std": out.ic_std.numpy(), "gen_states": out.gen_states.numpy(),
            "factors": out.factors.numpy(), "log_rates": out.output_params[..., 0].numpy(),
            "recon": float(recon), "kl": float(kl), "l2": float(l2), "total": float(total)}

X_chk = LZ_SEGS["learn"][:8]                               # 8 synthetic spike segments
LIB = POOL_LF.submit(lfads_library_check, SYN_HP, 0, X_chk).result()  # the library, in the separate process
MINE = lfads_forward_np(X_chk.astype(np.float64), LIB["weights"], SYN_HP)  # the manual computation, with the same weights
w_bad = {k: v.copy() for k, v in LIB["weights"].items()}   # check of the check: the same weights...
w_bad["decoder.rnn.cell.gen_cell.weight_hh"][0, 0] += 1e-3  # ...with a single generator weight changed by 0.001
BAD = lfads_forward_np(X_chk.astype(np.float64), w_bad, SYN_HP)
OUT_NAMES = {"ic_mean": "initial condition: mean", "ic_std": "initial condition: uncertainty",
             "gen_states": "generator states", "factors": "factors", "log_rates": "log-frequencies"}
display(pd.DataFrame({lab: {"shape": str(LIB[name].shape), "values: absolute maximum": f"{np.abs(LIB[name]).max():.3f}",
                            "maximum difference by hand − library": f"{np.abs(MINE[name] - LIB[name]).max():.1e}",
                            "same, with a weight changed by 0.001": f"{np.abs(BAD[name] - LIB[name]).max():.1e}"}
                      for name, lab in OUT_NAMES.items()}).T)
print("Model parameters:", sum(v.size for k, v in LIB["weights"].items() if not k.endswith("bias_hh")),
      "(without the b_hh offsets, which the library keeps fixed)")

# %% [markdown]
# **Reading.**
#
# - **The manual computation coincides with the library**: in all the blocks, from the initial condition to the
#   log-rates, the maximum difference is about 10⁻⁷, the rounding of the library's 32-bit
#   numbers.
# - **The comparison is sensitive**: changing a single generator weight by 0.001 (one out of more than 12 000) the
#   difference in the generator states rises to 2·10⁻⁵, about 200 times the rounding. The initial
#   condition instead stays identical, as it must: that weight comes after the encoder. Had we got a
#   formula wrong, for example the order of the gates, we would see it.
# - The model has about **71 000 parameters** for 30 neurons. For comparison, a segment contains 900
#   counts, and the 400 segments on which we learn 360 000 (which however overlap and are almost all 0).
# - **What the check does not cover**: the weights are the initial, random ones, and the computation is the
#   evaluation one (mean of the initial condition, no dropout). The random draw of the initial
#   condition and the coordinated dropout, used in training, belong to the library and we did not redo them
#   by hand; and our computation has no controller.
#
# ### 10.4 · The cost function, by hand
#
# LFADS is trained by minimizing a sum of three terms, averaged over segments:
#
# 1. **The reconstruction error** (Poisson): how improbable the observed counts are, given the
#    model's rates. For a bin with rate λ and k spikes, the Poisson probability is
#    λᵏ e^(−λ) / k!, and the error is its logarithm with the sign changed:
#    $$\mathrm{NLL} = \lambda - k \log\lambda + \log k!$$
#    The lower it is, the better the model explains the data.
# 2. **The KL term**: how far the initial condition of a segment (mean μ, uncertainty σ) departs from the
#    **prior** one (mean m, variance 0.1, the same for all segments). For each of the 64
#    dimensions:
#    $$\mathrm{KL} = \log\frac{\sigma_p}{\sigma} + \frac{\sigma^2 + (\mu - m)^2}{2\sigma_p^2} - \frac12$$
#    It is **a price for each piece of information** that the encoder passes to the generator. Without a price,
#    the encoder could write the whole segment into the initial condition, and the generator copy it back out.
# 3. **The L2 penalty** on the recurrent weights of the generator: it prefers simple rules of motion.
#
# Put together, error and KL form the **lower bound of the likelihood** (*ELBO*, with the sign
# changed), the quantity that every variational autoencoder minimizes (Kingma & Welling 2014; review, §4.2):
# for each segment, the sum of the errors over all bins and neurons, plus the KL.
#
# **The weight of the KL term.** The library does not sum the errors: it takes their **mean** over segments, bins and neurons
# (`recon_reduce_mean=True`), while the KL remains a sum over the 64 dimensions. For the cost to be
# proportional to the true ELBO, the KL must therefore be divided by the number of values reconstructed in a segment,
# 30 bins × 30 neurons = 900: this is the weight chosen in `SYN_HP`. With a larger weight the initial condition
# lets less information through; with a smaller one, more. It is one of the choices of §11.
#
# **The ramps.** At the start of training the KL and L2 weights start from (almost) 0 and grow to the
# full value in 80 epochs: first the model learns to reconstruct, then the price is charged. At epoch
# 0 they are 1/81.
#
# **A library detail.** The library does not compute the term log k! exactly: for k ≥ 2 it uses
# Stirling's approximation, k log k − k + ½ log(2πk) (the `full=True` option of PyTorch's `poisson_nll_loss`); for k = 0 and 1 it is 0 in both cases. It does not depend on the model, so it does not change
# the training; it only changes the value of the cost. We compute both versions.

# %%
from scipy.special import gammaln                           # log k! = gammaln(k + 1)

def stirling(k):                                            # the approximation of log k! used by PyTorch (full=True): only for k > 1, otherwise 0
    k_safe = np.where(k > 1, k, 1.0)                        # for k ≤ 1 any value, which is then not used (avoids log 0)
    return np.where(k > 1, k_safe * np.log(k_safe) - k_safe + 0.5 * np.log(2 * np.pi * k_safe), 0.0)

def poisson_nll(k, log_rate, exact=True):                   # Poisson error of each bin and neuron: λ − k log λ + log k!
    return np.exp(log_rate) - k * log_rate + (gammaln(k + 1) if exact else stirling(k))

def kl_normal(mu, sd, prior_mean, prior_var):               # KL between the initial condition (μ, σ) and the prior one, summed over dimensions, averaged over segments
    sp = np.sqrt(prior_var)
    return np.mean(np.sum(np.log(sp / sd) + (sd ** 2 + (mu - prior_mean) ** 2) / (2 * prior_var) - 0.5, axis=1))

def l2_gen(w, hp):                                          # the library's L2 penalty: ½ ‖recurrent weights‖², times its weight, divided by the number of weights
    terms = [(w["decoder.rnn.cell.gen_cell.weight_hh"], hp["l2_gen_scale"]),
             (w["encoder.ic_enc.fwd_gru.cell.weight_hh"], hp["l2_ic_enc_scale"]),
             (w["encoder.ic_enc.bwd_gru.cell.weight_hh"], hp["l2_ic_enc_scale"])]
    used = [(W_, s_) for W_, s_ in terms if s_ > 0]         # like the library: only the weights with a nonzero penalty
    return sum(s_ * 0.5 * np.sum(W_ ** 2) for W_, s_ in used) / (sum(W_.size for W_, _ in used) + 1e-8)

ramp0 = 1 / (SYN_HP["ramp_epochs"] + 1)                     # the ramps at epoch 0: (0 + 1) / (80 + 1)
k_chk = X_chk.astype(np.float64)
recon_stir = poisson_nll(k_chk, MINE["log_rates"], exact=False).mean()  # reconstruction error like the library (Stirling), mean over everything
recon_exact = poisson_nll(k_chk, MINE["log_rates"], exact=True).mean()  # the same with exact log k!
kl_mine = kl_normal(MINE["ic_mean"], MINE["ic_std"], LIB["weights"]["ic_prior.mean"], np.exp(LIB["weights"]["ic_prior.logvar"]))
l2_mine = l2_gen(LIB["weights"], SYN_HP)
total_mine = SYN_HP["loss_scale"] * (recon_stir + ramp0 * l2_mine + ramp0 * SYN_HP["kl_ic_scale"] * kl_mine)  # the library's cost formula
cost_rows = {"reconstruction error (Stirling)": (recon_stir, LIB["recon"]), "KL term": (kl_mine, LIB["kl"]),
             "L2 penalty": (l2_mine, LIB["l2"]), "total cost (ramps at epoch 0)": (total_mine, LIB["total"])}
display(pd.DataFrame({name: {"by hand": f"{a:.6g}", "library": f"{b:.6g}", "relative difference": f"{abs(a - b) / abs(b):.1e}"}
                      for name, (a, b) in cost_rows.items()}).T)
n_val = SEG_LEN * N_NEU                                     # values reconstructed in a segment: 900
elbo_neg = poisson_nll(k_chk, MINE["log_rates"], exact=True).sum((1, 2)).mean() + kl_mine  # −ELBO per segment, with exact log k!
print(f"Error with exact log k!: {recon_exact:.4f} (Stirling: {recon_stir:.4f}); difference due only to bins with 2 or more spikes")
true_chk = np.stack([RATE_LZ[a:a + SEG_LEN] for a in LZ_START["learn"][:8]])  # the true rates of the same 8 segments
mean_rate = SPK_LZ[:8_000].mean(0)                          # the mean rate of each neuron over the 160 s on which we learn
print(f"Comparison terms for the error (Stirling), on the same segments: with the true rates (the perfect model) "
      f"{poisson_nll(k_chk, np.log(true_chk), exact=False).mean():.4f}; with the mean rate of each neuron alone "
      f"{poisson_nll(k_chk, np.broadcast_to(np.log(mean_rate), k_chk.shape), exact=False).mean():.4f}")
print(f"Negative ELBO, per segment: {elbo_neg:.1f}; library cost with full ramps and without L2, × {n_val}: "
      f"{(recon_stir + SYN_HP['kl_ic_scale'] * kl_mine) * n_val:.1f} (the difference is the Stirling correction: "
      f"{(recon_exact - recon_stir) * n_val:.1f})")

# %% [markdown]
# **Reading.**
#
# - **The manual cost coincides with that of the library** in all three terms and in the total
#   (relative differences between 10⁻⁸ and 10⁻⁷).
# - **The newly created model reconstructs worse than one that knows only the mean rate**: error 1.011
#   per value, against 0.491 for the mean rate and 0.479 for the true rates. The reason: with the initial
#   weights the log-rates are close to 0, i.e. about 1 spike per bin, five times too many. The first
#   thing training will learn is the level of each neuron.
# - **The information about the hidden state lies in a small difference**: between the mean rate (0.491)
#   and the true rates (0.479) there is 2.5%. This is why we measure in bits per spike, which looks at exactly
#   that difference (§9.4, §12).
# - **The KL term weighs little now, but not at steady state.** With the initial weights it is 218 per segment: the uncertainty
#   of the initial condition (about 1) is much wider than the prior one (0.32). At epoch 0, with weight
#   1/900 and the ramp at 1/81, it adds just 0.003 to the cost; with the full weight it would add 0.24 per
#   value, about half the error of a good model (0.48). The KL weight therefore matters, and this is why it is in the
#   grid of §11.
# - **The ELBO**: per segment, the summed error plus KL is 1128.0; the library cost with full ramps and
#   without L2, multiplied by 900, is 1127.5. The difference, 0.5, is entirely due to Stirling's
#   approximation of log k!, which does not depend on the model.
#
# **End of §10a.** In §10b: training on these data, the comparison between LFADS and the ground truth (the Lorenz state
# and the rates) and with the GPFA, the seed check and the computation time on CPU and GPU.

# %% [markdown]
# ## 10b · LFADS trained on the synthetic data
#
# Now we really train LFADS on the Lorenz spikes of §10.1 and check, with the ground truth in hand,
# what it has learned. Three questions:
#
# 1. **Is it learning or memorizing?** Training curves on the seen segments and on the validation ones,
#    and, only here where we know it, the distance from the truth (§10.6).
# 2. **Does it recover the hidden state and the true rates?** Better or worse than the GPFA and than smoothing,
#    on equal terms (§10.7)?
# 3. **The controls**: the seams between segments, the manual computation with the trained weights, the seed, the timings
#    on CPU and GPU (§10.8).
#
# ### 10.5 · The training loop
#
# A training is made of **epochs**: in each epoch LFADS sees all 400 segments, shuffled, in groups
# of 64 (the last group is smaller). For each group:
#
# 1. the **library cost** (§10.4), with the two things done only in training: the
#    initial condition **drawn at random** within its uncertainty, and the **coordinated dropout**;
# 2. the **derivatives** of the cost with respect to all the weights (computed by `torch`);
# 3. one step of the **AdamW** optimizer (Loshchilov & Hutter 2019), with the library's learning rate
#    (4·10⁻³).
#
# **The coordinated dropout** (Keshtkaran & Pandarinath 2019): for each group 30% of the values
# (bin × neuron), chosen at random, are hidden from the encoder, and the error is measured **only on those hidden values**. The
# network can therefore not merely copy the input to the output: to guess a value it has not
# seen it has to use the others, i.e. the common structure. It is like an exercise in which some
# words of a text are covered and you are asked to guess them from the rest.
#
# **After each epoch** we measure the reconstruction error on the **validation** segments, with the mean of the
# initial condition and without dropout, and keep the version of the model with the lowest error. We
# stop when the error has not improved for 200 epochs, after 1000 at most. Before the KL and L2 ramps
# are finished (80 epochs) no version is chosen: a model "without prices" is not what we
# want. These are the rules of the library's trainer (`configs/single.yaml`: 1000 epochs, patience
# 200, initial period = end of the ramps).
#
# **Two technical details, because we do not use the Lightning trainer** (§10.3):
#
# - the KL and L2 ramps read the epoch number from the trainer; without a trainer they would stay stuck
#   at epoch 0, so we tell them ourselves at each epoch;
# - the library **clips the derivatives** when their length exceeds 200, but with the cost multiplied
#   by 10⁴ (`loss_scale`); with our cost, multiplied by 1, the same threshold is 200 / 10⁴ = 0.02.
#
# **Saving to disk**, as for CEBRA (§3.1): each trained model is saved in
# `~/tesi_h1_work/cache_04/lfads` with a name that is the fingerprint of everything that determines it (the code of
# `lfads_fit` and `make_lfads`, the parameters, the seed, all the data). Rerunning the notebook reads it back;
# `RETRAIN = True` (§3.1) redoes everything from scratch.

# %%
import copy                                                 # to keep a copy of the best version of the model

CACHE_LF = CACHE / "lfads"                                  # the saved LFADS models, in the same folder as CEBRA's (§3.1), outside the Desktop
CACHE_LF.mkdir(parents=True, exist_ok=True)
CLIP_LF = 200 / 1e4                                         # the library's gradient clipping threshold (200 with the cost × 10⁴), for our cost × 1

def lfads_fit(hp, seed, train, val, apply, watch=None, max_epochs=1000, patience=200, batch_size=64, clip=CLIP_LF,
              device="cpu", model_path=None, rate_cols=None):
    """Trains a library LFADS (make_lfads) on the segments `train` (segments × bins × neurons) and picks the version with
    the lowest error on the segments `val`. Returns, for each array of `apply`, factors and rates (mean of the
    initial condition, without dropout; of the rates only the columns rate_cols, if given), the training
    history and the weights. watch = {name: (segments, true rates or None[, columns hidden from the encoder])}: other
    measures at each epoch (error, bits per spike, R² with the ground truth). Runs in the separate process."""
    import copy, time                                       # copy of the best version; duration
    import numpy as np                                      # numpy BEFORE torch (§10.3)
    import torch
    from lfads_torch.tuples import SessionBatch             # the library's data format
    torch.set_num_threads(1)                                # one thread: the computations are small (§10.8)
    torch.manual_seed(seed)                                 # initial weights, coordinated dropout, dropout, draws of the initial condition
    rng = np.random.default_rng(seed)                       # the order of the segments in each epoch
    model = make_lfads(hp)
    model.__class__ = type("LFADSconEpoch", (type(model),), {"current_epoch": 0})  # an epoch number that we can write ourselves (for the ramps)
    model = model.to(device)
    model.log = model.log_dict = lambda *a, **k: None       # outside Lightning there is nowhere to write the measures
    opt = model.configure_optimizers()                      # AdamW with the learning rate of hp, like the library
    n_in = hp["n_in"]                                       # the neurons the encoder sees (the first n_in; in §12 not all of them)

    def as_batch(X):                                        # numpy segments -> the library's format
        Xt = torch.tensor(X, dtype=torch.float32, device=device)
        return SessionBatch(encod_data=Xt[..., :n_in], recon_data=Xt,  # the encoder sees the first n_in neurons, all of them are reconstructed
                            ext_input=torch.zeros(len(Xt), Xt.shape[1], 0, device=device),
                            truth=torch.zeros(0), sv_mask=torch.ones_like(Xt))

    def evaluate(X, chunk=256):                             # factors, log-rates and mean error, with the mean of the initial condition and without dropout
        model.eval()
        fac, logr, err = [], [], []
        with torch.no_grad():
            for i in range(0, len(X), chunk):
                sb = as_batch(X[i:i + chunk])
                out = model({0: sb}, sample_posteriors=False, output_means=False)[0]
                fac.append(out.factors.cpu().numpy())
                logr.append(out.output_params[..., 0].cpu().numpy())
                err.append(model.recon[0].compute_loss(sb.recon_data, out.output_params).sum().item())  # the library's Poisson error (Stirling)
        return np.concatenate(fac), np.concatenate(logr), sum(err) / X.size

    def bits(logr, X, cols):                                # bits per spike on the columns cols: gain over the mean rate of each channel in these data (§9.4)
        k, lr = X[..., cols].reshape(-1, len(cols)), logr[..., cols].reshape(-1, len(cols))
        lam0 = np.maximum(k.mean(0), 1e-9)                  # the mean rate of each channel (never exactly 0)
        nll_model = np.sum(np.exp(lr) - k * lr)             # Poisson error of the model (without log k!, which cancels)
        nll_null = np.sum(lam0 - k * np.log(lam0))          # error of one who knows only the mean
        return float((nll_null - nll_model) / k.sum() / np.log(2))

    def r2_rates(logr, truth):                              # R² between the model rates and the true ones, averaged over neurons
        p_, t_ = np.exp(logr).reshape(-1, truth.shape[-1]), truth.reshape(-1, truth.shape[-1])
        return float(np.mean(1 - ((t_ - p_) ** 2).sum(0) / ((t_ - t_.mean(0)) ** 2).sum(0)))

    hist, best_err, best_state, best_epoch = [], np.inf, None, 0
    t0, steps = time.time(), 0
    for epoch in range(1, max_epochs + 1):
        model.current_epoch = epoch - 1                     # the library counts epochs from 0
        model.train()                                       # training mode: dropout on
        order = rng.permutation(len(train))                 # the segments in a new order at each epoch
        costs = []
        for i in range(0, len(train), batch_size):          # in groups of 64 segments
            loss = model._shared_step({0: (as_batch(train[order[i:i + batch_size]]), ())}, i, "train")  # the library's cost (with CD and random draw)
            opt.zero_grad()                                 # reset the derivatives of the previous step
            loss.backward()                                 # the derivatives of the cost with respect to all the weights
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)  # clipping of derivatives that are too long
            opt.step()                                      # one AdamW step
            costs.append(loss.item()); steps += 1
        val_err = evaluate(val)[2]                          # error on the validation segments
        row = {"epoch": epoch, "costo di training": float(np.mean(costs)), "errore, validazione": val_err}
        for name, (Xw, truth, *hid) in (watch or {}).items():  # extra measures (for the synthetic data, also the distance from the ground truth)
            _, lw, ew = evaluate(Xw)
            row[f"errore, {name}"] = ew
            if truth is not None:
                row[f"R² frequenze, {name}"] = r2_rates(lw, truth)
            if hid:                                         # channels hidden from the encoder: bits per spike separately for visible and hidden
                vis = np.setdiff1d(np.arange(Xw.shape[-1]), hid[0])
                row[f"bit per spike visibili, {name}"] = bits(lw, Xw, vis)
                row[f"bit per spike nascosti, {name}"] = bits(lw, Xw, np.asarray(hid[0]))
            else:
                row[f"bit per spike, {name}"] = bits(lw, Xw, np.arange(Xw.shape[-1]))
        hist.append(row)
        if epoch >= hp["ramp_epochs"] and val_err < best_err:  # after the ramps: we keep the best version on the validation
            best_err, best_state, best_epoch = val_err, copy.deepcopy(model.state_dict()), epoch
        if best_epoch > 0 and epoch - best_epoch >= patience:  # 200 epochs without improvement: we stop
            break
    if best_state is not None:
        model.load_state_dict(best_state)                   # the best version
    seconds = time.time() - t0
    if model_path is not None:
        torch.save(model.state_dict(), model_path)          # the model, on disk
    outs = [evaluate(X)[:2] for X in apply]                 # factors and log-rates of each array to project
    keep = slice(None) if rate_cols is None else np.asarray(rate_cols)  # which rates to keep (to avoid filling the disk)
    return {"factors": [f_.astype(np.float32) for f_, _ in outs], "rates": [np.exp(l_[..., keep]).astype(np.float32) for _, l_ in outs],
            "history": hist, "best_epoch": best_epoch, "epochs": len(hist), "steps": steps, "seconds": seconds,
            "weights": {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}}

def lf_key(job):                                            # fingerprint of an LFADS training: it changes if anything that determines it changes
    h = hashlib.sha1()
    for f in (lfads_fit, make_lfads):
        h.update(inspect.getsource(f).encode())             # the code
    h.update(json.dumps({k: job.get(k) for k in ("hp", "seed", "max_epochs", "patience", "batch_size", "clip", "device", "rate_cols")},
                        sort_keys=True, default=str).encode())  # parameters, seed, training rules, rates kept
    arrays = [job["train"], job["val"], *job["apply"]]      # the data...
    for name, (Xw, truth, *hid) in sorted((job.get("watch") or {}).items()):
        h.update(name.encode()); arrays += [Xw] + ([] if truth is None else [truth]) + [np.asarray(c_) for c_ in hid]  # ...and those of the extra measures
    for a in arrays:
        h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()[:16]

def run_lfads_cached(jobs):                                 # runs the trainings that are not already on disk, 4 at a time; then reads everything back from disk
    keys = [lf_key(j) for j in jobs]
    todo = [i for i, k in enumerate(keys) if RETRAIN or not (CACHE_LF / f"{k}.npz").exists()]
    futures = {i: POOL_LF.submit(lfads_fit, **dict(jobs[i], model_path=CACHE_LF / f"{keys[i]}.pt")) for i in todo}
    for i, fut in futures.items():                          # new results are saved as soon as they arrive
        r = fut.result()
        np.savez(CACHE_LF / f"{keys[i]}.npz", history=json.dumps(r["history"]),
                 info=json.dumps({k: r[k] for k in ("best_epoch", "epochs", "steps", "seconds")}),
                 **{f"factors_{j}": f_ for j, f_ in enumerate(r["factors"])}, **{f"rates_{j}": x_ for j, x_ in enumerate(r["rates"])})
        np.savez(CACHE_LF / f"{keys[i]}_w.npz", **r["weights"])  # the weights, also as numpy arrays (for the manual computation)
    out = []
    for k in keys:                                          # all the results, in job order, read from disk
        z, zw = np.load(CACHE_LF / f"{k}.npz"), np.load(CACHE_LF / f"{k}_w.npz")
        n_ = sum(f_.startswith("factors_") for f_ in z.files)
        out.append({"factors": [z[f"factors_{j}"] for j in range(n_)], "rates": [z[f"rates_{j}"] for j in range(n_)],
                    "history": pd.DataFrame(json.loads(str(z["history"]))).set_index("epoch"),
                    **json.loads(str(z["info"])), "weights": {f_: zw[f_] for f_ in zw.files}, "key": k})
    print(f"LFADS trainings: {len(todo)} run now, {len(jobs) - len(todo)} read back from disk")
    return out

# %% [markdown]
# ### 10.6 · Is it learning or memorizing?
#
# Four trainings in parallel, one per process:
#
# - **LFADS as in `SYN_HP`** (coordinated dropout at 30%), with **3 seeds** (0, 1, 2);
# - the same **without coordinated dropout** (seed 0), to see what happens without the protection
#   against copying.
#
# At every epoch of each training we measure the reconstruction error on 50 **seen** segments
# (drawn from those it learns on) and on the 50 **validation** segments. In addition, only because here
# we know it, the **distance from the ground truth**: the R² between the model rates and the true ones. It does not enter
# model selection (which looks only at the validation error): it serves to check whether the validation
# error is a good judge. About 5 minutes the first time.

# %%
LZ_TRUE_SEGS = {k: np.stack([RATE_LZ[a:a + SEG_LEN] for a in st]) for k, st in LZ_START.items()}  # the true rates, segment by segment
LZ_WATCH = {"pezzi visti": (LZ_SEGS["learn"][:50], LZ_TRUE_SEGS["learn"][:50]),  # 50 segments among those it learns on
            "validation": (LZ_SEGS["validation"], LZ_TRUE_SEGS["validation"])}

def syn_job(hp, seed):                                      # one training on the synthetic data
    return dict(hp=hp, seed=seed, train=LZ_SEGS["learn"], val=LZ_SEGS["validation"],
                apply=[LZ_SEGS[k] for k in LZ_PARTS], watch=LZ_WATCH)  # project all segments: learn, validation, test

SYN_RUNS = [("con CD", 0), ("con CD", 1), ("con CD", 2), ("senza CD", 0)]  # (version, seed)
SYN_HPS = {"con CD": SYN_HP, "senza CD": dict(SYN_HP, cd_rate=0.0)}
res_syn_lf = dict(zip(SYN_RUNS, run_lfads_cached([syn_job(SYN_HPS[v], sd) for v, sd in SYN_RUNS])))
display(pd.DataFrame({f"{v}, seed {sd}": {"epochs": r["epochs"], "best epoch": r["best_epoch"],
                                         "duration (minutes)": round(r["seconds"] / 60, 1),
                                         "ms per step, measurements included (4 in parallel)": round(r["seconds"] / r["steps"] * 1000, 1)}
                      for (v, sd), r in res_syn_lf.items()}).T)

# %%
REF_LZ = {name: {"perfect": poisson_nll(Xw.astype(float), np.log(truth), exact=False).mean(),  # the error with the true rates...
                 "solo la media": poisson_nll(Xw.astype(float), np.broadcast_to(np.log(mean_rate), Xw.shape), exact=False).mean()}  # ...and with the mean rate alone
          for name, (Xw, truth) in LZ_WATCH.items()}
display(pd.DataFrame(REF_LZ).T.round(4).rename_axis("reference reconstruction error"))
fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
for (v, sd), ls in ((("con CD", 0), "-"), (("senza CD", 0), "--")):
    h_ = res_syn_lf[(v, sd)]["history"]
    best = res_syn_lf[(v, sd)]["best_epoch"]
    ax[0].plot(h_.index, h_["costo di training"], ls, color="0.3", lw=1.2, label=f"{v}")
    for name, c in (("pezzi visti", "C0"), ("validation", "C3")):  # the error as distance from the perfect model on those segments (0 = perfect)
        ax[1].plot(h_.index, (h_[f"errore, {name}"] - REF_LZ[name]["perfect"]) * 1000, ls, color=c, lw=1.2, label=f"{name}, {v}")
    for col, c in (("R² frequenze, pezzi visti", "C0"), ("R² frequenze, validazione", "C3")):
        ax[2].plot(h_.index, h_[col], ls, color=c, lw=1.2, label=f"{col.split(', ')[1]}, {v}")
    for a_ in ax[1:]:
        a_.axvline(best, color="0.5", ls=ls, lw=0.8)
for name, c in (("pezzi visti", "C0"), ("validation", "C3")):
    ax[1].axhline((REF_LZ[name]["solo la media"] - REF_LZ[name]["perfect"]) * 1000, color=c, ls=":", lw=1, label=f"mean rate only, {name}")
ax[1].axhline(0, color="k", lw=0.8, label="perfect model")
ax[0].set(xlabel="epoch", ylabel="cost (mean per value)", title="The cost that LFADS minimizes\n(with CD and random draws)", ylim=(None, 0.6))
ax[1].set(xlabel="epoch", ylabel="error − error of the perfect model (× 1000)", ylim=(-6, 32),
          title="Error on seen and validation segments\n(vertical lines: selected version)")
ax[2].set(xlabel="epoch", ylabel="R² with the true rates", ylim=(0, 1), title="Distance from the ground truth (synthetic data only)")
for a_ in ax:
    a_.legend(fontsize=7)
fig.tight_layout(); fig.savefig(FIGS / "04_lfads_synthetic_training.png", bbox_inches="tight"); plt.show()
for v, sd in (("con CD", 0), ("senza CD", 0)):
    h_ = res_syn_lf[(v, sd)]["history"]
    b_ = res_syn_lf[(v, sd)]["best_epoch"]
    print(f"{v}, seed {sd}: at the selected version (epoch {b_}) error seen {h_.loc[b_, 'errore, pezzi visti']:.4f}, "
          f"validation {h_.loc[b_, 'errore, validazione']:.4f}; R² rates seen {h_.loc[b_, 'R² frequenze, pezzi visti']:.3f}, "
          f"validation {h_.loc[b_, 'R² frequenze, validazione']:.3f}; highest validation R² during training "
          f"{h_['R² frequenze, validazione'].max():.3f} (epoch {h_['R² frequenze, validazione'].idxmax()})")

# %% [markdown]
# **Reading.**
#
# - **Duration**: with coordinated dropout each training ran 880-960 epochs (version selected between
#   epoch 678 and 759), about 3 minutes; without it, training stopped at 354 epochs (version from epoch 154).
#   About 30 ms per step, measurements included, with 4 trainings in parallel.
# - **LFADS learns.** On the validation segments the error starts from that of a model that knows only the
#   mean rate (0.5351) and with coordinated dropout falls to 0.5080: it closes 92% of the distance to the perfect
#   model (0.5057). The rates reach an R² of 0.77 with the true ones.
# - **Seen segments versus new segments.** With coordinated dropout, on the seen segments the error reaches exactly
#   the level of the perfect model (0.4922 versus 0.4924); on the validation ones it stays slightly above: a
#   small adaptation to the segments it learns on, which is normal.
# - **Without coordinated dropout, what Keshtkaran & Pandarinath (2019) describe happens**: the error
#   falls **below** that of the perfect model, on the seen segments (0.4904) and even on the
#   validation ones (0.5052 versus 0.5057). No model can explain new spikes better than the true
#   rates: if it appears to, it is because it reads those same spikes and copies part of them. **The validation
#   error alone does not notice.**
# - **Yet here copying does no harm**: compared with the ground truth, the model without coordinated dropout is
#   better (validation rate R² 0.82 versus 0.77) and learns much faster (a few
#   tens of epochs versus about 300). The authors themselves warn that coordinated dropout can lead to
#   learning less when neurons are few, like our 30.
# - **The validation error as a judge**: with coordinated dropout the selected version (epoch 707)
#   is not the closest to the ground truth (epoch 429, R² 0.781 versus 0.766); the difference is small.
# - **In the figure**: on the left the cost that LFADS minimizes (with coordinated dropout it stays higher, because it
#   must guess values it does not see); in the middle the error as distance from the perfect model (the 0) and from
#   a model that knows only the mean (dotted lines): the dashed curves, without coordinated dropout,
#   fall below 0; on the right the distance from the ground truth, which with coordinated dropout rises more slowly.
# - **The consequence for §11**: coordinated dropout (0 or 0.3) goes into the grid, and the choice must be made
#   with a measure that copying cannot fool: the **hidden channels** of the validation trials (§9.4).
#
# ### 10.7 · What it learned: the Lorenz state and the rates, against GPFA and smoothing
#
# **The LFADS outputs over the whole recording.** Each segment (training, validation and test) goes through
# the trained model, and the segments are stitched together with the rule of §9.2. The `stitch` function of §9.2 works
# on the real trials; here we write the version for an arbitrary stretch, and check that on the real trials
# it gives the same result.
#
# **The comparison baselines, on equal terms**, estimated on the same 400 training segments:
#
# - **GPFA** with the same number of factors (10), on the square root of the counts, with the settings of 03 and
#   of §7b (500 iterations, initial time scale 100 ms), then the same segments stitched in the same way;
# - **smoothing**: the spikes smoothed with σ = 50 ms (the choice of 02_new), with no model at all.
#
# **The measures, on the 20 s of test**, which no method has seen:
#
# 1. **the Lorenz state**: a linear regression from the factors (or from the 30 smoothed neurons) to the 3
#    true variables, fitted on the 160 s of training and judged on the test (R² per variable). This is the measure of
#    Sussillo et al. (2016), Table 1;
# 2. **the rates**: how closely the estimated rates resemble the true ones, neuron by neuron, with two
#    measures: the **R²** (scale also counts: an estimate with the right shape but twice the size gives a low R²) and the
#    **correlation** (shape only). Each method gives the rates according to its **own model**:
#    - LFADS: its rates;
#    - GPFA: its model describes the **square root** of the counts, √k, as a prediction μ plus Gaussian
#      noise of variance R (one per neuron). The expected value of the count is therefore
#      E[k] = E[(√k)²] = μ² + R. Squaring the prediction alone, μ², would greatly underestimate
#      the rates, because with counts this small the square root "squashes" the values;
#    - smoothing: the smoothed spikes themselves.
#
#    We also tried a per-neuron Poisson regression (GLM), from the factors to the spikes, as the
#    Neural Latents Benchmark does for GPFA: here we discard it, because the regression learns to copy the
#    spikes it receives as input, and on the test the exponential makes the predictions explode (we show this
#    below).
#
# We do not measure bits per spike here: on the test segments each method reads the same spikes that it then
# "predicts", and the comparison against those spikes would reward the method that copies (§9.4). With the true rates this
# problem does not arise.

# %%
from sklearn.linear_model import PoissonRegressor           # Poisson regression (GLM): from the factors to the spikes of one neuron

def stitch_at(Y_seg, starts, a0, a1):                       # the stitching of §9.2 for an arbitrary stretch [a0, a1) with the segments starting at `starts`
    num, den = np.zeros((a1 - a0, Y_seg.shape[2])), np.zeros((a1 - a0, 1))
    for a, y, w in zip(np.asarray(starts) - a0, Y_seg, segment_weights(len(Y_seg))):
        num[a:a + SEG_LEN] += w[:, None] * y
        den[a:a + SEG_LEN] += w[:, None]
    return num / den

print("Stitching for an arbitrary stretch matches that of §9.2 on the real trials:",
      all(np.array_equal(stitch_at(SEGS[t], SEG_START[t], *TRIAL_SPAN[t]), stitch(SEGS[t], t)) for t in TRIALS))

def stitch_lz(per_part):                                    # stitches the outputs of the three parts (learn, validation, test) and puts them in sequence: 10 000 bins
    return np.concatenate([stitch_at(Y, LZ_START[k], *LZ_PARTS[k]) for k, Y in zip(LZ_PARTS, per_part)])

FIT_LZ, TEST_LZ = slice(*LZ_PARTS["learn"]), slice(*LZ_PARTS["test"])  # the bins on which the regressions are fitted, and the test ones

def gpfa_seqs(segs):                                        # segments of counts -> the elephant format, with the square root of the counts (as in 03 and §7b)
    return np.array([(s_.shape[0], np.sqrt(s_.astype(float)).T) for s_ in segs], dtype=[("T", int), ("y", "O")])

def gpfa_fit_segments(segs, x_dim):                         # GPFA with x_dim factors fitted on the given segments
    np.random.seed(0)                                       # fixed starting point
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        params, _ = gpfa_core.fit(gpfa_seqs(segs), x_dim=x_dim, bin_width=20.0, em_max_iters=500, tau_init=100.0)
    return params

def gpfa_sqrt_pred(segs, params):                           # the GPFA prediction μ = C x + d for the square root of the counts, in each segment: (segments, bins, neurons)
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        lat, _ = gpfa_core.exact_inference_with_ll(gpfa_seqs(segs), params, get_ll=False)
    return np.stack([(params["C"] @ x_ + params["d"][:, None]).T for x_ in lat["latent_variable"]])

def gpfa_infer_segments(segs, params):                      # the ordered factors (orthonormalize) of each segment: (segments, bins, factors)
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        lat, _ = gpfa_core.exact_inference_with_ll(gpfa_seqs(segs), params, get_ll=False)
        _, orth = gpfa_core.orthonormalize(params, lat)
    return np.stack([x_.T for x_ in orth["latent_variable_orth"]])

GP_LZ = gpfa_fit_segments(LZ_SEGS["learn"], SYN_HP["fac_dim"])  # GPFA on equal terms: same segments, 10 factors
F_GP_LZ = stitch_lz([gpfa_infer_segments(LZ_SEGS[k], GP_LZ) for k in LZ_PARTS])  # its factors over the whole recording
SM_LZ = np.concatenate([gaussian_filter1d(SPK_LZ[a0:a1].astype(float), sigma=50 / 20, axis=0, mode="nearest")  # smoothing σ = 50 ms (2.5 bins),
                        for a0, a1 in LZ_PARTS.values()])   # part by part, like the segments

def lorenz_r2(F):                                           # R² of the 3 Lorenz variables obtained linearly from F (fitted on training, judged on test)
    reg = LinearRegression().fit(F[FIT_LZ], LZ_Z[FIT_LZ])
    return r2_score(LZ_Z[TEST_LZ], reg.predict(F[TEST_LZ]), multioutput="raw_values")

def rate_r2(R):                                             # R² between estimated and true rates on the test, mean over the 30 neurons
    return float(np.mean(r2_score(RATE_LZ[TEST_LZ], R[TEST_LZ], multioutput="raw_values")))

def rate_corr(R):                                           # correlation between estimated and true rates on the test, mean over the 30 neurons
    return float(np.mean([np.corrcoef(RATE_LZ[TEST_LZ, n], R[TEST_LZ, n])[0, 1] for n in range(N_NEU)]))

def glm_rates(F):                                           # rates from a per-neuron Poisson regression, from the factors to the spikes, fitted on training
    sc = StandardScaler().fit(F[FIT_LZ])
    Fs = sc.transform(F)
    return np.column_stack([PoissonRegressor(alpha=1e-4, max_iter=1000).fit(Fs[FIT_LZ], SPK_LZ[FIT_LZ, n]).predict(Fs)
                            for n in range(N_NEU)])

F_LF_LZ = {run: stitch_lz(r["factors"]) for run, r in res_syn_lf.items()}  # the stitched LFADS factors, for each training
R_LF_LZ = {run: stitch_lz(r["rates"]) for run, r in res_syn_lf.items()}    # its stitched rates
R_GP_LZ = stitch_lz([gpfa_sqrt_pred(LZ_SEGS[k], GP_LZ) for k in LZ_PARTS]) ** 2 + np.diag(GP_LZ["R"])  # the GPFA rates: μ² + R

def lz_row(F, R):                                           # one row of the table: state (from F) and rates (R)
    return {**dict(zip("xyz", lorenz_r2(F))), "rates: R²": rate_r2(R), "rates: correlation": rate_corr(R)}

rows_lz = {"smoothing σ = 50 ms (30 neurons)": [lz_row(SM_LZ, SM_LZ)],
           "GPFA, 10 factors": [lz_row(F_GP_LZ, R_GP_LZ)]}
for v in ("con CD", "senza CD"):
    rows_lz[f"LFADS, 10 factors, {v}"] = [lz_row(F_LF_LZ[(v_, sd)], R_LF_LZ[(v_, sd)]) for v_, sd in SYN_RUNS if v_ == v]

def cell_fmt(vals):                                         # "mean (min-max)" if there are several seeds
    vals = np.array([x_ for x_ in vals if x_ is not None and not np.isnan(x_)])
    if len(vals) == 0:
        return ""
    return f"{vals.mean():.3f}" if len(vals) == 1 else f"{vals.mean():.3f} ({vals.min():.3f}-{vals.max():.3f})"

COLS_LZ = ["x", "y", "z", "rates: R²", "rates: correlation"]
tab_lz = pd.DataFrame({name: {c: cell_fmt([r_.get(c, np.nan) for r_ in runs]) for c in COLS_LZ} for name, runs in rows_lz.items()}).T
tab_lz.loc["Sussillo et al. 2016 (other data): GPFA"] = ["0.713", "0.725", "0.325", "", ""]
tab_lz.loc["Sussillo et al. 2016 (other data): LFADS"] = ["0.826", "0.900", "0.821", "", ""]
display(tab_lz.rename(columns={"x": "state: R² of x", "y": "state: R² of y", "z": "state: R² of z"}))
print(f"Mean rate on the test (spikes per bin): true {RATE_LZ[TEST_LZ].mean():.3f}; GPFA with μ² + R {R_GP_LZ[TEST_LZ].mean():.3f}, "
      f"with μ² only {(R_GP_LZ - np.diag(GP_LZ['R']))[TEST_LZ].mean():.3f}; LFADS (seed 0) {R_LF_LZ[('con CD', 0)][TEST_LZ].mean():.3f}")
glm_max = {name: glm_rates(F)[TEST_LZ].max() / io.BIN_S for name, F in (("smoothing", SM_LZ), ("GPFA", F_GP_LZ), ("LFADS (seme 0)", F_LF_LZ[("con CD", 0)]))}
print("Discarded alternative, the GLM from factors to spikes: maximum predicted rate on the test (spikes/s)",
      {k: round(v) for k, v in glm_max.items()}, f"| true maximum {RATE_LZ[TEST_LZ].max() / io.BIN_S:.0f}")

# %%
show_t = slice(LZ_PARTS["test"][0], LZ_PARTS["test"][0] + 300)  # the first 6 s of the test
t_s = np.arange(300) * io.BIN_S
reg_lf = LinearRegression().fit(F_LF_LZ[("con CD", 0)][FIT_LZ], LZ_Z[FIT_LZ])  # the linear maps fitted on training
reg_gp = LinearRegression().fit(F_GP_LZ[FIT_LZ], LZ_Z[FIT_LZ])
fig, ax = plt.subplots(1, 2, figsize=(16, 4.6), gridspec_kw={"width_ratios": (1.2, 1)})
for j, lab in enumerate("xyz"):
    off = 4 * (2 - j)
    ax[0].plot(t_s, LZ_Z[show_t, j] + off, color="k", lw=2, label="true" if j == 0 else None)
    ax[0].plot(t_s, reg_lf.predict(F_LF_LZ[("con CD", 0)][show_t])[:, j] + off, color="C3", lw=1.3, label="LFADS (seed 0)" if j == 0 else None)
    ax[0].plot(t_s, reg_gp.predict(F_GP_LZ[show_t])[:, j] + off, color="C0", lw=1.0, label="GPFA" if j == 0 else None)
    ax[0].text(-0.15, off, lab, fontsize=11, ha="right", va="center")
ax[0].set(xlabel="time from test start (s)", yticks=[], title="The Lorenz state on the test: true and obtained linearly from the factors")
ax[0].legend(fontsize=8, loc="upper right", ncol=3)
for n_, off in ((0, 0), (2, 1)):
    sc_ = RATE_LZ[:, n_].max() / io.BIN_S
    ax[1].plot(t_s, RATE_LZ[show_t, n_] / io.BIN_S / sc_ + off * 1.2, color="k", lw=2, label="true" if n_ == 0 else None)
    ax[1].plot(t_s, R_LF_LZ[("con CD", 0)][show_t, n_] / io.BIN_S / sc_ + off * 1.2, color="C3", lw=1.3, label="LFADS" if n_ == 0 else None)
    ax[1].plot(t_s, R_GP_LZ[show_t, n_] / io.BIN_S / sc_ + off * 1.2, color="C0", lw=1.0, label="GPFA (μ² + R)" if n_ == 0 else None)
    ax[1].plot(t_s, SM_LZ[show_t, n_] / io.BIN_S / sc_ + off * 1.2, color="0.6", lw=0.8, label="smoothed spikes (σ = 50 ms)" if n_ == 0 else None)
    ax[1].text(-0.15, off * 1.2 + 0.2, f"neuron {n_}", fontsize=9, ha="right", va="center")
ax[1].set(xlabel="time from test start (s)", yticks=[], title="The rates of 2 neurons on the test (scale of each neuron)")
ax[1].legend(fontsize=8, loc="upper right", ncol=4)
fig.tight_layout(); fig.savefig(FIGS / "04_lfads_synthetic_recovery.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# **Reading.**
#
# - **LFADS recovers the Lorenz state better than GPFA**: R² 0.96 versus 0.90 for x and 0.94 versus 0.85
#   for y, with all three seeds; for z the difference is small (0.90 versus 0.89, and the worst LFADS
#   seed, 0.888, is on par with GPFA). Smoothing is below both (0.80-0.86) even though it starts from 30
#   numbers instead of 10.
# - **GPFA does much better here than in the paper** (z: 0.89 versus 0.33). The data are different: a
#   system that is slower relative to the bins, a continuous recording, 10 factors. The large gap reported by
#   Sussillo et al. on z does not reproduce; the one on x and y does, but smaller.
# - **The rates**: LFADS reaches an R² of 0.82 (0.80-0.84) with the true ones, GPFA 0.51, and
#   smoothing a negative value (−4.9: the noise of the smoothed spikes is larger than the true
#   variations). Looking at shape only (correlation) the gap narrows: 0.93 versus 0.88. The figure
#   shows why: where its prediction μ is close to 0, the GPFA rates do not fall below the
#   noise level R, and remain almost flat (neuron 0).
# - **The formula matters**: squaring the GPFA prediction alone, the mean rate on the test
#   would be 0.043 spikes per bin instead of 0.199; with μ² + R it is 0.197, which is correct. LFADS gives 0.196.
# - **The discarded GLM**: from the smoothed spikes it predicts up to 1563 spikes/s on the test, from GPFA up to 354,
#   when the true rates do not exceed 155. It copies the input spikes, and the exponential amplifies.
# - **What it does not say**: these data are generated **exactly** under the LFADS assumptions (rates =
#   exponential of a linear combination, Poisson noise, fixed dynamics rules, no external
#   input). The example shows that LFADS works when its assumptions hold, not that it will win on
#   real data. A single synthetic example, non-optimized parameters, GPFA with 10 factors not chosen with
#   cross-validation.
#
# ### 10.8 · Checks
#
# Four checks, all fast:
#
# - **(a) The seams.** In the 200 ms shared by two segments, how much do the factors given by the segment
#   that ends and by the one that begins differ? And at which point of the segment is LFADS most precise? If the edges
#   were much worse than the centre, stitching with graded weights would matter all the more.
# - **(b) The hand computation with the trained weights**: the comparison of §10.3 used the initial, random weights.
#   We repeat it with the weights of the trained model (seed 0), on the test segments.
# - **(c) The seed**: two short trainings (15 epochs) with the same seed must be identical, bit for
#   bit; one with a different seed must differ. They are not saved to disk: they are redone at every run.
# - **(d) The timings**: milliseconds per training step on the CPU and on the Mac GPU (MPS), for the
#   shape of the synthetic data and for that of the real data (160 channels, 25 factors), one training at a
#   time.

# %%
R_lf_test = res_syn_lf[("con CD", 0)]["factors"][2]         # the factors of the 50 test segments, not stitched (seed 0)
R_gp_test = gpfa_infer_segments(LZ_SEGS["test"], GP_LZ)     # the same for GPFA
st_test = LZ_START["test"]

def seam_gap(Fseg, starts):                                 # in the zones shared by two consecutive segments: mean difference between their factors, in standard deviations
    sd_ = Fseg.reshape(-1, Fseg.shape[2]).std(0)
    gaps = []
    for i in range(len(starts) - 1):
        ov = starts[i] + SEG_LEN - starts[i + 1]            # how many bins they share
        if ov > 0:
            gaps.append(np.abs(Fseg[i, -ov:] - Fseg[i + 1, :ov]) / sd_)
    return float(np.mean(np.concatenate(gaps)))

print(f"(a) Zones shared by consecutive segments: the factors of the two segments differ on average by {seam_gap(R_lf_test, st_test):.2f} "
      f"standard deviations (LFADS) and by {seam_gap(R_gp_test, st_test):.2f} (GPFA)")
pos_r2 = [np.mean(r2_score(LZ_TRUE_SEGS["test"][:, p_], res_syn_lf[("con CD", 0)]["rates"][2][:, p_], multioutput="raw_values"))
          for p_ in range(SEG_LEN)]                         # R² of the LFADS rates, position by position within the segment
print("    R² of the LFADS rates by position in the segment (bins 1, 5, 10, 15, 20, 25, 30):",
      [round(float(pos_r2[p_]), 3) for p_ in (0, 4, 9, 14, 19, 24, 29)])
W_auto = segment_weights(len(st_test))                      # three ways of stitching the test rates: the AutoLFADS weights (§9.2)...
W_flat = np.ones_like(W_auto)                               # ...the simple mean...
W_cut = np.ones_like(W_auto); W_cut[1:, :SEG_OVL // 2] = 1e-12; W_cut[:-1, -SEG_OVL // 2:] = 1e-12  # ...and the hard cut in the middle of the shared zone

def stitch_test_w(Y_seg, W):                                # stitching of the test segments with given weights
    a0, a1 = LZ_PARTS["test"]
    num, den = np.zeros((a1 - a0, Y_seg.shape[2])), np.zeros((a1 - a0, 1))
    for a, y, w in zip(st_test - a0, Y_seg, W):
        num[a:a + SEG_LEN] += w[:, None] * y
        den[a:a + SEG_LEN] += w[:, None]
    return num / den

print("    R² of the stitched test rates:", {lab: round(float(np.mean(r2_score(RATE_LZ[TEST_LZ], stitch_test_w(res_syn_lf[("con CD", 0)]["rates"][2], W_),
                                                                                 multioutput="raw_values"))), 3)
                                               for lab, W_ in (("AutoLFADS weights", W_auto), ("simple mean", W_flat), ("cut in the middle", W_cut))})

W_tr = res_syn_lf[("con CD", 0)]["weights"]                 # (b) the trained weights, as numpy arrays
MINE_tr = lfads_forward_np(LZ_SEGS["test"].astype(np.float64), {k: v.astype(np.float64) for k, v in W_tr.items()}, SYN_HP)
print(f"(b) Trained weights, 50 test segments: maximum difference hand − library in the factors "
      f"{np.abs(MINE_tr['factors'] - R_lf_test).max():.1e} (values up to {np.abs(R_lf_test).max():.2f}), in the rates "
      f"{np.abs(np.exp(MINE_tr['log_rates']) - res_syn_lf[('con CD', 0)]['rates'][2]).max():.1e}")

short = [POOL_LF.submit(lfads_fit, **dict(syn_job(SYN_HP, sd), max_epochs=15, watch=None)) for sd in (0, 0, 1)]  # (c) seed 0 twice, seed 1 once
short = [f_.result() for f_ in short]
same_w = all(np.array_equal(short[0]["weights"][k], short[1]["weights"][k]) for k in short[0]["weights"])
diff_w = max(np.abs(short[0]["weights"][k] - short[2]["weights"][k]).max() for k in short[0]["weights"] if short[0]["weights"][k].size)  # (the weights of the switched-off controller have 0 elements)
print(f"(c) Seed 0 twice: weights identical bit for bit {same_w}, cost identical at every epoch "
      f"{[h_['costo di training'] for h_ in short[0]['history']] == [h_['costo di training'] for h_ in short[1]['history']]}; "
      f"seed 1: maximum difference in the weights {diff_w:.2f}")

def lfads_step_time(hp, device, n_steps=30, n_seg=256, seed=0):
    """Milliseconds per training step (median after 5 warm-up steps), on fake data with the shape of hp."""
    import time
    import numpy as np                                      # numpy BEFORE torch
    import torch
    from lfads_torch.tuples import SessionBatch
    torch.set_num_threads(1); torch.manual_seed(seed)
    model = make_lfads(hp)
    model.__class__ = type("LFADSconEpoch", (type(model),), {"current_epoch": 0})
    model = model.to(device); model.train()
    model.log = model.log_dict = lambda *a, **k: None
    opt = model.configure_optimizers()
    X = torch.tensor(np.random.default_rng(seed).poisson(0.2, size=(n_seg, hp["seg_len"], hp["n_out"])), dtype=torch.float32, device=device)
    times = []
    for s_ in range(n_steps):
        idx = torch.randint(0, n_seg, (64,), device=device)
        sb = SessionBatch(X[idx][..., :hp["n_in"]], X[idx], torch.zeros(64, hp["seg_len"], 0, device=device), torch.zeros(0), torch.ones_like(X[idx]))
        t0 = time.time()
        loss = model._shared_step({0: (sb, ())}, s_, "train")
        opt.zero_grad(); loss.backward(); opt.step()
        if device == "mps":
            torch.mps.synchronize()                         # on the GPU the computations are asynchronous: wait for them to finish
        times.append(time.time() - t0)
    return float(np.median(times[5:]) * 1000)

REAL_SHAPE = dict(SYN_HP, n_in=N_CH, n_out=N_CH, fac_dim=25, kl_ic_scale=1 / (SEG_LEN * N_CH))  # the shape of the real data
tab_time = pd.DataFrame({name: {dev: POOL_LF.submit(lfads_step_time, hp_, dev).result() for dev in ("cpu", "mps")}  # one at a time
                         for name, hp_ in (("synthetic data (30 neurons, 10 factors)", SYN_HP),
                                           ("real data shape (160 channels, 25 factors)", REAL_SHAPE))}).T
tab_time["GPU / CPU"] = tab_time["mps"] / tab_time["cpu"]
display(tab_time.rename(columns={"cpu": "CPU (ms per step)", "mps": "Mac GPU, MPS (ms per step)"}).round(1))

# %% [markdown]
# **Reading.**
#
# - **(a) The seams matter less than expected, but the segment edges are weak.** In the shared zone the
#   factors of the two segments differ on average by 0.41 standard deviations (GPFA: 0.35): the two segments do not
#   agree, especially near the edges. Within the segment LFADS is precise at the centre (R² 0.83-0.87)
#   and worse at the edges: 0.74 at the last bin, **0.44 at the first**. The AutoLFADS weights give little weight precisely to
#   the start of the new segment, but compared with the simple mean or a hard cut the difference is minimal
#   (R² 0.840, 0.833, 0.835). **For the real data**: the first segment of each trial has no segment before it, so
#   roughly the first 100 ms of each trial (the tail of the preceding `Release`, §1.5) will be less reliable.
# - **(b) The hand computation matches the library even with the trained weights**: maximum difference
#   1.6·10⁻⁶ in the factors (which reach 2.2) and 6·10⁻⁷ in the rates.
# - **(c) The seed works**: two trainings with the same seed give weights identical bit for bit and the same
#   cost at every epoch; with a different seed the weights differ. This holds on this computer, with these versions
#   (torch 2.7.1, CPU); this is why the models are saved to disk.
# - **(d) The CPU is faster than the GPU**, by 4 to 7 times. The exact milliseconds change at every run
#   and with the load on the computer (those of the last run are in the table; across all the runs made they
#   were between 20 and 60 ms per step on the CPU and between 140 and 260 on the GPU, the highest with the Mac in
#   energy-saving mode). For the real data: about 400 segments per block, hence about 5 minutes
#   per training, 4 at a time.
#
# ### Summary of §10
#
# | question | answer | where |
# |---|---|---|
# | Does the hand-written LFADS match the library? | yes: 10⁻⁷ with the initial weights, 10⁻⁶ with the trained ones; the comparison detects a change of 0.001 in a single weight | §10.3, §10.8 |
# | does the hand-computed cost match? | yes, in all three terms; the KL weight 1/(bins × neurons) gives the true ELBO; the library approximates log k! with Stirling (does not change the training) | §10.4 |
# | Does LFADS recover the hidden state? | yes: R² 0.96, 0.94, 0.90 (GPFA 0.90, 0.85, 0.89; smoothing 0.82, 0.80, 0.86) | §10.7 |
# | and the rates? | R² 0.82 versus 0.51 for GPFA (correlation 0.93 versus 0.88) | §10.7 |
# | is coordinated dropout needed? | not here: without it, LFADS is closer to the ground truth; but the validation error falls below the perfect model (copying), so it is not a reliable judge | §10.6 |
# | the segment edges? | the first bin of a segment is much worse than the centre (R² 0.44 versus 0.83-0.87); stitching with weights helps little | §10.8 |
# | reproducibility and timings | same seed = same model; CPU 4-7 times faster than GPU; about 3 minutes per synthetic model | §10.6, §10.8 |
#
# **What we carry over to the real data**: the `SYN_HP` parameters as a starting point, coordinated
# dropout (0 or 0.3) and the KL weight in the grid of §11, selection with the hidden channels of the
# validation trials, the CPU with 4 trainings in parallel.
#
# **What we did not verify**: a single synthetic system, built with the LFADS assumptions;
# non-optimized parameters; GPFA with 10 factors fixed by us; copying without coordinated dropout we
# saw in the error, not measured directly.

# %% [markdown]
# ## 11 · LFADS on the real data: a first training
#
# Before measuring anything with the common yardstick (§12), we look at LFADS at work on the real data,
# with **a single configuration**, in **block 0** (test trials 3, 8 and 14, which are not examined here).
# §11 is divided into two parts:
#
# - **§11a (this one)**: a first training. Is it learning or memorizing? What do the
#   reconstructed rates look like? Can it generate credible fake trials?
# - **§11b**: the small parameter grid, chosen on the validation trials only.
#
# ### 11.1 · The starting configuration, and two forms of the model
#
# **The parameters** are those of the synthetic example (`SYN_HP`, §10.3), with three changes due to the data:
#
# | parameter | synthetic | real data | why |
# |---|---|---|---|
# | channels | 30 | 160 | the active channels (§9.1) |
# | factors | 10 | **25** | the same number as GPFA (02_new, §10.2): the equal-terms comparison of §12 |
# | weight of the KL term | 1 / (30 × 30) | 1 / (30 bins × 160 channels) | always the true ELBO (§10.4) |
#
# Coordinated dropout 0.3, dropout 0.05, encoder, initial condition and generator of 64: as in §10.
# In §10b LFADS did better without coordinated dropout, but its validation error did not see
# the copying; we therefore start from the protected version, and the choice is made by the grid of §11b.
#
# **Two forms of the same model**:
#
# - **all channels**: the encoder reads the 160 channels and the model reconstructs all of them. This is the form we
#   will use for the common yardstick (§12);
# - **120 visible + 40 hidden** (§9.4): the encoder reads only the 120 visible channels, but the model reconstructs
#   all 160. During training the 40 hidden channels serve only as "answers" to guess; in the validation and test trials
#   the model never sees them. The **bits per spike on the hidden channels** are the
#   measure that the copying of §10.6 cannot fool: a channel that the encoder does not read cannot be
#   copied.
#
# **Two validation pairs**, as for CEBRA (§3.2): 12 and 15 (the rule of §9.3), and 6 and 7, in the middle of the
# session, with a different seed. In all cases LFADS learns on the other 10 training trials of block 0.
#
# **What we measure at every epoch**, on two groups of trials of the same shape (two trials, one even and one
# odd): trials **4 and 5**, which the model sees (as in §3.2), and the 2 **validation** trials. In
# **bits per spike**, that is how much better the model explains the spikes than a model that knows only the mean rate
# of each channel (§9.4): on the channels that the encoder reads and, for the second form, on the hidden channels.
#
# The outputs for the test trials are computed and saved for §12, but are not examined here.
#
# 4 trainings (2 forms × 2 validation pairs), one per process: about 7 minutes the first time.

# %%
REAL_HP = dict(SYN_HP, n_in=N_CH, n_out=N_CH, fac_dim=K_PCA,  # 160 channels, 25 factors like GPFA
               kl_ic_scale=1 / (SEG_LEN * N_CH))            # the KL weight that gives the true ELBO: 1 / (30 bins × 160 channels)
HID_HP = dict(REAL_HP, n_in=len(HELD_IN))                   # the encoder reads only the 120 visible channels; all 160 are reconstructed
ORDER_HID = np.r_[HELD_IN, HIDDEN]                          # the channels reordered: first the 120 visible, then the 40 hidden
SEGS_HID = {t: SEGS[t][..., ORDER_HID] for t in TRIALS}     # the segments with the channels in this order
HID_COLS = list(range(len(HELD_IN), N_CH))                  # the columns of the hidden channels after reordering: 120-159

def real_job(hp, seed, val_trials, segs, hidden=None, k=0):  # one training on the real data in block k: learns on the training trials except val_trials
    fit = [int(t) for t in TRIALS if t not in TEST_TRIALS[k] and t not in val_trials]  # the 10 trials it learns on
    hid = [] if hidden is None else [hidden]                # the columns hidden from the encoder (if any)
    return dict(hp=hp, seed=seed, train=np.concatenate([segs[t] for t in fit]),
                val=np.concatenate([segs[t] for t in val_trials]),
                apply=[segs[t] for t in TRIALS],             # the outputs of all 15 trials (the test ones are needed for §12)
                watch={"trial visti (4, 5)": (np.concatenate([segs[t] for t in CHECK_TRIALS]), None, *hid),
                       "validation": (np.concatenate([segs[t] for t in val_trials]), None, *hid)})

FORMS = {"tutti i canali": (REAL_HP, SEGS, None), "120 visibili + 40 nascosti": (HID_HP, SEGS_HID, HID_COLS)}
RUNS_11 = [(form, vt, sd) for vt, sd in ((VAL_TRIALS, 0), (VAL_TRIALS_B, 1)) for form in FORMS]  # (form, validation pair, seed)
res11 = dict(zip(RUNS_11, run_lfads_cached([real_job(FORMS[f][0], sd, vt, FORMS[f][1], FORMS[f][2]) for f, vt, sd in RUNS_11])))
n_par_real = sum(v.size for k_, v in res11[RUNS_11[0]]["weights"].items() if not k_.endswith("bias_hh"))
print(f"Model parameters (all channels): {n_par_real}; segments it learns on: {len(real_job(REAL_HP, 0, VAL_TRIALS, SEGS)['train'])}")

rows11 = {}
for (f, vt, sd), r in res11.items():
    h_, b_ = r["history"], r["best_epoch"]
    row = {"epochs": r["epochs"], "best epoch": b_, "duration (minutes)": round(r["seconds"] / 60, 1)}
    for g in ("trial visti (4, 5)", "validation"):         # the bits per spike at the selected version
        if f == "tutti i canali":
            row[f"bit/spike, {g}"] = round(h_.loc[b_, f"bit per spike, {g}"], 3)
        else:
            row[f"bit/spike, {g}"] = round(h_.loc[b_, f"bit per spike visibili, {g}"], 3)
            row[f"bit/spike nascosti, {g}"] = round(h_.loc[b_, f"bit per spike nascosti, {g}"], 3)
    rows11[f"{f}; validation {vt[0]}-{vt[1]} (seed {sd})"] = row
display(pd.DataFrame(rows11).T.fillna(""))

# %%
fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
for j, (vt, ls) in enumerate(((VAL_TRIALS, "-"), (VAL_TRIALS_B, "--"))):
    sd = 0 if vt == VAL_TRIALS else 1
    h_, b_ = res11[("tutti i canali", vt, sd)]["history"], res11[("tutti i canali", vt, sd)]["best_epoch"]
    for g, c in (("trial visti (4, 5)", "C0"), ("validation", "C3")):  # left: the "all channels" form
        ax[0].plot(h_.index, h_[f"bit per spike, {g}"], ls, color=c, lw=1, label=f"{g}, val. {vt[0]}-{vt[1]}")
    ax[0].axvline(b_, color="0.5", ls=ls, lw=0.8)
    h_, b_ = res11[("120 visibili + 40 nascosti", vt, sd)]["history"], res11[("120 visibili + 40 nascosti", vt, sd)]["best_epoch"]
    for g, c in (("trial visti (4, 5)", "C0"), ("validation", "C3")):  # middle and right: visible and hidden of the second form
        ax[1].plot(h_.index, h_[f"bit per spike visibili, {g}"], ls, color=c, lw=1, label=f"{g}, val. {vt[0]}-{vt[1]}")
        ax[2].plot(h_.index, h_[f"bit per spike nascosti, {g}"], ls, color=c, lw=1, label=f"{g}, val. {vt[0]}-{vt[1]}")
    for a_ in ax[1:]:
        a_.axvline(b_, color="0.5", ls=ls, lw=0.8)
for a_, ttl in zip(ax, ("All channels: bits per spike\n(the encoder reads the same channels)",
                        "120 visible + 40 hidden:\nthe 120 channels that the encoder reads",
                        "120 visible + 40 hidden:\nthe 40 hidden channels (copying is impossible)")):
    a_.axhline(0, color="k", lw=0.8)
    a_.set(xlabel="epoch", ylabel="bits per spike (0 = mean rate only)", title=ttl, ylim=(-0.02, 0.06))  # the first epochs, very negative, stay out
    a_.legend(fontsize=7, loc="lower right")
fig.tight_layout(); fig.savefig(FIGS / "04_lfads_real_training.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# **Reading.**
#
# - **LFADS learns something, but little.** On the validation trials, at the selected version, it explains the spikes
#   better than the mean rate alone by 0.015-0.028 bits per spike on the channels it reads, and by 0.011-0.015
#   on the hidden channels. In the synthetic data the perfect model reached 0.22 (§10.1). Values this low
#   are not necessarily a defect of the model: in 02_new (§9.2) the fraction of variability that a channel
#   shares with the population was a few percentage points, and only that part can be predicted.
# - **On the hidden channels, where copying is impossible, the model memorizes**: on the seen trials (4 and 5) the bits
#   per spike keep rising up to 0.03-0.046, on the validation trials they stop at about
#   0.01-0.015 already after 100-200 epochs. The gap, 2-3 times, is what the model learns about the trials
#   it trains on and does not hold for new ones. It is the same phenomenon found for CEBRA (§3.2).
# - **The stopping rule looks at the reconstruction error**, and selects late versions (epochs 237-929;
#   with all channels and the 12-15 pair it did not stop before the maximum of 1000). On the hidden channels,
#   beyond 200 epochs nothing more is gained: training longer only serves to memorize.
# - **Reading a channel helps little**: on the validation trials the bits per spike of the channels that the encoder reads
#   (0.015-0.028) are only slightly above those of the hidden channels.
# - **The 12-15 pair, at the end of the session, does worse than 6-7** on all measures (for example, hidden
#   0.011 versus 0.015). For CEBRA the two pairs were equivalent (§3.2); here the difference is there, but is
#   confounded with the seed (0 versus 1). This is the doubt of §9.3 about validation at the end of the session.
#
# ### A baseline without a model
#
# How many bits per spike are "many" on the hidden channels? The Neural Latents Benchmark uses as its
# simplest reference **smoothing** (Pei et al. 2021): the spikes of the visible channels are smoothed and a
# Poisson regression (GLM) for each hidden channel, fitted on the trials used for learning, predicts them
# on the new trials. Here the GLM cannot copy anything (we discarded it in §10.7 precisely for this
# reason): the hidden channels are not among its inputs. We use the 120 visible channels smoothed with
# σ = 50 ms, the same 10 trials as LFADS, and three values of the regression penalty (0.001, 0.1 and 1).
#
# To compare on the **same bins**, we recompute the LFADS bits per spike here on the **stitched** rates
# of the validation trials: during training they were computed on the segments, where the bins of the
# shared zones count twice.

# %%
def bits_per_spike(rates, X):                               # bits per spike: gain over the mean rate of each channel, in these data (§9.4)
    k, lam = X.reshape(-1, X.shape[-1]), rates.reshape(-1, X.shape[-1])
    lam0 = np.maximum(k.mean(0), 1e-9)                      # the mean rate of each channel (never exactly 0)
    return float((np.sum(lam0 - k * np.log(lam0)) - np.sum(lam - k * np.log(lam))) / k.sum() / np.log(2))

SM_VIS = io.smooth_rates(s.spikes, sigma_ms=50)[:, ACTIVE][:, HELD_IN]  # the 120 visible channels smoothed (σ = 50 ms)

def smooth_glm_rates(fit_trials, alpha, C=None):            # rates of the hidden channels from a Poisson regression on the smoothed visible ones, fitted on the trials fit_trials
    C = COUNTS if C is None else C                          # the counts to predict (in §13.4 the time-shifted ones)
    fit_ = np.isin(trial, fit_trials)
    F = StandardScaler().fit(SM_VIS[fit_]).transform(SM_VIS)
    return np.column_stack([PoissonRegressor(alpha=alpha, max_iter=500).fit(F[fit_], C[fit_, h]).predict(F) for h in HIDDEN])

rows_ref = {}
for vt, sd in ((VAL_TRIALS, 0), (VAL_TRIALS_B, 1)):
    val_ = np.isin(trial, vt)                               # the bins of the 2 validation trials
    fit_tr = [int(t) for t in TRIALS if t not in TEST_TRIALS[0] and t not in vt]
    r_ = res11[("120 visibili + 40 nascosti", vt, sd)]["rates"]
    R_hid = np.concatenate([stitch(r_[list(TRIALS).index(t)], t) for t in vt])[:, HID_COLS]  # LFADS: stitched rates of the hidden channels
    col = {"LFADS (120 + 40)": bits_per_spike(R_hid, COUNTS[val_][:, HIDDEN])}
    for alpha in (1e-3, 1e-1, 1.0):
        col[f"smoothing + GLM, penalità {alpha:g}"] = bits_per_spike(smooth_glm_rates(fit_tr, alpha)[val_], COUNTS[val_][:, HIDDEN])
    rows_ref[f"validazione {vt[0]}-{vt[1]}"] = col
display(pd.DataFrame(rows_ref).round(4).rename_axis("bits per spike, 40 hidden channels, validation trials"))

# %% [markdown]
# **Reading.**
#
# - **The first LFADS achieves about half of the simplest reference.** On the same bins as the
#   validation trials, on the hidden channels: LFADS 0.011 and 0.018 bits per spike, smoothing with the GLM 0.028 and
#   0.033 (penalty 0.1; 0.027 and 0.032 with penalty 1). With too small a penalty (0.001) even the
#   GLM memorizes and falls to −0.002 and 0.010.
# - **It is not yet a verdict on LFADS**: it is a single configuration, with the parameters of the synthetic
#   example, and the curves above show that it memorizes. But it sets the bar for §11b: the grid
#   must find a configuration that exceeds 0.028-0.033 bits per spike on the hidden channels of the validation
#   trials, otherwise LFADS is not worth its complexity on these data.
# - Stitching, the LFADS bits per spike on trials 12-15 remain those of the segments (0.011); on trials 6-7
#   they rise from 0.015 to 0.018 (in the segments the bins of the shared zones count twice).
# ### 11.2 · The reconstructed rates, next to the smoothed ones
#
# One validation trial (12), 8 seconds. For three channels that the encoder reads, the LFADS rate
# ("all channels" form) next to the firing rate smoothed with σ = 50 ms, the one used by part A. For
# one **hidden** channel, the rate that LFADS derives without ever having read it ("120 + 40" form). We choose the three
# channels with a rule: the most modulated (highest standard deviation of the smoothed firing rate),
# the one in 20th position and the one in 60th; the hidden channel is the most modulated among the hidden ones.
#
# Note: the smoothed firing rate **is not the ground truth**, as the true rate was in §10. It is a
# noisy estimate, which in §10.7 had a correlation of only 0.51 with the true rates.

# %%
R11_all = {t: stitch(r_, t) for t, r_ in zip(TRIALS, res11[("tutti i canali", VAL_TRIALS, 0)]["rates"])}  # stitched rates, trial by trial
R11_hid = {t: stitch(r_, t) for t, r_ in zip(TRIALS, res11[("120 visibili + 40 nascosti", VAL_TRIALS, 0)]["rates"])}
SM_RATE = io.smooth_rates(s.spikes, sigma_ms=50)[:, ACTIVE]  # smoothed firing rate (σ = 50 ms) of the 160 channels, in spikes per second (not z-scored)
mod_rank = np.argsort(SM_RATE.std(0))[::-1]                 # channels from most to least modulated
show_ch = [mod_rank[0], mod_rank[19], mod_rank[59]]         # the three visible channels shown
hid_show = HIDDEN[np.argmax(SM_RATE[:, HIDDEN].std(0))]     # the most modulated hidden channel
t_val = VAL_TRIALS[0]                                       # trial 12
a0_, a1_ = TRIAL_SPAN[t_val]
win_ = np.arange(a0_ + 150, a0_ + 550)                      # 8 s, from the third second of the trial
t_w = (win_ - a0_) * io.BIN_S

fig, axes = plt.subplots(5, 1, figsize=(14, 9), sharex=True, gridspec_kw={"height_ratios": (0.35, 1, 1, 1, 1)})
for b_ in win_:                                             # top: the task phases
    axes[0].axvspan((b_ - a0_) * io.BIN_S, (b_ - a0_ + 1) * io.BIN_S, color=io.PHASE_COLORS.get(phase[b_], "white"), lw=0)
axes[0].set(yticks=[], title=f"Validation trial {t_val}: band = task phase")
ph_here = [p for p in ["Presentation", "Intertrial", *io.MOVEMENT_PHASES] if p in set(phase[win_])]
axes[0].legend([plt.Rectangle((0, 0), 1, 1, color=io.PHASE_COLORS[p]) for p in ph_here], ph_here, ncol=len(ph_here),
               fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.05), frameon=False)
panels11 = [(ch, R11_all[t_val][:, ch], "LFADS, all channels", "C3", "visible") for ch in show_ch]
panels11.append((hid_show, R11_hid[t_val][:, int(np.flatnonzero(ORDER_HID == hid_show)[0])], "LFADS, never having read it", "C0", "NASCOSTO"))
for a_, (ch, r_lf, lab, c, kind) in zip(axes[1:], panels11):
    spk = COUNTS[win_, ch]
    a_.vlines(t_w[spk > 0], 0, spk[spk > 0] / io.BIN_S * 0.15, color="0.75", lw=0.8)  # the spikes (height ∝ count)
    a_.plot(t_w, SM_RATE[win_, ch], color="0.3", lw=1.2, label="smoothed, σ = 50 ms")
    a_.plot(t_w, r_lf[win_ - a0_] / io.BIN_S, color=c, lw=1.6, label=lab)
    a_.set(ylabel="spike/s", title=f"channel {ch} ({kind}; mean rate {rate_hz[ch]:.0f} Hz)")
    a_.legend(fontsize=7, loc="upper right", ncol=2)
axes[-1].set_xlabel(f"seconds from the start of trial {t_val}")
fig.tight_layout(); fig.savefig(FIGS / "04_lfads_real_rates.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# **Reading.**
#
# - **The LFADS rates follow only the slow changes.** In the most modulated channel (35, 51 Hz on
#   average) they rise from about 55 to about 110 spikes/s during `Orient` and `Shape`, like the smoothed
#   firing rate; in channels 99 and 142 they stay almost flat, while the smoothed firing rate oscillates a lot.
#   Hidden channel 102, which the model never read, also has a rate that changes slowly
#   (from about 35 to about 50 spikes/s).
# - **The model "decides" that almost all rapid oscillations are noise.** With σ = 50 ms and about one
#   spike per bin the smoothed firing rate is very noisy (in §10.7 it had a correlation of only 0.51
#   with the true rates); in addition it contains the 18 Hz rhythm of §11.3. Who is right, without a ground truth,
#   is told only by the hidden channels: and there the slow LFADS rates explain about 0.01 bits per spike.
#
# ### 11.3 · LFADS generates fake trials
#
# This is the test that CEBRA cannot do (§2): a generative model can **produce new data**. The procedure,
# with the second half of our hand-written LFADS (§10.2): an initial condition is drawn at random from the
# **prior** distribution (learned mean, variance 0.1), the generator unrolls it for 30 bins, the factors
# give the rates, and Poisson spikes are drawn from these. No real data enter the procedure.
#
# We compare 400 fake segments with the 400 real segments the model learned on ("all channels" form,
# validation 12-15), and with a third version: the spikes drawn from the rates that the model assigns
# to the real segments (the initial condition this time comes from the encoder, that is "posterior"). Four
# properties:
#
# 1. the **mean rate of each channel**;
# 2. the **correlations between pairs of channels**, on spikes summed in 100 ms windows;
# 3. the **distribution of the spikes of the whole population** in a bin;
# 4. the **autocorrelation over time** of the population spikes: how much the current state resembles
#    the one a few bins later. This is the property that depends on the dynamics rules.

# %%
def lfads_sample_prior(n, w, hp, seed=0):
    """Fake segments: initial condition drawn from the prior distribution, then the second half of lfads_forward_np
    (generator, factors, rates), then Poisson spikes. Returns (rates, spikes)."""
    rng_ = np.random.default_rng(seed)
    ic = w["ic_prior.mean"] + np.exp(0.5 * w["ic_prior.logvar"]) * rng_.normal(size=(n, hp["ic_dim"]))  # from the prior distribution
    g = ic @ w["decoder.ic_to_g0.weight"].T + w["decoder.ic_to_g0.bias"]  # the starting state of the generator
    W_fac = w["decoder.rnn.cell.fac_linear.weight"]
    W_fac = W_fac / np.linalg.norm(W_fac, axis=1, keepdims=True)  # normalized factor directions, as in the library
    lam = []
    for _ in range(hp["seg_len"]):                          # 30 bins, only the generator rules
        g = gru_step(np.zeros((n, 0)), g, w, "decoder.rnn.cell.gen_cell.", hp["cell_clip"])
        lam.append(np.exp((g @ W_fac.T) @ w["readout.0.weight"].T + w["readout.0.bias"]))  # factors -> rates
    lam = np.stack(lam, 1)
    return lam, rng_.poisson(lam).astype(np.float32)

W11 = {k_: v.astype(np.float64) for k_, v in res11[("tutti i canali", VAL_TRIALS, 0)]["weights"].items()}
REAL11 = real_job(REAL_HP, 0, VAL_TRIALS, SEGS)["train"]    # the 400 real segments it learned on
fit11 = [int(t) for t in TRIALS if t not in TEST_TRIALS[0] and t not in VAL_TRIALS]
POST11 = np.concatenate([res11[("tutti i canali", VAL_TRIALS, 0)]["rates"][list(TRIALS).index(t)] for t in fit11])  # the model rates on the real segments
_, FAKE11 = lfads_sample_prior(len(REAL11), W11, REAL_HP, seed=0)  # 400 fake segments
POSTSPK11 = np.random.default_rng(1).poisson(POST11).astype(np.float32)  # spikes drawn from the "posterior" rates
SETS11 = {"true": REAL11, "fake (prior)": FAKE11, "from posterior rates": POSTSPK11}

def pair_corr(X):                                           # correlations between pairs of channels, on spikes summed in windows of 5 bins (100 ms) within the segments
    Y = X.reshape(len(X), SEG_LEN // 5, 5, X.shape[2]).sum(2).reshape(-1, X.shape[2])
    return np.corrcoef(Y.T)[np.triu_indices(X.shape[2], 1)]

def pop_autocorr(X, lags=range(1, 11)):                     # autocorrelation over time of the population spikes, within the segments
    P = X.sum(2)                                            # spikes of the whole population, bin by bin: (segments, 30)
    P = P - P.mean()
    return np.array([np.sum(P[:, :-l] * P[:, l:]) / np.sum(P[:, :-l] ** 2) for l in lags])

stats11 = {name: {"rates": X.mean((0, 1)) / io.BIN_S, "corr": pair_corr(X), "pop": X.sum(2).ravel(), "ac": pop_autocorr(X)}
           for name, X in SETS11.items()}
display(pd.DataFrame({name: {"spikes per bin, population (mean)": st_["pop"].mean(),
                             "same, standard deviation": st_["pop"].std(),
                             "correlation with the true: channel rates": np.corrcoef(stats11["true"]["rates"], st_["rates"])[0, 1],
                             "correlation with the true: pair correlations": np.corrcoef(stats11["true"]["corr"], st_["corr"])[0, 1],
                             "mean correlation between pairs": st_["corr"].mean(),
                             "autocorrelation at 1 bin": st_["ac"][0], "autocorrelation at 10 bins (200 ms)": st_["ac"][-1]}
                      for name, st_ in stats11.items()}).round(3))

# %%
fig, ax = plt.subplots(1, 4, figsize=(17, 3.9))
cols11 = {"true": "k", "fake (prior)": "C3", "from posterior rates": "C0"}
for name in ("fake (prior)", "from posterior rates"):
    ax[0].scatter(stats11["true"]["rates"], stats11[name]["rates"], s=8, alpha=0.6, color=cols11[name], label=name)
    ax[1].scatter(stats11["true"]["corr"], stats11[name]["corr"], s=2, alpha=0.15, color=cols11[name], label=name)
lim0 = (0, max(stats11["true"]["rates"].max(), stats11["fake (prior)"]["rates"].max()) * 1.05)
ax[0].plot(lim0, lim0, color="0.5", lw=0.8); ax[0].set(xlim=lim0, ylim=lim0, xlabel="real segments (Hz)", ylabel="other segments (Hz)",
                                                        title="Mean rate of each channel")
ax[1].plot((-0.2, 0.6), (-0.2, 0.6), color="0.5", lw=0.8); ax[1].set(xlabel="real segments", ylabel="other segments",
                                                                  title="Correlation between pairs of channels (100 ms)")
bins_pop = np.arange(0, max(st_["pop"].max() for st_ in stats11.values()) + 2) - 0.5
for name, st_ in stats11.items():
    ax[2].hist(st_["pop"], bins=bins_pop, histtype="step", density=True, color=cols11[name], lw=1.5, label=name)
    ax[3].plot(np.arange(1, 11) * 20, st_["ac"], "o-", color=cols11[name], ms=4, label=name)
ax[2].set(xlabel="spikes of the whole population in one bin", ylabel="fraction of bins", title="Distribution of the population spikes")
ax[3].set(xlabel="lag (ms)", ylabel="autocorrelazione", title="How much the population 'remembers' over time")
for a_ in ax:
    a_.legend(fontsize=7)
fig.tight_layout(); fig.savefig(FIGS / "04_lfads_generated.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# **Reading.**
#
# - **LFADS generates data with the right level for each channel**: the mean rates of the fake segments
#   match the true ones (correlation 0.999), and the population emits on average the same number of
#   spikes per bin (44.9 versus 45.8).
# - **But the real population fluctuates much more**: standard deviation 10.3 spikes per bin versus 7.1 for the
#   fake ones. If the spikes were only Poisson noise around fixed rates it would be about 6.8: the real
#   data have a lot more shared variability, LFADS almost none.
# - **The correlations between pairs of channels are only partly reproduced**: between real and fake they
#   resemble each other with a correlation of 0.47, and the mean correlation between pairs is 0.007 in the real data, 0.002 in the
#   fake.
# - **The real autocorrelation changes sign** from one lag to the next (−0.08 at 20 ms, then high values at
#   60 and 100 ms), that of the fake data is flat around 0.11-0.17: the model has slow shared fluctuations,
#   but not the oscillation. We see it in the cell below.
# - **The fake data resemble the model reconstructions** (the spikes drawn from the "posterior" rates
#   have almost the same properties): what is missing does not depend on the prior distribution, it is missing
#   from the model itself.
#
# ### An 18 Hz rhythm in the counts
#
# The autocorrelation of the real segments (figure above, on the right) changes sign from one lag to the next: this is the
# sign of an **oscillation**. In the previous notebooks it could not emerge: smoothing with σ = 50 ms
# almost completely removes frequencies above 10 Hz. We characterize it with the **spectrum** of the spikes of the
# whole population (how much variance there is at each frequency):
#
# - with the **exact spike times**, counted at 1 ms: with 20 ms bins no frequencies above
#   25 Hz are visible, and a higher frequency could "disguise itself" as a lower one. We measure the mean power in
#   a few bands, relative to that between 60 and 90 Hz, and look for narrow lines above 35 Hz;
# - separately in the **pauses** and in the **movement**, on 1 s windows (50 bins, one every 200 ms) that
#   fall entirely inside a pause or entirely inside a movement phase, averaging the spectra of the
#   windows (Welch's method). Beta oscillations of the motor cortex (13-30 Hz) usually decrease
#   during movement, even when only attempted or imagined (Kilavik et al. 2013, review): if this rhythm
#   were of that kind, it should be weaker during movement.

# %%
P_pop = COUNTS.sum(1).astype(float)                         # spikes of the whole population in each 20 ms bin
st_pop = np.concatenate([s.spike_times[u] for u in ACTIVE]) # the times of all the spikes of the 160 channels
h1, _ = np.histogram(st_pop, bins=np.arange(s.times[0], s.times[-1] + io.BIN_S, 0.001))  # counted at 1 ms
f1 = np.fft.rfftfreq(len(h1), d=0.001)
S1 = np.abs(np.fft.rfft(h1 - h1.mean())) ** 2              # the spectrum at 1 ms
ref_band = (f1 > 60) & (f1 < 90)                            # reference band
k_peak = np.argmax(S1 * (f1 > 5))                           # the frequency with most power above 5 Hz
print(f"Spectrum at 1 ms: peak at {f1[k_peak]:.2f} Hz")
print("Mean power per band, relative to the 60-90 Hz band:",
      {f"{lo}-{hi} Hz": round(float(S1[(f1 >= lo) & (f1 < hi)].mean() / S1[ref_band].mean()), 2)
       for lo, hi in ((10, 15), (16, 20), (21, 26), (30, 40), (45, 55))})
k_hi = np.flatnonzero(f1 > 35)[np.argmax(S1[f1 > 35])]      # the highest line above 35 Hz
print(f"Above 35 Hz the highest line is at {f1[k_hi]:.1f} Hz ({S1[k_hi] / np.median(S1[np.abs(f1 - f1[k_hi]) < 2]):.0f} times the ±2 Hz around it); "
      f"20 ms counting lets through a fraction {np.sinc(f1[k_hi] * io.BIN_S) ** 2:.1e} of the power")

def welch_pop(mask, L=50, step=10):                         # mean spectrum of the population spikes over windows of L bins all inside `mask`
    starts = [a for a in range(0, n_bins - L, step) if mask[a:a + L].all()]
    win = np.hanning(L)                                     # a window that vanishes at the edges (attenuates spurious frequencies)
    S = np.mean([np.abs(np.fft.rfft((P_pop[a:a + L] - P_pop[a:a + L].mean()) * win)) ** 2 for a in starts], axis=0)
    return np.fft.rfftfreq(L, io.BIN_S), S, len(starts)

WELCH = {"pause": welch_pop(~MOV), "movement": welch_pop(MOV)}
f_w = WELCH["pause"][0]
k18 = np.argmin(np.abs(f_w - 18))                           # the Welch grid frequency closest to 18 Hz
side = (np.abs(f_w - 18) > 3) & (f_w > 8)                   # the comparison frequencies: above 8 Hz and more than 3 Hz away from the peak
for name, (_, S_, n_) in WELCH.items():
    print(f"{name}: {n_} windows; power at {f_w[k18]:.0f} Hz = {S_[k18] / np.median(S_[side]):.1f} times the typical power above 8 Hz; "
          f"absolute power at {f_w[k18]:.0f} Hz {S_[k18]:.0f}")
print(f"Smoothing with σ = 50 ms lets through at 18 Hz a fraction {np.exp(-(2 * np.pi * 18 * 0.05) ** 2):.0e} of the power")

fig, ax = plt.subplots(1, 2, figsize=(13, 3.8))
for name, c in (("pause", "0.4"), ("movement", "C3")):
    f_, S_, n_ = WELCH[name]
    ax[0].plot(f_, S_ / np.median(S_[side]), "o-", ms=3, color=c, label=f"{name} ({n_} windows of 1 s)")
ax[0].set(xlabel="frequency (Hz)", ylabel="power / typical power above 8 Hz", xlim=(0, 25), yscale="log",
          title="Spectrum of the population spikes (20 ms bins)")
ax[0].legend(fontsize=8)
zoom = (f1 > 5) & (f1 < 40)
ax[1].plot(f1[zoom], uniform_filter1d(S1, 25)[zoom] / S1[ref_band].mean(), color="k", lw=1)  # moving average over 25 neighboring frequencies (0.1 Hz)
ax[1].set(xlabel="frequency (Hz)", ylabel="power / mean power between 60 and 90 Hz", title="With the exact spike times (1 ms)")
fig.tight_layout(); fig.savefig(FIGS / "04_lfads_beta.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# **Reading.**
#
# - **There is a real rhythm around 18 Hz.** With the exact spike times the peak is at 18.1 Hz, so
#   it is not a higher frequency "in disguise". It is broad, not a thin line: in the 16-20 Hz band the
#   power is 6.2 times that between 60 and 90 Hz, between 21 and 26 Hz 2.1 times, below 15 Hz the same as at 60-90 Hz.
#   It falls in the beta band (13-30 Hz).
# - **It does not decrease during movement**: relative to the neighboring frequencies it is 5.7 times in the pauses and 5.8 in
#   movement, and in absolute value too it is almost the same. Motor beta oscillations usually
#   decrease during movement, even when only attempted (Kilavik et al. 2013); here they do not. So **we do not know
#   where it comes from**: a physiological rhythm unrelated to movement, or a technical source. We did not
#   investigate it.
# - A second, narrow line is at 265.6 Hz (18 times its surroundings): probably technical; 20 ms counting
#   lets through only 0.25% of it, so it does not matter.
# - **For the previous notebooks nothing changes**: smoothing with σ = 50 ms lets through at 18 Hz a
#   fraction 10⁻¹⁴ of the power. **For LFADS and GPFA it does**: they receive the raw counts, rhythm included,
#   and neither of them reproduces it. It is a part of the shared variability of the population that, apparently,
#   has nothing to do with the task phases (the same level in pause and in movement); we did not
#   verify whether it carries information about the task.
#
# ### 11.4 · The segment edges, on the real data
#
# In §10.8 the first bin of each segment was much less precise than the centre. On the real data there is no ground truth,
# but there are the hidden channels: we measure the bits per spike of the 40 hidden channels **position by
# position** within the segments of the 2 validation trials ("120 + 40" form), and the disagreement between consecutive
# segments in the shared zone ("all channels" form).

# %%
r_hid = res11[("120 visibili + 40 nascosti", VAL_TRIALS, 0)]["rates"]
X_val_hid = np.concatenate([SEGS_HID[t] for t in VAL_TRIALS])[..., HID_COLS]  # the real spikes of the hidden channels, validation segments
R_val_hid = np.concatenate([r_hid[list(TRIALS).index(t)] for t in VAL_TRIALS])[..., HID_COLS]  # the LFADS rates for those channels
print(f"Hidden channels, validation trials: {bits_per_spike(R_val_hid, X_val_hid):.3f} bits per spike on the segments "
      f"(the training history, selected epoch: "
      f"{res11[('120 visibili + 40 nascosti', VAL_TRIALS, 0)]['history'].loc[res11[('120 visibili + 40 nascosti', VAL_TRIALS, 0)]['best_epoch'], 'bit per spike nascosti, validazione']:.3f})")
pos_bits = [bits_per_spike(R_val_hid[:, p_:p_ + 3], X_val_hid[:, p_:p_ + 3]) for p_ in range(0, SEG_LEN, 3)]  # in groups of 3 bins (60 ms)
print("Bits per spike of the hidden channels by position in the segment (groups of 3 bins: 1-3, 4-6, ..., 28-30):", [round(b_, 3) for b_ in pos_bits])
F_val = [res11[("tutti i canali", VAL_TRIALS, 0)]["factors"][list(TRIALS).index(t)] for t in VAL_TRIALS]
print(f"Zones shared by consecutive segments, validation trials: the factors differ on average by "
      f"{np.mean([seam_gap(F_, SEG_START[t]) for F_, t in zip(F_val, VAL_TRIALS)]):.2f} standard deviations (synthetic: 0.41)")

# %% [markdown]
# **Reading.**
#
# - **Position by position, on the real data the signal is too weak to see the edges**: the bits per
#   spike of the hidden channels range from −0.006 (the first 3 bins of the segment) to 0.011, and oscillate with no
#   clear trend. The first group is the lowest, as in the synthetic data (§10.8), but the difference is
#   within the noise.
# - **Consecutive segments disagree more than in the synthetic data**: in the shared zone the factors
#   differ on average by 0.65 standard deviations (0.41 in the synthetic data). The stitched trajectories may
#   have "joins": in §12 we will check how much they weigh on the roughness.
#
# ### Summary of §11a
#
# | question | answer | where |
# |---|---|---|
# | Does LFADS learn from the real data? | little: 0.011-0.015 bits per spike on the hidden channels of the validation trials | §11.1 |
# | does it memorize? | yes: on the seen trials the hidden ones reach 0.03-0.046; the gap opens after 100-200 epochs | §11.1 |
# | is it better than the reference without a model? | no: smoothing with the GLM gives 0.028-0.033, about double | §11.1 |
# | what rates does it reconstruct? | slow ones: it follows the variations linked to the phases, treats the rest as noise | §11.2 |
# | does it generate credible data? | the level of each channel yes; the shared variability and the autocorrelation no | §11.3 |
# | what is in the counts that the model does not see? | a rhythm at about 18 Hz, the same in pause and in movement, of unclear origin | §11.3 |
#
# **What we carry over to §11b.** The configuration of the synthetic example is **not enough**: LFADS memorizes
# and stays below the simplest reference. The grid must therefore explore more regularization (dropout,
# KL weight, smaller generator) and coordinated dropout, choosing with the **bits per spike on the hidden
# channels** of the validation trials, with the smoothing + GLM bar.
#
# **What we did not verify**: a single configuration and two seeds; the difference between the two
# validation pairs is confounded with the seed; the origin of the 18 Hz rhythm and whether it carries information about the task.

# %% [markdown]
# ## 11b · The parameter grid, on the validation trials only
#
# In §11a LFADS memorized and stayed below the simplest reference. Here we try to correct it with
# more regularization, changing the same "knobs" that AutoLFADS searches by itself (`scripts/run_pbt.py`
# of the library): the dropout, the coordinated dropout and the weight of the KL term. Since we cannot run
# AutoLFADS (§9), we use a **small grid**:
#
# | knob | values | AutoLFADS range | what it does |
# |---|---|---|---|
# | dropout | 0.05 (§11a), 0.25, 0.5 | 0-0.6 | randomly switches off a part of the numbers inside the network at every step: the network cannot rely on a single detail |
# | weight of the KL term | × 1 (the true ELBO, §10.4), × 5 | 10⁻⁶-10⁻³ (× 5 here is 1.04·10⁻³) | makes the information passing through the initial condition cost more |
# | coordinated dropout | 0, 0.3 (§11a) | 0.01-0.7 | hides a part of the values from the encoder and measures the error only on those (§10.5) |
#
# The generator size (64), the L2 penalty (10⁻²) and the learning rate
# (4·10⁻³) stay fixed: they are among the alternatives of §14.
#
# **The scheme** is that of §11a: "120 visible + 40 hidden" form, block 0, the two validation
# pairs (12-15 with seed 0, 6-7 with seed 1). 12 configurations × 2 pairs = 24 trainings; the
# configuration of §11a is already on disk. Of the new models we save only the rates of the hidden channels.
#
# **The rule, fixed before looking at the results.** For each training we take the curve of the bits per
# spike on the hidden channels of the validation trials (recorded at every epoch), smooth it with a moving
# average over 21 epochs (the measurements of a single epoch are noisy, §11a) and read its **maximum** and
# the epoch at which it falls. The **score** of a configuration is the mean of the two maxima (one per pair).
# The highest score wins. The **number of epochs** that we will use in §12 is the mean of the two peak
# epochs of the winning configuration, rounded up to a multiple of 50: as for CEBRA (§3.2),
# the number of steps is chosen on validation and then stays fixed.
#
# **Why not the reconstruction error**: it is what the training stopping rule uses, but
# §10.6 showed that without coordinated dropout copying fools it; the hidden channels do not.
#
# **The bar**: smoothing with Poisson regression, 0.028 and 0.033 bits per spike on the two pairs
# (§11a). It should be remembered that its penalty was also the best of the three tried on the same trials, and that
# the maximum of a noisy curve is a little optimistic: the definitive measure, on the test trials, is in §12.
#
# 22 new trainings, 4 at a time: about 30-40 minutes the first time.

# %%
GRID_DROP, GRID_KL, GRID_CD = (0.05, 0.25, 0.5), (1, 5), (0.0, 0.3)  # the grid values
CENTER = (0.05, 1, 0.3)                                     # the configuration of §11a

def grid_hp(drop, kl, cd):                                  # the parameters of a grid configuration ("120 + 40" form)
    return dict(HID_HP, dropout=drop, kl_ic_scale=kl / (SEG_LEN * N_CH), cd_rate=cd)

GRID = list(product(GRID_DROP, GRID_KL, GRID_CD))           # 12 configurations
PAIRS_11 = ((VAL_TRIALS, 0), (VAL_TRIALS_B, 1))             # the two validation pairs, with the seeds of §11a
jobs_g, keys_g = [], []
for cfg in GRID:
    for vt, sd in PAIRS_11:
        j_ = real_job(grid_hp(*cfg), sd, vt, SEGS_HID, HID_COLS)  # as in §11a
        if cfg != CENTER:
            j_["rate_cols"] = HID_COLS                      # of the rates we keep only the 40 hidden ones (the §11a configuration is already on disk with all of them)
        jobs_g.append(j_); keys_g.append((cfg, vt))
print("The §11a configuration coincides with the grid one:", grid_hp(*CENTER) == HID_HP)
res_g = dict(zip(keys_g, run_lfads_cached(jobs_g)))

SMOOTH_EP = 21                                              # the moving average over epochs

def smooth_curve(h, col):                                   # a column of the history, smoothed over 21 epochs
    return uniform_filter1d(h[col].values, SMOOTH_EP, mode="nearest")

def grid_row(cfg):                                          # the measures of a configuration: maximum of the smoothed curve, epoch, seen trials, for each pair
    row, vals, eps = {}, [], []
    for vt, _ in PAIRS_11:
        r_ = res_g[(cfg, vt)]
        h_ = r_["history"]
        v_ = smooth_curve(h_, "bit per spike nascosti, validazione")
        i_ = int(np.argmax(v_))                             # the epoch of the maximum (position in the history)
        lab = f"{vt[0]}-{vt[1]}"
        row[f"hidden, val. {lab}"] = v_[i_]
        row[f"epoch, val. {lab}"] = int(h_.index[i_])
        row[f"seen (4, 5), val. {lab}"] = smooth_curve(h_, "bit per spike nascosti, trial visti (4, 5)")[i_]
        row[f"epochs run, val. {lab}"] = r_["epochs"]
        vals.append(v_[i_]); eps.append(int(h_.index[i_]))
    row["score"] = np.mean(vals)
    row["peak epoch (mean)"] = np.mean(eps)
    return row

def cfg_name(c):                                            # the name of a configuration
    return f"dropout {c[0]}, KL × {c[1]}, CD {c[2]}"

def grid_table(cfgs):                                       # the table of a group of configurations, from the highest score
    t_ = pd.DataFrame({cfg_name(c): grid_row(c) for c in cfgs}).T
    return t_.sort_values("score", ascending=False)

REF_GLM = {f"{vt[0]}-{vt[1]}": rows_ref[f"validazione {vt[0]}-{vt[1]}"]["smoothing + GLM, penalità 0.1"] for vt, _ in PAIRS_11}
REF_GLM_MEAN = np.mean(list(REF_GLM.values()))              # the bar: smoothing + GLM from §11a, mean of the two pairs
tab_g1 = grid_table(GRID)
display(tab_g1.round(4))
best1 = max(GRID, key=lambda c: grid_row(c)["score"])
print(f"First round: the best is {cfg_name(best1)} (score {grid_row(best1)['score']:.4f}); "
      f"bar {REF_GLM} -> mean {REF_GLM_MEAN:.4f}; above the bar: {int((tab_g1['score'] > REF_GLM_MEAN).sum())} of {len(GRID)}")
print("Configurations that learn nothing (score below 0):", int((tab_g1["score"] < 0).sum()), "su", len(GRID))

# %% [markdown]
# **Reading.**
#
# - **More regularization does not help: it makes the model stop learning.** 9 configurations out of 12 have a score
#   below 0. On the 12-15 pair they all stop at around −0.015 bits per spike, that is at the level of a model that
#   always predicts the mean rate of the training trials: the encoder no longer passes any information
#   to the generator. This happens with KL weight × 5 and with dropout at 0.25-0.5. It is the phenomenon known in
#   variational autoencoders as *collapse*: if every piece of information passed costs too much, it pays not to
#   pass any, like a student who, if every word costs too much, hands in a blank
#   sheet. In some configurations one pair learns and the other does not: this too says that we are at the limit.
# - **Coordinated dropout makes it worse**: with dropout 0.05 and KL × 1, 0.026 without versus 0.011 with
#   coordinated dropout (the §11a configuration). It is the same outcome as in the synthetic data (§10.6), and here
#   it does not depend on copying: the hidden channels cannot be copied.
# - **The best, dropout 0.05, KL × 1, without coordinated dropout, is on the edge of the grid** for all
#   three knobs, on the side of the lowest regularization. It gives 0.025 and 0.026, below the bar
#   (0.028 and 0.033). Contrary to the expectation of §11a: the model memorizes, but the classic remedy, more
#   regularization, switches it off instead of correcting it.
#
# ### The second round, towards less regularization
#
# We decided on this round **after** seeing the first, and we state it: the best configuration
# is on the edge of the grid, on the side of the lowest regularization for all three knobs, and a
# maximum on the edge does not say whether going beyond would do better. We therefore extend in that direction:
#
# - coordinated dropout fixed at 0, because 0.3 is worse at every level of the other two knobs;
# - dropout 0, 0.02 (the starting value of the library configuration for MC_Maze-S) and 0.05;
# - KL weight × 0.3 and × 1.
#
# 5 new configurations × 2 pairs = 10 trainings, about 12-15 minutes. The rule stays the same and applies
# to all 17 configurations together. If the winner were still on the edge of the KL weight, we will
# state it and leave it to §14.

# %%
GRID2 = [(0.0, 0.3, 0.0), (0.0, 1, 0.0), (0.02, 0.3, 0.0), (0.02, 1, 0.0), (0.05, 0.3, 0.0)]  # the new configurations
jobs_g2, keys_g2 = [], []
for cfg in GRID2:
    for vt, sd in PAIRS_11:
        j_ = real_job(grid_hp(*cfg), sd, vt, SEGS_HID, HID_COLS)
        j_["rate_cols"] = HID_COLS
        jobs_g2.append(j_); keys_g2.append((cfg, vt))
res_g.update(zip(keys_g2, run_lfads_cached(jobs_g2)))
GRID_ALL = GRID + GRID2                                     # the 17 configurations
tab_g = grid_table(GRID_ALL)
display(tab_g.round(4))
BEST_CFG = max(GRID_ALL, key=lambda c: grid_row(c)["score"])  # the winning configuration
N_EP_LF = int(np.ceil(grid_row(BEST_CFG)["peak epoch (mean)"] / 50) * 50)  # the number of epochs for §12
print(f"Selected configuration: {cfg_name(BEST_CFG)}; score {grid_row(BEST_CFG)['score']:.4f} "
      f"(bar {REF_GLM_MEAN:.4f}); number of epochs for §12: N_EP_LF = {N_EP_LF}")
print(f"Configurations above the bar: {int((tab_g['score'] > REF_GLM_MEAN).sum())} of {len(GRID_ALL)}")

# %%
fig, ax = plt.subplots(1, 2, figsize=(16, 5.6), gridspec_kw={"width_ratios": (1.15, 1)})
order_g = list(tab_g.index)[::-1]                           # from the lowest score (bottom) to the highest (top)
for yi, name in enumerate(order_g):
    cfg = [c for c in GRID_ALL if cfg_name(c) == name][0]
    row = grid_row(cfg)
    v12, v67 = row["hidden, val. 12-15"], row["hidden, val. 6-7"]
    col = "C3" if cfg == BEST_CFG else ("0.3" if cfg == CENTER else ("C1" if cfg in GRID2 else "C0"))
    ax[0].plot([v12, v67], [yi, yi], color=col, lw=1.5, alpha=0.6)
    ax[0].scatter([v12], [yi], color=col, marker="<", s=30); ax[0].scatter([v67], [yi], color=col, marker=">", s=30)
    ax[0].scatter(row["score"], yi, color=col, s=70, edgecolor="k", zorder=3)
ax[0].axvspan(min(REF_GLM.values()), max(REF_GLM.values()), color="C2", alpha=0.2, label="smoothing + GLM (the two pairs)")
ax[0].axvline(0, color="k", lw=0.8)
ax[0].set(yticks=range(len(order_g)), yticklabels=order_g, xlabel="bits per spike, hidden channels, validation (maximum of the smoothed curve)",
          title="◀ pair 12-15, ▶ pair 6-7, ● mean\n(red: selected; grey: §11a; orange: second round)")
ax[0].legend(fontsize=8, loc="lower right")
for cfg, c in ((BEST_CFG, "C3"), (CENTER, "0.3")):
    for (vt, _), ls in zip(PAIRS_11, ("-", "--")):
        h_ = res_g[(cfg, vt)]["history"]
        ax[1].plot(h_.index, smooth_curve(h_, "bit per spike nascosti, validazione"), ls, color=c, lw=1.5,
                   label=f"{'choice' if cfg == BEST_CFG else '§11a'}, validation {vt[0]}-{vt[1]}")
        ax[1].plot(h_.index, smooth_curve(h_, "bit per spike nascosti, trial visti (4, 5)"), ls, color=c, lw=0.8, alpha=0.5)
ax[1].axhspan(min(REF_GLM.values()), max(REF_GLM.values()), color="C2", alpha=0.2)
ax[1].axvline(N_EP_LF, color="k", ls=":", lw=1, label=f"epochs for §12: {N_EP_LF}")
ax[1].set(xlabel="epoch", ylabel="bits per spike, hidden channels (smoothed over 21 epochs)", ylim=(-0.01, 0.08),
          title="Curves of the selected configuration and of §11a\n(thin lines: seen trials 4 and 5)")
ax[1].legend(fontsize=7, loc="upper left")
fig.tight_layout(); fig.savefig(FIGS / "04_lfads_grid.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# **Reading.**
#
# - **The selected configuration: dropout 0.05, KL weight × 0.3, without coordinated dropout.** On the hidden
#   channels of the validation trials it gives 0.026 (pair 12-15) and 0.028 (pair 6-7), score 0.0270. The
#   two peak epochs are 103 and 257: **N_EP_LF = 200**.
# - **The first six configurations are in practice equivalent** (scores from 0.022 to 0.027): all without
#   coordinated dropout, with dropout up to 0.05 and KL weight up to × 1. The differences among them are
#   of the order of the difference between the two pairs of the same configuration, with a single seed per
#   pair: the choice among these six is not significant. What counts is the group: no coordinated dropout, little
#   regularization.
# - **The dropout has its maximum in the interior** (0.05 better than 0, 0.02 and 0.25). **The KL weight does not**: ×
#   0.3 is only slightly better than × 1 (0.0270 versus 0.0259), and is the lowest value tried. As declared
#   before looking, even lower values are left for §14.
# - **None of the 17 configurations exceeds smoothing with the regression** (0.0305 on average): the best
#   stays below by 11%. On the 12-15 pair the distance is small (0.026 versus 0.028), on 6-7 it is not (0.028
#   versus 0.033).
# - **Relative to §11a the gain is large** (from 0.011 to 0.027), and comes mainly from removing
#   coordinated dropout.
# - **The model still memorizes a lot**: on the seen trials the hidden channels reach 0.068-0.092, three times
#   the validation. In the figure on the right the validation curves rise up to about 100 epochs, stay
#   flat until about 250 and then decline slowly, while those of the seen trials keep rising. Regularization
#   does not close this gap; what remains to try is more data (the other two sessions, §14) and
#   the "how many trials are needed" curve (§13).
# - **Limits**: the choice is made on the same trials on which we measure (the maximum of a noisy curve is a
#   little optimistic; this also holds for the GLM penalty); a single seed per pair and only block 0. The real
#   measure, on the test trials and on 5 blocks, is in §12.
#
# ### Summary of §11
#
# **What we carry over to §12:**
#
# | choice | value | from where |
# |---|---|---|
# | encoder, initial condition, generator | 64 | §10.3 (the sizes are among the alternatives of §14) |
# | factors | 25 | as for GPFA (§11.1) |
# | dropout | 0.05 | grid, maximum in the interior (§11b) |
# | coordinated dropout | 0 | grid: with 0.3 always worse (§11b); same outcome in the synthetic data (§10.6) |
# | KL weight | 0.3 / (30 × 160) = 6.25·10⁻⁵ | grid, on the edge: lower values in §14 (§11b) |
# | L2 on the generator, learning rate | 10⁻², 4·10⁻³ | §10.3 (among the alternatives of §14) |
# | epochs | **200, fixed** | the peak of the hidden channels on validation (§11b); in §12 the stopping rule on the reconstruction error is not used, as it selected late versions (§11a) |
# | two forms | all channels (common yardstick); 120 + 40 (hidden channels) | §11.1 |
# | bar for the hidden channels | smoothing + Poisson regression | §11a |
#
# **What we learned about LFADS with these data**: it learns little (about 0.027 bits per spike on the hidden
# channels) and memorizes a lot; strong regularization makes it collapse, coordinated dropout makes it
# worse; with the best configuration found it stays below the simplest reference on the validation
# trials. The counts contain a rhythm at about 18 Hz that the model does not reproduce (§11.3).
#
# **What we did not verify**: that the configuration chosen on the "120 + 40" form is the best
# also for the all-channels form (we assume it); that 200 epochs are also right in the other
# blocks; KL weight values below × 0.3; more seeds per configuration.

# %% [markdown]
# ## 12 · LFADS with the common yardstick, and the equal-terms comparison with GPFA
#
# Now the real measurement, on the **test trials** of the 5 blocks of 03. In each block LFADS is trained on the **12
# training trials** and the 3 test trials go through the already trained model, without retraining it. With the
# fixed epochs (§11b) the validation trials are no longer needed for stopping: like GPFA, CEBRA and all
# the other methods, LFADS learns on all 12 training trials.
#
# §12 is divided into two parts:
#
# - **§12a (this one)**: phase decoding with the common yardstick and the single-trial trajectories;
# - **§12b**: the measure proper to generative models, the hidden channels of the test trials.
#
# ### 12.1 · The three models on equal terms
#
# Three models of the same family, starting from the same counts of the 12 training trials and giving
# 25 factors per bin:
#
# | model | what it adds | data | from where |
# |---|---|---|---|
# | **FA** | separates the shared signal from the noise of each channel; no model of time | square root of the counts, all bins of the 12 trials | 02_new, §9 and §10.4 ("GPFA without time") |
# | **GPFA on equal terms** | + factors that change gradually in time | square root of the counts, **the same 600 ms segments as LFADS** | 02_new, §10 |
# | **LFADS** | + learned nonlinear dynamics rules, and exponential link to the rates | counts, the same segments | §11b |
#
# **GPFA on equal terms is fitted on whole segments.** `elephant`, during fitting, splits each sequence into
# segments of 20 bins (02_new, §10.1): with our 30-bin segments it would make two overlapping 20-bin ones,
# and GPFA would look at 400 ms at a time while LFADS sees 600. We therefore use whole
# segments, as in the test of 02_new (§10.6), where fitting on whole windows gave a better model
# on never-seen data. It costs the same as the other (measured: about 2.5 minutes per fit). For the rest, the
# settings of 02_new: 25 factors, 500 iterations, initial time scale 100 ms. The fits are
# saved to disk (`cache_04/gpfa`), like the LFADS models.
#
# **GPFA "as in §7b"** stays as a fourth row: the one fitted on the 1 s windows from the start
# of the phases, that is the GPFA of the numbers of 03 (but here on the 1106 points).
#
# **LFADS**: the configuration chosen in §11b (dropout 0.05, KL weight × 0.3, without coordinated
# dropout), "all channels" form, **200 epochs**, 5 seeds. The training loop (§10.5) keeps the
# version with the lowest error on a control set; here we give it the training segments themselves as the control set,
# whose error falls throughout the training, and so the version kept is in
# practice the last one. We check this below. (Changing the loop to always keep the last version
# would have changed the fingerprint, and therefore required retraining, all the models already saved.)
#
# **The ordered factors.** For the comparison "with 2-3 coordinates" we take the first factors **ordered by
# importance**, as for GPFA (`orthonormalize`, 02_new §10.5) and for FA (02_new §9.4): the
# matrix that goes from the factors to the output (for LFADS from the factors to the log-rates) is decomposed with the SVD, and the
# factors are rotated along its directions, from the strongest. The "ceiling" is the classifier on all 25.
#
# **For the trajectories** we need, as for CEBRA (§4.2), models fitted on **all 15 trials**: the
# factors of different blocks have different orientations and cannot be mixed. They describe, they do not
# measure generalization.
#
# 30 LFADS trainings (25 per block + 5 on all trials) and 6 GPFA fits: about 40 minutes the
# first time (measured with the Mac in energy-saving mode: 33 minutes for LFADS, 4 trainings at a time,
# and 5 for GPFA, 3 at a time).

# %%
from sklearn.decomposition import FactorAnalysis            # FA (02_new, §9)
import pickle                                               # to save the fitted GPFAs to disk

LF_HP = dict(REAL_HP, dropout=BEST_CFG[0], kl_ic_scale=BEST_CFG[1] / (SEG_LEN * N_CH), cd_rate=BEST_CFG[2])  # the §11b configuration, all channels
SEEDS_LF = range(5)                                         # 5 seeds, as for CEBRA (§4)
KEEP_CH = [int(mod_rank[0])]                                # of the rates we save only one channel (the most modulated, §11.2): here the factors are needed

def train_trials(k):                                        # the 12 training trials of block k (k = "all": all 15)
    return [int(t) for t in TRIALS] if k == "all" else [int(t) for t in TRIALS if t not in TEST_TRIALS[k]]

def lf12_job(seed, trs, hp=None, segs=None, rate_cols=None):  # LFADS on trials trs for N_EP_LF epochs; projects all 15 trials
    tr_ = np.concatenate([(SEGS if segs is None else segs)[t] for t in trs])
    return dict(hp=LF_HP if hp is None else hp, seed=seed, train=tr_, val=tr_,  # control set = the training segments (in practice the last version is kept)
                apply=[(SEGS if segs is None else segs)[t] for t in TRIALS], max_epochs=N_EP_LF, patience=N_EP_LF,
                rate_cols=KEEP_CH if rate_cols is None else rate_cols)

RUNS12 = [(k, sd) for k in list(range(5)) + ["all"] for sd in SEEDS_LF]  # (block, seed): 25 per block + 5 on all trials
res12 = dict(zip(RUNS12, run_lfads_cached([lf12_job(sd, train_trials(k)) for k, sd in RUNS12])))

def order_by_readout(F, W):                                 # factors ordered by importance: SVD of the output matrix W (channels × factors)
    U_, S_, Vt_ = np.linalg.svd(W, full_matrices=False)
    return F @ Vt_.T * S_                                   # coordinates along the directions of W, from the strongest (like orthonormalize of GPFA)

def stitch_all(per_trial):                                  # stitches the outputs of all 15 trials and puts them in sequence: 12233 bins
    return np.concatenate([stitch(y_, t) for t, y_ in zip(TRIALS, per_trial)])

F_LF12 = {run: order_by_readout(stitch_all(r["factors"]), r["weights"]["readout.0.weight"]) for run, r in res12.items()}
ep_kept = np.array([r["best_epoch"] for r in res12.values()])
print(f"LFADS: {len(res12)} models; version kept: epoch {ep_kept.min()}-{ep_kept.max()} of {N_EP_LF} "
      f"(equal to {N_EP_LF} in {int((ep_kept == N_EP_LF).sum())} cases); mean duration {np.mean([r['seconds'] for r in res12.values()]) / 60:.1f} minutes")

# %%
CACHE_GP = CACHE / "gpfa"                                   # the fitted GPFAs, on disk
CACHE_GP.mkdir(parents=True, exist_ok=True)
GP_ITERS = 500                                              # fitting iterations (02_new, §10.1)

def gpfa_fit_whole(segs, x_dim, iters):
    """GPFA on whole segments (without the elephant 20-bin cut, as in 02_new §10.6). Runs in a separate process."""
    import contextlib, os                                   # to silence the elephant messages
    import numpy as np
    from elephant.gpfa import gpfa_core
    cut_orig = gpfa_core.gpfa_util.cut_trials               # the function that cuts the segments...
    gpfa_core.gpfa_util.cut_trials = lambda seqs, seg_length=np.inf: cut_orig(seqs, seg_length=np.inf)  # ...replaced by one that does not cut
    try:
        np.random.seed(0)                                   # fixed starting point
        seqs = np.array([(s_.shape[0], np.sqrt(s_.astype(float)).T) for s_ in segs], dtype=[("T", int), ("y", "O")])  # square root of the counts
        with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
            params, _ = gpfa_core.fit(seqs, x_dim=x_dim, bin_width=20.0, em_max_iters=iters, tau_init=100.0)
    finally:
        gpfa_core.gpfa_util.cut_trials = cut_orig           # in any case the original function is restored
    return params

def gp_key(segs, x_dim, iters):                             # fingerprint of a GPFA fit: code, settings, data
    h = hashlib.sha1(inspect.getsource(gpfa_fit_whole).encode())
    h.update(json.dumps([x_dim, iters]).encode()); h.update(np.ascontiguousarray(segs).tobytes())
    return h.hexdigest()[:16]

def gpfa_cached(seg_sets, x_dim=K_PCA, iters=GP_ITERS):     # the GPFAs of several sets of segments: reloaded from disk or fitted 3 at a time
    keys = [gp_key(sg, x_dim, iters) for sg in seg_sets]
    todo = [i for i, k_ in enumerate(keys) if RETRAIN or not (CACHE_GP / f"{k_}.pkl").exists()]
    fits = Parallel(n_jobs=3)(delayed(gpfa_fit_whole)(seg_sets[i], x_dim, iters) for i in todo)
    for i, p_ in zip(todo, fits):
        with open(CACHE_GP / f"{keys[i]}.pkl", "wb") as f_:
            pickle.dump(p_, f_)
    print(f"GPFA: {len(todo)} fitted now, {len(keys) - len(todo)} reloaded from disk")
    out = []
    for k_ in keys:
        with open(CACHE_GP / f"{k_}.pkl", "rb") as f_:
            out.append(pickle.load(f_))
    return out

BLOCKS12 = list(range(5)) + ["all"]
GP12 = dict(zip(BLOCKS12, gpfa_cached([np.concatenate([SEGS[t] for t in train_trials(k)]) for k in BLOCKS12])))
F_GP12 = {k: stitch_all([gpfa_infer_segments(SEGS[t], GP12[k]) for t in TRIALS]) for k in BLOCKS12}  # ordered factors, whole session

def fa_factors(trs):                                        # FA with 25 factors on the square root of the counts of trials trs (all their bins); ordered factors of the whole session
    Y = np.sqrt(COUNTS.astype(float))
    fa_ = FactorAnalysis(n_components=K_PCA, svd_method="lapack", random_state=0).fit(Y[np.isin(trial, trs)])
    return order_by_readout(fa_.transform(Y), fa_.components_.T)  # ordered like those of GPFA (02_new, §9.4)

F_FA12 = {k: fa_factors(train_trials(k)) for k in BLOCKS12}
print("Factors per bin:", {"FA": F_FA12[0].shape, "GPFA alla pari": F_GP12[0].shape, "LFADS": F_LF12[(0, 0)].shape})

# %% [markdown]
# **Reading.**
#
# - **The version kept is in practice the last one**, as we wanted: in all 30 models it is between epoch 196
#   and 200 (in 12 exactly 200). The "200 fixed epochs" are therefore 200 to within 4.
# - **A training lasts 4.1 minutes on average** (4 at a time on the CPU, with the Mac in energy
#   saving mode; without it, 1.8 minutes: §12.4). Models and GPFAs stay on disk: rerunning the notebook
#   retrains nothing.
# - **The three models give 25 factors for each of the 12 233 bins of the session**: FA bin by bin, GPFA
#   and LFADS by stitching the segments. In each block the factors of the 3 test trials come from models whose weights
#   never saw those trials. The choices of §11b (configuration, 200 epochs) instead were
#   made on the trials of block 0, some of which are test trials in the other blocks: the same limit
#   as the number of CEBRA steps in part A.
#
# ### 12.2 · Phase decoding
#
# The common yardstick of §1.2, on the same 1106 points and the same 5 blocks: 10-neighbor classifier
# fitted on the points of the training trials, judged on those of the test trials, balanced accuracy
# (chance 12.5%). For each of the four models (FA, GPFA as in §7b, GPFA on equal terms, LFADS):
#
# - the **first 2, 3 and 7 ordered factors**;
# - the **3-coordinate** maps built on the 25 factors with **PCA, Isomap and UMAP** (as in 03 on the GPFA
#   factors), fitted on the training points only;
# - the **ceiling**: the classifier on all 25 factors.
#
# LFADS has 5 seeds (for UMAP, the UMAP seed is the same as the LFADS one); for FA and GPFA, UMAP has 5 seeds.
# Alongside, the numbers of part A on the 25 PCs of the smoothed firing rates (PCA, Isomap, UMAP, CEBRA-Time, ceiling)
# and the clock (§5.3).
#
# **Without `SnapTo`** (§9.5): the same computation for the first 3 factors and for the ceiling, removing the
# `SnapTo` points from both training and test (7 phases: chance 14.3%).
# %%
GETTERS12 = {"FA": [lambda k: F_FA12[k]],                   # for each model, a function block -> factors of the whole session (one per seed)
             "GPFA come nel §7b": [lambda k: X_GP[k]],
             "GPFA alla pari": [lambda k: F_GP12[k]],
             "LFADS": [lambda k, sd=sd: F_LF12[(k, sd)] for sd in SEEDS_LF]}
MEASURES12 = {"primi 2 fattori": lambda g, i: decode_map(fold_inp(lambda k: g(k)[:, :2]), emb_none),
              "primi 3 fattori": lambda g, i: decode_map(fold_inp(lambda k: g(k)[:, :3]), emb_none),
              "primi 7 fattori": lambda g, i: decode_map(fold_inp(lambda k: g(k)[:, :7]), emb_none),
              "PCA, d = 3": lambda g, i: decode_map(fold_inp(g), emb_pca(3)),
              "Isomap, d = 3": lambda g, i: decode_map(fold_inp(g), emb_isomap(3)),
              "UMAP, d = 3": None,                          # computed in parallel in run_measure
              "tetto: tutti i 25 fattori": lambda g, i: decode_map(fold_inp(g), emb_none)}

def run_measure(name, meas):                                # the accuracies of a model for a measure: one per seed (5 for UMAP even with a single model)
    gs = GETTERS12[name]
    if meas == "UMAP, d = 3":                               # UMAP in parallel: only the 1106 points of each block are passed to the processes, not the whole session
        inps = [fold_inp(g) for g in gs] if len(gs) > 1 else [fold_inp(gs[0])] * 5
        return Parallel(n_jobs=5)(delayed(decode_map)(inp_, emb_umap(3, seed=i)) for i, inp_ in enumerate(inps))
    return [MEASURES12[meas](g, i) for i, g in enumerate(gs)]

ACC12 = {(name, meas): run_measure(name, meas) for name in GETTERS12 for meas in MEASURES12}
tab12 = pd.DataFrame({name: {meas: fmt(ACC12[(name, meas)]) for meas in MEASURES12} for name in GETTERS12}).T
tab12.loc["parte A: 25 PC dei firing rate smussati"] = [fmt(ACC["PCA, d = 2"]), fmt(ACC["PCA, d = 3"]), fmt(HI_ACC[("PCA", 7)]),
                                                        fmt(ACC["PCA, d = 3"]), fmt(ACC["Isomap, d = 3"]), fmt(ACC["UMAP, d = 3"]),
                                                        fmt(ACC["tetto, 25 dimensioni"])]
display(tab12)
print(f"For comparison (part A, 25 PCs): CEBRA-Time with 2 and 3 free coordinates {fmt(ACC['CEBRA-Time, 3 dimensioni'])} and "
      f"{fmt(ACC['CEBRA-Time, 4 dimensioni'])}; the clock alone {acc_clock:.1f}; chance 12.5")

# %%
keep_ns = phase[bins] != "SnapTo"                           # the evaluation points that are not SnapTo

def decode_subset(get, keep, d=None):                       # the common yardstick only on the points `keep` (training and test), with the first d factors (None = all)
    idx = np.flatnonzero(keep)
    pred = pd.Series(index=idx, dtype=object)
    for k, (trn, tst) in enumerate(FOLDS):
        trn_k, tst_k = trn[keep[trn]], tst[keep[tst]]       # training and test points of block k, without SnapTo
        F = get(k) if d is None else get(k)[:, :d]
        clf = KNeighborsClassifier(n_neighbors=K_NN).fit(F[bins[trn_k]], phase[bins[trn_k]])
        pred.loc[tst_k] = clf.predict(F[bins[tst_k]])
    return balanced_accuracy_score(phase[bins[idx]], pred.loc[idx].values) * 100

rows_ns = {}
for name, gs in GETTERS12.items():
    rows_ns[name] = {"primi 3 fattori": fmt([decode_subset(g, keep_ns, 3) for g in gs]),
                     "tetto: tutti i 25 fattori": fmt([decode_subset(g, keep_ns) for g in gs])}
rows_ns["parte A: 25 PC dei firing rate smussati"] = {"primi 3 fattori": fmt(decode_subset(lambda k: PCA(3).fit(X25[k][bins[FOLDS[k][0]]]).transform(X25[k]), keep_ns)),
                                                      "tetto: tutti i 25 fattori": fmt(decode_subset(lambda k: X25[k], keep_ns))}
display(pd.DataFrame(rows_ns).T.rename_axis("without SnapTo (7 phases, chance 14.3%)"))

# %%
MEAS_PLOT = ["primi 2 fattori", "primi 3 fattori", "PCA, d = 3", "Isomap, d = 3", "UMAP, d = 3", "tetto: tutti i 25 fattori"]
COL12 = {"FA": "C1", "GPFA come nel §7b": "C4", "GPFA alla pari": "C0", "LFADS": "C3"}
fig, ax = plt.subplots(figsize=(14, 5))
w12 = 0.2
for i, name in enumerate(GETTERS12):
    for j, meas in enumerate(MEAS_PLOT):
        v = np.atleast_1d(ACC12[(name, meas)])
        x = j + (i - 1.5) * w12
        ax.bar(x, v.mean(), width=w12 * 0.9, color=COL12[name], label=name if j == 0 else None)
        if len(v) > 1:
            ax.plot([x, x], [v.min(), v.max()], color="k", lw=1)  # minimum-maximum across seeds
for yv, lab, ls in ((ACC["tetto, 25 dimensioni"], f"part A: ceiling on the 25 PCs ({ACC['tetto, 25 dimensioni']:.1f}%)", "--"),
                    (np.mean(ACC["UMAP, d = 3"]), f"part A: UMAP d = 3 on the 25 PCs ({np.mean(ACC['UMAP, d = 3']):.1f}%)", "-."),
                    (acc_clock, f"the clock alone ({acc_clock:.1f}%)", ":"), (12.5, "chance level", "-")):
    ax.axhline(yv, color="0.35", ls=ls, lw=1, label=lab)
ax.set(xticks=range(len(MEAS_PLOT)), xticklabels=MEAS_PLOT, ylim=(0, 60), ylabel="phase recognized (%)",
       title="Phase decoding on never-seen trials: FA, GPFA and LFADS (black bars: minimum-maximum across seeds)")
ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
fig.tight_layout(); fig.savefig(FIGS / "04_lfads_phase.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# **Reading.**
#
# - **LFADS does not beat the equal-terms GPFA: it is below on every measure.** With all 25 factors (the ceiling)
#   it recognizes the phase in 46.5% of the points (44.7-47.7 across the 5 seeds), the equal-terms GPFA in 51.4%: none of the
#   5 seeds reaches GPFA. With the 3-coordinate maps the gap remains: UMAP 45.8 versus 50.4, Isomap 40.0
#   versus 44.9. Only with the 3-component PCA is it small (30.9 versus 32.5) and within the variation across the LFADS seeds
#   (27.9-34.7).
# - **The largest gap is in the first ordered factors**: with 2 and 3 factors LFADS reaches 17.9% and
#   20.9%, GPFA 32.1% and 30.9%; with 7, 32.2% versus 41.5%. The directions that weigh most in the
#   LFADS rates are not those that distinguish the phases: the 3-component PCA on the same 25 factors, which
#   looks for the directions of maximum variance, reaches 30.9%. Why, we did not verify.
# - **The two GPFAs are equivalent**: ceiling 51.4 (equal terms: 600 ms segments, all bins of the
#   training trials) versus 50.7 (§7b: 1 s windows from the start of the phases); with UMAP 50.4 versus 50.5. The
#   cut into 600 ms segments does not penalize GPFA, so it does not explain the LFADS gap.
# - **The model of time matters more than everything else.** FA, which treats each bin on its own, reaches
#   25.8% with all factors and 17.4% with the first 2. Adding factors that change gradually (GPFA)
#   brings the ceiling to 51.4%. Adding nonlinear dynamics rules (LFADS) adds nothing more, and in fact
#   takes away about 5 points. Why the first FA factors are worth so little we see in the checks of
#   §12.3.
# - **Relative to the references**: only the two GPFAs exceed the clock (48.5%). LFADS stays below, but above the
#   25 PCs of the smoothed firing rates of part A (ceiling 42.2, 3-coordinate UMAP 42.6) and above CEBRA-Time
#   (37.7 and 40.4 with 2 and 3 free coordinates).
# - **Without `SnapTo` (7 phases, chance 14.3%) the order does not change**: equal-terms GPFA ceiling 58.5, §7b GPFA
#   57.9, LFADS 53.1 (50.6-54.1), 25 PCs of part A 48.0, FA 29.5. With the first 3 factors LFADS stays the
#   lowest after FA (23.6 versus 35.5 and 38.6 of the two GPFAs). `SnapTo` does not decide the comparison.
# - **The figure** summarizes the table: the red bars (LFADS) are below the blue ones (equal-terms GPFA) in every
#   group, and the minimum-maximum across seeds (the black lines) reaches GPFA only with the 3-component PCA.
# - **What it does not say: why LFADS stays below.** Three hypotheses, which we will check: (1) LFADS learns from 12
#   trials, about 3 minutes of recording, and in §11 it memorized the seen trials (§13: training versus test,
#   and how many trials are needed); (2) the configuration was chosen on the hidden channels, not on the phase, and the
#   KL weight was on the edge of the grid (§14); (3) recognizing the phase is not the task LFADS was
#   built for: its measure, predicting the hidden channels of the test trials, is in §12b.
#
# ### 12.3 · The single-trial trajectories
#
# The measures of §4.2 (roughness and consistency, also by task version) on the 119 windows of 1 s,
# with the **first 3 ordered factors** of the models fitted on **all 15 trials**. Alongside, the numbers of
# part A (PCA, Isomap, UMAP, CEBRA-Time on the 25 PCs) and those of the GPFA of 02_new (roughness 0.011,
# consistency 0.11 mixing the versions and 0.137 per version; fitted on the windows themselves).
#
# It should be kept in mind that for the equal-terms GPFA and LFADS each window is stitched from 600 ms segments (§9.2),
# while the GPFA of 02_new saw each 1 s window whole.

# %%
def traj_of(F):                                             # the measures of §4.2 with the first 3 ordered factors
    return traj_stats([w_[:, :3] for w_ in cut(F)])

TRAJ12 = {"FA (tutti i trial)": [traj_of(F_FA12["all"])],
          "GPFA alla pari (tutti i trial)": [traj_of(F_GP12["all"])],
          "LFADS (tutti i trial, 5 semi)": [traj_of(F_LF12[("all", sd)]) for sd in SEEDS_LF],
          **{f"parte A: {name}": TRAJ[name] for name in ("PCA", "Isomap", "UMAP", "CEBRA-Time, 4 dimensioni")}}
def fmt3(v):                                                # "mean (minimum-maximum)" with 3 decimals (the fmt of part A has one: it is for percentages)
    v = np.atleast_1d(v)
    return f"{v.mean():.3f}" if len(v) == 1 else f"{v.mean():.3f} ({v.min():.3f}-{v.max():.3f})"

rows_t12 = {}
for name, runs in TRAJ12.items():
    d_ = pd.DataFrame(runs)
    rows_t12[name] = {c: fmt3(d_[c].values) for c in d_.columns}
rows_t12["GPFA of 02_new (§10.5, whole windows)"] = {"seghettatura": "0.011", "consistenza, tutte le ripetizioni": "0.11"}
display(pd.DataFrame(rows_t12).T.fillna("not computed"))

# %%
show12 = {"FA": cut(F_FA12["all"]), "GPFA alla pari": cut(F_GP12["all"]), "LFADS (seme 0)": cut(F_LF12[("all", 0)])}
fig, ax = plt.subplots(1, 3, figsize=(15, 3.8), sharey=True)
for a_, (name, tr_) in zip(ax, show12.items()):
    R = np.stack([tr_[i][:, 0] for i in idx_ph])            # the first ordered factor of each repetition of Release
    R = (R - R.mean()) / R.std()                            # same scale for all models
    for odd, c, lab in ((True, "C1", "odd trials"), (False, "C0", "even trials")):
        sel = WIN_ODD[idx_ph] == odd
        for r_ in R[sel]:
            a_.plot(t_win, r_, color=c, lw=0.7, alpha=0.45)
        a_.plot(t_win, R[sel].mean(0), color=c, lw=2.5, label=f"{lab} (mean)")
    a_.set(title=f"{name}: first ordered factor", xlabel="ms from the start of the phase")
ax[0].set_ylabel("z"); ax[0].legend(fontsize=8)
fig.suptitle(f"The {len(idx_ph)} repetitions of {PH_SHOW}, colored by task version (models fitted on all trials)", y=1.02)
fig.tight_layout(); fig.savefig(FIGS / "04_lfads_trajectories.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# **Reading.**
#
# - **LFADS is almost as smooth as GPFA**: roughness 0.028 (0.018-0.048 across seeds) versus 0.017 for the
#   equal-terms GPFA and 0.013 for PCA; much less than CEBRA-Time (0.074), Isomap (0.275) and UMAP (0.596). Almost
#   all of its roughness comes, however, from a few windows, those with the trial change (checks below). FA, which has no model of
#   time, reaches 7.3: like noise without memory, in fact a little more
#   (checks below).
# - **Consistency: LFADS and equal-terms GPFA.** On all repetitions LFADS 0.109 (0.090-0.135), the equal-terms GPFA
#   0.118; separating the two task versions 0.132 (0.103-0.155) versus 0.138. The difference is within the
#   variation across the LFADS seeds. UMAP is a little higher (0.125 and 0.144), CEBRA-Time a little lower (0.106
#   and 0.122), FA almost zero (0.029 and 0.034).
# - **For all methods consistency rises when the versions are separated**, and the even trials are more similar to each other
#   than the odd ones (equal-terms GPFA 0.159 versus 0.118, LFADS 0.155 versus 0.109), as for the part A methods
#   in the bottom rows.
# - **Equal-terms GPFA and GPFA of 02_new**: roughness 0.017 versus 0.011, consistency 0.118 versus 0.11 (per
#   version 0.138 versus 0.137). The
#   stitched segments make GPFA a little less smooth, not less consistent.
# - **The figure** (the 14 repetitions of `Release`, first ordered factor): FA is noise around an almost
#   flat mean; GPFA and LFADS give smooth curves, and in both the mean of the odd trials descends towards
#   200 ms more than that of the even ones. Between 760 and 840 ms there is a jump, much stronger in LFADS: it is the trial
#   change, where the models restart from scratch (checks below).
# - **What it does not say**: these models are fitted on all 15 trials, so they describe the trajectories
#   but do not measure generalization; and the first 3 ordered factors are not the best LFADS directions
#   for distinguishing the phases (§12.2).

# %% [markdown]
# #### Two checks on the trajectories: the joins and the 18 Hz rhythm
#
# **The joins.** In §11.4 the factors of two consecutive LFADS segments, in the zone they share,
# differed on average by 0.65 standard deviations, and we had deferred the question to here: how much do the
# joins weigh on the roughness? We divide the bins of the 119 windows into two groups: those **in the zone shared
# by two segments**, where stitching averages two segments, and those **covered by a single segment**. Then
# we compute the roughness in each group (each second difference goes to its central bin). If the
# joins matter, the roughness is higher in the shared zones. FA and the part A PCA are not
# stitched: for them the two groups must give the same number, and this checks the measure itself.
#
# **The largest join: the trial change.** The segments never cross the boundary between two trials
# (§9.2): there GPFA and LFADS restart from scratch. The trial change falls inside `Release` (§1.5), hence
# inside the 1 s windows of `Release`. We count which windows contain it and recompute the
# roughness without them.
#
# **The 18 Hz rhythm.** The FA factors are rougher than noise without memory. For noise
# that changes randomly from one bin to the next the roughness is 6 (hand computation: the second difference
# x[t+1] − 2x[t] + x[t−1] has variance 1 + 4 + 1 = 6 times that of x), in fact 6 × 50/49 = 6.12 because the
# z-score of each window uses the variance of its 50 bins. An oscillation that changes direction at almost every bin raises the
# second difference further: the ~18 Hz rhythm of §11.3 has one period every 2.8 bins. To
# find out we measure what share of the power of the first 3 ordered factors falls between 16 and 20 Hz, with the
# mean spectrum over 1 s windows (as in §11.3).

# %%
COVER = np.zeros(n_bins, int)                               # in how many segments each bin of the session lies
for t in TRIALS:
    for a in SEG_START[t]:
        COVER[a:a + SEG_LEN] += 1                           # each segment covers 30 bins from its start
IN_OVL = COVER > 1                                          # true for the bins in the zone shared by two segments (where stitching averages)
MID = np.concatenate([np.arange(w["start"] + 1, w["stop"] - 1) for w in windows])  # the central bins of the second differences, window after window

def rough_split(F):                                         # roughness of the first 3 factors in the bins of the shared zones and in the bins of a single segment
    d2 = np.concatenate([(np.diff((T_ - T_.mean(0)) / (T_.std(0) + 1e-9), n=2, axis=0) ** 2).mean(1)  # as roughness (§4.2): one value per central bin
                         for T_ in (F[w["start"]:w["stop"], :3] for w in windows)])
    return d2[IN_OVL[MID]].mean(), d2[~IN_OVL[MID]].mean()

def share_18(F, L=50, step=25):                             # share of the power of the first 3 factors between 16 and 20 Hz, mean spectrum over 1 s windows
    win = np.hanning(L)                                     # the same window as §11.3
    S = np.mean([np.abs(np.fft.rfft((F[a:a + L, :3] - F[a:a + L, :3].mean(0)) * win[:, None], axis=0)) ** 2
                 for a in range(0, n_bins - L, step)], axis=0)  # power per frequency, one column per factor
    f_ = np.fft.rfftfreq(L, io.BIN_S)                       # the frequencies: from 0 to 25 Hz every 1 Hz
    return S[(f_ >= 16) & (f_ <= 20)].sum(0) / S[1:].sum(0)  # share of the total power (without 0 Hz)

def cross_tr(w):                                            # true if the trial changes within window w
    return bool((np.diff(trial[w["start"]:w["stop"]]) != 0).any())

CUT_MS = [20 * (int(np.flatnonzero(np.diff(trial[w["start"]:w["stop"]]))[0]) + 1) for w in windows if cross_tr(w)]  # where the change falls, in ms from the start of the window
print(f"Windows with a trial change: {len(CUT_MS)} of {len(windows)}, of phases {sorted({w['tag'] for w in windows if cross_tr(w)})}; "
      f"the change falls between {min(CUT_MS)} and {max(CUT_MS)} ms from the start of the window")

CTRL12 = {"FA (tutti i trial)": [F_FA12["all"]],
          "GPFA alla pari (tutti i trial)": [F_GP12["all"]],
          "LFADS (tutti i trial, 5 semi)": [F_LF12[("all", sd)] for sd in SEEDS_LF],
          "parte A: PCA (non ricucita)": [PCA(3).fit(X_all).transform(Z25)]}
rows_c = {}
for name, Fs in CTRL12.items():
    sp = np.array([rough_split(F_) for F_ in Fs])           # (seeds, 2): shared zones, single segment
    q = 100 * np.array([share_18(F_) for F_ in Fs])         # (seeds, 3): one value per factor, in %
    nb_ = [roughness([F_[w["start"]:w["stop"], :3] for w in windows if not cross_tr(w)]) for F_ in Fs]  # without the windows with the trial change
    rows_c[name] = {"roughness, shared zones": fmt3(sp[:, 0]), "roughness, single segment": fmt3(sp[:, 1]),
                    "roughness, without trial change": fmt3(nb_),
                    "16-20 Hz, factor 1 (%)": fmt(q[:, 0]), "16-20 Hz, factor 2 (%)": fmt(q[:, 1]), "16-20 Hz, factor 3 (%)": fmt(q[:, 2])}
display(pd.DataFrame(rows_c).T)
rng_w = np.random.default_rng(0)                            # noise without memory, for the hand computation
print(f"Central bins in the shared zones: {IN_OVL[MID].mean() * 100:.1f}%. Noise without memory (2000 windows of 50 bins): "
      f"roughness {roughness([rng_w.standard_normal((50, 3)) for _ in range(2000)]):.2f} (by hand: 6 × 50/49 = {6 * 50 / 49:.2f}); "
      f"16-20 Hz share {100 * share_18(rng_w.standard_normal((n_bins, 3))).mean():.1f}% (by hand: 5 frequencies out of 25 = 20%)")

# %% [markdown]
# **Reading.**
#
# - **The check of the measure works**: for FA and the part A PCA, which are not stitched, the two
#   groups of bins give the same number (FA 7.324 and 7.317, PCA 0.013 and 0.012).
# - **Where the segments are stitched the trajectories are smooth**: in the shared zones the roughness is 0.005
#   (0.004-0.006) for LFADS and 0.012 for the equal-terms GPFA, like PCA. The disagreement between consecutive segments
#   seen in §11.4 is absorbed by the weighted mean, which passes gradually from one segment to the next.
# - **All of the extra roughness of LFADS comes from the trial change.** The change falls in 14 windows out of
#   119, all of `Release`, between 760 and 840 ms from the start. Without those 14 windows LFADS drops from 0.028 to
#   0.005 (0.004-0.005): **the smoothest of all**, more than the equal-terms GPFA (0.012) and PCA (0.013).
#   In the bins covered by a single segment, including the start of each trial, it rises instead to 0.051 (0.031-0.090). At the
#   trial change the model restarts from the initial condition of the first segment alone, and the first bin of a
#   segment was already the weakest point in the synthetic data (§10b).
# - **The first FA factors are in large part the 18 Hz rhythm**: between 16 and 20 Hz falls 51.4% of the
#   power of the first factor, 24.4% of the second and 31.1% of the third. For noise without memory it is
#   20.1%, like the hand computation (5 frequencies out of 25 = 20%); the roughness of the noise too, 6.12,
#   matches the hand computation (6 × 50/49). The rhythm of §11.3 is an activity shared by many channels, which is exactly what
#   FA looks for, and FA has no way to prefer slow signals: it puts it in the strongest factors. It is
#   consistent with the roughness above that of noise (an oscillation that changes direction every 1.4 bins raises
#   the second difference) and with the first 2-3 FA factors at 17% in phase decoding (§12.2).
# - **GPFA and LFADS do not pick up the rhythm**: between 16 and 20 Hz falls 0.0% and 0.2% (0.1-0.5) of the power
#   of the first factor, as for the PCA of the smoothed firing rates. GPFA, with factors that change
#   gradually, could not; LFADS in principle could, but does not, in agreement with §11.3
#   (the trials generated by LFADS did not have the oscillating autocorrelation of the real data).
# - **What it does not say**: the check looks at the first 3 ordered factors and the 1 s windows, not all
#   25 factors; that the jump at the trial change also weighs on phase decoding is unlikely
#   (few evaluation points fall near a boundary, §1.5), but we did not measure it; the origin of the
#   18 Hz rhythm remains unknown (§11.3).
#
# ### Summary of §12a
#
# **First answer to the question of part B**, for phase decoding and the trajectories: **no**,
# on these data the nonlinearity and the learned dynamics rules of LFADS do not improve on GPFA. The
# LFADS trajectories are smoother within trials and as consistent as those of GPFA, but decode the
# phase about 5 points worse.
#
# | question | answer | where |
# |---|---|---|
# | Does LFADS decode the phase better than the equal-terms GPFA? | **no**: 46.5% (44.7-47.7) versus 51.4% with all 25 factors; below also with the 3-coordinate maps and without `SnapTo` | §12.2 |
# | and compared with the other methods? | above the 25 PCs of part A (42.2%) and CEBRA-Time; below the clock (48.5%), which only the GPFAs exceed | §12.2 |
# | what matters most? | having a model of time: FA 25.8%, GPFA 51.4%; the nonlinearity of LFADS adds nothing | §12.2 |
# | are the LFADS trajectories smooth and repeatable? | smooth: 0.005 without the trial change (equal-terms GPFA 0.012); consistent like GPFA (0.109 versus 0.118) | §12.3 |
# | do the joins between segments matter? | no; the trial change matters, where LFADS restarts from scratch | §12.3 |
# | why does FA do so badly? | its first factors are in large part the 18 Hz rhythm | §12.3 |
#
# **What we did not verify**: why LFADS stays below GPFA (little data and memorization, §13;
# configuration, §14); whether the gap changes in the measure proper to generative models, the hidden channels
# of the test trials (§12b); everything holds for a single session and for the 5 blocks fixed in 03.

# %% [markdown]
# ### 12.4 · The measure for generative models: the hidden channels of the test trials (§12b)
#
# Phase decoding (§12.2) asks the factors to separate the phases, something none of
# these models was trained for. Now the measure they are built for: **predicting never-seen activity**
# (§9.4). In each block each model learns on the 12 training trials, where it sees all 160 channels; in the 3
# test trials it receives only the **120 visible channels** and must guess the **40 hidden** ones. The score is in
# bits per spike (§9.4): 0 = "always say the mean rate of the channel", more = better.
#
# Each trial is a test trial in only one block: putting the predictions of the 5 blocks in sequence gives a
# prediction of the **whole session** made only on never-seen trials, and the score is computed on the 12 233 bins
# (the benchmark uses all bins of the test trials). Alongside, the score block by block.
#
# | model | how it predicts the hidden channels of a test trial | from where |
# |---|---|---|
# | **smoothing + GLM** (the reference) | smooths the 120 visible (σ = 50 ms) and, for each hidden one, a Poisson regression fitted on the training trials; penalty 0.1 | §11a, the benchmark reference |
# | **FA** | derives the factors of each bin from the visible channels only, then the model's expected count for the hidden ones | §12.1 |
# | **Equal-terms GPFA** | the same GPFA as §12a (fitted on the 160 channels of the training segments); in the test segments it derives the factors from the visible channels only, then the expected count | §12.1 |
# | **LFADS** | the "120 visible + 40 hidden" form of §11: the encoder reads the visible ones, the output gives all 160. §11b configuration, 200 epochs, 5 seeds | §11, §12.1 |
#
# **The expected count of FA and GPFA.** These two models describe the **square root** of the counts, with Gaussian
# noise. If the square root of a count y has mean μ and variance v, then the mean of y is μ² + v. The
# variance v has two parts: the channel's own noise (R for GPFA, Ψ for FA) and the uncertainty about the
# factors, which cannot be derived exactly from the 120 visible channels alone (Σ, carried onto the channel by its weights:
# c Σ cᵀ). In §10b, with all channels visible, we had used μ² + R; here we use the full formula, and
# μ² + R stays as a variant.
#
# **A second GPFA variant**: instead of the model's expected count, a Poisson regression from the
# 25 factors (derived from the visible channels) to each hidden channel, fitted on the training trials, as the
# benchmark references do with models that do not predict the counts by themselves. Penalty 0.1, as for
# smoothing, without choosing it on purpose.
#
# **Three hand computations**, before the scores: (1) the bits per spike with the formula of §11a and with the
# Poisson likelihood of `scipy`; (2) the FA factors with our formula, using all 160
# channels, against those of `scikit-learn`; (3) the GPFA prediction with our restricted model,
# using all 160 channels, against that of §10b.
#
# 25 LFADS trainings (5 blocks × 5 seeds), 4 at a time, and 10 regressions of 40 channels: about 12 minutes
# the first time (measured with the Mac not in energy-saving mode; in energy-saving mode the trainings
# take more than double, §12.1).

# %%
from scipy.stats import poisson                             # the Poisson likelihood of the library, for the hand computation
from sklearn.linear_model import PoissonRegressor           # the Poisson regression (already used in §11a)

LF_HID_HP = dict(LF_HP, n_in=len(HELD_IN))                  # the §12a configuration, but the encoder reads only the 120 visible channels
RUNS12B = [(k, sd) for k in range(5) for sd in SEEDS_LF]    # (block, seed): 25 trainings
res12b = dict(zip(RUNS12B, run_lfads_cached([lf12_job(sd, train_trials(k), hp=LF_HID_HP, segs=SEGS_HID, rate_cols=HID_COLS)
                                             for k, sd in RUNS12B])))  # 12 training trials, 200 epochs; saves the rates of the 40 hidden ones
ep_b = np.array([r["best_epoch"] for r in res12b.values()])
print(f"LFADS (120 + 40): {len(res12b)} models; version kept: epoch {ep_b.min()}-{ep_b.max()} of {N_EP_LF}; "
      f"mean duration {np.mean([r['seconds'] for r in res12b.values()]) / 60:.1f} minutes")

Y_HID = COUNTS[:, HIDDEN].astype(float)                     # the true counts of the 40 hidden ones, over the whole session

def test_only(per_block):                                   # from "block -> prediction of the whole session" to the session made only of test trials
    out = np.empty_like(per_block[0])
    for k in range(5):
        tb = np.isin(trial, TEST_TRIALS[k])                 # the bins of the 3 test trials of block k
        out[tb] = per_block[k][tb]                          # taken from the model of block k, which never saw them
    return out

PRED_LF = [test_only([stitch_all(res12b[(k, sd)]["rates"]) for k in range(5)]) for sd in SEEDS_LF]  # per seed: expected counts of the 40 hidden ones, 12233 × 40

# %%
def gpfa_vis(segs, params, ch):                             # the GPFA factors derived from the channels ch only (mean and uncertainty), segment by segment
    p_ = dict(params, C=params["C"][ch], d=params["d"][ch], R=params["R"][np.ix_(ch, ch)])  # the model restricted to the channels ch
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        lat, _ = gpfa_core.exact_inference_with_ll(gpfa_seqs(segs[..., ch]), p_, get_ll=False)
    return lat["latent_variable"], lat["Vsm"]               # per segment: factors (25 × bins) and uncertainty (25 × 25 × bins)

def gpfa_hidden(segs, params):                              # expected counts of the hidden ones in each segment, from the visible only: full and μ² + R variant
    X_, V_ = gpfa_vis(segs, params, HELD_IN)
    C_h, d_h, r_h = params["C"][HIDDEN], params["d"][HIDDEN], np.diag(params["R"])[HIDDEN]  # weights, means and noise of the hidden ones
    mu = np.stack([(C_h @ x_ + d_h[:, None]).T for x_ in X_])  # mean of the square root of the counts: (segments, bins, 40)
    unc = np.stack([np.einsum("hi,ijt,hj->th", C_h, v_, C_h) for v_ in V_])  # uncertainty on the factors carried onto each channel: c Σ cᵀ
    return mu ** 2 + r_h + unc, mu ** 2 + r_h

def fa_hidden(trs, C=None):                                 # FA on the 160 channels of trials trs; expected counts of the hidden ones over the whole session, from the visible only
    Y = np.sqrt((COUNTS if C is None else C).astype(float))  # the counts (in §13.4 the time-shifted ones)
    fa_ = FactorAnalysis(n_components=K_PCA, svd_method="lapack", random_state=0).fit(Y[np.isin(trial, trs)])
    L, psi, m = fa_.components_.T, fa_.noise_variance_, fa_.mean_  # weights (160 × 25), noise of each channel (Ψ), means
    Lv, Lh = L[HELD_IN], L[HIDDEN]
    S = np.linalg.inv(np.eye(K_PCA) + Lv.T @ (Lv / psi[HELD_IN, None]))  # uncertainty on the factors, given the 120 visible (Σ)
    X = ((Y[:, HELD_IN] - m[HELD_IN]) / psi[HELD_IN]) @ Lv @ S  # factors of each bin from the visible only: Σ Λᵀ Ψ⁻¹ (y − m)
    mu = X @ Lh.T + m[HIDDEN]                               # mean of the square root of the counts of the hidden ones
    return mu ** 2 + psi[HIDDEN] + np.einsum("hi,ij,hj->h", Lh, S, Lh)

# (1) bits per spike: formula of §11a against the Poisson likelihood of scipy (log k! cancels in the difference)
lam_ = PRED_LF[0]
by_scipy = (poisson.logpmf(Y_HID, lam_).sum() - poisson.logpmf(Y_HID, Y_HID.mean(0)).sum()) / Y_HID.sum() / np.log(2)
print(f"(1) bits per spike of LFADS (seed 0): formula of §11a {bits_per_spike(lam_, Y_HID):.6f}, with scipy {by_scipy:.6f}")
# (2) FA factors: our formula with all 160 channels against scikit-learn's transform
Yq = np.sqrt(COUNTS.astype(float))
fa_chk = FactorAnalysis(n_components=K_PCA, svd_method="lapack", random_state=0).fit(Yq[np.isin(trial, train_trials(0))])
L_, psi_ = fa_chk.components_.T, fa_chk.noise_variance_
X_hand = ((Yq - fa_chk.mean_) / psi_) @ L_ @ np.linalg.inv(np.eye(K_PCA) + L_.T @ (L_ / psi_[:, None]))
print(f"(2) FA, hand-computed factors against scikit-learn: maximum difference {np.abs(X_hand - fa_chk.transform(Yq)).max():.1e} "
      f"(the factors reach {np.abs(X_hand).max():.1f})")
# (3) GPFA: the restricted model, with all channels, against the prediction of §10b
t_chk = int(TEST_TRIALS[0][0])
X_chk, _ = gpfa_vis(SEGS[t_chk], GP12[0], np.arange(N_CH))
mu_chk = np.stack([(GP12[0]["C"] @ x_ + GP12[0]["d"][:, None]).T for x_ in X_chk])
print(f"(3) GPFA, model restricted to all 160 channels against gpfa_sqrt_pred (§10b), trial {t_chk}: "
      f"maximum difference {np.abs(mu_chk - gpfa_sqrt_pred(SEGS[t_chk], GP12[0])).max():.1e}")

# %%
GP_PIECES = {int(t): gpfa_hidden(SEGS[t], GP12[k]) for k in range(5) for t in TEST_TRIALS[k]}  # each trial with the GPFA of the block in which it is a test trial
PRED_GP = np.concatenate([stitch(GP_PIECES[int(t)][0], t) for t in TRIALS])     # full expected count, stitched
PRED_GP_R = np.concatenate([stitch(GP_PIECES[int(t)][1], t) for t in TRIALS])   # μ² + R variant (§10b)
PRED_FA = test_only([fa_hidden(train_trials(k)) for k in range(5)])
PRED_GLM = test_only([smooth_glm_rates(train_trials(k), 0.1) for k in range(5)])  # the smoothing + GLM of §11a, on the 12 training trials

def gpfa_glm(k, segs=None, gp=None, C=None, trs=None):     # variant: Poisson regression from the 25 GPFA factors (derived from the visible ones) to the hidden ones, in block k
    segs, gp, C = (SEGS if segs is None else segs), (GP12 if gp is None else gp), (COUNTS if C is None else C)  # in §13.4: the time-shifted data
    lat = np.concatenate([stitch(np.stack([x_.T for x_ in gpfa_vis(segs[t], gp[k], HELD_IN)[0]]), t) for t in TRIALS])  # factors of the whole session
    fit_ = np.isin(trial, train_trials(k) if trs is None else trs)  # the regression is fitted on the 12 training trials (in §13b on fewer trials)
    F = StandardScaler().fit(lat[fit_]).transform(lat)
    return np.column_stack([PoissonRegressor(alpha=0.1, max_iter=500).fit(F[fit_], C[fit_, h]).predict(F) for h in HIDDEN])

PRED_GPGLM = test_only([gpfa_glm(k) for k in range(5)])

PRED12B = {"smoothing + GLM (reference)": [PRED_GLM], "FA": [PRED_FA], "GPFA alla pari": [PRED_GP],
           "LFADS (5 semi)": PRED_LF,
           "variante: GPFA, μ² + R (§10b)": [PRED_GP_R], "variante: GPFA + regressione di Poisson": [PRED_GPGLM]}

def fmt4(v):                                                # "mean [minimum, maximum]" with 4 decimals (square brackets because the values can be negative)
    v = np.atleast_1d(v)
    return f"{v.mean():.4f}" if len(v) == 1 else f"{v.mean():.4f} [{v.min():.4f}, {v.max():.4f}]"

BPS12 = {}                                                  # (model, where) -> one value per seed
for name, P in PRED12B.items():
    BPS12[(name, "tutta la sessione")] = [bits_per_spike(p_, Y_HID) for p_ in P]
    for k in range(5):
        tb = np.isin(trial, TEST_TRIALS[k])                 # the 3 test trials of block k
        BPS12[(name, f"blocco {k}")] = [bits_per_spike(p_[tb], Y_HID[tb]) for p_ in P]
WHERE12 = ["tutta la sessione"] + [f"blocco {k}" for k in range(5)]
display(pd.DataFrame({name: {w: fmt4(BPS12[(name, w)]) for w in WHERE12} for name in PRED12B}).T
        .rename_axis("bits per spike, 40 hidden channels, test trials; LFADS: mean [minimum, maximum] across seeds"))
lf_vs = np.array([[BPS12[("LFADS (5 semi)", f"blocco {k}")][i] > BPS12[("smoothing + GLM (reference)", f"blocco {k}")][0]
                   for k in range(5)] for i in range(len(SEEDS_LF))])
gp_vs = [BPS12[("LFADS (5 semi)", f"blocco {k}")] > np.array(BPS12[("GPFA alla pari", f"blocco {k}")]) for k in range(5)]
print(f"LFADS above smoothing + GLM: {int(lf_vs.sum())} cases of {lf_vs.size} (block × seed); "
      f"above the equal-terms GPFA: {int(np.sum(gp_vs))} of {lf_vs.size}")

# %%
BAND_OF = np.array([next(b for b, b_ in enumerate(bands) if h in b_) for h in HIDDEN])  # the frequency band (0 = the slowest) of each hidden channel
BAND_LAB = [f"band {b + 1}: {rate_hz[HIDDEN[BAND_OF == b]].min():.1f}-{rate_hz[HIDDEN[BAND_OF == b]].max():.1f} Hz" for b in range(4)]
rows_band = {name: {BAND_LAB[b]: fmt4([bits_per_spike(p_[:, BAND_OF == b], Y_HID[:, BAND_OF == b]) for p_ in P]) for b in range(4)}
             for name, P in PRED12B.items()}
display(pd.DataFrame(rows_band).T.rename_axis("bits per spike by frequency band (10 hidden each), whole session"))

# %%
MAIN12B = ["smoothing + GLM (reference)", "FA", "GPFA alla pari", "LFADS (5 semi)"]
COL12B = {"smoothing + GLM (reference)": "0.45", "FA": "C1", "GPFA alla pari": "C0", "LFADS (5 semi)": "C3"}
fig, ax = plt.subplots(1, 2, figsize=(15, 4.6), gridspec_kw={"width_ratios": (1.5, 1)})
for i, name in enumerate(MAIN12B):                          # left: the whole session and the 5 blocks
    for j, w in enumerate(WHERE12):
        v = np.array(BPS12[(name, w)])
        x = j + (i - 1.5) * 0.2
        ax[0].bar(x, v.mean(), width=0.18, color=COL12B[name], label=name if j == 0 else None)
        if len(v) > 1:
            ax[0].plot([x, x], [v.min(), v.max()], color="k", lw=1)  # minimum-maximum across seeds
ax[0].axhline(0, color="k", lw=0.8)
ax[0].set(xticks=range(len(WHERE12)), xticklabels=WHERE12, ylabel="bits per spike, 40 hidden",
          title="Hidden channels of the test trials (black bars: minimum-maximum across seeds)")
ax[0].legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.08))  # below the plot, so as not to cover the bars
for i, name in enumerate(MAIN12B):                          # right: by frequency band
    for b in range(4):
        v = np.array([bits_per_spike(p_[:, BAND_OF == b], Y_HID[:, BAND_OF == b]) for p_ in PRED12B[name]])
        x = b + (i - 1.5) * 0.2
        ax[1].bar(x, v.mean(), width=0.18, color=COL12B[name])
        if len(v) > 1:
            ax[1].plot([x, x], [v.min(), v.max()], color="k", lw=1)
ax[1].axhline(0, color="k", lw=0.8)
ax[1].set(xticks=range(4), xticklabels=[l_.replace(": ", "\n") for l_ in BAND_LAB], ylabel="bits per spike, 10 hidden",
          title="By frequency band, whole session")
fig.tight_layout(); fig.savefig(FIGS / "04_lfads_cobps.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# **Reading.**
#
# - **The trainings**: in all 25 models the version kept is between epoch 194 and 200, in practice
#   the last one. 1.8 minutes per model, with the Mac no longer in energy-saving mode (in §12a it was 4.1).
# - **The three hand computations check out**: the bits per spike with our formula and with `scipy` coincide
#   (0.039313); the FA factors with our formula differ from those of `scikit-learn` by about
#   10⁻¹⁵ (the exact value changes with the number of threads); the restricted GPFA, used with all channels, gives
#   exactly the prediction of §10b
#   (difference 0). The predictions of the hidden channels therefore rest on verified formulas.
# - **On its own measure LFADS beats neither GPFA nor the reference.** On the whole session: LFADS 0.0388 bits
#   per spike [0.0383, 0.0393 across the 5 seeds], the equal-terms GPFA 0.0422, smoothing + GLM 0.0416; below is
#   only FA (0.0358). Block by block LFADS exceeds smoothing + GLM in 4 cases out of 25 (block × seed) and
#   GPFA in 2 out of 25; it is on par only in block 2 (0.0417, like the reference; GPFA 0.0427).
# - **It confirms §11b**: there, on the validation trials, no configuration exceeded smoothing + GLM;
#   here, on never-seen trials and with 12 training trials instead of 10, LFADS stays about 7% below (0.0388
#   versus 0.0416).
# - **On this measure the model of time helps little**: GPFA exceeds FA (0.0422 versus 0.0358), but
#   smoothing + GLM, which has no shared population model, is already almost there (0.0416).
# - **How one goes from factors to counts matters a lot.** The GPFA + Poisson regression variant is the
#   best of all, 0.0474 (14% above the reference): the same GPFA factors, read with a
#   Poisson regression instead of the model's expected count, predict better. The uncertainty on the
#   factors, instead, does not matter: with μ² + R GPFA gives 0.0421 versus 0.0422.
# - **By frequency band the picture changes** (table and figure on the right):
#   - **slowest channels** (0.5-2.5 Hz): only the two Poisson regressions gain anything (0.0158 and
#     0.0153); GPFA and FA stay at zero; LFADS goes below zero (−0.0087 [−0.0129, −0.0028]), that is it does worse
#     than saying the mean rate of the channel;
#   - **3.5-7.8 Hz**: **LFADS is the best** (0.0821, versus 0.0709 for the reference, 0.0780 for GPFA with
#     the Poisson regression and 0.0441 for the equal-terms GPFA);
#   - **10 to 45 Hz**: LFADS is below GPFA (0.0646 versus 0.0704; 0.0210 versus 0.0299). These channels
#     carry most of the spikes, and the overall score is the mean of the band scores weighted by their
#     spikes: this is where LFADS loses.
#   - A plausible explanation, not verified, for the two slow bands: the expected count of GPFA and FA comes
#     from Gaussian noise on the square root of the counts, a poor description when spikes are rare;
#     the models with Poisson noise (the regressions and LFADS) do much better between 3.5 and 7.8 Hz.
# - **What it does not say**: (1) the 40 hidden channels are a single draw (§9.4); (2) why LFADS goes below
#   zero on the slowest channels: one hypothesis is the memorization of §11 (the readout of those channels learns from the
#   training trials details that do not repeat), to be checked in §13; (3) GPFA had a
#   variant with the Poisson regression and LFADS did not: the same regression on the LFADS factors
#   we did not try (§14); (4) the numbers of the Neural Latents Benchmark are not comparable with these
#   (other data, isolated neurons instead of channels).
#
# ### 12.5 · A hidden channel, up close
#
# The same trial and the same channel as §11.2 (trial 12, the most modulated hidden channel, 8 seconds), but now with
# the models of the block in which trial 12 is a test trial: none of them has ever seen it. In grey the firing rate
# smoothed with σ = 50 ms, which is not the ground truth (§11.2) but helps the eye; at the top the true spikes.

# %%
k12 = next(k for k in range(5) if 12 in TEST_TRIALS[k])     # the block in which trial 12 is a test trial
h12 = int(np.flatnonzero(HIDDEN == hid_show)[0])            # the column of the channel shown among the 40 hidden ones
fig, ax = plt.subplots(2, 1, figsize=(14, 4.6), sharex=True, gridspec_kw={"height_ratios": (0.25, 1)})
ax[0].eventplot(t_w[COUNTS[win_, hid_show] > 0], color="k", lineoffsets=0.5, linelengths=0.8)  # the bins with at least one spike
ax[0].set(yticks=[], title=f"Trial 12, hidden channel {hid_show} (block {k12}: the trial is a test trial); at the top the bins with spikes")
ax[1].plot(t_w, SM_RATE[win_, hid_show], color="0.6", lw=2.5, label="smoothed firing rate (σ = 50 ms), only to guide the eye")
for name, p_, c_, ls_ in (("smoothing + GLM", PRED_GLM, "0.2", "--"), ("GPFA alla pari", PRED_GP, "C0", "-"),
                          ("LFADS, seed 0", PRED_LF[0], "C3", "-")):
    ax[1].plot(t_w, p_[win_, h12] / io.BIN_S, lw=1.3, color=c_, ls=ls_, label=name)  # expected counts per bin -> spikes per second
ax[1].set(xlabel="seconds from the start of the trial", ylabel="spikes per second"); ax[1].legend(fontsize=8, ncol=2)
fig.tight_layout(); fig.savefig(FIGS / "04_lfads_hidden_example.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# **Reading.**
#
# - **All three models follow the slow trend of the channel** without having ever seen it in this trial:
#   the rise between 6 and 7.5 s, the oscillations between 8 and 10 s.
# - **LFADS is smoother**: its rates change over longer times, while smoothing + GLM and
#   GPFA also follow the more rapid oscillations.
# - **None goes down as far as the smoothed firing rate** (for example between 4.8 and 5.2 s, or around 7.8 s): the models
#   stay between 30 and 45 spikes per second. The smoothed firing rate, however, is not the ground truth (§11.2), and its
#   deepest dips may be noise.
# - **What it does not say**: it is a single channel in a single trial, chosen with a rule (the most
#   modulated hidden one), not to make a model look good; the comparison is in the numbers of §12.4.
#
# ### Summary of §12b
#
# | question | answer | where |
# |---|---|---|
# | Does LFADS predict the hidden channels of the test trials better than the equal-terms GPFA? | **no**: 0.0388 [0.0383, 0.0393] versus 0.0422 bits per spike; better only in 2 cases out of 25 (block × seed) | §12.4 |
# | and better than smoothing + GLM? | **no**: 0.0416; better only in 4 cases out of 25 | §12.4 |
# | who predicts best? | GPFA with a Poisson regression on the factors: 0.0474 | §12.4 |
# | where is LFADS better, and where worse? | the best on the channels between 3.5 and 7.8 Hz (0.0821); below zero on the slowest, below GPFA on the fastest | §12.4 |
# | do the hand computations check out? | yes: bits per spike, FA factors, restricted GPFA | §12.4 |
#
# ### The answer of §12 to the question of part B
#
# On these data, with 12 training trials, **the nonlinearity and the learned dynamics rules of LFADS do not
# improve on GPFA**: neither in phase decoding (46.5% versus 51.4%, §12.2), nor in the measure for
# generative models (0.0388 versus 0.0422 bits per spike, §12.4). LFADS gives smoother trajectories within
# trials and as consistent as those of GPFA (§12.3), and is the best on a single band of channels.
#
# **What we did not verify**: why LFADS stays below (§13: memorization, how many trials are needed;
# §14: configuration, KL weight, segment length, controller, Poisson regression on its factors);
# a single draw of the hidden channels; a single session.

# %% [markdown]
# ## 13 · The checks
#
# §12 gave a negative answer: LFADS does not improve on GPFA. Before trusting it, four checks
# on the results of §12, plus one on the amount of data:
#
# - **§13a (this part)**
#   - **13.1** how much the results change with the **seed**;
#   - **13.2** how much better the models do on **seen trials** than on new trials (memorization);
#   - **13.3** the **Poisson twin**: without any link to the task, phase decoding must
#     fall to chance;
#   - **13.4** the **time-shifted hidden channels**: without any link between channels, the score must fall to 0;
# - **§13b**: **13.5-13.7** how many trials are needed, for LFADS and for GPFA.
#
# ### 13.1 · How much the seed matters
#
# **Are the gaps of §12 larger than the variation across seeds?** We read it from the numbers already
# computed.
#
# **Do the factors change from one seed to another?** The part A measure (§5.2): the R² of the linear
# transformation that brings the factors of one seed as close as possible to those of another, on the 1106 evaluation
# points, for each ordered pair of seeds (20 pairs). R² = 1 means "the same factors, up to
# a linear transformation". With all 25 factors and with the first 3 ordered, for the models fitted on all
# trials and for those of the blocks. GPFA and FA give a single model (fixed starting point), PCA does not
# depend on chance.
#
# **Do LFADS and GPFA see the same thing?** The same measure between the two models: how well the 25 GPFA factors
# explain those of LFADS, and vice versa.
# %%
acc_lf = np.array(ACC12[("LFADS", "tetto: tutti i 25 fattori")])           # LFADS ceiling, one value per seed (§12.2)
acc_gp = ACC12[("GPFA alla pari", "tetto: tutti i 25 fattori")][0]
b_lf = np.array(BPS12[("LFADS (5 semi)", "tutta la sessione")])           # LFADS hidden channels, one value per seed (§12.4)
b_gp = BPS12[("GPFA alla pari", "tutta la sessione")][0]
print(f"Phase (ceiling): LFADS from {acc_lf.min():.1f} to {acc_lf.max():.1f} across the 5 seeds (range {np.ptp(acc_lf):.1f} points); "
      f"equal-terms GPFA {acc_gp:.1f}, {acc_gp - acc_lf.max():.1f} points above the best seed")
print(f"Hidden channels: LFADS from {b_lf.min():.4f} to {b_lf.max():.4f} (range {np.ptp(b_lf):.4f}); "
      f"equal-terms GPFA {b_gp:.4f}, {b_gp - b_lf.max():.4f} above the best seed")

cons12 = {}
for d_, lab in ((K_PCA, "25 fattori"), (3, "primi 3 fattori ordinati")):
    cons12[f"LFADS, {lab}, stimato su tutti i trial"] = run_consistency([F_LF12[("all", sd)][bins, :d_] for sd in SEEDS_LF])
    cons12[f"LFADS, {lab}, modelli dei blocchi (media sui 5)"] = np.array(
        [run_consistency([F_LF12[(k, sd)][bins, :d_] for sd in SEEDS_LF]).mean() for k in range(5)])  # one mean per block
cons12["parte A: CEBRA-Time, 3 dimensioni, stimato su tutti i trial"] = cons["CEBRA-Time, 3 dimensioni (stimato su tutti i trial)"]
cons12["parte A: UMAP, d = 3"] = cons["UMAP, d = 3"]
G_ = F_GP12["all"][bins]                                  # the 25 GPFA factors fitted on all trials, on the 1106 points
cons12["dai 25 fattori della GPFA a quelli di LFADS"] = np.array(
    [LinearRegression().fit(G_, F_LF12[("all", sd)][bins]).score(G_, F_LF12[("all", sd)][bins]) for sd in SEEDS_LF])
cons12["dai 25 fattori di LFADS a quelli della GPFA"] = np.array(
    [LinearRegression().fit(F_LF12[("all", sd)][bins], G_).score(F_LF12[("all", sd)][bins], G_) for sd in SEEDS_LF])
display(pd.Series({name: mean_range(v * 100) for name, v in cons12.items()} | {"GPFA, FA (single model), PCA": "100 (by construction)"},
                  name="R² (%), mean (min-max)").to_frame())

# %% [markdown]
# **Reading.**
#
# - **The §12 gaps do not depend on the seed.** In phase decoding the 5 LFADS seeds range from
#   44.7% to 47.7% (3.0 points), and the time-matched GPFA sits 3.7 points above the best seed. On the hidden
#   channels the seeds range from 0.0383 to 0.0393, and the GPFA sits 0.0029 above the best one: a gap three times
#   the variation between seeds.
# - **The 25 LFADS factors, taken together, are fairly stable across seeds**: the R² from one seed to another is
#   88.7% (86.7-89.5), and 87.4% with the block models.
# - **The first 3 ordered factors, instead, change a lot**: 40.4%, with pairs of seeds dropping to 29.7%;
#   also lower than the 3-dimensional maps of part A (CEBRA-Time 63.4%, UMAP 93.3%).
#   The directions that weigh most in the rates are not the same from one seed to another: together with §12.2
#   (first 3 factors at 20.9%), this says that for LFADS the "first factors" are not a reliable description.
#   What counts is the whole 25-dimensional space.
# - **The GPFA factors explain almost all of the LFADS ones (85.8%), not the other way round (66.5%).** In plain
#   words, almost everything LFADS describes is also in the GPFA, whereas about a third of what the GPFA
#   describes is not found, linearly, in LFADS. Note: the R² is the mean over the 25 dimensions,
#   each with the same weight, the weakest ones included; we did not check which parts of the GPFA are
#   missing.
# - **What it does not say**: the measures are on the 1106 evaluation points, that is, only in the movement phases,
#   and they are linear: two factors related in a nonlinear way come out as different.
#
# ### 13.2 · Seen trials and new trials
#
# At §11 LFADS predicted the hidden channels of the trials it had learned from three times better than those of new trials.
# Here is the same question for all the models of §12.4, each in its own block: the bits per spike of the 40
# hidden channels on the **12 training trials** (seen: their hidden channels were used to fit the model) and on the **3
# test trials** (new), then the mean over the 5 blocks. Every model does a little better on the seen trials; what matters is how much.
#
# Note: the "new trials" column is the mean over the 5 blocks, so it is close to but not equal to the score for the "whole
# session" of §12.4, which pools all the bins.

# %%
FULL13 = {"smoothing + GLM (reference)": [[smooth_glm_rates(train_trials(k), 0.1)] for k in range(5)],  # [block][seed] -> prediction for the whole session
          "FA": [[fa_hidden(train_trials(k))] for k in range(5)],
          "GPFA alla pari": [[np.concatenate([stitch(gpfa_hidden(SEGS[t], GP12[k])[0], t) for t in TRIALS])] for k in range(5)],
          "GPFA + regressione di Poisson": [[gpfa_glm(k)] for k in range(5)],
          "LFADS (5 semi)": [[stitch_all(res12b[(k, sd)]["rates"]) for sd in SEEDS_LF] for k in range(5)]}

def seen_new(full, Y):                                      # bits per spike on seen and on new trials, block by block: (seeds, 5 blocks, 2)
    out = np.zeros((len(full[0]), 5, 2))
    for k in range(5):
        trn_b, tst_b = np.isin(trial, train_trials(k)), np.isin(trial, TEST_TRIALS[k])  # the bins of the 12 training trials and of the 3 test trials
        for i, p_ in enumerate(full[k]):
            out[i, k] = bits_per_spike(p_[trn_b], Y[trn_b]), bits_per_spike(p_[tst_b], Y[tst_b])
    return out

SN13 = {name: seen_new(full, Y_HID) for name, full in FULL13.items()}
display(pd.DataFrame({name: {"seen trials (training)": fmt4(v[:, :, 0].mean(1)), "new trials (test)": fmt4(v[:, :, 1].mean(1)),
                             "difference": fmt4((v[:, :, 0] - v[:, :, 1]).mean(1)),
                             "new / seen": fmt4(v[:, :, 1].mean(1) / v[:, :, 0].mean(1))} for name, v in SN13.items()}).T
        .rename_axis("bits per spike, 40 hidden channels, mean over the 5 blocks; LFADS: mean [min, max] across seeds"))

# %%
fig, ax = plt.subplots(figsize=(11, 4.6))
for i, (name, v) in enumerate(SN13.items()):
    vb = v.mean(0)                                          # (5 blocks, 2): for LFADS, mean over seeds
    for k in range(5):
        ax.plot([i - 0.15, i + 0.15], vb[k], color="0.75", lw=1, marker="o", ms=3)  # one block: from seen to new trials
    ax.plot([i - 0.15, i + 0.15], vb.mean(0), color="k", lw=2.5, marker="o", ms=6, label="mean over blocks" if i == 0 else None)
ax.axhline(0, color="k", lw=0.8)
ax.set(xticks=range(len(SN13)), xticklabels=[n_.replace(" (", "\n(").replace(" + ", "\n+ ") for n_ in SN13],
       ylabel="bits per spike, 40 hidden channels", title="Left to right in each pair: seen trials (training) → new trials (test); the 5 blocks in grey")
ax.legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIGS / "04_lfads_overfit.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# **Reading.**
#
# - **All models do better on seen trials**, but not by the same amount. On new trials the following remains:
#   - about two thirds of the score for FA (0.68), time-matched GPFA (0.66) and GPFA with Poisson regression
#     (0.64);
#   - less than half for smoothing + GLM (0.44) and for LFADS (0.40 [0.39, 0.41]).
# - **LFADS has the largest difference** (0.0527 bits per spike, against 0.0482 for smoothing + GLM and 0.0197
#   for the time-matched GPFA) and the highest score on seen trials (0.0882): it is the model that learns the most from the
#   training trials, and the part that does not carry over to new trials is the largest. This is the memorization already seen
#   at §11, now on 5 blocks and on test trials.
# - **Memorization alone does not explain the ranking**: smoothing + GLM memorizes almost as much as LFADS,
#   yet does better on new trials (0.0383 against 0.0355, means over blocks). The GPFA memorizes less and on
#   new trials is on par with the reference (0.0389).
# - **The figure**: in each pair of points, the first is the score on seen trials and the second the one on new trials;
#   the grey lines of the 5 blocks are all close to the mean, so the picture is the same in every block.
# - **What it does not say**: whether the gap closes with more training trials (§13b, how many trials are needed);
#   and the seen trials here are those whose hidden channels were used to fit the model, so their
#   score is not a measure of generalization.
#
# ### 13.3 · The Poisson twin
#
# **The question**: does the 46.5% of LFADS in phase decoding really come from the link between activity and
# task, or does the LFADS → stitching → common-metric chain create it on its own? A delicate point: the segments restart at
# the beginning of every trial, and the time since trial onset alone decodes the phase at 48.5% (the clock,
# §5.3). If the stitched factors carried that clock with them, decoding would stay high even on
# data with no link to the task.
#
# **The Poisson twin.** We start from the smoothed rates (σ = 50 ms) of the 160 channels, take their
# square root and pass it through the twin of part A (§5.1): same Fourier amplitudes, random phases, **the same
# for all channels**. Then we square again and draw the counts at random from the Poisson distribution.
# The twin preserves:
#
# - the mean rate of each channel (computed by hand: random phases change neither the mean nor the variance
#   of the square root, and the mean of the square is mean² + variance);
# - the correlations between channels (same phases for all) and the speed at which activity changes over time;
# - the trial boundaries and the 600 ms segments, which are the same;
#
# but the activity no longer has any link with the task phases. The square root serves to return to positive rates
# (the twin of the square root can go below zero; its square cannot).
#
# On 3 twins (seeds 0-2) we train LFADS as in §12a (12 training trials per block, 200 epochs, LFADS seed
# = twin seed): 15 training runs. On twin 0 we also run the time-matched GPFA (5 fits) and, as a
# term of comparison, the classifier directly on the twin's rates. The phase labels are
# the true ones: **decoding must drop to chance** (12.5%).

# %%
TW_SEEDS = range(3)                                         # 3 twins
def poisson_twin(seed):                                     # Poisson twin: random-phase twin of the square root of the smoothed rates, squared, then random counts
    R_tw = phase_randomize(np.sqrt(SM_RATE), seed) ** 2     # twin rates (Hz), always positive
    return np.random.default_rng(1000 + seed).poisson(R_tw * io.BIN_S).astype(np.float32), R_tw  # counts per 20 ms bin

TWIN13 = {sd: poisson_twin(sd) for sd in TW_SEEDS}
C_tw, R_tw0 = TWIN13[0]
Q_tw0 = phase_randomize(np.sqrt(SM_RATE), 0)                # the square root of twin 0, before squaring
iu = np.triu_indices(N_CH, 1)
print(f"Mean rate of each channel, twin against data: maximum difference {np.abs(R_tw0.mean(0) - SM_RATE.mean(0)).max():.1e} Hz "
      f"(mean rates range from {SM_RATE.mean(0).min():.1f} to {SM_RATE.mean(0).max():.1f} Hz)")
print(f"Covariances between channels of the square root, twin against data: maximum difference "
      f"{np.abs(np.cov(Q_tw0.T) - np.cov(np.sqrt(SM_RATE).T)).max():.1e} (the largest is {np.abs(np.cov(np.sqrt(SM_RATE).T)).max():.2f})")
print(f"Correlations between pairs of channels of the rates: twin against data, correlation "
      f"{np.corrcoef(np.corrcoef(R_tw0.T)[iu], np.corrcoef(SM_RATE.T)[iu])[0, 1]:.3f}")
print(f"Counts: zero bins {np.mean(C_tw == 0) * 100:.1f}% in the twin, {np.mean(COUNTS == 0) * 100:.1f}% in the data; "
      f"total spikes {C_tw.sum() / COUNTS.sum():.3f} times those of the data")
print(f"Link with the real data: mean correlation, channel by channel, between twin rates and real rates "
      f"{np.mean([np.corrcoef(R_tw0[:, c], SM_RATE[:, c])[0, 1] for c in range(N_CH)]):.3f}")

# %%
SEGS_TW = {sd: {t: cut_segments(TWIN13[sd][0], t) for t in TRIALS} for sd in TW_SEEDS}  # the count segments of each twin
RUNS_TW = [(k, sd) for k in range(5) for sd in TW_SEEDS]
res_tw13 = dict(zip(RUNS_TW, run_lfads_cached([lf12_job(sd, train_trials(k), segs=SEGS_TW[sd]) for k, sd in RUNS_TW])))
F_LF_TW = {run: order_by_readout(stitch_all(r["factors"]), r["weights"]["readout.0.weight"]) for run, r in res_tw13.items()}
GP_TW = dict(zip(range(5), gpfa_cached([np.concatenate([SEGS_TW[0][t] for t in train_trials(k)]) for k in range(5)])))  # GPFA on twin 0
F_GP_TW = {k: stitch_all([gpfa_infer_segments(SEGS_TW[0][t], GP_TW[k]) for t in TRIALS]) for k in range(5)}
Z_TW = StandardScaler().fit_transform(R_tw0)                # the rates of twin 0, z-scored: the term of comparison

rows_tw = {"LFADS on the twin (3 twins)": {"first 3 factors": fmt([decode_map(fold_inp(lambda k, sd=sd: F_LF_TW[(k, sd)][:, :3]), emb_none) for sd in TW_SEEDS]),
                                             "ceiling: all dimensions": fmt([decode_map(fold_inp(lambda k, sd=sd: F_LF_TW[(k, sd)]), emb_none) for sd in TW_SEEDS])},
           "time-matched GPFA on twin 0": {"first 3 factors": fmt(decode_map(fold_inp(lambda k: F_GP_TW[k][:, :3]), emb_none)),
                                            "ceiling: all dimensions": fmt(decode_map(fold_inp(lambda k: F_GP_TW[k]), emb_none))},
           "classifier on the rates of twin 0 (160 channels)": {"ceiling: all dimensions": fmt(decode_map(fold_inp(lambda k: Z_TW), emb_none))},
           "for comparison, real data: LFADS (§12.2)": {"first 3 factors": fmt(ACC12[("LFADS", "primi 3 fattori")]), "ceiling: all dimensions": fmt(acc_lf)},
           "for comparison, real data: time-matched GPFA (§12.2)": {"first 3 factors": fmt(ACC12[("GPFA alla pari", "primi 3 fattori")]), "ceiling: all dimensions": fmt(acc_gp)}}
display(pd.DataFrame(rows_tw).T.fillna("—").rename_axis("phase decoded (%), chance 12.5"))

# %% [markdown]
# **Reading.**
#
# - **The twin is well made**: it preserves the mean rate of each channel (maximum difference 5.4·10⁻⁶ Hz,
#   the precision of 32-bit numbers) and the covariances between channels of the square root (maximum difference 6.4·10⁻⁸),
#   as expected from the hand calculation; the correlations between pairs of channels of the rates stay almost
#   the same (0.921 between the two matrices), because squaring changes them a little. The counts are similar to the
#   real ones (79.1% of zero bins against 77.3%, same number of spikes), and the twin has no link with the real data
#   (mean correlation 0.014).
# - **On the twin everything drops to chance**: LFADS 11.0% with all factors (7.8-13.9 across the 3 twins) and 11.8%
#   with the first 3; the GPFA 13.8% and 15.3%; the classifier on the twin's rates 13.8%. All around
#   12.5%, with the fluctuations of a few points expected from 1106 points and 8 classes.
# - **So the 46.5% of LFADS comes from the link between activity and task**, not from the LFADS → stitching →
#   common-metric chain. In particular the stitched factors do **not** carry the clock since trial onset with them: the
#   segments and trial boundaries in the twin are the same as in the real data, and the clock alone would be worth
#   48.5%.
# - **What it does not say**: the twin is Gaussian on the square root and then Poisson, so it rules out any
#   structure that has the same correlations and the same speed as the data, not every possible
#   nonlinear structure; the GPFA was tried on one twin only.
#
# ### 13.4 · Hidden channels shifted in time
#
# **The question**: do the positive bits per spike of §12.4 really come from the coordination between channels, or does the
# chain (model, stitching, scoring) produce them even without it? We shift in time the whole recording of
# each of the 40 hidden channels, by a different random amount for each channel (between a quarter and three quarters of the
# session, i.e. between 61 and 183 s, with wrap-around: what leaves on one side re-enters on the other). Each
# hidden channel keeps its mean rate and its time course; the visible channels stay as they are. Only the link
# between hidden and visible channels at the same instant is lost, which is exactly what is needed to
# predict them.
#
# We redo the whole of §12.4 on the shifted data: LFADS ("120 + 40" form, 3 seeds × 5 blocks), the time-matched GPFA
# (5 fits), FA, smoothing + GLM and the GPFA with Poisson regression. **On new trials the score must
# drop to about 0.** On seen trials, instead, a model that memorizes can still go above 0.
#
# A caveat: the task repeats (one trial about every 16 s), so a shifted channel may by chance fall
# in a similar task phase and retain some link with the visible channels.

# %%
rng_sh = np.random.default_rng(0)                           # fixed seed: the same shifts at every run
SHIFT = rng_sh.integers(n_bins // 4, 3 * n_bins // 4, size=len(HIDDEN))  # by how many bins each hidden channel is shifted
COUNTS_SH = COUNTS.copy()
COUNTS_SH[:, HIDDEN] = np.column_stack([np.roll(COUNTS[:, h], sh) for h, sh in zip(HIDDEN, SHIFT)])  # rotation in time, channel by channel
Y_SH = COUNTS_SH[:, HIDDEN].astype(float)                   # the shifted hidden channels: what the models must predict
SEGS_SH = {t: cut_segments(COUNTS_SH, t) for t in TRIALS}   # the segments of the shifted data, channels in the original order (GPFA)
SEGS_SH_HID = {t: SEGS_SH[t][..., ORDER_HID] for t in TRIALS}  # and with the 120 visible channels first, then the 40 hidden ones (LFADS)
SM_SH = io.smooth_rates(COUNTS_SH.astype(float), sigma_ms=50)  # smoothed firing rates of the shifted data, for the check below
R_sh = np.corrcoef(SM_SH.T)
print(f"Shifts: from {SHIFT.min() * io.BIN_S:.0f} to {SHIFT.max() * io.BIN_S:.0f} s. Highest correlation of a hidden channel with a "
      f"visible one: median {np.median(R_sh[np.ix_(HIDDEN, HELD_IN)].max(1)):.2f} (before: {np.median(best_vis):.2f})")

RUNS_SH = [(k, sd) for k in range(5) for sd in range(3)]
res_sh = dict(zip(RUNS_SH, run_lfads_cached([lf12_job(sd, train_trials(k), hp=LF_HID_HP, segs=SEGS_SH_HID, rate_cols=HID_COLS)
                                             for k, sd in RUNS_SH])))
GP_SH = dict(zip(range(5), gpfa_cached([np.concatenate([SEGS_SH[t] for t in train_trials(k)]) for k in range(5)])))
FULL_SH = {"smoothing + GLM (reference)": [[smooth_glm_rates(train_trials(k), 0.1, C=COUNTS_SH)] for k in range(5)],
           "FA": [[fa_hidden(train_trials(k), C=COUNTS_SH)] for k in range(5)],
           "GPFA alla pari": [[np.concatenate([stitch(gpfa_hidden(SEGS_SH[t], GP_SH[k])[0], t) for t in TRIALS])] for k in range(5)],
           "GPFA + regressione di Poisson": [[gpfa_glm(k, segs=SEGS_SH, gp=GP_SH, C=COUNTS_SH)] for k in range(5)],
           "LFADS (3 semi)": [[stitch_all(res_sh[(k, sd)]["rates"]) for sd in range(3)] for k in range(5)]}
SN_SH = {name: seen_new(full, Y_SH) for name, full in FULL_SH.items()}
REAL_NAME = {"LFADS (3 semi)": "LFADS (5 semi)", "GPFA + regressione di Poisson": "variante: GPFA + regressione di Poisson"}
display(pd.DataFrame({name: {"real data, new trials (§12.4)": fmt4(BPS12[(REAL_NAME.get(name, name), "tutta la sessione")]),
                             "shifted, new trials (whole session)": fmt4([bits_per_spike(test_only([full[k][i] for k in range(5)]), Y_SH)
                                                                              for i in range(len(full[0]))]),
                             "shifted, seen trials (mean over blocks)": fmt4(SN_SH[name][:, :, 0].mean(1))}
                      for name, full in FULL_SH.items()}).T.rename_axis("bits per spike, 40 hidden channels"))

# %% [markdown]
# **Reading.**
#
# - **The shift removes the link between hidden and visible channels**: the highest correlation of a hidden channel with
#   a visible one drops from 0.18 to 0.07 (medians). The shifts drawn range from 61 to 180 s.
# - **On new trials everything falls below zero**: smoothing + GLM −0.0234, LFADS −0.0179 [−0.0190, −0.0162],
#   time-matched GPFA −0.0122, GPFA with Poisson regression −0.0109, FA −0.0044. Without coordination between
#   channels no model gains anything over the mean rate: the positive scores of §12.4 really come
#   from the shared structure of the population.
# - **Below zero, not at zero**: each model learned from the training trials a link that does not exist in the new trials,
#   and applies it anyway, doing worse than predicting the mean rate. The models with the most freedom (smoothing
#   + GLM, a regression with 120 inputs per channel, and LFADS) lose the most.
# - **On seen trials memorization shows up by itself**: with shifted hidden channels LFADS reaches 0.0398 and
#   smoothing + GLM 0.0352, as much as (or more than) what LFADS achieves on new trials with the real data (0.0388).
#   A score on seen trials therefore says nothing about structure; the GPFA (0.0143) and FA (0.0055) invent
#   much less.
# - **What it does not say**: a single shift (seed 0) and 3 LFADS seeds instead of 5; and the repetition
#   of the task (one trial about every 16 s) could leave some residual link, which however here is not enough to
#   bring any model above zero.
#
# ### Summary of §13a
#
# | check | result | where |
# |---|---|---|
# | do the LFADS-GPFA gaps hold when the seed changes? | **yes**: the GPFA is above the best LFADS seed (3.7 points on phase, 0.0029 bits per spike on hidden channels) | §13.1 |
# | are the LFADS factors stable across seeds? | the 25 together yes (R² 88.7%); the first 3 ordered no (40.4%) | §13.1 |
# | do LFADS and GPFA describe the same thing? | largely: the GPFA explains 85.8% of LFADS, LFADS 66.5% of the GPFA | §13.1 |
# | does LFADS memorize? | **yes**, like smoothing + GLM: on new trials 40% of the seen-trial score remains (GPFA 66%) | §13.2 |
# | does the chain create phase decoding from nothing? | **no**: on the Poisson twin everything is at chance (LFADS 11.0%) | §13.3 |
# | does it create positive scores on hidden channels without coordination? | **no**: with hidden channels shifted in time everything is below zero on new trials | §13.4 |
#
# **What we did not check**: whether with more data LFADS reaches or exceeds the GPFA (§13b); a single
# shift of the hidden channels and a single twin for the GPFA.

# %% [markdown]
# ### 13.5 · How many trials are needed: the training runs
#
# **The question.** LFADS has many more parameters than the GPFA and with 12 training trials it memorizes (§13.2). If
# its gap to the GPFA depends on the amount of data, removing training trials should widen it,
# and the slope of the curve says whether LFADS is still improving when it reaches 12 trials: that is, whether with more data
# the verdict of §12 could change. We cannot go beyond 12 (the session has 15 trials, and 3
# are test trials), so the curve indicates a direction, not the point at which the two curves would cross.
#
# **How.** In each block the 3 test trials stay the same, and the models receive **4, 7 or 10** of the 12
# training trials (at 12: the models of §12):
#
# - the subsets are **nested** (the 4 lie within the 7, the 7 within the 10): the difference between one point
#   of the curve and the next comes from the added trials, not from a new draw;
# - they are **balanced between odd and even** (the two versions of the task): 2 + 2, 4 + 3 (or 3 + 4), 5 + 5;
# - the order in which trials are added is random (fixed seed, different for each block), so the 5 blocks
#   also give 5 different subsets.
#
# **Only what the model learns changes.** The common metric stays the same: the 10-neighbor classifier
# is always fitted on the points of the 12 training trials (with the factors given by the model trained on fewer trials) and
# is scored on the 3 test trials; the hidden channels are scored on the test trials as in §12.4. In this way the curve
# measures the representation, not the classifier.
#
# **The models**: LFADS in the two forms of §12 (all channels for the phase, 120 visible + 40 hidden for the
# hidden channels), with 3 seeds instead of 5; the time-matched GPFA and FA; for the hidden channels also smoothing + GLM and the
# GPFA with Poisson regression. All are fitted on the same reduced trials, with the settings of
# §12: the LFADS parameters are **not** chosen again for each amount of data (AutoLFADS would do that).
#
# **How long to train LFADS.** With the 200 epochs of §12, fewer trials also mean fewer training steps
# (one step = a batch of 64 segments): about 600 with 4 trials, 1600 with 12. An LFADS trained on 4 trials
# could therefore do worse because it took fewer steps, not because it saw less data. To check this,
# at 4 trials we also train a version with **the same steps as §12** (about 530 epochs). The KL
# and L2 ramps stay at 80 epochs.
#
# **A precedent.** The Neural Latents Benchmark (Pei et al., 2021) measures the same thing on three monkey
# sessions (MC_Maze) with 500, 250 and 100 training trials: AutoLFADS remains the best, with a wide margin
# over the GPFA, even with 100 trials. But there each size is a different session (in their table the 250-trial
# session gives almost all methods lower scores than the 100-trial one), a trial lasts 700 ms, the task has 27
# repeated conditions and the AutoLFADS parameters are chosen again for each dataset. Here the subsets
# come from the same session and the parameters stay those of §11b.
#
# 120 new LFADS training runs (90 at 200 epochs, 30 controls; the 30 at 12 trials are already on disk) and 15
# new GPFA fits: about 45 minutes the first time.

# %%
from itertools import zip_longest                           # to alternate odd and even

SIZES13 = (4, 7, 10, 12)                                    # how many training trials the model receives (12: the models of §12)
SEEDS13 = range(3)                                          # 3 LFADS seeds (5 in §12)

def trial_order(k):                                         # the order in which the 12 training trials of block k are added
    rng_ = np.random.default_rng(100 + k)                   # fixed seed, different for each block
    odd = [int(t) for t in rng_.permutation([t for t in train_trials(k) if t % 2 == 1])]  # the odd ones, in random order
    even = [int(t) for t in rng_.permutation([t for t in train_trials(k) if t % 2 == 0])]  # the even ones, in random order
    first, second = (odd, even) if k % 2 == 0 else (even, odd)  # blocks 0, 2, 4 start with an odd trial, the others with an even one
    return [t for pair in zip_longest(first, second) for t in pair if t is not None]  # alternating; when one group runs out, the rest of the other follows

TRIAL_ORDER = {k: trial_order(k) for k in range(5)}

def subset(k, n):                                           # the first n trials of that order, put back in time order (n = 12: train_trials(k))
    return sorted(TRIAL_ORDER[k][:n])

def n_batch(trs):                                           # batches of 64 segments in one training epoch (the last may be smaller)
    return int(np.ceil(n_seg(trs) / 64))

def ep_same_steps(k, n):                                    # the epochs with which n trials take at least the steps of §12 (200 epochs with the 12 trials)
    return int(np.ceil(N_EP_LF * n_batch(train_trials(k)) / n_batch(subset(k, n))))

def mm(v):                                                  # "min-max" of a series of integers (a single number if they are all equal)
    return f"{min(v)}" if min(v) == max(v) else f"{min(v)}-{max(v)}"

for k in range(5):
    print(f"Block {k}: test trials {[int(t) for t in TEST_TRIALS[k]]}; order of the 12 training trials {TRIAL_ORDER[k]}")
print("Annidati:", all(set(subset(k, a)) <= set(subset(k, b)) for k in range(5) for a, b in zip(SIZES13, SIZES13[1:])),
      "| at 12 trials the same as §12:", all(subset(k, 12) == train_trials(k) for k in range(5)))
display(pd.DataFrame({f"{n} trial": {"odd + even, blocks 0-4": ", ".join(f"{sum(t % 2 for t in subset(k, n))}+{sum(t % 2 == 0 for t in subset(k, n))}"
                                                                              for k in range(5)),
                                     "seconds of recording": mm([int(np.isin(trial, subset(k, n)).sum() * io.BIN_S) for k in range(5)]),
                                     "600 ms segments": mm([n_seg(subset(k, n)) for k in range(5)]),
                                     "steps in 200 epochs": mm([N_EP_LF * n_batch(subset(k, n)) for k in range(5)]),
                                     "epochs to match the steps of §12": mm([ep_same_steps(k, n) for k in range(5)])} for n in SIZES13}))

# %%
FORMS13 = {"tutti i canali": {}, "120 + 40": dict(hp=LF_HID_HP, segs=SEGS_HID, rate_cols=HID_COLS)}  # the two forms of §12: phase (§12a), hidden channels (§12b)

def job13(form, n, k, sd, rule):                            # one training run: form, number of trials, block, seed, rule
    j_ = lf12_job(sd, subset(k, n), **FORMS13[form])        # as in §12, but on the reduced trials
    if rule == "passi del §12":                             # the control: more epochs, until the steps match those of §12
        j_.update(max_epochs=ep_same_steps(k, n), patience=ep_same_steps(k, n))
    return j_

RUNS13 = ([(f_, n, k, sd, "200 epoche") for f_ in FORMS13 for n in SIZES13 for k in range(5) for sd in SEEDS13]  # 120 (the 30 at 12 trials are from §12)
          + [(f_, 4, k, sd, "passi del §12") for f_ in FORMS13 for k in range(5) for sd in SEEDS13])  # 30 controls, at 4 trials
res13 = dict(zip(RUNS13, run_lfads_cached([job13(*r) for r in RUNS13])))

KEYS13 = [(n, k) for n in SIZES13 for k in range(5)]        # (number of trials, block)
GP13 = dict(zip(KEYS13, gpfa_cached([np.concatenate([SEGS[t] for t in subset(k, n)]) for n, k in KEYS13])))  # at 12 trials: the GPFAs of §12

st13 = pd.DataFrame([{"trial": n, "rule": rule, "epochs": r["epochs"], "steps": r["steps"], "prima della fine": r["epochs"] - r["best_epoch"]}
                     for (f_, n, k, sd, rule), r in res13.items()])
display(pd.DataFrame({f"{n} trial, {rule}": {"trainings": len(g_), "epochs": mm(g_["epochs"]), "steps": mm(g_["steps"]),
                                             "kept version: epochs before the end (maximum)": int(g_["prima della fine"].max())}
                      for (rule, n), g_ in st13.groupby(["rule", "trial"], sort=False)}).T)

# %% [markdown]
# **Reading.**
#
# - **The subsets are as intended**: nested, with 2 + 2 odd and even trials at 4 trials, 4 + 3 or 3 + 4 at
#   7, 5 + 5 at 10; at 12 they are exactly the trials of §12. In seconds of recording: about 65 s with 4 trials,
#   115 with 7, 160 with 10, 195 with 12.
# - **The training steps**: with 200 epochs, 600 at 4 trials, 1000 at 7, 1400 at 10, 1600 at 12. The control at
#   4 trials runs 534 epochs, that is 1602 steps, like §12.
# - **The kept version is in practice the last one** in all 150 models: at most 6 epochs before the end
#   with 200 epochs, 9 with 534. As in §12, the fixed epochs are really fixed.
# - The 120 new training runs and the 15 GPFAs took 47 minutes; the 30 models at 12 trials and the 5 GPFAs of
#   §12 were read back from disk.
#
# ### 13.6 · Phase, with fewer trials
#
# The common metric of §12.2 on the ordered factors of each model: the ceiling (all 25 factors) and the first 3.
# At 12 trials the numbers must be exactly those of §12.2 (for LFADS seeds 0-2): this checks that the
# subsets and the cache work as intended.

# %%
F_LF13 = {run: order_by_readout(stitch_all(r["factors"]), r["weights"]["readout.0.weight"])  # ordered factors, whole session
          for run, r in res13.items() if run[0] == "tutti i canali"}
F_GP13 = {nk: stitch_all([gpfa_infer_segments(SEGS[t], GP13[nk]) for t in TRIALS]) for nk in KEYS13}
F_FA13 = {(n, k): fa_factors(subset(k, n)) for n, k in KEYS13}

GET13 = {}                                                  # (model, number of trials) -> for each seed, a block -> factors function
for n in SIZES13:
    GET13[("FA", n)] = [lambda k, n=n: F_FA13[(n, k)]]
    GET13[("GPFA alla pari", n)] = [lambda k, n=n: F_GP13[(n, k)]]
    GET13[("LFADS (3 semi)", n)] = [lambda k, n=n, sd=sd: F_LF13[("tutti i canali", n, k, sd, "200 epoche")] for sd in SEEDS13]
GET13[("LFADS, passi del §12", 4)] = [lambda k, sd=sd: F_LF13[("tutti i canali", 4, k, sd, "passi del §12")] for sd in SEEDS13]
MEAS13 = {"tetto: tutti i 25 fattori": K_PCA, "primi 3 fattori": 3}  # how many columns of the ordered factors the classifier sees
ACC13 = {(name, n, meas): [decode_map(fold_inp(lambda k, g=g, d_=d_: g(k)[:, :d_]), emb_none) for g in gs]
         for (name, n), gs in GET13.items() for meas, d_ in MEAS13.items()}

NAME12 = {"FA": "FA", "GPFA alla pari": "GPFA alla pari", "LFADS (3 semi)": "LFADS"}  # the names used in §12.2
d_chk = max(np.abs(np.array(ACC13[(nm, 12, meas)]) - np.array(ACC12[(nm12, meas)][:len(SEEDS13) if nm12 == "LFADS" else 1])).max()
            for nm, nm12 in NAME12.items() for meas in MEAS13)
print(f"Check: at 12 trials, maximum difference from the numbers of §12.2: {d_chk:.1e} points")

def fmt1s(v):                                               # "mean [min, max]" with one decimal (square brackets because it can be negative)
    v = np.atleast_1d(v)
    return f"{v.mean():.1f}" if len(v) == 1 else f"{v.mean():.1f} [{v.min():.1f}, {v.max():.1f}]"

NAMES_PH13 = ["FA", "GPFA alla pari", "LFADS (3 semi)", "LFADS, passi del §12"]
tab13 = pd.DataFrame({(meas, name): {f"{n} trial": fmt(ACC13[(name, n, meas)]) if (name, n, meas) in ACC13 else "—" for n in SIZES13}
                      | {"from 4 to 12 trials": f"{np.mean(ACC13[(name, 12, meas)]) - np.mean(ACC13[(name, 4, meas)]):+.1f}"
                         if (name, 12, meas) in ACC13 else "—"}
                      for meas in MEAS13 for name in NAMES_PH13}).T
display(tab13.rename_axis(["phase decoded (%), chance 12.5", "model"]))
for meas in MEAS13:
    gap_ = {n: ACC13[("GPFA alla pari", n, meas)][0] - np.array(ACC13[("LFADS (3 semi)", n, meas)]) for n in SIZES13}
    print(f"Time-matched GPFA minus LFADS, {meas} (points; [min, max] across LFADS seeds): "
          + "; ".join(f"{n} trial {fmt1s(v)}" for n, v in gap_.items()))

# %% [markdown]
# **Reading.**
#
# - **The check passes**: at 12 trials the same numbers as §12.2 (difference 0). With seeds 0-2 the LFADS
#   ceiling is 47.0 (45.9-47.7); with 5 seeds it was 46.5.
# - **With more trials everything improves a little, LFADS more than the others**: from the ceiling at 4 trials to that at 12,
#   LFADS gains 6.3 points (from 40.7 to 47.0), the time-matched GPFA 4.7 (from 46.7 to 51.4), FA 2.7.
# - **But LFADS stays below the GPFA at every amount of data**: the gap is 6.0 points at 4 trials, 7.2 at 7, 5.2 at
#   10 and 4.5 at 12, and no LFADS seed reaches the GPFA (the smallest gap, seed by seed, is 3.7
#   points). From 7 to 12 trials it shrinks by 2.7 points, but not regularly: the differences between neighboring points
#   are of the order of the variation between seeds (up to 3.5 points).
# - **The GPFA needs little data**: with 4 trials, about 65 seconds of recording, it decodes the phase for
#   46.7% of the points, as much as LFADS with 12 trials (47.0%).
# - **The first 3 ordered LFADS factors do not improve with data**: they stay between 20 and 23%, against 30-32% for the
#   GPFA, a gap of about 10 points at every amount. This confirms §13.1: for LFADS the "first factors"
#   do not describe the useful part.
# - **The 200-epoch rule does not penalize LFADS with few trials**: at 4 trials, with the steps of §12, it reaches
#   39.8 on the ceiling (against 40.7) and 19.0 on the first 3 (against 23.1). More steps on the same few data do not
#   help: the curve is not steeper because of training that is too short.
# - **What it does not say**: where the two curves would meet with more than 12 trials (we do not extrapolate); the
#   points of the curve are not independent (the subsets are nested) and each block has a single
#   subset per amount of data.
#
# ### 13.7 · The hidden channels, with fewer trials
#
# The §12.4 score: bits per spike of the 40 hidden channels, on the session made of the test trials only, each
# predicted by the model of its block. Each model learns the link between visible and hidden channels only from the reduced
# trials. Here too, at 12 trials the numbers must be those of §12.4. In addition we count in how many cases
# (block × seed) LFADS beats the time-matched GPFA on the 3 test trials of the block.

# %%
def gpfa_hidden_test(gp_of):                                # expected counts of the hidden channels of each test trial, from the GPFA of its block (gp_of: block -> parameters)
    pieces = {int(t): gpfa_hidden(SEGS[t], gp_of(k))[0] for k in range(5) for t in TEST_TRIALS[k]}
    return np.concatenate([stitch(pieces[int(t)], t) for t in TRIALS])  # stitched and concatenated: the session made of test trials only

PRED13 = {}                                                 # (model, number of trials) -> predictions of the 40 hidden channels, one per seed
for n in SIZES13:
    PRED13[("smoothing + GLM (reference)", n)] = [test_only([smooth_glm_rates(subset(k, n), 0.1) for k in range(5)])]
    PRED13[("FA", n)] = [test_only([fa_hidden(subset(k, n)) for k in range(5)])]
    PRED13[("GPFA alla pari", n)] = [gpfa_hidden_test(lambda k, n=n: GP13[(n, k)])]
    PRED13[("GPFA + regressione di Poisson", n)] = [test_only([gpfa_glm(k, gp={k: GP13[(n, k)]}, trs=subset(k, n)) for k in range(5)])]
    PRED13[("LFADS (3 semi)", n)] = [test_only([stitch_all(res13[("120 + 40", n, k, sd, "200 epoche")]["rates"]) for k in range(5)])
                                     for sd in SEEDS13]
PRED13[("LFADS, passi del §12", 4)] = [test_only([stitch_all(res13[("120 + 40", 4, k, sd, "passi del §12")]["rates"]) for k in range(5)])
                                       for sd in SEEDS13]
BPS13 = {key: [bits_per_spike(p_, Y_HID) for p_ in P] for key, P in PRED13.items()}  # bits per spike, one per seed

NAME12B = {"smoothing + GLM (reference)": "smoothing + GLM (reference)", "FA": "FA", "GPFA alla pari": "GPFA alla pari",
           "GPFA + regressione di Poisson": "variante: GPFA + regressione di Poisson", "LFADS (3 semi)": "LFADS (5 semi)"}  # the names used in §12.4
d_chk_b = max(np.abs(np.array(BPS13[(nm, 12)]) - np.array(BPS12[(nm12, "tutta la sessione")][:len(BPS13[(nm, 12)])])).max()
              for nm, nm12 in NAME12B.items())
print(f"Check: at 12 trials, maximum difference from the numbers of §12.4: {d_chk_b:.1e} bits per spike")

NAMES_B13 = list(NAME12B) + ["LFADS, passi del §12"]
display(pd.DataFrame({name: {f"{n} trial": fmt4(BPS13[(name, n)]) if (name, n) in BPS13 else "—" for n in SIZES13}
                            | {"from 4 to 12 trials": f"{np.mean(BPS13[(name, 12)]) - np.mean(BPS13[(name, 4)]):+.4f}" if (name, 12) in BPS13 else "—"}
                      for name in NAMES_B13}).T.rename_axis("bits per spike, 40 hidden channels, test trials; LFADS: mean [min, max] across seeds"))

def bps_blocks(P):                                          # bits per spike on the 3 test trials of each block: (seeds, 5 blocks)
    return np.array([[bits_per_spike(p_[np.isin(trial, TEST_TRIALS[k])], Y_HID[np.isin(trial, TEST_TRIALS[k])]) for k in range(5)] for p_ in P])

for n in SIZES13:
    gap_ = BPS13[("GPFA alla pari", n)][0] - np.array(BPS13[("LFADS (3 semi)", n)])
    above = bps_blocks(PRED13[("LFADS (3 semi)", n)]) > bps_blocks(PRED13[("GPFA alla pari", n)])  # (3 seeds, 5 blocks)
    print(f"{n:2d} trials: time-matched GPFA minus LFADS {fmt4(gap_)} bits per spike; LFADS above the time-matched GPFA in "
          f"{int(above.sum())} cases out of {above.size} (block × seed)")

# %%
COL13 = {"smoothing + GLM (reference)": "0.45", "FA": "C1", "GPFA alla pari": "C0", "GPFA + regressione di Poisson": "C2", "LFADS (3 semi)": "C3"}
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
curves = ((ax[0], ["FA", "GPFA alla pari", "LFADS (3 semi)"], lambda name, n: ACC13[(name, n, "tetto: tutti i 25 fattori")],
           "phase decoded (%), ceiling: 25 factors (chance 12.5)", "Phase decoding, test trials"),
          (ax[1], list(COL13), lambda name, n: BPS13[(name, n)], "bits per spike, 40 hidden channels", "Hidden channels of the test trials"))
for a_, names, get_, ylab, title in curves:
    for name in names:
        v = np.array([get_(name, n) for n in SIZES13])      # (4 amounts of trials, seeds)
        a_.plot(SIZES13, v.mean(1), color=COL13[name], marker="o", lw=2, label=name)
        if v.shape[1] > 1:
            a_.fill_between(SIZES13, v.min(1), v.max(1), color=COL13[name], alpha=0.2)  # min-max across seeds
    v4 = np.array(get_("LFADS, passi del §12", 4))           # the 4-trial control, shifted slightly to the right so as not to cover the point
    a_.errorbar(4.3, v4.mean(), yerr=[[v4.mean() - v4.min()], [v4.max() - v4.mean()]], color="C3", marker="s", mfc="white",
                ls="none", capsize=3, label="LFADS at 4 trials, with the steps of §12")
    a_.set(xticks=SIZES13, xlabel="training trials given to the model (out of 12)", ylabel=ylab, title=title)
ax[1].axhline(0, color="k", lw=0.8)
ax[0].legend(fontsize=8); ax[1].legend(fontsize=8, ncol=2)
fig.tight_layout(); fig.savefig(FIGS / "04_lfads_trials.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# **Reading.**
#
# - **The check passes**: at 12 trials the same numbers as §12.4 (difference 0).
# - **On the hidden channels the gap between GPFA and LFADS shrinks regularly**: 0.0114 bits per spike at 4
#   trials, 0.0084 at 7, 0.0050 at 10, 0.0035 at 12, about 0.001 for each additional trial (a little less in the last ones).
#   LFADS gains twice as much as the GPFA (+0.0159 against +0.0080 from 4 to 12 trials). However it stays below at every
#   amount: it beats the time-matched GPFA in 0 cases out of 15 (block × seed) at 4, 7 and 10 trials, in 1 out of 15 at 12.
# - **The ranking changes with the data, and it is instructive**:
#   - **smoothing + GLM**, which has 120 free coefficients for each hidden channel, is the one that suffers most
#     with little data (0.0096 at 4 trials) and gains most (+0.0321): it is below LFADS at 4 trials, on par at 7
#     (0.0306), above at 10 and 12;
#   - **LFADS beats FA only from 10 trials up** (0.0358 against 0.0347): with less data the linear model
#     without dynamics does better;
#   - the **GPFA with Poisson regression** remains the best at every amount of data, and the time-matched GPFA
#     the second among the factor models.
#
#   In plain words: the more freedom a model has, the more data it needs. LFADS sits in the middle, between the linear
#   factor models, which already do well with little data, and channel-by-channel regression.
# - **Here too the 200-epoch rule does not penalize LFADS**: at 4 trials, with the steps of §12, it drops from
#   0.0228 to 0.0142. More steps on the same data mean more memorization (§13.2), and on new
#   trials this has a cost.
# - **The figure**: phase on the left, hidden channels on the right; the red band is the min-max across the 3 seeds
#   of LFADS, the empty square the control with the steps of §12.
# - **What it does not say**: whether with more than 12 trials LFADS would reach the GPFA. The trend on the hidden channels
#   says the gap shrinks, the one on the phase is less clear; to find out more data are needed, that is the
#   other sessions (§14).
#
# ### Summary of §13b
#
# | question | result | where |
# |---|---|---|
# | with fewer trials does LFADS lose more than the GPFA? | **yes**: from 4 to 12 trials LFADS gains 6.3 points on phase and 0.016 bits per spike on hidden channels, the GPFA 4.7 points and 0.008 | §13.6, §13.7 |
# | does the gap close within 12 trials? | **no**: LFADS stays below the GPFA at every amount of data (phase: from 4.5 to 7.2 points; hidden channels: from 0.0114 to 0.0035) | §13.6, §13.7 |
# | does the 200-epoch rule penalize LFADS with few trials? | **no**: with the same steps as §12, at 4 trials it does worse (phase 39.8 against 40.7; hidden channels 0.0142 against 0.0228) | §13.5-§13.7 |
# | how much data does the GPFA need? | little: with 4 trials (about 65 s) it decodes the phase as well as LFADS with 12 (46.7 against 47.0) | §13.6 |
#
# **§13 as a whole**: the negative result of §12 is not an artifact (seed, twin, shift of the hidden
# channels); LFADS memorizes the training trials more than the GPFA; with more trials the gap shrinks, but
# within the data of this session it does not close.
#
# **What we did not check**: more than 12 training trials (the other two sessions, §14); a single
# trial order per block; the LFADS parameters, the GLM penalty and the number of GPFA factors were not
# chosen again for each amount of data.

# %% [markdown]
# ## 14 · Sensitivity and alternatives
#
# §12 gave a negative answer and §13 checked that it is not an artifact. One question remains: does the
# answer depend on the **choices** made for LFADS? The configuration comes from a small grid on the validation trials
# of block 0 (§11b); the network sizes, the segment length and the absence of the
# controller were fixed by hand. Here we change **one thing at a time**, leaving everything else as in
# §12, and redo the two measures.
#
# - **§14a (this part)**: the sizes (factors, generator) and the regularization that the grid did not
#   explore (KL weight × 0.3 lower, L2 penalty on the generator);
# - **§14b**: the segment length and the controller;
# - **§14c**: LFADS without dynamics (the "nonlinear FA") and Poisson regression on the LFADS factors;
# - **§14d**: more data, with the other two sessions of the same day.
#
# **The scheme of each variant** is that of §12: 12 training trials per block, 5 blocks, 200 epochs,
# the two forms (all channels for the phase, 120 + 40 for the hidden channels), with **seeds 0-2**. The term of
# comparison is the §12 configuration with the same seeds (the models are already on disk): each variant is
# compared with the main one **seed by seed**.
#
# **The measures**: the phase with the common metric (all factors, i.e. the ceiling, and the first 3 ordered); the bits
# per spike of the 40 hidden channels on the test trials (§12.4); how much of the hidden-channel score remains going from seen
# to new trials (§13.2), because memorization is the main suspect.
#
# **The rule, fixed before looking**: the §12 verdict changes if a variant brings LFADS **above the time-matched
# GPFA with the same number of factors**, with all 3 seeds, on one of the two measures (phase ceiling or
# hidden channels). For the variants with 10 and 40 factors we also fit the time-matched GPFA with 10 and 40 factors.

# %% [markdown]
# ### 14.1 · Sizes and regularization
#
# | variant | what changes (in §12) | why try it |
# |---|---|---|
# | factors 10 | 10 factors (25) | fewer dimensions in which to describe the activity: less room to memorize |
# | factors 40 | 40 factors (25) | more room, in case 25 were too few for a nonlinear model |
# | generator 32 | 32 units in the generator (64) | a smaller network, with fewer weights to learn from little data |
# | generator 128 | 128 units (64) | a larger network: richer dynamics, if the data contain them |
# | KL weight × 0.1 | 0.1 / (30 × 160) (0.3 / (30 × 160)) | the chosen value was at the edge of the grid (§11b) |
# | L2 on the generator × 10 | 10⁻¹ (10⁻²) | smaller recurrent weights: simpler dynamics, hence less memorization |
#
# We do not change the dropout: in the §11b grid its maximum fell in the interior (0.05 better than 0,
# 0.02 and 0.25). The encoder and the initial condition stay at 64.
#
# 180 new training runs (6 variants × 2 forms × 5 blocks × 3 seeds; the 30 of the main one are from §12) and 10
# GPFA fits (10 and 40 factors): about 1 hour 35 minutes the first time. A 10-epoch trial run (4
# at a time) gave about 10% more time for the 128-unit generator, and for the other variants the same
# time as the main one.

# %%
SEEDS14 = range(3)                                          # seeds 0-2, the same as §13b
VARIANTS14A = {"principale (§12)": {},                       # what changes with respect to the §12 configuration
               "fattori 10": dict(fac_dim=10),
               "fattori 40": dict(fac_dim=40),
               "generatore 32": dict(gen_dim=32),
               "generatore 128": dict(gen_dim=128),
               "peso del KL × 0.1": dict(kl_ic_scale=LF_HP["kl_ic_scale"] / 3),  # from 0.3 to 0.1 times the true ELBO
               "L2 sul generatore × 10": dict(l2_gen_scale=LF_HP["l2_gen_scale"] * 10)}
FORMS14 = {"tutti i canali": (LF_HP, {}),                   # (starting configuration, data) of the two forms of §12
           "120 + 40": (LF_HID_HP, dict(segs=SEGS_HID, rate_cols=HID_COLS))}

def job14(var, form, k, sd):                                # one training run: variant, form, block, seed; the rest as in §12
    hp_, data_ = FORMS14[form]
    return lf12_job(sd, train_trials(k), hp=dict(hp_, **VARIANTS14A[var]), **data_)

RUNS14A = [(v, f_, k, sd) for v in VARIANTS14A for f_ in FORMS14 for k in range(5) for sd in SEEDS14]
res14a = dict(zip(RUNS14A, run_lfads_cached([job14(*r) for r in RUNS14A])))

GP14 = {d_: dict(zip(range(5), gpfa_cached([np.concatenate([SEGS[t] for t in train_trials(k)]) for k in range(5)], x_dim=d_)))
        for d_ in (10, 40)}                                 # the time-matched GPFA with 10 and 40 factors
GP14[K_PCA] = {k: GP12[k] for k in range(5)}                # with 25: the one from §12

def shapes(w):                                              # the sizes of a model read from its weights: factors, generator
    return w["readout.0.weight"].shape[1], w["decoder.rnn.cell.gen_cell.weight_hh"].shape[1]

display(pd.DataFrame({v: {"factors, generator (from the weights)": ", ".join(sorted({"{}, {}".format(*shapes(res14a[(v, f_, k, sd)]["weights"]))
                                                                             for f_ in FORMS14 for k in range(5) for sd in SEEDS14})),
                          "kept version (epoch)": mm([res14a[(v, f_, k, sd)]["best_epoch"] for f_ in FORMS14 for k in range(5) for sd in SEEDS14])}
                      for v in VARIANTS14A}).T)

# %% [markdown]
# **Reading.**
#
# - **The variants are the intended ones**: the saved weights have 10 and 40 factors, a generator of 32 and 128 units;
#   the KL weight and the L2 penalty are not visible in the weights, but are part of the configuration and hence
#   of the fingerprint of each model (§10.5).
# - **The kept version is in practice the last one** in all 210 models (between epoch 193 and 200), as in §12.
# - The 30 models of the main configuration and the 25-factor GPFAs were read back from disk. The duration of this
#   run is not indicative: the computer was running other notebooks at the same time and went to sleep for
#   almost two hours (the run then resumed on its own). Without other computations the estimate written above holds.
#
# ### 14.2 · The two measures, variant by variant

# %%
F_LF14 = {run: order_by_readout(stitch_all(r["factors"]), r["weights"]["readout.0.weight"])  # ordered factors, whole session
          for run, r in res14a.items() if run[1] == "tutti i canali"}
F_GP14 = {d_: {k: stitch_all([gpfa_infer_segments(SEGS[t], GP14[d_][k]) for t in TRIALS]) for k in range(5)} for d_ in (10, 40)}
F_GP14[K_PCA] = {k: F_GP12[k] for k in range(5)}

def acc14(get):                                             # phase decoded: (ceiling, first 3) for a block -> factors function
    return decode_map(fold_inp(get), emb_none), decode_map(fold_inp(lambda k: get(k)[:, :3]), emb_none)

def full_lf14(var):                                         # [block][seed] -> expected counts of the 40 hidden channels over the whole session (120 + 40 form)
    return [[stitch_all(res14a[(var, "120 + 40", k, sd)]["rates"]) for sd in SEEDS14] for k in range(5)]

def full_gp14(d_):                                          # the same for the time-matched GPFA with d_ factors (one model per block)
    if d_ == K_PCA:
        return FULL13["GPFA alla pari"]                     # with 25 factors: already computed in §13.2
    return [[np.concatenate([stitch(gpfa_hidden(SEGS[t], GP14[d_][k])[0], t) for t in TRIALS])] for k in range(5)]

def fmt2(v):                                                # "mean [min, max]" with 2 decimals
    v = np.atleast_1d(v)
    return f"{v.mean():.2f}" if len(v) == 1 else f"{v.mean():.2f} [{v.min():.2f}, {v.max():.2f}]"

M14 = {}                                                    # name -> dict of measures, each a list (one value per seed)
for v in VARIANTS14A:
    ph = [acc14(lambda k, sd=sd: F_LF14[(v, "tutti i canali", k, sd)]) for sd in SEEDS14]
    full = full_lf14(v)
    sn = seen_new(full, Y_HID)                              # (seeds, 5 blocks, 2): bits per spike on seen and new trials (§13.2)
    M14[v] = {"ceiling": [a for a, _ in ph], "primi 3": [b for _, b in ph],
              "hidden": [bits_per_spike(test_only([full[k][i] for k in range(5)]), Y_HID) for i in range(len(SEEDS14))],
              "nuovi / visti": list(sn[:, :, 1].mean(1) / sn[:, :, 0].mean(1))}
for d_ in (10, K_PCA, 40):
    a_, b_ = acc14(lambda k, d_=d_: F_GP14[d_][k])
    full = full_gp14(d_)
    sn = seen_new(full, Y_HID)
    M14[f"GPFA alla pari, {d_} fattori"] = {"ceiling": [a_], "primi 3": [b_],
                                            "hidden": [bits_per_spike(test_only([full[k][0] for k in range(5)]), Y_HID)],
                                            "nuovi / visti": [sn[0, :, 1].mean() / sn[0, :, 0].mean()]}

GP_SAME = {v: f"GPFA alla pari, {VARIANTS14A[v].get('fac_dim', K_PCA)} fattori" for v in VARIANTS14A}  # the GPFA with the same factors
chk14 = max(abs(np.mean(M14["principale (§12)"]["ceiling"]) - np.mean(ACC13[("LFADS (3 semi)", 12, "tetto: tutti i 25 fattori")])),
            abs(np.mean(M14["principale (§12)"]["hidden"]) - np.mean(BPS13[("LFADS (3 semi)", 12)])),
            abs(M14[f"GPFA alla pari, {K_PCA} fattori"]["hidden"][0] - BPS12[("GPFA alla pari", "tutta la sessione")][0]))
print(f"Check: the main configuration and the 25-factor GPFA give the same numbers as §13b and §12.4 (maximum difference {chk14:.1e})")

display(pd.DataFrame({name: {"phase, ceiling (%)": fmt(m["ceiling"]), "phase, first 3 (%)": fmt(m["primi 3"]),
                             "hidden, test trials (bits per spike)": fmt4(m["hidden"]), "new / seen": fmt2(m["nuovi / visti"])}
                      for name, m in M14.items()}).T.rename_axis("mean (min-max) across the 3 seeds; chance 12.5%"))

P14 = M14["principale (§12)"]
display(pd.DataFrame({v: {"ceiling: difference from the main configuration (points)": fmt1s(np.subtract(M14[v]["ceiling"], P14["ceiling"])),
                          "hidden: difference from the main configuration": fmt4(np.subtract(M14[v]["hidden"], P14["hidden"])),
                          "GPFA with the same factors": GP_SAME[v].replace("GPFA alla pari, ", ""),
                          "seeds above the GPFA: ceiling": f"{int(np.sum(np.array(M14[v]['ceiling']) > M14[GP_SAME[v]]['ceiling'][0]))} of 3",
                          "seeds above the GPFA: hidden": f"{int(np.sum(np.array(M14[v]['hidden']) > M14[GP_SAME[v]]['hidden'][0]))} of 3"}
                      for v in VARIANTS14A}).T.rename_axis("seed by seed; [min, max] across the 3 seeds"))

# %%
ROWS14 = list(VARIANTS14A)[::-1]                            # from top to bottom as in the table
fig, ax = plt.subplots(1, 2, figsize=(15, 4.8), sharey=True)
for a_, meas, xlab in ((ax[0], "ceiling", "phase decoded (%), all factors"), (ax[1], "hidden", "bits per spike, 40 hidden channels, test trials")):
    for y, v in enumerate(ROWS14):
        vals = np.array(M14[v][meas])
        a_.scatter(vals, np.full(len(vals), y), color="C3", s=18, alpha=0.6, zorder=3)  # the 3 seeds
        a_.scatter(vals.mean(), y, color="C3", s=70, marker="D", zorder=4, label="LFADS: mean of the 3 seeds (small points: the seeds)" if y == 0 else None)
        g_ = M14[GP_SAME[v]][meas][0]
        a_.scatter(g_, y, color="C0", s=70, marker="|", linewidths=3, zorder=4, label="time-matched GPFA with the same factors" if y == 0 else None)
    a_.axvline(np.mean(P14[meas]), color="C3", ls=":", lw=1)  # the main configuration, for comparison
    a_.set(xlabel=xlab, yticks=range(len(ROWS14)), yticklabels=ROWS14)
    a_.grid(axis="y", alpha=0.3)
ax[0].legend(fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.13))  # below the plot, so as not to cover the points
ax[0].set_title("Phase (dashed line: the main configuration)"); ax[1].set_title("Hidden channels of the test trials")
fig.tight_layout(); fig.savefig(FIGS / "04_lfads_variants_a.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# **Reading.**
#
# - **The check passes**: the main configuration and the 25-factor GPFA give exactly the numbers of §13b and §12.4.
# - **No variant changes the verdict**, according to the rule fixed before looking: none brings LFADS
#   above the time-matched GPFA with the same factors with all 3 seeds, either on phase or on hidden channels. The closest
#   case is the one with 10 factors on the hidden channels: one seed out of three beats the 10-factor GPFA by about 0.0001
#   bits per spike, but the mean stays below (0.0321 against 0.0329).
# - **Three variants make LFADS worse in all 3 seeds**, seed by seed with respect to the main configuration:
#   - **10 factors**: −5.7 points on phase and −0.0066 bits per spike on the hidden channels. There are too few
#     dimensions; but the GPFA also suffers (44.7% and 0.0329 with 10 factors), so the two get closer;
#   - **generator of 128**: −1.7 points and −0.0071. This is the variant that memorizes most: on new trials only
#     25% of the seen-trial score remains (40% for the main configuration);
#   - **KL weight × 0.1**: −1.6 points and −0.0033, and it memorizes more (33%). The value chosen in §11b was at the
#     edge of the grid: going beyond, toward less regularization, is worse, not better.
# - **Two variants change little**: 40 factors (+0.1 points, −0.0007, memorizes more: 35%) and generator of 32
#   (−1.1 points, −0.0005: memorizes less, 48%, but does not gain on new trials).
# - **The only one that does a little better is L2 on the generator × 10**: +0.5 points on phase in all 3 seeds
#   (from +0.2 to +0.8) and +0.0006 on the hidden channels (from about 0 to +0.0013), with a little less memorization (42%). The
#   gain is however small compared with the gap to the GPFA, which remains 4.0 points and 0.0029 bits per spike.
#   It is the highest value we tried: we do not know about even higher values.
# - **Memorizing more never helps**: the variants with more capacity or less regularization (generator of
#   128, KL × 0.1, 40 factors) retain 25-35% of the seen-trial score on new trials, and do worse or
#   the same on the hidden channels; those with less capacity (generator of 32, 10 factors) memorize less (47-48%) but
#   do not do better. The GPFA retains 62-72%.
# - **The comparison is fair to the GPFA too**: with 40 factors it reaches 51.1% and 0.0435 (a little better on the hidden channels
#   than with 25), with 10 factors worse (44.7% and 0.0329). The 25 factors of §12 do not favor it, and LFADS with
#   40 factors does as well as with 25.
# - **The first 3 ordered LFADS factors stay between 19 and 21%** in all variants, against 31-34% for the
#   GPFA: no variant resolves this point (§12.2, §13.1).
# - **The figure**: for each variant the 3 LFADS seeds (small points) and their mean (diamond), and the time-matched
#   GPFA with the same factors (blue tick); the dashed line is the main configuration.
# - **What it does not say**: we changed one knob at a time, so combinations (for example fewer
#   factors and more L2 together) are not tested; neither are L2 values above × 10; the starting
#   configuration comes from block 0 (§11b); 3 seeds per variant.
#
# ### Summary of §14a
#
# | variant | phase, ceiling: difference from the main configuration (points) | hidden: difference (bits per spike) | what remains on new trials (new / seen) | above the GPFA with the same factors? |
# |---|---|---|---|---|
# | main (§12) | 47.0% | 0.0387 | 0.40 | no |
# | factors 10 | −5.7 | −0.0066 | 0.47 | no (on hidden channels 1 seed out of 3, by about 0.0001) |
# | factors 40 | +0.1 | −0.0007 | 0.35 | no |
# | generator 32 | −1.1 | −0.0005 | 0.48 | no |
# | generator 128 | −1.7 | −0.0071 | 0.25 | no |
# | KL weight × 0.1 | −1.6 | −0.0033 | 0.33 | no |
# | L2 on the generator × 10 | +0.5 | +0.0006 | 0.42 | no |
#
# **In short**: the §12 verdict does not depend on the network sizes nor on the regularization that the
# grid had not explored. More capacity makes the model memorize more and does not help; the only small gain comes from
# a stronger L2 penalty, too small to bring the GPFA within reach.

# %% [markdown]
# ### 14.3 · Segment length and the controller
#
# **The segments.** In §9.2 we chose 600 ms segments, and among the alternatives there were longer segments. Here
# we try two:
#
# - **400 ms**: the generator has to unroll by itself for less time, but each segment sees less context;
# - **1 s**: the duration of the windows of the common metric (§4.2). More context, but the generator has to unroll
#   by itself for a whole second, often straddling two phases.
#
# The overlap stays at 200 ms, with the same stitching (§9.2): with 400 ms segments almost every bin lies
# in two segments, with 1 s segments almost always in one only. The rest of LFADS stays as in §12, with two
# adaptations:
#
# - the KL weight stays 0.3 times that of the true ELBO, so it becomes 0.3 / (bins of the segment × 160): the reconstruction
#   error is a mean over bins and channels, the KL a sum per segment (§10.4);
# - 200 epochs, that is the same number of passes over the data: with 400 ms segments an epoch has twice the steps
#   (3200 in all), with 1 s segments half (800).
#
# The time-matched GPFA is fitted **on the same segments**, so the comparison stays like for like.
#
# **The controller.** Without a controller, everything that happens in a segment must already be written in the initial
# state: after that the generator runs by itself. The controller adds **inferred inputs**: at each bin a
# small network, the controller, reads what a second encoder has seen of the data around that bin and passes
# 4 numbers to the generator, a "push" that can follow events that the initial state could not have predicted (for
# example a cue). This is the full version of LFADS (Pandarinath et al., 2018). The settings are those of the
# lfads-torch configuration for MC_RTT, the continuous recording of the Neural Latents Benchmark, scaled to our
# sizes:
#
# | part | here | MC_RTT (lfads-torch) |
# |---|---|---|
# | input encoder | 64 units, bidirectional GRU | 80 |
# | controller | 64 units | 80 |
# | inferred inputs | 4 | 4 |
# | input delay | 1 bin | 1 |
# | prior distribution of the inputs | autoregressive: time constant 10 bins (200 ms), variance 0.1 | same |
# | KL weight of the inputs | same as the initial condition, 0.3 / (30 × 160) | chosen by AutoLFADS |
# | L2 penalty on the input encoder and controller | 0 | chosen by AutoLFADS, starting from 0 |
#
# The L2 penalty on the controller stays at 0 because the library divides the total penalty by the number of
# penalized weights: adding the controller would effectively halve the penalty on the generator, and the variant
# would change two things at once. 600 ms segments, the rest as in §12.
#
# The risk is known: more freedom means more room to memorize (§13.2, §14a). For this reason, besides the two
# measures, we check whether the model really **uses** the inputs: if the KL term pushes them to zero, the controller
# is there but switched off.
#
# **The rule** is that of §14a: the §12 verdict changes if a variant brings LFADS above the time-matched GPFA
# (on the same segments) with all 3 seeds, on one of the two measures.
#
# **The hand checks**:
#
# - the new cutting and stitching functions, with 600 ms segments, must give exactly the segments and the
#   stitching of §9.2; with 400 ms and 1 s, stitching the segments of the counts must return the counts;
# - the prior probability of the inputs, computed by hand, must coincide with that of the library.
#
# 90 new training runs (3 variants × 2 forms × 5 blocks × 3 seeds) and 10 GPFA fits: about 1 hour the first
# time. A 10-epoch trial run (4 at a time) gave, per training run, 132 s with 400 ms segments, 84 s with
# 1 s segments and 158 s with the controller, against 95 s for the main configuration.

# %%
OVL14 = SEG_OVL                                             # the overlap stays at 10 bins (200 ms)
LENS14 = {"pezzi da 400 ms": 20, "pezzi da 1 s": 50}        # bins per segment of the two variants

def starts_L(t, L):                                         # like segment_starts (§9.2), for segments of L bins overlapping by 10
    a0, a1 = TRIAL_SPAN[t]
    st = list(range(a0, a1 - L + 1, L - OVL14))             # one segment every L − 10 bins...
    if st[-1] != a1 - L:
        st.append(a1 - L)                                   # ...plus a last segment that ends exactly with the trial
    return np.array(st)

def cut_L(X, t, L):                                         # like cut_segments: the L-bin segments of trial t of a signal X
    return np.stack([X[a:a + L] for a in starts_L(t, L)])

def stitch_L(Y_seg, t, L):                                  # like stitch: weighted mean, with the AutoLFADS ramps on the 10 shared bins
    a0, a1 = TRIAL_SPAN[t]
    W = np.ones((len(Y_seg), L))                            # 1 where the segment stands alone
    W[1:, :OVL14] = RAMP_IN                                 # rising ramp at the start (except for the first segment of the trial)
    W[:-1, -OVL14:] = RAMP_OUT                              # falling ramp at the end (except for the last)
    num, den = np.zeros((a1 - a0, Y_seg.shape[2])), np.zeros((a1 - a0, 1))
    for a, y, w in zip(starts_L(t, L) - a0, Y_seg, W):
        num[a:a + L] += w[:, None] * y                      # sum of the weighted values...
        den[a:a + L] += w[:, None]                          # ...and of the weights
    return num / den

def stitch_all_L(per_trial, L):                             # like stitch_all: the 15 stitched trials, concatenated
    return np.concatenate([stitch_L(y_, t, L) for t, y_ in zip(TRIALS, per_trial)])

rng_chk = np.random.default_rng(0)
Y_chk = {t: rng_chk.normal(size=SEGS[t].shape[:2] + (3,)) for t in TRIALS}  # random values, different in each segment
print("600 ms segments: the same as §9.2:", all(np.array_equal(cut_L(COUNTS, t, SEG_LEN), SEGS[t]) for t in TRIALS),
      f"| stitching of random values, maximum difference from stitch: "
      f"{max(np.abs(stitch_L(Y_chk[t], t, SEG_LEN) - stitch(Y_chk[t], t)).max() for t in TRIALS):.1e}")
SEGS14 = {L: {t: cut_L(COUNTS, t, L) for t in TRIALS} for L in LENS14.values()}  # the count segments, for each of the two lengths
SEGS14_HID = {L: {t: s_[..., ORDER_HID] for t, s_ in seg.items()} for L, seg in SEGS14.items()}  # the 120 visible channels first, then the 40 hidden ones
for name, L in LENS14.items():
    back = np.concatenate([stitch_L(SEGS14[L][t].astype(float), t, L) for t in TRIALS])
    print(f"{name}: {sum(len(v) for v in SEGS14[L].values())} segments, from {min(len(v) for v in SEGS14[L].values())} a "
          f"{max(len(v) for v in SEGS14[L].values())} per trial ({sum(len(SEGS14[L][t]) for t in train_trials(0))} in the 12 training trials "
          f"of block 0); stitching of the counts, maximum difference {np.abs(back - COUNTS).max():.1e}")

# %%
CO14 = dict(ci_enc_dim=64, con_dim=64, co_dim=4, co_tau=10.0, co_nvar=0.1,  # input encoder, controller, inputs; autoregressive prior
            kl_co_scale=LF_HP["kl_ic_scale"], l2_ci_enc_scale=0.0, l2_con_scale=0.0)  # KL of the inputs as for the initial condition; no L2

def make_lfads_co(hp):
    """Like make_lfads (§10.3), but with the controller: a second encoder, the controller and 4 inferred inputs. Called in the separate process."""
    import numpy as np                                      # numpy BEFORE torch (§10.3)
    import torch
    from torch import nn
    from lfads_torch.model import LFADS                     # the library model
    from lfads_torch.modules import augmentations           # coordinated dropout (here 0, as in §12)
    from lfads_torch.modules.priors import AutoregressiveMultivariateNormal, MultivariateNormal  # priors of the inputs and of the initial condition
    from lfads_torch.modules.recons import Poisson          # the noise model: Poisson
    from lfads_torch.modules.readin_readout import FanInLinear  # the linear readout from the factors to the log-rates
    cd = augmentations.CoordinatedDropout(cd_rate=hp["cd_rate"], cd_pass_rate=0.0, ic_enc_seq_len=0)
    return LFADS(encod_data_dim=hp["n_in"], encod_seq_len=hp["seg_len"], recon_seq_len=hp["seg_len"],  # data shape
                 ext_input_dim=0, ic_enc_seq_len=0,         # no external input; the encoder reads the whole segment
                 ic_enc_dim=hp["enc_dim"], ci_enc_dim=hp["ci_enc_dim"], ci_lag=1, con_dim=hp["con_dim"], co_dim=hp["co_dim"],  # the controller is on
                 ic_dim=hp["ic_dim"], gen_dim=hp["gen_dim"], fac_dim=hp["fac_dim"], dropout_rate=hp["dropout"],
                 reconstruction=nn.ModuleList([Poisson()]), variational=True,
                 co_prior=AutoregressiveMultivariateNormal(tau=hp["co_tau"], nvar=hp["co_nvar"], shape=hp["co_dim"]),  # prior of the inputs
                 ic_prior=MultivariateNormal(mean=0.0, variance=hp["ic_prior_var"], shape=hp["ic_dim"]),
                 ic_post_var_min=hp["ic_post_var_min"], cell_clip=hp["cell_clip"],
                 train_aug_stack=augmentations.AugmentationStack([cd], batch_order=[0], loss_order=[0]),
                 infer_aug_stack=augmentations.AugmentationStack(),
                 readin=nn.ModuleList([nn.Identity()]), readout=nn.ModuleList([FanInLinear(hp["fac_dim"], hp["n_out"])]),
                 loss_scale=hp["loss_scale"], recon_reduce_mean=True,
                 lr_scheduler=False, lr_init=hp["lr"], lr_stop=1e-5, lr_decay=0.95, lr_patience=6,
                 lr_adam_beta1=0.9, lr_adam_beta2=0.999, lr_adam_epsilon=1e-8, weight_decay=0.0,
                 l2_start_epoch=0, l2_increase_epoch=hp["ramp_epochs"], l2_ic_enc_scale=hp["l2_ic_enc_scale"],
                 l2_ci_enc_scale=hp["l2_ci_enc_scale"], l2_gen_scale=hp["l2_gen_scale"], l2_con_scale=hp["l2_con_scale"],
                 kl_start_epoch=0, kl_increase_epoch=hp["ramp_epochs"], kl_ic_scale=hp["kl_ic_scale"], kl_co_scale=hp["kl_co_scale"])

def run_lfads_cached_co(jobs):                              # run_lfads_cached (§10.5) with the controller model
    global make_lfads                                       # during the call the name make_lfads refers to make_lfads_co: so lfads_fit,
    orig = make_lfads                                       # in the separate process, builds the model with the controller, and the fingerprint
    make_lfads = make_lfads_co                              # of each model (lf_key) contains the code of make_lfads_co
    try:
        return run_lfads_cached(jobs)
    finally:
        make_lfads = orig                                   # in any case the original function is restored

def co_prior_check(tau, nvar, seed):                        # log-probability of some sequences under the library's autoregressive prior (separate process)
    import numpy as np                                      # numpy BEFORE torch
    import torch
    from lfads_torch.modules.priors import AutoregressiveMultivariateNormal
    torch.manual_seed(seed)
    prior = AutoregressiveMultivariateNormal(tau=tau, nvar=nvar, shape=4)  # 4 inputs, as in CO14
    x = torch.randn(3, 30, 4)                               # 3 segments of 30 bins, 4 inputs: random values
    with torch.no_grad():
        return x.numpy().astype(np.float64), prior.log_prob(x).numpy().astype(np.float64)

x_co, lp_lib = POOL_LF.submit(co_prior_check, CO14["co_tau"], CO14["co_nvar"], 0).result()
alpha = np.exp(-1 / CO14["co_tau"])                         # how much of each input remains at the next bin: e^(−1/τ)
var0 = CO14["co_nvar"] / (1 - alpha ** 2)                   # variance at the first bin: the stationary one of the process
mu_co = np.concatenate([np.zeros_like(x_co[:, :1]), alpha * x_co[:, :-1]], axis=1)  # mean of each bin: α × the previous bin (0 at the first)
var_co = np.concatenate([np.full_like(x_co[:, :1], var0), np.full_like(x_co[:, 1:], CO14["co_nvar"])], axis=1)  # variance of each bin
lp_hand = (-0.5 * np.log(2 * np.pi * var_co) - (x_co - mu_co) ** 2 / (2 * var_co)).sum(axis=(1, 2))  # sum over bins and inputs, segment by segment
print(f"Autoregressive prior of the inputs: log-probability by hand {np.round(lp_hand, 3)}, library {np.round(lp_lib, 3)}; "
      f"maximum difference {np.abs(lp_hand - lp_lib).max():.1e}. Stationary deviation of the prior: {np.sqrt(var0):.2f}")

# %%
def job14b(var, form, k, sd):                               # one training run of §14b: variant, form, block, seed; the rest as in §12
    hp_, data_ = FORMS14[form]                              # configuration and data of the two forms (§14a)
    if var == "controller":
        return lf12_job(sd, train_trials(k), hp=dict(hp_, **CO14), **data_)  # 600 ms segments, with the controller
    L = LENS14[var]
    hp_L = dict(hp_, seg_len=L, kl_ic_scale=hp_["kl_ic_scale"] * SEG_LEN / L)  # KL weight: 0.3 / (L × 160)
    if form == "tutti i canali":
        return lf12_job(sd, train_trials(k), hp=hp_L, segs=SEGS14[L])
    return lf12_job(sd, train_trials(k), hp=hp_L, segs=SEGS14_HID[L], rate_cols=HID_COLS)

RUNS14B = [(v, f_, k, sd) for v in LENS14 for f_ in FORMS14 for k in range(5) for sd in SEEDS14]           # 400 ms and 1 s segments: 60
RUNS14B_CO = [("controller", f_, k, sd) for f_ in FORMS14 for k in range(5) for sd in SEEDS14]           # controller: 30
res14b = dict(zip(RUNS14B, run_lfads_cached([job14b(*r) for r in RUNS14B])))
res14b |= dict(zip(RUNS14B_CO, run_lfads_cached_co([job14b(*r) for r in RUNS14B_CO])))
GP14B = {L: dict(zip(range(5), gpfa_cached([np.concatenate([SEGS14[L][t] for t in train_trials(k)]) for k in range(5)])))
         for L in LENS14.values()}                          # the time-matched GPFA on the same 400 ms and 1 s segments

def co_dim_of(w):                                           # how many inferred inputs a model has, read from the weights (0 = no controller)
    return next((v.shape[0] // 2 for k_, v in w.items() if k_.endswith("co_linear.weight")), 0)

display(pd.DataFrame({v: {"bins per segment (from the outputs)": mm([res14b[(v, f_, k, sd)]["factors"][0].shape[1] for f_ in FORMS14 for k in range(5) for sd in SEEDS14]),
                          "inferred inputs (from the weights)": mm([co_dim_of(res14b[(v, f_, k, sd)]["weights"]) for f_ in FORMS14 for k in range(5) for sd in SEEDS14]),
                          "steps": mm([res14b[(v, f_, k, sd)]["steps"] for f_ in FORMS14 for k in range(5) for sd in SEEDS14]),
                          "kept version (epoch)": mm([res14b[(v, f_, k, sd)]["best_epoch"] for f_ in FORMS14 for k in range(5) for sd in SEEDS14])}
                      for v in list(LENS14) + ["controller"]}).T)

# %% [markdown]
# **Reading.**
#
# - **The hand checks pass.** With 600 ms segments the new functions give exactly the segments and the stitching of
#   §9.2 (difference 0); with 400 ms and 1 s, stitching the segments of the counts returns the counts (difference 9·10⁻¹⁶,
#   rounding). The prior probability of the inputs computed by hand coincides with that of the library
#   (difference 7.5·10⁻⁵, the precision of 32-bit numbers).
# - **The segments**: 1215 of 400 ms (from 66 to 96 per trial) and 310 of 1 s (from 17 to 24), against the 610 of 600 ms.
# - **The variants are the intended ones**: the outputs have 20, 50 and 30 bins per segment, and only the controller models have
#   the 4 inferred inputs in their weights. Steps: 3200 with 400 ms segments, 800 with 1 s segments, 1600 with the controller; the
#   kept version is in practice the last one (between epoch 192 and 200).
# - The 90 training runs and the 10 GPFA fits took about 1 hour 20 minutes, more than the estimate: the 400
#   ms segments cost more than expected.
#
# ### 14.4 · The two measures, and whether the controller is on
#
# The same measures as §14.2, with the main configuration of §14a (seeds 0-2) and the time-matched GPFA on the same segments.
# For the controller, in addition: the **inferred inputs** of the all-channel models on the segments of the test
# trials, that is their amplitude (the root of the mean of the squared posterior means) and their uncertainty
# (the posterior deviation), to be compared with the stationary deviation of the prior. Inputs with amplitude
# close to 0 and uncertainty equal to that of the prior would mean a switched-off controller.

# %%
GEOM14B = {"principale (§12)": SEG_LEN, "pezzi da 400 ms": 20, "pezzi da 1 s": 50, "controller": SEG_LEN}  # bins per segment of each variant
RES14B = {**{r: res14a[r] for r in RUNS14A if r[0] == "principale (§12)"}, **res14b}  # the main configuration comes from §14a

def fac14b(v, k, sd):                                       # ordered factors of variant v (all-channels form), whole session
    r = RES14B[(v, "tutti i canali", k, sd)]
    return order_by_readout(stitch_all_L(r["factors"], GEOM14B[v]), r["weights"]["readout.0.weight"])

M14B = {}                                                   # name -> measures, one list per seed (as M14)
for v, L in GEOM14B.items():
    ph = [acc14(lambda k, sd=sd: fac14b(v, k, sd)) for sd in SEEDS14]
    full = [[stitch_all_L(RES14B[(v, "120 + 40", k, sd)]["rates"], L) for sd in SEEDS14] for k in range(5)]  # [block][seed] -> hidden channels, whole session
    sn = seen_new(full, Y_HID)
    M14B[v] = {"ceiling": [a for a, _ in ph], "primi 3": [b for _, b in ph],
               "hidden": [bits_per_spike(test_only([full[k][i] for k in range(5)]), Y_HID) for i in range(len(SEEDS14))],
               "nuovi / visti": list(sn[:, :, 1].mean(1) / sn[:, :, 0].mean(1))}
M14B["GPFA alla pari, pezzi da 600 ms"] = M14[f"GPFA alla pari, {K_PCA} fattori"]  # from §14a
for name, L in LENS14.items():
    a_, b_ = acc14(lambda k, L=L: stitch_all_L([gpfa_infer_segments(SEGS14[L][t], GP14B[L][k]) for t in TRIALS], L))
    full = [[np.concatenate([stitch_L(gpfa_hidden(SEGS14[L][t], GP14B[L][k])[0], t, L) for t in TRIALS])] for k in range(5)]
    sn = seen_new(full, Y_HID)
    M14B[f"GPFA alla pari, {name}"] = {"ceiling": [a_], "primi 3": [b_],
                                       "hidden": [bits_per_spike(test_only([full[k][0] for k in range(5)]), Y_HID)],
                                       "nuovi / visti": [sn[0, :, 1].mean() / sn[0, :, 0].mean()]}

GP_SAME14B = {"principale (§12)": "GPFA alla pari, pezzi da 600 ms", "pezzi da 400 ms": "GPFA alla pari, pezzi da 400 ms",
              "pezzi da 1 s": "GPFA alla pari, pezzi da 1 s", "controller": "GPFA alla pari, pezzi da 600 ms"}  # the GPFA on the same segments
print(f"Check: the main configuration gives the same numbers as §14a (maximum difference "
      f"{max(np.abs(np.subtract(M14B['principale (§12)'][m], M14['principale (§12)'][m])).max() for m in ('ceiling', 'hidden')):.1e})")

display(pd.DataFrame({name: {"phase, ceiling (%)": fmt(m["ceiling"]), "phase, first 3 (%)": fmt(m["primi 3"]),
                             "nascosti, trial di test (bit per spike)": fmt4(m["hidden"]), "nuovi / visti": fmt2(m["nuovi / visti"])}
                      for name, m in M14B.items()}).T.rename_axis("media (minimo-massimo) tra i 3 semi; a caso 12.5%"))

P14B = M14B["principale (§12)"]
display(pd.DataFrame({v: {"ceiling: difference from the main configuration (points)": fmt1s(np.subtract(M14B[v]["ceiling"], P14B["ceiling"])),
                          "hidden: difference from the main configuration": fmt4(np.subtract(M14B[v]["hidden"], P14B["hidden"])),
                          "time-matched GPFA": GP_SAME14B[v].replace("GPFA alla pari, ", ""),
                          "semi sopra la GPFA: tetto": f"{int(np.sum(np.array(M14B[v]['ceiling']) > M14B[GP_SAME14B[v]]['ceiling'][0]))} su 3",
                          "semi sopra la GPFA: nascosti": f"{int(np.sum(np.array(M14B[v]['hidden']) > M14B[GP_SAME14B[v]]['hidden'][0]))} su 3"}
                      for v in GEOM14B}).T.rename_axis("seed by seed; [min, max] across the 3 seeds"))

# %%
def co_inputs(hp, model_path, X):                           # the inferred inputs of a model with the controller saved on disk, on the segments X (separate process)
    import numpy as np                                      # numpy BEFORE torch
    import torch
    from lfads_torch.tuples import SessionBatch
    torch.set_num_threads(1)
    model = make_lfads_co(hp)
    model.load_state_dict(torch.load(model_path))           # the weights of the kept version
    model.eval()                                            # no dropout
    Xt = torch.tensor(X, dtype=torch.float32)
    sb = SessionBatch(encod_data=Xt[..., :hp["n_in"]], recon_data=Xt, ext_input=torch.zeros(len(Xt), Xt.shape[1], 0),
                      truth=torch.zeros(0), sv_mask=torch.ones_like(Xt))
    with torch.no_grad():
        out = model({0: sb}, sample_posteriors=False, output_means=False)[0]  # with the posterior mean, as for the factors
    return out.co_means.numpy(), out.co_stds.numpy()

CO_IN = {(k, sd): POOL_LF.submit(co_inputs, job14b("controller", "tutti i canali", k, sd)["hp"],
                                 CACHE_LF / f"{res14b[('controller', 'tutti i canali', k, sd)]['key']}.pt",
                                 np.concatenate([SEGS[t] for t in TEST_TRIALS[k]])) for k in range(5) for sd in SEEDS14}
CO_IN = {key: f_.result() for key, f_ in CO_IN.items()}    # for each block and seed: (means, deviations) of the inputs, segments × 30 bins × 4
amp = np.array([np.sqrt(np.mean(m_ ** 2)) for m_, _ in CO_IN.values()])  # amplitude of the inferred inputs
unc = np.array([np.mean(s_) for _, s_ in CO_IN.values()])  # mean posterior uncertainty
amp_dim = np.array([np.sqrt(np.mean(m_ ** 2, axis=(0, 1))) for m_, _ in CO_IN.values()])  # amplitude of each of the 4 inputs
print(f"Inferred inputs on the test trials (15 models): amplitude {fmt2(amp)}, posterior uncertainty {fmt2(unc)}; "
      f"stationary deviation of the prior {np.sqrt(var0):.2f}")
print(f"Amplitude of each of the 4 inputs, sorted from the largest (mean of the 15 models): {np.round(np.sort(amp_dim, axis=1)[:, ::-1].mean(0), 2)}")

# %%
ROWS14B = list(GEOM14B)[::-1]                               # from top to bottom as in the table
fig, ax = plt.subplots(1, 2, figsize=(15, 3.8), sharey=True)
for a_, meas, xlab in ((ax[0], "ceiling", "phase decoded (%), all factors"), (ax[1], "hidden", "bits per spike, 40 hidden channels, test trials")):
    for y, v in enumerate(ROWS14B):
        vals = np.array(M14B[v][meas])
        a_.scatter(vals, np.full(len(vals), y), color="C3", s=18, alpha=0.6, zorder=3)  # the 3 seeds
        a_.scatter(vals.mean(), y, color="C3", s=70, marker="D", zorder=4, label="LFADS: mean of the 3 seeds (small points: the seeds)" if y == 0 else None)
        a_.scatter(M14B[GP_SAME14B[v]][meas][0], y, color="C0", s=70, marker="|", linewidths=3, zorder=4,
                   label="time-matched GPFA on the same segments" if y == 0 else None)
    a_.axvline(np.mean(P14B[meas]), color="C3", ls=":", lw=1)  # the main configuration, for comparison
    a_.set(xlabel=xlab, yticks=range(len(ROWS14B)), yticklabels=ROWS14B)
    a_.grid(axis="y", alpha=0.3)
ax[0].legend(fontsize=8, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.18))  # below the plot, so as not to cover the points
ax[0].set_title("Phase (dashed line: the main configuration)"); ax[1].set_title("Hidden channels of the test trials")
fig.tight_layout(); fig.savefig(FIGS / "04_lfads_variants_b.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# **Reading.**
#
# - **The check passes**: the main configuration gives the same numbers as §14a.
# - **400 ms segments: in practice like the main configuration.** On phase −0.7 points (in all 3 seeds, from −1.1 to −0.4);
#   on the hidden channels +0.0036 (0.0423), on par with the GPFA on the same segments (0.0420): 2 seeds out of 3 above, so the
#   rule is not satisfied.
# - **1 s segments: worse on both measures**, −5.3 points on phase and −0.0091 on the hidden channels (0.0296): the
#   generator has to unroll by itself for a whole second, and cannot manage it. The GPFA, instead, is not affected by the length of the
#   segments (from 50.3 to 51.4% on phase, from 0.0420 to 0.0430 on the hidden channels).
# - **The controller changes things, in two opposite directions**:
#   - on **phase** it gets worse: −5.2 points (41.8%), and 17.0% with the first 3 factors;
#   - on the **hidden channels** it reaches **0.0631** bits per spike (from 0.0628 to 0.0636 across seeds): +0.0244 over the main configuration,
#     above the time-matched GPFA (0.0422) with all 3 seeds, and above the GPFA with Poisson regression
#     (0.0474) too, the highest score so far. **By the rule fixed before looking, on the hidden channels the §12 verdict
#     changes.**
# - **The controller is on**: the inferred inputs have an amplitude of 0.29 (0.27-0.31 across models) with
#   a posterior uncertainty of 0.15, against the prior deviation of 0.74. The model uses all 4 of them (amplitudes
#   from 0.26 to 0.31) and knows well what they are worth.
# - **It memorizes like the main configuration**: on new trials 40% of the seen-trial score remains (37-40% for all the
#   LFADS variants, 66-67% for the GPFA).
# - **In plain words**: without a controller the generator has to guess from the initial state everything that
#   happens in the 600 ms of the segment; with the controller it can be "pushed" bin by bin, following what the visible
#   channels do instant by instant. This helps predict the hidden channels, but makes the factors less useful for
#   decoding the phase.
# - **The figure**: for each variant the 3 LFADS seeds (small points) and their mean (diamond), and the time-matched GPFA
#   on the same segments (blue tick); the dashed line is the main configuration.
# - **What it does not say**: whether the gain on the hidden channels is real or an artifact, and where it comes from. We check it
#   below.
#
# #### The controller beats the GPFA on the hidden channels: three checks
#
# A gain this large must be checked before it is believed (the rule of the whole notebook: when a number is
# surprisingly good, look for the error first). The input encoder reads only the 120 visible channels, like the one
# for the initial state: the hidden channels never enter the model when it computes its predictions. Three checks:
#
# 1. **Hidden channels shifted in time** (§13.4): the same 15 training runs of the 120 + 40 form, on the data in which each
#    hidden channel is shifted by a minute or more. If the gain comes from the coordination between channels, on new trials the
#    score must fall below zero, as for the other models; if the chain (controller, stitching, scoring) created it
#    on its own, it would stay above zero.
# 2. **Where the gain comes from**: the score by rate band of the channels (§12.4), and how much of the variation
#    of the predicted rates lies between 16 and 20 Hz, the rhythm of the counts that LFADS without a controller does not reproduce
#    (§11.3, §12.3).
# 3. **Smoothing + GLM with less smoothing** (σ = 20 and 10 ms instead of 50): if the controller wins because it follows the
#    fast fluctuations common to visible and hidden channels, a reference that preserves them should also gain.
#
# 15 new training runs with the controller: about 12 minutes.

# %%
def job14b_sh(k, sd):                                       # the controller, 120 + 40 form, on the data with the hidden channels shifted in time (§13.4)
    return lf12_job(sd, train_trials(k), hp=dict(LF_HID_HP, **CO14), segs=SEGS_SH_HID, rate_cols=HID_COLS)

res14b_sh = dict(zip([(k, sd) for k in range(5) for sd in SEEDS14],
                     run_lfads_cached_co([job14b_sh(k, sd) for k in range(5) for sd in SEEDS14])))
full_sh_co = [[stitch_all(res14b_sh[(k, sd)]["rates"]) for sd in SEEDS14] for k in range(5)]  # [block][seed] -> predicted shifted hidden channels
sn_sh_co = seen_new(full_sh_co, Y_SH)
b_sh_lf = [bits_per_spike(test_only([FULL_SH["LFADS (3 semi)"][k][i] for k in range(5)]), Y_SH) for i in range(len(SEEDS14))]
print(f"(1) Shifted hidden channels, new trials: controller {fmt4([bits_per_spike(test_only([full_sh_co[k][i] for k in range(5)]), Y_SH) for i in range(len(SEEDS14))])}, "
      f"without controller (§13.4) {fmt4(b_sh_lf)}; seen trials: controller {fmt4(sn_sh_co[:, :, 0].mean(1))}, "
      f"without controller {fmt4(SN_SH['LFADS (3 semi)'][:, :, 0].mean(1))}")

P_CO = [test_only([stitch_all(res14b[("controller", "120 + 40", k, sd)]["rates"]) for k in range(5)]) for sd in SEEDS14]  # session made of test trials only
P_LF = [test_only([stitch_all(res14a[("principale (§12)", "120 + 40", k, sd)]["rates"]) for k in range(5)]) for sd in SEEDS14]

def share18_rates(P, L=50, step=25):                        # share of the variation between 16 and 20 Hz in the predicted rates (mean over channels), 1 s windows
    win = np.hanning(L)                                     # the same window as §11.3
    S = np.mean([np.abs(np.fft.rfft((P[a:a + L] - P[a:a + L].mean(0)) * win[:, None], axis=0)) ** 2 for a in range(0, len(P) - L, step)], axis=0)
    f_ = np.fft.rfftfreq(L, io.BIN_S)                       # from 0 to 25 Hz, every 1 Hz
    return float(np.mean(S[(f_ >= 16) & (f_ <= 20)].sum(0) / S[1:].sum(0)))

SMV = {sg: io.smooth_rates(s.spikes, sigma_ms=sg)[:, ACTIVE][:, HELD_IN] for sg in (20, 10)}  # the 120 visible channels smoothed with a smaller σ

def glm_sigma(sg, k):                                       # the smoothing + GLM of §11a with a different σ, fitted on the 12 training trials of block k
    fit_ = np.isin(trial, train_trials(k))
    F = StandardScaler().fit(SMV[sg][fit_]).transform(SMV[sg])
    return np.column_stack([PoissonRegressor(alpha=0.1, max_iter=500).fit(F[fit_], COUNTS[fit_, h]).predict(F) for h in HIDDEN])

P_GLM_S = {sg: test_only([glm_sigma(sg, k) for k in range(5)]) for sg in SMV}
CHK14B = {"LFADS with the controller (3 seeds)": P_CO, "LFADS, main configuration (3 seeds)": P_LF, "time-matched GPFA": [PRED_GP],
          "GPFA + Poisson regression": [PRED_GPGLM], "smoothing (σ = 50 ms) + GLM": [PRED_GLM],
          "smoothing (σ = 20 ms) + GLM": [P_GLM_S[20]], "smoothing (σ = 10 ms) + GLM": [P_GLM_S[10]]}
display(pd.DataFrame({name: {"all 40": fmt4([bits_per_spike(p_, Y_HID) for p_ in P]),
                             **{BAND_LAB[b]: fmt4([bits_per_spike(p_[:, BAND_OF == b], Y_HID[:, BAND_OF == b]) for p_ in P]) for b in range(4)},
                             "variation between 16 and 20 Hz (%)": f"{np.mean([share18_rates(p_) for p_ in P]) * 100:.1f}"}
                      for name, P in CHK14B.items()}).T.rename_axis("(2, 3) bits per spike of the hidden channels, test trials, by band; share of the variation of the predictions at 16-20 Hz"))
print(f"For comparison, in the true counts of the 40 hidden channels the share between 16 and 20 Hz is {share18_rates(Y_HID) * 100:.1f}% "
      f"(it includes Poisson noise, which is the same at all frequencies)")

# %% [markdown]
# **Reading.**
#
# - **(1) The gain comes from real coordination between channels.** With the hidden channels shifted in time the controller
#   falls below zero on new trials (−0.0181), exactly like LFADS without a controller (−0.0179, §13.4); on seen
#   trials it reaches 0.0425, the same memorization as before (0.0398). The controller → stitching → scoring chain does not
#   create positive scores on its own.
# - **(2) The gain lies in the channels above 3.5 Hz and goes through the 18 Hz rhythm.** The controller is the best in three
#   bands out of four (0.1011, 0.0948 and 0.0443 bits per spike in bands 2, 3 and 4), but falls below zero in the slowest
#   (−0.0147), where the GPFA with regression and smoothing + GLM win (about 0.015). Its predictions have
#   32.0% of the variation between 16 and 20 Hz, against 0.1% for LFADS without a controller and for the GPFA: they follow the
#   18 Hz rhythm of the counts (§11.3) that the other models, all smoother, cancel.
# - **(3) Following fast fluctuations is not enough.** Smoothing + GLM with less smoothing does not gain (0.0386 with
#   σ = 20 ms, 0.0413 with σ = 10 ms, against 0.0416 with σ = 50 ms), even though with σ = 10 ms its predictions too
#   have 18.3% of the variation between 16 and 20 Hz. The controller manages to use the rhythm because it estimates it from all
#   channels together, in the few factors of the model; a channel-by-channel regression on 120 noisy inputs does not.
# - **What it does not say**: why the controller worsens the phase (an unverified hypothesis: the inputs absorb
#   variations that the factors without a controller used to distinguish the phases); whether with more data (§14d) the picture
#   for the controller changes; and the shift of the hidden channels is a single one (seed 0).
#
# ### Summary of §14b
#
# | variant | phase, ceiling: difference from the main configuration | hidden: difference | above the GPFA on the same segments? |
# |---|---|---|---|
# | main (§12): 600 ms segments, no controller | 47.0% | 0.0387 | no |
# | 400 ms segments | −0.7 | +0.0036 | on hidden channels 2 seeds out of 3 (on par) |
# | 1 s segments | −5.3 | −0.0091 | no |
# | controller | −5.2 | **+0.0244** | **on hidden channels yes, 3 seeds out of 3** (0.0631 against 0.0422; above the GPFA with regression too, 0.0474); on phase no |
#
# **In short**: the segment length does not change the verdict (600 or 400 ms are fine, 1 s is worse). The controller
# does, but only on the generative measure: with the inferred inputs LFADS becomes the best model of the hidden channels,
# because it follows the 18 Hz rhythm shared by the population. As a description of the task phases, however, it gets worse.

# %% [markdown]
# ### 14.5 · LFADS without dynamics: a VAE that looks at one bin at a time
#
# **The question.** LFADS adds two things at once to the GPFA: a **nonlinear** map from the factors to the rates
# (with Poisson noise) and a **dynamics** (the generator). To understand which of the two matters, we need the cell
# that is missing from this table:
#
# | | without time: one bin at a time | with time |
# |---|---|---|
# | **linear** | FA | GPFA (factors that change gradually) |
# | **nonlinear** | **per-bin VAE** (here) | LFADS (a dynamics) |
#
# **How.** The same LFADS as §12 with segments **one bin** long: the encoder reads a single bin and derives the initial
# state from it, the generator takes a single step, and from there come the factors and the rates of that bin. Each bin is treated
# on its own, as in FA, but the map from the factors to the rates goes through a network and the noise is Poisson: this is the
# "nonlinear FA". The rest as in §12: KL weight 0.3 / (1 × 160), 200 epochs, batches of 64 bins (154 steps per
# epoch), the two forms, 3 seeds, 5 blocks: 30 training runs, about 35 minutes the first time (a 4-epoch trial run
# gave about 4-5 minutes per training run).
#
# **The check**: without time, the order of the bins does not matter. We also give the already trained model a test trial
# with its bins shuffled: after putting the results back in order, the factors of each bin must be the same.
#
# **What we expect**: if nonlinearity helps, the per-bin VAE beats FA; if the LFADS dynamics helps,
# LFADS beats the per-bin VAE. The more useful comparison for the question of the thesis is the second.

# %%
SEGS1 = {t: COUNTS[TRIAL_SPAN[t][0]:TRIAL_SPAN[t][1], None, :] for t in TRIALS}  # one-bin "segments": (bins of the trial, 1, 160)
SEGS1_HID = {t: s_[..., ORDER_HID] for t, s_ in SEGS1.items()}  # the 120 visible channels first, then the 40 hidden ones
T_SHUF = int(TEST_TRIALS[0][0])                             # for the check: a test trial of block 0...
PERM = np.random.default_rng(0).permutation(len(SEGS1[T_SHUF]))  # ...with its bins in random order

def job14c(form, k, sd):                                    # one training run of the per-bin VAE: form, block, seed; the rest as in §12
    hp_, _ = FORMS14[form]
    hp1 = dict(hp_, seg_len=1, kl_ic_scale=hp_["kl_ic_scale"] * SEG_LEN)  # one-bin segments; KL weight 0.3 / (1 × 160)
    if form == "tutti i canali":
        j_ = lf12_job(sd, train_trials(k), hp=hp1, segs=SEGS1)
    else:
        j_ = lf12_job(sd, train_trials(k), hp=hp1, segs=SEGS1_HID, rate_cols=HID_COLS)
    j_["apply"] = j_["apply"] + [j_["apply"][list(TRIALS).index(T_SHUF)][PERM]]  # in addition: the check trial, shuffled
    return j_

RUNS14C = [(f_, k, sd) for f_ in FORMS14 for k in range(5) for sd in SEEDS14]
res14c = dict(zip(RUNS14C, run_lfads_cached([job14c(*r) for r in RUNS14C])))
i_sh = list(TRIALS).index(T_SHUF)
d_sh = max(np.abs(r["factors"][len(TRIALS)][np.argsort(PERM)] - r["factors"][i_sh]).max() for r in res14c.values())  # put back in order
print(f"Check: with the bins of trial {T_SHUF} shuffled, maximum difference of the factors (put back in order) {d_sh:.1e} "
      f"(the factors reach {max(np.abs(r['factors'][i_sh]).max() for r in res14c.values()):.1f})")
display(pd.DataFrame({f_: {"steps": mm([res14c[(f_, k, sd)]["steps"] for k in range(5) for sd in SEEDS14]),
                           "kept version (epoch)": mm([res14c[(f_, k, sd)]["best_epoch"] for k in range(5) for sd in SEEDS14])}
                      for f_ in FORMS14}).T)

# %%
def fac14c(k, sd):                                          # ordered factors of the per-bin VAE (all-channels form), whole session
    r = res14c[("tutti i canali", k, sd)]
    return order_by_readout(np.concatenate([f_[:, 0, :] for f_ in r["factors"][:len(TRIALS)]]), r["weights"]["readout.0.weight"])

ph14c = [acc14(lambda k, sd=sd: fac14c(k, sd)) for sd in SEEDS14]
full14c = [[np.concatenate([x_[:, 0, :] for x_ in res14c[("120 + 40", k, sd)]["rates"][:len(TRIALS)]]) for sd in SEEDS14] for k in range(5)]
sn14c = seen_new(full14c, Y_HID)
M14C = {"FA": {"ceiling": ACC12[("FA", "tetto: tutti i 25 fattori")], "primi 3": ACC12[("FA", "primi 3 fattori")],  # from §12 and §13.2
               "hidden": BPS12[("FA", "tutta la sessione")], "nuovi / visti": list(SN13["FA"][:, :, 1].mean(1) / SN13["FA"][:, :, 0].mean(1))},
        "VAE per bin (3 semi)": {"ceiling": [a for a, _ in ph14c], "primi 3": [b for _, b in ph14c],
                                 "hidden": [bits_per_spike(test_only([full14c[k][i] for k in range(5)]), Y_HID) for i in range(len(SEEDS14))],
                                 "nuovi / visti": list(sn14c[:, :, 1].mean(1) / sn14c[:, :, 0].mean(1))},
        "GPFA alla pari": M14[f"GPFA alla pari, {K_PCA} fattori"],
        "LFADS (principale, semi 0-2)": M14["principale (§12)"],
        "LFADS con il controller (§14b)": M14B["controller"]}
display(pd.DataFrame({name: {"phase, ceiling (%)": fmt(m["ceiling"]), "phase, first 3 (%)": fmt(m["primi 3"]),
                             "nascosti, trial di test (bit per spike)": fmt4(m["hidden"]), "nuovi / visti": fmt2(m["nuovi / visti"])}
                      for name, m in M14C.items()}).T.rename_axis("media (minimo-massimo) tra i semi; a caso 12.5%"))
for meas, f_ in (("ceiling", lambda v: f"{np.mean(v):.1f}%"), ("hidden", lambda v: f"{np.mean(v):.4f}")):
    display(pd.DataFrame({"without time": {"linear": "FA " + f_(M14C["FA"][meas]), "non linear": "per-bin VAE " + f_(M14C["VAE per bin (3 semi)"][meas])},
                          "with time": {"linear": "GPFA " + f_(M14C["GPFA alla pari"][meas]),
                                           "non linear": "LFADS " + f_(M14C["LFADS (principale, semi 0-2)"][meas])}})
            .rename_axis("phase, ceiling" if meas == "ceiling" else "hidden, bits per spike"))

# %% [markdown]
# **Reading.**
#
# - **The check passes**: with the bins of trial 3 shuffled, the factors of each bin are identical (difference 0). The
#   per-bin VAE does not use time, as intended.
# - **The training runs**: about 30 400-31 000 steps (154 per epoch); the kept version falls between epoch 138 and 200,
#   not always at the last one, because here the error on the training bins also oscillates a little at the end of training.
# - **The 2 × 2 table says that time matters, not nonlinearity.**
#   - **Phase**: without time FA 25.8% and per-bin VAE 17.1%; with time GPFA 51.4% and LFADS 47.0%. Adding
#     time is worth about 25-30 points; adding nonlinearity is worth nothing, in fact it subtracts (−8.7 without time,
#     −4.4 with time).
#   - **Hidden channels**: FA 0.0358 and per-bin VAE 0.0348, on par; GPFA 0.0422 and LFADS 0.0387. Here too the
#     step forward comes from time, and the nonlinear network does not improve on it.
# - **The per-bin VAE memorizes less than LFADS** (on new trials 54% of the seen-trial score remains, against 40%)
#   but more than FA (68%).
# - **The controller (§14b) is the only exception**, and only on the hidden channels (0.0631).
# - **What it does not say**: the VAE network is that of LFADS with a single step (an encoder and a 64-unit generator),
#   not a network chosen specifically for one bin at a time; 3 seeds.
#
# ### 14.6 · Poisson regression on the LFADS factors
#
# In §12.4 the best variant on the hidden channels was the GPFA with a **Poisson regression** from its factors
# to the hidden channels (0.0474), in place of its output (linear on the square root of the counts). It is fair to give LFADS the
# same possibility: from the 25 factors of the 120 + 40 form, which the encoder derives from the visible channels only, a Poisson
# regression for each hidden channel, fitted on the 12 training trials with the same penalty (0.1). No new training
# runs: the models are those of §12.4, with their 5 seeds.
#
# In §14b the controller already brought LFADS above the GPFA on the hidden channels. Here the question is different: for LFADS **without**
# a controller, is a better output enough? **The rule, fixed before looking**: yes, if LFADS with the regression
# beats the GPFA with the regression with all 5 seeds.

# %%
def lfads_glm(k, sd):                                       # Poisson regression from the 25 LFADS factors (120 + 40 form) to the 40 hidden channels, in block k
    lat = stitch_all(res12b[(k, sd)]["factors"])            # the factors of the whole session, derived from the 120 visible channels only
    fit_ = np.isin(trial, train_trials(k))                  # the regression is fitted on the 12 training trials
    F = StandardScaler().fit(lat[fit_]).transform(lat)
    return np.column_stack([PoissonRegressor(alpha=0.1, max_iter=500).fit(F[fit_], COUNTS[fit_, h]).predict(F) for h in HIDDEN])

FULL14C = [[lfads_glm(k, sd) for sd in SEEDS_LF] for k in range(5)]  # [block][seed] -> prediction of the hidden channels over the whole session
lat_chk = stitch_all(res12b[(0, 0)]["factors"]); fit_chk = np.isin(trial, train_trials(0))
sc_chk = StandardScaler().fit(lat_chk[fit_chk]); reg_chk = PoissonRegressor(alpha=0.1, max_iter=500).fit(sc_chk.transform(lat_chk[fit_chk]), COUNTS[fit_chk, HIDDEN[0]])
hand = np.exp(sc_chk.transform(lat_chk) @ reg_chk.coef_ + reg_chk.intercept_)  # the hand prediction: e^(standardized factors · β + b)
print(f"Hand check (block 0, seed 0, first hidden channel): maximum difference from scikit-learn {np.abs(hand - FULL14C[0][0][:, 0]).max():.1e}")

b_lfglm = [bits_per_spike(test_only([FULL14C[k][i] for k in range(5)]), Y_HID) for i in range(len(SEEDS_LF))]
sn_lfglm = seen_new(FULL14C, Y_HID)
b_gpglm = BPS12[("variante: GPFA + regressione di Poisson", "tutta la sessione")][0]
def ratio(sn_):                                             # new / seen, seed by seed, from a (seeds, 5 blocks, 2) array from seen_new
    return list(sn_[:, :, 1].mean(1) / sn_[:, :, 0].mean(1))

rows14c = {"smoothing + GLM (riferimento, §12.4)": (BPS12[("smoothing + GLM (reference)", "tutta la sessione")], ratio(SN13["smoothing + GLM (reference)"])),
           "LFADS, la sua uscita (§12.4, 5 semi)": (BPS12[("LFADS (5 semi)", "tutta la sessione")], ratio(SN13["LFADS (5 semi)"])),
           "LFADS + regressione di Poisson (5 semi)": (b_lfglm, ratio(sn_lfglm)),
           "LFADS con il controller, la sua uscita (§14b, 3 semi)": (M14B["controller"]["hidden"], M14B["controller"]["nuovi / visti"]),
           "GPFA alla pari, la sua uscita (§12.4)": (BPS12[("GPFA alla pari", "tutta la sessione")], ratio(SN13["GPFA alla pari"])),
           "GPFA + regressione di Poisson (§12.4)": ([b_gpglm], ratio(SN13["GPFA + regressione di Poisson"]))}
display(pd.DataFrame({name: {"nascosti, trial di test (bit per spike)": fmt4(b_), "nuovi / visti": fmt2(r_)}
                      for name, (b_, r_) in rows14c.items()}).T.rename_axis("40 hidden channels; for LFADS mean [min, max] across seeds"))
above14c = np.array([[bits_per_spike(FULL14C[k][i][np.isin(trial, TEST_TRIALS[k])], Y_HID[np.isin(trial, TEST_TRIALS[k])]) >
                      BPS12[("variante: GPFA + regressione di Poisson", f"blocco {k}")][0] for k in range(5)] for i in range(len(SEEDS_LF))])
print(f"LFADS + regression above GPFA + regression: {int((np.array(b_lfglm) > b_gpglm).sum())} seeds out of {len(SEEDS_LF)} (whole session), "
      f"{int(above14c.sum())} cases out of {above14c.size} (block × seed); mean gap {b_gpglm - np.mean(b_lfglm):.4f} bits per spike")

# %%
fig, ax = plt.subplots(1, 2, figsize=(15, 4.6), gridspec_kw={"width_ratios": (1, 1.4)})
names_ph = ["FA", "VAE per bin (3 semi)", "GPFA alla pari", "LFADS (principale, semi 0-2)"]
cols_ph = ["C1", "C5", "C0", "C3"]
for i, (name, c_) in enumerate(zip(names_ph, cols_ph)):    # left: the 2 × 2 table, on phase
    v = np.atleast_1d(M14C[name]["ceiling"])
    ax[0].bar(i, v.mean(), color=c_, width=0.6)
    if len(v) > 1:
        ax[0].plot([i, i], [v.min(), v.max()], color="k", lw=1)  # min-max across seeds
ax[0].set(xticks=range(4), xticklabels=["FA\nlinear,\nwithout time", "Per-bin VAE\nnonlinear,\nwithout time", "GPFA\nlinear,\nwith time",
                                        "LFADS\nnonlinear,\nwith time"], ylabel="phase decoded (%), all factors",
          title="Phase decoding (chance 12.5%)")
for i, (name, (b_, _)) in enumerate(list(rows14c.items()) + [("VAE per bin (3 semi)", (M14C["VAE per bin (3 semi)"]["hidden"], None)),
                                                             ("FA", (M14C["FA"]["hidden"], None))]):  # right: the hidden channels
    v = np.atleast_1d(b_)
    c_ = "C2" if "regression" in name else ("0.45" if "smoothing" in name else ("C4" if "controller" in name else ("C3" if "LFADS" in name else ("C0" if "GPFA" in name else ("C5" if "VAE" in name else "C1")))))
    ax[1].barh(i, v.mean(), color=c_, height=0.6)
    if len(v) > 1:
        ax[1].plot([v.min(), v.max()], [i, i], color="k", lw=1)
ax[1].set(yticks=range(len(rows14c) + 2), yticklabels=[n_.replace(", la sua uscita", "").replace(" (", "\n(") for n_ in list(rows14c) + ["VAE per bin (3 semi)", "FA (§12.4)"]],
          xlabel="bits per spike, 40 hidden channels, test trials", title="Hidden channels (green: with Poisson regression)")
ax[1].invert_yaxis()
fig.tight_layout(); fig.savefig(FIGS / "04_lfads_nodynamics.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# **Reading.**
#
# - **The hand check passes**: the regression prediction coincides with that of scikit-learn (difference 0).
# - **A better output helps LFADS without a controller only a little**: with Poisson regression on its factors it rises from
#   0.0388 to 0.0404, but stays below the GPFA with the same regression (0.0474) in all 5 seeds and in all 25 cases
#   (block × seed), by 0.0070 bits per spike on average. By the rule fixed beforehand, the verdict does not change: LFADS
#   without a controller is not missing the output, it is missing the factors.
# - **The regression makes the model memorize a little less** (on new trials 44% of the seen-trial score remains, against 40%).
# - **The figure**: on the left the 2 × 2 table on phase; on the right all models on the hidden channels (in green those with
#   Poisson regression).
#
# ### Summary of §14c
#
# | question | answer | where |
# |---|---|---|
# | does nonlinearity without time (per-bin VAE) beat FA? | **no**: phase 17.1% against 25.8%, hidden channels 0.0348 against 0.0358 | §14.5 |
# | what does the LFADS dynamics add? | time: from 17.1% to 47.0% on phase, from 0.0348 to 0.0387 on hidden channels; but the GPFA, which is linear, does more | §14.5 |
# | does a Poisson regression on the factors bring LFADS (without controller) to the GPFA? | **no**: 0.0404 against 0.0474, in none of the 25 cases | §14.6 |
#
# **In short**: in these data the gain comes from modeling time; the nonlinear map, alone or together with the
# dynamics, adds nothing. The only exception remains the controller, on the hidden channels.

# %% [markdown]
# ### 14.7 · More data: the other two sessions of the same day
#
# **The question.** In §13b the gap between LFADS and the GPFA on the hidden channels shrank with more training
# trials (from 0.0114 bits per spike with 4 trials to 0.0035 with 12), without closing; beyond 12 trials, with a single
# session, we could not go. Here we add the other two sessions recorded on the same day (19250108T111022 and
# 19250108T111455: 8 trials each, about 2 minutes each). In each block the models learn from our 12 training
# trials **plus 16 new trials**: 28 in all, more than double the data.
#
# **How.** The three sessions come from the same electrode array, with the same 176 channels: we use our 160
# active channels in the other two as well (of these, 159 and 160 are active there). The new sessions are cut into 600 ms segments
# within each of their trials, like ours (§9.2), and the segments are added to the training ones. Everything else
# stays the same: the 3 test trials of each block (always from our session), the common metric (the classifier
# learns from the points of our 12 training trials), 200 epochs (which with more data mean more steps, about
# 3800 instead of 1600), 3 seeds. The time-matched GPFA, FA and smoothing + GLM also learn from the new sessions.
# In the two "with Poisson regression" variants (GPFA and LFADS) the factor models use the extra data, whereas
# the regression from the factors to the hidden channels stays fitted on our 12 trials, as in §12.4.
#
# **Discarded alternatives**:
#
# - a **multi-session** LFADS (a different input layer for each session, as in AutoLFADS) is needed when the
#   channels change from one session to another; here they are the same, and we check this below;
# - **all 13 calibration sessions**: more data, but from different days, with the changes in the signals from one
#   day to the next.
#
# **LFADS with the controller.** After §14b we also try LFADS with the controller on more data (an addition decided after
# seeing §14b, and we state it): the question for it is whether more data recover the phase, where it was below.
#
# **The rule** is that of §14a: the §12 verdict changes if, with the same data, LFADS beats the time-matched GPFA
# with all 3 seeds, on one of the two measures.
#
# **The checks**: the channels are really the same (the mean rate of each channel in the new sessions against that of
# ours); each new segment lies within a trial; the new functions for FA and GLM, without the new sessions, return
# exactly those of §12.
#
# 60 new training runs (with and without the controller, 2 forms × 5 blocks × 3 seeds) and 5 GPFA fits, on 2.4 times the
# data: about 1 hour 30 the first time.

# %%
EXTRA_SID = ("19250108T111022", "19250108T111455")         # the other two sessions of the same day
EXTRA = {sid: io.load_session(SESS.parent / SESS.name.replace("19250108T110520", sid), bins="legacy") for sid in EXTRA_SID}  # same bins as the rest of the notebook
X_EX = {sid: s_.spikes[:, ACTIVE].astype(np.float32) for sid, s_ in EXTRA.items()}  # the counts of our 160 active channels
pieces_ex, inside_ex = [], []
for sid, s_ in EXTRA.items():
    for t in np.unique(s_.trial_num):                       # each trial of the session, as in ours (§9.2)
        a0, a1 = int(np.flatnonzero(s_.trial_num == t)[0]), int(np.flatnonzero(s_.trial_num == t)[-1]) + 1
        st = segment_starts(a0, a1)                         # one segment every 20 bins, the last aligned to the end of the trial
        pieces_ex.append(np.stack([X_EX[sid][a:a + SEG_LEN] for a in st]))
        inside_ex.append(all(np.all(s_.trial_num[a:a + SEG_LEN] == t) for a in st))
SEGS_EX = np.concatenate(pieces_ex)                         # the segments of the two new sessions
SEGS_EX_HID = SEGS_EX[..., ORDER_HID]                       # and with the 120 visible channels first, then the 40 hidden ones
rate_ours = COUNTS.mean(0) / io.BIN_S                       # mean rate of each channel (Hz), in our session
for sid, x_ in X_EX.items():
    r_ = x_.mean(0) / io.BIN_S
    print(f"{sid}: {len(np.unique(EXTRA[sid].trial_num))} trials, {x_.shape[0]} bins; mean rate of the channels: correlation with "
          f"our session {np.corrcoef(r_, rate_ours)[0, 1]:.3f} (with the channel order shuffled: "
          f"{np.corrcoef(np.random.default_rng(0).permutation(r_), rate_ours)[0, 1]:.3f}); mean {r_.mean():.2f} Hz against {rate_ours.mean():.2f}")
print(f"New segments: {len(SEGS_EX)} (the 12 training trials of a block contain about {n_seg(train_trials(0))}); "
      f"every segment within its trial: {all(inside_ex)}")

# %%
def job14d(form, k, sd):                                    # as in §12, with the segments of the new sessions added to the training segments
    hp_, data_ = FORMS14[form]
    j_ = lf12_job(sd, train_trials(k), hp=hp_, **data_)
    j_["train"] = j_["val"] = np.concatenate([j_["train"], SEGS_EX if form == "tutti i canali" else SEGS_EX_HID])
    return j_

def job14d_co(form, k, sd):                                 # the same, with the controller of §14b
    j_ = job14d(form, k, sd)
    j_["hp"] = dict(j_["hp"], **CO14)
    return j_

RUNS14D = [(f_, k, sd) for f_ in FORMS14 for k in range(5) for sd in SEEDS14]
res14d = dict(zip(RUNS14D, run_lfads_cached([job14d(*r) for r in RUNS14D])))
res14d_co = dict(zip(RUNS14D, run_lfads_cached_co([job14d_co(*r) for r in RUNS14D])))
GP14D = dict(zip(range(5), gpfa_cached([np.concatenate([SEGS[t] for t in train_trials(k)] + [SEGS_EX]) for k in range(5)])))  # time-matched GPFA, same data
display(pd.DataFrame({f"{lab}, {f_}": {"training segments": mm([len(job14d(f_, k, 0)["train"]) for k in range(5)]),
                                        "inferred inputs (from the weights)": mm([co_dim_of(r_[(f_, k, sd)]["weights"]) for k in range(5) for sd in SEEDS14]),
                                        "steps": mm([r_[(f_, k, sd)]["steps"] for k in range(5) for sd in SEEDS14]),
                                        "kept version (epoch)": mm([r_[(f_, k, sd)]["best_epoch"] for k in range(5) for sd in SEEDS14])}
                      for lab, r_ in (("without controller", res14d), ("with the controller", res14d_co)) for f_ in FORMS14}).T)

# %% [markdown]
# **Reading.**
#
# - **The channels are the same in the three sessions**: the mean rate of each channel in the new sessions correlates 0.986
#   and 0.983 with ours (about 0 if the channel order is shuffled), and the mean is almost equal (14.1 and 14.0 Hz
#   against 14.2).
# - **The extra data**: 662 new segments (16 trials), added to the roughly 488 of the 12 training trials: 1145-1153
#   segments, 3600-3800 steps in 200 epochs. Each new segment lies within its trial.
# - The models with the controller have the 4 inferred inputs, those without do not; the kept version is in practice the last
#   (between epoch 182 and 200).
# - The 60 training runs and the 5 GPFAs took about 2 hours.
#
# ### 14.8 · The two measures, with more data
#
# For each model, the score with our 12 trials (from §12, §13, §14b and §14c; for LFADS seeds 0-2) and
# the one with the 28 trials. The figure reuses the curves of §13b ("how many trials are needed") and adds the point with the new
# sessions.

# %%
SM_VIS_EX = [io.smooth_rates(s_.spikes, sigma_ms=50)[:, ACTIVE][:, HELD_IN] for s_ in EXTRA.values()]  # the 120 visible channels smoothed, new sessions

def fa_extra(k, extra=True):                                # FA with 25 factors on the 12 training trials (and on the new sessions): ordered factors and hidden channels
    Y = np.sqrt(COUNTS.astype(float))
    Y_fit = np.concatenate([Y[np.isin(trial, train_trials(k))]] + ([np.sqrt(x_.astype(float)) for x_ in X_EX.values()] if extra else []))
    fa_ = FactorAnalysis(n_components=K_PCA, svd_method="lapack", random_state=0).fit(Y_fit)
    L, psi, m = fa_.components_.T, fa_.noise_variance_, fa_.mean_  # like fa_hidden (§12.4)
    Lv, Lh = L[HELD_IN], L[HIDDEN]
    S = np.linalg.inv(np.eye(K_PCA) + Lv.T @ (Lv / psi[HELD_IN, None]))
    X = ((Y[:, HELD_IN] - m[HELD_IN]) / psi[HELD_IN]) @ Lv @ S  # factors of each bin from the visible channels only
    return order_by_readout(fa_.transform(Y), L), (X @ Lh.T + m[HIDDEN]) ** 2 + psi[HIDDEN] + np.einsum("hi,ij,hj->h", Lh, S, Lh)

def glm_extra(k, extra=True):                               # smoothing + GLM (§11a) fitted on the 12 training trials (and on the new sessions)
    fit_ = np.isin(trial, train_trials(k))
    F_fit = np.concatenate([SM_VIS[fit_]] + (SM_VIS_EX if extra else []))
    C_fit = np.concatenate([COUNTS[fit_]] + (list(X_EX.values()) if extra else []))
    sc = StandardScaler().fit(F_fit)
    return np.column_stack([PoissonRegressor(alpha=0.1, max_iter=500).fit(sc.transform(F_fit), C_fit[:, h]).predict(sc.transform(SM_VIS))
                            for h in HIDDEN])

def glm_on(lat, k):                                         # Poisson regression from any factors (whole session) to the hidden channels, on the 12 training trials of block k
    fit_ = np.isin(trial, train_trials(k))
    F = StandardScaler().fit(lat[fit_]).transform(lat)
    return np.column_stack([PoissonRegressor(alpha=0.1, max_iter=500).fit(F[fit_], COUNTS[fit_, h]).predict(F) for h in HIDDEN])

f_chk, h_chk = fa_extra(0, extra=False)
print(f"Check: without the new sessions, fa_extra = fa_factors and fa_hidden (maximum differences {np.abs(f_chk - F_FA12[0]).max():.1e} e "
      f"{np.abs(h_chk - fa_hidden(train_trials(0))).max():.1e}); glm_extra = smooth_glm_rates "
      f"({np.abs(glm_extra(0, extra=False) - smooth_glm_rates(train_trials(0), 0.1)).max():.1e})")

FA14D = {k: fa_extra(k) for k in range(5)}                  # block -> (factors, hidden channels), with the new sessions
F_GP14D = {k: stitch_all([gpfa_infer_segments(SEGS[t], GP14D[k]) for t in TRIALS]) for k in range(5)}
FULL14D = {"smoothing + GLM": [[glm_extra(k)] for k in range(5)],
           "FA": [[FA14D[k][1]] for k in range(5)],
           "GPFA alla pari": [[np.concatenate([stitch(gpfa_hidden(SEGS[t], GP14D[k])[0], t) for t in TRIALS])] for k in range(5)],
           "GPFA + regressione di Poisson": [[gpfa_glm(k, gp=GP14D)] for k in range(5)],
           "LFADS (3 semi)": [[stitch_all(res14d[("120 + 40", k, sd)]["rates"]) for sd in SEEDS14] for k in range(5)],
           "LFADS con il controller (3 semi)": [[stitch_all(res14d_co[("120 + 40", k, sd)]["rates"]) for sd in SEEDS14] for k in range(5)],
           "LFADS + regressione di Poisson (3 semi)": [[glm_on(stitch_all(res14d[("120 + 40", k, sd)]["factors"]), k) for sd in SEEDS14] for k in range(5)]}
B14D = {name: [bits_per_spike(test_only([full[k][i] for k in range(5)]), Y_HID) for i in range(len(full[0]))] for name, full in FULL14D.items()}
SN14D = {name: seen_new(full, Y_HID) for name, full in FULL14D.items()}
B12 = {"smoothing + GLM": BPS12[("smoothing + GLM (reference)", "tutta la sessione")], "FA": BPS12[("FA", "tutta la sessione")],
       "GPFA alla pari": BPS12[("GPFA alla pari", "tutta la sessione")],
       "GPFA + regressione di Poisson": BPS12[("variante: GPFA + regressione di Poisson", "tutta la sessione")],
       "LFADS (3 semi)": M14["principale (§12)"]["hidden"], "LFADS con il controller (3 semi)": M14B["controller"]["hidden"],
       "LFADS + regressione di Poisson (3 semi)": b_lfglm[:len(SEEDS14)]}  # 12 trial
display(pd.DataFrame({name: {"12 trials": fmt4(B12[name]), "12 + 16 trials": fmt4(B14D[name]),
                             "gain": f"{np.mean(B14D[name]) - np.mean(B12[name]):+.4f}",
                             "new / seen, 12 + 16 trials": fmt2(SN14D[name][:, :, 1].mean(1) / SN14D[name][:, :, 0].mean(1))}
                      for name in FULL14D}).T.rename_axis("bits per spike, 40 hidden channels, test trials; LFADS: mean [min, max] across the 3 seeds"))

ACC14D = {"FA": [acc14(lambda k: FA14D[k][0])],
          "GPFA alla pari": [acc14(lambda k: F_GP14D[k])],
          "LFADS (3 semi)": [acc14(lambda k, sd=sd: order_by_readout(stitch_all(res14d[("tutti i canali", k, sd)]["factors"]),
                                                                      res14d[("tutti i canali", k, sd)]["weights"]["readout.0.weight"])) for sd in SEEDS14],
          "LFADS con il controller (3 semi)": [acc14(lambda k, sd=sd: order_by_readout(stitch_all(res14d_co[("tutti i canali", k, sd)]["factors"]),
                                                                                        res14d_co[("tutti i canali", k, sd)]["weights"]["readout.0.weight"]))
                                               for sd in SEEDS14]}
A12 = {"FA": (ACC12[("FA", "tetto: tutti i 25 fattori")], ACC12[("FA", "primi 3 fattori")]),
       "GPFA alla pari": (M14[f"GPFA alla pari, {K_PCA} fattori"]["ceiling"], M14[f"GPFA alla pari, {K_PCA} fattori"]["primi 3"]),
       "LFADS (3 semi)": (M14["principale (§12)"]["ceiling"], M14["principale (§12)"]["primi 3"]),
       "LFADS con il controller (3 semi)": (M14B["controller"]["ceiling"], M14B["controller"]["primi 3"])}
display(pd.DataFrame({name: {"ceiling, 12 trials": fmt(A12[name][0]), "ceiling, 12 + 16 trials": fmt([a for a, _ in v]),
                             "first 3, 12 trials": fmt(A12[name][1]), "first 3, 12 + 16 trials": fmt([b for _, b in v])}
                      for name, v in ACC14D.items()}).T.rename_axis("phase decoded (%), chance 12.5"))
for lab in ("LFADS (3 semi)", "LFADS con il controller (3 semi)"):
    for meas, gp_v, lf_v in (("phase, ceiling (points)", ACC14D["GPFA alla pari"][0][0], [a for a, _ in ACC14D[lab]]),
                             ("hidden (bits per spike)", B14D["GPFA alla pari"][0], B14D[lab])):
        print(f"With 12 + 16 trials, time-matched GPFA minus {lab}, {meas}: {(fmt1s if 'phase' in meas else fmt4)(gp_v - np.array(lf_v))}; semi sopra la GPFA: "
              f"{int(np.sum(np.array(lf_v) > gp_v))} of {len(lf_v)}")

# %%
fig, ax = plt.subplots(1, 2, figsize=(15, 5))
curves14d = ((ax[0], ["FA", "GPFA alla pari", "LFADS (3 semi)"], lambda name, n: ACC13[(name, n, "tetto: tutti i 25 fattori")],
              lambda name: [a for a, _ in ACC14D[name]], "phase decoded (%), ceiling (chance 12.5)", "Phase decoding, test trials"),
             (ax[1], ["smoothing + GLM (reference)", "FA", "GPFA alla pari", "GPFA + regressione di Poisson", "LFADS (3 semi)"],
              lambda name, n: BPS13[(name, n)], lambda name: B14D[name.replace(" (riferimento)", "")], "bits per spike, 40 hidden channels",
              "Hidden channels of the test trials"))
for a_, names, get13, get14, ylab, title in curves14d:
    for name in names:
        v = np.array([get13(name, n) for n in SIZES13])     # the curve of §13b: 4, 7, 10, 12 trials
        a_.plot(SIZES13, v.mean(1), color=COL13[name], marker="o", lw=2, label=name)
        if v.shape[1] > 1:
            a_.fill_between(SIZES13, v.min(1), v.max(1), color=COL13[name], alpha=0.2)  # min-max across seeds
        w_ = np.array(get14(name))                          # the new point: 12 + 16 trials
        a_.plot([12, 28], [v.mean(1)[-1], w_.mean()], color=COL13[name], lw=1, ls=":")  # dotted: we move on to the new sessions
        a_.errorbar(28, w_.mean(), yerr=[[w_.mean() - w_.min()], [w_.max() - w_.mean()]], color=COL13[name], marker="s", mfc="white",
                    ls="none", capsize=3)
    a_.set(xticks=list(SIZES13) + [28], xlabel="training trials (28 = our 12 + the 16 of the other two sessions; empty squares)",
           ylabel=ylab, title=title)
for a_, v12, v28 in ((ax[0], M14B["controller"]["ceiling"], [a for a, _ in ACC14D["LFADS con il controller (3 semi)"]]),
                     (ax[1], M14B["controller"]["hidden"], B14D["LFADS con il controller (3 semi)"])):  # the controller: only 12 and 28 trials
    a_.plot([12, 28], [np.mean(v12), np.mean(v28)], color="C4", lw=1, ls=":", marker="D", ms=6, label="LFADS with the controller (§14b)")
ax[1].axhline(0, color="k", lw=0.8)
ax[0].legend(fontsize=8); ax[1].legend(fontsize=8, ncol=2)
fig.tight_layout(); fig.savefig(FIGS / "04_lfads_moredata.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# **Reading.**
#
# - **The check passes**: without the new sessions, the new functions for FA and GLM return exactly those of §12.
# - **With more than double the data, LFADS without a controller reaches the GPFA.**
#   - **Phase**: 51.7% (from 50.7 to 52.4 across seeds) against 51.1% for the time-matched GPFA; LFADS gains 4.7 points, the
#     GPFA none (51.4% with 12 trials). 2 seeds out of 3 above the GPFA: by the rule (3 out of 3) this is a tie, not an
#     overtaking.
#   - **Hidden channels**: 0.0460 against 0.0449, with all 3 seeds above: **by the rule, on the hidden channels the
#     verdict changes even without a controller**. LFADS gains 0.0073, the GPFA 0.0027.
# - **With the controller the advantage on the hidden channels grows**: 0.0742 (+0.0111), the highest score in the whole notebook.
#   On phase it rises to 45.8%, but stays 5.3 points below the GPFA.
# - **Those that gain most from the data are those with the most freedom**: smoothing + GLM (+0.0079, to 0.0495) and LFADS; FA
#   gains nothing (+0.0002), the GPFA little. With 28 trials smoothing + GLM and the GPFA with Poisson regression
#   (0.0498) still stay above LFADS without a controller; only LFADS with the controller beats them.
# - **Memorization drops for everyone**: on new trials 50-54% of the seen-trial score remains for LFADS (it was
#   40%), 79% for the GPFA.
# - **The figure**: the curves of §13b ("how many trials are needed") with the 28-trial point (empty squares; the dotted line
#   is a reminder that the 16 extra trials come from other sessions), and LFADS with the controller (purple diamonds) at 12 and 28 trials.
#   The LFADS curves keep rising, the GPFA curve on phase is flat.
# - **What it does not say**: the extra trials come from sessions recorded a few minutes later, not from the same one (the
#   28-trial point is not strictly on the same curve as §13b); a single way of adding the data (no
#   multi-session model); what would happen with even more data.
#
# ### Summary of §14d
#
# | question | answer | where |
# |---|---|---|
# | with 28 trials instead of 12, does LFADS reach the GPFA on phase? | **it reaches it**: 51.7% against 51.1%, but with 2 seeds out of 3 (a tie by the rule) | §14.8 |
# | and on the hidden channels? | **it beats it** with all 3 seeds: 0.0460 against 0.0449 | §14.8 |
# | and with the controller? | on the hidden channels even more (0.0742); on phase it stays below (45.8%) | §14.8 |
# | who gains from the data? | LFADS and smoothing + GLM; the GPFA almost nothing, FA nothing | §14.8 |
#
# **In short**: the gap of §12 was largely a matter of data. With 12 trials LFADS memorizes and stays
# below; with 28 it reaches the GPFA on phase and beats it on the hidden channels, while the GPFA, which with little data was
# already at its maximum, does not improve.

# %% [markdown]
# ## 15 · LFADS summary, final table and conclusions
#
# ### 15.1 · The path, in brief
#
# Part B put LFADS to the test on the same data and with the same measures as all the other methods. The details,
# the choices and the alternatives are in the summaries at the end of each section; here is the thread.
#
# | step | what | main result | summary |
# |---|---|---|---|
# | §9 · the data | raw counts at 20 ms; 600 ms segments within each trial, stitched; 40 hidden channels | stitching returns the counts; no segment crosses two trials | §9 |
# | §10 · the idea | LFADS written by hand on a Lorenz system, against the `lfads-torch` library | coincides with the library (10⁻⁷); recovers the hidden state better than the GPFA (R² 0.90-0.96 against 0.85-0.90) | §10 |
# | §11 · first training and grid | 17 configurations, chosen on the validation trials only | dropout 0.05, KL × 0.3, no coordinated dropout, 200 epochs; it memorizes a lot, strong regularization switches it off | §11 |
# | §12 · the like-for-like comparison | common metric, trajectories, hidden channels; GPFA on the same segments | with 12 trials below the GPFA on phase (46.5% against 51.4%) and on the hidden channels (0.0388 against 0.0422) | §12a, §12b |
# | §13 · the checks | seeds, memorization, Poisson twin, shifted hidden channels, how many trials are needed | the result is not an artifact; LFADS memorizes; with more trials the gap shrinks | §13a, §13b |
# | §14 · sensitivity and alternatives | sizes, regularization, segments, controller, no dynamics, more data | no network variant changes the verdict; the controller (on hidden channels) and more data do | §14a-§14d |
#
# **The question of part B**: *on these data, do the nonlinearity and the dynamics of LFADS improve the single-trial
# trajectories compared with the GPFA?*

# %% [markdown]
# ### 15.2 · The final table: all methods, the same measures
#
# Built by the code with the numbers of this notebook: the table of §8 (part A) with, in addition, the models of
# part B. Three measures:
#
# - **the phase decoded on never-seen trials** (the common metric: 1106 points, 5 blocks, 10-neighbor classifier,
#   chance 12.5%), with 2, 3 and 7 free coordinates and with all the dimensions of the model (the ceiling). For FA, GPFA,
#   LFADS and the per-bin VAE the coordinates are the first factors ordered by importance; for CEBRA, on the sphere, the
#   free coordinates are one fewer than the dimensions;
# - **the hidden channels of the test trials** (bits per spike, §12.4): only for the models that reconstruct the activity;
# - **the trajectories** (§4.2): roughness, consistency between repetitions of the same phase, and stability across seeds.
#
# Where there is a range: mean (min-max) across seeds.

# %%
FIN = final_dec.copy()                                      # the table of §8: 2, 3 and 7 free coordinates
FIN["tutte le dimensioni"] = ""
FIN.loc["PCA", "tutte le dimensioni"] = f"{ACC['tetto, 25 dimensioni']:.1f}"  # the ceiling from 03: the classifier on the 25 PCs
FIN.loc["GPFA (primi fattori)", "tutte le dimensioni"] = fmt(ACC12[("GPFA come nel §7b", "tetto: tutti i 25 fattori")])

def fin_row(lab, a2=None, a3=None, a7=None, ceiling=None):    # a new row, in the same format as the table of §8
    return {"labels": lab, "2 coordinate libere": fmt(a2) if a2 is not None else "", "3 coordinate libere": fmt(a3) if a3 is not None else "",
            "7 coordinate libere": fmt(a7) if a7 is not None else "", "tutte le dimensioni": fmt(ceiling) if ceiling is not None else ""}

for name, key in (("FA (primi fattori ordinati)", "FA"), ("GPFA alla pari (primi fattori ordinati)", "GPFA alla pari"),
                  ("LFADS (primi fattori ordinati, 5 semi)", "LFADS")):
    FIN.loc[name] = fin_row("no", ACC12[(key, "primi 2 fattori")], ACC12[(key, "primi 3 fattori")], ACC12[(key, "primi 7 fattori")],
                            ACC12[(key, "tetto: tutti i 25 fattori")])
FIN.loc["VAE per bin (primi fattori ordinati, 3 semi)"] = fin_row("no", a3=M14C["VAE per bin (3 semi)"]["primi 3"],
                                                                  ceiling=M14C["VAE per bin (3 semi)"]["ceiling"])
FIN.loc["LFADS con il controller (primi fattori ordinati, 3 semi)"] = fin_row("no", a3=M14B["controller"]["primi 3"], ceiling=M14B["controller"]["ceiling"])
for name, key in (("UMAP sui fattori della GPFA alla pari", "GPFA alla pari"), ("UMAP sui fattori di LFADS (5 semi)", "LFADS")):
    FIN.loc[name] = fin_row("no", a3=ACC12[(key, "UMAP, d = 3")])
ORDER15 = ["PCA", "Isomap", "UMAP", "CEBRA-Time (3, 4 e 8 dimensioni)", "FA (primi fattori ordinati)", "GPFA (primi fattori)",
           "GPFA alla pari (primi fattori ordinati)", "LFADS (primi fattori ordinati, 5 semi)", "LFADS con il controller (primi fattori ordinati, 3 semi)",
           "VAE per bin (primi fattori ordinati, 3 semi)",
           "UMAP sui fattori della GPFA alla pari", "UMAP sui fattori di LFADS (5 semi)"]
ORDER15 += [m_ for m_ in FIN.index if m_ not in ORDER15]  # the methods with labels (§8) at the bottom
display(FIN.loc[ORDER15].rename_axis("phase decoded (%), never-seen trials; chance 12.5, the clock alone 48.5"))

HID15 = {"smoothing + GLM (reference)": (BPS12[("smoothing + GLM (reference)", "tutta la sessione")], B14D["smoothing + GLM"]),
         "FA": (BPS12[("FA", "tutta la sessione")], B14D["FA"]),
         "VAE per bin (3 semi)": (M14C["VAE per bin (3 semi)"]["hidden"], None),
         "GPFA alla pari": (BPS12[("GPFA alla pari", "tutta la sessione")], B14D["GPFA alla pari"]),
         "GPFA + regressione di Poisson": (BPS12[("variante: GPFA + regressione di Poisson", "tutta la sessione")], B14D["GPFA + regressione di Poisson"]),
         "LFADS (§12.4: 5 semi; §14d: 3 semi)": (BPS12[("LFADS (5 semi)", "tutta la sessione")], B14D["LFADS (3 semi)"]),
         "LFADS + regressione di Poisson (§14c: 5 semi; §14d: 3 semi)": (b_lfglm, B14D["LFADS + regressione di Poisson (3 semi)"]),
         "LFADS con il controller (§14b, §14d: 3 semi)": (M14B["controller"]["hidden"], B14D["LFADS con il controller (3 semi)"])}
display(pd.DataFrame({name: {"12 training trials (§12.4, §14c)": fmt4(a_), "12 + 16 trials, with the other two sessions (§14d)": fmt4(b_) if b_ is not None else "—"}
                      for name, (a_, b_) in HID15.items()}).T.rename_axis("hidden channels of the test trials (bits per spike, 40 channels)"))

STAB15 = {"parte A: PCA": "100 (by construction)", "parte A: Isomap": "100 (by construction)",
          "parte A: UMAP": mean_range(cons["UMAP, d = 3"] * 100) + " (d = 3)",
          "parte A: CEBRA-Time, 4 dimensioni": mean_range(cons["CEBRA-Time, 4 dimensioni (stimato su tutti i trial)"] * 100),
          "FA (tutti i trial)": "100 (single model)", "GPFA alla pari (tutti i trial)": "100 (single model)",
          "LFADS (tutti i trial, 5 semi)": (mean_range(cons12["LFADS, primi 3 fattori ordinati, stimato su tutti i trial"] * 100) + " (primi 3); "
                                            + mean_range(cons12["LFADS, 25 fattori, stimato su tutti i trial"] * 100) + " (25)")}
TRAJ15 = pd.DataFrame(rows_t12).T
TRAJ15["stability across seeds (R², %)"] = [STAB15.get(i_, "") for i_ in TRAJ15.index]
display(TRAJ15.fillna("not computed").rename_axis("trajectories (§4.2, §12.3): first 3 factors or coordinates"))

# %%
UNSUP15 = {"PCA": (ACC["PCA, d = 3"], ACC["tetto, 25 dimensioni"]), "Isomap": (ACC["Isomap, d = 3"], None), "UMAP": (ACC["UMAP, d = 3"], None),
           "CEBRA-Time (4 dimensioni)": (ACC["CEBRA-Time, 4 dimensioni"], None),
           "FA": (ACC12[("FA", "primi 3 fattori")], ACC12[("FA", "tetto: tutti i 25 fattori")]),
           "VAE per bin": (M14C["VAE per bin (3 semi)"]["primi 3"], M14C["VAE per bin (3 semi)"]["ceiling"]),
           "GPFA alla pari": (ACC12[("GPFA alla pari", "primi 3 fattori")], ACC12[("GPFA alla pari", "tetto: tutti i 25 fattori")]),
           "LFADS": (ACC12[("LFADS", "primi 3 fattori")], ACC12[("LFADS", "tetto: tutti i 25 fattori")]),
           "LFADS con il controller": (M14B["controller"]["primi 3"], M14B["controller"]["ceiling"])}  # (3 free coordinates, all dimensions)
COL15 = {"PCA": "C1", "Isomap": "C9", "UMAP": "C2", "CEBRA-Time (4 dimensioni)": "C6", "FA": "C1", "VAE per bin": "C5",
         "GPFA alla pari": "C0", "LFADS": "C3", "LFADS con il controller": "C4"}
fig, ax = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw={"width_ratios": (1.2, 1)})
for i, (name, (a3, at)) in enumerate(UNSUP15.items()):    # left: the phase
    v = np.atleast_1d(a3)
    ax[0].barh(i, v.mean(), color=COL15[name], height=0.6, alpha=0.85, label="3 free coordinates" if i == 0 else None)
    if len(v) > 1:
        ax[0].plot([v.min(), v.max()], [i, i], color="k", lw=1)  # min-max across seeds
    if at is not None:
        ax[0].scatter(np.mean(at), i, marker="D", s=50, color="k", zorder=3, label="all dimensions (ceiling)" if name == "PCA" else None)
for xv, lab, ls in ((acc_clock, f"the clock alone ({acc_clock:.1f}%)", ":"), (12.5, "at chance", "-")):
    ax[0].axvline(xv, color="0.3", ls=ls, lw=1, label=lab)
ax[0].set(yticks=range(len(UNSUP15)), yticklabels=list(UNSUP15), xlabel="phase decoded (%), never-seen trials",
          title="Phase decoding, methods without labels")
ax[0].invert_yaxis(); ax[0].legend(fontsize=8, loc="lower right")
for i, (name, (a_, b_)) in enumerate(HID15.items()):      # right: the hidden channels
    v = np.atleast_1d(a_)
    c_ = "C2" if "regression" in name else ("0.45" if "smoothing" in name else ("C4" if "controller" in name else ("C3" if "LFADS" in name else ("C0" if "GPFA" in name else ("C5" if "VAE" in name else "C1")))))
    ax[1].barh(i, v.mean(), color=c_, height=0.6, label="12 training trials" if i == 0 else None)
    if len(v) > 1:
        ax[1].plot([v.min(), v.max()], [i, i], color="k", lw=1)
    if b_ is not None:
        ax[1].scatter(np.mean(b_), i, marker="s", s=45, facecolor="white", edgecolor="k", zorder=3,
                      label="12 + 16 trials, with the other two sessions" if i == 0 else None)
ax[1].set(yticks=range(len(HID15)), yticklabels=[n_.split(" (")[0] for n_ in HID15], xlabel="bits per spike, 40 hidden channels, test trials",
          title="Hidden channels (only the models that reconstruct the activity)")
ax[1].invert_yaxis(); ax[1].legend(fontsize=8, loc="lower right")
fig.tight_layout(); fig.savefig(FIGS / "04_final.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# **Reading.**
#
# - **Among the methods without labels, with few coordinates**, the picture from part A does not change: UMAP on the 25 PCs remains the
#   best with 2-3 free coordinates (41.2% and 42.6%). The first ordered LFADS factors are among the lowest in the
#   table (17.9% and 20.9% with 2 and 3), below PCA; only FA, the per-bin VAE and LFADS with the controller do worse.
#   For the nonlinear models the "first factors" are not a good description (§12.2, §13.1).
# - **With all the dimensions** the two GPFAs (50.7% and 51.4%) remain the best with 12 training trials, above
#   the clock (48.5%); LFADS reaches 46.5%, the controller lowers it, the per-bin VAE and FA stay below 26%. With 28
#   trials LFADS reaches the level of the GPFA (§14d; not in the table, which uses 12 trials for everyone).
# - **On the hidden channels** LFADS with the controller is the best, with 12 trials (0.0631) and with 28 (0.0742). Without a
#   controller LFADS is below the GPFA with 12 trials and above with 28.
# - **The trajectories**: those of LFADS are smooth within trials but restart at every change of trial, and are
#   as consistent as those of the GPFA (§12.3); its 25 factors are stable across seeds, the first 3 are not.
#
# ### 15.3 · Conclusions for the thesis
#
# **The answer to the question, in three points.**
#
# 1. **For single-trial trajectories and phase decoding of the task, no**: with the data of one session (12
#    training trials) the nonlinearity and the dynamics of LFADS do not improve on the GPFA, in any of the variants
#    tried (§12, §14a-§14c). With more than double the data (the other two sessions of the same day) LFADS
#    reaches it, but does not beat it (§14d).
# 2. **As a generative model of the hidden channels, yes, under two conditions**: with the controller (the inferred inputs)
#    LFADS beats everything already with 12 trials, because it follows the 18 Hz rhythm shared by the population that the smoother
#    models cancel (§14b, with three checks); without the controller it beats the GPFA only with more data (§14d).
# 3. **What matters in these data is modeling time, not nonlinearity**: from FA to the GPFA about 25
#    points are gained on phase, whereas the nonlinear map, without time (per-bin VAE) or with time (LFADS with
#    12 trials), adds nothing (§14c).
#
# **Why.** LFADS has many more parameters than the GPFA and with few trials it memorizes: on new trials it retains 40% of the
# seen-trial score, the GPFA 66% (§13.2). Every way of increasing its freedom (larger generator, lower
# KL) makes it worse, every way of restraining it switches it off (§11b, §14a); more data, instead, help it, and the GPFA, which with little
# data is already at its maximum, no longer improves (§13b, §14d).
#
# **What this means for a BCI.** On a short session, to describe the task phases and obtain
# interpretable trajectories, the GPFA is enough and is more robust (a single model, stable, little data). LFADS becomes interesting
# when more data are available or when predicting the activity of the channels matters (for example to reconstruct lost channels):
# there the full version, with the controller, is the best.
#
# > **Rule for the thesis.** An LFADS result must be reported with: the configuration (sizes, dropout, KL weight,
# > coordinated dropout, controller yes or no) and how it was chosen; the number of training trials and the epochs; the
# > segment length and the stitching; the seeds (mean and min-max) and the stability of the factors across seeds; the
# > **like-for-like** comparison with the GPFA (same segments, same trials) and with smoothing + GLM; the measure
# > (phase decoding with all factors and with a few, hidden channels in bits per spike) and the ratio between
# > new trials and seen trials. For example: *"LFADS (lfads-torch, 25 factors, generator 64, dropout 0.05, KL × 0.3,
# > no controller, 200 epochs, 600 ms segments) trained on 12 trials: phase decoded on never-seen trials 46.5%
# > (44.7-47.7, 5 seeds) against 51.4% for the GPFA on the same segments; hidden channels 0.0388 against 0.0422 bits per
# > spike; with the controller 0.0631; with 28 trials 51.7% and 0.0460"*.
#
# ### 15.4 · What we did not check
#
# - **AutoLFADS** (the automatic parameter search, Keshtkaran et al. 2022): not feasible on a laptop;
#   parameters chosen with a small grid on block 0 (§11b) and varied one at a time (§14), never in combination.
# - **The controller** in a single configuration (4 inputs, library priors); why it worsens the phase; the
#   controller with fewer or more inputs.
# - **The linear generator** and a per-bin VAE with a purpose-built network (here it is LFADS with 1-bin segments).
# - **More data** only from the same day, added as extra trials; no multi-session LFADS, no sessions from
#   different days.
# - **A single test session** and the 5 blocks of 03; a single draw of the hidden channels.
# - **Kinematics decoding** (the FALCON benchmark) was not done. Before doing it the binning must be corrected:
#   our row k corresponds to row k+1 of FALCON (bin k contains the spikes of the 20 ms *after* sample k instead of
#   before). It changes nothing in the measures of this notebook based on the phases; it changes slightly
#   the numbers that use the kinematics (part A, §6).
#
# ### 15.5 · References of part B
#
# - Sussillo, D., Jozefowicz, R., Abbott, L. F., Pandarinath, C. (2016). LFADS - Latent Factor Analysis via Dynamical
#   Systems. *arXiv*:1608.06315.
# - Pandarinath, C., O'Shea, D. J., Collins, J., et al. (2018). Inferring single-trial neural population dynamics using
#   sequential auto-encoders. *Nature Methods*, 15, 805–815.
# - Keshtkaran, M. R., Sedler, A. R., Chowdhury, R. H., et al. (2022). A large-scale neural network training framework
#   for generalized estimation of single-trial population dynamics. *Nature Methods*, 19, 1572–1577. (AutoLFADS.)
# - Sedler, A. R., Pandarinath, C. (2023). lfads-torch: a modular and extensible implementation of latent factor
#   analysis via dynamical systems. *arXiv*:2309.01230.
# - Pei, F., Ye, J., Zoltowski, D., et al. (2021). Neural Latents Benchmark '21: evaluating latent variable models of
#   neural population activity. *NeurIPS Datasets and Benchmarks Track*.
# - Kingma, D. P., Welling, M. (2014). Auto-encoding variational Bayes. *International Conference on Learning
#   Representations*.
# - Yu, B. M., et al. (2009). Gaussian-process factor analysis for low-dimensional single-trial analysis of neural
#   population activity. *Journal of Neurophysiology*, 102, 614–635.
# - Lorenz, E. N. (1963). Deterministic nonperiodic flow. *Journal of the Atmospheric Sciences*, 20, 130–141.
