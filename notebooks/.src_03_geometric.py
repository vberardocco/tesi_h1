# %% [markdown]
# # 03 · Nonlinear geometric methods: Isomap (and then UMAP) · FALCON H1
#
# **Goal.** In notebook 02 we described the population activity with **linear** methods
# (PCA, FA, GPFA). All three make the same assumption: that the data lie in a
# **flat subspace**, like a sheet laid out flat. Here we drop this assumption using the methods of the
# "geometric" family (review, §4.1), which reconstruct the manifold by looking only at **who is close
# to whom**:
#
# - **Isomap** (part A, this notebook): preserves the distances measured **along** the manifold;
# - **UMAP** (part B, §9, in brief): preserves the **neighbors**, not the distances.
#
# For Isomap there are three questions:
#
# 1. **Is the manifold curved?** Does measuring distances along the manifold, instead of in a straight line,
#    describe the data better?
# 2. **How many dimensions are needed** if curvature is taken into account? This is the distinction between the
#    *embedding* dimension (what PCA sees) and the *intrinsic* dimension (review, §2.4; Jazayeri &
#    Ostojic 2021).
# 3. **Are the task phases decoded better** in Isomap coordinates than in PCA coordinates?
#
# The pipeline:
#
# ```
# spike_times ──(as in 02: 20 ms bins, 160 active channels, σ = 50 ms, z-score)──► Z   12233 × 160
#
# Z ──(§2: movement phases only, one bin in 5, PCA to 25)────────────────────────► X   ~1100 × 25
#
# X ──(§3: Isomap = neighbor graph → geodesic distances → MDS)───────────────────► Isomap coordinates
#
# then:  §4  what we see                    §5  is the manifold curved? (residual variance + surrogates)
#        §6  are the phases decoded better? §7  sensitivity analysis and alternatives
#        §8  Isomap summary                 §9  part B: UMAP in brief
# ```
#
# Same session as notebooks 02 and 02_new (`held-in-calib`, `19250108T110520`), so the numbers
# are comparable.
#
# **Run time:** about 6-7 minutes: the GPFA fits of §7.2 and the UMAP repetitions of §9.

# %% [markdown]
# ## 0 · Setup

# %%
import os                                         # to set an environment variable before the other imports
os.environ.setdefault("KMP_WARNINGS", "off")      # hides a harmless OpenMP message that would appear with UMAP (§9)
os.environ.setdefault("NUMBA_THREADING_LAYER", "workqueue")  # numba (used by UMAP) does not load a second copy of OpenMP: together with the scikit-learn one the kernel sometimes hangs
import sys                                        # to add src/ to the paths where Python looks for modules
import warnings                                   # to hide harmless warnings
from pathlib import Path                          # file paths, independent of the operating system

import numpy as np                                # arrays and linear algebra
import pandas as pd                               # tables for the summaries
import matplotlib.pyplot as plt                   # figures
from scipy.sparse.csgraph import shortest_path, connected_components  # shortest paths on a graph; separate pieces of a graph
from scipy.spatial.distance import pdist, squareform, cdist  # pairwise distances; from "list of pairs" to matrix; distances between two sets
from sklearn.decomposition import PCA             # PCA (pre-reduction and comparison)
from sklearn.preprocessing import StandardScaler  # column-by-column z-score
from sklearn.neighbors import NearestNeighbors, kneighbors_graph, KNeighborsClassifier  # neighbor search; neighbor graph; neighbor-based classifier
from sklearn.manifold import Isomap               # scikit-learn Isomap, to check our hand computation
from sklearn.model_selection import GroupKFold    # cross-validation that keeps the bins of the same trial together
from sklearn.metrics import balanced_accuracy_score  # accuracy that weights all phases equally
from sklearn.datasets import make_swiss_roll      # the "rolled sheet": classic synthetic example of a curved manifold

ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()  # project root, whichever folder the notebook starts from
sys.path.insert(0, str(ROOT / "src"))             # makes the project package importable
from thesis_h1 import io                            # data loading, phases, colors (checked step by step in 02_new)

warnings.filterwarnings("ignore")                 # scikit-learn and elephant print harmless warnings (e.g. graph with several pieces)
FIGS = ROOT / "figures"                           # folder of the PNG figures
plt.rcParams.update({"figure.dpi": 110, "axes.grid": True, "grid.alpha": 0.25,
                     "axes.spines.top": False, "axes.spines.right": False})  # same plotting style as the other notebooks

SESS = ROOT / "data/raw/sub-HumanPitt-held-in-calib/sub-HumanPitt-held-in-calib_ses-19250108T110520.nwb"  # analyzed session

# %% [markdown]
# ## 1 · The data: the same as in notebook 02
#
# We start from the same data as the PCA of notebook 02: 20 ms bins, 160 active channels
# (≥ 0.5 Hz), Gaussian smoothing with σ = 50 ms, per-channel z-score. In 02_new every step was
# redone by hand and compared with the functions of `thesis_h1.io`, and they coincided: here we use
# the functions directly.
#
# To each bin we attach two labels, which **never** enter the methods (all are
# unsupervised) but are used to evaluate them:
#
# - the **task phase** in which the bin falls (`Reach`, `Orient`, ...);
# - the **trial** it belongs to.

# %%
s = io.load_session(SESS)                                   # 20 ms counts, kinematics, task phases
ACTIVE = io.active_units(s, min_hz=0.5)                     # the 160 active channels
Z = StandardScaler().fit_transform(                         # per-channel z-score...
    io.smooth_rates(s.spikes, sigma_ms=50)[:, ACTIVE].astype(float))  # ...of the firing rates smoothed with σ = 50 ms
n_bins = s.n_bins                                           # number of bins in the session

def base_tag(tag):                                          # "Presentation3" -> "Presentation": one label for all presentations
    return "Presentation" if tag.startswith("Presentation") else tag

phase = np.array([base_tag(p) for p in io.phase_per_bin(s)], dtype=object)  # phase of each bin ("" if no phase covers it)
trial = s.trial_num                                         # trial of each bin
MOV = np.isin(phase, io.MOVEMENT_PHASES)                    # true for the bins that fall in a movement phase

print(f"Z: {Z.shape[0]} bins × {Z.shape[1]} channels, {len(np.unique(trial))} trials")
print(f"Bins in movement phases: {MOV.sum()} of {n_bins} ({MOV.mean() * 100:.0f}%)")
print(pd.Series(phase[MOV]).value_counts().reindex(io.MOVEMENT_PHASES).rename("bins per phase").to_frame().T.to_string())

# %% [markdown]
# ## 2 · The neighbor graph: which data to give to Isomap
#
# ### Why the choice of input matters so much
#
# Isomap (and UMAP too, in part B) always starts the same way: each point "shakes hands" with its
# **k nearest points** (here k = 10). Everything else in the method is built on this network of
# handshakes. Choosing the input data therefore means **choosing who ends up close to whom**.
#
# What do we want? That a `Grasp` instant of trial 3 be close to `Grasp` instants of
# **other** trials: only then does the network link the repetitions and describe a **common** manifold, not the
# story of a single trial.
#
# For each point we split its 10 neighbors into three groups:
#
# - **time chain**: same trial and within 200 ms. It is "itself a moment before or after";
# - **same phase, other trial**: what we want;
# - **other**: different phase, or same trial but far in time.
#
# and we compare four possible inputs.

# %%
K_NN = 10                                                   # neighbors per point (the choice is checked in §7.1)
CHAIN_BINS = 10                                             # "time chain": same trial and within 10 bins = 200 ms

def knn_index(Xp, k=K_NN, forbid=None):                     # indices of the k neighbors of each point (excludes the point itself)
    D = squareform(pdist(Xp))                               # n × n matrix of straight-line distances
    np.fill_diagonal(D, np.inf)                             # a point is not its own neighbor
    if forbid is not None:                                  # forbidden pairs (needed for the Theiler window, §7.2)
        D[forbid] = np.inf
    return np.argpartition(D, k, axis=1)[:, :k]             # for each row, the k columns with the smallest distance

def neighbor_mix(nb, b, rows=None):                         # what kind of neighbors: three fractions summing to 1
    tr_, ph_ = trial[b], phase[b]                           # trial and phase of each point (b = bin of each point)
    chain = (tr_[nb] == tr_[:, None]) & (np.abs(b[nb] - b[:, None]) <= CHAIN_BINS)   # same trial, within 200 ms
    good = ~chain & (ph_[nb] == ph_[:, None]) & (tr_[nb] != tr_[:, None])            # same phase, other trial
    rows = np.ones(len(b), bool) if rows is None else rows  # which points to average over (all, or only the movement ones)
    return {"catena del tempo": chain[rows].mean(), "stessa fase, altro trial": good[rows].mean(),
            "other": 1 - chain[rows].mean() - good[rows].mean()}

def chance_good(b):                                         # the "same phase, other trial" fraction if the neighbors were chosen at random
    ph_, tr_ = phase[b], trial[b]
    return np.mean([np.mean((ph_ == ph_[i]) & (tr_ != tr_[i])) for i in range(len(b))])

STEP = 5                                                    # one bin in 5 = one every 100 ms
K_PCA = 25                                                  # dimensions kept by PCA (rationale below)
bins_all = np.flatnonzero(MOV)                              # all bins of the movement phases (one every 20 ms)
bins = bins_all[bins_all % STEP == 0]                       # those on a regular 100 ms grid
pca_in = PCA(n_components=K_PCA).fit(Z[bins_all])           # PCA fitted on all bins of the movement phases
Z25 = pca_in.transform(Z)                                   # the whole session projected onto the 25 directions
Zc = StandardScaler().fit_transform(s.spikes[:, ACTIVE].astype(float))  # raw z-scored counts, for the first comparison

inputs = {                                                  # the four inputs: (data of the whole session, bins used as points)
    "conteggi grezzi, 160 canali": (Zc, bins_all),
    "smussati σ = 50 ms, 160 canali": (Z, bins_all),
    "smussati + PCA a 25": (Z25, bins_all),
    "smussati + PCA a 25, un bin ogni 5": (Z25, bins),
}
mix = {name: neighbor_mix(knn_index(Xf[b]), b) for name, (Xf, b) in inputs.items()}  # neighbor composition for each input
tab_mix = (pd.DataFrame(mix).T * 100).round(1)
tab_mix["points"] = [len(b) for _, b in inputs.values()]
tab_mix["'stessa fase, altro trial' a caso"] = [round(chance_good(b) * 100, 1) for _, b in inputs.values()]
display(tab_mix)

# %%
cats, cols = ["catena del tempo", "stessa fase, altro trial", "other"], ["C3", "C2", "0.82"]
fig, ax = plt.subplots(figsize=(11, 3.3))
for i, m in enumerate(mix.values()):                        # one bar per input
    left = 0
    for c, col in zip(cats, cols):                          # the three groups one after the other, up to 100%
        ax.barh(i, m[c] * 100, left=left, color=col, label=c if i == 0 else None)
        if m[c] > 0.04:                                     # writes the percentage only if the piece is wide enough
            ax.text(left + m[c] * 50, i, f"{m[c] * 100:.0f}%", ha="center", va="center", fontsize=9)
        left += m[c] * 100
ax.set_yticks(range(len(mix))); ax.set_yticklabels(list(mix)); ax.invert_yaxis()
ax.set(xlabel="% of the 10 neighbors", xlim=(0, 100), title="Who are the 10 neighbors of each point?")
ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, -0.22), fontsize=9)
fig.tight_layout(); fig.savefig(FIGS / "03_neighbor_mix.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# **Reading.** There are two opposite traps.
#
# 1. **Raw counts: noise decides.** The "same phase, other trial" neighbors are barely above
#    chance. It is like comparing two very grainy photos pixel by pixel: the grain makes the difference,
#    not the subject.
# 2. **Smoothed data: the time chain.** Smoothing makes each bin almost a copy of the one
#    20 ms before (the two contain largely the same spikes). The nearest neighbor of
#    each point becomes "itself a moment before": this happens for almost all neighbors. It is like in a
#    film: the frame most similar to any frame is the one just before, and one never finds
#    the same scene shot in another take. On this network Isomap would reconstruct the **time within
#    each trial**, not the manifold.
#
# **PCA to 25 dimensions alone is not enough**: the chain remains. The decisive step is **one bin in
# 5**: the "copies" disappear, the neighbors have to be sought in the other repetitions, and those of the same
# phase in other trials become about twice chance. Subsampling therefore is not only for
# speed: above all it avoids this trap.
#
# ### The choices for the rest of the notebook
#
# - **Only the bins of the movement phases**: each point has a phase label and we can measure
#   how well the phases separate. Each bin appears **only once** (we do not use the 1 s windows of
#   notebook 02, which overlap when a phase lasts less than 1 s and would create
#   duplicate points at zero distance).
# - **PCA to 25 dimensions before Isomap.** It removes the directions with almost only noise
#   independent across channels. It does not "straighten" the manifold: a curved road on a map stays curved
#   even if the altitude is removed. 25 is the number of factors chosen by the GPFA cross-validation in
#   notebook 02, so we do not choose it ourselves, and it is larger than the number of dimensions we
#   expect to find: with 10 dimensions Isomap could never answer "12".
# - **One bin in 5 (100 ms).**
#
# All three choices are checked in §7.

# %%
X = Z25[bins]                                               # the points Isomap works on: n × 25
ph, tr = phase[bins], trial[bins]                           # phase and trial of each point
print(f"X: {X.shape[0]} points × {X.shape[1]} dimensions")
print(f"Variance of the movement bins kept by the 25 components: {pca_in.explained_variance_ratio_.sum() * 100:.0f}%")

# %% [markdown]
# The 25 components keep about a third of the variance. That is not little: the rest is largely the
# independent noise of the individual channels which, in notebook 02, pushed the participation ratio to
# ~106.
#
# ## 3 · Isomap: what it does
#
# ### The idea
#
# Isomap (Tenenbaum, de Silva & Langford 2000) is **a PCA that measures distances along the manifold
# instead of in a straight line**. Three steps:
#
# 1. **Neighbor graph.** Each point is linked to its k = 10 nearest neighbors: these are the
#    "roads" between points that certainly lie on the same piece of manifold.
# 2. **Geodesic distances.** The distance between any two points becomes the length of the
#    **shortest path along the roads**: the distance "on foot", not "as the crow flies".
# 3. **Classical MDS** (*multidimensional scaling*). We look for coordinates in d dimensions in which the
#    ordinary (straight-line) distances reproduce the geodesic distances as well as possible. It is the
#    same computation as PCA (eigenvalues and eigenvectors), done on the geodesic distances.
#
# If in step 3 straight-line distances were used, the result would be **exactly
# PCA** (we check this in §3.5). The whole difference between PCA and Isomap therefore lies in how
# distances are measured.
#
# ### An example on synthetic data: the swiss roll
#
# Before the real data, let us see what Isomap does on a manifold whose shape we know: a 2D sheet
# rolled up in 3D space (*swiss roll*). Two points on neighboring turns of the roll are close in a
# straight line but far apart along the sheet.
#
# To say "how many dimensions are needed" we use the **residual variance** (Tenenbaum 2000): how much of the
# distances that the method wants to preserve is **not** reproduced by the distances in the d-dimensional
# map. It equals 1 − r², where r is the correlation between the two lists of distances (all pairs of
# points). For Isomap the distances to preserve are the geodesic ones; for PCA the straight-line
# ones. The number of dimensions at which the curve **stops decreasing** (the "elbow") is the estimate
# of the dimension.

# %%
def residual_variance(d_target, Y):                         # 1 - r² between the distances to preserve and those in the map Y
    return 1 - np.corrcoef(d_target, pdist(Y))[0, 1] ** 2   # d_target: list of the distances of all pairs (pdist format)

def upper(M):                                               # from n × n matrix to list of pairs (i < j), in the same order as pdist
    return M[np.triu_indices(len(M), k=1)]

X_roll, pos_roll = make_swiss_roll(n_samples=1500, noise=0.3, random_state=0)  # 1500 points on the roll; pos_roll = position along the sheet
roll_pca = PCA(3).fit(X_roll)                               # PCA on the 3D data
iso_roll = Isomap(n_neighbors=K_NN, n_components=3).fit(X_roll)  # Isomap with the same k as the real data
Y_roll_pca, Y_roll_iso = roll_pca.transform(X_roll), iso_roll.embedding_  # coordinates in the two maps

rv_roll = pd.DataFrame({                                    # residual variance for d = 1, 2, 3
    "Isomap (geodesic distances)": [residual_variance(upper(iso_roll.dist_matrix_), Y_roll_iso[:, :d]) for d in (1, 2, 3)],
    "PCA (straight-line distances)": [residual_variance(pdist(X_roll), Y_roll_pca[:, :d]) for d in (1, 2, 3)],
}, index=pd.Index([1, 2, 3], name="dimensions d"))
display(rv_roll.round(3).T)

fig = plt.figure(figsize=(16, 3.9))
ax0 = fig.add_subplot(1, 4, 1, projection="3d")            # the roll in 3D space
ax0.scatter(*X_roll.T, c=pos_roll, cmap="viridis", s=4)
ax0.set(title="The swiss roll (3D)\ncolor = position along the sheet"); ax0.view_init(10, -70)
ax0.set_xticklabels([]); ax0.set_yticklabels([]); ax0.set_zticklabels([])
for k, (Y_, ttl) in enumerate([(Y_roll_pca, "PCA: the turns overlap"),
                               (Y_roll_iso, "Isomap: the sheet unrolls")], start=2):
    a_ = fig.add_subplot(1, 4, k)
    a_.scatter(Y_[:, 0], Y_[:, 1], c=pos_roll, cmap="viridis", s=4)   # same color: if the colors stay in order, the sheet is unrolled
    a_.set(title=ttl, xlabel="dimension 1", ylabel="dimension 2")
a_ = fig.add_subplot(1, 4, 4)
for col, c in zip(rv_roll.columns, ("C0", "C1")):
    a_.plot(rv_roll.index, rv_roll[col], "o-", color=c, lw=2, label=col)
a_.set(xlabel="dimensions d", ylabel="residual variance", xticks=[1, 2, 3], title="How many dimensions are needed?")
a_.legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIGS / "03_swiss_roll.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# **Reading.** In the roll, PCA squashes the turns onto one another: distant colors (points far
# apart along the sheet) end up overlapping. Isomap unrolls the sheet and the colors stay in order.
# The residual variance says the same thing in numbers: Isomap is almost at zero already with 1 dimension (the
# sheet is much longer than wide, so the length carries almost all the weight) and at zero with **2**, the
# intrinsic dimension of the sheet. PCA needs **3**, the embedding dimension. This is the signature of a curved manifold that we will look for in the real data: **Isomap drops
# faster than PCA**.
#
# Now we redo the three steps by hand on the real data.
#
# ### 3.1 · Step 1: the neighbor graph

# %%
G = kneighbors_graph(X, n_neighbors=K_NN, mode="distance")  # sparse n × n matrix: G[i, j] = distance if j is among the 10 neighbors of i, otherwise empty
n_pieces, _ = connected_components(G, directed=False)       # how many separate pieces the network is split into
print(f"{X.shape[0]} points, {G.nnz} links (= {X.shape[0]} × {K_NN}), separate pieces: {n_pieces}")

# %% [markdown]
# The network is **a single piece**: from every point one can reach every other by walking along the
# roads. If it were broken, no geodesic distance would exist between two pieces.
#
# ### 3.2 · Step 2: the geodesic distances
#
# For each pair of points we look for the shortest path along the roads (Dijkstra's algorithm,
# `method="D"`). `directed=False` means that each road works both ways: if j is among the neighbors
# of i, one can go from i to j and from j to i, even if i is not among the neighbors of j.

# %%
D_geo = shortest_path(G, method="D", directed=False)        # n × n: length of the shortest path along the graph
D_euc = squareform(pdist(X))                                # n × n: straight-line distance, for comparison
ratio = upper(D_geo) / upper(D_euc)                         # for each pair: how much longer the walking route is
print(f"Geodesic distance / straight-line distance: median {np.median(ratio):.2f} "
      f"(10% of pairs below {np.percentile(ratio, 10):.2f}, 10% above {np.percentile(ratio, 90):.2f})")
print(f"Correlation between the two distances: real data {np.corrcoef(upper(D_geo), upper(D_euc))[0, 1]:.2f}, "
      f"swiss roll {np.corrcoef(upper(iso_roll.dist_matrix_), pdist(X_roll))[0, 1]:.2f}")

# %%
rng = np.random.default_rng(0)                              # fixed seed: same pairs at every run
fig, ax = plt.subplots(1, 2, figsize=(12, 4.3))
for a_, (dg, de, ttl) in zip(ax, [(upper(iso_roll.dist_matrix_), pdist(X_roll), "Swiss roll"),
                                  (upper(D_geo), upper(D_euc), "Real data (X, 25 dimensions)")]):
    pick = rng.choice(len(dg), 20000, replace=False)        # 20000 random pairs, to avoid drawing millions of points
    a_.scatter(de[pick], dg[pick], s=2, alpha=0.25, color="C0")
    lim = max(de[pick].max(), dg[pick].max())
    a_.plot([0, lim], [0, lim], "--", color="0.4", lw=1, label="geodesic = straight line")
    a_.set(xlabel="straight-line distance", ylabel="geodesic distance (on foot)", title=ttl)
    a_.legend(loc="upper left", fontsize=8)
fig.tight_layout(); fig.savefig(FIGS / "03_geodesic_vs_euclidean.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# **Reading.**
#
# - **In the swiss roll** the figure fans out: many pairs are close in a straight line
#   (left) but very far on foot (top). These are the points on adjacent turns of the roll: it is
#   the information that PCA does not see.
# - **In the real data** the walking route is on average about **twice** the straight line, but the two
#   distances go almost in step (high correlation). A constant ratio says little about
#   curvature: if the cloud is noisy, even on a flat surface the path along the graph
#   zig-zags between the points and ends up longer than the straight line. There is no roll-like "fan".
#
# Nothing can be concluded from this figure yet: we need the measures of §5.
#
# ### 3.3 · Step 3: classical MDS
#
# We have the geodesic distances; now we look for coordinates that reproduce them. The trick is to go
# from **distances** to **inner products**, i.e. to a matrix that says how much two points "point in
# the same direction" relative to the center of the cloud. For points with true (centered) coordinates Y,
# the inner products are B = Y Yᵀ, and they are obtained from the distances by **double centering**:
#
# B = −½ · H · D² · H,   with H = I − 1/n (subtracts the mean of rows and columns).
#
# Then, as in PCA: eigenvectors and eigenvalues of B, and **coordinates = eigenvector × √eigenvalue**.
# The first axis is the one along which the geodesic distances are largest, and so on.

# %%
n = X.shape[0]                                              # number of points
H = np.eye(n) - 1.0 / n                                     # centering matrix: identity minus 1/n in every element
B = -0.5 * H @ (D_geo ** 2) @ H                             # double centering: from geodesic distances to inner products (n × n)
eigval, eigvec = np.linalg.eigh(B)                          # eigenvalues and eigenvectors (B is symmetric), in increasing order
order = np.argsort(eigval)[::-1]                            # indices for decreasing order
eigval, eigvec = eigval[order], eigvec[:, order]            # first the axis with the largest distances
D_MAX = 15                                                  # how many coordinates we keep
Y_iso = eigvec[:, :D_MAX] * np.sqrt(eigval[:D_MAX])         # Isomap coordinates: eigenvector × square root of the eigenvalue
print(f"Isomap coordinates: {Y_iso.shape}")
print(f"First 5 eigenvalues: {np.round(eigval[:5], 0)}")

def negative_share(Dg):                                     # how much the negative eigenvalues "weigh" relative to the positive ones
    n_ = len(Dg)
    ev_ = np.linalg.eigvalsh(-0.5 * (np.eye(n_) - 1.0 / n_) @ (Dg ** 2) @ (np.eye(n_) - 1.0 / n_))
    return -ev_[ev_ < 0].sum() / ev_[ev_ > 0].sum()

print(f"Negative eigenvalues: {(eigval < 0).sum()} of {n}; weight relative to the positive ones: "
      f"real data {negative_share(D_geo) * 100:.0f}%, swiss roll {negative_share(iso_roll.dist_matrix_) * 100:.0f}%")

# %% [markdown]
# **The negative eigenvalues.** If the geodesic distances were exactly the distances between points of
# a flat space, B would have no negative eigenvalues. The negative eigenvalues measure how much the
# geodesic distances **cannot be drawn in any flat space**, whatever the number
# of dimensions. In the swiss roll they weigh little: once unrolled, the sheet is flat. In the real
# data they weigh a lot: the distances along the graph are distorted by noise (the zig-zag of §3.2), and
# no map reproduces them well. It is a first sign that on these data Isomap works in difficult
# conditions, as the review warns: *"degrades when the manifold is (...) strongly corrupted by
# noise"*.
#
# ### 3.4 · Check: it coincides with scikit-learn

# %%
iso = Isomap(n_neighbors=K_NN, n_components=D_MAX).fit(X)  # the same thing, with scikit-learn
print("Same geodesic distances:", np.allclose(iso.dist_matrix_, D_geo))
print("Correlation (absolute value) between our coordinates and sklearn's, per axis:",
      np.round([abs(np.corrcoef(Y_iso[:, k], iso.embedding_[:, k])[0, 1]) for k in range(D_MAX)], 6))

# %% [markdown]
# They coincide. As for PCA (02_new, §6), the **sign** of each axis is arbitrary: an axis may
# come out flipped from one computation to another. From here on we use `iso`, which also has the `transform` method
# to project new points (needed in §6 and §6.1).
#
# ### 3.5 · The bridge with PCA
#
# If every point is linked to **all** the others (k = n − 1), the shortest path between two points is
# the direct one: the geodesic distances become the straight-line ones, and Isomap must give
# exactly PCA.

# %%
iso_full = Isomap(n_neighbors=n - 1, n_components=3).fit(X)  # complete graph: every point linked to all the others
Y_pca3 = PCA(3).fit_transform(X)                            # PCA on the same points
print("Geodesic distances = straight-line distances:", np.allclose(iso_full.dist_matrix_, D_euc))
print("Correlation (absolute value) Isomap with k = n − 1 vs PCA, per axis:",
      np.round([abs(np.corrcoef(iso_full.embedding_[:, k], Y_pca3[:, k])[0, 1]) for k in range(3)], 6))

# %% [markdown]
# Identical. **Isomap contains PCA as a special case**: with small k it measures the distances
# along the manifold, with k = n − 1 it goes back to PCA. The number of neighbors k decides how "locally"
# we look at the manifold (we vary it in §7.1).
#
# ## 4 · What we see: Isomap and PCA on the same points
#
# At the top, each point of X in the first two Isomap and PCA coordinates, colored by phase;
# the two panels have **the same axes**, so the difference in scale between the maps is also visible.
# At the bottom, the **mean trajectories** of each condition: we project all the bins (at 20 ms) of the
# 1 s windows of notebook 02 and average the repetitions. As in 02_new (§8), a
# condition is a pair (phase, task version): the task alternates two versions between odd
# and even trials (we show this in §9.2), so the repetitions of a phase are not all replicas of the
# same movement. There are 16 conditions, with 7-8 repetitions each. For Isomap we use `transform`, which
# places each new point starting from its neighbors among the points of X.
#
# The comparison PCA is fitted **on the same points X**: same data, only the way of
# measuring distances changes.

# %%
pca_x = PCA(n_components=D_MAX).fit(X)                      # PCA on the same points as Isomap
Y_pca = pca_x.transform(X)
windows = io.phase_windows(s, win_s=1.0, align="start")     # the 119 1 s windows from the start of each phase (notebook 02)
win_X = [Z25[w["start"]:w["stop"]] for w in windows]        # each window in the 25-dimensional space (50 bins × 25)
traj = {"Isomap": [iso.transform(x_) for x_ in win_X],      # each window in Isomap coordinates...
        "PCA": [pca_x.transform(x_) for x_ in win_X]}       # ...and in PCA coordinates

VERSIONS = ("odd", "even")                              # the two task versions: odd and even trials (§9.2)
def cond_of(w):                                             # the condition of a window: (phase, version), as in 02_new
    return (w["tag"], "odd" if w["trial"] % 2 == 1 else "even")
CONDS = [(p, v) for p in io.MOVEMENT_PHASES for v in VERSIONS]  # the 16 conditions
VER_LS = {"odd": "-", "even": "--"}                     # solid line = odd trials, dashed = even trials

fig, ax = plt.subplots(2, 2, figsize=(12, 9.5))
ax[0, 1].sharex(ax[0, 0]); ax[0, 1].sharey(ax[0, 0])        # top row: same axes for Isomap and PCA, so the difference in scale is visible
for j, (name, Y_) in enumerate([("Isomap", iso.embedding_), ("PCA", Y_pca)]):
    for p in io.MOVEMENT_PHASES:                            # cloud of points, one color per phase
        m = ph == p
        ax[0, j].scatter(Y_[m, 0], Y_[m, 1], s=6, alpha=0.5, color=io.PHASE_COLORS[p])
    ax[0, j].set(title=f"{name}: all points ({len(X)})", xlabel=f"{name} 1", ylabel=f"{name} 2")
    for (p, v) in CONDS:                                    # mean trajectory of each condition
        avg = np.mean([t_[:, :2] for t_, w in zip(traj[name], windows) if cond_of(w) == (p, v)], axis=0)
        ax[1, j].plot(avg[:, 0], avg[:, 1], color=io.PHASE_COLORS[p], ls=VER_LS[v], lw=2)
        ax[1, j].plot(avg[0, 0], avg[0, 1], "o", color=io.PHASE_COLORS[p], ms=6)  # dot = start of the phase
    ax[1, j].set(title=f"{name}: mean of the repetitions of each condition", xlabel=f"{name} 1", ylabel=f"{name} 2")
handles = ([plt.Line2D([], [], color=io.PHASE_COLORS[p], lw=2, label=p) for p in io.MOVEMENT_PHASES]
           + [plt.Line2D([], [], color="0.3", ls=VER_LS[v], lw=1.6, label=f"trial {v}") for v in VERSIONS])
fig.legend(handles=handles, ncol=10, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(FIGS / "03_isomap_vs_pca.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# **Reading.**
#
# - In both maps there is a large central group where `Reach`, `Orient`,
#   `SnapTo`, `Carry` and `Orient2` overlap, and a few phases that separate from it: `Shape` (red), `Release`
#   (gray) and partly `Grasp` (purple). These are the hand phases, the same ones that in notebook 02
#   were the most reproducible, consistent with the position of the arrays in the hand area.
# - **Isomap gives more room to the hand phases**: `Shape` and `Release` move further from the
#   central group, and their mean trajectories are long and well separated. PCA separates them along
#   the first component, but squashes them more onto the others. Beware of the scale, though: in the two
#   top panels, which have the same axes, the Isomap cloud extends over more than twice
#   that of PCA. The Isomap coordinates reproduce distances measured **along the graph**,
#   which are always greater than or equal to the straight-line ones (median ratio 1.93, §3.2).
#   Part of this "more room" is therefore expected by construction: how much remains is told by the
#   measures of §5 and §6. The bottom panels instead have free axes, to read the shape of the
#   trajectories.
# - **The two task versions are already visible here.** In the bottom panels the means of the odd trials
#   (solid line) and of the even ones (dashed) of `Shape` go in different directions; in Isomap the
#   long trajectory of `Shape` is almost only that of the even trials. This is the difference between the two
#   versions that we measure in §9.2, and it is the reason why the means are taken per condition.
# - The mean Isomap trajectories are **rough**, those of PCA smooth. `transform` places
#   each new point starting from its neighbors among the points of X, and when the neighbors change the point
#   jumps (we measure this in §6.1).
#
# The figures say where to look, not how large the difference is: for that we need the measures
# of §5 and §6.
#
# ## 5 · Is the manifold curved?
#
# ### 5.1 · The residual variance on the real data
#
# It is the measure of the swiss roll (§3): for each number of dimensions d, how much of the distances to
# preserve the map does **not** reproduce. If the manifold is curved, the Isomap curve drops faster
# than that of PCA and has an elbow at the intrinsic dimension.

# %%
D_LIST = np.arange(1, D_MAX + 1)                            # d = 1, ..., 15

def rv_curves(Xp, k=K_NN, forbid=None):                     # residual variance of Isomap and PCA for d = 1..15
    iso_ = isomap_on(Xp, k, D_MAX, forbid)
    Yp_ = PCA(D_MAX, svd_solver="full").fit_transform(Xp)   # svd_solver="full": exact computation, without the random part of the fast solver
    r_iso = [residual_variance(upper(iso_.dist_matrix_), iso_.embedding_[:, :d]) for d in D_LIST]
    r_pca = [residual_variance(pdist(Xp), Yp_[:, :d]) for d in D_LIST]
    return np.array(r_iso), np.array(r_pca)

def isomap_on(Xp, k, d, forbid=None):                       # Isomap that also accepts forbidden pairs (Theiler window, §7.2)
    D = squareform(pdist(Xp))                               # straight-line distances
    if forbid is not None:
        D[forbid] = 1e3 * D.max()                           # a forbidden pair has a huge distance: it will never be chosen as a neighbor
    return Isomap(n_neighbors=k, n_components=d, metric="precomputed").fit(D)

rv_iso, rv_pca = rv_curves(X)
display(pd.DataFrame({"Isomap": rv_iso, "PCA": rv_pca, "Isomap advantage (PCA − Isomap)": rv_pca - rv_iso},
                     index=pd.Index(D_LIST, name="d")).round(3).T)

# %% [markdown]
# **Reading.**
#
# - **Isomap is slightly better than PCA with few dimensions**: about 3.5 percentage points less
#   residual variance with d = 1, 2, 3. The advantage shrinks and disappears around d = 7; with
#   many dimensions PCA wins. This is expected: with enough dimensions PCA reproduces
#   the straight-line distances by construction, whereas the geodesic distances cannot be reproduced
#   completely in any flat space (the negative eigenvalues of §3.3).
# - **No elbow.** The two curves drop gently. Unlike the swiss roll, there is
#   no number of dimensions beyond which "nothing more is needed".
#
# The advantage of Isomap is small. Is it a sign of curvature or an effect of the way Isomap
# treats noise?
#
# ### 5.2 · The decisive control: a "twin" of the data without curvature
#
# Isomap beats PCA by a little. But that little could come from how Isomap treats noise and the
# time chain, not from a real curvature. To find out we need a term of comparison: a dataset
# **twin** of X, with the same properties that PCA sees, but **flat by construction**.
#
# **How it is built: random-phase surrogates** (Theiler et al. 1992; Prichard & Theiler 1994).
# Any signal in time can be written as a sum of oscillations of different frequencies (Fourier).
# For each oscillation there are two numbers:
#
# - the **amplitude**, which says how much that frequency weighs: from it come the variance, the
#   correlations between dimensions and how fast the signal changes in time;
# - the **phase**, which says when the oscillation peaks: it is the precise alignment of the
#   phases that creates shapes, such as curves and separate groups.
#
# We keep the amplitudes and **shuffle the phases at random**, with the same shuffling for all 25
# dimensions. It is like an orchestra with the same instruments at the same volume, but in which each one
# comes in at a random moment: the "timbre" remains, the melody disappears. The twin has the same
# covariance (hence the same PCA) and the same speed of variation (hence the same time
# chain), but it is a Gaussian cloud **without curvature**.
#
# We build it on the whole session (the continuous signal) and then take the same bins as X.
# We repeat with 5 different shufflings.

# %%
def phase_randomize(Xt, seed):                              # surrogate: same Fourier amplitudes, random phases (equal for all columns)
    rng_ = np.random.default_rng(seed)
    F = np.fft.rfft(Xt - Xt.mean(0), axis=0)                # spectrum of each column: one complex number per frequency
    phases = np.exp(2j * np.pi * rng_.uniform(size=F.shape[0]))  # one random phase for each frequency
    phases[0] = 1                                           # the constant component is left untouched
    if len(Xt) % 2 == 0:
        phases[-1] = 1                                      # neither is the one at the maximum frequency (it must stay real)
    return np.fft.irfft(F * phases[:, None], n=len(Xt), axis=0) + Xt.mean(0)  # back to the time domain

N_SUR = 5                                                   # how many surrogates
sur = [phase_randomize(Z25, seed)[bins] for seed in range(N_SUR)]  # the twins of X, on the same bins
c_real, c_sur = np.cov(X.T), np.cov(sur[0].T)               # covariances between the 25 dimensions
print(f"Covariance of the twin vs real data: correlation between the elements {np.corrcoef(c_real.ravel(), c_sur.ravel())[0, 1]:.3f}")
print(f"Time chain (neighbors of the same trial within 200 ms): real data "
      f"{neighbor_mix(knn_index(X), bins)['catena del tempo'] * 100:.0f}%, twin "
      f"{neighbor_mix(knn_index(sur[0]), bins)['catena del tempo'] * 100:.0f}%")

rv_sur = [rv_curves(Xs) for Xs in sur]                      # residual variance of Isomap and PCA on each twin
rv_iso_s = np.array([r[0] for r in rv_sur]); rv_pca_s = np.array([r[1] for r in rv_sur])
adv_real, adv_sur = rv_pca - rv_iso, rv_pca_s - rv_iso_s    # Isomap advantage: on the data and on each twin
display(pd.DataFrame({"Isomap advantage, real data": adv_real,
                      "Isomap advantage, twins (mean)": adv_sur.mean(0),
                      "twins, minimum": adv_sur.min(0), "twins, maximum": adv_sur.max(0)},
                     index=pd.Index(D_LIST, name="d")).round(3).T)

# %%
fig, ax = plt.subplots(1, 2, figsize=(13, 4.3))
ax[0].plot(D_LIST, rv_iso, "o-", color="C0", lw=2, ms=4, label="Isomap, real data")
ax[0].plot(D_LIST, rv_pca, "o-", color="C1", lw=2, ms=4, label="PCA, real data")
ax[0].plot(D_LIST, rv_iso_s.mean(0), "--", color="C0", lw=1.5, label="Isomap, twins without curvature")
ax[0].plot(D_LIST, rv_pca_s.mean(0), "--", color="C1", lw=1.5, label="PCA, twins without curvature")
ax[0].set(xlabel="dimensions d", ylabel="residual variance", ylim=(0, 1), title="Residual variance")
ax[0].legend(fontsize=8)
for a_s in adv_sur:
    ax[1].plot(D_LIST, a_s * 100, color="0.6", lw=1)        # one twin per line
ax[1].plot([], [], color="0.6", label=f"twins without curvature ({N_SUR})")
ax[1].plot(D_LIST, adv_real * 100, "o-", color="C3", lw=2.2, ms=5, label="real data")
ax[1].axhline(0, color="0.3", lw=0.8)
ax[1].set(xlabel="dimensions d", ylabel="PCA − Isomap (percentage points)",
          title="Isomap advantage over PCA (above zero = Isomap better)")
ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIGS / "03_residual_variance.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# **Reading.**
#
# - The twin really has the same "linear" properties as the data: same covariance (correlation
#   0.97 between the elements) and same time chain (20% versus 21%).
# - **On the twin without curvature Isomap always loses**, and by a lot: from −4 points with d = 1 to −21
#   with d = 15, in all 5 twins. This is what one expects: on a flat and
#   noisy cloud, the distances along the graph are just straight-line distances distorted by the zig-zag.
# - **On the real data Isomap wins instead of losing** (up to d = 6). With d = 3: +3.6 points versus
#   −9.6 on average in the twins (−5.3 in the best one). The distance between the red curve and the gray ones,
#   about 13 points with d = 3, is the part of structure that a flat cloud with the same variances,
#   the same correlations and the same speed in time does **not** have.
# - In the left panel both methods have a much lower residual variance on the real
#   data than on the twins: the distances of the real data are dominated by a few large excursions (the hand
#   phases of §4), easier to reproduce with few dimensions.
#
# **Conclusion**, with an honest caveat. The control rules out that the advantage of Isomap
# comes from noise or the time chain: there is **nonlinear** structure. The twin, however, is also
# **Gaussian**, the real data are not. The control therefore says "there is something that a linear
# Gaussian model does not explain": it may be curvature, but also separate groups or rare extreme values. To
# tell them apart we would need a twin that also preserves the distribution of values (IAAFT surrogates,
# Schreiber & Schmitz 1996); we did not do this here.
#
# **Answers to questions 1 and 2.** The manifold has a nonlinear component, but a **modest** one. Isomap
# does **not** find a small, sharp intrinsic dimension in the single-trial data: the curve has no
# elbow.
#
# ### 5.3 · A second way to estimate the intrinsic dimension: TwoNN
#
# The residual variance looks at **all** the distances, even between distant points. There are
# **local** estimators, which look only at the very nearest neighbors. The most widely used is **TwoNN** (Facco et al. 2017): for each
# point one takes the ratio μ = r₂ / r₁ between the distance to the second and to the first neighbor. In a
# d-dimensional manifold, μ follows a known law that depends only on d: a straight line is enough to estimate d.
# We announced it as a "method-independent" estimator; let us see whether it holds up.
#
# We apply it to all the movement bins (one every 20 ms) and to one bin in 5, also with the
# **Theiler window** (Theiler 1986): points of the same trial within 100 or 200 ms
# are forbidden as neighbors, to remove the time chain. And we compare it with the twin.

# %%
def twonn(Xp, forbid=None, discard=0.1):                    # TwoNN estimate of the intrinsic dimension (Facco et al. 2017)
    D = squareform(pdist(Xp)); np.fill_diagonal(D, np.inf)  # distances, without the point itself
    if forbid is not None:
        D[forbid] = np.inf                                  # forbidden pairs
    r = np.partition(D, 1, axis=1)[:, :2]                   # distance to the first and second neighbor
    mu = np.sort(r[:, 1] / r[:, 0])                         # ratio μ = r2 / r1, in increasing order
    n_ = len(mu); mu = mu[: int(n_ * (1 - discard))]        # discards the largest 10% of the ratios (the noisiest)
    F = np.arange(1, len(mu) + 1) / n_                      # fraction of points with ratio ≤ μ
    x_, y_ = np.log(mu), -np.log(1 - F)                     # in these coordinates the law is a straight line through the origin...
    return (x_ @ y_) / (x_ @ x_)                            # ...whose slope is d

def theiler_mask(b, win_bins):                              # pairs of the same trial closer than win_bins (diagonal excluded)
    m = (trial[b][:, None] == trial[b][None, :]) & (np.abs(b[:, None] - b[None, :]) <= win_bins)
    np.fill_diagonal(m, False)
    return m

Z25_sur = phase_randomize(Z25, 0)                           # a twin of the whole session
rows = []
for step in (1, 5):                                         # all bins, or one in 5
    b_ = bins_all[bins_all % step == 0]
    for w_ms in (0, 100, 200):                              # Theiler window: none, ±100 ms, ±200 ms
        fb = theiler_mask(b_, w_ms // 20) if w_ms else None
        rows.append({"one bin every": step, "Theiler window (ms)": w_ms,
                     "TwoNN, real data": twonn(Z25[b_], fb), "TwoNN, twin": twonn(Z25_sur[b_], fb)})
display(pd.DataFrame(rows).set_index(["one bin every", "Theiler window (ms)"]).round(1))

# %% [markdown]
# **Reading.**
#
# - The estimates range **from 7.5 to 68** depending on how many points we keep and how wide the
#   Theiler window is: there is no stable value.
# - Above all, **the twin without curvature gives almost the same numbers** as the real data (at
#   most 5 apart out of 70, usually 1-2). TwoNN looks only at the very nearest neighbors, and at that scale the data are dominated by noise and by the
#   structure in time, which the twin preserves. Whatever number it gives, it is not measuring the
#   manifold.
#
# We do not use TwoNN as an estimate of the intrinsic dimension: on these data the global comparison of
# §5.2, which has a term of comparison, is more reliable. TwoNN remains among the alternatives tried.
#
# ## 6 · Are the phases decoded better in Isomap coordinates?
#
# A more concrete measure: from the coordinates of an instant, can we tell **in which task phase**
# the participant is? If Isomap "unrolls" the manifold, different phases should end up in different
# zones of the map.
#
# - **Neighbor classifier** (k = 10): for a new point we look at the 10 nearest training points
#   in the map and the most voted phase wins. It is the simplest classifier that does not assume linear
#   boundaries, so it does not disadvantage Isomap.
# - **Balanced accuracy**: the mean, over the 8 phases, of the fraction of points recognized. It weights
#   long and short phases equally. **Chance: 1/8 = 12.5%**.
# - **Cross-validation by trial** (5 blocks of 3 trials): PCA, Isomap and the classifier are fitted
#   on the training trials; the points of the test trials are projected with `transform`. No bin of the
#   same trial ends up in both training and test.
#
# We compare Isomap and PCA with the same number of dimensions d.

# %%
D_DEC = (2, 3, 5, 10, 25)                                   # dimensions tried

def decode_clean(source, b, K=K_PCA, fit_bins=bins_all, k=K_NN, d_list=D_DEC, forbid_win=0):
    """Balanced accuracy (%) of Isomap and PCA for each d, with cross-validation by trial.
    source: data of the whole session; b: bins used as points; K: dimensions of the initial PCA (None = none);
    fit_bins: bins on which the initial PCA is fitted; forbid_win: Theiler window in bins (0 = none)."""
    keep = MOV[b]                                           # the classifier learns and is evaluated only on movement bins
    pred = {(m, d): np.empty(len(b), dtype=object) for m in ("Isomap", "PCA") for d in d_list}
    for trn, tst in GroupKFold(n_splits=5).split(b, groups=trial[b]):  # 5 blocks of 3 trials: each trial in the test set exactly once
        if K is None:                                       # no initial PCA
            Xtr, Xte = source[b[trn]], source[b[tst]]
        else:                                               # the initial PCA sees only the training trials
            train_trials = np.unique(trial[b[trn]])
            pca_f = PCA(K).fit(source[fit_bins[np.isin(trial[fit_bins], train_trials)]])
            Xtr, Xte = pca_f.transform(source[b[trn]]), pca_f.transform(source[b[tst]])
        fb = theiler_mask(b[trn], forbid_win) if forbid_win else None
        iso_f = isomap_on(Xtr, k, max(d_list), fb)          # Isomap on the training points only
        pca_d = PCA(max(d_list), svd_solver="full").fit(Xtr)  # PCA on the same points (exact computation: with 160 columns the default has a random part)
        E = {"Isomap": (iso_f.embedding_, iso_f.transform(cdist(Xte, Xtr))),  # test points projected with transform
             "PCA": (pca_d.transform(Xtr), pca_d.transform(Xte))}
        for m, (Etr, Ete) in E.items():
            for d in d_list:                                # the first d coordinates
                clf = KNeighborsClassifier(n_neighbors=10).fit(Etr[keep[trn], :d], phase[b[trn]][keep[trn]])
                pred[(m, d)][tst] = clf.predict(Ete[:, :d])
    return {(m, d): balanced_accuracy_score(phase[b][keep], pred[(m, d)][keep]) * 100 for (m, d) in pred}

acc = decode_clean(Z, bins)                                 # main configuration
tab_dec = pd.DataFrame({m: [acc[(m, d)] for d in D_DEC] for m in ("Isomap", "PCA")},
                       index=pd.Index(D_DEC, name="dimensioni d"))
display(tab_dec.round(1).T)

# %%
fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(D_DEC, tab_dec["Isomap"], "o-", color="C0", lw=2, ms=7, label="Isomap")
ax.plot(D_DEC, tab_dec["PCA"], "s-", color="C1", lw=2, ms=7, label="PCA")
ax.axhline(100 / 8, color="0.5", ls="--", lw=1, label="chance (12.5%)")
ax.set(xscale="log", xticks=D_DEC, xticklabels=D_DEC, xlabel="dimensions d", ylabel="balanced accuracy (%)",
       ylim=(0, 50), title="Decoding the task phase from the coordinates (never-seen trials)")
ax.legend()
fig.tight_layout(); fig.savefig(FIGS / "03_phase_decoding.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# **Reading.**
#
# - All the values are **well above chance** (12.5%): even 2 dimensions contain information
#   about the phase.
# - **With few dimensions (2-5) Isomap is better than PCA**: +6 points with d = 2, +3.6 with d = 3. This is
#   consistent with the residual variance: Isomap puts more structure in the first coordinates.
# - **With 10-25 dimensions PCA wins** (about 42% versus 37-39%), and it is also the highest value
#   overall. Isomap therefore does not find more information: it **compresses** what is there into fewer
#   dimensions. It is useful for looking at the data in 2-3 dimensions, not for decoding the phases better.
# - The absolute values remain low (about 40%) because the arm phases (`Reach`, `Orient`,
#   `Carry`, `Orient2`) are confused with one another, as seen in §4 and as the almost zero consistency
#   of these phases in notebook 02 said.
#
# ### 6.1 · The single-trial trajectories
#
# Last measure, the same as in notebook 02: we take the 119 windows of 1 s (all bins, one every
# 20 ms), project them with Isomap and with PCA, and compute on the first 3 dimensions the
# **roughness** (how much the trajectory jumps from one bin to the next) and the **consistency** (how much the
# repetitions of the same condition resemble one another, i.e. same phase and same task version,
# averaged over the 16 conditions). The functions are those of 02_new (§9.4).

# %%
def roughness(trajs):                                       # mean roughness (02_new, §9.4)
    return float(np.mean([(np.diff((T_ - T_.mean(0)) / (T_.std(0) + 1e-9), n=2, axis=0) ** 2).mean()
                          for T_ in trajs]))                # z-score of each dimension, then squared second difference

def consistency(trajs):                                     # mean correlation between repetitions of the same condition (02_new, §9.4)
    conds_ = [cond_of(w) for w in windows]
    out = []
    for c in CONDS:
        R = np.stack([t_ for t_, cw in zip(trajs, conds_) if cw == c])  # (repetitions, 50 bins, 3 dimensions)
        out.append(np.mean([np.corrcoef(R[:, :, j])[np.triu_indices(len(R), 1)].mean() for j in range(R.shape[2])]))
    return float(np.mean(out))                              # mean over the 16 conditions

tab_traj = pd.DataFrame([{"method": m, "roughness (lower = better)": roughness([t_[:, :3] for t_ in traj[m]]),
                          "consistency (higher = better)": consistency([t_[:, :3] for t_ in traj[m]])}
                         for m in ("PCA", "Isomap")]).set_index("method")
display(tab_traj.round(4))

PH_SHOW = ("Release", "odd")                            # one of the most reproducible conditions in notebook 02
idx_ph = [i for i, w in enumerate(windows) if cond_of(w) == PH_SHOW]
t_win = np.arange(50) * 20                                  # ms from the start of the phase
fig, ax = plt.subplots(1, 2, figsize=(12, 3.6), sharey=True)
for a_, m in zip(ax, ("PCA", "Isomap")):
    R = np.stack([traj[m][i][:, 0] for i in idx_ph])        # first dimension of each repetition
    R = (R - R.mean()) / R.std()                            # same scale for the two methods
    for r_ in R:
        a_.plot(t_win, r_, color=io.PHASE_COLORS[PH_SHOW[0]], lw=0.8, alpha=0.6)
    a_.plot(t_win, R.mean(0), color="0.1", lw=2.2)          # mean of the repetitions
    a_.set(title=f"{m}: first dimension", xlabel="ms from the start of the phase")
ax[0].set_ylabel("z")
fig.suptitle(f"The {len(idx_ph)} repetitions of {PH_SHOW[0]} in {PH_SHOW[1]} trials, one per line (black = mean)", y=1.02)
fig.tight_layout(); fig.savefig(FIGS / "03_trajectories.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# **Reading.**
#
# - **Similar consistency**, slightly higher for Isomap (0.122 versus 0.109): the repetitions of the
#   same condition resemble one another a little more.
# - **Roughness about 20 times higher.** It is a structural limit of Isomap: it does not produce a smooth
#   function leading from the data to the coordinates. `transform` places each new point
#   by looking at its 10 neighbors among the points of X; when the group of neighbors changes, the point jumps.
#   In the figure this shows up as a jitter on the Isomap curves, absent from those of PCA.
# - It is the third common limit of geometric methods pointed out by the review (§4.1): they give a descriptive
#   map, not a model of the data. For single-trial trajectories the reference remains the
#   GPFA of 02_new, §10.5 (roughness 0.011, consistency 0.137 with the same per-condition measure).
#
# ## 7 · Sensitivity analysis and alternatives tried
#
# We change **one choice at a time** relative to the main configuration and, for each variant,
# compute the same measures:
#
# - the composition of the neighbors (§2);
# - the **advantage of Isomap over PCA** in the residual variance with d = 3 (§5), on the data and on a
#   twin without curvature (§5.2): what matters is the **difference** between the two;
# - phase decoding with d = 3, for Isomap and for PCA, always with the clean estimate of
#   §6.
#
# **A discarded shortcut.** To save time one could fit Isomap just once on
# **all** the points (it does not see the labels, so it seems harmless) and cross-validate only
# the classifier. We check it on the main configuration.

# %%
def decode_shortcut(E, b):                                  # the shortcut: coordinates fitted on all points, CV only on the classifier
    keep = MOV[b]
    E_, ph_, tr_ = E[keep], phase[b][keep], trial[b][keep]
    pred = np.empty(len(E_), dtype=object)
    for trn, tst in GroupKFold(n_splits=5).split(E_, groups=tr_):
        pred[tst] = KNeighborsClassifier(n_neighbors=10).fit(E_[trn], ph_[trn]).predict(E_[tst])
    return balanced_accuracy_score(ph_, pred) * 100

print(f"Phase decoded with d = 3, Isomap: clean estimate {tab_dec.loc[3, 'Isomap']:.1f}%, "
      f"shortcut {decode_shortcut(iso.embedding_[:, :3], bins):.1f}%")
print(f"                              PCA:    clean estimate {tab_dec.loc[3, 'PCA']:.1f}%, "
      f"shortcut {decode_shortcut(Y_pca[:, :3], bins):.1f}%")

# %% [markdown]
# **The shortcut is optimistic**: Isomap goes from 36.0% to 41.0% (+5 points), PCA from 32.4% to 34.5%
# (+2). The advantage of Isomap would seem almost twice as large. The reason: if Isomap is fitted on all the
# points, the points of the test trials also contribute to building the map. The labels do not
# enter, but the geometry does, and the map adapts to the points on which it is then evaluated. **From here on
# we always use the clean estimate**, even though it is slower.
#
# ### 7.1 · The number of neighbors k

# %%
D_EVAL = 3                                                  # dimensions used in the summary measures

def evaluate(source, b, K=K_PCA, fit_bins=bins_all, k=K_NN, forbid_win=0, surrogate=True, decode=True):
    """All the measures for one input: source = data of the whole session, b = bins used as points.
    The PCAs use svd_solver="full" (exact computation): with 160 columns the default would choose a fast
    solver that has a random part, and the result would change a little from one run to another."""
    Xf = PCA(K, svd_solver="full").fit(source[fit_bins]).transform(source) if K else source  # input of the whole session (for neighbors, residual variance, twin)
    Xp = Xf[b]
    fb = theiler_mask(b, forbid_win) if forbid_win else None
    mix_ = neighbor_mix(knn_index(Xp, k, fb), b, MOV[b])
    iso_ = isomap_on(Xp, k, D_EVAL, fb)
    adv = residual_variance(pdist(Xp), PCA(D_EVAL, svd_solver="full").fit_transform(Xp)) - \
          residual_variance(upper(iso_.dist_matrix_), iso_.embedding_)
    if surrogate:                                           # same measure on a twin without curvature
        Xs = phase_randomize(Xf, 0)[b]
        iso_s = isomap_on(Xs, k, D_EVAL, fb)
        adv_s = residual_variance(pdist(Xs), PCA(D_EVAL, svd_solver="full").fit_transform(Xs)) - \
                residual_variance(upper(iso_s.dist_matrix_), iso_s.embedding_)
    else:
        adv_s = np.nan
    acc_ = (decode_clean(source, b, K=K, fit_bins=fit_bins, k=k, d_list=(D_EVAL,), forbid_win=forbid_win) if decode
            else {("Isomap", D_EVAL): np.nan, ("PCA", D_EVAL): np.nan})
    return {"points": len(b), "graph pieces": connected_components(kneighbors_graph(Xp, k), directed=False)[0] if fb is None else np.nan,
            "time chain (%)": mix_["catena del tempo"] * 100,
            "same phase, other trial (%)": mix_["stessa fase, altro trial"] * 100,
            "Isomap advantage, data": adv * 100, "Isomap advantage, twin": adv_s * 100,
            "phase decoded, Isomap (%)": acc_[("Isomap", D_EVAL)],
            "phase decoded, PCA (%)": acc_[("PCA", D_EVAL)]}

K_LIST = (5, 10, 15, 20, 30, 50)
tab_k = pd.DataFrame([evaluate(Z, bins, k=k) for k in K_LIST], index=pd.Index(K_LIST, name="neighbors k"))
display(tab_k.round(1))

# %%
fig, ax = plt.subplots(1, 2, figsize=(12.5, 4))
ax[0].plot(K_LIST, tab_k["Isomap advantage, data"], "o-", color="C3", lw=2, ms=7, label="real data")
ax[0].plot(K_LIST, tab_k["Isomap advantage, twin"], "s--", color="0.55", lw=1.5, ms=7, label="twin without curvature")
ax[0].axhline(0, color="0.3", lw=0.8)
ax[0].set(xlabel="neighbors k", ylabel="PCA − Isomap, d = 3 (% points)", xticks=K_LIST,
          title="Isomap advantage in the residual variance")
ax[0].legend()
ax[1].plot(K_LIST, tab_k["phase decoded, Isomap (%)"], "o-", color="C0", lw=2, ms=7, label="Isomap")
ax[1].axhline(tab_k["phase decoded, PCA (%)"].iloc[0], color="C1", lw=2, label="PCA (does not depend on k)")
ax[1].axhline(12.5, color="0.5", ls="--", lw=1, label="chance level")
ax[1].set(xlabel="neighbors k", ylabel="balanced accuracy, d = 3 (%)", xticks=K_LIST, ylim=(0, 50),
          title="Phase decoding")
ax[1].legend()
fig.tight_layout(); fig.savefig(FIGS / "03_sensitivity_k.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# **Reading.**
#
# - **Residual variance.** The advantage of Isomap on the data is positive for k = 5-20 and disappears from 30
#   upward. This is expected: with many neighbors the "roads" become straight-line shortcuts and Isomap
#   approaches PCA (with k = n − 1 they coincide, §3.5). On the twin Isomap always loses (from −8 to
#   −12): the gap between data and twin is there for every k.
# - **Phase decoding.** Isomap ranges between 31.5% and 36.0%, PCA is at 32.4%. The advantage of §6
#   (+3.6 points) is the **largest** of the series: with the other values of k it ranges from −1 to +2 points.
#   k = 10 was chosen before seeing these numbers, but the advantage of Isomap in decoding
#   the phases must be reported for what it is: **small and dependent on k**.
# - The time chain decreases as k grows (with more neighbors, those "from the same moment" are
#   a smaller fraction), while the "same phase, other trial" neighbors stay around 23-28%.
#
# ### 7.2 · The alternative inputs
#
# We change **one choice at a time** relative to the main configuration (smoothed σ = 50 ms,
# PCA to 25, one bin in 5, movement phases only, k = 10):
#
# | variant | what changes | why try it |
# |---|---|---|
# | raw counts | no smoothing | trap 1 of §2 |
# | PCA to 10 / to 50 | how many dimensions PCA keeps | 25 must not be the limit of the answer |
# | no PCA | Isomap on the 160 channels | does the pre-reduction change the result? |
# | one bin in 2 / in 10 | the subsampling grid | how much the chosen step matters |
# | Theiler window | all bins (20 ms), but those of the same trial within 200 ms are forbidden as neighbors | removes the chain without discarding points |
# | whole session | also `Presentation` and `Intertrial` | do the pause moments change the map? |
# | per-condition means | the 16 mean trajectories, phase × version (800 points) | the "clean" signal of 02_new, §8 |
# | GPFA factors | the 25 GPFA factors in place of PCA | the cleanest possible input |

# %%
all_bins = np.arange(0, n_bins, STEP)                       # whole session, one bin in 5
alts = {                                                    # each variant changes only one thing relative to "main"
    "main": dict(source=Z, b=bins),
    "raw counts (PCA to 25)": dict(source=Zc, b=bins),
    "PCA to 10": dict(source=Z, b=bins, K=10),
    "PCA to 50": dict(source=Z, b=bins, K=50),
    "no PCA (160 channels)": dict(source=Z, b=bins, K=None),
    "one bin in 2": dict(source=Z, b=bins_all[bins_all % 2 == 0]),
    "one bin in 10": dict(source=Z, b=bins_all[bins_all % 10 == 0]),
    "Theiler window ±200 ms (all bins)": dict(source=Z, b=bins_all, forbid_win=CHAIN_BINS),
    "whole session": dict(source=Z, b=all_bins, fit_bins=np.arange(n_bins)),  # PCA fitted on the whole session
}
rows_alt = {name: evaluate(**kw) for name, kw in alts.items()}

# %% [markdown]
# **The per-condition means** (02_new, §8): for each condition (phase × version), the mean of the
# 7-8 repetitions of the 1 s window, projected onto the same 25 directions. They are 16 curves of 50
# points. Here it makes no sense
# to speak of trials or cross-validation: we only measure the advantage of Isomap in the residual
# variance.

# %%
PSTH = np.concatenate([np.mean([Z25[w["start"]:w["stop"]] for w in windows if cond_of(w) == c], axis=0)
                       for c in CONDS])                     # (16 conditions × 50 bins, 25)
iso_avg = isomap_on(PSTH, K_NN, D_EVAL)
rows_alt[f"per-condition means ({len(PSTH)} points)"] = {
    "points": len(PSTH),
    "graph pieces": connected_components(kneighbors_graph(PSTH, K_NN), directed=False)[0],
    "Isomap advantage, data": (residual_variance(pdist(PSTH), PCA(D_EVAL).fit_transform(PSTH))
                               - residual_variance(upper(iso_avg.dist_matrix_), iso_avg.embedding_)) * 100}

# %% [markdown]
# **The GPFA factors.** We fit the GPFA exactly as in 02_new (§10): square root of the
# counts, 1 s windows, 25 factors, 500 iterations, factors rotated and sorted by importance.
# The windows overlap, so from each window we keep only the bins that belong to its own
# phase: each bin appears only once. Then one bin in 5, as in the main configuration.
#
# For phase decoding the clean estimate requires a GPFA **for each block** of the
# cross-validation, fitted on the training trials only: the factors of the test trials are then computed
# with the training parameters, like `transform` for PCA. The 5 GPFAs run in parallel. Altogether
# this takes a few minutes.
#
# **The twin of the factors.** The surrogates of §5.2 require a continuous signal, and the factors
# exist only inside the windows. The GPFA **input**, the square root of the counts, is however continuous:
# we shuffle its Fourier phases and apply to it the same GPFA, with the parameters fitted
# on the real data. This is what happens for the other variants, where the same PCA is applied to the real data
# and to the twin. No new fit is needed, so we repeat it with 5 seeds; the table shows seed
# 0, as for the other rows. As a control, for seed 0 we also refit the GPFA directly
# on the twin.

# %%
import contextlib, logging, os                              # to silence the progress messages of elephant
from joblib import Parallel, delayed                        # to fit the 5 GPFAs in parallel
from elephant.gpfa import gpfa_core                         # the GPFA computation functions (as in 02_new)
import elephant.utils
elephant.utils.logger.setLevel(logging.ERROR)

X_counts = s.spikes[:, ACTIVE]                              # raw counts of the 160 active channels
Y_sqrt = np.sqrt(X_counts)                                  # their square root, over the whole session: it is the GPFA input

def make_seqs(win_list, Y=None):                            # the data in elephant's format (02_new, §10); Y = input, normally the square root of the counts
    Y = Y_sqrt if Y is None else Y
    return np.array([(w["stop"] - w["start"], Y[w["start"]:w["stop"]].T)  # (bins, channels × bins)
                     for w in win_list], dtype=[("T", int), ("y", "O")])

def fit_gpfa(win_list, Y=None):                             # GPFA with 25 factors on the given windows (02_new, §10); Y = input, see make_seqs
    np.random.seed(0)                                       # fixed starting point
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        params, _ = gpfa_core.fit(make_seqs(win_list, Y), x_dim=K_PCA, bin_width=20.0, em_max_iters=500, tau_init=100.0)
    return params

def gpfa_per_bin(params, Y=None):                           # the 25 sorted factors of each bin, computed with the given parameters (Y = input, see make_seqs)
    with open(os.devnull, "w") as devnull, contextlib.redirect_stdout(devnull):
        lat, _ = gpfa_core.exact_inference_with_ll(make_seqs(windows, Y), params, get_ll=False)  # factors in each window
        _, orth = gpfa_core.orthonormalize(params, lat)     # rotated and sorted by importance
    out = np.full((n_bins, K_PCA), np.nan)                  # a vector of 25 factors per bin (NaN where there is none)
    for w, x_ in zip(windows, orth["latent_variable_orth"]):
        own = phase[w["start"]:w["stop"]] == w["tag"]       # the bins of the window that belong to its own phase
        out[w["start"]:w["stop"]][own] = x_.T[own]
    return out

params_full = fit_gpfa(windows)                             # GPFA on all windows: for neighbors and residual variance
gp_full = gpfa_per_bin(params_full)                         # its factors, bin by bin
win_trial = np.full(n_bins, -1)                             # the trial of the window each bin comes from
for w in windows:
    own = phase[w["start"]:w["stop"]] == w["tag"]
    win_trial[w["start"]:w["stop"]][own] = w["trial"]
b_gp = np.flatnonzero(~np.isnan(gp_full[:, 0]))             # bins that have a GPFA value
b_gp = b_gp[(b_gp % STEP == 0) & (win_trial[b_gp] == trial[b_gp])]  # one bin in 5 (and trial of the bin = trial of the window)
row = evaluate(np.nan_to_num(gp_full), b_gp, K=None, surrogate=False, decode=False)  # neighbors and residual variance (the twin is computed below)
print(f"Points with GPFA factors: {len(b_gp)}, of which outside the movement phases: {(~MOV[b_gp]).sum()}")
print(pd.Series(phase[b_gp]).value_counts().rename("points per phase").to_frame().T.to_string())

def isomap_gain(Xp):                                        # advantage of Isomap over PCA in the residual variance with d = 3, as in evaluate
    iso_ = isomap_on(Xp, K_NN, D_EVAL)
    return (residual_variance(pdist(Xp), PCA(D_EVAL).fit_transform(Xp))
            - residual_variance(upper(iso_.dist_matrix_), iso_.embedding_)) * 100

gp_twins = [gpfa_per_bin(params_full, phase_randomize(Y_sqrt, seed)) for seed in range(N_SUR)]  # the twins: the same GPFA on the input with shuffled Fourier phases
gp_twin = gp_twins[0]                                       # seed 0, the same one used by evaluate for the other variants
gain_tw = [isomap_gain(G_[b_gp]) for G_ in gp_twins]
row["Isomap advantage, twin"] = gain_tw[0]
print(f"Twin of the GPFA factors (seed 0): covariance vs real data, correlation between the elements "
      f"{np.corrcoef(np.cov(gp_full[b_gp].T).ravel(), np.cov(gp_twin[b_gp].T).ravel())[0, 1]:.3f}; time chain "
      f"{neighbor_mix(knn_index(gp_twin[b_gp]), b_gp)['catena del tempo'] * 100:.0f}% (real data {row['time chain (%)']:.0f}%)")
Y_tw = phase_randomize(Y_sqrt, 0)                           # control: what if the GPFA is refitted directly on the twin (seed 0)?
gain_refit = isomap_gain(gpfa_per_bin(fit_gpfa(windows, Y_tw), Y_tw)[b_gp])
print(f"Isomap advantage on the {N_SUR} twins of the GPFA factors: from {min(gain_tw):.1f} to {max(gain_tw):.1f}, mean {np.mean(gain_tw):.1f} "
      f"(real data {row['Isomap advantage, data']:.1f}); with the GPFA refitted on the twin: {gain_refit:.1f}")

def gpfa_train(train_trials):                               # GPFA fitted on the training trials only; factors of all bins
    return gpfa_per_bin(fit_gpfa([w for w in windows if w["trial"] in train_trials]))

folds_gp = list(GroupKFold(n_splits=5).split(b_gp, groups=trial[b_gp]))  # 5 blocks of 3 trials, as in §6
gp_fold = Parallel(n_jobs=5)(delayed(gpfa_train)(np.unique(trial[b_gp[trn]])) for trn, _ in folds_gp)  # the 5 GPFAs in parallel
pred_gp = {m: np.empty(len(b_gp), dtype=object) for m in ("Isomap", "PCA")}
for (trn, tst), G_ in zip(folds_gp, gp_fold):
    Xtr, Xte = G_[b_gp[trn]], G_[b_gp[tst]]
    iso_f, pca_f = Isomap(n_neighbors=K_NN, n_components=D_EVAL).fit(Xtr), PCA(D_EVAL).fit(Xtr)
    for m, (Etr, Ete) in {"Isomap": (iso_f.embedding_, iso_f.transform(Xte)),
                          "PCA": (pca_f.transform(Xtr), pca_f.transform(Xte))}.items():
        pred_gp[m][tst] = KNeighborsClassifier(n_neighbors=10).fit(Etr, phase[b_gp[trn]]).predict(Ete)
row["phase decoded, Isomap (%)"] = balanced_accuracy_score(phase[b_gp], pred_gp["Isomap"]) * 100
row["phase decoded, PCA (%)"] = balanced_accuracy_score(phase[b_gp], pred_gp["PCA"]) * 100
rows_alt["GPFA factors (d = 25)"] = row

tab_alt = pd.DataFrame(rows_alt).T
display(tab_alt.round(1))

# %%
alt = tab_alt.drop(index=f"per-condition means ({len(PSTH)} points)").astype(float)  # without the means: variant not comparable (§7.2)
short = {"main": "main", "raw counts (PCA to 25)": "raw counts",
         "PCA to 10": "PCA to 10", "PCA to 50": "PCA to 50", "no PCA (160 channels)": "no PCA",
         "one bin in 2": "one bin in 2", "one bin in 10": "one bin in 10",
         "Theiler window ±200 ms (all bins)": "Theiler ±200 ms",
         "whole session": "whole session", "GPFA factors (d = 25)": "GPFA factors"}
lab = [short[i] for i in alt.index]                         # short labels for the axis
y = np.arange(len(alt))[::-1]                               # top to bottom, in table order

fig, ax = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
for a_, (c1_, c2_) in zip(ax, (("phase decoded, PCA (%)", "phase decoded, Isomap (%)"),
                               ("Isomap advantage, twin", "Isomap advantage, data"))):
    for yi, (_, r) in zip(y, alt.iterrows()):               # one segment per variant: what matters is the distance between the two points
        a_.plot([r[c1_], r[c2_]], [yi, yi], color="0.8", lw=2, zorder=1)
    a_.axhline(y[0] - 0.5, color="0.85", lw=0.8)            # separates the main configuration from the variants
ax[0].scatter(alt["phase decoded, PCA (%)"], y, color="C1", s=50, label="PCA", zorder=2)
ax[0].scatter(alt["phase decoded, Isomap (%)"], y, color="C0", s=50, label="Isomap", zorder=2)
ax[0].axvline(12.5, color="0.5", ls="--", lw=1, label="chance (12.5%)")
ax[0].set(yticks=y, yticklabels=lab, xlim=(0, 50), xlabel=f"phase decoded, d = {D_EVAL} (%)",
          title="Phase decoding (never-seen trials)")
ax[0].legend(loc="lower left", fontsize=8)
ax[1].scatter(alt["Isomap advantage, data"], y, color="C3", s=50, label="real data", zorder=2)
ax[1].scatter(alt["Isomap advantage, twin"], y, color="0.55", marker="s", s=40,
              label="twin without curvature", zorder=2)
ax[1].axvline(0, color="0.3", lw=0.8)
ax[1].set(xlabel=f"PCA − Isomap, d = {D_EVAL} (percentage points)", title="Isomap advantage in the residual variance")
ax[1].legend(loc="lower right", fontsize=8)
fig.tight_layout(); fig.savefig(FIGS / "03_alternatives.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# **Reading**, variant by variant (advantage = PCA − Isomap in the residual variance with d = 3; phases
# = decoding with d = 3, clean estimate).
#
# The figure summarizes the table. On the left, phase decoding: one segment per variant,
# from PCA to Isomap. On the right, the advantage of Isomap in the residual variance, on the real data and on the
# twin: what matters is the distance between the two points. Only the per-condition means are missing (a non-informative
# variant).
#
# - **Raw counts**: Isomap worse than PCA (−6.9 points) and phases at 21-22% for both. This is
#   trap 1 of §2: the graph is made of noise.
# - **Dimensions of the initial PCA.** With 10 or 25 dimensions Isomap decodes the phases better than
#   PCA (+3 / +3.6 points); with 50 or 160 worse (−3 / −2). The more noise directions enter the graph,
#   the worse Isomap does, while PCA stays at 32-33%. The advantage in the residual variance instead does not
#   follow a clear trend (1.8, 3.6, 8.6, 0.9): it is a global measure, less tied to
#   phase decoding.
# - **Subsampling step.** With one bin in 2 the time chain returns to 52%; with one
#   in 10 it drops to 6%, but only 554 points remain. The advantage in the residual variance is similar
#   (2.6-3.7) and always far above the twin.
# - **Theiler window** (all 5507 bins, neighbors of the same trial within 200 ms forbidden): the
#   chain disappears by construction, the "same phase, other trial" neighbors are the most numerous (36%), and
#   Isomap decodes the phases better than PCA by 6 points (36.6% versus 30.4%). It is the best variant
#   among those based on PCA, and leads to the **same conclusion** as the main
#   configuration.
# - **Whole session**: the pause moments take up many places among the neighbors ("same phase, other
#   trial" drops to 16%). PCA gets much worse at decoding the phases (25.6%), probably
#   because its first directions mostly describe the difference between pause and movement (we did not
#   verify this); Isomap holds up better (32.8%).
# - **Per-condition means** (16 curves, phase × version): the graph breaks into 2 pieces (some
#   mean curves are too far from the others), Isomap has to link them artificially, and the
#   result is worse than PCA (−4.7 points). With few smooth, separate curves this variant
#   is **not informative**.
# - **GPFA factors**: in the residual variance Isomap is on par with PCA (−0.7), but on the 5 twins passed
#   through the same GPFA it would lose by 8-11 points (−9.8 if the GPFA is refitted on the twin): the gap
#   between data and twin, about 9 points, is in line with the other variants. In phase
#   decoding Isomap has instead the **largest advantage of all**: 40.7% versus 30.5%, with GPFA and Isomap
#   fitted only on the training trials. The points are all in movement phases, but there are 872 and not
#   1106: the bins beyond the first second of the long phases are missing.
#
# Cleaning the input in time, therefore, does not change how much nonlinear structure there is in the global
# distances, but it greatly improves how the phases are arranged **locally**. This is what
# UMAP looks at (part B, §9).
#
# ## 8 · Isomap summary
#
# | step | choice | why | alternatives (§7) |
# |---|---|---|---|
# | data | movement-phase bins, smoothed σ = 50 ms, z-score | the same as the PCA of notebook 02; each point has a phase | whole session |
# | initial PCA | 25 dimensions | removes noise independent across channels; 25 = factors chosen for the GPFA; more than the expected dimensions | 10, 50, none (160) |
# | subsampling | one bin in 5 (100 ms) | breaks the time chain: neighbors "from the same moment" from 95% to 20% | one bin in 2 or in 10; Theiler window |
# | neighbors k | 10 | standard value, fixed before seeing the results | 5, 15, 20, 30, 50 |
# | dimensions for the measures | d = 3 | comparison with the trajectories of notebook 02 | 1-15 (residual variance), 2-25 (phases) |
#
# ### Alternatives tried in this notebook
#
# | what | values tried | effect | where |
# |---|---|---|---|
# | input for the neighbor graph | raw, smoothed, + PCA to 25, + one bin in 5 | "same phase, other trial" neighbors: 16, 2, 5, 27% (chance 13%) | §2 |
# | hand computation | our Isomap vs scikit-learn; k = n − 1 vs PCA | identical; with k = n − 1 Isomap is PCA | §3.4, §3.5 |
# | twin without curvature | 5 random-phase surrogates | on the twins Isomap always loses (−10 points on average with d = 3); on the data it wins (+3.6) | §5.2 |
# | local estimator TwoNN | one bin in 1 or 5; Theiler 0, 100, 200 ms | estimates from 7.5 to 68, the same on the twin: not usable | §5.3 |
# | phase-decoding estimate | clean vs shortcut | the shortcut is optimistic (+5 points Isomap, +2 PCA) | §7 |
# | neighbors k | 5, 10, 15, 20, 30, 50 | advantage in the residual variance up to k = 20; phases 31.5-36% (PCA 32.4%) | §7.1 |
# | dimensions of the initial PCA | 10, 25, 50, 160 | Isomap better than PCA with 10-25, worse with 50-160 | §7.2 |
# | step / time chain | every 2, 5, 10 bins; Theiler | chain 52, 20, 6, 0%; with Theiler +6 points on the phases | §7.2 |
# | which instants | movement phases vs whole session | with the whole session PCA collapses (25.6%), Isomap holds up (32.8%) | §7.2 |
# | per-condition means | 16 conditions phase × version, 800 points | graph in 2 pieces, Isomap worse than PCA (−4.7): not informative | §7.2 |
# | GPFA factors as input | 25 factors, GPFA fitted per block | phases: Isomap 40.7% versus PCA 30.5%, the largest advantage; residual variance: −0.7 on the data, −8.3 on the twin | §7.2 |
#
# ### What we learned about Isomap
#
# 1. **How the points are chosen matters more than the method.** On smoothed data the neighbor graph is
#    a time chain, on raw counts it is noise. Subsampling (or the Theiler
#    window) is needed after smoothing.
# 2. **There is nonlinear structure, but it is modest.** Isomap reproduces the distances slightly better than
#    PCA with 1-3 dimensions (+3.6 points), whereas on a linear Gaussian twin it would lose by
#    about 10: the difference, about 13 points, is structure that a linear Gaussian model does not
#    explain.
# 3. **No small intrinsic dimension in the single-trial data.** The residual variance has no
#    elbow, and the local estimator TwoNN on these data measures noise.
# 4. **Isomap compresses, it does not add.** With 2-5 coordinates it decodes the phases slightly better than PCA;
#    with 10-25 PCA is better. With the main configuration the advantage is small and depends on
#    k.
# 5. **The advantage in decoding the phases grows with a cleaner input**: +6 points with the
#    Theiler window, +10 with the GPFA factors. What improves is not global: with the GPFA factors
#    the distance from the twin in the residual variance does not grow (about 9 points, versus 13 for the
#    main configuration). "Local" needs to be qualified, though: on the 25 PCs the extra structure of the
#    twins is largest in neighborhoods of 20-50 points, not among the very nearest neighbors (§9.2, neighbors preserved as
#    the scale varies). Whether this is the scale that improves with the GPFA factors we did not
#    measure.
# 6. **Limits**: rough trajectories (neighbor-based `transform`), no model of the data, geodesic
#    distances strongly distorted by noise (negative eigenvalues equal to 63% of the positive ones, versus
#    4% for the swiss roll).
#
# > **Rule for the thesis.** An Isomap result must be reported with all the choices that produce it
# > and with its term of comparison, for example: *"Isomap (k = 10) on 25 PCs of the smoothed
# > firing rates (σ = 50 ms), one point every 100 ms in the movement phases: residual variance with 3
# > dimensions 0.38 versus 0.41 for PCA; on random-phase surrogates Isomap is always worse than
# > PCA (−10 points on average)"*.

# %% [markdown]
# ## 9 · Part B: UMAP, in brief
#
# UMAP (McInnes et al. 2018) starts from the same neighbor graph as Isomap (§2), but then does something
# different. Isomap measures the distances along the graph and looks for a map that reproduces **all** of them,
# even between distant points. UMAP only asks that **neighbors stay close** and that non-neighbors do not
# end up next to each other. It is like seating guests at a party so that each one has
# their own friends around them, without caring how far apart the various groups are.
#
# Three consequences, to keep in mind before looking at any map:
#
# - **Distances in the map are not interpretable**: neither those between distant groups, nor the size
#   of the groups. For this reason the residual variance of §5 makes no sense here, and UMAP says nothing about the
#   intrinsic dimension.
# - **The result depends on chance**: the map is adjusted with random steps. We fix the seed and
#   repeat the main measures (phase decoding, neighbors preserved on the real data, `Shape`
#   islands) with 5 different seeds. The maps of §9.2, the twins and the trajectories of §9.4 use a
#   single seed.
# - **Two parameters**: `n_neighbors` (how many neighbors per point: it is the k of Isomap, and we use the same
#   value, 10) and `min_dist` (how much the points may pile up in the map; standard value 0.1).
#
# This part is deliberately brief. It uses the same points, the same cross-validation blocks and
# the same twins as part A, and answers four questions:
#
# 1. With 2-3 coordinates, does UMAP decode the phases better than Isomap and PCA? (§9.1)
# 2. What does UMAP draw on a cloud without structure? (§9.2)
# 3. How much does the result depend on the seed and on the parameters? (§9.3)
# 4. What do the single-trial trajectories look like? (§9.4)
#
# ### 9.1 · Are the phases decoded better?
#
# Same measure as §6: 10-neighbor classifier, balanced accuracy (chance 12.5%),
# cross-validation by trial with the map fitted **only on the training trials**. We try it on two
# inputs: the 25 PCs of the main configuration and the 25 GPFA factors, which in §7.2 were the best
# input.
#
# We add a term of comparison: the classifier **without any map**, directly on the
# 25 dimensions. We call it the **ceiling**: a 2-3-dimensional map built from the neighbor graph
# can preserve the information that the neighbors contain, we do not expect it to add any. It is a
# term of comparison, not a mathematical limit. For the 25 PCs it is the "PCA with d = 25" of §6 (42.2%).

# %%
from umap import UMAP                                       # UMAP (umap-learn package)

def folds_pca(source, b):                                   # the 5 blocks of §6: the 25-dimensional PCA sees only the training trials
    out = []
    for trn, tst in GroupKFold(n_splits=5).split(b, groups=trial[b]):
        pca_f = PCA(K_PCA).fit(source[bins_all[np.isin(trial[bins_all], np.unique(trial[b[trn]]))]])
        out.append((pca_f.transform(source[b[trn]]), pca_f.transform(source[b[tst]]), trn, tst))
    return out

INPUTS_U = {                                                # the two inputs: (bins used as points, [(training, test, indices) per block])
    "25 PC (principale)": (bins, folds_pca(Z, bins)),       # same points and same blocks as §6
    "25 fattori GPFA": (b_gp, [(G_[b_gp[trn]], G_[b_gp[tst]], trn, tst)  # the 5 GPFAs per block of §7.2
                               for (trn, tst), G_ in zip(folds_gp, gp_fold)]),
}

def decode_map(inp, embed):                                 # balanced accuracy (%) with any map, clean estimate
    b, folds = inp
    pred = np.empty(len(b), dtype=object)
    for Xtr, Xte, trn, tst in folds:
        Etr, Ete = embed(Xtr, Xte)                          # the map is fitted on the training set; the test points are projected
        pred[tst] = KNeighborsClassifier(n_neighbors=10).fit(Etr, phase[b[trn]]).predict(Ete)
    return balanced_accuracy_score(phase[b], pred) * 100

def emb_umap(d, n_neighbors=K_NN, min_dist=0.1, seed=0):    # UMAP: n_jobs=1 because with a fixed seed it runs on a single core anyway
    def f(Xtr, Xte):
        u = UMAP(n_neighbors=n_neighbors, n_components=d, min_dist=min_dist, random_state=seed, n_jobs=1).fit(Xtr)
        return u.embedding_, u.transform(Xte)
    return f

def emb_isomap(d):                                          # Isomap as in §6
    def f(Xtr, Xte):
        i_ = Isomap(n_neighbors=K_NN, n_components=d).fit(Xtr)
        return i_.embedding_, i_.transform(Xte)
    return f

def emb_pca(d):                                             # PCA as in §6
    def f(Xtr, Xte):
        p_ = PCA(d).fit(Xtr)
        return p_.transform(Xtr), p_.transform(Xte)
    return f

def emb_none(Xtr, Xte):                                     # no map: the classifier works on the 25 dimensions
    return Xtr, Xte

SEEDS = range(5)                                            # the steps with which UMAP adjusts the map are random: we repeat with 5 seeds

def umap_seeds(name, **kw):                                 # the 5 accuracies (one per seed) of a UMAP configuration
    return np.array(Parallel(n_jobs=5)(delayed(decode_map)(INPUTS_U[name], emb_umap(seed=sd, **kw)) for sd in SEEDS))

rows_u = {}
for name, inp in INPUTS_U.items():
    col = {f"PCA, d = {d}": f"{decode_map(inp, emb_pca(d)):.1f}" for d in (2, 3)}
    col.update({f"Isomap, d = {d}": f"{decode_map(inp, emb_isomap(d)):.1f}" for d in (2, 3)})
    for d in (2, 3):
        a_ = umap_seeds(name, d=d)
        col[f"UMAP, d = {d} (mean of the 5 seeds; min-max)"] = f"{a_.mean():.1f} ({a_.min():.1f}-{a_.max():.1f})"
    col["no map, 25 dimensions (the ceiling)"] = f"{decode_map(inp, emb_none):.1f}"
    rows_u[name] = col
tab_u = pd.DataFrame(rows_u)
display(tab_u)
print(f"Ceiling of the 25 PCs on the same {len(b_gp)} points as the GPFA factors: "
      f"{decode_map((b_gp, folds_pca(Z, b_gp)), emb_none):.1f}%")

# %% [markdown]
# **Reading.**
#
# - **UMAP reaches the ceiling with 2-3 coordinates**: with d = 3, 42.6% versus 42.2% on the 25 PCs and 50.5%
#   versus 50.8% on the GPFA factors. Isomap with d = 3 stops at 36.0% and 40.7%, PCA at 32.4% and
#   30.5%.
# - **It does not exceed it appreciably.** The means are within about half a point of the ceiling, less
#   than the difference between one seed and another. UMAP does not find new information: it puts into 2-3
#   coordinates almost all that the neighbors contain in 25 dimensions. The conclusion of
#   §6 on Isomap holds (it compresses, does not add), only UMAP compresses much better.
# - **In part this is expected by construction.** UMAP's `transform` places a new point in the middle of
#   its training neighbors in 25 dimensions. The neighbor classifier in the map therefore queries
#   points of the same zone: not the same 10 (in §9.2 we will see that about 4 out of 10 remain), but they
#   vote on average in the same way. The measure says that the map preserves the phase information
#   contained in the neighborhoods, not that UMAP has discovered a shape.
# - **The ceiling depends on the input**: 42.2% on the 25 PCs, 50.8% on the GPFA factors. The two inputs do not have
#   the same points (1106 versus 872, because the GPFA factors exist only in the first second of each
#   phase, §7.2), but the difference remains at equal points: on the same 872 the ceiling of the 25 PCs is
#   43.0%. It is lesson 1 of Isomap (§8): how the points are prepared matters.
# - **Seeds matter little**: between the best and the worst seed there are 1-2 points.
#
# **A check on the input.** If UMAP preserves only the information that the neighbors contain, with a
# noisier input it must drop together with the ceiling. We try it without PCA, on the same 1106 points
# and the same blocks: the 160 smoothed channels and the raw counts of §2.

# %%
def folds_plain(src):                                       # no PCA: the data enter as they are, same blocks as §6
    return [(src[bins[trn]], src[bins[tst]], trn, tst)
            for trn, tst in GroupKFold(n_splits=5).split(bins, groups=trial[bins])]

for name, src in {"smoothed, 160 channels (no PCA)": Z, "raw counts, 160 channels": Zc}.items():
    inp = (bins, folds_plain(src))
    a_ = np.array([decode_map(inp, emb_umap(d=3, seed=sd)) for sd in SEEDS])   # 5 seeds
    print(f"{name}: UMAP d = 3 {a_.mean():.1f}% ({a_.min():.1f}-{a_.max():.1f}) | "
          f"no map {decode_map(inp, emb_none):.1f}%")

# %% [markdown]
# **Reading.** UMAP drops together with the ceiling: 40.2% versus 40.7% on the 160 smoothed channels, 17.0%
# versus 16.8% on the raw counts (chance 12.5%). Smoothing matters a lot (from 17.0% to 40.2%), the
# PCA to 25 shifts little (from 40.2% to 42.6%). To go up one has to raise the ceiling, i.e. give UMAP a
# cleaner input, such as the GPFA factors (50.5%).
#
# ### 9.2 · What does UMAP draw on a cloud without structure?
#
# The known risk of 2-dimensional maps such as UMAP is that they distort distances and neighborhoods, and that
# shapes and groups in the map are partly a product of the method (Chari & Pachter 2023). To
# check this we use the **twins without curvature** of §5.2: same covariance and same time
# chain as the real data, but no structure tied to the phases.
#
# First we look at the 2-dimensional maps: at the top the real data and the GPFA factors, at the bottom the 5
# twins of the 25 PCs and the twin of the GPFA factors (§7.2). Here UMAP is fitted on all points and with
# a single seed: fine for looking, not for measuring (it is the shortcut of §7). In the twins the phases
# do not exist: each point has the color of the corresponding bin in the real data, only for comparison.

# %%
def umap_fit(Xp, d, seed=0):                                # UMAP fitted on all the given points: for looking at the map, not for measuring
    return UMAP(n_neighbors=K_NN, n_components=d, random_state=seed, n_jobs=1).fit_transform(Xp)

def draw_map(a_, Xp, ph_, title, size=6):                   # a 2-dimensional UMAP map (seed 0), one color per phase
    Yu = umap_fit(Xp, 2)
    for p in io.MOVEMENT_PHASES:
        m = ph_ == p
        a_.scatter(Yu[m, 0], Yu[m, 1], s=size, alpha=0.6, color=io.PHASE_COLORS[p])
    a_.set(title=title, xlabel="UMAP 1", ylabel="UMAP 2")
    a_.set_aspect("equal", adjustable="datalim")            # same scale on both axes

fig = plt.figure(figsize=(16, 9.2))
gs = fig.add_gridspec(2, 12, height_ratios=(1.9, 1))        # two large maps at the top, six small ones at the bottom
draw_map(fig.add_subplot(gs[0, :6]), X, ph, f"UMAP: real data, 25 PCs ({len(X)} points)")
draw_map(fig.add_subplot(gs[0, 6:]), gp_full[b_gp], phase[b_gp],          # GPFA fitted on all windows (§7.2)
         f"UMAP: GPFA factors ({len(b_gp)} points)")
for i, Xs in enumerate(sur):                                # the 5 twins of §5.2: there the phases do not exist, the colors serve only as a comparison
    draw_map(fig.add_subplot(gs[1, 2 * i:2 * i + 2]), Xs, ph, f"twin {i + 1} of the 25 PCs", size=3)
draw_map(fig.add_subplot(gs[1, 10:]), gp_twin[b_gp], phase[b_gp], "twin of the GPFA factors", size=3)  # the twin of §7.2
handles = [plt.Line2D([], [], color=io.PHASE_COLORS[p], marker="o", ls="", ms=5, label=p) for p in io.MOVEMENT_PHASES]
fig.legend(handles=handles, ncol=8, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.01))
fig.tight_layout(rect=(0, 0.03, 1, 1)); fig.savefig(FIGS / "03_umap_maps.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# Then a measure. For each point we take its 10 neighbors in 25 dimensions and count how many
# remain among its 10 neighbors in the map: these are the **preserved neighbors**. It is the local measure
# corresponding to the residual variance of §5 and, as in §5.2, what matters is the **difference between data and
# twins**. On the real data we repeat UMAP with 5 seeds; for the twins we use the 5 twins, with one seed
# each. For the advantage of UMAP over PCA we report the mean, minimum and maximum.

# %%
def kept_neighbors(Xp, Y, k=K_NN):                          # of the k neighbors of each point in 25 dimensions, how many remain among the k neighbors in the map (%)
    a_, b_ = knn_index(Xp, k), knn_index(Y, k)
    return np.mean([len(np.intersect1d(a_[i], b_[i])) for i in range(len(a_))]) / k * 100

def kept_all(Xp, d, seed=0):                                # neighbors preserved by the three methods on the same points
    return {"PCA": kept_neighbors(Xp, PCA(d).fit_transform(Xp)),
            "Isomap": kept_neighbors(Xp, Isomap(n_neighbors=K_NN, n_components=d).fit_transform(Xp)),
            "UMAP": kept_neighbors(Xp, umap_fit(Xp, d, seed))}

rows_kn = {}
for d in (2, 3):
    runs = {f"d = {d}, dati veri (5 semi)": pd.DataFrame([kept_all(X, d, sd) for sd in SEEDS]),   # real data: 5 UMAP seeds
            f"d = {d}, gemelli (5)": pd.DataFrame([kept_all(Xs, d) for Xs in sur])}               # the 5 twins of §5.2, seed 0
    for label, r_ in runs.items():
        gap = r_["UMAP"] - r_["PCA"]                        # advantage of UMAP over PCA in each repetition
        rows_kn[label] = {**r_.mean(), "UMAP − PCA": gap.mean(), "UMAP − PCA, minimum": gap.min(),
                          "UMAP − PCA, maximum": gap.max(), "UMAP / PCA": r_["UMAP"].mean() / r_["PCA"].mean()}
tab_kn = pd.DataFrame(rows_kn).T
display(tab_kn.round(1))

# %% [markdown]
# **Reading.**
#
# - **On the twins UMAP does not draw islands.** In all 6 (the 5 of the 25 PCs and the one of the GPFA factors)
#   the map is a single blob with mixed colors. Only short filaments of a few
#   points remain, probably bins close in time from the same trial (we did not verify this). In the
#   twin of the GPFA factors they are more numerous, as expected from factors smooth in time, but
#   no phase forms an island. Covariance and time chain alone are therefore not enough to
#   produce the islands of the real data.
# - **The control has two limits.** The twin is Gaussian (§5.2): it does not exclude that UMAP exaggerates the
#   separation between groups that are connected in 25 dimensions. And of the 5 twins of the GPFA factors
#   (§7.2) here we look at only one, seed 0.
# - **In the real data the hand phases separate**, the same as in §4: `Shape`, `Release` and
#   `Grasp`. With the GPFA factors the islands are sharper. The arm phases (`Reach`, `Orient`,
#   `SnapTo`, `Carry`, `Orient2`) remain a single mixed group in both maps.
# - **UMAP preserves many more neighbors**: with d = 3, 38.6%, versus 12.0% for PCA and 14.6% for
#   Isomap. But it does so on the twins too (32.6% versus 9.8%): in both cases about 3 times PCA.
#   The advantage is therefore almost entirely a property of the method, which holds even for a Gaussian cloud
#   without structure.
# - **In points, the advantage is slightly larger on the data**: with d = 3, 26.6 (across seeds 26.5-26.8)
#   versus 22.8 (across twins 22.4-23.3); with d = 2, 26.8 versus 20.6. The intervals do not
#   touch, but the difference is small, and as a ratio it disappears.
# - It is consistent with §5.2, but here the sign change that made the control decisive there is missing:
#   PCA also preserves a few more neighbors on the data (12.0% versus 9.8%).

# %% [markdown]
# ### Neighbors preserved as the scale varies
#
# The measure above counts the neighbors preserved among the **10** nearest. But 10 is a choice: with a
# smaller number one looks at a more local scale, with a larger one at a more global scale. It is the
# question left open by §7.2: **at what scale** do the real data have more structure than the twins,
# local or global?
#
# There are two different k's, not to be confused:
#
# - the **k of the method** (`n_neighbors` of UMAP and of Isomap): how many neighbors the method uses to
#   build the map. It stays fixed at 10 (`K_NN`), as in the whole notebook;
# - the **k of the measure**: how many neighbors we count to say whether the map preserved them. It is the only one
#   that we vary: 5, 10, 20, 50, 100, 200. On 1106 points this means from less than 1% to 18%
#   of the points.
#
# The maps are 3-dimensional and are fitted only once for each set of points: the real data
# (UMAP with 5 seeds) and the 5 twins (one seed each), as in the measure above. There is no
# classifier or test, so the maps are fitted on all points.
#
# We add a reference, the **random map**: if the neighbors in the map were chosen at random, k out of
# n − 1 would remain. This value rises with k even without any structure (0.5% with k = 5,
# 18% with k = 200).

# %%
K_EVAL = (5, 10, 20, 50, 100, 200)                          # k of the MEASURE: how many neighbors are counted (the k of the METHOD stays K_NN = 10)
METHODS = ("PCA", "Isomap", "UMAP")                         # the three methods, in the order of the tables of §9.2

def maps3(Xp, seeds=(0,)):                                  # the three 3-dimensional maps of a set of points, fitted only once
    return {"PCA": [PCA(D_EVAL).fit_transform(Xp)],
            "Isomap": [Isomap(n_neighbors=K_NN, n_components=D_EVAL).fit_transform(Xp)],
            "UMAP": [umap_fit(Xp, D_EVAL, sd) for sd in seeds]}  # UMAP: one map per seed

def kept_by_k(Xp, maps):                                    # for each method: neighbors preserved (%), one row per map and one column per k
    return {m: np.array([[kept_neighbors(Xp, Y, k) for k in K_EVAL] for Y in Ys]) for m, Ys in maps.items()}

kept_data = kept_by_k(X, maps3(X, SEEDS))                   # real data: UMAP with the 5 seeds (5 rows), PCA and Isomap a single map (1 row)
kept_twin = [kept_by_k(Xs, maps3(Xs)) for Xs in sur]        # the 5 twins of §5.2, one UMAP seed per twin
kept_sur = {m: np.concatenate([kt[m] for kt in kept_twin]) for m in METHODS}  # per method: 5 rows (the twins) × 6 columns (the k's)
chance_k = np.array(K_EVAL) / (len(X) - 1) * 100            # "random map": with neighbors chosen at random in the map, k out of n − 1 would remain

def mean_range(A):                                          # "mean (minimum-maximum)" for each k, from a repetitions × k matrix
    return [f"{a.mean():.1f} ({a.min():.1f}-{a.max():.1f})" for a in A.T]

rows_ke = {}
for m in METHODS:
    d_, g_ = kept_data[m], kept_sur[m]                      # real data and twins of this method
    if len(d_) > 1:                                         # UMAP: 5 seeds, so mean and minimum-maximum
        rows_ke[f"{m}, real data (mean of the 5 seeds; min-max)"] = mean_range(d_)
    else:                                                   # PCA and Isomap: a single map
        rows_ke[f"{m}, real data"] = [f"{v:.1f}" for v in d_[0]]
    rows_ke[f"{m}, twins (mean of the 5; min-max)"] = mean_range(g_)
    rows_ke[f"{m}, data − twins"] = [f"{v:+.1f}" for v in d_.mean(0) - g_.mean(0)]  # difference between the two means, in percentage points
rows_ke["random map"] = [f"{v:.1f}" for v in chance_k]
tab_ke = pd.DataFrame(rows_ke, index=pd.Index(K_EVAL, name="k of the measure")).T
display(tab_ke)

u_d, u_g = kept_data["UMAP"].mean(0), kept_sur["UMAP"].mean(0)  # UMAP: mean of the seeds (data) and of the twins, for each k
tab_gain = pd.DataFrame({                                   # the advantage of UMAP over the other two methods, for each k
    "UMAP − PCA, real data": u_d - kept_data["PCA"][0], "UMAP − PCA, twins": u_g - kept_sur["PCA"].mean(0),
    "UMAP − Isomap, real data": u_d - kept_data["Isomap"][0], "UMAP − Isomap, twins": u_g - kept_sur["Isomap"].mean(0),
    "UMAP / PCA, real data": u_d / kept_data["PCA"][0], "UMAP / PCA, twins": u_g / kept_sur["PCA"].mean(0),
}, index=pd.Index(K_EVAL, name="k of the measure")).T
display(tab_gain.round(1))

COL_M = {"PCA": "C1", "Isomap": "C0", "UMAP": "C2"}         # one color per method (PCA and Isomap as in §5 and §6)
fig, ax = plt.subplots(1, 2, figsize=(13, 4.3))
for m in METHODS:
    ax[0].plot(K_EVAL, kept_data[m].mean(0), "o-", color=COL_M[m], lw=2, ms=5, label=f"{m}, real data")       # solid line: real data
    ax[0].plot(K_EVAL, kept_sur[m].mean(0), "s--", color=COL_M[m], lw=1.5, ms=4, label=f"{m}, twins")         # dashed: mean of the 5 twins
    ax[1].plot(K_EVAL, kept_data[m].mean(0) - kept_sur[m].mean(0), "o-", color=COL_M[m], lw=2, ms=5, label=m)  # how much more on the data
ax[0].plot(K_EVAL, chance_k, ":", color="0.4", lw=1.5, label="random map")
ax[1].axhline(0, color="0.3", lw=0.8)                       # above zero: more neighbors are preserved on the real data than on the twins
ax[0].set(xscale="log", xticks=K_EVAL, xticklabels=K_EVAL, xlabel="k of the measure (how many neighbors are counted)",
          ylabel="preserved neighbors (%)", title=f"Neighbors preserved in the {D_EVAL}-dimensional maps")
ax[1].set(xscale="log", xticks=K_EVAL, xticklabels=K_EVAL, xlabel="k of the measure (how many neighbors are counted)",
          ylabel="data − twins (percentage points)", title="How much more on the real data than on the twins")
ax[0].minorticks_off(); ax[1].minorticks_off()              # only ticks at the K_EVAL values
ax[0].legend(fontsize=8, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.17))  # legend below the plot, so as not to cover the curves
ax[1].legend(fontsize=8)
fig.tight_layout(); fig.savefig(FIGS / "03_kept_vs_k.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# **Reading.** With k = 10 the numbers of the previous table are recovered.
#
# - **The advantage of UMAP is largest at the smallest scale and disappears at the largest.** On the real
#   data, with k = 5 UMAP preserves 30.7 points more than PCA and 29.0 more than Isomap. The advantage drops
#   at each step (with k = 50: 11.7 and 10.3) and with k = 200 it is almost null: +0.5 over PCA, +1.8 over
#   Isomap. On the twins it is lost earlier: from k = 100 UMAP preserves fewer neighbors than PCA (−0.5, then
#   −5.2 with k = 200). UMAP keeps 36-40% of the neighbors from k = 5 to k = 100, whereas PCA and Isomap
#   rise with k (from 9-11% to 43-44%): it is a map that preserves small neighborhoods, as its
#   `n_neighbors` = 10 says, not the overall arrangement.
# - **The difference between data and twins is largest at an intermediate scale, for all three
#   methods.** UMAP: +9.3 points with k = 50 (+8.1 with k = 20, +7.8 with k = 100). Isomap: +5.6 with
#   k = 50 (+5.4 with k = 20). PCA: +3.1 with k = 20. At the smallest scale it is smaller (k = 5: UMAP
#   +2.0, Isomap +3.3, PCA +1.6) and at the largest it drops again (k = 200: UMAP +5.4, Isomap
#   +2.8, PCA −0.3, i.e. nothing).
# - **How sharp these numbers are.** For UMAP the profile is sharp: with k = 5 the twins
#   preserve 36.6-38.6% of the neighbors versus 39.6% for the data, with k = 50 26.6-27.5% versus
#   36.3%. For Isomap and PCA the shape is the same, but between the maximum and the extremes there are only
#   1.5-3.5 points, little more than the spread between one twin and another (usually 1-2 points). In absolute terms
#   the differences remain small, at most 9 points out of 100. And the usual limits apply: the
#   twins are Gaussian (§5.2) and for each there is a single UMAP seed.
# - **Why so little at the smallest scale?** With k = 5 about a third of the neighbors are time chain
#   (§7.1), and the twins have the same chain by construction. This is consistent with a small difference, but we did not
#   verify it by removing the chain (for example with the Theiler window
#   of §7.2).
#
# **The "local structure, not global" hypothesis of §7.2 is only partly confirmed, and needs to be
# qualified.**
#
# - **"Not global" holds.** At the largest scale the extra structure relative to the twins shrinks for
#   all methods, and for PCA it disappears.
# - **"Local" needs qualification.** The extra structure is not in the very nearest neighbors, but in neighborhoods of
#   a few tens of points: 20-50 out of 1106, i.e. 2-5%. To give an idea, a phase in a trial
#   occupies on average about 9 points: it is the scale of a few repetitions of the same phase, not that
#   of adjacent instants. This correspondence is our reading, not a measurement.
# - **One piece remains open.** The hypothesis arose from the comparison between the 25 PCs and the GPFA factors
#   (§7.2): a cleaner input improves the arrangement of the phases. Here we measured only the
#   25 PCs, so we do not know whether cleaning the input increases the structure at exactly this scale.

# %% [markdown]
# **The `Shape` islands.** In the map of the real data `Shape` (red) does not form one group but two;
# in the map of the GPFA factors the pieces are three. They are not single trials. Let us look at the kinematics, on the
# 25 PCs.

# %%
from sklearn.cluster import KMeans                          # to split the points of a phase into two groups

G3 = io.KIN_LABELS.index("g3")                              # one of the grasp degrees of freedom
odd = trial % 2 == 1                                        # true for the bins of odd trials
tab_g3 = pd.DataFrame({p: {"odd trials": np.nanmean(s.kin[(phase == p) & odd, G3]),   # mean value of g3 in each phase
                           "even trials": np.nanmean(s.kin[(phase == p) & ~odd, G3])} for p in io.MOVEMENT_PHASES})
display(tab_g3.round(2))
ends = np.array([s.kin[np.flatnonzero((phase == "Shape") & (trial == t))[[0, -1]], G3] for t in np.unique(trial)])  # g3 at the first and last bin of Shape
is_odd_t = np.unique(trial) % 2 == 1
print(f"g3 from the start to the end of Shape: odd trials from {ends[is_odd_t, 0].mean():.2f} to {ends[is_odd_t, 1].mean():.2f}, "
      f"even trials from {ends[~is_odd_t, 0].mean():.2f} to {ends[~is_odd_t, 1].mean():.2f}")

def split2(Y, p):                                           # splits the points of phase p in a map into 2 groups
    return KMeans(2, n_init=10, random_state=0).fit_predict(Y[ph == p]).astype(bool)

def match(lab, truth):                                      # agreement (%) between two 2-group splits, up to swapping the names
    return max((lab == truth).mean(), (lab != truth).mean()) * 100

def null_match(p, n=1000):                                  # the same agreement if "odd/even" were assigned at random to the trials (phase p)
    rng_p, tr_p = np.random.default_rng(0), tr[ph == p]     # fixed seed for each phase; the trial of each point of the phase
    t_p, lab_pca = np.unique(tr_p), split2(Y_pca[:, :2], p) # the trials present; the 2 groups of the 2-dimensional PCA
    out = []
    for _ in range(n):
        fake = dict(zip(t_p, rng_p.permutation(t_p % 2 == 1)))   # odd/even shuffled among the trials
        out.append(match(lab_pca, np.array([fake[t] for t in tr_p])))
    return np.array(out)

PH_ISL = ("Shape", "Grasp", "Release")                      # the hand phases: those that separate from the rest in the maps
Yu = {d: [umap_fit(X, d, sd) for sd in SEEDS] for d in (2, 3)}  # UMAP: 5 maps per number of dimensions, fitted only once
MAPS_ISL = {**{f"PCA, d = {d}": [Y_pca[:, :d]] for d in (2, 3)},   # the same maps for all three phases
            **{f"Isomap, d = {d}": [iso.embedding_[:, :d]] for d in (2, 3)},
            **{f"UMAP, d = {d} (mean of the 5 seeds)": Yu[d] for d in (2, 3)}}
rows_isl = {}
for p in PH_ISL:
    odd_p = tr[ph == p] % 2 == 1                            # for each point of the phase: does it come from an odd trial?
    rows_isl[p] = {name: np.mean([match(split2(Y_, p), odd_p) for Y_ in Ys]) for name, Ys in MAPS_ISL.items()}
    nul = null_match(p)
    rows_isl[p].update({"chance: mean": nul.mean(), "chance: 95th percentile": np.percentile(nul, 95)})
tab_isl = pd.DataFrame(rows_isl).T
display(tab_isl.round(1))
print("Points and trials per phase:", {p: (int((ph == p).sum()), len(np.unique(tr[ph == p]))) for p in PH_ISL})

# %% [markdown]
# The grasp degree of freedom g3 differs between odd and even trials in **all** phases: it is
# 0.9 versus 0.1 before `Shape`, 0.04 versus 0.96 in `Grasp`. During `Shape` it moves in opposite
# directions: in odd trials it falls from 0.90 to 0.18, in even ones it rises from 0.10 to 0.82. The task **alternates
# two versions** of the movement.
#
# If we split the points of `Shape` in the map into two groups, with UMAP and with Isomap the two groups
# coincide almost perfectly with odd and even trials (98-100%); with the 2-dimensional PCA in
# 79%. The term of comparison is not 50%: points of the same trial stay together, so
# the 15 trials count and not the 92 points. Randomly shuffling odd and even among the trials, the agreement is on
# average 59% and 95 times out of 100 it stays below 72%.

# %% [markdown]
# **Reading.** The two islands coincide with odd and even trials **only for `Shape`**.
#
# - **`Shape`**: 98-100% with Isomap and UMAP, 79-88% with PCA, versus a chance 95th percentile of
#   71.7%.
# - **`Grasp`**: 52-67% with all methods, never above the chance 95th percentile (69.0%).
# - **`Release`**: 52-53%, below even the chance mean (54.7%).
#
# In `Grasp` and `Release`, therefore, the split of the map into two groups does not follow the two versions of the
# task, even though g3 is very different between odd and even trials (0.04 versus 0.96 in `Grasp`, 0.16
# versus 0.85 in `Release`, table above). Chance is computed, as for `Shape`, on the two groups
# of the 2-dimensional PCA: for the other methods it is an approximate reference.

# %% [markdown]
# Two consequences:
#
# - The two `Shape` islands correspond to a real difference in the task. For `Grasp` and `Release`
#   they do not: there the split of the map into two groups does not follow the two versions (52-67%, within chance).
# - The 15 repetitions of a phase are **not replicas** of the same movement: at least g3 changes from
#   one trial to another, in all phases. This is why the means and the consistency (02_new; here §4,
#   §6.1, §7.2 and §9.4) use 16 conditions, phase × version, instead of the 8 phases. Among the hand
#   phases, the split of the map follows the two versions only in `Shape`; for the arm phases we did not
#   check.
#
# ### 9.3 · How much does chance matter? Seeds and parameters
#
# We repeat the measure of §9.1 with d = 3 changing one parameter at a time, always with 5 seeds:
# `n_neighbors` = 5, 10, 20, 50 and `min_dist` = 0, 0.1, 0.5.

# %%
NN_LIST = (5, 10, 20, 50)                                   # n_neighbors: how large the "neighborhood" of each point is
MD_LIST = (0.0, 0.1, 0.5)                                   # min_dist: how much the points may pile up in the map

sens = {}                                                   # (input, parameter, value) -> the 5 accuracies, one per seed
for name in INPUTS_U:
    for nn in NN_LIST:
        sens[(name, "n_neighbors", nn)] = umap_seeds(name, d=D_EVAL, n_neighbors=nn)
    for md in MD_LIST:
        sens[(name, "min_dist", md)] = umap_seeds(name, d=D_EVAL, min_dist=md)
tab_sens = pd.DataFrame({(n_, p_, str(v_)): {"mean": a_.mean(), "minimum": a_.min(), "max": a_.max()}
                         for (n_, p_, v_), a_ in sens.items()}).T
display(tab_sens.round(1))

fig, ax = plt.subplots(1, 2, figsize=(12.5, 4), sharey=True)
for a_, (par, vals) in zip(ax, (("n_neighbors", NN_LIST), ("min_dist", MD_LIST))):
    for name, c_ in zip(INPUTS_U, ("C0", "C2")):
        A = np.array([sens[(name, par, v)] for v in vals])  # (parameter values, 5 seeds)
        a_.errorbar(range(len(vals)), A.mean(1), yerr=[A.mean(1) - A.min(1), A.max(1) - A.mean(1)], fmt="o-",
                    color=c_, lw=2, capsize=3, label=f"UMAP, {name}")
        a_.axhline(float(tab_u.loc["no map, 25 dimensions (the ceiling)", name]), color=c_, ls="--", lw=1)
        a_.axhline(float(tab_u.loc[f"Isomap, d = {D_EVAL}", name]), color=c_, ls=":", lw=1)
    a_.set(xticks=range(len(vals)), xticklabels=vals, xlabel=par, title=f"Varying {par}")
ax[0].set_ylabel("balanced accuracy (%)")
ax[0].plot([], [], color="0.4", ls="--", lw=1, label="no map, 25 dimensions (the ceiling)")
ax[0].plot([], [], color="0.4", ls=":", lw=1, label=f"Isomap, d = {D_EVAL}")
ax[0].legend(fontsize=8, loc="lower left"); ax[0].set_ylim(25, 55)
fig.suptitle(f"UMAP with d = {D_EVAL}: phase decoding (dot = mean of the 5 seeds, bar = minimum-maximum)", y=1.0)
fig.tight_layout(); fig.savefig(FIGS / "03_umap_sensitivity.png", bbox_inches="tight"); plt.show()

# %% [markdown]
# **Reading.**
#
# - **With `n_neighbors` from 5 to 20 the result does not change**: 42.3-42.6% on the 25 PCs, 49.7-50.9% on the
#   GPFA factors, always at the ceiling level.
# - **With 50 neighbors it drops** (38.6% and 47.6%): with a wide neighborhood UMAP looks less locally and
#   loses part of the information. It still remains above Isomap with d = 3 (36.0% and 40.7%).
# - **`min_dist` does not matter** for this measure (41.9-42.7% and 49.9-50.6%): it changes the look of the
#   map, not who is close to whom.
# - **Across seeds** there are at most 3 points of difference (usually 1-2).
#
# In §7.1 the advantage of Isomap depended on k and was small. That of UMAP in decoding the
# phases with 2-3 coordinates is **large and stable**, as long as the neighborhood stays small.
#
# ### 9.4 · The single-trial trajectories
#
# The same measure as §6.1: the 119 windows of 1 s projected into the 3-dimensional map with
# `transform`, then roughness and consistency.

# %%
u3 = UMAP(n_neighbors=K_NN, n_components=3, random_state=0, n_jobs=1).fit(X)  # 3-dimensional UMAP on the same points as §4
traj["UMAP"] = np.split(u3.transform(np.concatenate(win_X)), len(win_X))      # the 119 windows of 1 s, projected with transform
tab_traj_u = pd.DataFrame([{"method": m, "roughness (lower = better)": roughness([t_[:, :3] for t_ in traj[m]]),
                            "consistency (higher = better)": consistency([t_[:, :3] for t_ in traj[m]])}
                           for m in ("PCA", "Isomap", "UMAP")]).set_index("method")
display(tab_traj_u.round(4))

# %% [markdown]
# **Reading.**
#
# - **Even higher roughness**: 0.59 versus 0.27 for Isomap and 0.013 for PCA. UMAP's
#   `transform` also works through neighbors, and the point jumps when the neighbors change.
# - **Higher consistency** (0.144 versus 0.122 and 0.109, with a single seed): in the UMAP coordinates
#   the repetitions of the same condition (phase × version, §9.2) have a slightly more
#   similar course in time.
# - The limit of §6.1 applies: UMAP gives a map, not a model of the data. For single-trial
#   trajectories the reference remains the GPFA: similar consistency (0.137), but roughness about 50
#   times lower (0.011 versus 0.59).
#
# ### 9.5 · UMAP summary
#
# | what | values tried | effect | where |
# |---|---|---|---|
# | phase decoding, d = 3 | UMAP, Isomap, PCA, no map; 25 PCs and GPFA factors | UMAP 42.6% and 50.5%, at the ceiling level (42.2% and 50.8%); Isomap 36.0% and 40.7%; PCA 32.4% and 30.5% | §9.1 |
# | ceiling at equal points | 25 PCs on the 872 points of the GPFA factors | 43.0%, versus 50.8% for the GPFA factors | §9.1 |
# | UMAP input | raw counts, smoothed without PCA, 25 PCs, GPFA factors | 17.0%, 40.2%, 42.6%, 50.5%, always at the level of the no-map reference | §9.1 |
# | twins without curvature, maps | 5 twins of the 25 PCs and 1 of the GPFA factors, one seed | no island in any of the 6 | §9.2 |
# | twins without curvature, preserved neighbors | 5 twins; 5 seeds on the data | UMAP about 3 times PCA on both data and twins; UMAP − PCA with d = 3: 26.6 points on the data, 22.8 on the twins | §9.2 |
# | neighbors preserved as the scale varies | k = 5-200 (k of the measure; that of the method stays 10) | UMAP − PCA on the data: +30.7 points with k = 5, +0.5 with k = 200; data − twins largest at an intermediate scale (k = 20-50: UMAP +9.3, Isomap +5.6, PCA +3.1), smaller with k = 5 and with k = 200 | §9.2 |
# | the two `Shape` islands | split into 2 groups vs odd and even trials | they coincide: UMAP and Isomap 98-100%, PCA 79-88% (chance 59%) | §9.2 |
# | seeds | 5 | 1-3 points between the best and the worst | §9.1, §9.3 |
# | `n_neighbors` | 5, 10, 20, 50 | stable up to 20, drops at 50 (38.6% and 47.6%) | §9.3 |
# | `min_dist` | 0, 0.1, 0.5 | no effect on the measure | §9.3 |
# | single-trial trajectories | PCA, Isomap, UMAP | roughness 0.013, 0.27, 0.59; per-condition consistency 0.109, 0.122, 0.144 | §9.4 |
#
# **What we learned about UMAP**
#
# 1. **Among the three methods tried, UMAP is the one that best preserves the phase information in 2-3
#    coordinates.** It decodes the phases as well as the classifier on the 25 dimensions, much better than
#    Isomap and PCA, and the result changes little with the seed, with `min_dist` and with `n_neighbors`
#    between 5 and 20 (with 50 it drops by 3-4 points). It is a good map for seeing which phases separate,
#    not for reading distances, group sizes or trajectories.
# 2. **It does not add information.** It stays at the ceiling level, slightly above or slightly below depending on the
#    seed: what it shows was already in the neighbors.
# 3. **It is consistent with the hypothesis of §7.2 only in part, and does not prove it.** The phase
#    information lies in the neighborhoods in 25 dimensions (the ceiling is reached without any map) and UMAP, which
#    preserves the neighborhoods, brings it into 2-3 coordinates; Isomap and PCA lose part of it. But UMAP
#    preserves the neighborhoods almost as well on a flat twin: its success proves neither
#    that the structure is nonlinear nor that the manifold is curved. And as the scale varies (§9.2)
#    "local" needs qualification: the extra structure relative to the twins is largest in neighborhoods of 20-50 points
#    (UMAP +9.3 points with k = 50, +2.0 with k = 5), not among the very nearest neighbors, and it shrinks at the largest
#    scale. We measured this only on the 25 PCs.
# 4. **The islands do not arise from the method applied to a Gaussian cloud**: they do not appear
#    on the twins. For `Shape` they correspond to a real difference in the task (the two versions, odd
#    and even trials); for `Grasp` and `Release` they do not: their split into two groups does not follow the two
#    versions (52-67%, within chance).
# 5. **The repetitions of a phase are not replicas**: the task alternates two versions between odd
#    and even trials. This is why means and consistency, here and in 02_new, use 16 conditions (phase × version).
# 6. **Limits**: distances in the map are not interpretable, there is no dimension estimate,
#    the trajectories are the roughest of all.
#
# > **Rule for the thesis.** UMAP is used to **show**, not to **measure** the shape of the manifold.
# > A result must be reported with its ceiling and with the twin, for example: *"UMAP (n_neighbors =
# > 10, min_dist = 0.1, 5 seeds) on 25 PCs of the smoothed firing rates (σ = 50 ms), one point every 100 ms
# > in the movement phases: balanced accuracy in decoding the phase with 3 coordinates 42.6%
# > (41.9-43.2 across seeds), versus 42.2% for the classifier on the 25 dimensions and 32.4% for PCA with
# > 3 components; on 5 random-phase surrogates the map shows no groups"*.

# %% [markdown]
# ## References
#
# - Tenenbaum, J. B., de Silva, V., Langford, J. C. (2000). A global geometric framework for
#   nonlinear dimensionality reduction. *Science*, 290(5500), 2319–2323.
# - Jazayeri, M., Ostojic, S. (2021). Interpreting neural computations by examining intrinsic and
#   embedding dimensionality of neural activity. *Current Opinion in Neurobiology*, 70, 113–120.
# - Theiler, J. (1986). Spurious dimension from correlation algorithms applied to limited
#   time-series data. *Physical Review A*, 34(3), 2427–2432.
# - Theiler, J., Eubank, S., Longtin, A., Galdrikian, B., Farmer, J. D. (1992). Testing for
#   nonlinearity in time series: the method of surrogate data. *Physica D*, 58, 77–94.
# - Prichard, D., Theiler, J. (1994). Generating surrogate data for time series with several
#   simultaneously measured variables. *Physical Review Letters*, 73(7), 951–954.
# - Schreiber, T., Schmitz, A. (1996). Improved surrogate data for nonlinearity tests. *Physical
#   Review Letters*, 77(4), 635–638.
# - Facco, E., d'Errico, M., Rodriguez, A., Laio, A. (2017). Estimating the intrinsic dimension of
#   datasets by a minimal neighborhood information. *Scientific Reports*, 7, 12140.
# - Yu, B. M., et al. (2009). Gaussian-process factor analysis for low-dimensional single-trial
#   analysis of neural population activity. *Journal of Neurophysiology*, 102, 614–635.
# - McInnes, L., Healy, J., Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection
#   for dimension reduction. *arXiv*:1802.03426.
# - Chari, T., Pachter, L. (2023). The specious art of single-cell genomics. *PLoS Computational
#   Biology*, 19(8), e1011288.
