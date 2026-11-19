"""Step 1 (literature comparison), first run: the two check numbers on S1, recomputed and logged.

Data: X_counts (counts at 20 ms, FALCON bins, 160 active channels >= 0.5 Hz) from
~/thesis_h1_work/rep/19250108T110520/02_rep.pkl, i.e. the replication with FALCON bins.

- PR of the single-trial PCA: Gaussian smoothing sigma = 50 ms (mode="nearest"), per-channel z-score,
  all bins of the session, 160 channels. Expected 106.1 (notebook 02, § 6).
- FA dimensions for 95% of the shared variance: z-scored raw counts, K = 15,
  svd_method="lapack", random_state=0; eigenvalues of W W^T. Expected 14 (notebook 02, § 9.3).

If a value does not match, the script exits with an error AFTER logging the numbers obtained.
Uso: PYTHONPATH=~/thesis_h1_git/src python scripts/literature_check.py
"""
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.decomposition import FactorAnalysis
from sklearn.preprocessing import StandardScaler

from thesis_h1 import results

SID, SESSION = "19250108T110520", "S1_set_1"
PKL = Path.home() / "thesis_h1_work/rep" / SID / "02_rep.pkl"
BIN_S, SIGMA_MS, K_FA = 0.02, 50, 15
EXPECTED = {"PR": 106.1, "dim95": 14}
KEYS = dict(experiment="literature", decoder=None, session=SESSION, bins="falcon", split="session",
            condition="check", fold=None, k_shot=None, H_ms=None, seed=None)
results.check_keys(method="PCA", latent_dim=None, **KEYS)

X = pickle.load(open(PKL, "rb"))["arrays"]["X_counts"].astype(float)       # 12,233 bins x 160 channels
print(f"{SID}: X_counts {X.shape}")

# PCA PR: sigma = 50 ms, z-score
Z = StandardScaler().fit_transform(gaussian_filter1d(X, SIGMA_MS / 1000 / BIN_S, axis=0, mode="nearest") / BIN_S)
lam = np.linalg.eigvalsh(np.cov(Z, rowvar=False))[::-1]
PR = lam.sum() ** 2 / (lam ** 2).sum()

# FA, K = 15, z-scored raw counts: dimensions for 95% of the shared part
fa = FactorAnalysis(n_components=K_FA, svd_method="lapack", random_state=0).fit(StandardScaler().fit_transform(X))
W = fa.components_.T
eig = np.linalg.eigvalsh(W @ W.T)[::-1][:K_FA]
dim95 = int(np.searchsorted(np.cumsum(eig) / eig.sum(), 0.95) + 1)

print(f"PR = {PR:.4f} (expected {EXPECTED['PR']}); FA dimensions for 95% = {dim95} (expected {EXPECTED['dim95']})")
results.log_run({"PR": PR, "n_channels": X.shape[1], "n_bin": X.shape[0]},
                method="PCA", latent_dim=None, **KEYS)
results.log_run({"dim95_shared": dim95, "n_channels": X.shape[1], "n_bin": X.shape[0], "n_iter": fa.n_iter_},
                method="FA", latent_dim=K_FA, **KEYS)
ok = round(PR, 1) == EXPECTED["PR"] and dim95 == EXPECTED["dim95"]
print("check:", "OK" if ok else "DOES NOT MATCH")
sys.exit(0 if ok else 1)
