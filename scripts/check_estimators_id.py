"""Step 1 (R9): are the repo's intrinsic dimension estimators correct?

Generates data of known dimension, without noise, embedded in 160 dimensions, and compares
the repo functions with three independent references:
  - DADApy (Laio's group, authors of TwoNN): Data.compute_id_2NN
  - scikit-dimension: skdim.id.TwoNN e skdim.id.MLE
  - reference implementations in this file (Facco 2017; Levina & Bickel 2005)

Test 1 · Cases of known dimension (independent points).
Test 2 · Theiler window, on time-ordered data smoothed like H1 (sigma 2.5 bins):
         - 3 slow signals: the window does not matter, the estimate stays close to 3;
         - 160 channels of slow noise: without the window the 10 neighbors are consecutive
           time points, the MLE measures the trajectory and drops below 2; with the window
           it rises to values unrelated to 160. Here only repo against reference is
           compared: the "true" value cannot be estimated.

Usage (from the repo, with the environment active):
    PYTHONPATH=$HOME/thesis_h1_git/src python scripts/check_estimators_id.py --out <cartella>

Before running it, link the repo functions in the ADAPTER below.
If a function is not linked, the script prints only the references.

External libraries: optional. The reference implementations in this file were
tested on 5 Nov against DADApy 0.3.1 and scikit-dimension 0.5.0 and agree to the
second decimal (collaudo_riferimenti.md). Do not install DADApy in the thesis
environment: it requires numpy < 2 and would downgrade numpy.
"""
import argparse
import csv
import os
import sys
import time

import numpy as np
from sklearn.neighbors import NearestNeighbors

# =====================================================================
# ADAPTER: link the repo functions here (phase 2, R9).
# Each function receives X with shape (n_points, n_dimensions) and returns a float.
# Leave "raise NotImplementedError" for the functions the repo does not have.
# =====================================================================


# Linked to src/thesis_h1/literature2.py: estimate_id (R9) and estimates_theiler (R9b).
from thesis_h1 import literature2  # noqa: E402

_cache_theiler = {}


def _theiler(X, w):
    # estimates_theiler computes TwoNN, MLE10, MLE20 and CorrInt together: one call per (X, w), reused by TwoNN and MLE.
    # A single group (a single "trial"), time = row index.
    key = (id(X), w)
    if key not in _cache_theiler:
        _cache_theiler[key] = literature2.estimates_theiler(X, np.zeros(len(X)), np.arange(len(X)), [w])
    return _cache_theiler[key]


def twonn_repo(X):
    """Repo TwoNN, without Theiler window."""
    return literature2.estimate_id(X, "TwoNN")


def mle_repo(X, k=10):
    """Repo MLE (Levina & Bickel) with k neighbors, without Theiler window."""
    if k not in (10, 20):
        raise NotImplementedError  # the repo only has MLE10 and MLE20
    return literature2.estimate_id(X, f"MLE{k}")


def twonn_repo_theiler(X, w):
    """Repo TwoNN with a Theiler window of w bins (X in time order)."""
    return _theiler(X, w)[(w, "TwoNN")]


def mle_repo_theiler(X, w, k=10):
    """Repo MLE with a Theiler window of w bins (X in time order)."""
    if k not in (10, 20):
        raise NotImplementedError
    return _theiler(X, w)[(w, f"MLE{k}")]


# =====================================================================
# References
# =====================================================================

D_AMBIENTE = 160


def _neighbors(X, n_neighbors):
    nn = NearestNeighbors(n_neighbors=n_neighbors + 1, algorithm="brute").fit(X)
    dist, idx = nn.kneighbors(X)
    return dist[:, 1:], idx[:, 1:]  # drop the point itself


def _neighbors_theiler(X, n_neighbors, w):
    """First n_neighbors at temporal distance > w (row indices = time)."""
    if w <= 0:
        return _neighbors(X, n_neighbors)
    q = n_neighbors + 2 * w + 1
    nn = NearestNeighbors(n_neighbors=q + 1, algorithm="brute").fit(X)
    dist, idx = nn.kneighbors(X)
    rows = np.arange(len(X))[:, None]
    out = np.abs(idx - rows) > w
    out_d = np.empty((len(X), n_neighbors))
    for i in range(len(X)):
        out_d[i] = dist[i, out[i]][:n_neighbors]
    return out_d, None


def twonn_facco(dist, deviation=0.1):
    """Facco et al. 2017: line through the origin on -log(1-F) against log(mu), discarding the highest 10%."""
    r1, r2 = dist[:, 0], dist[:, 1]
    ok = r1 > 0
    mu = np.sort(r2[ok] / r1[ok])
    n = len(mu)
    F = np.arange(1, n + 1) / n
    keep = int(np.floor(n * (1 - deviation)))
    x, y = np.log(mu[:keep]), -np.log(1 - F[:keep])
    return float(np.dot(x, y) / np.dot(x, x))


def twonn_max_likelihood(dist):
    """Variant: maximum likelihood estimate of d for f(mu) = d mu^(-d-1)."""
    r1, r2 = dist[:, 0], dist[:, 1]
    ok = r1 > 0
    return float((ok.sum() - 1) / np.log(r2[ok] / r1[ok]).sum())


def mle_levina_bickel(dist, k=10):
    """Levina & Bickel 2005: mean of the local estimates, normalization k-1."""
    T = dist[:, :k]
    ok = T[:, 0] > 0
    T = T[ok]
    s = np.log(T[:, k - 1:k] / T[:, : k - 1]).sum(axis=1)
    return float(np.mean((k - 1) / s))


def mle_mackay_ghahramani(dist, k=10):
    """MacKay & Ghahramani 2005: inverse of the mean of the inverses of the local estimates."""
    T = dist[:, :k]
    ok = T[:, 0] > 0
    T = T[ok]
    s = np.log(T[:, k - 1:k] / T[:, : k - 1]).sum(axis=1)
    return float(1.0 / np.mean(s / (k - 1)))


def dadapy_twonn(X):
    from dadapy import Data

    d = Data(np.ascontiguousarray(X, dtype=float), verbose=False)
    out = d.compute_id_2NN()
    return float(out[0])


def skdim_twonn(X):
    import skdim

    return float(skdim.id.TwoNN().fit(X).dimension_)


def skdim_mle(X, k=10):
    import skdim

    return float(skdim.id.MLE().fit(X, n_neighbors=k).dimension_)


# =====================================================================
# Data of known dimension
# =====================================================================


def embed(Z, rng, D=D_AMBIENTE):
    """Random rotation: from R^m to R^D without changing distances."""
    Q, _ = np.linalg.qr(rng.standard_normal((D, Z.shape[1])))
    return Z @ Q.T


def sphere(n, d, rng):
    """Uniform points on the sphere S^d (dimension d, embedded in R^(d+1))."""
    Z = rng.standard_normal((n, d + 1))
    return Z / np.linalg.norm(Z, axis=1, keepdims=True)


def cases_known(n, rng):
    t = rng.uniform(0, 3 * np.pi, n)
    swiss = np.c_[t * np.cos(t), rng.uniform(0, 10, n), t * np.sin(t)]
    return [
        ("circle", 1, sphere(n, 1, rng)),
        ("plane (square)", 2, rng.uniform(0, 1, (n, 2))),
        ("swiss roll (curved)", 2, swiss),
        ("sphere S^3", 3, sphere(n, 3, rng)),
        ("hypercube 5D", 5, rng.uniform(0, 1, (n, 5))),
        ("sphere S^6 (Altan's d)", 6, sphere(n, 6, rng)),
        ("hypercube 10D", 10, rng.uniform(0, 1, (n, 10))),
    ]


def trajectory_slow(T, d, sigma_bin, rng):
    """d independent slow signals (smoothed Gaussian noise), in time order."""
    from scipy.ndimage import gaussian_filter1d

    Z = gaussian_filter1d(rng.standard_normal((T, d)), sigma_bin, axis=0, mode="wrap")
    return Z / Z.std(axis=0)


# =====================================================================
# Execution
# =====================================================================


_warned = set()


def trial(f, *args, **kw):
    try:
        return f(*args, **kw)
    except NotImplementedError:
        return None
    except ImportError as e:  # reference library not installed: warn only once
        if e.name not in _warned:
            _warned.add(e.name)
            print(f"  (library {e.name} not installed: empty column)", file=sys.stderr)
        return None


def fmt(v):
    return "—" if v is None else f"{v:.2f}"


def deviation_rel(a, b):
    if a is None or b is None:
        return None
    return abs(a - b) / max(abs(b), 1e-9)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=4000, help="points per case (test 1)")
    ap.add_argument("--k", type=int, default=10, help="neighbors for MLE")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=".", help="folder for check_id.csv and .md")
    ap.add_argument("--tol", type=float, default=0.05, help="allowed relative deviation from the references")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.seed)
    t0 = time.time()

    rows = []
    print(f"== Test 1 · cases of known dimension (n = {a.n}, D = {D_AMBIENTE}, k = {a.k}, seed {a.seed})")
    print(f"{'chance':26s} {'d':>3s} | {'Facco':>6s} {'DADApy':>6s} {'skdim':>6s} {'repo':>6s} | "
          f"{'LB':>6s} {'MG':>6s} {'skdim':>6s} {'repo':>6s} | outcome")
    for name, d_true, Z in cases_known(a.n, rng):
        X = embed(Z, rng)
        dist, _ = _neighbors(X, max(a.k, 2))
        r = dict(
            test="1", chance=name, d_true=d_true, w="",
            twonn_facco=twonn_facco(dist),
            twonn_mv=twonn_max_likelihood(dist),
            twonn_dadapy=trial(dadapy_twonn, X),
            twonn_skdim=trial(skdim_twonn, X),
            twonn_repo=trial(twonn_repo, X),
            mle_lb=mle_levina_bickel(dist, a.k),
            mle_mg=mle_mackay_ghahramani(dist, a.k),
            mle_skdim=trial(skdim_mle, X, a.k),
            mle_repo=trial(mle_repo, X, k=a.k),
        )
        outcomes = []
        if r["twonn_repo"] is not None:
            ref = [v for v in (r["twonn_facco"], r["twonn_dadapy"], r["twonn_skdim"], r["twonn_mv"]) if v is not None]
            outcomes.append("TwoNN ok" if min(deviation_rel(r["twonn_repo"], v) for v in ref) <= a.tol else "TwoNN DIFFERENT")
        if r["mle_repo"] is not None:
            ref = [v for v in (r["mle_lb"], r["mle_mg"], r["mle_skdim"]) if v is not None]
            outcomes.append("MLE ok" if min(deviation_rel(r["mle_repo"], v) for v in ref) <= a.tol else "MLE DIFFERENT")
        r["outcome"] = ", ".join(outcomes) if outcomes else "references only"
        rows.append(r)
        print(f"{name:26s} {d_true:3d} | {fmt(r['twonn_facco']):>6s} {fmt(r['twonn_dadapy']):>6s} "
              f"{fmt(r['twonn_skdim']):>6s} {fmt(r['twonn_repo']):>6s} | {fmt(r['mle_lb']):>6s} "
              f"{fmt(r['mle_mg']):>6s} {fmt(r['mle_skdim']):>6s} {fmt(r['mle_repo']):>6s} | {r['outcome']}")

    # 12,000 bins like an H1 session; sigma 2.5 bins = 50 ms with 20 ms bins
    T, sigma = 12000, 2.5
    scenarios = [
        ("3 slow signals", 3, embed(trajectory_slow(T, 3, sigma, rng), rng)),
        ("160 channels of slow noise", 160, trajectory_slow(T, D_AMBIENTE, sigma, rng)),
    ]
    print(f"\n== Test 2 · Theiler window (T = {T} bins in time order, sigma = {sigma} bins)")
    print(f"{'scenario':28s} {'w':>3s} | {'Facco':>6s} {'MV':>6s} {'repo':>6s} | "
          f"{'LB':>6s} {'MG':>6s} {'repo':>6s} | outcome")
    for name, d_true, X in scenarios:
        for w in (0, 1, 3, 5, 10, 20, 40):
            dist, _ = _neighbors_theiler(X, max(a.k, 2), w)
            r = dict(
                test="2", chance=name, d_true=d_true, w=w,
                twonn_facco=twonn_facco(dist), twonn_mv=twonn_max_likelihood(dist),
                twonn_dadapy=None, twonn_skdim=None,
                twonn_repo=trial(twonn_repo_theiler, X, w),
                mle_lb=mle_levina_bickel(dist, a.k), mle_mg=mle_mackay_ghahramani(dist, a.k),
                mle_skdim=None,
                mle_repo=trial(mle_repo_theiler, X, w, k=a.k),
            )
            outcomes = []
            if r["twonn_repo"] is not None:
                outcomes.append("TwoNN ok" if min(deviation_rel(r["twonn_repo"], r["twonn_facco"]),
                                               deviation_rel(r["twonn_repo"], r["twonn_mv"])) <= a.tol
                             else "TwoNN DIFFERENT")
            if r["mle_repo"] is not None:
                outcomes.append("MLE ok" if min(deviation_rel(r["mle_repo"], r["mle_lb"]),
                                             deviation_rel(r["mle_repo"], r["mle_mg"])) <= a.tol
                             else "MLE DIFFERENT")
            r["outcome"] = ", ".join(outcomes) if outcomes else "references only"
            rows.append(r)
            print(f"{name:28s} {w:3d} | {fmt(r['twonn_facco']):>6s} {fmt(r['twonn_mv']):>6s} "
                  f"{fmt(r['twonn_repo']):>6s} | {fmt(r['mle_lb']):>6s} {fmt(r['mle_mg']):>6s} "
                  f"{fmt(r['mle_repo']):>6s} | {r['outcome']}")

    fields = ["test", "chance", "d_true", "w", "twonn_facco", "twonn_mv", "twonn_dadapy", "twonn_skdim",
             "twonn_repo", "mle_lb", "mle_mg", "mle_skdim", "mle_repo", "outcome"]
    with open(os.path.join(a.out, "check_id.csv"), "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=fields)
        wr.writeheader()
        for r in rows:
            wr.writerow({c: ("" if r[c] is None else r[c]) for c in fields})

    different = [r for r in rows if "DIFFERENT" in r["outcome"]]
    linked = any(r["twonn_repo"] is not None or r["mle_repo"] is not None for r in rows)
    if not linked:
        verdict = "No repo function linked: only the references were printed."
    elif different:
        verdict = (f"The repo deviates from the references in {len(different)} of {len(rows)} rows "
                    f"(tolerance {a.tol:.0%}): check the implementation before interpreting R9.")
    else:
        verdict = (f"The repo agrees with at least one reference in all rows (tolerance {a.tol:.0%}): "
                    "the failure to replicate Altan does not depend on an error in the estimators.")
    with open(os.path.join(a.out, "check_id.md"), "w") as f:
        f.write("# Check of the intrinsic dimension estimators (step 1, R9)\n\n")
        f.write(f"Parameters: n = {a.n}, D = {D_AMBIENTE}, k = {a.k}, seed {a.seed}, tolerance {a.tol:.0%}.\n\n")
        f.write("| test | case | true d | w | TwoNN Facco | TwoNN DADApy | TwoNN skdim | TwoNN repo | "
                "MLE LB | MLE MG | MLE skdim | MLE repo | outcome |\n|" + "---|" * 13 + "\n")
        for r in rows:
            f.write(f"| {r['test']} | {r['chance']} | {r['d_true']} | {r['w']} | {fmt(r['twonn_facco'])} | "
                    f"{fmt(r['twonn_dadapy'])} | {fmt(r['twonn_skdim'])} | {fmt(r['twonn_repo'])} | "
                    f"{fmt(r['mle_lb'])} | {fmt(r['mle_mg'])} | {fmt(r['mle_skdim'])} | "
                    f"{fmt(r['mle_repo'])} | {r['outcome']} |\n")
        f.write(f"\n**Verdict.** {verdict}\n")
    print(f"\nVerdict: {verdict}")
    print(f"Files: {os.path.join(a.out, 'check_id.csv')} and .md ({time.time() - t0:.0f} s)")
    print("== END")


if __name__ == "__main__":
    main()
