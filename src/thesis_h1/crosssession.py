"""Few-shot cross-session protocol on FALCON H1: 14 held-out sessions, 3 calibration trials each.

Scheme (see docs/crosssession_protocol.md for the choices and the analysis plan):
- each method (PCA, FA, GPFA, Isomap, UMAP, CEBRA, LFADS) and its linear readout are fitted ONCE on the
  13 held-in sessions (calib), with the hyperparameters and the latent dimension already chosen in notebooks 02_new, 03,
  04; then they are frozen;
- on each held-out session: leave-one-trial-out rotation, 2 trials adapt and 1 tests -> 3 folds;
- conditions (cumulative): zero_shot, recenter, align, fewshot; plus scratch (method and readout on the 2 trials only,
  except LFADS) and the common reference ridge_rates (method "rates"); for FA also fa_native;
- every trial is a sequence of its own: smoothing, GPFA and LFADS pieces, CEBRA windows and decoder history
  start and end inside the trial. So nothing that is estimated in the adaptation (means, z-score,
  FA, alignment, readout) sees the data of the test trial, not even at the edges;
- λ (fewshot) and α (scratch, ridge_rates) are chosen with a pseudo-few-shot on the held-in sessions (method and readout
  fitted on days S0-S3, pseudo-sessions of 3 consecutive trials from the 4 files of S4 and S5), then they are frozen.

Metric: variance-weighted R² (= sklearn r2_score(multioutput="variance_weighted"), as the falcon_challenge
evaluator) on the bins with eval_mask; per fold and on the concatenated predictions of the 3 folds (results.log_concat).
"""
from __future__ import annotations

import contextlib
import hashlib
import io as _io
import json
import os
import pickle
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from . import io
from .decoding import lagged, r2_vw

WORK = Path.home() / "thesis_h1_work"
DATA = Path(os.environ.get("THESIS_DATA", Path.home() / "000954"))
XS = Path(os.environ.get("THESIS_XS", WORK / "xs"))      # cache of the data and models, pseudo-few-shot choices

BIN_S = io.BIN_S
SIGMA_BINS = 50 / 20                    # Gaussian smoothing σ = 50 ms (02_new, 03, 04)
MIN_HZ = 0.5                            # active channels: mean rate ≥ 0.5 Hz on the held-in sessions (02_new)
H = 50                                  # decoder history: 50 bins = 1000 ms (ridge_h50 of 05)
EXPERIMENT, SPLIT, DECODER, H_MS, BINS = "crosssession", "calib3_loto", "ridge", 1000, "falcon"
K_SHOT = 2                              # trials with kinematics used in the adaptation

PSEUDO_TRAIN_DAYS = ("S0", "S1", "S2", "S3")   # pseudo-few-shot: model on these days...
PSEUDO_TEST_DAYS = ("S4", "S5")                # ...pseudo-sessions of 3 trials from these
METHODS = ("PCA", "FA", "GPFA", "Isomap", "UMAP", "CEBRA", "LFADS")
SEEDS = {"PCA": [0], "FA": [0], "GPFA": [0], "Isomap": [0], "UMAP": [0, 1, 2], "CEBRA": [0, 1, 2], "LFADS": [0, 1, 2]}
CONDITIONS = ("zero_shot", "recenter", "align", "fewshot", "fewshot_gain", "scratch")
NO_ADAPT = ("zero_shot", "no_adapt")    # conditions without adaptation trials (k_shot = 0)
DIM_RULE = 25                           # d of the maps (Isomap, UMAP, CEBRA) from the second round (31/10): the rule of PCA and LFADS, i.e. the
                                        # d of GPFA chosen with cross-validation (02_new §10); before d = 3 (03, 04)
RIDGE_H50 = Path(__file__).resolve().parents[2] / "models" / "ridge_h50.npz"   # channel decoder of 05 (b797b3e)

ALPHA_SRC_GRID = 10.0 ** np.arange(-2, 8.01, 0.5)   # α of the held-in readout (leave-one-day-out choice on the held-in sessions)
LAM_GRID = 10.0 ** np.arange(-2, 15.01, 0.5)        # λ of fewshot (towards the held-in weights); up to 1e15 (second round, 31/10:
                                                    # the grid up to 1e9 had the optimum at the edge for FA and LFADS)
ALPHA_GRID = 10.0 ** np.arange(-2, 8.01, 0.5)       # α of scratch and ridge_rates
EPS_REL = 1e-8                                      # eps on the diagonal of the covariances, relative to the mean eigenvalue


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


@contextlib.contextmanager
def quiet():
    with open(os.devnull, "w") as dn, contextlib.redirect_stdout(dn):
        yield


# ============================================================================================
# Data: a session = list of trials; every trial is a sequence of its own
# ============================================================================================
@dataclass
class Trial:
    num: int                  # TrialNum in the file
    counts: np.ndarray        # bins × active channels (int16)
    vel: np.ndarray           # bins × 7, velocity (FALCON target)
    mask: np.ndarray          # bins, eval_mask
    mov: np.ndarray           # bins, true in the 8 movement phases
    windows: list             # 1 s windows at the start of the movement phases, (start, end) within the trial
    counts_all: np.ndarray = None   # bins × 176 channels (for the channel decoder, ridge_channels)
    phases: list = None             # movement phases present in the trial

    @property
    def version(self):
        """Task version: odd and even trials alternate two versions (g3 in Grasp with opposite sign, 03 §9.2;
        checked on all held-in and held-out trials on 31/10)."""
        return "odd" if self.num % 2 else "even"

    def __len__(self):
        return len(self.counts)


@dataclass
class Sess:
    key: str                  # FALCON key, e.g. "S6_set_1" (for the pseudo-sessions "S4_set_1#b0")
    day: str
    sid: str
    group: str                # held_in / held_out / pseudo
    trials: list = field(repr=False)


def _load_file(path, group):
    from falcon_challenge.config import FalconConfig, FalconTask
    s = io.load_session(path, bins="falcon")                       # FALCON bins (= load_nwb)
    key = FalconConfig(FalconTask.h1).hash_dataset(Path(path).stem)
    ph = io.phase_per_bin(s)
    mov = np.isin(ph, io.MOVEMENT_PHASES)
    wins = io.phase_windows(s, win_s=1.0, align="start")           # 02_new §10: 1 s windows, movement phases
    trials = []
    for a, b in io.trial_slices(s):
        tw = [(w["start"] - a, w["stop"] - a) for w in wins if w["start"] >= a and w["stop"] <= b]
        trials.append(Trial(int(s.trial_num[a]), s.spikes[a:b], s.vel[a:b].astype(np.float64),
                            s.eval_mask[a:b].astype(bool), mov[a:b], tw, s.spikes[a:b].copy(),
                            sorted(set(ph[a:b]) & set(io.MOVEMENT_PHASES))))
    return Sess(key, key.split("_")[0], s.session_id, group, trials)


def load_data(refresh=False):
    """All calib sessions (held-in and held-out) and the active channels on the held-in sessions. Cache in XS/data.pkl."""
    cache = XS / "data.pkl"
    if cache.exists() and not refresh:
        with open(cache, "rb") as f:
            out = pickle.load(f)
        if getattr(next(iter(out["sessions"].values())).trials[0], "counts_all", None) is not None:
            return out                                              # 31/10 cache without counts_all: rebuild
    sessions = {}
    for group, split in (("held_in", "held-in-calib"), ("held_out", "held-out-calib")):
        for p in sorted(DATA.glob(f"sub-HumanPitt-{split}/*.nwb")):
            s = _load_file(p, group)
            sessions[s.key] = s
    hi = [t for s in sessions.values() if s.group == "held_in" for t in s.trials]
    hz = np.sum([t.counts.sum(0) for t in hi], axis=0) / (sum(len(t) for t in hi) * BIN_S)
    active = np.flatnonzero(hz >= MIN_HZ)
    for s in sessions.values():
        for t in s.trials:
            t.counts = np.ascontiguousarray(t.counts[:, active])
    out = dict(sessions=sessions, active=active, rate_hz=hz)
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".tmp")
    with open(tmp, "wb") as f:
        pickle.dump(out, f)
    os.replace(tmp, cache)
    return out


def heldin(data, days=None):
    return [s for s in data["sessions"].values() if s.group == "held_in" and (days is None or s.day in days)]


def heldout(data):
    return [s for s in data["sessions"].values() if s.group == "held_out"]


def pseudo_sessions(data):
    """Pseudo-sessions of the days PSEUDO_TEST_DAYS: blocks of 3 consecutive trials (like the held-out ones)."""
    out = []
    for s in heldin(data, PSEUDO_TEST_DAYS):
        for j in range(len(s.trials) // 3):
            out.append(Sess(f"{s.key}#b{j}", s.day, s.sid, "pseudo", s.trials[3 * j:3 * j + 3]))
    return out


# ============================================================================================
# Input representations (per trial)
# ============================================================================================
def rep(trial, kind):
    x = trial.counts.astype(np.float64)
    if kind == "counts":
        return x
    if kind == "sqrt":
        return np.sqrt(x)
    if kind == "rates":                                             # smoothing within the trial
        from scipy.ndimage import gaussian_filter1d
        return gaussian_filter1d(x, SIGMA_BINS, axis=0, mode="nearest") / BIN_S
    raise ValueError(kind)


def zstats(X):
    mu, sd = X.mean(0), X.std(0)
    sd = np.where(sd > 1e-12, sd, 1.0)
    return mu, sd


# ============================================================================================
# Methods: fit(trials) and encode(list of arrays in the representation of the method) -> list of latents
# ============================================================================================
class Method:
    name, rep, dim, cacheable = "", "", 0, True

    def __init__(self, seed=0, dim=None):
        self.seed = seed
        self.dim = type(self).dim if dim is None else int(dim)

    def fit(self, trials):
        raise NotImplementedError

    def encode(self, Xs):
        raise NotImplementedError


class PCAm(Method):
    """PCA with 25 components on the smoothed (σ 50 ms) z-scored firing rates, fitted on the bins of the movement phases (03, 04 §1)."""
    name, rep, dim = "PCA", "rates", 25

    def fit(self, trials):
        from sklearn.decomposition import PCA
        Xs = [rep(t, "rates") for t in trials]
        X = np.concatenate(Xs)
        self.mu, self.sd = zstats(X)
        mov = np.concatenate([t.mov for t in trials])
        self.pca = PCA(self.dim, svd_solver="full").fit((X[mov] - self.mu) / self.sd)
        return self

    def encode(self, Xs):
        return [self.pca.transform((x - self.mu) / self.sd) for x in Xs]


def _points(P, trials, step, n_max):
    """Points for Isomap/UMAP: movement bins, one every `step` within each trial (03 §2); at most n_max drawn at random."""
    pts = np.concatenate([p[t.mov & (np.arange(len(t)) % step == 0)] for p, t in zip(P, trials)])
    if len(pts) > n_max:
        pts = pts[np.sort(np.random.default_rng(0).choice(len(pts), n_max, replace=False))]
    return pts


class IsomapM(Method):
    """Isomap (k = 10, d = 3) on the 25 PCs, points = movement bins one every 5 (03 §2, §6; 04 §1.2).
    On the held-in sessions the points would be ~20 000: for memory (geodesic distance matrix) 5000 are kept,
    drawn at random. Each bin is projected with transform."""
    name, rep, dim, cacheable = "Isomap", "rates", 3, False
    K, STEP, N_MAX = 10, 5, 5000

    def _embedder(self):
        from sklearn.manifold import Isomap
        return Isomap(n_neighbors=self.K, n_components=self.dim)

    def fit(self, trials):
        self.pc = PCAm().fit(trials)
        pts = _points(self.pc.encode([rep(t, "rates") for t in trials]), trials, self.STEP, self.N_MAX)
        self.n_points = len(pts)
        np.random.seed(self.seed)                                   # initial ARPACK vector
        self.emb = self._embedder().fit(pts)
        return self

    def encode(self, Xs):
        return [self.emb.transform(p) for p in self.pc.encode(Xs)]


class UMAPm(IsomapM):
    """UMAP (n_neighbors = 10, min_dist = 0.1, d = 3) on the same points as Isomap (03 §9.1; 04 §1.2)."""
    name, cacheable = "UMAP", True
    N_MAX = 5000

    def _embedder(self):
        from umap import UMAP
        return UMAP(n_neighbors=self.K, n_components=self.dim, min_dist=0.1, random_state=self.seed, n_jobs=1)


class FAm(Method):
    """FA with K* = 15 factors on the z-scored raw counts (02_new §9), latents = posterior mean."""
    name, rep, dim = "FA", "counts", 15

    def fit(self, trials):
        from sklearn.decomposition import FactorAnalysis
        X = np.concatenate([rep(t, "counts") for t in trials])
        self.mu, self.sd = zstats(X)
        self.fa = FactorAnalysis(self.dim, svd_method="lapack", random_state=0).fit((X - self.mu) / self.sd)
        return self

    def encode(self, Xs):
        return [self.fa.transform((x - self.mu) / self.sd) for x in Xs]


def _gp_seqs(arrs):
    return np.array([(a.shape[0], a.T) for a in arrs], dtype=[("T", int), ("y", "O")])


class GPFAm(Method):
    """GPFA with d = 25 on the square root of the counts (02_new §10): fitted on the 1 s windows at the start of the
    movement phases (500 iterations, initial τ 100 ms, elephant pieces of 20 bins); orthonormalized latents of each
    bin from the 2 s pieces every 1 s, each taken from the piece in which it is most central (04 §7b), within the trial."""
    name, rep, dim = "GPFA", "sqrt", 25
    ITERS, TAU0, PIECE, KEEP = 500, 100.0, 100, 50

    def fit(self, trials):
        from elephant.gpfa import gpfa_core
        import elephant.utils, logging
        elephant.utils.logger.setLevel(logging.ERROR)
        segs = [np.sqrt(t.counts[a:b].astype(np.float64)) for t in trials for a, b in t.windows]
        self.n_windows = len(segs)
        np.random.seed(0)                                           # fixed starting point (02_new)
        with quiet():
            self.params, _ = gpfa_core.fit(_gp_seqs(segs), x_dim=self.dim, bin_width=20.0,
                                           em_max_iters=self.ITERS, tau_init=self.TAU0)
        return self

    def encode(self, Xs):
        from elephant.gpfa import gpfa_core
        pieces, owner = [], []
        for i, x in enumerate(Xs):
            T = len(x)
            L = min(self.PIECE, T)
            st = list(range(0, T - L + 1, self.KEEP))
            if st[-1] != T - L:
                st.append(T - L)
            centers = np.array(st) + L / 2
            first = len(pieces)
            pieces += [x[a:a + L] for a in st]
            own = np.argmin(np.abs(centers[None, :] - np.arange(T)[:, None]), axis=1)
            owner.append((first, np.array(st), own))
        with quiet():
            lat, _ = gpfa_core.exact_inference_with_ll(_gp_seqs(pieces), self.params, get_ll=False)
            _, orth = gpfa_core.orthonormalize(self.params, lat)
        Z = orth["latent_variable_orth"]
        out = []
        for (first, st, own), x in zip(owner, Xs):
            out.append(np.stack([Z[first + o][:, b - st[o]] for b, o in enumerate(own)]))
        return out


# --------------------------------------------------------------------------------------------
# CEBRA (in process; numpy imported before torch)
# --------------------------------------------------------------------------------------------
CEBRA_CFG = dict(model_architecture="offset10-model", batch_size=512, learning_rate=3e-4, temperature=1.0,
                 time_offsets=10, distance="cosine", output_dimension=3, max_iterations=1000)  # 04 §2.3, §3


def _cebra_fit(X, cfg, seed):
    import torch
    import cebra
    import cebra.distributions.base as cdb
    torch.set_num_threads(1)
    if not hasattr(cdb.HasGenerator, "orig_init"):                  # seed fix (04 §2.7)
        cdb.HasGenerator.orig_init = cdb.HasGenerator.__init__

        def seeded_init(self, device, seed):
            cdb.HasGenerator.orig_init(self, device=device, seed=seed)
            if cdb.HasGenerator.fixed_seed is not None:
                self._seed = cdb.HasGenerator.fixed_seed * 1000 + cdb.HasGenerator.counter
                cdb.HasGenerator.counter += 1
                self._generator = torch.Generator(device=self.device).manual_seed(self._seed)
        cdb.HasGenerator.__init__ = seeded_init
    cdb.HasGenerator.fixed_seed = seed
    cdb.HasGenerator.counter = 0
    torch.manual_seed(seed)
    np.random.seed(seed)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    model = cebra.CEBRA(**cfg, device=device, verbose=False)
    model.fit(X)
    return model


class CEBRAm(Method):
    """CEBRA-Time on the 25 PCs (04 §4): offset10-model, batch 512, lr 3e-4, τ = 1, Δ = 10 bins, cosine, 3 dimensions,
    1000 steps; trained on all bins (pauses included, 04 §1.4), each trial projected on its own."""
    name, rep, dim, cacheable = "CEBRA", "rates", 3, False

    def fit(self, trials):
        self.pc = PCAm().fit(trials)
        X = np.concatenate(self.pc.encode([rep(t, "rates") for t in trials])).astype(np.float32)
        self.model = _cebra_fit(X, dict(CEBRA_CFG, output_dimension=self.dim), self.seed)
        return self

    def encode(self, Xs):
        return [np.asarray(self.model.transform(p.astype(np.float32)), dtype=np.float64) for p in self.pc.encode(Xs)]


# --------------------------------------------------------------------------------------------
# LFADS (lfads-torch with the training loop of 04 §10.5; numpy imported before torch)
# --------------------------------------------------------------------------------------------
SEG_LEN, SEG_OVL = 30, 10                    # 600 ms pieces overlapping by 200 ms (04 §9.2)
SEG_STEP = SEG_LEN - SEG_OVL


def lfads_hp(n_ch):
    """LF_HP of 04 (§11b, §12): REAL_HP (= SYN_HP with 25 factors) with dropout 0.05, KL × 0.3, CD 0."""
    return dict(n_in=n_ch, n_out=n_ch, seg_len=SEG_LEN, enc_dim=64, ic_dim=64, gen_dim=64, fac_dim=25,
                ic_prior_var=0.1, ic_post_var_min=1e-4, cell_clip=5.0, dropout=0.05, cd_rate=0.0,
                kl_ic_scale=0.3 / (SEG_LEN * n_ch), l2_gen_scale=1e-2, l2_ic_enc_scale=0.0,
                ramp_epochs=80, lr=4e-3, loss_scale=1.0)


def make_lfads(hp):
    """Copy of make_lfads of 04 (§10.3): LFADS from the library without controller."""
    import torch  # noqa: F401
    from torch import nn
    from lfads_torch.model import LFADS
    from lfads_torch.modules import augmentations
    from lfads_torch.modules.priors import MultivariateNormal, Null
    from lfads_torch.modules.recons import Poisson
    from lfads_torch.modules.readin_readout import FanInLinear
    cd = augmentations.CoordinatedDropout(cd_rate=hp["cd_rate"], cd_pass_rate=0.0, ic_enc_seq_len=0)
    return LFADS(encod_data_dim=hp["n_in"], encod_seq_len=hp["seg_len"], recon_seq_len=hp["seg_len"],
                 ext_input_dim=0, ic_enc_seq_len=0,
                 ic_enc_dim=hp["enc_dim"], ci_enc_dim=0, ci_lag=1, con_dim=0, co_dim=0,
                 ic_dim=hp["ic_dim"], gen_dim=hp["gen_dim"], fac_dim=hp["fac_dim"], dropout_rate=hp["dropout"],
                 reconstruction=nn.ModuleList([Poisson()]), variational=True, co_prior=Null(),
                 ic_prior=MultivariateNormal(mean=0.0, variance=hp["ic_prior_var"], shape=hp["ic_dim"]),
                 ic_post_var_min=hp["ic_post_var_min"], cell_clip=hp["cell_clip"],
                 train_aug_stack=augmentations.AugmentationStack([cd], batch_order=[0], loss_order=[0]),
                 infer_aug_stack=augmentations.AugmentationStack(),
                 readin=nn.ModuleList([nn.Identity()]), readout=nn.ModuleList([FanInLinear(hp["fac_dim"], hp["n_out"])]),
                 loss_scale=hp["loss_scale"], recon_reduce_mean=True,
                 lr_scheduler=False, lr_init=hp["lr"], lr_stop=1e-5, lr_decay=0.95, lr_patience=6,
                 lr_adam_beta1=0.9, lr_adam_beta2=0.999, lr_adam_epsilon=1e-8, weight_decay=0.0,
                 l2_start_epoch=0, l2_increase_epoch=hp["ramp_epochs"], l2_ic_enc_scale=hp["l2_ic_enc_scale"],
                 l2_ci_enc_scale=0.0, l2_gen_scale=hp["l2_gen_scale"], l2_con_scale=0.0,
                 kl_start_epoch=0, kl_increase_epoch=hp["ramp_epochs"], kl_ic_scale=hp["kl_ic_scale"], kl_co_scale=0.0)


def _seg_starts(T):
    st = list(range(0, T - SEG_LEN + 1, SEG_STEP))
    if st[-1] != T - SEG_LEN:
        st.append(T - SEG_LEN)
    return np.array(st)


def _seg_weights(n):
    x = (np.arange(SEG_OVL) + 0.5) / SEG_OVL
    W = np.ones((n, SEG_LEN))
    W[1:, :SEG_OVL] = x ** 2                                        # AutoLFADS ramps (04 §9.2)
    W[:-1, -SEG_OVL:] = 1 - x ** 2
    return W


def cut_segments(x):
    return np.stack([x[a:a + SEG_LEN] for a in _seg_starts(len(x))])


def stitch(Y, T):
    num, den = np.zeros((T, Y.shape[2])), np.zeros((T, 1))
    for a, y, w in zip(_seg_starts(T), Y, _seg_weights(len(Y))):
        num[a:a + SEG_LEN] += w[:, None] * y
        den[a:a + SEG_LEN] += w[:, None]
    return num / den


class LFADSm(Method):
    """LFADS (04 §11b-§12): 25 factors, generator 64, 600 ms pieces every 400 ms within the trial, 200 epochs,
    version kept = lowest error on the training pieces after the ramps (val = training, as in §12).
    Latents = factors (mean of the initial condition, without dropout) stitched together and ordered by readout."""
    name, rep, dim = "LFADS", "counts", 25
    EPOCHS, BATCH, CLIP = 200, 64, 200 / 1e4

    def __getstate__(self):
        d = dict(self.__dict__)
        d.pop("_model", None)
        return d

    def _torch_model(self):
        if getattr(self, "_model", None) is None:
            import torch
            torch.set_num_threads(1)
            m = make_lfads(self.hp)
            m.load_state_dict({k: torch.tensor(v) for k, v in self.weights.items()})
            m.eval()
            self._model = m
        return self._model

    def fit(self, trials):
        import copy
        import torch
        from lfads_torch.tuples import SessionBatch
        torch.set_num_threads(1)
        torch.manual_seed(self.seed)
        rng = np.random.default_rng(self.seed)
        n_ch = trials[0].counts.shape[1]
        self.hp = lfads_hp(n_ch)
        train = np.concatenate([cut_segments(t.counts.astype(np.float32)) for t in trials])
        model = make_lfads(self.hp)
        model.__class__ = type("LFADSconEpoch", (type(model),), {"current_epoch": 0})
        model.log = model.log_dict = lambda *a, **k: None
        opt = model.configure_optimizers()

        def as_batch(X):
            Xt = torch.tensor(X, dtype=torch.float32)
            return SessionBatch(encod_data=Xt, recon_data=Xt, ext_input=torch.zeros(len(Xt), Xt.shape[1], 0),
                                truth=torch.zeros(0), sv_mask=torch.ones_like(Xt))

        def val_err(X, chunk=512):
            model.eval()
            err = 0.0
            with torch.no_grad():
                for i in range(0, len(X), chunk):
                    sb = as_batch(X[i:i + chunk])
                    out = model({0: sb}, sample_posteriors=False, output_means=False)[0]
                    err += model.recon[0].compute_loss(sb.recon_data, out.output_params).sum().item()
            return err / X.size

        best, best_state, best_epoch, hist = np.inf, None, 0, []
        t0 = time.time()
        for epoch in range(1, self.EPOCHS + 1):
            model.current_epoch = epoch - 1
            model.train()
            order = rng.permutation(len(train))
            costs = []
            for i in range(0, len(train), self.BATCH):
                loss = model._shared_step({0: (as_batch(train[order[i:i + self.BATCH]]), ())}, i, "train")
                opt.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), self.CLIP)
                opt.step()
                costs.append(loss.item())
            ve = val_err(train)
            hist.append((epoch, float(np.mean(costs)), ve))
            if epoch >= self.hp["ramp_epochs"] and ve < best:
                best, best_state, best_epoch = ve, copy.deepcopy(model.state_dict()), epoch
            if epoch % 10 == 0:
                log(f"LFADS seed {self.seed}: epoch {epoch}/{self.EPOCHS}, cost {np.mean(costs):.4f}, error {ve:.4f} "
                    f"({(time.time() - t0) / epoch:.1f} s/epoch)")
        model.load_state_dict(best_state)
        self.weights = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
        self.history, self.best_epoch, self.n_segments = hist, best_epoch, len(train)
        self.seconds = time.time() - t0
        model.eval()
        self._model = model
        W = self.weights["readout.0.weight"]                           # channels × factors
        _, S, Vt = np.linalg.svd(W, full_matrices=False)
        self.order = Vt.T * S                                           # order by readout (04 §12)
        return self

    def encode(self, Xs):
        import torch
        from lfads_torch.tuples import SessionBatch
        m = self._torch_model()
        out = []
        for x in Xs:
            S = cut_segments(np.asarray(x, np.float32))
            Xt = torch.tensor(S, dtype=torch.float32)
            sb = SessionBatch(encod_data=Xt, recon_data=Xt, ext_input=torch.zeros(len(Xt), Xt.shape[1], 0),
                              truth=torch.zeros(0), sv_mask=torch.ones_like(Xt))
            with torch.no_grad():
                f = m({0: sb}, sample_posteriors=False, output_means=False)[0].factors.numpy().astype(np.float64)
            out.append(stitch(f, len(x)) @ self.order)
        return out


METHOD_CLASS = {"PCA": PCAm, "FA": FAm, "GPFA": GPFAm, "Isomap": IsomapM, "UMAP": UMAPm, "CEBRA": CEBRAm, "LFADS": LFADSm}


def make_method(name, seed=0, dim=None):
    return METHOD_CLASS[name](seed=seed, dim=dim)


def default_dim(name):
    return DIM_RULE if name in ("Isomap", "UMAP", "CEBRA") else METHOD_CLASS[name].dim


# ============================================================================================
# Moment matching on the latents
# ============================================================================================
def sym_power(C, p, eps):
    """C^p for symmetric C (p = ±1/2), via eigh, with eps on the diagonal."""
    w, V = np.linalg.eigh((C + C.T) / 2)
    w = np.maximum(w, 0.0) + eps
    return (V * w ** p) @ V.T


def moments(Z):
    mu = Z.mean(0)
    Zc = Z - mu
    return mu, Zc.T @ Zc / len(Z)


class MomentMatch:
    """f(z) = (z − μ_a) · C_a^(−1/2) · C_s^(1/2) + μ_s: brings mean and covariance of the adaptation latents onto those
    of the held-in sessions. Symmetric roots via eigh; eps = EPS_REL × mean eigenvalue on the diagonal.
    Known limitation: it is blind to rotations (if the new latents are a rotation of the old ones, the moments match
    but the points do not)."""

    def __init__(self, Z_a, mu_s, C_s, eps_rel=EPS_REL):
        self.mu_a, C_a = moments(Z_a)
        self.mu_s = mu_s
        e_a = eps_rel * np.trace(C_a) / len(C_a)
        e_s = eps_rel * np.trace(C_s) / len(C_s)
        self.A = sym_power(C_a, -0.5, e_a) @ sym_power(C_s, 0.5, e_s)

    def __call__(self, Z):
        return (Z - self.mu_a) @ self.A + self.mu_s


# ============================================================================================
# Linear readout: ridge on the causal history (H bins) of the standardized latents
# ============================================================================================
def ridge_solve(X, Y, lams, B0=None, intercept="free"):
    """Ridge for several λ. intercept="free": unpenalized intercept (centered), solution towards 0.
    intercept="prior": min ||Y − [X 1] B||² + λ ||B − B0||², intercept included (B0 = held-in weights with
    the intercept in the last row): for λ -> ∞ one returns exactly to the held-in readout.
    Returns {λ: (W, b)}. Primal or dual form, whichever is smaller."""
    X = np.asarray(X, np.float64)
    Y = np.asarray(Y, np.float64)
    if intercept == "free":
        mx, my = X.mean(0), Y.mean(0)
        Xc, R = X - mx, Y - my
        base = np.zeros((X.shape[1], Y.shape[1]))
    elif intercept == "prior":
        Xc = np.hstack([X, np.ones((len(X), 1))])
        base = np.asarray(B0, np.float64)
        R = Y - Xc @ base
    else:
        raise ValueError(intercept)
    n, p = Xc.shape
    out = {}
    if p <= n:
        w, V = np.linalg.eigh(Xc.T @ Xc)
        VR = V.T @ (Xc.T @ R)
        for lam in lams:
            out[lam] = base + V @ (VR / (w + lam)[:, None])
    else:
        w, U = np.linalg.eigh(Xc @ Xc.T)
        UR = U.T @ R
        for lam in lams:
            out[lam] = base + Xc.T @ (U @ (UR / (w + lam)[:, None]))
    if intercept == "free":
        return {lam: (B, my - mx @ B) for lam, B in out.items()}
    return {lam: (B[:-1], B[-1]) for lam, B in out.items()}


class Readout:
    """Latents -> velocity: standardized z (m, s), history of H bins within the trial (decoding.lagged), ridge."""

    def __init__(self, m, s, W=None, b=None):
        self.m, self.s, self.W, self.b = m, s, W, b

    def feats(self, Z):
        return lagged(((Z - self.m) / self.s).astype(np.float32), H).astype(np.float64)

    def predict(self, Z, W=None, b=None):
        return self.feats(Z) @ (self.W if W is None else W) + (self.b if b is None else b)

    def train_rows(self, Zs, trials):
        X = np.concatenate([self.feats(z)[t.mask] for z, t in zip(Zs, trials)])
        Y = np.concatenate([t.vel[t.mask] for t in trials])
        return X, Y


def fit_readout_lodo(rd: Readout, Zs, trials, days, alphas=ALPHA_SRC_GRID):
    """Held-in readout: α chosen leave-one-day-out (R² of each day on its bins with eval_mask, files concatenated as
    in FALCON; mean over days), then fitted on all days. Accumulates X'X day by day."""
    p = H * Zs[0].shape[1]
    acc = {}
    for z, t, d in zip(Zs, trials, days):
        X = rd.feats(z)[t.mask]
        Y = t.vel[t.mask]
        a = acc.setdefault(d, dict(G=np.zeros((p, p)), g=np.zeros((p, 7)), sx=np.zeros(p), sy=np.zeros(7), n=0))
        a["G"] += X.T @ X
        a["g"] += X.T @ Y
        a["sx"] += X.sum(0)
        a["sy"] += Y.sum(0)
        a["n"] += len(X)

    def solve(keep, lams):
        G = sum(acc[d]["G"] for d in keep); g = sum(acc[d]["g"] for d in keep)
        sx = sum(acc[d]["sx"] for d in keep); sy = sum(acc[d]["sy"] for d in keep); n = sum(acc[d]["n"] for d in keep)
        mx, my = sx / n, sy / n
        w, V = np.linalg.eigh(G - n * np.outer(mx, mx))
        VR = V.T @ (g - n * np.outer(mx, my))
        out = {}
        for lam in lams:
            W = V @ (VR / (w + lam)[:, None])
            out[lam] = (W, my - mx @ W)
        return out

    uniq = sorted(acc)
    cv = {a: [] for a in alphas}
    for dout in uniq:
        sol = solve([d for d in uniq if d != dout], alphas)
        idx = [i for i, d in enumerate(days) if d == dout]
        Xd = [rd.feats(Zs[i]) for i in idx]
        y = np.concatenate([trials[i].vel[trials[i].mask] for i in idx])
        for a, (W, b) in sol.items():
            yh = np.concatenate([(x @ W + b)[trials[i].mask] for x, i in zip(Xd, idx)])
            cv[a].append(r2_vw(y, yh))
    score = {float(a): float(np.mean(v)) for a, v in cv.items()}
    best = max(score, key=score.get)
    rd.W, rd.b = solve(uniq, [best])[best]
    rd.alpha, rd.cv = best, score
    rd.alpha_at_edge = best in (float(min(alphas)), float(max(alphas)))
    return rd


# ============================================================================================
# The held-in model: method + moments + readout
# ============================================================================================
@dataclass
class Bundle:
    method: str
    seed: int
    tag: str                  # "heldin" (13 sessions) or "pseudo" (days PSEUDO_TRAIN_DAYS)
    model: Method
    mu_rep: np.ndarray        # held-in input mean (for the recentering)
    lat_mu: np.ndarray        # moments of the held-in latents (for the moment matching)
    lat_C: np.ndarray
    readout: Readout
    info: dict


def bundle_path(method, seed, tag, dim):
    return XS / "models" / f"{method}_d{dim}_{tag}_s{seed}.pkl"


def fit_bundle(method, seed, sessions, tag, use_cache=True, dim=None):
    M = make_method(method, seed, dim)
    p = bundle_path(method, seed, tag, M.dim)
    if use_cache and M.cacheable and p.exists():
        with open(p, "rb") as f:
            b = pickle.load(f)
        assert b.model.dim == M.dim and b.seed == seed, f"{p.name}: cached model differs from the requested one"
        log(f"{method} d = {M.dim} seed {seed} ({tag}): model re-read from {p.name}")
        return b
    trials = [t for s in sessions for t in s.trials]
    days = [s.day for s in sessions for _ in s.trials]
    t0 = time.time()
    M.fit(trials)
    t_fit = time.time() - t0
    reps = [rep(t, M.rep) for t in trials]
    mu_rep = np.concatenate(reps).mean(0)
    Zs = M.encode(reps)
    del reps
    lat_mu, lat_C = moments(np.concatenate(Zs))
    sd = np.sqrt(np.diag(lat_C))
    rd = fit_readout_lodo(Readout(lat_mu, np.where(sd > 1e-12, sd, 1.0)), Zs, trials, days)
    info = dict(fit_seconds=t_fit, total_seconds=time.time() - t0, n_trials=len(trials), n_bins=int(sum(len(t) for t in trials)),
                dim=M.dim,
                sessions=[s.key for s in sessions], alpha=rd.alpha, alpha_at_edge=rd.alpha_at_edge,
                lodo_r2=rd.cv[rd.alpha])
    for k in ("n_points", "n_windows", "best_epoch", "n_segments", "seconds"):
        if hasattr(M, k):
            info[k] = getattr(M, k)
    b = Bundle(method, seed, tag, M, mu_rep, lat_mu, lat_C, rd, info)
    log(f"{method} d = {M.dim} seed {seed} ({tag}): fitted in {info['total_seconds'] / 60:.1f} min; readout α = {rd.alpha:.3g}, "
        f"R² leave-one-day-out = {info['lodo_r2']:.3f}{' (α AT EDGE)' if rd.alpha_at_edge else ''}")
    if M.cacheable:
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p.with_suffix(".tmp"), "wb") as f:
            pickle.dump(b, f)
        os.replace(p.with_suffix(".tmp"), p)
    return b


# ============================================================================================
# Evaluation of one session: 3 folds, all conditions.
# fa_native: FA on the 2 trials (counts minus the mean of the 2 trials, divided by the held-in deviation), same K;
# loadings rotated onto the held-in ones with orthogonal Procrustes (as in Degenhart et al. 2020, but with all channels);
# the rotated latents enter the held-in readout.
# ============================================================================================
@dataclass
class Adapter:
    """Everything that is estimated in the adaptation. Built by `adapt`, which receives ONLY the 2 adaptation trials."""
    mu_a: np.ndarray = None           # input mean of the 2 trials (recenter)
    f: MomentMatch = None             # moment matching (align)
    fewshot: dict = None              # {λ: (W, b)}
    gain: tuple = None                # fewshot_gain: (gain, offset) per DoF on top of align
    fa_native: tuple = None           # (FA of the 2 trials, Procrustes rotation, mean)
    scratch: tuple = None             # (method fitted on the 2 trials, readout, {α: (W, b)})
    rates: tuple = None               # (readout on the rates, {α: (W, b)})
    n_bins: int = 0
    n_bins_mask: int = 0


def adapt(b: Bundle, tad, lams, alphas, scratch=True):
    """Adaptation to the 2 trials `tad` (with kinematics): recenter, align, fewshot, fa_native (FA), scratch."""
    M = b.model
    reps = [rep(t, M.rep) for t in tad]
    A = Adapter(n_bins=int(sum(len(t) for t in tad)), n_bins_mask=int(sum(t.mask.sum() for t in tad)))
    A.mu_a = np.concatenate(reps).mean(0)                           # recenter: means of the 2 trials
    lat = M.encode([r - A.mu_a + b.mu_rep for r in reps])
    A.f = MomentMatch(np.concatenate(lat), b.lat_mu, b.lat_C)       # align: moments of the latents of the 2 trials
    X, Y = b.readout.train_rows([A.f(z) for z in lat], tad)         # fewshot: ridge towards the held-in weights
    B0 = np.vstack([b.readout.W, b.readout.b[None, :]])
    A.fewshot = ridge_solve(X, Y, lams, B0, intercept="prior")
    A.gain = fit_gain(X @ b.readout.W + b.readout.b, Y)            # fewshot_gain: gain and offset per DoF
    if M.name == "FA":                                              # fa_native: FA of the 2 trials, Procrustes on the loadings
        from sklearn.decomposition import FactorAnalysis
        fa = FactorAnalysis(M.dim, svd_method="lapack", random_state=0).fit((np.concatenate(reps) - A.mu_a) / M.sd)
        U, _, Vt = np.linalg.svd(fa.components_ @ M.fa.components_.T)   # L_newᵀ L_src (K × K)
        A.fa_native = (fa, U @ Vt)
    if scratch:                                                     # scratch: method and readout on the 2 trials only
        M2 = make_method(M.name, M.seed, M.dim).fit(tad)
        lz = M2.encode([rep(t, M.rep) for t in tad])
        rd2 = Readout(*zstats(np.concatenate(lz)))
        X2, _ = rd2.train_rows(lz, tad)
        A.scratch = (M2, rd2, ridge_solve(X2, Y, alphas, intercept="free"))
    return A


def predict_trial(b: Bundle, A: Adapter, trial, z0=None):
    """Predictions on the test trial for each condition: {(condition, hyperparameter or None): bins × 7}."""
    M = b.model
    x = rep(trial, M.rep)
    out = {}
    z0 = M.encode([x])[0] if z0 is None else z0
    out[("zero_shot", None)] = b.readout.predict(z0)
    z = M.encode([x - A.mu_a + b.mu_rep])[0]
    out[("recenter", None)] = b.readout.predict(z)
    za = A.f(z)
    out[("align", None)] = b.readout.predict(za)
    out[("fewshot_gain", None)] = out[("align", None)] * A.gain[0] + A.gain[1]
    Xte = b.readout.feats(za)
    for lam, (W, c) in A.fewshot.items():
        out[("fewshot", lam)] = Xte @ W + c
    if A.fa_native is not None:
        fa, R = A.fa_native
        out[("fa_native", None)] = b.readout.predict(fa.transform((x - A.mu_a) / M.sd) @ R)
    if A.scratch is not None:
        M2, rd2, sol = A.scratch
        X2 = rd2.feats(M2.encode([rep(trial, M.rep)])[0])
        for a, (W, c) in sol.items():
            out[("scratch", a)] = X2 @ W + c
    return out


def fit_gain(yhat, y):
    """fewshot_gain: for each DoF, y ≈ g · ŷ + o by least squares on the bins of the 2 trials (2 parameters per DoF)."""
    g, o = np.empty(y.shape[1]), np.empty(y.shape[1])
    for j in range(y.shape[1]):
        A = np.column_stack([yhat[:, j], np.ones(len(y))])
        g[j], o[j] = np.linalg.lstsq(A, y[:, j], rcond=None)[0]
    return g, o


def folds(sess):
    """Leave-one-trial-out rotation: (test trial, adaptation trials)."""
    n = len(sess.trials)
    return [(k, [j for j in range(n) if j != k]) for k in range(n)]


def eval_session(b: Bundle, sess: Sess, lams, alphas, scratch=True):
    """Predictions of the test trial of each fold, per condition: {(condition, hyperparameter or None): [3 arrays]}."""
    tr = sess.trials
    z0 = b.model.encode([rep(t, b.model.rep) for t in tr])          # zero-shot: does not depend on the fold
    out = {}
    for k, ad in folds(sess):
        A = adapt(b, [tr[j] for j in ad], lams, alphas, scratch)
        for key, yh in predict_trial(b, A, tr[k], z0[k]).items():
            out.setdefault(key, [None] * len(tr))[k] = yh
    return out


def eval_rates(sess: Sess, alphas):
    """ridge_rates: ridge (history H) on the smoothed rates of the 2 trials, z-scored with their statistics."""
    tr = sess.trials
    out = {}
    for k, ad in folds(sess):
        reps = [rep(tr[j], "rates") for j in ad]
        rd = Readout(*zstats(np.concatenate(reps)))
        X, Y = rd.train_rows(reps, [tr[j] for j in ad])
        Xte = rd.feats(rep(tr[k], "rates"))
        for a, (W, c) in ridge_solve(X, Y, alphas, intercept="free").items():
            out.setdefault(("ridge_rates", a), [None] * len(tr))[k] = Xte @ W + c
    return out


def concat_r2(sess, preds):
    y = np.concatenate([t.vel[t.mask] for p, t in zip(preds, sess.trials) if p is not None])
    yh = np.concatenate([p[t.mask] for p, t in zip(preds, sess.trials) if p is not None])
    return float(r2_vw(y, yh))


def adapt_bins(sess):
    """Bins of the 2 adaptation trials of each fold: all of them and those with eval_mask."""
    tr = sess.trials
    return [dict(fold=k, bins=int(sum(len(tr[j]) for j in range(3) if j != k)),
                 bins_mask=int(sum(tr[j].mask.sum() for j in range(3) if j != k))) for k in range(3)]


# ============================================================================================
# Pseudo-few-shot on the held-in sessions: choice of λ and α
# ============================================================================================
def tune_path(method, dim=None):
    return XS / "tune" / (f"{method}.json" if dim is None else f"{method}_d{dim}.json")


def _pick(scores):
    best = max(scores, key=scores.get)
    vals = sorted(scores)
    return best, best in (vals[0], vals[-1])


def tune(method, data, seed=0, n_jobs=1, dim=None):
    """Pseudo-few-shot on the held-in sessions: method and readout (or channel decoder) fitted on days S0-S3, then the conditions
    on the 20 pseudo-sessions (blocks of 3 trials of S4 and S5) with all the grids. Each pseudo-session is excluded from the
    fit of the method and of the readout (indeed whole days S4 and S5 are: stricter than leave-one-session-out).
    λ and α = highest mean of the concatenated R²; for λ the whole curve is saved and compared with the asymptote
    (λ -> ∞ = align; for ridge_channels = recenter)."""
    ps = pseudo_sessions(data)
    t0 = time.time()
    b = None
    if method == "rates":
        res = _par(n_jobs, lambda s: eval_rates(s, ALPHA_GRID), ps)
    elif method == "ridge_channels":
        dec = ChannelDecoder.fit(heldin(data, PSEUDO_TRAIN_DAYS))
        res = _par(n_jobs, lambda s: eval_channels(dec, s, LAM_GRID), ps)
    else:
        b = fit_bundle(method, seed, heldin(data, PSEUDO_TRAIN_DAYS), "pseudo", dim=dim)
        dim = b.model.dim
        res = _par(n_jobs, lambda s: eval_session(b, s, LAM_GRID, ALPHA_GRID, scratch=method != "LFADS"), ps)
    per = {}
    for s, r in zip(ps, res):
        for key, preds in r.items():
            per.setdefault(key, []).append(concat_r2(s, preds))
    mean = {k: float(np.mean(v)) for k, v in per.items()}
    out = dict(method=method, dim=dim, seed=seed, pseudo_train_days=PSEUDO_TRAIN_DAYS, pseudo_sessions=[s.key for s in ps],
               seconds=time.time() - t0)
    if b is not None:
        out["bundle"] = {k: v for k, v in b.info.items() if k != "sessions"}
    for cond in ("zero_shot", "recenter", "align", "fa_native", "fewshot_gain"):
        if (cond, None) in mean:
            out[cond] = dict(mean_r2=mean[(cond, None)], per_session=per[(cond, None)])
    for cond, grid_name in (("fewshot", "lam"), ("scratch", "alpha"), ("ridge_rates", "alpha")):
        sc = {k[1]: v for k, v in mean.items() if k[0] == cond}
        if sc:
            best, edge = _pick(sc)
            out[cond] = {grid_name: best, "at_edge": edge, "mean_r2": sc[best], "per_session": per[(cond, best)],
                         "grid": {f"{g:.4g}": v for g, v in sorted(sc.items())}}
    if "fewshot" in out:                                            # asymptote: λ -> ∞ goes back to the held-in readout
        ref = "recenter" if method == "ridge_channels" else "align"
        f = out["fewshot"]
        f["asymptote"] = ref
        f["asymptote_r2"] = mean[(ref, None)]
        f["gain_over_asymptote"] = f["mean_r2"] - mean[(ref, None)]
        f["interior"] = not f["at_edge"]
    name = method if method in ("rates", "ridge_channels") else method
    p = tune_path(name, None if method in ("rates", "ridge_channels") else dim)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    if "fewshot" in out:                                            # the R²(λ) curve, also as CSV
        with open(p.with_suffix(".lambda_curve.csv"), "w") as fh:
            fh.write("lam,mean_r2\n" + "".join(f"{float(k)!r},{v!r}\n" for k, v in out["fewshot"]["grid"].items()))
    log(f"pseudo-few-shot {method} (d = {dim}): " + ", ".join(
        f"{c} {out[c]['mean_r2']:.4f}" + (f" ({g} {out[c][g]:.3g}{' AT EDGE' if out[c]['at_edge'] else ''})" if g else "")
        for c, g in (("zero_shot", None), ("recenter", None), ("align", None), ("fa_native", None), ("fewshot_gain", None),
                     ("fewshot", "lam"), ("scratch", "alpha"), ("ridge_rates", "alpha")) if c in out))
    return out


# ============================================================================================
# Reference: the causal ridge decoder on the channels (05), same scale of conditions
# ============================================================================================
class ChannelDecoder:
    """Causal ridge decoder on the 176 channels of 05 (models/ridge_h50.npz, commit b797b3e): per-channel z-score with held-in mean
    and deviation, history of H = 50 bins, W and b. Trained on whole files; here it is applied trial by trial (the history
    restarts at every trial), like all the other methods. Conditions: zero_shot, recenter (mean of the 2 trials in place of
    the held-in one), fewshot (recenter + ridge towards the held-in weights), fewshot_gain (recenter + gain and offset)."""

    def __init__(self, W, b, mu, sd, H):
        self.W, self.b, self.mu, self.sd, self.H = W, b, mu, sd, int(H)

    @classmethod
    def load(cls, path=RIDGE_H50):
        m = np.load(path)
        return cls(m["W"], m["b"], m["mu"], m["sd"], int(m["H"]))

    @classmethod
    def fit(cls, sessions, H=50, alpha=10 ** 4.5):
        """Same recipe as scripts/05_ridge_fit_final.py (whole files = trials in a row, scaler on all bins, fit on the
        bins with eval_mask, α 10^4.5); needed for the pseudo-few-shot (days S0-S3)."""
        import scipy.linalg
        files = [np.concatenate([t.counts_all for t in s.trials]).astype(np.float32) for s in sessions]
        vels = [np.concatenate([t.vel for t in s.trials]) for s in sessions]
        masks = [np.concatenate([t.mask for t in s.trials]) for s in sessions]
        allc = np.concatenate(files)
        mu, sd = allc.mean(0), allc.std(0)
        sd[sd == 0] = 1.0
        del allc
        p = files[0].shape[1] * H
        G, g, sx, sy, n = np.zeros((p, p)), np.zeros((p, 7)), np.zeros(p), np.zeros(7), 0
        for c, v, m in zip(files, vels, masks):
            X = lagged((c - mu) / sd, H)
            rows = np.flatnonzero(m)
            for i in range(0, len(rows), 2000):
                r = rows[i:i + 2000]
                Xc, y = X[r].astype(np.float64), v[r]
                G += Xc.T @ Xc; g += Xc.T @ y; sx += Xc.sum(0); sy += y.sum(0); n += len(r)
            del X
        mx, my = sx / n, sy / n
        G -= n * np.outer(mx, mx)
        G[np.diag_indices(p)] += alpha
        W = scipy.linalg.solve(G, g - n * np.outer(mx, my), assume_a="pos", overwrite_a=True)
        return cls(W, my - mx @ W, mu.astype(np.float64), sd.astype(np.float64), H)

    def feats(self, counts_all, mu=None):
        z = (counts_all.astype(np.float64) - (self.mu if mu is None else mu)) / self.sd
        return lagged(z.astype(np.float32), self.H).astype(np.float64)

    def predict(self, counts_all, mu=None):
        return self.feats(counts_all, mu) @ self.W + self.b


def eval_channels(dec: ChannelDecoder, sess: Sess, lams):
    tr = sess.trials
    out = {}

    def put(key, k, y):
        out.setdefault(key, [None] * len(tr))[k] = y

    for k, ad in folds(sess):
        tad = [tr[j] for j in ad]                                   # adaptation: the 2 trials only
        mu_a = np.concatenate([t.counts_all for t in tad]).astype(np.float64).mean(0)
        X = np.concatenate([dec.feats(t.counts_all, mu_a)[t.mask] for t in tad])
        Y = np.concatenate([t.vel[t.mask] for t in tad])
        few = ridge_solve(X, Y, lams, np.vstack([dec.W, dec.b[None, :]]), intercept="prior")
        gain = fit_gain(X @ dec.W + dec.b, Y)
        del X
        t = tr[k]                                                   # test
        put(("zero_shot", None), k, dec.predict(t.counts_all))
        Xt = dec.feats(t.counts_all, mu_a)
        rec = Xt @ dec.W + dec.b
        put(("recenter", None), k, rec)
        put(("fewshot_gain", None), k, rec * gain[0] + gain[1])
        for lam, (W, c) in few.items():
            put(("fewshot", lam), k, Xt @ W + c)
    return out


# ============================================================================================
# Diagnosis of moment matching without drift across days (held-in sessions)
# ============================================================================================
def diag_within(b: Bundle, sess: Sess, block=3):
    """In a held-in session, for each block of 3 consecutive trials: source = the OTHER trials of the same session
    (readout refitted on them with the held-in α scaled by the number of bins; moments of their latents; input mean),
    leave-one-trial-out rotation within the block as on the held-out sessions (2 trials adapt, 1 tests). The method (manifold)
    remains the held-in one. Conditions: no_adapt, recenter, align. Between source and block there is no drift across days:
    if align gets worse here, the drop does not come from drift but from how the 2 trials represent the session."""
    M = b.model
    tr = sess.trials
    n = len(tr)
    reps = [rep(t, M.rep) for t in tr]
    lat = M.encode(reps)
    out = {}

    def put(key, k, y):
        out.setdefault(key, [None] * n)[k] = y

    for j in range(n // block):
        blk = list(range(block * j, block * (j + 1)))
        rest = [i for i in range(n) if i not in blk]
        Zr = [lat[i] for i in rest]
        mu_s, C_s = moments(np.concatenate(Zr))
        sd = np.sqrt(np.diag(C_s))
        rd = Readout(mu_s, np.where(sd > 1e-12, sd, 1.0))
        X, Y = rd.train_rows(Zr, [tr[i] for i in rest])
        alpha = b.readout.alpha * sum(len(tr[i]) for i in rest) / b.info["n_bins"]
        rd.W, rd.b = ridge_solve(X, Y, [alpha])[alpha]
        mu_rep_s = np.concatenate([reps[i] for i in rest]).mean(0)
        for k in blk:
            ad = [i for i in blk if i != k]
            put(("no_adapt", None), k, rd.predict(lat[k]))
            mu_a = np.concatenate([reps[i] for i in ad]).mean(0)
            z = M.encode([reps[i] - mu_a + mu_rep_s for i in ad + [k]])
            put(("recenter", None), k, rd.predict(z[-1]))
            f = MomentMatch(np.concatenate(z[:-1]), mu_s, C_s)
            put(("align", None), k, rd.predict(f(z[-1])))
    return out


def _par(n_jobs, fn, items):
    if n_jobs == 1:
        return [fn(x) for x in items]
    from joblib import Parallel, delayed
    return Parallel(n_jobs=n_jobs, backend="loky")(delayed(fn)(x) for x in items)


# ============================================================================================
# Registry: keys, runs already done, writing
# ============================================================================================
def run_keys(method, dim, session, condition, seed, fold=None, experiment=EXPERIMENT, split=SPLIT, decoder=DECODER):
    return dict(experiment=experiment, method=method, latent_dim=dim, decoder=decoder, session=session, bins=BINS,
                split=split, condition=condition, fold=fold, k_shot=0 if condition in NO_ADAPT else K_SHOT,
                H_ms=H_MS, seed=seed)


def expected_conditions(method):
    if method == "rates":
        return ["ridge_rates"]
    if method == "ridge_channels":
        return ["zero_shot", "recenter", "fewshot", "fewshot_gain"]
    c = ["zero_shot", "recenter", "align", "fewshot", "fewshot_gain"] + (["scratch"] if method != "LFADS" else [])
    return c + (["fa_native"] if method == "FA" else [])


def done_runs(path=None, experiment=EXPERIMENT, dim="any"):
    """{(method, seed, session, condition)} with the R² concat row already in the registry (experiment and, if given, d)."""
    import pandas as pd
    from . import results as res
    p = Path(path or res.RUNS)
    if not p.exists():
        return set()
    df = res.load_runs(p)
    df = df[(df.experiment == experiment) & (df.fold == "concat") & (df.metric == "r2")]
    if dim != "any":
        df = df[df.latent_dim.isna()] if dim is None else df[df.latent_dim == dim]
    return {(str(m), None if pd.isna(s) else int(s), str(ss), str(c))
            for m, s, ss, c in zip(df.method, df.seed, df.session, df.condition)}


HP_NAME = {"fewshot": "lam", "scratch": "alpha", "ridge_rates": "alpha"}


def log_session(method, dim, sess, preds, seed, alpha_src=None, experiment=EXPERIMENT, split=SPLIT, decoder=DECODER):
    """Log the folds (those with a prediction) + concat for each condition. In the hyperparameters (extra metrics
    of the fold): λ, α, held-in α."""

    from . import results as res
    for (cond, hp), yh in preds.items():
        extra = {HP_NAME[cond]: hp} if hp is not None else {}
        if alpha_src is not None and cond != "ridge_rates":
            extra["alpha_src"] = alpha_src
        kw = dict(experiment=experiment, split=split, decoder=decoder)
        done = [k for k, y in enumerate(yh) if y is not None]
        for k in done:
            t = sess.trials[k]
            res.log_fold(t.vel, yh[k], t.mask, extra=extra or None, **run_keys(method, dim, sess.key, cond, seed, fold=k, **kw))
        kk = run_keys(method, dim, sess.key, cond, seed, **kw)
        kk.pop("fold")
        res.log_concat(done, **kk)


def code_hash(*objs):
    h = hashlib.sha1()
    for o in objs:
        h.update(json.dumps(o, sort_keys=True, default=str).encode())
    return h.hexdigest()[:12]
