"""Prove del moment matching, del ridge verso i pesi held-in, della metrica e dell'adattamento senza leakage
(src/tesi_h1/crosssession.py).

Uso: python tests/test_alignment.py
"""
import os
import shutil
import sys
import tempfile
from pathlib import Path

TMP = Path(tempfile.mkdtemp(prefix="tesi_xs_"))
os.environ["TESI_XS"] = str(TMP)                                # cache dei modelli in una cartella temporanea
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import numpy as np  # noqa: E402
from tesi_h1 import crosssession as xs  # noqa: E402
from tesi_h1.decoding import r2_vw  # noqa: E402

FAIL = []


def check(name, ok, detail=""):
    print(f"[{'ok' if ok else 'ERRORE'}] {name}" + (f" -- {detail}" if detail else ""))
    if not ok:
        FAIL.append(name)


def rel_err(Z, ref):
    return float(np.linalg.norm(Z - ref) / np.linalg.norm(ref - ref.mean(0)))


# --- dati sintetici: seme 0, d = 8, z_src di 5000 campioni da una gaussiana anisotropa -----------------
rng = np.random.default_rng(0)
d, n = 8, 5000
A = rng.normal(size=(d, d))
C_true = A @ A.T + 0.5 * np.eye(d)
mu_true = rng.normal(size=d)
L = np.linalg.cholesky(C_true)
z_src = mu_true + rng.normal(size=(n, d)) @ L.T
mu_s, C_s = xs.moments(z_src)

# 1. spostamento + guadagno: media e covarianza portate su quelle di z_src
gain = rng.uniform(0.5, 2.0, size=d)
shift = rng.normal(scale=3.0, size=d)
z_t = z_src * gain + shift
f = xs.MomentMatch(z_t, mu_s, C_s)
mu_f, C_f = xs.moments(f(z_t))
dm, dC = float(np.abs(mu_f - mu_s).max()), float(np.abs(C_f - C_s).max())
check("1. shift + guadagno: |Δmedia| < 1e-6", dm < 1e-6, f"{dm:.2e}")
check("1. shift + guadagno: |Δcovarianza| < 1e-6", dC < 1e-6, f"{dC:.2e}")
print(f"     (errore puntuale relativo dopo shift + guadagno: {rel_err(f(z_t), z_src):.3f}; "
      "il guadagno non isotropo non si inverte esattamente con le radici simmetriche)")

# 2. stessa sessione: un altro campione della stessa distribuzione -> f quasi identità
z_new = mu_true + rng.normal(size=(n, d)) @ L.T
f2 = xs.MomentMatch(z_new, mu_s, C_s)
e2 = rel_err(f2(z_new), z_new)
check("2. stessa sessione: errore puntuale relativo ~0.03", 0.01 < e2 < 0.06, f"{e2:.4f}")

# 3. rotazione pura: la covarianza combacia, i punti no (limite noto: il moment matching è cieco alle rotazioni)
Q, _ = np.linalg.qr(rng.normal(size=(d, d)))
z_rot = (z_src - mu_s) @ Q + mu_s
f3 = xs.MomentMatch(z_rot, mu_s, C_s)
_, C3 = xs.moments(f3(z_rot))
dC3, e3 = float(np.abs(C3 - C_s).max()), rel_err(f3(z_rot), z_src)
check("3. rotazione: |Δcovarianza| < 1e-6", dC3 < 1e-6, f"{dC3:.2e}")
check("3. rotazione: errore puntuale relativo alto (> 0.3)", e3 > 0.3, f"{e3:.3f} (senza allineamento: {rel_err(z_rot, z_src):.3f})")

# 4. ridge verso i pesi held-in: λ -> ∞ restituisce i pesi held-in, λ -> 0 i minimi quadrati
X = rng.normal(size=(400, 30))
Y = X @ rng.normal(size=(30, 7)) + rng.normal(size=(400, 7))
B0 = rng.normal(size=(31, 7))
sol = xs.ridge_solve(X, Y, [1e-10, 1e14], B0, intercept="prior")
W_inf, b_inf = sol[1e14]
check("4. λ -> ∞: pesi e intercetta held-in", np.allclose(W_inf, B0[:-1], atol=1e-6) and np.allclose(b_inf, B0[-1], atol=1e-6))
Xa = np.hstack([X, np.ones((400, 1))])
ols = np.linalg.lstsq(Xa, Y, rcond=None)[0]
W0, b0 = sol[1e-10]
check("4. λ -> 0: minimi quadrati", np.allclose(W0, ols[:-1], atol=1e-6) and np.allclose(b0, ols[-1], atol=1e-6))
Xw = rng.normal(size=(50, 300))                                   # p > n: forma duale
Yw = rng.normal(size=(50, 7))
B0w = rng.normal(size=(301, 7))
lam = 3.0
Xwa = np.hstack([Xw, np.ones((50, 1))])
direct = B0w + np.linalg.solve(Xwa.T @ Xwa + lam * np.eye(301), Xwa.T @ (Yw - Xwa @ B0w))
Wd, bd = xs.ridge_solve(Xw, Yw, [lam], B0w, intercept="prior")[lam]
check("4. forma duale = soluzione diretta", np.allclose(np.vstack([Wd, bd]), direct, atol=1e-8))
Wf, bf = xs.ridge_solve(Xw, Yw, [lam], intercept="free")[lam]
mx, my = Xw.mean(0), Yw.mean(0)
direct_f = np.linalg.solve((Xw - mx).T @ (Xw - mx) + lam * np.eye(300), (Xw - mx).T @ (Yw - my))
check("4. intercetta libera, forma duale = diretta", np.allclose(Wf, direct_f, atol=1e-8) and np.allclose(bf, my - mx @ direct_f))

# 5. la metrica è quella dell'evaluator di FALCON: sklearn r2_score(multioutput="variance_weighted")
from sklearn.metrics import r2_score  # noqa: E402
yt = rng.normal(size=(800, 7)) * rng.uniform(0.1, 3, size=7)
yp = yt + rng.normal(size=yt.shape)
check("5. r2_vw = sklearn variance_weighted", abs(r2_vw(yt, yp) - r2_score(yt, yp, multioutput="variance_weighted")) < 1e-12)
from falcon_challenge.evaluator import FalconEvaluator  # noqa: E402
m = rng.random(800) > 0.1
r_falcon = FalconEvaluator.compute_metrics_regression(yp, yt[m], m, {"S9": [800]})["R2 Mean"]
check("5. = compute_metrics_regression di falcon_challenge (un file, eval_mask)",
      abs(r_falcon - r2_vw(yt[m], yp[m])) < 1e-12, f"{r_falcon:.6f} vs {r2_vw(yt[m], yp[m]):.6f}")

# 6. niente leakage: l'adattamento non vede il trial di test; cambiare il trial di test non cambia nulla del fold
T, C = 700, 40
def fake_trial(seed):
    r = np.random.default_rng(seed)
    lat = np.cumsum(r.normal(size=(T, 3)), 0) / 10
    rate = np.exp(0.5 + lat @ r.normal(size=(3, C)) * 0.3)
    return xs.Trial(seed, r.poisson(rate * 0.02 * 20).astype(np.int16), np.gradient(lat, axis=0) @ r.normal(size=(3, 7)),
                    r.random(T) > 0.05, r.random(T) > 0.3, [(a, a + 50) for a in range(0, T - 50, 90)])
train = [fake_trial(s) for s in range(12)]
sess_like = [xs.Sess(f"S{d_}_set_1", f"S{d_}", "x", "held_in", train[4 * d_:4 * d_ + 4]) for d_ in range(3)]
b = xs.fit_bundle("PCA", 0, sess_like, "prova", use_cache=False)
tr = [fake_trial(100 + i) for i in range(3)]
s1 = xs.Sess("S9_set_1", "S9", "x", "held_out", tr)
alt = fake_trial(999)                                             # trial di test diverso: neurale e cinematica
s2 = xs.Sess("S9_set_1", "S9", "x", "held_out", [tr[0], tr[1], alt])
p1 = xs.eval_session(b, s1, [10.0], [10.0])
p2 = xs.eval_session(b, s2, [10.0], [10.0])
A1 = xs.adapt(b, [tr[0], tr[1]], [10.0], [10.0])
same = all(np.array_equal(xs.predict_trial(b, A1, tr[2])[k], p1[k][2]) for k in p1)
check("6. eval_session = adapt(2 trial) + predict_trial(test)", same)
pa = xs.predict_trial(b, A1, alt)
check("6. fold 2 con un altro trial di test = stesso adattamento applicato al nuovo trial",
      all(np.allclose(pa[k], p2[k][2]) for k in p2))
tr_v = xs.Trial(tr[2].num, tr[2].counts, tr[2].vel * 5 + 1, tr[2].mask, tr[2].mov, tr[2].windows)
p3 = xs.eval_session(b, xs.Sess("S9_set_1", "S9", "x", "held_out", [tr[0], tr[1], tr_v]), [10.0], [10.0])
check("6. la cinematica del trial di test non entra nelle sue previsioni", all(np.array_equal(p1[k][2], p3[k][2]) for k in p1))
check("6. ...ma entra negli altri fold (dove è un trial di adattamento)",
      not np.array_equal(p1[("fewshot", 10.0)][0], p3[("fewshot", 10.0)][0]))
check("6. condizioni attese per PCA", sorted({k[0] for k in p1}) == sorted(xs.expected_conditions("PCA")))

print()
shutil.rmtree(TMP, ignore_errors=True)
if FAIL:
    print(f"{len(FAIL)} prove fallite: {FAIL}")
    sys.exit(1)
print("tutte le prove passate")
