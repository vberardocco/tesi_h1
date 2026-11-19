"""Test of the generator of the R9 checks (post hoc; rules in docs/literature/rules_r9_checks.md, commit 067e8f5).

T1: the FA share measured as in slide 5 is within ± 0.02 of the requested value, with the calibrated g.
T2: smoothed expected rates, without Poisson, d = 6, one bin every 10 and every 20: TwoNN and MLE10 between 4.5 and 7.5.
T3: as T2 with d = 3: between 2.5 and 4.5, median below that of T2.
Post-hoc modification (option 1): T2 and T3 only at the g of level 0.167, on the medians of the 3 seeds.
The expected rates depend on g (exp(g S)): this is why T2 and T3 use the calibrated g of level 0.167 (for σ_lat).
Requires the S1 frequencies (literature_cache/phase2/S1_freq_a.npy) and the calibration (scripts/literature_r9c_calibrate.py).
"""
import json
from pathlib import Path

import numpy as np
import pytest

from thesis_h1 import literature2

C = Path.home() / "thesis_h1_work/literature_cache"
F_FREQ, F_CAL = C / "phase2/S1_freq_a.npy", C / "r9c/calibration.json"
pytestmark = pytest.mark.skipif(not (F_FREQ.exists() and F_CAL.exists()), reason="calibration missing")
CAL = json.loads(F_CAL.read_text()) if F_CAL.exists() else {}
LIVELLI = sorted(CAL)
SEMI = (0, 1, 2)


def _estimates(d, g, sigma_lat, seed, m):
    gen = literature2.generate_check(np.load(F_FREQ), g, seed, d=d, sigma_lat=sigma_lat)
    X = literature2.preprocessa_a(gen["rates"])
    return {s: float(np.median(v)) for s, v in literature2.estimates_thinning(X, m).items()}


def test_reading_thinning():
    """The readings of the rules on hand-written cases."""
    assert literature2.reading_thinning(6.0, 7.4, 6) == dict(stable=True, around=True)
    assert literature2.reading_thinning(6.0, 7.6, 6) == dict(stable=False, around=False)
    assert literature2.reading_thinning(20.0, 21.0, 6) == dict(stable=True, around=False)


def test_thin_sfasamenti():
    X = np.arange(25)[:, None]
    assert [len(literature2.thin(X, 10, o)) for o in range(10)] == [3] * 5 + [2] * 5
    assert literature2.thin(X, 10, 3)[:, 0].tolist() == [3, 13, 23]


@pytest.mark.parametrize("level", LIVELLI)
def test_T1_share_fa(level):
    c = CAL[level]
    gen = literature2.generate_check(np.load(F_FREQ), c["g"], 0, sigma_lat=c["sigma_lat"])
    q = literature2.share_fa_slide5(gen["counts"], n_jobs=6)
    assert abs(q - c["share"]) <= 0.02, (level, q)


@pytest.mark.parametrize("level", [k for k in LIVELLI if k.startswith("0.167_")])
def test_T2_T3_without_noise(level):
    """Post-hoc modification (option 1): only at the g of level 0.167, on the medians of the 3 seeds at m = 10 and m = 20."""
    c = CAL[level]
    for m in (10, 20):
        t2 = [_estimates(6, c["g"], c["sigma_lat"], s, m) for s in SEMI]
        t3 = [_estimates(3, c["g"], c["sigma_lat"], s, m) for s in SEMI]
        for stim in ("TwoNN", "MLE10"):
            v2, v3 = np.median([r[stim] for r in t2]), np.median([r[stim] for r in t3])
            assert 4.5 <= v2 <= 7.5, ("T2", level, m, stim, v2)
            assert 2.5 <= v3 <= 4.5, ("T3", level, m, stim, v3)
            assert v3 < v2, ("T3 median", level, m, stim)

