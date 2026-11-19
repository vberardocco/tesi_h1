"""Peak of the population spectrum (the `ctrl_hz` key of slide 10), with the FALCON bins and with the legacy ones.

The value comes from §11 of 04 ("An 18 Hz rhythm in the counts", cell 145 of the source), which the extended 04 does not
rerun (§11 is among the removed cells). Here the same code: times of all spikes of the active channels, counted in
1 ms bins from the start of the session, power spectrum, frequency with the most power above 5 Hz. The spike times
do not depend on the bins; only the active channels and the start of the 1 ms grid (s.times[0]) depend on them.

Usage: python peak_18hz.py [--registry]   (with --registry it writes the row to the run registry, experiment replica_04_extended)
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path.home() / "thesis_h1_work" / "src"))
from thesis_h1 import io  # noqa: E402

SESS = (Path.home() / "thesis_h1_work" / "data/raw/sub-HumanPitt-held-in-calib"
        / "sub-HumanPitt-held-in-calib_ses-19250108T110520.nwb")


def peak(bins):
    s = io.load_session(SESS, bins=bins)
    active = io.active_units(s, min_hz=0.5)
    st_pop = np.concatenate([s.spike_times[u] for u in active])
    h1, _ = np.histogram(st_pop, bins=np.arange(s.times[0], s.times[-1] + io.BIN_S, 0.001))
    f1 = np.fft.rfftfreq(len(h1), d=0.001)
    s1 = np.abs(np.fft.rfft(h1 - h1.mean())) ** 2
    k = np.argmax(s1 * (f1 > 5))
    return float(f1[k]), len(active)


if __name__ == "__main__":
    out = {b: peak(b) for b in ("falcon", "legacy")}
    for b, (f, n) in out.items():
        print(f"bin {b}: {n} active channels, peak at {f:.2f} Hz")
    if "--registry" in sys.argv:
        from thesis_h1 import results
        results.log_run({"peak_hz": out["falcon"][0], "channels_active": float(out["falcon"][1])},
                        experiment="replica_04_extended", method="spectrum_population", latent_dim=None, decoder=None,
                        session="S1_set_1", bins="falcon", split=None, condition="11_peak_spectrum", fold=None,
                        k_shot=None, H_ms=None, seed=None)
        print("written to the run registry")
