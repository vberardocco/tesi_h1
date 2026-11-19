"""Numbers of slide 10 marked "legacy bins" and VAE of §14.5: old value → new, and run registry (night of 16 Nov, Step 2d).

Old: the outputs of notebooks/04_deep.ipynb (legacy bins), read with the same expressions as scripts/literature_slide_data.py.
New: the outputs of the extended 04 (rep/19250108T110520/04_rep_extended.out.ipynb, FALCON bins), with the same expressions.
The spectrum peak (ctrl_hz) is already in the run registry (peak_18hz.py, 18.08 Hz with both bins).

Usage: python registry_04_extended.py NEW.out.ipynb [--registry]
Without --registry it only prints the table (markdown); with --registry it writes the new values with results.log_run
(experiment "replica_04_extended", session "S1_set_1", bins "falcon"; means across seeds, empty seed).
"""
import json
import re
import sys
from pathlib import Path

GIT = Path.home() / "thesis_h1_git"
OLD = GIT / "notebooks/04_deep.ipynb"


def nb_out(path, *markers):
    nb = json.load(open(path))
    for c in nb["cells"]:
        for o in c.get("outputs", []):
            t = "".join(o.get("text", "")) if "text" in o else "".join(o.get("data", {}).get("text/plain", ""))
            if all(m in t for m in markers):
                return t
    raise KeyError(f"{path}: no output with {markers}")


def search(text, regex, source):
    m = re.search(regex, text, re.S)
    if not m:
        raise KeyError(f"{source}: cannot find {regex!r}")
    return float(m.group(1))


def extract(path):
    """{key: value} from the outputs of a 04 (same expressions as scripts/literature_slide_data.py)."""
    v = {}
    t = nb_out(path, "fase riconosciuta (%), a caso 12.5", "tetto, 12 + 16 trial")
    v["l28_phase_gpfa"] = search(t, r"GPFA alla pari\s+[\d.]+\s+([\d.]+)", path)
    v["l28_phase_lfads"] = search(t, r"LFADS \(3 semi\)\s+[\d.]+ \([^)]*\)\s+([\d.]+)", path)
    t = nb_out(path, "12 + 16 trial", "bit per spike, 40 nascosti")
    b28 = t[t.index("12 + 16 trial"):]
    v["l28_bps_lfads"] = search(b28, r"LFADS \(3 semi\)\s+([\d.]+)", path)
    v["l28_bps_gpfa"] = search(b28, r"GPFA alla pari\s+([\d.]+)", path)
    v["l28_bps_smooth"] = search(b28, r"smoothing \+ GLM\s+([\d.]+)", path)
    t = nb_out(path, "LFADS con il controller, la sua uscita")
    v["ctrl_bps"] = search(t, r"LFADS con il controller, la sua uscita[^\n]*?\s(0\.\d+) \[", path)
    t = nb_out(path, "LFADS con il controller (§14b)", "VAE per bin")
    v["ctrl_phase"] = search(t, r"LFADS con il controller \(§14b\)\s+([\d.]+)", path)
    t = nb_out(path, "fase, tetto (%)", "VAE per bin (3 semi)", "nuovi / visti")   # the table of §14.5
    vae = [float(x) for x in re.findall(r"VAE per bin \(3 semi\)\s+([\d.]+)", t)]
    if len(vae) != 4:
        raise KeyError(f"{path}: table of §14.5 with {len(vae)} VAE values instead of 4")
    v["vae_phase_ceiling"], v["vae_phase_first3"], v["vae_bps"], v["vae_new_seen"] = vae
    return v


# key: (digits, method, condition, metric)
REG = {
    "l28_phase_gpfa": (1, "gpfa_at_even", "14d_12+16_trial", "phase_ceiling_pct"),
    "l28_phase_lfads": (1, "lfads", "14d_12+16_trial", "phase_ceiling_pct"),
    "l28_bps_lfads": (4, "lfads", "14d_12+16_trial", "bps_hidden"),
    "l28_bps_gpfa": (4, "gpfa_at_even", "14d_12+16_trial", "bps_hidden"),
    "l28_bps_smooth": (4, "smoothing_glm", "14d_12+16_trial", "bps_hidden"),
    "ctrl_bps": (4, "lfads_controller", "14b_controller", "bps_hidden"),
    "ctrl_phase": (1, "lfads_controller", "14b_controller", "phase_ceiling_pct"),
    "vae_phase_ceiling": (1, "vae_per_bin", "14.5_vae_per_bin", "phase_ceiling_pct"),
    "vae_phase_first3": (1, "vae_per_bin", "14.5_vae_per_bin", "phase_first3_pct"),
    "vae_bps": (4, "vae_per_bin", "14.5_vae_per_bin", "bps_hidden"),
    "vae_new_seen": (2, "vae_per_bin", "14.5_vae_per_bin", "new_su_seen"),
}
# the old values as they are in results/soglia/tracciabilita.md (check of the reading)
TRACC = {"l28_phase_gpfa": 51.1, "l28_phase_lfads": 51.7, "l28_bps_lfads": 0.0460, "l28_bps_gpfa": 0.0449,
         "l28_bps_smooth": 0.0495, "ctrl_bps": 0.0631, "ctrl_phase": 41.8}


def it(x, nd):
    return f"{x:.{nd}f}".replace(".", ",")


def main(new, registry=False):
    old, nw = extract(OLD), extract(new)
    for k, x in TRACC.items():
        if round(old[k], REG[k][0]) != x:
            sys.exit(f"{k}: the old value read ({old[k]}) is not the one in tracciabilita.md ({x})")
    print("| key | old (legacy bins) | new (FALCON bins) | difference |")
    print("|---|---|---|---|")
    for k, (nd, *_r) in REG.items():
        print(f"| `{k}` | {it(old[k], nd)} | {it(nw[k], nd)} | {'+' if nw[k] >= old[k] else '−'}{it(abs(nw[k] - old[k]), nd)} |")
    if registry:
        sys.path.insert(0, str(Path.home() / "thesis_h1_work" / "src"))
        from thesis_h1 import results
        for k, (nd, method, cond, metric) in REG.items():
            results.log_run({metric: nw[k]}, experiment="replica_04_extended", method=method, latent_dim=None, decoder=None,
                            session="S1_set_1", bins="falcon", split=None, condition=cond, fold=None, k_shot=None,
                            H_ms=None, seed=None)
        print(f"written to the run registry: {len(REG)} rows")


if __name__ == "__main__":
    a = [x for x in sys.argv[1:] if not x.startswith("--")]
    main(Path(a[0]), registry="--registry" in sys.argv)
