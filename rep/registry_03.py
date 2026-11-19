"""03 rerun with the FALCON bins (night of 16 Nov, Step 3d): numbers of the † A8-A10 of the draft, and run registry.

Reads from the outputs of the copy executed last night (~/thesis_h1_work/rep/night_16_11/nb/03_geometric_figen.out.ipynb) and,
for comparison, from the original notebook (notebooks/03_geometric.ipynb, legacy bins):
- the table of the decoded phase for Isomap and PCA at d = 2, 3, 5, 10, 25 (original cell 33);
- the table of the preserved neighbors for PCA, Isomap, UMAP at d = 2 and 3, real data (UMAP: mean of 5 seeds) and twins
  (mean of 5) (original cell 57).
Usage: python registry_03.py [--registry]   (with --registry it writes the new values: experiment "replica_03", S1_set_1,
bins "falcon")
"""
import json
import re
import sys
from pathlib import Path

GIT = Path.home() / "thesis_h1_git"
W = Path.home() / "thesis_h1_work"
OLD = GIT / "notebooks/03_geometric.ipynb"
NEW = W / "rep/night_16_11/nb/03_geometric_figen.out.ipynb"


def out_with(path, *marks):
    nb = json.load(open(path))
    for c in nb["cells"]:
        for o in c.get("outputs", []):
            t = "".join(o.get("text", "")) if "text" in o else "".join(o.get("data", {}).get("text/plain", ""))
            if all(m in t for m in marks):
                return t
    raise KeyError(f"{path}: no output with {marks}")


PHASE = re.compile(r"dimensioni d\s+2\s+3\s+5\s+10\s+25\s*\nIsomap(\s+[\d.]+){5}\s*\nPCA(\s+[\d.]+){5}")


def phase(path):
    nb = json.load(open(path))
    t = next(x for c in nb["cells"] for o in c.get("outputs", [])
             for x in ["".join(o.get("text", "")) if "text" in o else "".join(o.get("data", {}).get("text/plain", ""))]
             if PHASE.search(x))
    t = PHASE.search(t).group(0)
    head = re.search(r"dimensioni d\s+([\d\s]+?)\n", t).group(1).split()
    out = {}
    for meth in ("Isomap", "PCA"):
        vals = re.search(rf"^{meth}\s+([\d.\s]+)$", t, re.M).group(1).split()
        out.update({(meth.lower(), int(d)): float(v) for d, v in zip(head, vals)})
    return out


def neighbors(path):
    t = out_with(path, "dati veri (5 semi)", "gemelli (5)", "UMAP − PCA")
    out = {}
    for d in (2, 3):
        for cond, lab in (("data", r"dati veri \(5 semi\)"), ("twins", r"gemelli \(5\)")):
            m = re.search(rf"d = {d}, {lab}\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)", t)
            for meth, v in zip(("pca", "isomap", "umap"), m.groups()):
                out[(meth, d, cond)] = float(v)
    return out


def main(registry=False):
    fo, fn = phase(OLD), phase(NEW)
    vo, vn = neighbors(OLD), neighbors(NEW)
    print("| decoded phase (%) | d | original | FALCON bins |\n|---|---|---|---|")
    for k in sorted(fn, key=lambda k: (k[0], k[1])):
        print(f"| {k[0]} | {k[1]} | {fo[k]} | {fn[k]} |")
    print("\n| preserved neighbors (%) | d | data / twins | original | FALCON bins |\n|---|---|---|---|---|")
    for k in sorted(vn):
        print(f"| {k[0]} | {k[1]} | {k[2]} | {vo[k]} | {vn[k]} |")
    if registry:
        sys.path.insert(0, str(W / "src"))
        from thesis_h1 import results
        n = 0
        for (meth, d), v in fn.items():
            results.log_run({"phase_pct": v}, experiment="replica_03", method=meth, latent_dim=d, decoder="knn10",
                            session="S1_set_1", bins="falcon", split=None, condition="phase_trial_never_seen", fold=None,
                            k_shot=None, H_ms=None, seed=None)
            n += 1
        for (meth, d, cond), v in vn.items():
            results.log_run({"neighbors_kept_pct": v}, experiment="replica_03", method=meth, latent_dim=d,
                            decoder=None, session="S1_set_1", bins="falcon", split=None,
                            condition=f"neighbors_{cond}", fold=None, k_shot=None, H_ms=None, seed=None)
            n += 1
        print(f"written to the run registry: {n} rows")


if __name__ == "__main__":
    main(registry="--registry" in sys.argv)
