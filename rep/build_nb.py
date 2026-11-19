"""Builds the replication notebook of a session from a source in percent format.

Cells are removed by index (sweeps, alternatives, excluded methods); pieces of text can be replaced within a cell
and new cells inserted before a cell. The index is that of cells.py / py2nb (markdown and code counted together).
"""
import json
from pathlib import Path
import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook


def parse(src):
    cells, kind, buf = [], None, []
    def flush():
        if kind is not None:
            cells.append((kind, "\n".join(buf).strip("\n")))
    for line in open(src, encoding="utf-8").read().split("\n"):
        if line.startswith("# %% [markdown]"):
            flush(); kind, buf = "md", []
        elif line.startswith("# %%"):
            flush(); kind, buf = "code", []
        else:
            buf.append(line)
    flush()
    return cells


def build(src, out, skip=(), replace=None, insert_before=None, append=None, title=None):
    replace, insert_before = replace or {}, insert_before or {}
    cells = parse(src)
    nb_cells = []
    if title:
        nb_cells.append(new_markdown_cell(title))
    for i, (kind, text) in enumerate(cells):
        for code in insert_before.get(i, []):
            nb_cells.append(new_code_cell(code))
        if i in skip:
            if kind == "code":                                 # of the removed cells only functions, classes and imports are kept
                import ast
                tree = ast.parse(text)
                def const_assign(nd):                          # simple constants (e.g. N_B, TAU = 512, 1.0; CLIP = 200 / 1e4): needed for default values
                    return isinstance(nd, ast.Assign) and all(
                        isinstance(x, (ast.Name, ast.Tuple, ast.Store)) for tg in nd.targets for x in ast.walk(tg)) and not any(
                        isinstance(x, (ast.Name, ast.Call, ast.Attribute, ast.Subscript, ast.Lambda, ast.comprehension))
                        for x in ast.walk(nd.value))
                keep = [ast.get_source_segment(text, nd) for nd in tree.body
                        if isinstance(nd, (ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)) or const_assign(nd)]
                if keep:
                    nb_cells.append(new_code_cell(f"# REPLICATION: only the definitions of the removed cell {i}\n" + "\n\n".join(keep)))
            continue
        for old, new in replace.get(i, []):
            assert old in text, f"cell {i}: text to replace not found: {old[:60]!r}"
            text = text.replace(old, new)
        if kind == "md":
            import re
            nb_cells.append(new_markdown_cell("\n".join(re.sub(r"^# ?", "", l) for l in text.split("\n"))))
        elif text.strip():
            nb_cells.append(new_code_cell(text))
    for code in append or []:
        nb_cells.append(new_code_cell(code))
    nb = new_notebook(cells=nb_cells)
    nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    nbformat.write(nb, out)
    return len(nb_cells)


def header(sid, rep_root, extra=""):
    """First added cell: session, replication folders, FALCON bins."""
    return f'''# ===== REPLICATION (added): session {sid}, FALCON bins =====
SID = "{sid}"
REP = Path("{rep_root}") / SID                            # results and figures of this session
REP.mkdir(parents=True, exist_ok=True)
SESS = ROOT / f"data/raw/sub-HumanPitt-held-in-calib/sub-HumanPitt-held-in-calib_ses-{{SID}}.nwb"
FIGS = REP / "figures"; FIGS.mkdir(exist_ok=True)
exec(open("{Path(rep_root) / 'falcon_patch.py'}", encoding="utf-8").read())
{extra}'''


FOOTER = '''# ===== REPLICATION (added): save the numbers and tables of the notebook =====
import pickle, numbers
_num = {k: float(v) for k, v in globals().items() if not k.startswith("_") and isinstance(v, numbers.Real) and not isinstance(v, bool)}
_tabs = {k: v for k, v in globals().items() if not k.startswith("_") and isinstance(v, pd.DataFrame)}
_arrs = {k: v for k, v in globals().items() if not k.startswith("_") and isinstance(v, np.ndarray) and v.size <= 2_000_000}
_dicts = {}
for k, v in globals().items():                               # result dictionaries (accuracies, measures): only if they can be pickled
    if not k.startswith("_") and isinstance(v, dict) and k.isupper():
        try:
            pickle.dumps(v); _dicts[k] = v
        except Exception:
            pass
with open(REP / f"{OUT_NAME}.pkl", "wb") as f:
    pickle.dump(dict(numbers=_num, tables=_tabs, arrays=_arrs, dicts=_dicts), f)
print(f"saved {len(_num)} numbers, {len(_tabs)} tables, {len(_arrs)} arrays, {len(_dicts)} dictionaries in {REP / (OUT_NAME + '.pkl')}")
'''
