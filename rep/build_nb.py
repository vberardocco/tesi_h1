"""Costruisce il notebook di replica di una sessione da un sorgente in formato percent.

Le celle si tolgono per indice (sweep, alternative, metodi esclusi); si possono sostituire pezzi di testo in una cella
e inserire celle nuove prima di una cella. L'indice è quello di cells.py / py2nb (markdown e codice contati insieme).
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
            if kind == "code":                                 # delle celle tolte si tengono solo funzioni, classi e import
                import ast
                tree = ast.parse(text)
                def const_assign(nd):                          # costanti semplici (es. N_B, TAU = 512, 1.0; CLIP = 200 / 1e4): servono ai valori predefiniti
                    return isinstance(nd, ast.Assign) and all(
                        isinstance(x, (ast.Name, ast.Tuple, ast.Store)) for tg in nd.targets for x in ast.walk(tg)) and not any(
                        isinstance(x, (ast.Name, ast.Call, ast.Attribute, ast.Subscript, ast.Lambda, ast.comprehension))
                        for x in ast.walk(nd.value))
                keep = [ast.get_source_segment(text, nd) for nd in tree.body
                        if isinstance(nd, (ast.FunctionDef, ast.ClassDef, ast.Import, ast.ImportFrom)) or const_assign(nd)]
                if keep:
                    nb_cells.append(new_code_cell(f"# REPLICA: solo le definizioni della cella tolta {i}\n" + "\n\n".join(keep)))
            continue
        for old, new in replace.get(i, []):
            assert old in text, f"cella {i}: testo da sostituire non trovato: {old[:60]!r}"
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
    """Prima cella aggiunta: sessione, cartelle della replica, bin di FALCON."""
    return f'''# ===== REPLICA (aggiunta): sessione {sid}, bin di FALCON =====
SID = "{sid}"
REP = Path("{rep_root}") / SID                            # risultati e figure di questa sessione
REP.mkdir(parents=True, exist_ok=True)
SESS = ROOT / f"data/raw/sub-HumanPitt-held-in-calib/sub-HumanPitt-held-in-calib_ses-{{SID}}.nwb"
FIGS = REP / "figures"; FIGS.mkdir(exist_ok=True)
exec(open("{Path(rep_root) / 'falcon_patch.py'}", encoding="utf-8").read())
{extra}'''


FOOTER = '''# ===== REPLICA (aggiunta): salva i numeri e le tabelle del notebook =====
import pickle, numbers
_num = {k: float(v) for k, v in globals().items() if not k.startswith("_") and isinstance(v, numbers.Real) and not isinstance(v, bool)}
_tabs = {k: v for k, v in globals().items() if not k.startswith("_") and isinstance(v, pd.DataFrame)}
_arrs = {k: v for k, v in globals().items() if not k.startswith("_") and isinstance(v, np.ndarray) and v.size <= 2_000_000}
_dicts = {}
for k, v in globals().items():                               # dizionari di risultati (accuratezze, misure): solo se si salvano
    if not k.startswith("_") and isinstance(v, dict) and k.isupper():
        try:
            pickle.dumps(v); _dicts[k] = v
        except Exception:
            pass
with open(REP / f"{OUT_NAME}.pkl", "wb") as f:
    pickle.dump(dict(numbers=_num, tables=_tabs, arrays=_arrs, dicts=_dicts), f)
print(f"salvati {len(_num)} numeri, {len(_tabs)} tabelle, {len(_arrs)} array, {len(_dicts)} dizionari in {REP / (OUT_NAME + '.pkl')}")
'''
