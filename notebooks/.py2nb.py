"""Convert a source file in percent format (# %% / # %% [markdown]) into an .ipynb notebook.

Usage:  python .py2nb.py .src_02_new.py 02_new.ipynb
Then:   jupyter nbconvert --to notebook --execute --inplace 02_new.ipynb
"""
import re
import sys

import nbformat
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

src, dst = sys.argv[1], sys.argv[2]
cells, kind, buf = [], None, []


def flush():
    if kind is None:
        return
    text = "\n".join(buf).strip("\n")
    if not text:
        return
    if kind == "md":
        lines = [re.sub(r"^# ?", "", line) for line in text.split("\n")]
        cells.append(new_markdown_cell("\n".join(lines)))
    else:
        cells.append(new_code_cell(text))


for line in open(src, encoding="utf-8").read().split("\n"):
    if line.startswith("# %% [markdown]"):
        flush()
        kind, buf = "md", []
    elif line.startswith("# %%"):
        flush()
        kind, buf = "code", []
    else:
        buf.append(line)
flush()

nb = new_notebook(cells=cells)
nb.metadata["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb.metadata["language_info"] = {"name": "python"}
nbformat.write(nb, dst)
print(f"{len(cells)} cells -> {dst}")
