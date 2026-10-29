"""Per una scelta di celle tenute, elenca i nomi usati da celle tenute che sono definiti solo in celle tolte
(escluse funzioni, classi e import, che build_nb conserva). Uso: from deps import missing."""
import ast, builtins
from build_nb import parse


def names(text):
    tree = ast.parse(text)
    stored, loaded, defs = set(), set(), set()
    for nd in tree.body:
        if isinstance(nd, (ast.FunctionDef, ast.ClassDef)):
            defs.add(nd.name)
        elif isinstance(nd, (ast.Import, ast.ImportFrom)):
            for a in nd.names:
                defs.add((a.asname or a.name).split(".")[0])
    for nd in ast.walk(tree):
        if isinstance(nd, ast.Name):
            (stored if isinstance(nd.ctx, ast.Store) else loaded).add(nd.id)
        elif isinstance(nd, (ast.FunctionDef, ast.ClassDef)):
            stored.add(nd.name)
        elif isinstance(nd, ast.arg):
            stored.add(nd.arg)
    return stored, loaded, defs


def missing(src, skip, injected=()):
    cells = parse(src)
    have = set(dir(builtins)) | set(injected) | {"display", "get_ipython", "__file__"}
    out = []
    info = []
    for i, (kind, text) in enumerate(cells):
        if kind != "code" or not text.strip():
            info.append(None); continue
        info.append(names(text))
    for i, (kind, text) in enumerate(cells):
        if info[i] is None:
            continue
        st, ld, df = info[i]
        if i in skip:
            have |= df
            continue
        need = {n for n in ld - have - st}
        for n in sorted(need):
            src_cells = [j for j in range(i) if info[j] is not None and n in info[j][0] and j in skip]
            if src_cells:
                out.append((i, n, src_cells[-1]))
        have |= st | df
    return out
