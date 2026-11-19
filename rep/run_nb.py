"""Runs a notebook with a per-cell log and writes the executed notebook (even if it stops with an error).

Usage: python run_nb.py NOTEBOOK.ipynb OUTPUT.ipynb LOG
Exit code 0 only if all cells ran successfully.
"""
import sys, time
from pathlib import Path
import nbformat
from nbclient import NotebookClient

src, out, log_path = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
nb = nbformat.read(src, as_version=4)
log = open(log_path, "a")
T0 = time.time()


def stamp(msg):
    log.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} [{(time.time() - T0) / 60:7.1f} min] {msg}\n"); log.flush()


def on_start(cell=None, cell_index=None, **kw):
    first = next((l for l in cell.source.splitlines() if l.strip()), "")[:90]
    stamp(f"start of cell {cell_index}/{len(nb.cells)}: {first}")


def on_done(cell=None, cell_index=None, **kw):
    stamp(f"end of cell {cell_index}")


stamp(f"=== start {src.name}")
client = NotebookClient(nb, timeout=24 * 3600, kernel_name="python3",
                        resources={"metadata": {"path": str(Path.home() / "thesis_h1_work" / "notebooks")}},
                        on_cell_start=on_start, on_cell_executed=on_done)
ok = True
try:
    client.execute()
    stamp("=== DONE")
except Exception as e:
    ok = False
    stamp(f"=== ERROR {type(e).__name__}: {str(e)[-1500:]}")
nbformat.write(nb, out)
sys.exit(0 if ok else 1)
