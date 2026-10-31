"""results/sessions.csv: una riga per file NWB di H1 (calib), con la chiave FALCON, la data e i giorni dall'ultima
sessione held-in.

Colonne: session (chiave FALCON, es. S5_set_1), giorno (S5), sid (19250120T115044), data (1925-01-20, da
session_start_time del file NWB), gruppo (held_in / held_out), giorni_da_held_in (giorni dall'ultima data held-in;
0 = S5, negativo per le held-in precedenti). Le chiavi vengono dal caricatore di FALCON (hash_dataset).
results.add_days ricalcola giorni_da_held_in dalla data, quindi la colonna nel file è solo per chi lo legge a mano.
Controllo: la data di session_start_time coincide con quella nel nome del file.

Uso: python scripts/make_sessions_csv.py [cartella dei dati, default ~/000954]  ->  RESULTS/sessions.csv
"""
import csv
import sys
from datetime import date
from pathlib import Path

import h5py

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from falcon_challenge.config import FalconConfig, FalconTask  # noqa: E402
from tesi_h1 import results as res  # noqa: E402

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "000954"
cfg = FalconConfig(FalconTask.h1)
rows = []
for gruppo, split in (("held_in", "held-in-calib"), ("held_out", "held-out-calib")):
    for f in sorted(root.glob(f"sub-HumanPitt-{split}/*.nwb")):
        sid = f.stem.split("_ses-")[1].split("_")[0]
        with h5py.File(f, "r") as h:
            start = h["session_start_time"][()].decode()           # es. 1925-01-20T11:50:44-05:00
        data = start[:10]
        assert data == f"{sid[:4]}-{sid[4:6]}-{sid[6:8]}", f"{f.name}: session_start_time {start} diverso dal nome"
        key = cfg.hash_dataset(f.stem)
        rows.append(dict(session=key, giorno=key.split("_")[0], sid=sid, data=data, gruppo=gruppo))
assert rows, f"nessun file NWB in {root}"
last = max(date.fromisoformat(r["data"]) for r in rows if r["gruppo"] == "held_in")
for r in rows:
    r["giorni_da_held_in"] = (date.fromisoformat(r["data"]) - last).days
res.SESSIONS.parent.mkdir(parents=True, exist_ok=True)
with open(res.SESSIONS, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"{len(rows)} sessioni -> {res.SESSIONS} (ultima held-in: {last})")
