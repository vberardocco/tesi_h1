"""results/sessions.csv: una riga per file NWB di H1 (calib), con la chiave FALCON e la data.

Colonne: session (chiave FALCON, es. S5_set_1), giorno (S5), sid (19250120T115044), data (1925-01-20),
gruppo (held_in / held_out). Le chiavi vengono dal caricatore di FALCON (hash_dataset), senza leggere i dati.
Serve per i giorni dall'ultima sessione held-in (results.add_days); non va ripetuto in ogni riga del registro.

Uso: python scripts/make_sessions_csv.py [cartella dei dati, default ~/000954]  ->  RESULTS/sessions.csv
"""
import csv
import sys
from pathlib import Path

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
        key = cfg.hash_dataset(f.stem)
        rows.append(dict(session=key, giorno=key.split("_")[0], sid=sid,
                         data=f"{sid[:4]}-{sid[4:6]}-{sid[6:8]}", gruppo=gruppo))
assert rows, f"nessun file NWB in {root}"
res.SESSIONS.parent.mkdir(parents=True, exist_ok=True)
with open(res.SESSIONS, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"{len(rows)} sessioni -> {res.SESSIONS}")
