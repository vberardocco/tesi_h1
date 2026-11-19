"""results/sessions.csv: one row per H1 NWB file (calib), with the FALCON key, the date and the days since the last
held-in session.

Columns: session (FALCON key, e.g. S5_set_1), day (S5), sid (19250120T115044), data (the date, 1925-01-20, from
session_start_time of the NWB file), group (held_in / held_out), days_from_held_in (days since the last held-in date;
0 = S5, negative for the earlier held-in sessions). The keys come from the FALCON loader (hash_dataset).
results.add_days recomputes days_from_held_in from the date, so the column in the file is only for people reading it by hand.
Check: the date in session_start_time matches the one in the file name.

Usage: python scripts/make_sessions_csv.py [data folder, default ~/000954]  ->  RESULTS/sessions.csv
"""
import csv
import sys
from datetime import date
from pathlib import Path

import h5py

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
from falcon_challenge.config import FalconConfig, FalconTask  # noqa: E402
from thesis_h1 import results as res  # noqa: E402

root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path.home() / "000954"
cfg = FalconConfig(FalconTask.h1)
rows = []
for group, split in (("held_in", "held-in-calib"), ("held_out", "held-out-calib")):
    for f in sorted(root.glob(f"sub-HumanPitt-{split}/*.nwb")):
        sid = f.stem.split("_ses-")[1].split("_")[0]
        with h5py.File(f, "r") as h:
            start = h["session_start_time"][()].decode()           # e.g. 1925-01-20T11:50:44-05:00
        data = start[:10]
        assert data == f"{sid[:4]}-{sid[4:6]}-{sid[6:8]}", f"{f.name}: session_start_time {start} differs from the file name"
        key = cfg.hash_dataset(f.stem)
        rows.append(dict(session=key, day=key.split("_")[0], sid=sid, data=data, group=group))
assert rows, f"no NWB files in {root}"
last = max(date.fromisoformat(r["data"]) for r in rows if r["group"] == "held_in")
for r in rows:
    r["days_from_held_in"] = (date.fromisoformat(r["data"]) - last).days
res.SESSIONS.parent.mkdir(parents=True, exist_ok=True)
with open(res.SESSIONS, "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
print(f"{len(rows)} sessions -> {res.SESSIONS} (last held-in: {last})")
