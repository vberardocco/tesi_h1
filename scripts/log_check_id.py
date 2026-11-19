"""Log the verdict of scripts/check_estimators_id.py (step 1 of R9) to h1_runs.csv.

Usage: PYTHONPATH=$HOME/thesis_h1_git/src python scripts/log_check_id.py <folder containing check_id.csv>
"""
import sys
from pathlib import Path

import pandas as pd

from thesis_h1 import results

folder = Path(sys.argv[1]).expanduser()
df = pd.read_csv(folder / "check_id.csv")
outcome = df["outcome"].astype(str)
metrics = {
    "rows": len(df),
    "rows_different": int(outcome.str.contains("DIFFERENT").sum()),
    "twonn_different": int(outcome.str.contains("TwoNN DIFFERENT").sum()),
    "mle_different": int(outcome.str.contains("MLE DIFFERENT").sum()),
    "rows_different_test1": int(outcome[df.test == 1].str.contains("DIFFERENT").sum()),
    "rows_different_test2": int(outcome[df.test == 2].str.contains("DIFFERENT").sum()),
}
metrics["repo_matches"] = float(metrics["rows_different"] == 0)
results.log_run(metrics, experiment="literature", method="TwoNN+MLE10", latent_dim=None, decoder=None,
                session=None, bins="falcon", split="session", condition="9_check_estimators_id", fold=None,
                k_shot=None, H_ms=None, seed=0)
print(metrics)
