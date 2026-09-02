"""Score a handful of profiles with the real sklearn pipeline, for the JS parity test."""
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from export_model import CAT, NUM, load_frame, split

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent

BASE = {
    "age": 38, "campaign": 1, "previous": 0, "job": "admin", "marital": "married",
    "education": "university degree", "housing": "yes", "loan": "no", "contact": "cellular",
    "month": "may", "day_of_week": "thu", "poutcome": "nonexistent",
    "was_contacted_before": "no", "is_default_status_known": "yes",
    "nr.employed": 5099.1, "euribor3m": 1.34, "cons.conf.idx": -46.6,
}

CASES = [
    BASE,
    {**BASE, "age": 62, "job": "retired", "education": "basic 4 years", "housing": "no",
     "month": "oct", "previous": 2, "poutcome": "success", "was_contacted_before": "yes",
     "nr.employed": 5017.5, "euribor3m": 0.74, "cons.conf.idx": -28.7},
    {**BASE, "age": 44, "job": "blue-collar", "education": "basic 9 years",
     "contact": "telephone", "campaign": 11,
     "nr.employed": 5228.1, "euribor3m": 4.95, "cons.conf.idx": -40.4},
    # unknown levels must reach the NaN one-hot column, not the dropped-first level
    {**BASE, "job": "unknown", "marital": "unknown", "education": "unknown", "age": 25},
    {**BASE, "age": 95, "campaign": 20, "job": "student", "month": "mar", "day_of_week": "mon"},
]


def main():
    rows = []
    for case in CASES:
        row = dict(case)
        for key in ("job", "marital", "education"):
            if row[key] == "unknown":
                row[key] = np.nan  # sklearn only treats real NaN as the missing category
        rows.append(row)

    frame = pd.DataFrame(rows)[NUM + CAT]
    pipe = joblib.load(ROOT / "01_bank_marketing_model.sav")
    proba = pipe.predict_proba(frame)[:, 1]

    # a random sample of real held-out rows, as a broader check
    _, X_test, _, _ = split(load_frame())
    sample = X_test.sample(40, random_state=7)
    sample_proba = pipe.predict_proba(sample)[:, 1]
    sample_rows = sample.where(pd.notna(sample), None).to_dict("records")
    for row in sample_rows:
        for key, value in row.items():
            if value is None:
                row[key] = "unknown"

    fixtures = {
        "cases": [
            {"row": case, "expected": float(p)} for case, p in zip(CASES, proba)
        ] + [
            {"row": row, "expected": float(p)} for row, p in zip(sample_rows, sample_proba)
        ]
    }
    out = ROOT / "tools" / "fixtures.json"
    out.write_text(json.dumps(fixtures, indent=1), encoding="utf-8")
    print(f"wrote {len(fixtures['cases'])} fixtures ->", out.name)
    print("probabilities:", [round(float(p), 4) for p in proba])


if __name__ == "__main__":
    main()
