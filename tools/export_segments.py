"""Refit the k=4 K-Prototypes segmentation and export full persona profiles."""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from kmodes.kprototypes import KPrototypes
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "web" / "src" / "data"

NUM = ["age", "duration", "campaign", "pdays", "previous", "emp.var.rate",
       "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
CAT = ["job", "marital", "education", "default", "housing", "loan", "contact",
       "month", "day_of_week", "poutcome"]

# Elbow costs as fitted in clustering_kprototypes.ipynb (k = 2..8)
ELBOW = [362920.96, 311496.52, 289219.69, 268031.26, 254788.22, 235852.02, 221883.85]

NAMES = {
    "Loyal Responders": (
        "Previously converted, low-rate era",
        "Reached on a cellular line an average of 22 days after a successful prior campaign, "
        "during the low-interest window. One in every 1.6 calls lands a subscription.",
    ),
    "Promising Prospects": (
        "Cheap money, few prior attempts",
        "Contacted about twice, during the same low-rate macro regime, but with no prior "
        "campaign history. Converts at nearly double the portfolio baseline.",
    ),
    "Unengaged Majority": (
        "The default cellular pool",
        "Nearly two thirds of the book. Standard cellular contact during high-rate months, "
        "no prior relationship. This is where the wasted dialling budget goes.",
    ),
    "Resistant Non-Responders": (
        "Over-dialled landline segment",
        "Called 12 times on average via telephone with nothing to show for it. "
        "Every additional call here destroys value.",
    ),
}


def main():
    df = pd.read_csv(ROOT / "raw_data.csv", sep=";")
    target = df["y"].map({"yes": 1, "no": 0})
    X = df[NUM + CAT].copy()
    cat_idx = [X.columns.get_loc(c) for c in CAT]

    X_scaled = X.copy()
    X_scaled[NUM] = StandardScaler().fit_transform(X[NUM])

    print("fitting k=4 ...")
    km = KPrototypes(n_clusters=4, init="Cao", random_state=42, n_jobs=-1)
    labels = km.fit_predict(X_scaled, categorical=cat_idx)

    res = X.copy()
    res["cluster"] = labels
    res["y"] = target

    segments = []
    for c, grp in res.groupby("cluster"):
        rate = float(grp["y"].mean() * 100)
        segments.append({
            "cluster": int(c),
            "size": int(len(grp)),
            "share": round(len(grp) / len(res) * 100, 1),
            "rate": round(rate, 1),
            "lift": round(rate / (target.mean() * 100), 2),
            "numeric": {k: round(float(grp[k].mean()), 2) for k in NUM},
            "modes": {k: str(grp[k].mode().iloc[0]) for k in CAT},
            "topJobs": [{"label": str(i), "pct": round(float(v) * 100, 1)}
                        for i, v in grp["job"].value_counts(normalize=True).head(3).items()],
        })

    segments.sort(key=lambda s: -s["rate"])
    ordered = list(NAMES.items())
    for seg, (name, (tag, blurb)) in zip(segments, ordered):
        seg["name"] = name
        seg["tagline"] = tag
        seg["blurb"] = blurb

    payload = {
        "baseline": round(float(target.mean() * 100), 1),
        "total": int(len(res)),
        "elbow": [{"k": k, "cost": round(c, 1)} for k, c in zip(range(2, 9), ELBOW)],
        "chosenK": 4,
        "segments": segments,
    }
    (OUT / "segments.json").write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
    for s in segments:
        print(f"  {s['name']:<26} n={s['size']:>6}  rate={s['rate']:>5}%  lift={s['lift']}x")


if __name__ == "__main__":
    main()
