"""Export the deployed LightGBM pipeline to a compact JSON the browser can score.

Reproduces the exact training split used for 01_bank_marketing_model.sav, flattens
the booster into typed arrays, and verifies the JS-equivalent traversal against
sklearn's predict_proba before writing anything out.
"""
import json, warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "web" / "src" / "data"
OUT.mkdir(parents=True, exist_ok=True)

EDU = {
    "basic.4y": "basic 4 years", "basic.6y": "basic 6 years", "basic.9y": "basic 9 years",
    "high.school": "high school", "professional.course": "professional course",
    "university.degree": "university degree", "illiterate": "illiterate",
}
NUM = ["age", "campaign", "previous", "cons.conf.idx", "euribor3m", "nr.employed"]
CAT = ["job", "marital", "education", "housing", "loan", "contact", "month",
       "day_of_week", "poutcome", "is_default_status_known", "was_contacted_before"]


def load_frame():
    df = pd.read_csv(ROOT / "raw_data.csv", sep=";", na_values="unknown")
    df = df.dropna(subset=["housing", "loan"])
    df["job"] = df["job"].str.replace(".", "", regex=False)
    df["education"] = df["education"].map(EDU)
    df["was_contacted_before"] = np.where(df["pdays"] == 999, "no", "yes")
    df["is_default_status_known"] = "yes"
    return df


def split(df):
    y = df["y"].map({"no": 0, "yes": 1})
    X = df[NUM + CAT]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def flatten_trees(booster):
    """Turn the nested dump into parallel arrays: node i is internal if left/right >= 0."""
    dump = booster.dump_model()
    trees = []
    for t in dump["tree_info"]:
        feat, thr, left, right, leaf = [], [], [], [], []

        def walk(node):
            if "leaf_value" in node:
                leaf.append(node["leaf_value"])
                return -len(leaf)  # encode leaf k as -(k+1)
            assert node["decision_type"] == "<=", node["decision_type"]
            idx = len(feat)
            feat.append(node["split_feature"])
            thr.append(node["threshold"])
            left.append(0)
            right.append(0)
            left[idx] = walk(node["left_child"])
            right[idx] = walk(node["right_child"])
            return idx

        root = walk(t["tree_structure"])
        trees.append({
            "root": root,
            "feature": feat,
            "threshold": [round(v, 9) for v in thr],
            "left": left,
            "right": right,
            "leaf": [round(v, 9) for v in leaf],
        })
    return trees


def score(trees, rows):
    """Reference implementation of the traversal the browser will run."""
    out = np.zeros(len(rows))
    for i, x in enumerate(rows):
        total = 0.0
        for t in trees:
            n = t["root"]
            while n >= 0:
                n = t["left"][n] if x[t["feature"][n]] <= t["threshold"][n] else t["right"][n]
            total += t["leaf"][-n - 1]
        out[i] = total
    return 1.0 / (1.0 + np.exp(-out))


def main():
    df = load_frame()
    X_train, X_test, y_train, y_test = split(df)
    pipe = joblib.load(ROOT / "01_bank_marketing_model.sav")
    pre = pipe.named_steps["preprocessor"]
    clf = pipe.named_steps["classifier"]
    scaler = pre.named_transformers_["num"].named_steps["scaler"]
    encoder = pre.named_transformers_["cat"].named_steps["encoder"]

    trees = flatten_trees(clf.booster_)

    # --- verify the flattened traversal matches sklearn exactly -------------
    sample = X_test.iloc[:400]
    expected = pipe.predict_proba(sample)[:, 1]
    got = score(trees, pre.transform(sample))
    err = float(np.abs(expected - got).max())
    print(f"max |sklearn - flattened| on 400 rows: {err:.3e}")
    # rounding thresholds/leaves to 9dp keeps the payload small; drift is ~1e-9 in probability
    assert err < 1e-6, "flattened booster does not reproduce sklearn output"

    # --- feature layout the browser must build ------------------------------
    dropped = [str(c[0]) for c in encoder.categories_]  # drop='first'
    cat_spec = []
    for col, cats in zip(CAT, encoder.categories_):
        kept = [None if (isinstance(c, float) and np.isnan(c)) else str(c) for c in cats[1:]]
        first = cats[0]
        cat_spec.append({
            "name": col,
            "base": None if (isinstance(first, float) and np.isnan(first)) else str(first),
            "levels": kept,
        })

    model = {
        "numeric": [{"name": n, "mean": float(m), "scale": float(s)}
                    for n, m, s in zip(NUM, scaler.mean_, scaler.scale_)],
        "categorical": cat_spec,
        "nFeatures": int(clf.n_features_in_),
        "trees": trees,
    }
    (OUT / "model.json").write_text(json.dumps(model, separators=(",", ":")), encoding="utf-8")
    print("model.json", (OUT / "model.json").stat().st_size // 1024, "KB",
          "| trees:", len(trees), "| dropped-first:", dropped[:3], "...")

    # --- test-set scores drive the threshold explorer client side ----------
    proba = pipe.predict_proba(X_test)[:, 1]
    np.save(ROOT / "tools" / "_test_proba.npy", proba)
    np.save(ROOT / "tools" / "_test_y.npy", y_test.values)
    print("test rows:", len(proba), "positives:", int(y_test.sum()))

    # baseline row (population medians/modes) for the leave-one-out explainer
    baseline = {c: (float(X_train[c].median()) if c in NUM else str(X_train[c].mode()[0]))
                for c in NUM + CAT}
    (OUT / "baseline.json").write_text(json.dumps(baseline, indent=1), encoding="utf-8")
    print("baseline:", baseline)


if __name__ == "__main__":
    main()
