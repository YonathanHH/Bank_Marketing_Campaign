"""Compute every number the website shows, straight from raw_data.csv and the saved model.

Nothing on the site is hand-typed: EDA aggregates, the honest benchmark table, the
leakage exhibit and the test-set score vector are all written from here.
"""
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (average_precision_score, fbeta_score,
                             precision_recall_curve, precision_score,
                             recall_score, roc_auc_score, roc_curve)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.tree import DecisionTreeClassifier

from export_model import CAT, NUM, load_frame, split

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "web" / "src" / "data"

MONTH_ORDER = ["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
DAY_ORDER = ["mon", "tue", "wed", "thu", "fri"]


def rate_table(df, col, order=None):
    g = df.groupby(col, observed=True)["target"].agg(n="count", yes="sum")
    g["rate"] = (g["yes"] / g["n"] * 100).round(2)
    g = g.reindex(order) if order else g.sort_values("rate", ascending=False)
    return [{"label": str(i), "n": int(r.n), "yes": int(r.yes), "rate": float(r.rate)}
            for i, r in g.dropna().iterrows()]


def curve_points(x, y, k=160):
    """Even-stride downsample that always keeps the endpoints."""
    idx = np.arange(len(x)) if len(x) <= k else np.unique(np.linspace(0, len(x) - 1, k).astype(int))
    return [[round(float(x[i]), 4), round(float(y[i]), 4)] for i in idx]


def build_eda(df):
    df = df.copy()
    df["target"] = df["y"].map({"no": 0, "yes": 1})
    n, yes = len(df), int(df["target"].sum())

    df["age_band"] = pd.cut(df["age"], [17, 25, 35, 45, 55, 65, 100],
                            labels=["18-25", "26-35", "36-45", "46-55", "56-65", "65+"])
    df["campaign_band"] = pd.cut(df["campaign"], [0, 1, 2, 3, 5, 10, 100],
                                 labels=["1", "2", "3", "4-5", "6-10", "11+"])
    df["prev_band"] = pd.cut(df["previous"], [-1, 0, 1, 2, 100], labels=["0", "1", "2", "3+"])
    df["euribor_band"] = pd.cut(df["euribor3m"], [0, 1, 2, 3, 4, 6],
                                labels=["under 1%", "1-2%", "2-3%", "3-4%", "over 4%"])

    dur = df.groupby("target")["duration"]
    return {
        "rows": n,
        "positives": yes,
        "rate": round(yes / n * 100, 2),
        "job": rate_table(df, "job"),
        "education": rate_table(df, "education"),
        "month": rate_table(df, "month", MONTH_ORDER),
        "day": rate_table(df, "day_of_week", DAY_ORDER),
        "contact": rate_table(df, "contact"),
        "poutcome": rate_table(df, "poutcome"),
        "marital": rate_table(df, "marital"),
        "housing": rate_table(df, "housing"),
        "age": rate_table(df, "age_band"),
        "campaignBand": rate_table(df, "campaign_band"),
        "previousBand": rate_table(df, "prev_band"),
        "euribor": rate_table(df, "euribor_band"),
        "durationByClass": {
            "no": round(float(dur.mean().loc[0]), 1),
            "yes": round(float(dur.mean().loc[1]), 1),
        },
        "missingness": [{"label": c, "pct": round(float(df[c].isna().mean() * 100), 2)}
                        for c in ["default", "education", "housing", "loan", "job", "marital"]],
    }


def make_preprocessor(num=None, cat=None):
    return ColumnTransformer([
        ("num", StandardScaler(), num or NUM),
        ("cat", OneHotEncoder(drop="first", handle_unknown="ignore", sparse_output=False), cat or CAT),
    ])


def evaluate(name, proba, y_true, thr=0.5):
    pred = (proba >= thr).astype(int)
    p, r = precision_score(y_true, pred), recall_score(y_true, pred)
    return {
        "model": name,
        "rocAuc": round(float(roc_auc_score(y_true, proba)), 4),
        "prAuc": round(float(average_precision_score(y_true, proba)), 4),
        "precision": round(float(p), 4),
        "recall": round(float(r), 4),
        "f1": round(float(2 * p * r / (p + r)) if p + r else 0.0, 4),
        "f2": round(float(fbeta_score(y_true, pred, beta=2)), 4),
        "accuracy": round(float((pred == y_true).mean()), 4),
    }


def main():
    df = load_frame()
    eda = build_eda(df)
    (OUT / "eda.json").write_text(json.dumps(eda, separators=(",", ":")), encoding="utf-8")
    print("eda.json ok -- baseline rate", eda["rate"], "%")

    X_train, X_test, y_train, y_test = split(df)
    yte = y_test.values
    pipe = joblib.load(ROOT / "01_bank_marketing_model.sav")
    proba = pipe.predict_proba(X_test)[:, 1]

    # --- honest benchmark on the deployable (no-duration) feature set ------
    candidates = {
        "Logistic Regression": LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42),
        "Decision Tree": DecisionTreeClassifier(max_depth=7, class_weight="balanced", random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(random_state=42),
    }
    bench = []
    for name, est in candidates.items():
        m = Pipeline([("pre", make_preprocessor()), ("clf", est)]).fit(X_train, y_train)
        bench.append(evaluate(name, m.predict_proba(X_test)[:, 1], yte))
        print("  benchmarked", name, bench[-1]["rocAuc"])
    bench.append(evaluate("LightGBM (tuned, deployed)", proba, yte))
    bench[-1]["deployed"] = True

    # --- the leakage exhibit: identical model, plus call duration ----------
    dnum = NUM + ["duration"]
    leaky = Pipeline([
        ("pre", make_preprocessor(dnum, CAT)),
        ("clf", LGBMClassifier(class_weight="balanced", learning_rate=0.03, max_depth=7,
                               n_estimators=250, random_state=42, verbose=-1)),
    ]).fit(df.loc[X_train.index, dnum + CAT], y_train)
    leaky_proba = leaky.predict_proba(df.loc[X_test.index, dnum + CAT])[:, 1]
    leaky_fpr, leaky_tpr, _ = roc_curve(yte, leaky_proba)
    leakage = {
        "withDuration": evaluate("Call duration included", leaky_proba, yte),
        "withoutDuration": evaluate("Call duration excluded", proba, yte),
        "roc": curve_points(leaky_fpr, leaky_tpr),
    }

    # --- curves and decile gains -------------------------------------------
    fpr, tpr, _ = roc_curve(yte, proba)
    prec, rec, _ = precision_recall_curve(yte, proba)
    order = np.argsort(-proba)
    ranked = yte[order]
    cum = np.cumsum(ranked) / ranked.sum()
    deciles = []
    for i in range(10):
        lo, hi = int(len(yte) * i / 10), int(len(yte) * (i + 1) / 10)
        deciles.append({
            "decile": i + 1,
            "captured": round(float(cum[hi - 1] * 100), 1),
            "rate": round(float(ranked[lo:hi].mean() * 100), 1),
        })

    names = pipe.named_steps["preprocessor"].get_feature_names_out()
    gain = pipe.named_steps["classifier"].booster_.feature_importance("gain")
    top = sorted(zip(names, gain), key=lambda t: -t[1])[:14]

    metrics = {
        "train": {"rows": int(len(y_train)), "positives": int(y_train.sum())},
        "test": {"rows": int(len(yte)), "positives": int(yte.sum())},
        "headline": evaluate("deployed", proba, yte),
        "benchmark": bench,
        "leakage": leakage,
        "roc": curve_points(fpr, tpr),
        "pr": curve_points(rec[::-1], prec[::-1]),
        "deciles": deciles,
        "importance": [{"feature": str(n), "gain": round(float(g) / float(gain.sum()) * 100, 2)}
                       for n, g in top],
        # score vector powers the client-side threshold explorer (probability x1000)
        "scores": (np.round(proba, 3) * 1000).astype(int).tolist(),
        "labels": yte.astype(int).tolist(),
    }
    (OUT / "metrics.json").write_text(json.dumps(metrics, separators=(",", ":")), encoding="utf-8")
    print("metrics.json", (OUT / "metrics.json").stat().st_size // 1024, "KB")
    print(json.dumps(metrics["headline"], indent=1))
    print("leakage:", leakage["withDuration"]["rocAuc"], "vs", leakage["withoutDuration"]["rocAuc"])


if __name__ == "__main__":
    main()
