# 🏦 Bank Marketing Campaign - Subscription Prediction

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![PyCaret](https://img.shields.io/badge/AutoML-PyCaret-orange)](https://pycaret.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Notebook](https://img.shields.io/badge/Notebook-Jupyter-orange)]()

## 📌 Overview

This project predicts whether a bank client will subscribe to a term deposit
based on the UCI Portuguese Bank Marketing dataset (45,211 records, 17 features).
Two modeling approaches are benchmarked: a **Conventional ML pipeline** (sklearn-based)
and an **AutoML approach** using PyCaret — comparing development speed, performance,
and interpretability.

| Item            | Detail                                      |
|----------------|----------------------------------------------|
| Dataset         | UCI Bank Marketing (Portuguese bank, 2008-2010) |
| Target          | `y` — Term deposit subscription (yes/no)    |
| Imbalance Ratio | ~88% No / ~12% Yes                          |
| Problem Type    | Binary Classification                       |
| Evaluation      | F2-Score (recall-focused)                   |

More insights could be seen here: [Purwadhika full end-to-end project](https://github.com/jcdspurwadhika/JCDSJKTPM-34_Alpha)

## 💼 Business Problem

Running direct marketing campaigns is costly. The bank needs to identify
high-probability subscribers *before* calling, so the sales team focuses
their efforts. A model with high recall on class `yes` directly reduces
wasted call costs and improves conversion rate.

**Objectives:**
- Build a reliable classifier to flag likely subscribers
- Minimize false negatives (missed potential subscribers)
- Compare AutoML efficiency vs. manual pipeline development

## ⚙️ Methodology

### Approach 1: Conventional ML (Sklearn Pipeline)

- Preprocessing via `ColumnTransformer` (StandardScaler + OneHotEncoder)
- StratifiedKFold CV (n=5) across 8 baseline models
- Best model selected → SMOTE/class_weight resampling
- Hyperparameter tuning with `GridSearchCV`
- Explainability via SHAP values

### Approach 2: AutoML with PyCaret

- `setup()` handles preprocessing automatically
- `compare_models()` benchmarks 15+ algorithms simultaneously
- `tune_model()` with Optuna-backed Bayesian search
- `plot_model()` for native explainability plots
- Same train/test split as Approach 1 for fair comparison
## 📊 Results

| Model                     | Approach       | ROC-AUC | F2-Score | Recall (Yes) |
|--------------------------|----------------|---------|----------|--------------|
| LightGBM + balanced weight (tuned)  | Conventional   | 0.82   | 0.58    | 0.65          |
| Logistic Regression(PyCaret tuned) | AutoML         | 0.63   | 0.56    | 0.83         |

> Conventional ML gave more control over resampling and threshold tuning.
> PyCaret delivered comparable performance ~5x faster in development time.




