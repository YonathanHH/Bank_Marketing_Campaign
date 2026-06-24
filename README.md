# 🏦 Bank Marketing Campaign: Subscription Prediction and Customer Segmentation

[![Python](https://img.shields.io/badge/Python-3.10-blue)](https://python.org)
[![PyCaret](https://img.shields.io/badge/AutoML-PyCaret-orange)](https://pycaret.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Notebook](https://img.shields.io/badge/Notebook-Jupyter-orange)]()

## 📌 Overview

This project analyzes the UCI Portuguese Bank Marketing dataset (45,211 records, 17 features) through two complementary lenses: **supervised classification** to predict term deposit subscriptions, and **unsupervised clustering** to segment customers into actionable personas.

Two modeling approaches are benchmarked for classification: a **Conventional ML pipeline** (sklearn-based) and an **AutoML approach** using PyCaret, comparing development speed, performance, and interpretability. Customer segmentation is handled using **K-Prototypes**, chosen for its native support of mixed numeric and categorical data.

| Item | Detail |
|---|---|
| Dataset | UCI Bank Marketing (Portuguese bank, 2008-2010) |
| Target | `y` — Term deposit subscription (yes/no) |
| Imbalance Ratio | ~88% No / ~12% Yes |
| Problem Type | Binary Classification + Unsupervised Clustering |
| Evaluation | F2-Score (recall-focused) for classification |

More insights can be found here: [Purwadhika full end-to-end project](https://github.com/jcdspurwadhika/JCDSJKTPM-34_Alpha)

## 💼 Business Problem

Running direct marketing campaigns is costly. The bank needs to identify high-probability subscribers before calling, so the sales team can focus their efforts efficiently. A model with high recall on class `yes` directly reduces wasted call costs and improves conversion rate.

Beyond prediction, understanding **who** those likely subscribers are enables smarter campaign design. Clustering surfaces distinct customer personas with varying conversion rates, giving the marketing team a targeting framework that goes beyond a binary yes/no flag.

**Objectives:**
- Build a reliable classifier to flag likely subscribers
- Minimize false negatives (missed potential subscribers)
- Segment all customers into interpretable groups using K-Prototypes
- Identify which segments have the highest subscription rate above the 11% baseline
- Compare AutoML efficiency vs. manual pipeline development

## 📂 Repository Structure

```
Bank_Marketing_Campaign/
├── raw_data.csv                    # Original UCI dataset (semicolon-delimited)
├── processed_data.csv              # Preprocessed dataset used for classification
├── Final_Project_Alpha.ipynb       # Main classification notebook (Conventional + AutoML)
├── end_to_end_automl.ipynb         # Standalone AutoML exploration notebook
├── clustering_kprototypes.ipynb    # Customer segmentation with K-Prototypes
├── app.py                          # Streamlit deployment app
├── bank_marketing_model.sav        # Saved trained model (joblib)
└── requirements.txt                # Python dependencies
```

## ⚙️ Methodology

### Part 1: Classification

#### Approach 1: Conventional ML (Sklearn Pipeline)

- Preprocessing via `ColumnTransformer` (StandardScaler + OneHotEncoder)
- StratifiedKFold CV (n=5) across 8 baseline models
- Best model selected with SMOTE and class_weight resampling
- Hyperparameter tuning with `GridSearchCV`
- Explainability via SHAP values

#### Approach 2: AutoML with PyCaret

- `setup()` handles preprocessing automatically
- `compare_models()` benchmarks 15+ algorithms simultaneously
- `tune_model()` with Optuna-backed Bayesian search
- `plot_model()` for native explainability plots
- Same train/test split as Approach 1 for fair comparison

### Part 2: Customer Segmentation with K-Prototypes

K-Prototypes is selected over K-Means because the UCI Bank dataset contains a mix of numeric features (age, duration, campaign counts, macroeconomic indicators) and categorical features (job, marital status, education, contact type). K-Means cannot handle categorical data natively without lossy one-hot encoding that distorts cluster distances.

**Pipeline summary:**
1. Load `raw_data.csv` with original categorical string values intact
2. Apply `StandardScaler` to numeric features only to balance the dissimilarity measure
3. Determine optimal k using the Elbow Method (cost vs. number of clusters)
4. Fit final K-Prototypes model with `init='Cao'` for stable initialization
5. Cross-tabulate cluster labels with `y` to compute conversion rate per segment
6. Profile each cluster by mean (numeric) and mode (categorical)
7. Visualize with a normalized heatmap and bar chart of conversion rates
8. Assign business-friendly segment names and export `clustered_customers.csv`

> The clustering is run on the **full dataset** (all `y` values), not filtered to subscribers only. This preserves the contrast signal: a 38% conversion cluster is only meaningful relative to the 11% baseline.

## 📊 Classification Results

| Model | Approach | ROC-AUC | F2-Score | Recall (Yes) |
|---|---|---|---|---|
| LightGBM + balanced weight (tuned) | Conventional | 0.82 | 0.58 | 0.65 |
| Logistic Regression (PyCaret tuned) | AutoML | 0.63 | 0.56 | 0.83 |

Conventional ML provided more control over resampling and threshold tuning. PyCaret delivered comparable performance approximately 5x faster in development time.

## 🔍 Segmentation Results

After running the Elbow Method across k=2 to k=8, the cost curve flattens noticeably after k=5, with **k=4** selected as the optimal value balancing cluster quality and business interpretability. Four segments provide distinct, nameable customer personas without over-segmenting the data.

Each cluster is profiled by numeric means and categorical modes, then ranked by their subscription conversion rate against the 11% dataset baseline. High-converting clusters represent priority targeting groups for future campaigns.

## 🚀 Streamlit App

The classification model is deployed as an interactive Streamlit app. Users can input customer attributes and receive a real-time subscription probability score.

To run locally:
```bash
pip install -r requirements.txt
streamlit run app.py
```

## 🛠️ Requirements

```
streamlit
pandas
joblib
lightgbm
plotly
scikit-learn
kmodes
```

Install all dependencies with:
```bash
pip install -r requirements.txt
```

## 📚 References

Moro, S., Cortez, P., and Rita, P. (2014). A Data-Driven Approach to Predict the Success of Bank Telemarketing. *Decision Support Systems*, Elsevier, 62, 22-31.

UCI Machine Learning Repository: [Bank Marketing Dataset](https://archive.uci.edu/dataset/222/bank+marketing)

## 📄 License

This project is licensed under the MIT License.
