# Credit Risk Scorecard & Model Monitoring — `sklearn.Pipeline` with LightGBM

Two notebooks demonstrating production-style credit risk work: building a
scorecard as a `sklearn.pipeline.Pipeline`, and independently monitoring
that model in production using PSI, CSI, and statistical drift testing.

**Author:** Rishi Bavishi · [LinkedIn](https://www.linkedin.com/in/rishi-bavishi-25050845/) 

---

## ⚠️ A note on the data

This project uses a **hybrid synthetic dataset** (see [`data/generate_data.py`](data/generate_data.py)):

- **28 bureau / application features** are generated to match the schema, ranges, and
  statistical relationships of the real, publicly available **Lending Club** loan dataset
  ([Kaggle: wordsforthewise/lending-club](https://www.kaggle.com/datasets/wordsforthewise/lending-club)).
- **15 cash-flow / alternative-data features** (prefixed `cf_`) are fully synthetic. No public
  dataset contains real bank-transaction-level data — that's the proprietary territory products
  like Plaid operate in — so these are constructed to mirror the *kind* of signal that data
  provides, with a deliberately constructed underlying relationship to default risk.

No real, proprietary, or employer data is used anywhere in this repo.

---

## What's in this repo

| File | What it covers |
|---|---|
| [`notebooks/01_credit_risk_scorecard_pipeline.ipynb`](notebooks/01_credit_risk_scorecard_pipeline.ipynb) | The full pipeline: a custom `FeatureEngineer` transformer, `ColumnTransformer`-based imputation/encoding, a reusable pipeline-builder function, champion/challenger comparison (bureau-only vs. + cash-flow), ROC/AUC/KS evaluation, permutation feature importance, SHAP feature importance, sample loan with SHAP features' contribution to decisioning and a live demo scoring a single new application end-to-end |

| [`notebooks/02_model_monitoring_psi.ipynb`](notebooks/02_model_monitoring_psi.ipynb) | Independent model monitoring: Population Stability Index (PSI) on input features, Characteristic Stability Index (CSI) on the model's own score distribution, and Kolmogorov-Smirnov (KS) testing as a cross-check — with rolling monthly tracking that catches drift a single snapshot comparison would miss |
---

## Sample results

| | |
|---|---|
| ![ROC Curve](outputs/roc_curve.png) | ![Feature Importance](outputs/feature_importance.png) |
| ![SHAP Summary](outputs/shap_summary.png) | ![SHAP Single Loan](outputs/shap_single_loan.png) |
| ![Score Distribution Shift](outputs/score_distribution_shift.png) | ![PSI Summary](outputs/psi_summary.png) |
| ![PSI Drilldown](outputs/psi_drilldown.png) | ![Rolling PSI](outputs/rolling_psi.png) |

Adding the 15 cash-flow attributes lifted the Gini coefficient from **0.506 to 0.596** (a
**+0.090** absolute lift), with cash-flow attributes accounting for roughly half of the
top 15 most predictive features.

The monitoring notebook catches a gradual population drift in the recent vintages —
invisible in a single before/after snapshot, but clearly visible in the rolling
monthly PSI trend, which crosses the significant-shift threshold well before it
would show up in lagging performance metrics like charge-off rate.

---

## Why I built this

In my day-to-day work I build and validate ML-based credit decisioning models and evaluate
whether alternative data sources (e.g., cash-flow attributes) justify integration into
existing scorecards — quantified via Gini lift in a champion/challenger framework. This
notebook reconstructs that methodology on synthetic data, built the way it would actually
be deployed. Additionally, all deployed models need to be monitored for their stability and 
drift the second notebook captures just that.

---

## Tech stack

- Python 3.10+
- pandas, numpy
- **LightGBM** (gradient boosting model)
- scikit-learn (`Pipeline`, `ColumnTransformer`, custom transformers, permutation importance, metrics)
- SHAP feature importance
- scipy (statistical testing)
- matplotlib

## Running locally

```bash
git clone https://github.com/<your-username>/credit-risk-scorecard-pipeline.git
cd credit-risk-scorecard-pipeline
pip install -r requirements.txt
python data/generate_data.py        # regenerate the synthetic dataset
jupyter notebook notebooks/          # open and run the notebook
```

---

## Background

10+ years of experience in consumer lending, risk and finance analytics, currently as a Principal,
Risk and Analytics. My recent work has included evaluating alternative cash-flow data for credit
decisioning, using the same champion/challenger methodology demonstrated in this repo.
