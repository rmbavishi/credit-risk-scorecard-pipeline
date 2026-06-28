"""
generate_data.py
=================
Builds the modeling dataset for this portfolio using a hybrid approach:

  1. BUREAU / APPLICATION FEATURES — generated to match the real, well-known
     Lending Club loan dataset schema (column names, ranges, and relationships
     mirror the public Kaggle dataset: https://www.kaggle.com/datasets/wordsforthewise/lending-club).
     If you have downloaded the real `accepted_2007_to_2018Q4.csv` file, you can
     swap it in directly — this script's output schema matches it on the columns
     used below.

  2. CASH-FLOW / ALTERNATIVE-DATA FEATURES — synthetically generated, since no
     public dataset contains real bank-transaction-level data (this is the
     proprietary territory that products like Plaid operate in). These are
     clearly namespaced with a `cf_` prefix so it's always obvious in the
     notebook which features are real-data-style vs. synthetic-only.

Run:
    python data/generate_data.py
Output:
    data/loan_data.csv
"""

import numpy as np
import pandas as pd

RNG = np.random.default_rng(42)
N = 60_000

print(f"Generating {N:,} synthetic loan records (Lending-Club-schema + synthetic cash-flow)...")

# ---------------------------------------------------------------------------
# 1. IDENTIFIERS & LOAN TERMS  (real LC column names)
# ---------------------------------------------------------------------------
issue_d = pd.to_datetime(
    RNG.choice(pd.date_range("2019-01-01", "2024-12-01", freq="MS"), size=N)
)

loan_amnt = np.round(np.clip(RNG.lognormal(9.4, 0.55, N), 1000, 40000), -2)
term = RNG.choice([36, 60], size=N, p=[0.65, 0.35])
purpose = RNG.choice(
    ["debt_consolidation", "credit_card", "home_improvement", "major_purchase",
     "small_business", "medical", "car", "other"],
    size=N, p=[0.45, 0.20, 0.10, 0.08, 0.05, 0.05, 0.04, 0.03]
)
home_ownership = RNG.choice(["MORTGAGE", "RENT", "OWN", "OTHER"], size=N, p=[0.45, 0.40, 0.13, 0.02])
addr_state = RNG.choice(
    ["CA", "TX", "NY", "FL", "IL", "OH", "PA", "GA", "NC", "MI"], size=N
)

# ---------------------------------------------------------------------------
# 2. BUREAU / CREDIT FILE FEATURES  (real LC column names)
# ---------------------------------------------------------------------------
fico_range_low = np.clip(RNG.normal(670, 55, N), 300, 845).round().astype(int)
fico_range_high = fico_range_low + 4

# --- Deliberate drift injection for PSI monitoring demo ---
# Simulates a gradual marketing-mix shift toward lower-FICO applicants over
# the most recent ~6 months of the dataset (a realistic, common driver of
# model monitoring alerts in production).
months_from_start = ((issue_d.year - issue_d.year.min()) * 12 + issue_d.month).values
drift_window_start = months_from_start.max() - 5
in_drift_window = months_from_start >= drift_window_start
drift_intensity = np.where(in_drift_window, (months_from_start - drift_window_start + 1) / 6, 0)
fico_range_low = np.clip(
    fico_range_low - (drift_intensity * RNG.normal(35, 10, N)).astype(int), 300, 845
)
fico_range_high = fico_range_low + 4

annual_inc = np.round(np.clip(RNG.lognormal(10.9, 0.5, N), 15000, 350000), -2)
emp_length = RNG.choice(
    ["< 1 year", "1 year", "2 years", "3 years", "4 years", "5 years",
     "6 years", "7 years", "8 years", "9 years", "10+ years"],
    size=N, p=[0.10, 0.08, 0.09, 0.09, 0.08, 0.08, 0.07, 0.07, 0.07, 0.06, 0.21]
)
emp_length_years = pd.Series(emp_length).str.extract(r"(\d+)").astype(float)[0].fillna(0).values
emp_length_years = np.where(np.array(emp_length) == "10+ years", 10, emp_length_years)

dti = np.clip(RNG.normal(19, 9, N), 0, 60).round(2)
inq_last_6mths = RNG.poisson(0.8, N)
open_acc = np.clip(RNG.poisson(11, N), 1, 40)
total_acc = open_acc + RNG.poisson(8, N)
mort_acc = np.clip(RNG.poisson(1.2, N), 0, 10)
pub_rec = RNG.poisson(0.12, N)
revol_bal = np.round(np.clip(RNG.lognormal(8.7, 1.1, N), 0, 120000), -1)
revol_util = np.clip(RNG.beta(2, 3, N) * 100, 0, 150).round(1)
mths_since_earliest_cr_line = np.clip(RNG.normal(180, 90, N), 12, 600).round().astype(int)
delinq_2yrs = RNG.poisson(0.18, N)
pct_tl_nvr_dlq = np.clip(RNG.normal(94, 8, N), 40, 100).round(1)

# Loan grade (LC assigns this from a proprietary model — we derive a simplified
# version from FICO + DTI just to keep the schema realistic; not used as a
# model FEATURE since in production it would leak target information)
risk_index = (700 - fico_range_low) / 100 + dti / 40
grade = pd.cut(
    risk_index, bins=[-np.inf, 0.0, 0.5, 1.0, 1.5, 2.0, np.inf],
    labels=["A", "B", "C", "D", "E", "F"]
).astype(str)
int_rate = np.clip(7 + risk_index * 6 + RNG.normal(0, 1.5, N), 5.3, 30.99).round(2)
installment = np.round(
    loan_amnt * (int_rate / 1200) / (1 - (1 + int_rate / 1200) ** (-term)), 2
)

# ---------------------------------------------------------------------------
# 3. PAYMENT FREQUENCY  (real LC field is implicitly monthly; many installment
#    / fintech lenders offer borrower-selected pay frequency — keeping this
#    since it mirrors a genuine production risk driver in installment lending)
# ---------------------------------------------------------------------------
pay_frequency = RNG.choice(
    ["weekly", "biweekly", "semimonthly", "monthly"], size=N, p=[0.30, 0.32, 0.20, 0.18]
)

# ---------------------------------------------------------------------------
# 4. SYNTHETIC CASH-FLOW / ALTERNATIVE-DATA FEATURES  (cf_ prefix = NOT real-
#    data-style; this is the proprietary "Plaid/Brigit-style" layer)
# ---------------------------------------------------------------------------
# Underlying "true" financial health latent factor (unobserved) drives several
# correlated cash-flow signals, similar to how real bank-transaction features
# are internally correlated.
financial_health = RNG.normal(0, 1, N)

cf_avg_checking_balance = np.round(np.clip(
    1200 + 900 * financial_health + RNG.normal(0, 400, N), -500, 25000
), 2)
cf_nsf_count_6mo = np.clip(
    RNG.poisson(np.clip(1.5 - financial_health, 0, None)), 0, 15
)
cf_overdraft_count_6mo = np.clip(
    RNG.poisson(np.clip(1.0 - 0.8 * financial_health, 0, None)), 0, 15
)
cf_avg_daily_balance_volatility = np.round(np.clip(
    0.45 - 0.12 * financial_health + RNG.normal(0, 0.08, N), 0.05, 1.2
), 3)
cf_days_since_last_paycheck = np.clip(
    RNG.normal(14 - 2 * financial_health, 6, N), 0, 45
).round().astype(int)
cf_monthly_income_consistency = np.round(np.clip(
    0.85 + 0.08 * financial_health + RNG.normal(0, 0.07, N), 0.3, 1.0
), 3)  # 1.0 = perfectly consistent month to month
cf_num_income_deposits_90d = np.clip(
    RNG.poisson(np.clip(6 + 1.5 * financial_health, 1, None)), 1, 20
)
cf_essential_spend_ratio = np.round(np.clip(
    0.55 - 0.07 * financial_health + RNG.normal(0, 0.08, N), 0.15, 0.95
), 3)  # share of inflow spent on essentials (rent, utilities, groceries)
cf_discretionary_spend_ratio = np.round(np.clip(1 - cf_essential_spend_ratio - RNG.uniform(0, 0.1, N), 0.02, 0.6), 3)
cf_savings_rate = np.round(np.clip(
    0.05 + 0.04 * financial_health + RNG.normal(0, 0.03, N), -0.1, 0.4
), 3)
cf_num_bnpl_accounts = np.clip(RNG.poisson(np.clip(0.9 - 0.4 * financial_health, 0, None)), 0, 6)
cf_existing_loan_payment_ratio = np.round(np.clip(
    0.18 - 0.05 * financial_health + RNG.normal(0, 0.06, N), 0, 0.7
), 3)
cf_account_tenure_days = np.clip(RNG.normal(540 + 80 * financial_health, 200, N), 30, 3650).round().astype(int)
cf_negative_balance_days_90d = np.clip(
    RNG.poisson(np.clip(4 - 3 * financial_health, 0, None)), 0, 90
)
cf_income_source_count = np.clip(RNG.poisson(1.1), 1, 4)

print("Generated 15 synthetic cash-flow (cf_) features")

# ---------------------------------------------------------------------------
# 5. TARGET CONSTRUCTION
# ---------------------------------------------------------------------------
# Combine bureau-style signal with cash-flow signal + pay_frequency timing risk
# into a single underlying default propensity, then sample binary outcomes.

z = (
    -3.1
    - 0.020 * (fico_range_low - 670)
    + 0.035 * dti
    + 0.55 * (revol_util / 100)
    + 0.18 * inq_last_6mths
    + 0.30 * pub_rec
    + 0.55 * (pd.Series(pay_frequency) == "monthly").astype(int).values
    + 0.28 * (pd.Series(pay_frequency) == "semimonthly").astype(int).values
    # cash-flow layer (the "alternative data" lift our scorecard notebook tests)
    - 0.50 * financial_health
    + 0.12 * cf_nsf_count_6mo
    + 0.10 * cf_overdraft_count_6mo
    + 0.40 * cf_existing_loan_payment_ratio
    - 0.35 * cf_monthly_income_consistency
    + RNG.normal(0, 0.55, N)
)
prob_default = 1 / (1 + np.exp(-z))
charged_off = (RNG.random(N) < prob_default).astype(int)

# First payment default - early subset of charge-offs, weighted toward worst
# cash-flow signals (mirrors real production behavior)
fpd_z = (
    -3.2
    + 0.55 * (pd.Series(pay_frequency) == "monthly").astype(int).values
    - 0.45 * financial_health
    + 0.20 * cf_nsf_count_6mo
    + 0.015 * (cf_days_since_last_paycheck)
    + RNG.normal(0, 0.5, N)
)
prob_fpd = 1 / (1 + np.exp(-fpd_z))
first_payment_default = (RNG.random(N) < prob_fpd).astype(int) & charged_off

# ---------------------------------------------------------------------------
# 6. ASSEMBLE + INJECT REALISTIC DATA QUALITY ISSUES
#    (so the cleaning/imputation step in the notebook has real work to do)
# ---------------------------------------------------------------------------
df = pd.DataFrame({
    "loan_id": np.arange(1, N + 1),
    "issue_d": issue_d,
    "loan_amnt": loan_amnt,
    "term": term,
    "int_rate": int_rate,
    "installment": installment,
    "grade": grade,
    "purpose": purpose,
    "home_ownership": home_ownership,
    "addr_state": addr_state,
    "annual_inc": annual_inc,
    "emp_length": emp_length,
    "fico_range_low": fico_range_low,
    "fico_range_high": fico_range_high,
    "dti": dti,
    "inq_last_6mths": inq_last_6mths,
    "open_acc": open_acc,
    "total_acc": total_acc,
    "mort_acc": mort_acc,
    "pub_rec": pub_rec,
    "revol_bal": revol_bal,
    "revol_util": revol_util,
    "mths_since_earliest_cr_line": mths_since_earliest_cr_line,
    "delinq_2yrs": delinq_2yrs,
    "pct_tl_nvr_dlq": pct_tl_nvr_dlq,
    "pay_frequency": pay_frequency,
    # cash-flow / alternative-data layer
    "cf_avg_checking_balance": cf_avg_checking_balance,
    "cf_nsf_count_6mo": cf_nsf_count_6mo,
    "cf_overdraft_count_6mo": cf_overdraft_count_6mo,
    "cf_avg_daily_balance_volatility": cf_avg_daily_balance_volatility,
    "cf_days_since_last_paycheck": cf_days_since_last_paycheck,
    "cf_monthly_income_consistency": cf_monthly_income_consistency,
    "cf_num_income_deposits_90d": cf_num_income_deposits_90d,
    "cf_essential_spend_ratio": cf_essential_spend_ratio,
    "cf_discretionary_spend_ratio": cf_discretionary_spend_ratio,
    "cf_savings_rate": cf_savings_rate,
    "cf_num_bnpl_accounts": cf_num_bnpl_accounts,
    "cf_existing_loan_payment_ratio": cf_existing_loan_payment_ratio,
    "cf_account_tenure_days": cf_account_tenure_days,
    "cf_negative_balance_days_90d": cf_negative_balance_days_90d,
    "cf_income_source_count": cf_income_source_count,
    # targets
    "first_payment_default": first_payment_default.astype(int),
    "charged_off": charged_off,
})

# --- Inject realistic missingness (MCAR-ish, varying rates by column) ---
def inject_missing(series, rate, rng=RNG):
    s = series.copy()
    mask = rng.random(len(s)) < rate
    s[mask] = np.nan
    return s

df["mths_since_earliest_cr_line"] = inject_missing(df["mths_since_earliest_cr_line"], 0.03)
df["mort_acc"] = inject_missing(df["mort_acc"].astype(float), 0.05)
df["pct_tl_nvr_dlq"] = inject_missing(df["pct_tl_nvr_dlq"], 0.04)
df["revol_util"] = inject_missing(df["revol_util"], 0.02)
df["cf_avg_daily_balance_volatility"] = inject_missing(df["cf_avg_daily_balance_volatility"], 0.06)
df["emp_length"] = df["emp_length"].astype(object)
df.loc[RNG.random(N) < 0.04, "emp_length"] = np.nan

# --- Inject a few realistic outliers / data entry errors ---
outlier_idx = RNG.choice(N, size=25, replace=False)
df.loc[outlier_idx, "annual_inc"] = df.loc[outlier_idx, "annual_inc"] * 50  # fat-finger entry errors

df.to_csv("data/loan_data.csv", index=False)

print(f"\nSaved data/loan_data.csv")
print(f"Shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
print(f"Charge-off rate: {df['charged_off'].mean():.2%}")
print(f"FPD rate: {df['first_payment_default'].mean():.2%}")
print(f"\nMissing values by column (top 10):")
print(df.isna().sum().sort_values(ascending=False).head(10))
