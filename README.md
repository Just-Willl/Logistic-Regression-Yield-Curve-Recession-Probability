# UK Yield Curve — Recession Start (12-Month) Forecast

This script pulls UK yield and GDP data, builds a simple 10y–3m yield spread, and uses a logistic regression to estimate the chance that a UK recession **starts** within the next 12 months. It also makes a few clear plots.

## What it does (in order)
1. Download monthly UK yields (10y gilt, 3m T-bill, 3m IBOR) and quarterly UK real GDP.
2. Mark “recession quarters” as runs of ≥2 negative QoQ GDP and shade them in plots.
3. Splice the short rate: extend 3m T-bill using 3m IBOR via a simple overlap regression.
4. Create the feature: spread = (10y gilt) − (3m short rate).
5. Create the label: `1` if a recession **starts** at any point in the next 12 months, else `0`.
6. Train a logistic regression with time-series cross-validation (12-month gap to avoid leakage).
7. Refit on all data and print the **latest** probability that a recession starts within 12 months.
8. (Optional) Do a month-by-month walk-forward backtest and plot probabilities.
