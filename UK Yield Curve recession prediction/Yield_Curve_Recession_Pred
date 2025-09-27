# UK Yield Curve & Recession-Start Forecast (12 months ahead)
# -----------------------------------------------------------
# What this script does (end-to-end, tidy order):
# 1) Fetch monthly UK yields (10y gilt, 3m T-bill, 3m IBOR) + quarterly UK GDP (ONS).
# 2) Identify recession quarters (runs of >=2 negative QoQ GDP); plot GDP & yields with recession shading.
# 3) Splice short rate: extend 3m T-bill using 3m IBOR via overlap regression; plot results.
# 4) Build the single feature: yield spread = 10y gilt − 3m short rate (combined); plot with recessions.
# 5) Build the label (THIS is the model’s target, the “forecast indicator”):
#    recession_start_next12m = 1 if a recession STARTS at any point in the next 12 months; else 0.
#    (Recession “start” = first quarter of a run of >=2 consecutive negative QoQ GDP.)
# 6) Train & evaluate ONE logistic regression (second model only) using TimeSeriesSplit with a 12-month gap
#    to avoid shared-future leakage. Report pooled OOF (out-of-fold) ROC-AUC, PR-AUC, Brier, base rate.
# 7) Refit on all labeled data and print the latest forecast probability.

import io, requests, pandas as pd, numpy as np
import matplotlib.pyplot as plt
from cycler import cycler

# Modeling
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss

# Regression for splicing
import statsmodels.api as sm


# === Plot style (keep) ===
def apply_economist_style():
    plt.rcParams.update({
        "figure.figsize": (10, 4),
        "figure.facecolor": "white",
        "axes.facecolor":   "white",
        "axes.edgecolor":   "black",
        "axes.grid":        True,
        "grid.color":       "#e5e5e5",
        "grid.linestyle":   "-",
        "grid.linewidth":   0.7,
        "axes.axisbelow":   True,
        "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11,
        "axes.spines.top": False, "axes.spines.right": False,
        "legend.frameon": False,
        "lines.linewidth": 1.8,
        "axes.prop_cycle": cycler(color=[
            "#1f77b4","#E3120B","#4c78a8","#72b7b2","#8c8c8c","#b279a2"
        ])
    })
apply_economist_style()


# === 1) Fetch & prepare data ===
# FRED CSV endpoints (monthly)
gilt10 = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=IRLTLT01GBM156N",
                     parse_dates=["observation_date"], index_col="observation_date") \
         .rename(columns={"IRLTLT01GBM156N":"gilt10"})["gilt10"]
tb3m   = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=IR3TTS01GBM156N",
                     parse_dates=["observation_date"], index_col="observation_date") \
         .rename(columns={"IR3TTS01GBM156N":"tb3m"})["tb3m"]
ibor3m = pd.read_csv("https://fred.stlouisfed.org/graph/fredgraph.csv?id=IR3TIB01GBM156N",
                     parse_dates=["observation_date"], index_col="observation_date") \
         .rename(columns={"IR3TIB01GBM156N":"ibor3m"})["ibor3m"]

# ONS GDP (quarterly)
GEN_URL = ("https://www.ons.gov.uk/generator?format=csv"
           "&uri=/economy/grossdomesticproductgdp/timeseries/abmi/ukea")
hdrs = {
    "User-Agent": "Mozilla/5.0",
    "Referer": "https://www.ons.gov.uk/economy/grossdomesticproductgdp/timeseries/abmi/ukea",
    "Accept": "text/csv,application/octet-stream"
}
r = requests.get(GEN_URL, headers=hdrs, timeout=30)
r.raise_for_status()
gdp_raw = pd.read_csv(io.StringIO(r.text))

# Keep quarterly rows; clean index → quarter ends
gdp_q = gdp_raw.iloc[84:,:].reset_index(drop=True)   # first ~84 are annual
gdp_q.columns = ["TIME", "GDP"]
gdp = gdp_q.set_index("TIME")["GDP"].astype(float)
gdp.index = gdp.index.str.replace(' ', '')
gdp.index = pd.PeriodIndex(gdp.index, freq='Q').to_timestamp(how='end')
gdp = gdp.to_frame()

# Ensure sorted
gilt10 = gilt10.sort_index()
tb3m   = tb3m.sort_index()
ibor3m = ibor3m.sort_index()
gdp    = gdp.sort_index()

# Inspect heads/tails
print("10-Year Gilt Yield (head & tail):")
print(gilt10.head(), "\n", gilt10.tail())
print("\n3-Month T-Bill (head & tail):")
print(tb3m.head(), "\n", tb3m.tail())
print("\n3-Month IBOR (head & tail):")
print(ibor3m.head(), "\n", ibor3m.tail())
print("\nUK GDP quarterly (head & tail):")
print(gdp.head(), "\n", gdp.tail())


# === 2) Recession quarters & plots ===
# Recession quarter = any quarter with negative QoQ, AND (prev OR next) quarter also negative.
qoq = gdp["GDP"].pct_change()
neg = (qoq < 0)
recession_q = (neg & (neg.shift(1).fillna(False) | neg.shift(-1).fillna(False))).astype(int)
gdp["recession_q"] = recession_q
print("\nRecession quarter counts:\n", gdp["recession_q"].value_counts())

# Plot GDP with recession shading
fig, ax = plt.subplots(figsize=(10,4))
gdp["GDP"].plot(ax=ax, lw=1.2, label="Real GDP")
is_rec = gdp["recession_q"].astype(bool)
ax.fill_between(gdp.index, 0, 1, where=is_rec, transform=ax.get_xaxis_transform(),
                color="red", alpha=0.3)
ax.set_title("UK GDP with recession shading")
ax.set_ylabel("£m (chained-volume)")
ax.grid(True); ax.legend(); plt.tight_layout(); plt.show()

# Plot three yields with shading
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
plt.subplots_adjust(hspace=0.3)
gilt10.plot(ax=ax1, lw=1.2, title="10-Year Gilt Yield", ylabel="%", xlabel="")
ax1.fill_between(gdp.index, 0, 1, where=is_rec, transform=ax1.get_xaxis_transform(),
                 color="red", alpha=0.3); ax1.grid(True)
tb3m.plot(ax=ax2, lw=1.2, title="3-Month T-Bill Yield",   ylabel="%", xlabel="")
ax2.fill_between(gdp.index, 0, 1, where=is_rec, transform=ax2.get_xaxis_transform(),
                 color="red", alpha=0.3); ax2.grid(True)
ibor3m.plot(ax=ax3, lw=1.2, title="3-Month IBOR",         ylabel="%", xlabel="Date")
ax3.fill_between(gdp.index, 0, 1, where=is_rec, transform=ax3.get_xaxis_transform(),
                 color="red", alpha=0.3); ax3.grid(True)
xmin = min(s.index.min() for s in [gilt10, tb3m, ibor3m])
xmax = max(s.index.max() for s in [gilt10, tb3m, ibor3m])
for ax in (ax1, ax2, ax3): ax.set_xlim(xmin, xmax)
plt.tight_layout(); plt.show()


# === 3) Splice T-Bill & IBOR into a combined short rate; plots ===
idx = tb3m.index.union(ibor3m.index).sort_values()
tb = tb3m.reindex(idx)
ib = ibor3m.reindex(idx)

overlap = tb.notna() & ib.notna()
print(f"\nOverlap period: {tb[overlap].index.min().date()} to {tb[overlap].index.max().date()}")
print(f"Number of overlapping months: {int(overlap.sum())}")

# Scatter vs time (two series)
fig, ax = plt.subplots(figsize=(8,5))
ax.scatter(tb.index[overlap], tb[overlap], alpha=0.7, label="T-Bill")
ax.scatter(ib.index[overlap], ib[overlap], alpha=0.7, label="IBOR")
ax.set_title("3-Month Yields (scatter vs time)")
ax.set_xlabel("Date"); ax.set_ylabel("%")
ax.grid(True); ax.legend(); plt.tight_layout(); plt.show()

# Overlap regression: tbill = a + b*ibor
X = sm.add_constant(ib[overlap])
model = sm.OLS(tb[overlap], X).fit()
print(model.summary())
print("\nSplice regression coefficients:")
print(f"Intercept a: {model.params['const']:.4f}")
print(f"IBOR coef b: {model.params['ibor3m']:.4f}")

# Predict tbill where missing, using IBOR
tb_pred = model.predict(sm.add_constant(ib[~overlap]))
tb_combined = tb.copy()
tb_combined.update(tb_pred)
tb_combined.name = "tb3m_combined"

# Plot combined short rate
fig, ax = plt.subplots(figsize=(10,4))
tb_combined.plot(ax=ax, lw=1.2, title="3-Month T-Bill Yield (combined)", ylabel="%", xlabel="Date")
ax.grid(True); plt.tight_layout(); plt.show()

# Re-plot yields with combined short rate + recession shading
is_rec = gdp["recession_q"].astype(bool)
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
plt.subplots_adjust(hspace=0.3)
gilt10.plot(ax=ax1, lw=1.2, title="10-Year Gilt Yield", ylabel="%", xlabel="")
ax1.fill_between(gdp.index, 0, 1, where=is_rec, transform=ax1.get_xaxis_transform(),
                 color="red", alpha=0.3); ax1.grid(True)
tb_combined.plot(ax=ax2, lw=1.2, title="3-Month T-Bill (combined)", ylabel="%", xlabel="")
ax2.fill_between(gdp.index, 0, 1, where=is_rec, transform=ax2.get_xaxis_transform(),
                 color="red", alpha=0.3); ax2.grid(True)
ibor3m.plot(ax=ax3, lw=1.2, title="3-Month IBOR", ylabel="%", xlabel="Date")
ax3.fill_between(gdp.index, 0, 1, where=is_rec, transform=ax3.get_xaxis_transform(),
                 color="red", alpha=0.3); ax3.grid(True)
xmin = min(s.index.min() for s in [gilt10, tb_combined, ibor3m])
xmax = max(s.index.max() for s in [gilt10, tb_combined, ibor3m])
for ax in (ax1, ax2, ax3): ax.set_xlim(xmin, xmax)
plt.tight_layout(); plt.show()


# === 4) Single feature: Yield spread, with plot ===
spread = gilt10.reindex(tb_combined.index).astype(float) - tb_combined
fig, ax = plt.subplots(figsize=(10,4))
spread.plot(ax=ax, lw=1.2, title="Yield Spread (10y − 3m combined)", ylabel="%", xlabel="Date")
ax.fill_between(gdp.index, 0, 1, where=is_rec, transform=ax.get_xaxis_transform(),
                color='red', alpha=0.3)
ax.axhline(0, color='black', lw=0.8, ls='--')
ax.grid(True); plt.tight_layout(); plt.show()


# === 5) LABELS (EXPLAINED) =========================================
# "Recession indicator" (quarterly): a quarter is a recession quarter if QoQ growth < 0 and either the previous
# quarter or the next quarter is also QoQ < 0 (i.e., part of a run of >=2 negatives).
# "Forecast indicator" (monthly, the model's target): recession_start_next12m = 1 if a recession STARTS
# at any point in the next 12 months; else 0. A recession STARTS in the first quarter of a run of
# >=2 consecutive negative QoQ quarters.

# Find start-of-recession quarters (first negative of a >=2 run)
start_q = (neg & neg.shift(-1) & (~neg.shift(1).fillna(False))).astype(int)
gdp["recession_start_q"] = start_q
print("\nRecession START quarter counts:\n", gdp["recession_start_q"].value_counts())

# Map quarterly start flags to monthly index of the feature
H = 12  # forecast horizon in months
start_q_pi = start_q.copy()
start_q_pi.index = start_q_pi.index.to_period('Q')
start_m = pd.Series(
    start_q_pi.reindex(tb_combined.index.to_period('Q')).values,
    index=tb_combined.index
).fillna(0).astype(int)   # 1 in months of the quarter where the recession STARTS

# Monthly "recession START within next 12 months" label (shift by 1 month to avoid peeking)
yA = (start_m.shift(-1).rolling(window=H, min_periods=1).max()).astype(float)
yA.iloc[-H:] = pd.NA  # can't know the future for the last H months
yA = yA.rename("recession_start_next12m")
print("\nLabel counts (including NA for last 12 months):\n", yA.value_counts(dropna=False))

# Final modeling frame (one feature + label)
dfA = pd.DataFrame({"spread": spread}).join(yA).dropna()
print("\nModeling frame (head & tail):\n", dfA.head(), "\n", dfA.tail())


# === 6) ONE logistic model with TimeSeriesSplit(gap=12) ===
X = dfA[["spread"]]
y = dfA["recession_start_next12m"].astype(int)

# Split config. gap=12 matches the 12-month look-ahead in the label.
N_SPLITS = 6
TEST_LEN = 36  # months per fold’s test block
GAP = 12

pipe = make_pipeline(
    StandardScaler(with_mean=True),
    LogisticRegression(class_weight="balanced", solver="lbfgs", max_iter=1000, random_state=0)
)

tscv = TimeSeriesSplit(n_splits=N_SPLITS, test_size=TEST_LEN, gap=GAP)

# Collect out-of-fold predictions & per-fold metrics
oof = pd.Series(index=X.index, dtype=float)
rows = []

for fold, (tr, te) in enumerate(tscv.split(X), start=1):
    Xtr, ytr = X.iloc[tr], y.iloc[tr]
    Xte, yte = X.iloc[te], y.iloc[te]

    # Optional: use a rolling train window (e.g., last 240 months)
    # if len(Xtr) > 240:
    #     Xtr, ytr = Xtr.iloc[-240:], ytr.iloc[-240:]

    pipe.fit(Xtr, ytr)
    p = pipe.predict_proba(Xte)[:, 1]
    oof.iloc[te] = p

    # Metrics (handle no-positive folds cleanly)
    try:
        roc = roc_auc_score(yte, p)
    except ValueError:
        roc = float("nan")
    pr = average_precision_score(yte, p) if yte.sum() > 0 else 0.0
    br = brier_score_loss(yte, p)
    base = yte.mean()

    rows.append({
        "fold": fold,
        "train_start": Xtr.index[0], "train_end": Xtr.index[-1],
        "test_start":  Xte.index[0], "test_end":  Xte.index[-1],
        "roc_auc": roc, "pr_auc": pr, "brier": br, "base": base
    })

fold_tbl = pd.DataFrame(rows)
mask = oof.notna()
oof_roc  = roc_auc_score(y[mask], oof[mask])
oof_pr   = average_precision_score(y[mask], oof[mask])
oof_br   = brier_score_loss(y[mask], oof[mask])
oof_base = y[mask].mean()

print("\n=== TimeSeriesSplit OOF metrics (pooled across all test folds) ===")
print(f"oof_roc_auc: {oof_roc:.6f}")
print(f"oof_pr_auc:  {oof_pr:.6f}")
print(f"oof_brier:   {oof_br:.6f}")
print(f"oof_base:    {oof_base:.6f}")

print("\n=== Per-fold metrics ===")
print(fold_tbl.to_string(index=False))

# Refit on all labeled data (simple final fit) and print latest forecast probability
final_pipe = make_pipeline(
    StandardScaler(with_mean=True),
    LogisticRegression(class_weight="balanced", solver="lbfgs", max_iter=1000, random_state=0)
)
final_pipe.fit(X, y)

latest_month = spread.index.max()
latest_prob  = float(final_pipe.predict_proba(pd.DataFrame({"spread":[spread.iloc[-1]]}, index=[latest_month]))[:,1])

print("\n=== Latest forecast ===")
print(f"As of {latest_month.date()}, Pr(recession START within next 12 months) = {latest_prob:.3%}")

    # === Continuous walk-forward backtest (one prob per month, embargo = 12) ===
# For each month t, train on all data ≤ t-12m (so training labels don't share the future with t),
# then predict Pr(recession START within next 12m) for month t. This yields a continuous line.

H = 12  # embargo = forecast horizon
data_all = pd.concat([X, y], axis=1).dropna()  # only rows with known labels
idx_all = data_all.index

# Optional warm-up: require at least N training months before making predictions
MIN_TRAIN = 120  # e.g., 10 years
p_walk = pd.Series(index=idx_all, dtype=float, name="walkforward_prob")

for t in idx_all:
    # end training at t - 12 months (end of month), so no shared future with label window for t
    train_end = t - pd.offsets.MonthEnd(H)
    train_mask = idx_all <= train_end
    n_tr = int(train_mask.sum())
    if n_tr < MIN_TRAIN:
        continue  # not enough history yet

    Xtr, ytr = X.loc[train_mask], y.loc[train_mask]
    # fresh model each step (simple & transparent)
    wf_pipe = make_pipeline(
        StandardScaler(with_mean=True),
        LogisticRegression(class_weight="balanced", solver="lbfgs", max_iter=1000, random_state=0)
    )
    wf_pipe.fit(Xtr, ytr)
    p_walk.loc[t] = wf_pipe.predict_proba(X.loc[[t]])[:, 1][0]

print("\nWalk-forward coverage:")
print(f"predictions from {p_walk.first_valid_index().date()} to {p_walk.last_valid_index().date()}  "
      f"({p_walk.notna().sum()} months)")

# === Recession shading (monthly, aligned to plot index) ===
# If you already have a monthly recession series, reuse it; otherwise build it:
rec_m_q = gdp["recession_q"].copy()
rec_m_q.index = rec_m_q.index.to_period("Q")
rec_m = pd.Series(
    rec_m_q.reindex(spread.index.to_period("Q")).values,
    index=spread.index
).fillna(0).astype(int)  # 1 during recession months, else 0
is_rec_m = rec_m.astype(bool)

# === Plot: spread + walk-forward prob + recession shading ===
fig, ax1 = plt.subplots(figsize=(10,4))
spread.plot(ax=ax1, lw=1.0, label="10y–3m spread (%)")
ax1.axhline(0, color="black", lw=0.8, ls="--")
# Recession shading (full height)
ax1.fill_between(
    spread.index, 0, 1, where=is_rec_m.reindex(spread.index).fillna(False),
    transform=ax1.get_xaxis_transform(), color="red", alpha=0.3
)
ax1.set_ylabel("%"); ax1.set_title("Yield Spread & Walk-forward Probability (with recessions)")

ax2 = ax1.twinx()
p_walk.plot(ax=ax2, lw=1.2, alpha=0.9, color='red', label="Walk-forward Pr(start ≤ 12m)")
ax2.set_ylabel("Probability")

# Legend
l1, lab1 = ax1.get_legend_handles_labels()
l2, lab2 = ax2.get_legend_handles_labels()
ax1.legend(l1+l2, lab1+lab2, loc="upper left")
plt.tight_layout(); plt.show()




# Sign of the coefficient (should be negative)
lr = final_pipe.named_steps['logisticregression']
sc = final_pipe.named_steps['standardscaler']
print("Coef (β):", lr.coef_[0,0], "Intercept (α):", lr.intercept_[0])

# Correlation (should be strongly negative)
print("Corr(spread, OOF prob):", pd.Series(oof).corr(dfA['spread'].loc[oof.index]))

# Visual: overlay inverted, standardized spread against probs
z = (dfA['spread'] - sc.mean_[0]) / sc.scale_[0]
ax = p_walk.plot(lw=1.2, label="Pr(recession start ≤ 12m)")
(-z).rename("−z(spread)").rolling(3).mean().plot(ax=ax, lw=1.0, alpha=0.7)
plt.legend(); plt.show()
