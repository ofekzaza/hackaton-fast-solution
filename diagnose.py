import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

plt.rcParams.update({"figure.dpi": 150, "font.size": 11})

OUT = Path("figures")
OUT.mkdir(exist_ok=True)

import sys
sys.path.insert(0, str(Path("submissions/my_team").resolve()))
from model import BikeDemandModel

artifacts = joblib.load("submissions/my_team/weights.joblib")
model = artifacts.get("regressor", artifacts.get("model"))

pub = pd.read_csv("dataset/public_validation_targets.csv", low_memory=False)
pri = pd.read_csv("dataset/private_labels.csv", low_memory=False)

bike_model = BikeDemandModel()
bike_model.load_artifacts(artifacts)
preds = bike_model.predict(pub)

pri["prediction"] = preds
pri["prediction_clipped"] = np.maximum(0, pri["prediction"])
pri["abs_error"] = np.abs(pri["demand"] - pri["prediction_clipped"])
pri["signed_error"] = pri["demand"] - pri["prediction_clipped"]
pri["squared_error"] = (pri["demand"] - pri["prediction_clipped"]) ** 2

pub["hour"] = pd.to_datetime(pub["hour_ts"]).dt.hour
pub["weekday_name"] = pd.to_datetime(pub["date"]).dt.day_name()
pri["hour"] = pub["hour"].values
pri["weekday_name"] = pub["weekday_name"].values

print("=" * 60)
print("OVERALL METRICS")
print("=" * 60)
for city in ["city 1", "city 2"]:
    mask = pri["city"] == city
    sub = pri[mask]
    print(f"\n{city}:")
    print(f"  MAE:                {sub['abs_error'].mean():.4f}")
    print(f"  RMSE:               {np.sqrt(sub['squared_error'].mean()):.4f}")
    print(f"  Mean true demand:   {sub['demand'].mean():.4f}")
    print(f"  Mean prediction:    {sub['prediction_clipped'].mean():.4f}")
    print(f"  Bias (pred-true):   {sub['prediction_clipped'].mean() - sub['demand'].mean():.4f}")
    print(f"  Median abs error:   {sub['abs_error'].median():.4f}")
    print(f"  N rows:             {len(sub)}")

overall = pri
print(f"\nOVERALL:")
print(f"  MAE:                {overall['abs_error'].mean():.4f}")
print(f"  RMSE:               {np.sqrt(overall['squared_error'].mean()):.4f}")
print(f"  Mean true demand:   {overall['demand'].mean():.4f}")
print(f"  Mean prediction:    {overall['prediction_clipped'].mean():.4f}")
print(f"  Bias:               {overall['prediction_clipped'].mean() - overall['demand'].mean():.4f}")
print(f"  Median abs error:   {overall['abs_error'].median():.4f}")
print(f"  N rows:             {len(overall)}")
print()

# === FIGURE 1: Error distribution ===
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

ax = axes[0]
ax.hist(pri["abs_error"], bins=80, range=(0, 20), color="#4C72B0", ec="white", lw=0.3)
ax.axvline(pri["abs_error"].mean(), color="red", ls="--", lw=1.5, label=f'Mean={pri["abs_error"].mean():.2f}')
ax.axvline(pri["abs_error"].median(), color="orange", ls="--", lw=1.5, label=f'Median={pri["abs_error"].median():.2f}')
ax.set_xlabel("Absolute error")
ax.set_ylabel("Count")
ax.set_title("Distribution of absolute errors")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1]
ax.hist(pri["signed_error"], bins=80, range=(-10, 10), color="#55A868", ec="white", lw=0.3)
ax.axvline(0, color="red", ls="-", lw=0.8)
ax.axvline(pri["signed_error"].mean(), color="red", ls="--", lw=1.5,
           label=f'Mean={pri["signed_error"].mean():.2f}')
ax.set_xlabel("Signed error (true - pred)")
ax.set_ylabel("Count")
ax.set_title("Distribution of signed errors (bias)")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[2]
ax.hist(pri["demand"], bins=60, range=(0, 30), color="#DD8452", alpha=0.7, ec="white", lw=0.3,
        label=f"True (mean={pri['demand'].mean():.2f})")
ax.hist(pri["prediction_clipped"], bins=60, range=(0, 30), color="#4C72B0", alpha=0.5, ec="white", lw=0.3,
        label=f"Pred (mean={pri['prediction_clipped'].mean():.2f})")
ax.set_xlabel("Demand")
ax.set_ylabel("Count")
ax.set_title("Distribution: true vs predicted demand")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUT / "error_distribution.png", bbox_inches="tight")
print(f"Saved {OUT / 'error_distribution.png'}")

# === FIGURE 2: Calibration curves (mean prediction vs mean true by bin) ===
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

pri["pred_bin"] = pd.cut(pri["prediction_clipped"], bins=np.linspace(0, 25, 26), labels=False, include_lowest=True)
calib = pri.groupby("pred_bin").agg(
    mean_pred=("prediction_clipped", "mean"),
    mean_true=("demand", "mean"),
    count=("demand", "size"),
).reset_index()
calib = calib[calib["count"] > 50]

ax = axes[0]
ax.plot(calib["mean_pred"], calib["mean_true"], "o-", color="#4C72B0", ms=4)
ax.plot([0, 20], [0, 20], "r--", alpha=0.5, lw=1)
ax.set_xlabel("Mean predicted demand")
ax.set_ylabel("Mean true demand")
ax.set_title("Calibration curve (binned by prediction)")
ax.grid(alpha=0.3)

# Calibration by hour
ax = axes[1]
hourly = pri.groupby("hour").agg(
    mae=("abs_error", "mean"),
    rmse=("squared_error", lambda x: np.sqrt(x.mean())),
    mean_true=("demand", "mean"),
    mean_pred=("prediction_clipped", "mean"),
    n=("demand", "size"),
).reset_index()
ax.plot(hourly["hour"], hourly["mae"], "o-", label="MAE", color="#DD8452")
ax.plot(hourly["hour"], hourly["rmse"], "s--", label="RMSE", color="#4C72B0")
ax.set_xlabel("Hour of day")
ax.set_ylabel("Error")
ax.set_title("MAE and RMSE by hour")
ax.legend()
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUT / "calibration_curves.png", bbox_inches="tight")
print(f"Saved {OUT / 'calibration_curves.png'}")

# === FIGURE 3: Per-city and per-weekday breakdown ===
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ax = axes[0]
city_stats = pri.groupby("city").agg(
    mae=("abs_error", "mean"),
    bias=("signed_error", "mean"),
    mean_true=("demand", "mean"),
    mean_pred=("prediction_clipped", "mean"),
).reset_index()
x = np.arange(len(city_stats))
w = 0.25
ax.bar(x - w, city_stats["mae"], w, label="MAE", color="#DD8452")
ax.bar(x, city_stats["mean_true"], w, label="Mean true", color="#4C72B0")
ax.bar(x + w, city_stats["mean_pred"], w, label="Mean pred", color="#55A868")
ax.set_xticks(x)
ax.set_xticklabels(city_stats["city"])
ax.set_title("Per-city metrics")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

ax = axes[1]
wd_stats = pri.groupby("weekday_name").agg(
    mae=("abs_error", "mean"),
    mean_true=("demand", "mean"),
    mean_pred=("prediction_clipped", "mean"),
    n=("demand", "size"),
).reset_index()
weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
wd_stats["weekday_order"] = wd_stats["weekday_name"].map({d: i for i, d in enumerate(weekday_order)})
wd_stats = wd_stats.sort_values("weekday_order")
ax.plot(wd_stats["weekday_name"], wd_stats["mae"], "o-", label="MAE", color="#DD8452")
ax.plot(wd_stats["weekday_name"], wd_stats["mean_true"], "s--", label="Mean true", color="#4C72B0")
ax.plot(wd_stats["weekday_name"], wd_stats["mean_pred"], "^--", label="Mean pred", color="#55A868")
ax.set_xticklabels(wd_stats["weekday_name"], rotation=30, ha="right")
ax.set_title("Per-weekday metrics")
ax.legend(fontsize=8)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUT / "city_weekday_breakdown.png", bbox_inches="tight")
print(f"Saved {OUT / 'city_weekday_breakdown.png'}")

# === FIGURE 4: Zoomed scatter for low demand ===
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

sample = pri.sample(min(10000, len(pri)), random_state=42)

ax = axes[0]
mask = sample["demand"] <= 10
ax.scatter(sample.loc[mask, "demand"], sample.loc[mask, "prediction_clipped"],
           alpha=0.15, s=4, c="#4C72B0")
ax.plot([0, 10], [0, 10], "r--", alpha=0.5, lw=1)
ax.set_xlabel("True demand")
ax.set_ylabel("Predicted demand")
ax.set_title("Zoom: demand ≤ 10 rides")
ax.grid(alpha=0.3)

ax = axes[1]
# Bin true demand and show mean + std of predictions
true_bins = np.arange(0, 31, 1)
pri["true_bin"] = pd.cut(pri["demand"], bins=np.arange(-0.5, 31.5, 1), labels=np.arange(0, 31))
bin_stats = pri.groupby("true_bin", observed=True).agg(
    mean_pred=("prediction_clipped", "mean"),
    std_pred=("prediction_clipped", "std"),
    count=("prediction_clipped", "size"),
).reset_index()
bin_stats = bin_stats[bin_stats["count"] > 20]
ax.errorbar(bin_stats["true_bin"].astype(int), bin_stats["mean_pred"],
            yerr=bin_stats["std_pred"], fmt="o", capsize=2, ms=3, color="#4C72B0", alpha=0.7)
ax.plot([0, 30], [0, 30], "r--", alpha=0.5, lw=1)
ax.set_xlabel("True demand")
ax.set_ylabel("Mean predicted demand ± 1σ")
ax.set_title("Calibration with uncertainty (binned by true demand)")
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUT / "zoomed_calibration.png", bbox_inches="tight")
print(f"Saved {OUT / 'zoomed_calibration.png'}")

# === Figure 5: Q-Q plot ===
fig, ax = plt.subplots(figsize=(6, 5))
quantiles = np.linspace(0.001, 0.999, 200)
true_q = np.quantile(pri["demand"].values, quantiles)
pred_q = np.quantile(pri["prediction_clipped"].values, quantiles)
ax.plot(true_q, pred_q, "o", ms=1, color="#4C72B0", alpha=0.5)
ax.plot([0, 30], [0, 30], "r--", alpha=0.5, lw=1)
ax.set_xlabel("True demand quantiles")
ax.set_ylabel("Predicted demand quantiles")
ax.set_title("Q-Q plot of predicted vs true demand")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "qq_plot.png", bbox_inches="tight")
print(f"Saved {OUT / 'qq_plot.png'}")

# === Output key stats ===
print()
print("=" * 60)
print("DETAILED BREAKDOWNS")
print("=" * 60)

print("\nMAE by true demand bin:")
pri["true_bin_label"] = pd.cut(pri["demand"], bins=[-1, 0, 1, 2, 3, 5, 10, 20, 200],
                                labels=["0", "1", "2", "3", "4-5", "6-10", "11-20", ">20"])
for label, sub in pri.groupby("true_bin_label", observed=True):
    print(f"  true={label:6s}  n={len(sub):6d}  MAE={sub['abs_error'].mean():.4f}  "
          f"mean_true={sub['demand'].mean():.4f}  mean_pred={sub['prediction_clipped'].mean():.4f}")

print("\nMAE by hour (top/bottom 3):")
hourly_sorted = hourly.sort_values("mae")
for _, r in hourly_sorted.head(3).iterrows():
    print(f"  Best hour {int(r['hour']):2d}: MAE={r['mae']:.4f}  n={int(r['n']):6d}")
for _, r in hourly_sorted.tail(3).iterrows():
    print(f"  Worst hour {int(r['hour']):2d}: MAE={r['mae']:.4f}  n={int(r['n']):6d}")

pct_zero_true = (pri["demand"] == 0).mean() * 100
pct_zero_pred = (pri["prediction_clipped"] == 0).mean() * 100
print(f"\n% zero true demand:  {pct_zero_true:.1f}%")
print(f"% zero predictions:  {pct_zero_pred:.1f}%")

zero_true = pri[pri["demand"] == 0]
print(f"MAE on zero-demand rows only: {zero_true['abs_error'].mean():.4f} "
      f"(mean pred = {zero_true['prediction_clipped'].mean():.4f})")

non_zero = pri[pri["demand"] > 0]
print(f"MAE on non-zero-demand rows:  {non_zero['abs_error'].mean():.4f}")

# Prediction breakdown by weekend vs weekday
print(f"\nWeekday MAE:  {pri[pri['hour'].between(0,4) | pri['hour'].between(22,23)]['abs_error'].mean():.4f}")
print(f"Daytime MAE:  {pri[pri['hour'].between(6,21)]['abs_error'].mean():.4f}")
