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

import time
_t0 = time.time()
def log(msg):
    t = time.time() - _t0
    print(f"[{t:6.1f}s] {msg}")

log("Loading model...")
artifacts = joblib.load("submissions/my_team/weights.joblib")
model = artifacts.get("regressor", artifacts.get("model"))

log("Reading data...")
rides = pd.read_csv("dataset/train_set.csv", low_memory=False)
rides["hour_ts"] = pd.to_datetime(rides["hour_ts"])
rides["hour"] = rides["hour_ts"].dt.hour

pub = pd.read_csv("dataset/public_validation_targets.csv", low_memory=False)
pri = pd.read_csv("dataset/private_labels.csv", low_memory=False)

import sys
sys.path.insert(0, str(Path("submissions/my_team").resolve()))
from model import BikeDemandModel
bike_model = BikeDemandModel()
bike_model.load_artifacts(artifacts)
log("Running predictions...")
preds = bike_model.predict(pub)

from model import FEATURE_COLUMNS
feature_names = FEATURE_COLUMNS

pri["prediction"] = preds
pri["prediction_clipped"] = np.maximum(0, pri["prediction"])
pri["abs_error"] = np.abs(pri["demand"] - pri["prediction_clipped"])
pub["hour"] = pd.to_datetime(pub["hour_ts"]).dt.hour

log("Plotting...")
fig, axes = plt.subplots(2, 2, figsize=(12, 8))

ax = axes[0, 0]
city_counts = rides.groupby(["city", "hour_ts", "hour"]).size().reset_index(name="cnt")
for city in ["city 1", "city 2", "city 3"]:
    cd = city_counts[city_counts["city"] == city].groupby("hour")["cnt"].mean()
    ax.plot(cd.index, cd.values, label=city, marker="o", markersize=3)
ax.set_xlabel("Hour of day")
ax.set_ylabel("Mean demand")
ax.set_title("Hourly demand patterns by city")
ax.legend()
ax.grid(alpha=0.3)

ax = axes[0, 1]
city_mean = city_counts.groupby("city")["cnt"].mean().reset_index()
ax.bar(city_mean["city"], city_mean["cnt"], color=["#4C72B0", "#DD8452", "#55A868"])
ax.set_xlabel("City")
ax.set_ylabel("Mean demand per station-hour")
ax.set_title("Average demand by city")
ax.grid(alpha=0.3)
for i, v in enumerate(city_mean["cnt"]):
    ax.text(i, v + 0.02, f"{v:.2f}", ha="center", fontsize=9)

ax = axes[1, 0]
fnames = model.feature_name()
# Replace Column_N with actual feature names
fname_map = {f"Column_{i}": fn for i, fn in enumerate(feature_names)}
fi = pd.DataFrame({
    "feature": [fname_map.get(f, f) for f in fnames],
    "importance": model.feature_importance("gain"),
})
fi = fi.sort_values("importance", ascending=True).tail(20)
ax.barh(fi["feature"], fi["importance"], color="#4C72B0")
ax.set_xlabel("Feature importance (gain)")
ax.set_title("Top 20 feature importances")
ax.grid(alpha=0.3)

ax = axes[1, 1]
sample = pri.sample(min(10000, len(pri)), random_state=42)
ax.scatter(sample["demand"], sample["prediction_clipped"], alpha=0.1, s=5, c="#4C72B0")
lim = max(sample["demand"].max(), sample["prediction_clipped"].max())
ax.plot([0, lim], [0, lim], "r--", alpha=0.5, lw=1)
ax.set_xlabel("True demand")
ax.set_ylabel("Predicted demand")
ax.set_title(f"Predictions vs actual (n={len(sample)})")
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig(OUT / "model_analysis.png", bbox_inches="tight")
print(f"Saved {OUT / 'model_analysis.png'}")

fig, ax = plt.subplots(figsize=(10, 5))
pri["hour"] = pub["hour"].values
hourly_mae = pri.groupby("hour")["abs_error"].mean()
ax.bar(hourly_mae.index, hourly_mae.values, color="#DD8452")
ax.set_xlabel("Hour of day")
ax.set_ylabel("MAE")
ax.set_title("MAE by hour of day")
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "mae_by_hour.png", bbox_inches="tight")
print(f"Saved {OUT / 'mae_by_hour.png'}")

print("\nTop 10 features by gain:")
for _, r in fi.tail(10).iterrows():
    print(f"  {r['feature'][:38]:38s} {r['importance']:.2f}")

print(f"\nOverall MAE: {pri['abs_error'].mean():.4f}")
print(f"City 1 MAE: {pri[pri['city']=='city 1']['abs_error'].mean():.4f}")
print(f"City 2 MAE: {pri[pri['city']=='city 2']['abs_error'].mean():.4f}")
