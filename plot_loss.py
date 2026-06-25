import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import joblib
from pathlib import Path

plt.rcParams.update({"figure.dpi": 150, "font.size": 11})
OUT = Path("figures")
OUT.mkdir(exist_ok=True)

artifacts = joblib.load("submissions/my_team/weights.joblib")
result = artifacts["evals_result"]

fig, ax = plt.subplots(figsize=(10, 5))
valid_mae = result["valid_0"]["l1"]
iters = np.arange(1, len(valid_mae) + 1)
ax.plot(iters, valid_mae, label="Validation MAE", color="#4C72B0")
best_idx = np.argmin(valid_mae)
ax.axhline(valid_mae[best_idx], color="red", ls="--", alpha=0.5,
           label=f"Best: {valid_mae[best_idx]:.4f} at round {best_idx+1}")
ax.set_xlabel("Boosting round")
ax.set_ylabel("MAE")
ax.set_title("Validation loss curve during Tweedie training")
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT / "loss_curves.png", bbox_inches="tight")
print(f"Saved {OUT / 'loss_curves.png'}")
print(f"Best valid MAE: {valid_mae[best_idx]:.4f} at round {best_idx+1}")
