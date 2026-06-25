from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd

from model import FEATURE_COLUMNS, make_features_from_rides

DATA_ROOT = Path("../../dataset")
TRAIN_CSV = DATA_ROOT / "train_set.csv"
OUTPUT_WEIGHTS = "weights.joblib"


def main() -> None:
    print(f"Reading training data from {TRAIN_CSV}...")
    rides = pd.read_csv(TRAIN_CSV, low_memory=False)
    print(f"Read {len(rides):,} ride rows")

    print("Aggregating to station-hour demand and creating features...")
    train_df, artifacts = make_features_from_rides(rides)
    print(f"Created {len(train_df):,} station-hour training examples")
    d = train_df["demand"]
    print(
        f"Demand stats: mean={d.mean():.3f}, "
        f"median={np.median(d):.3f}, "
        f"max={d.max():.0f}"
    )
    zero_frac = (d == 0).mean()
    
    print(f"Zero-demand fraction: {zero_frac:.1%}")

    train_df = train_df.sort_values("hour_ts").reset_index(drop=True)

    # we have about several weeks of data, so use the last week for testing
    cutoff = pd.to_datetime(train_df["hour_ts"]).max() - pd.Timedelta("7D")
    train = train_df[pd.to_datetime(train_df["hour_ts"]) < cutoff].copy()
    valid = train_df[pd.to_datetime(train_df["hour_ts"]) >= cutoff].copy()

    print(f"Train: {len(train):,} rows  (up to {cutoff.date()}")
    print(f"Valid: {len(valid):,} rows  ({cutoff.date()} onwards)")

    feature_cols = [c for c in FEATURE_COLUMNS if c in train.columns]
    X_train = train[feature_cols].values
    y_train = train["demand"].values
    X_valid = valid[feature_cols].values
    y_valid = valid["demand"].values

    print(f"\nTraining Tweedie regressor on full data ({len(X_train):,} rows)...")

    params = {
        "objective": "tweedie",
        "tweedie_variance_power": 1.3,
        "metric": "mae",
        "boosting_type": "gbdt",
        "num_leaves": 255,
        "learning_rate": 0.02,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": -1,
        "num_threads": -1,
        "min_data_in_leaf": 20,
        "lambda_l1": 0.5,
        "lambda_l2": 0.5,
        "min_gain_to_split": 0.0,
        "random_state": 42,
    }

    evals_result = {}
    model = lgb.train(
        params,
        lgb.Dataset(X_train, y_train),
        num_boost_round=8000,
        valid_sets=[lgb.Dataset(X_valid, y_valid)],
        callbacks=[
            lgb.callback.early_stopping(100),
            lgb.callback.log_evaluation(200),
            lgb.callback.record_evaluation(evals_result),
        ],
    )
    print(f"  Trained {model.num_trees()} trees")

    val_preds = model.predict(X_valid)
    val_mae = np.abs(val_preds - y_valid).mean()
    print(f"\nValidation MAE: {val_mae:.4f}")

    artifacts["model"] = model
    artifacts["feature_columns"] = feature_cols
    artifacts["evals_result"] = evals_result

    joblib.dump(artifacts, OUTPUT_WEIGHTS)
    print(
        f"\nSaved {OUTPUT_WEIGHTS} ({Path(OUTPUT_WEIGHTS).stat().st_size / 1024 / 1024:.1f} MB)"
    )


if __name__ == "__main__":
    main()
