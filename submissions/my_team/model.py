import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

FEATURE_COLUMNS = [
    "city",
    "hour",
    "weekday",
    "working_day",
    "comfort_score", # how good is temperature/humidity/sky
    "outdoor_pleasantness", # how bad is rain/snow/wind
    "recreational_accessibility", # combined park area + bike-lane length
    "office_poi_count_1000m",
    "retail_poi_count_1000m",
    "restaurant_cafe_count_500m",
    "transit_stop_count_500m",
    "distance_to_nearest_rail_station",
    "distance_to_city_center",
  
    "station_mean_demand",
    "station_median_demand",
    "station_max_demand",
    "city_hour_mean_demand",
    "city_hour_median_demand",
    "city_hour_max_demand",
    "city_weekday_mean_demand",
    "city_mean_demand",
    "station_hour_mean_demand",
    "station_hour_median_demand",
    "station_hour_max_demand",
    "station_hour_std_demand",
    "is_morning_rush",
    "is_evening_rush",
    "rush_x_station_hour_mean",
    # lag fetures
    "lag_1d_demand",
    "lag_7d_demand",
    "lag_14d_demand",
    "lag_21d_demand",
    "lag_rolling_mean",
]

grp_cols = ["city", "start_station_id", "hour_ts"]

CATEGORICAL_COLUMNS = ["city"]


def _create_temporal_features(df):
    ts = pd.to_datetime(df["hour_ts"])
    df["hour_ts"] = ts

    if "hour" not in df.columns:
        df["hour"] = ts.dt.hour

    if "weekday" not in df.columns:
        df["weekday"] = ts.dt.weekday

    return df


def _encode_categoricals(df, encoders=None):
    out = df.copy()
    out["city"] = out["city"].astype(str)
    out["start_station_id"] = _normalize_station_id(out["start_station_id"])
    if encoders is not None:
        for col in CATEGORICAL_COLUMNS:
            le = encoders[col]
            mapper = {v: i for i, v in enumerate(le.classes_)}
            out[col] = out[col].map(mapper).fillna(0).astype(int)
    else:
        encoders = {}
        for col in CATEGORICAL_COLUMNS:
            le = LabelEncoder()
            out[col] = le.fit_transform(out[col].values)
            encoders[col] = le
    return out, encoders


def _merge_agg_features(df, artifacts):
    if "station_stats" in artifacts:
        df = df.merge(
            artifacts["station_stats"], on=["city", "start_station_id"], how="left"
        )
    
    if "city_hour_stats" in artifacts:
        df = df.merge(artifacts["city_hour_stats"], on=["city", "hour"], how="left")
    
    if "city_weekday_stats" in artifacts:
        df = df.merge(
            artifacts["city_weekday_stats"], on=["city", "weekday"], how="left"
        )
    
    if "city_mean_stats" in artifacts:
        df = df.merge(artifacts["city_mean_stats"], on=["city"], how="left")
    
    if "station_hour_stats" in artifacts:
        df = df.merge(
            artifacts["station_hour_stats"],
            on=grp_cols,
            how="left",
        )
    
    for col in FEATURE_COLUMNS:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)
    
    return df


def _merge_lag_features(df, demand_history):
    """
    For each row, look up how much demand there was at the same station
    1 day ago, 7 days ago, and 14 days ago.

    The demand_history DataFrame (city, start_station_id, hour_ts, demand)
    is stored in artifacts at training time so it's available at predict time.
    """
    demand_history = demand_history.copy()
    demand_history["hour_ts"] = pd.to_datetime(demand_history["hour_ts"])
    demand_history["city"] = demand_history["city"].astype(str)
    demand_history["start_station_id"] = _normalize_station_id(
        demand_history["start_station_id"]
    )

    df = df.copy()
    df["hour_ts"] = pd.to_datetime(df["hour_ts"])

    # Deduplicate after normalisation — defensive guard against any residual
    # duplicate keys that would cause a left-merge to explode row count.
    demand_history = demand_history.groupby(
        grp_cols, as_index=False
    )["demand"].sum()

    for col, delta in [
        ("lag_1d_demand", pd.Timedelta("1D")),
        ("lag_7d_demand", pd.Timedelta("7D")),
        ("lag_14d_demand", pd.Timedelta("14D")),
        ("lag_21d_demand", pd.Timedelta("21D")),
    ]:
        # Shift the history forward so that row at time T gets the value from T-delta
        shifted = demand_history[
            grp_cols + ["demand"]
        ].copy()
        shifted["hour_ts"] = shifted["hour_ts"] + delta
        shifted = shifted.rename(columns={"demand": col})
        df = df.merge(shifted, on=grp_cols, how="left")

    # avg 3 wks
    weekly = [
        c
        for c in ["lag_7d_demand", "lag_14d_demand", "lag_21d_demand"]
        if c in df.columns
    ]
    df["lag_rolling_mean"] = df[weekly].mean(axis=1) if weekly else 0.0

    for col in [
        "lag_1d_demand",
        "lag_7d_demand",
        "lag_14d_demand",
        "lag_21d_demand",
        "lag_rolling_mean",
    ]:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    return df


def _normalize_station_id(s):
    raw = s.astype(str).str.strip()
    nums = pd.to_numeric(raw, errors="coerce")
    mask = nums.notna() & (nums == nums.fillna(0).astype(int).astype(float))
    out = raw.copy()
    out[mask] = nums[mask].astype(int).astype(str)
    return out


def create_features(df, artifacts=None, is_train=False):
    df = df.copy()
    df["city"] = df["city"].astype(str)
    df["start_station_id"] = _normalize_station_id(df["start_station_id"])
    df = _create_temporal_features(df)
    if artifacts is not None:
        df = _merge_agg_features(df, artifacts)
        if "demand_history" in artifacts:
            df = _merge_lag_features(df, artifacts["demand_history"])

    # 1 most pleasant, 0 terriable hell on earth
    if "apparent_temperature" in df.columns and "relative_humidity_2m" in df.columns:
        temp_c = np.exp(-((df["apparent_temperature"] - 18.0) ** 2) / (2 * 12.0**2))
        hum_c = np.clip(1.0 - np.abs(df["relative_humidity_2m"] - 50.0) / 50.0, 0, 1)
        cloud_c = (
            (1.0 - np.clip(df["cloud_cover"] / 100.0, 0, 1))
            if "cloud_cover" in df.columns
            else 0.5
        )
        df["comfort_score"] = 0.5 * temp_c + 0.3 * hum_c + 0.2 * cloud_c

    # Precipitation - rain, snow, and wind.
    # 1 perfect, 0 extreme weather
    rain = (
        df["rain"].clip(0) if "rain" in df.columns else pd.Series(0.0, index=df.index)
    )
    snow = (
        df["snowfall"].clip(0)
        if "snowfall" in df.columns
        else pd.Series(0.0, index=df.index)
    )
    wind = (
        df["wind_speed_10m"].clip(0)
        if "wind_speed_10m" in df.columns
        else pd.Series(0.0, index=df.index)
    )
    df["outdoor_pleasantness"] = (
        np.exp(-rain / 2.0) * np.exp(-snow * 5.0) * np.exp(-wind / 25.0)
    )

    # any disabled bike for the unfortunate?
    if "park_area_500m" in df.columns and "bike_lane_length_500m" in df.columns:
        df["recreational_accessibility"] = (
            df["park_area_500m"] / 100_000.0 + df["bike_lane_length_500m"] / 10_000.0
        )

    # rush, nicky lauda
    if "hour" in df.columns:
        df["is_morning_rush"] = ((df["hour"] >= 7) & (df["hour"] <= 9)).astype(int)
        df["is_evening_rush"] = ((df["hour"] >= 17) & (df["hour"] <= 19)).astype(int)

    if "station_hour_mean_demand" in df.columns:
        is_rush = df["is_morning_rush"] | df["is_evening_rush"]
        df["rush_x_station_hour_mean"] = is_rush * df["station_hour_mean_demand"]
        df["rush_x_station_hour_mean"] = df["rush_x_station_hour_mean"].clip(upper=3.0)

    return df


def make_features_from_rides(rides_df):
    rides = rides_df.copy()

    # City 3 data filtering so no overfitting
    city_counts = rides["city"].value_counts()
    major_cities = city_counts[city_counts >= 10_000].index
    rides = rides[rides["city"].isin(major_cities)].copy()

    rides["hour_ts"] = pd.to_datetime(rides["hour_ts"])
    rides["hour"] = rides["hour_ts"].dt.hour
    rides["weekday"] = pd.to_datetime(rides["date"]).dt.weekday

    if "weekend" not in rides.columns:
        rides["weekend"] = rides["weekday"].isin([5, 6]).astype(int)
    
    rides["city"] = rides["city"].astype(str)
    rides["start_station_id"] = rides["start_station_id"].astype(str)

    obs_demand: pd.DataFrame = (
        rides.groupby(grp_cols, dropna=False, observed=True)
        .size()
        .reset_index(name="demand")

    )
    obs_demand = obs_demand.drop_duplicates()
    print(obs_demand.columns)
    print(obs_demand.__len__())
    rides_dedup = rides.groupby(grp_cols).first().reset_index()
    obs_demand = obs_demand.merge(rides_dedup, on=grp_cols, how="left")
    print(obs_demand.columns)
    print(obs_demand.__len__())

    obs_demand["start_station_id"] = _normalize_station_id(
        obs_demand["start_station_id"]
    )

    station_list = (
        obs_demand[["city", "start_station_id"]]
        .drop_duplicates()
        .reset_index(drop=True)
    )
    city_date_range = (
        obs_demand[["city", "hour_ts"]]
        .drop_duplicates()
        .groupby("city")["hour_ts"]
        .agg(["min", "max"])
        .reset_index()
    )

    meta_cols = [
        c
        for c in [
            "start_lat",
            "start_lng",
            "bike_lane_length_500m",
            "park_area_500m",
            "university_count_1000m",
            "office_poi_count_1000m",
            "retail_poi_count_1000m",
            "restaurant_cafe_count_500m",
            "transit_stop_count_500m",
            "distance_to_nearest_rail_station",
            "distance_to_city_center",
        ]
        if c in obs_demand.columns
    ]
    station_meta = (
        obs_demand[["city", "start_station_id"] + meta_cols]
        .groupby(["city", "start_station_id"])
        .first()
        .reset_index()
    )

    weather_cols = [
        c
        for c in [
            "temperature_2m",
            "relative_humidity_2m",
            "apparent_temperature",
            "precipitation",
            "rain",
            "snowfall",
            "cloud_cover",
            "wind_speed_10m",
        ]
        if c in obs_demand.columns
    ]
    city_hour_weather = obs_demand[["city", "hour_ts"] + weather_cols].drop_duplicates(
        ["city", "hour_ts"]
    )

    holiday_info = obs_demand[
        ["city", "hour_ts", "holiday", "working_day"]
    ].drop_duplicates(["city", "hour_ts"])

    all_parts = []
    for _, row in city_date_range.iterrows():
        city = row["city"]
        hours = pd.date_range(row["min"], row["max"], freq="h")
        stations = station_list[station_list["city"] == city]["start_station_id"].values
        n_hours = len(hours)
        all_parts.append(
            pd.DataFrame(
                {
                    "city": np.repeat(city, len(stations) * n_hours),
                    "start_station_id": np.repeat(stations, n_hours),
                    "hour_ts": np.tile(hours, len(stations)),
                }
            )
        )

    full = pd.concat(all_parts, ignore_index=True) if all_parts else pd.DataFrame()
    full["hour"] = full["hour_ts"].dt.hour
    full["weekday"] = full["hour_ts"].dt.weekday
    full["weekend"] = full["weekday"].isin([5, 6]).astype(int)
    full = full.merge(station_meta, on=["city", "start_station_id"], how="left")
    full = full.merge(city_hour_weather, on=["city", "hour_ts"], how="left")
    full = full.merge(holiday_info, on=["city", "hour_ts"], how="left")
    full["holiday"] = full["holiday"].fillna(0).astype(int)
    full["working_day"] = full["working_day"].fillna(1).astype(int)

    demand_lookup = obs_demand[["city", "start_station_id", "hour_ts", "demand"]]
    full = full.merge(
        demand_lookup, on=grp_cols, how="left"
    )
    full["demand"] = full["demand"].fillna(0).astype(int)

    station_hour_demand = full

    station_stats = (
        station_hour_demand.groupby(["city", "start_station_id"])["demand"]
        .agg(["mean", "median", "max"])
        .rename(
            columns={
                "mean": "station_mean_demand",
                "median": "station_median_demand",
                "max": "station_max_demand",
            }
        )
        .reset_index()
    )
    city_hour_stats = (
        station_hour_demand.groupby(["city", "hour"])["demand"]
        .agg(["mean", "median", "max"])
        .rename(
            columns={
                "mean": "city_hour_mean_demand",
                "median": "city_hour_median_demand",
                "max": "city_hour_max_demand",
            }
        )
        .reset_index()
    )
    city_weekday_stats = (
        station_hour_demand.groupby(["city", "weekday"])["demand"]
        .mean()
        .reset_index(name="city_weekday_mean_demand")
    )
    city_mean_stats = (
        station_hour_demand.groupby(["city"])["demand"]
        .mean()
        .reset_index(name="city_mean_demand")
    )
    station_hour_stats = (
        station_hour_demand.groupby(grp_cols)["demand"]
        .agg(["mean", "median", "max", "std"])
        .rename(
            columns={
                "mean": "station_hour_mean_demand",
                "median": "station_hour_median_demand",
                "max": "station_hour_max_demand",
                "std": "station_hour_std_demand",
            }
        )
        .reset_index()
    )
    station_hour_stats["station_hour_std_demand"] = station_hour_stats[
        "station_hour_std_demand"
    ].fillna(0.0)

    demand_history = (
        obs_demand[grp_cols + ["demand"]]
        .groupby(grp_cols, as_index=False)["demand"]
        .sum()
    )

    artifacts = {
        "station_stats": station_stats,
        "city_hour_stats": city_hour_stats,
        "city_weekday_stats": city_weekday_stats,
        "city_mean_stats": city_mean_stats,
        "station_hour_stats": station_hour_stats,
        "demand_history": demand_history,
    }

    train = create_features(station_hour_demand, artifacts, is_train=True)
    train, encoders = _encode_categoricals(train, encoders=None)
    artifacts["encoders"] = encoders
    train["demand"] = station_hour_demand["demand"].values

    return train, artifacts


class BikeDemandModel:
    def __init__(self):
        self.artifacts = None

    def load_artifacts(self, artifacts: dict) -> None:
        self.artifacts = artifacts

    def _prepare_x(self, test_df):
        X = create_features(test_df, self.artifacts, is_train=False)
        
        if not self.artifacts:
            raise RuntimeError("missing artifacts")
        
        encoders = self.artifacts.get("encoders", {})
        X, _ = _encode_categoricals(X, encoders=encoders)
        X = X.reindex(columns=FEATURE_COLUMNS, fill_value=0)
        for col in FEATURE_COLUMNS:
            if col not in X.columns:
                X[col] = 0
        return X

    def predict(self, test_df: pd.DataFrame) -> np.ndarray:
        if self.artifacts is None or "model" not in self.artifacts:
            raise RuntimeError("Model not loaded.")
        
        X = self._prepare_x(test_df)
        model = self.artifacts["model"]
        preds = model.predict(X)
        preds = np.where(preds < 0.6, 0.0, preds)

        return np.maximum(0.0, preds)
