"""Prepare one station's hourly checkouts and compare small forecasting baselines."""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
TIME_ZONE = "Europe/Oslo"
START = pd.Timestamp("2025-08-01", tz=TIME_ZONE)
VALIDATION = pd.Timestamp("2025-08-20", tz=TIME_ZONE)
TEST = pd.Timestamp("2025-08-25", tz=TIME_ZONE)
STOP = pd.Timestamp("2025-09-01", tz=TIME_ZONE)


def load_hourly_checkouts():
    path = HERE / "data/oslo-2025-08.csv"
    trips = pd.read_csv(
        path,
        usecols=["started_at", "start_station_id", "start_station_name"],
        dtype={"start_station_id": str},
    )
    trips["started_at"] = pd.to_datetime(
        trips.started_at, utc=True, format="mixed"
    ).dt.tz_convert(TIME_ZONE)
    training_trips = trips.loc[
        (trips.started_at >= START) & (trips.started_at < VALIDATION)
    ]
    station_id = training_trips.start_station_id.value_counts().idxmax()
    station_name = training_trips.loc[
        training_trips.start_station_id == station_id, "start_station_name"
    ].iloc[0]
    selected = trips.loc[trips.start_station_id == station_id]
    counts = selected.groupby(selected.started_at.dt.floor("h")).size()
    # Preserve missing hours: a slot with no recorded checkout has a count of zero.
    all_hours = pd.date_range(START, STOP, freq="h", inclusive="left")
    hourly = counts.reindex(all_hours, fill_value=0).rename("checkouts").to_frame()
    return station_id, station_name, len(trips), hourly


def make_examples(hourly):
    examples = hourly.copy()
    # For target hour t, these three counts have finished before t begins.
    for lag in (1, 2, 24):
        examples[f"lag_{lag}"] = examples.checkouts.shift(lag)
    phase = 2 * np.pi * examples.index.hour.to_numpy() / 24
    examples["hour_sin"] = np.sin(phase)
    examples["hour_cos"] = np.cos(phase)
    examples["weekend"] = (examples.index.dayofweek >= 5).astype(float)
    examples = examples.dropna()
    # Limit this introductory comparison to 05:00–01:00 next day.
    return examples.loc[(examples.index.hour >= 5) | (examples.index.hour == 0)]


def fit_ridge(x_train, y_train, strength):
    center = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale == 0] = 1
    x = (x_train - center) / scale
    y_center = y_train.mean()
    weights = np.linalg.solve(
        np.einsum("ni,nj->ij", x, x) / len(x) + strength * np.eye(x.shape[1]),
        np.einsum("ni,n->i", x, y_train - y_center) / len(x),
    )
    return center, scale, y_center, weights


def ridge_prediction(model, x):
    center, scale, y_center, weights = model
    return np.maximum(0, np.einsum("ni,i->n", (x - center) / scale, weights) + y_center)


def mae(prediction, target):
    return float(np.abs(prediction - target).mean())


def main():
    station_id, station_name, raw_rows, hourly = load_hourly_checkouts()
    examples = make_examples(hourly)
    feature_names = ["lag_1", "lag_2", "lag_24", "hour_sin", "hour_cos", "weekend"]
    x = examples[feature_names].to_numpy(dtype=float)
    y = examples.checkouts.to_numpy(dtype=float)
    masks = {
        "train": examples.index < VALIDATION,
        "validation": (examples.index >= VALIDATION) & (examples.index < TEST),
        "test": examples.index >= TEST,
    }
    assert examples.index.is_monotonic_increasing
    assert np.isfinite(x).all() and np.isfinite(y).all()
    assert all(mask.any() for mask in masks.values())

    print(
        f"Raw trips: {raw_rows:,}; train-selected station: {station_name} ({station_id})"
    )
    print("Target = checkouts during the hour that starts at the row timestamp.")
    print(examples.head(3).to_string())
    print("Features:", feature_names)
    for name, mask in masks.items():
        print(f"{name}: X={x[mask].shape}, y={y[mask].shape}")

    train, validation = masks["train"], masks["validation"]
    candidates = []
    for strength in (0.01, 0.1, 1.0):
        model = fit_ridge(x[train], y[train], strength)
        candidates.append(
            (
                mae(ridge_prediction(model, x[validation]), y[validation]),
                strength,
                model,
            )
        )
    _, strength, model = min(candidates, key=lambda result: result[0])
    predictions = {
        "Previous hour": examples.lag_1.to_numpy(),
        "Previous day same hour": examples.lag_24.to_numpy(),
        "Ridge regression": ridge_prediction(model, x),
    }
    print(f"Ridge strength chosen on validation: {strength}")
    for name, prediction in predictions.items():
        print(
            f"{name}: validation MAE={mae(prediction[validation], y[validation]):.3f}, "
            f"test MAE={mae(prediction[masks['test']], y[masks['test']]):.3f}"
        )


if __name__ == "__main__":
    main()
