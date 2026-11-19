"""One-hour-ahead forecasts for the Oslo City Bike pilot."""

import hashlib
import json
from pathlib import Path
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from replay import prepare_requests, replay


ROOT = Path(__file__).resolve().parent
TRIPS = ROOT / "data" / "oslo-2025-08.csv"
METADATA = ROOT / "stations.json"
TZ = "Europe/Oslo"
START = pd.Timestamp("2025-08-01", tz=TZ)
VALIDATION = pd.Timestamp("2025-08-20", tz=TZ)
TEST = pd.Timestamp("2025-08-25", tz=TZ)
STOP = pd.Timestamp("2025-09-01", tz=TZ)
STATION_COUNT = 16
NEIGHBOR_COUNT = 3
RIDGE_WEIGHTS = (0.001, 0.01, 0.1, 1.0)


def count_events(trips, times, station_ids, time_field, station_field):
    return (
        trips.groupby([times[time_field].dt.floor("h"), station_field]).size()
        .unstack(fill_value=0)
        .reindex(index=times["hours"], columns=station_ids, fill_value=0)
        .to_numpy(dtype=float)
    )


def historical_average(counts, hours, train_hours):
    hour_of_week_type = 24 * (hours.dayofweek >= 5).astype(int) + hours.hour
    means = np.stack([
        counts[train_hours & (hour_of_week_type == slot)].mean(axis=0)
        for slot in range(48)
    ])
    return means[hour_of_week_type]


def geographic_graph(stations):
    lat = np.radians([s["lat"] for s in stations])
    lon = np.radians([s["lon"] for s in stations])
    sin_lat = np.sin((lat[:, None] - lat) / 2)
    sin_lon = np.sin((lon[:, None] - lon) / 2)
    angle = sin_lat**2 + np.cos(lat[:, None]) * np.cos(lat) * sin_lon**2
    distances = 2 * 6371 * np.arcsin(np.sqrt(np.clip(angle, 0, 1)))
    np.fill_diagonal(distances, np.inf)
    neighbors = np.argsort(distances, axis=1)[:, :NEIGHBOR_COUNT]
    graph = np.zeros_like(distances)
    np.put_along_axis(graph, neighbors, 1 / NEIGHBOR_COUNT, axis=1)
    return graph


def temporal_features(counts, hours, graph=None):
    time_count, node_count, _ = counts.shape
    phase = 2 * np.pi * hours.hour.to_numpy() / 24
    weekend = (hours.dayofweek >= 5).astype(float)
    calendar = np.stack([
        np.sin(phase), np.cos(phase), np.sin(2 * phase), np.cos(2 * phase), weekend
    ], axis=-1)
    calendar = np.broadcast_to(calendar[:, None, :], (time_count, node_count, 5))
    station = np.broadcast_to(np.eye(node_count), (time_count, node_count, node_count))
    features = [calendar, station]
    for lag in (1, 2, 24):
        previous = np.zeros_like(counts)
        previous[lag:] = counts[:-lag]
        features.append(previous)
        if graph is not None:
            features.append(np.einsum("ij,tjk->tik", graph, previous))
    return np.concatenate(features, axis=-1)


def ridge_predict(features, target, train_hours, validation_hours):
    raw_train = features[train_hours].reshape(-1, features.shape[-1])
    labels = target[train_hours].reshape(-1, target.shape[-1])
    center = raw_train.mean(axis=0)
    scale = raw_train.std(axis=0)
    scale[scale == 0] = 1
    train = (raw_train - center) / scale
    label_mean = labels.mean(axis=0)
    gram = np.einsum("ni,nj->ij", train, train) / len(train)
    rhs = np.einsum("ni,nj->ij", train, labels - label_mean) / len(train)
    standardized = (features - center) / scale
    penalty = np.eye(gram.shape[0])
    runs = []
    for strength in RIDGE_WEIGHTS:
        coef = np.linalg.solve(gram + strength * penalty, rhs)
        prediction = np.clip(standardized @ coef + label_mean, 0, None)
        val_mae = float(np.abs(prediction[validation_hours] - target[validation_hours]).mean())
        runs.append((val_mae, strength, prediction, coef))
    val_mae, strength, prediction, coef = min(runs, key=lambda run: run[0])

    def estimate(new_features):
        return np.clip((new_features - center) / scale @ coef + label_mean, 0, None)

    return val_mae, strength, prediction, estimate


def evaluate(prediction, target, hours, test_hours):
    errors = prediction - target
    test_errors = errors[test_hours]
    rush_hours = test_hours & (hours.dayofweek < 5) & np.isin(hours.hour, [7, 8, 9, 15, 16, 17])
    return {
        "mae": float(np.abs(test_errors).mean()),
        "checkout_mae": float(np.abs(test_errors[:, :, 0]).mean()),
        "return_mae": float(np.abs(test_errors[:, :, 1]).mean()),
        "rmse": float(np.sqrt(np.mean(test_errors**2))),
        "rush_mae": float(np.abs(errors[rush_hours]).mean()),
    }


def plot_example(hours, counts, predictions, station):
    start = pd.Timestamp("2025-08-25 05:00", tz=TZ)
    stop = pd.Timestamp("2025-08-27 00:00", tz=TZ)
    mask = (hours >= start) & (hours < stop)
    labels = {
        "historical_average": "Historical average",
        "local_ridge": "Local ridge",
        "neighbor_ridge": "Ridge + neighbors",
    }
    figure, axes = plt.subplots(2, 1, figsize=(9, 5.5), sharex=True, layout="constrained")
    for index, axis in enumerate(axes):
        axis.plot(hours[mask], counts[mask, 0, index], color="black", linewidth=1.8, label="Observed")
        for key, prediction in predictions.items():
            axis.plot(hours[mask], prediction[mask, 0, index], linewidth=1.0, label=labels[key])
        axis.set_ylabel(["Checkouts / hour", "Returns / hour"][index])
        axis.grid(alpha=0.25)
    axes[0].legend(ncol=2, fontsize=8)
    axes[1].xaxis.set_major_formatter(matplotlib.dates.DateFormatter("%d %H:%M", tz=hours.tz))
    axes[1].set_xlabel("Day of August and local time")
    figure.suptitle(f"Oslo City Bike: {station['name']} ({station['station_id']})")
    figure.savefig(ROOT / "forecast.png", dpi=140)
    plt.close(figure)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    started = perf_counter()
    columns = ["started_at", "ended_at", "start_station_id", "end_station_id"]
    trips = pd.read_csv(TRIPS, usecols=columns, dtype={name: str for name in columns[2:]})
    snapshot = json.loads(METADATA.read_text(encoding="utf-8"))
    by_id = {str(s["station_id"]): s for s in snapshot["data"]["stations"]}
    hours = pd.date_range(START, STOP, freq="h", inclusive="left")
    times = {name: pd.to_datetime(trips[name], utc=True, format="mixed").dt.tz_convert(TZ)
             for name in columns[:2]}
    times["hours"] = hours
    start_times = times["started_at"]
    train_trips = trips[(start_times >= START) & (start_times < VALIDATION)]
    ranked = train_trips["start_station_id"].value_counts().rename_axis("station_id").reset_index()
    ranked = ranked.sort_values(["count", "station_id"], ascending=[False, True])
    ids = [sid for sid in ranked["station_id"] if sid in by_id][:STATION_COUNT]
    stations = [by_id[sid] for sid in ids]
    counts = np.stack([
        count_events(trips, times, ids, "started_at", "start_station_id"),
        count_events(trips, times, ids, "ended_at", "end_station_id"),
    ], axis=-1)

    active = (hours.hour == 0) | (hours.hour >= 5)
    train_all = (hours < VALIDATION) & (hours >= START + pd.Timedelta(days=1))
    train_hours = train_all & active
    validation_hours = (hours >= VALIDATION) & (hours < TEST) & active
    test_hours = (hours >= TEST) & active
    baseline = historical_average(counts, hours, train_all)
    predictions = {"historical_average": baseline}
    tuning = {"historical_average": {"validation_mae": float(
        np.abs(baseline[validation_hours] - counts[validation_hours]).mean())}}
    graph = geographic_graph(stations)
    estimators = {}
    for name, neighbor_graph in [("local_ridge", None), ("neighbor_ridge", graph)]:
        val_mae, strength, prediction, estimate = ridge_predict(
            temporal_features(counts, hours, neighbor_graph), counts, train_hours, validation_hours
        )
        predictions[name] = prediction
        estimators[name] = (estimate, neighbor_graph)
        tuning[name] = {"validation_mae": val_mae, "ridge_weight": strength}

    replay_start = hours.get_loc(TEST)
    replay_hours = hours[replay_start:]
    replay_active = active[replay_start:]
    warmup = counts[replay_start - 24:replay_start]
    forecast_hours = hours[replay_start - 24:]
    requests = prepare_requests(trips, times, ids, TEST, STOP)

    def online_forecast(name):
        estimate, adjacency = estimators[name]

        def predict(t, observed_counts):
            features = temporal_features(observed_counts[t:t + 25], forecast_hours[t:t + 25], adjacency)
            return estimate(features[-1])

        return predict

    policies = {
        "no_transfer": None,
        "stock_greedy": None,
        "historical_average_greedy": lambda t, _: baseline[replay_start + t],
        "local_ridge_greedy": online_forecast("local_ridge"),
        "neighbor_ridge_greedy": online_forecast("neighbor_ridge"),
    }
    replay_results = {
        name: replay(replay_hours, replay_active, stations, requests, warmup, name, forecast)
        for name, forecast in policies.items()
    }

    report = {
        "dataset": "Oslo City Bike August 2025",
        "raw_rows": len(trips),
        "raw_csv_sha256": sha256(TRIPS),
        "gbfs_source_sha256": snapshot["source_sha256"],
        "frozen_station_sha256": sha256(METADATA),
        "gbfs_last_updated_utc": pd.to_datetime(snapshot["last_updated"], unit="s", utc=True).isoformat(),
        "stations": [{key: s[key] for key in ("station_id", "name", "capacity")} for s in stations],
        "time_zone": TZ,
        "target": "hourly checkout and return counts; one hour ahead",
        "split": {
            "training": [str(START.date()), str(VALIDATION.date())],
            "validation": [str(VALIDATION.date()), str(TEST.date())],
            "test": [str(TEST.date()), str(STOP.date())],
            "right_endpoint_exclusive": True,
            "training_hours": int(train_hours.sum()),
            "validation_hours": int(validation_hours.sum()),
            "test_hours": int(test_hours.sum()),
        },
        "test_count_cells": int(counts[test_hours].size),
        "test": {name: evaluate(prediction, counts, hours, test_hours)
                 for name, prediction in predictions.items()},
        "tuning": tuning,
        "replay": {"recorded_request_count": len(requests), "policies": replay_results},
    }
    plot_example(hours, counts, predictions, stations[0])
    report["wall_seconds"] = round(perf_counter() - started, 3)
    (ROOT / "results.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n",
                                      encoding="utf-8")
    print(json.dumps(report["test"], indent=2))
    print(json.dumps(replay_results, indent=2))
    print(f"Elapsed: {report['wall_seconds']} s")


if __name__ == "__main__":
    main()
