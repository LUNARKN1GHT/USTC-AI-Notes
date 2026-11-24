"""Read KuaiRand-Pure and start a small, chronological long-view experiment."""

from pathlib import Path
import tarfile

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
ARCHIVE = ROOT / "data" / "KuaiRand-Pure.tar.gz"
MAX_ROWS = 20_000  # Set to None to use every eligible row in each split.
TAB = 1
LOG_FILES = {
    "earlier": "log_standard_4_08_to_4_21_pure.csv",
    "standard_test": "log_standard_4_22_to_5_08_pure.csv",
    "random_test": "log_random_4_22_to_5_08_pure.csv",
}


def read_logs():
    columns = ["user_id", "video_id", "date", "duration_ms", "long_view", "tab"]
    frames = {}
    with tarfile.open(ARCHIVE, "r:gz") as archive:
        for split, filename in LOG_FILES.items():
            with archive.extractfile("KuaiRand-Pure/data/" + filename) as source:
                frame = pd.read_csv(source, usecols=columns)
            # Keep one feed context and a duration known before the user watches.
            frames[split] = frame.loc[(frame.tab == TAB) & (frame.duration_ms > 0)]
    return frames


def small_sample(frame):
    if MAX_ROWS is None or len(frame) <= MAX_ROWS:
        return frame.copy()
    return frame.sample(n=MAX_ROWS, random_state=42).copy()


def features(frame, center, scale):
    duration = np.log1p(frame.duration_ms.to_numpy(dtype=float) / 1000)
    # First column is the intercept. Only the training set sets center and scale.
    return np.column_stack((np.ones(len(frame)), (duration - center) / scale))


def sigmoid(values):
    return 1 / (1 + np.exp(-np.clip(values, -30, 30)))


def fit_logistic(x, y):
    weights = np.zeros(x.shape[1])
    for _ in range(600):
        probabilities = sigmoid((x * weights).sum(axis=1))
        gradient = (x * (probabilities - y)[:, None]).mean(axis=0)
        weights -= 0.1 * gradient
    return weights


def evaluate(y, probabilities):
    p = np.clip(probabilities, 1e-8, 1 - 1e-8)
    return {
        "observed_rate": y.mean(),
        "mean_prediction": p.mean(),
        "log_loss": -(y * np.log(p) + (1 - y) * np.log1p(-p)).mean(),
        "brier": np.square(p - y).mean(),
        "accuracy": ((p >= 0.5) == y).mean(),
    }


def main():
    logs = read_logs()
    earlier = logs.pop("earlier")
    splits = {
        "train": earlier.loc[earlier.date < 20220418],
        "validation": earlier.loc[earlier.date >= 20220418],
        **logs,
    }
    assert splits["train"].date.max() < splits["validation"].date.min()
    assert splits["validation"].date.max() < splits["standard_test"].date.min()
    for name, frame in splits.items():
        print(f"{name}: eligible={len(frame):,}; dates={frame.date.min()}..{frame.date.max()}")
    splits = {name: small_sample(frame) for name, frame in splits.items()}
    print("\nRow cap per split:", MAX_ROWS, "(None means all eligible rows).")
    print(splits["train"][["user_id", "video_id", "duration_ms", "long_view"]].head(3))

    train = splits["train"]
    log_duration = np.log1p(train.duration_ms.to_numpy(dtype=float) / 1000)
    center, scale = log_duration.mean(), max(log_duration.std(), 1e-8)
    x_train = features(train, center, scale)
    y_train = train.long_view.to_numpy(dtype=float)
    assert set(np.unique(y_train)) == {0, 1}
    print("\nX_train shape:", x_train.shape, "y_train shape:", y_train.shape)
    weights = fit_logistic(x_train, y_train)
    print("Learned weights [intercept, standardized log-duration]:", weights)

    rows = []
    for split, frame in splits.items():
        if split == "train":
            continue
        y = frame.long_view.to_numpy(dtype=float)
        predictions = {
            "training_mean": np.full(len(frame), y_train.mean()),
            "duration_logistic": sigmoid((features(frame, center, scale) * weights).sum(axis=1)),
        }
        for model, probabilities in predictions.items():
            rows.append({"split": split, "model": model, "n": len(y), **evaluate(y, probabilities)})
    print("\n", pd.DataFrame(rows).to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
