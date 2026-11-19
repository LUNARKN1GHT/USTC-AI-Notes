"""Compare long-view predictions for standard and randomized KuaiRand exposures."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import tarfile
import tempfile

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "kuairand-pilot-matplotlib")
)
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "kuairand-pilot-cache"))

import numpy as np
import pandas as pd


HERE = Path(__file__).resolve().parent
ARCHIVE_URL = "https://zenodo.org/records/10439422/files/KuaiRand-Pure.tar.gz"
ARCHIVE_MD5 = "0820331067a3784d9691136f772b35a7"
FIT_START = 20220414
TEST_START = 20220422
TEST_END = 20220508
PRIOR_WEIGHT = 20
FEATURE_NAMES = (
    "log_duration",
    "log_user_history_count",
    "user_history_long_view_rate",
    "log_video_history_count",
    "video_history_long_view_rate",
)
LOG_NAMES = {
    "log_standard_4_08_to_4_21_pure.csv": "earlier_standard",
    "log_standard_4_22_to_5_08_pure.csv": "standard",
    "log_random_4_22_to_5_08_pure.csv": "randomized",
}
LOG_COLUMNS = {
    "user_id": "int32",
    "video_id": "int32",
    "date": "int32",
    "duration_ms": "int32",
    "tab": "int8",
    "long_view": "int8",
}


def read_logs(path, tab):
    checksum = hashlib.md5()  # Match the published checksum for this exact archive.
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            checksum.update(chunk)
    if checksum.hexdigest() != ARCHIVE_MD5:
        raise ValueError(f"Archive checksum mismatch. Download it again from {ARCHIVE_URL}")

    logs = {}
    with tarfile.open(path, "r:gz") as archive:
        for member in archive:
            name = LOG_NAMES.get(Path(member.name).name)
            if name is None or not member.isfile():
                continue
            with archive.extractfile(member) as source:
                frame = pd.read_csv(source, usecols=LOG_COLUMNS, dtype=LOG_COLUMNS)
            # Restrict to one feed and to exposures with a valid video duration.
            logs[name] = frame.loc[(frame.tab == tab) & (frame.duration_ms > 0)].drop(
                columns="tab"
            )
    if set(logs) != set(LOG_NAMES.values()):
        raise ValueError("The archive is missing one of the three expected logs")
    return logs


def history_statistics(history, key, prior):
    stats = history.groupby(key)["long_view"].agg(["count", "sum"])
    stats["rate"] = (stats["sum"] + PRIOR_WEIGHT * prior) / (stats["count"] + PRIOR_WEIGHT)
    return stats


def make_features(frame, prior, user_history, video_history):
    duration = np.log1p(frame.duration_ms.to_numpy() / 1000)
    user_count = frame.user_id.map(user_history["count"]).fillna(0).to_numpy()
    user_rate = frame.user_id.map(user_history["rate"]).fillna(prior).to_numpy()
    video_count = frame.video_id.map(video_history["count"]).fillna(0).to_numpy()
    video_rate = frame.video_id.map(video_history["rate"]).fillna(prior).to_numpy()
    return np.column_stack(
        (duration, np.log1p(user_count), user_rate, np.log1p(video_count), video_rate)
    )


def sigmoid(logits):
    return 1 / (1 + np.exp(-np.clip(logits, -35, 35)))


def fit_logistic(features, labels):
    """Optimize mean binary cross-entropy plus L2 by Newton's method."""
    center = features.mean(axis=0)
    scale = features.std(axis=0)
    scale[scale == 0] = 1
    design = np.column_stack((np.ones(len(features)), (features - center) / scale))
    weights = np.zeros(design.shape[1])
    penalty = np.eye(len(weights)) * 0.001
    penalty[0, 0] = 0  # Leave the intercept unpenalized.
    for _ in range(30):
        probabilities = sigmoid(np.einsum("ni,i->n", design, weights))
        gradient = (
            np.einsum("ni,n->i", design, probabilities - labels) / len(labels)
            + penalty @ weights
        )
        curvature = probabilities * (1 - probabilities)
        hessian = np.einsum("ni,nj,n->ij", design, design, curvature) / len(labels) + penalty
        step = np.linalg.solve(hessian, gradient)
        weights -= step
        if np.linalg.norm(step, ord=np.inf) < 1e-8:
            return center, scale, weights
    raise RuntimeError("Logistic optimization did not converge in 30 iterations")


def predict_logistic(model, features):
    center, scale, weights = model
    return sigmoid(weights[0] + np.einsum("ni,i->n", (features - center) / scale, weights[1:]))


def roc_auc(labels, probabilities):
    """Compute the probability that a positive outranks a negative, with half for ties."""
    order = np.argsort(probabilities)
    scores = probabilities[order]
    sorted_labels = labels[order]
    boundaries = np.r_[0, np.flatnonzero(scores[1:] != scores[:-1]) + 1, len(scores)]
    positives_by_score = np.add.reduceat(sorted_labels, boundaries[:-1])
    negatives_by_score = np.diff(boundaries) - positives_by_score
    negatives_below = np.cumsum(negatives_by_score) - negatives_by_score
    return float((positives_by_score * (negatives_below + 0.5 * negatives_by_score)).sum()) / (
        int(labels.sum()) * int((1 - labels).sum())
    )


def evaluate(labels, probabilities):
    p = np.clip(probabilities, 1e-12, 1 - 1e-12)
    loss = -(labels * np.log(p) + (1 - labels) * np.log1p(-p)).mean()
    return {
        "n": len(labels),
        "long_view_rate": float(labels.mean()),
        "mean_prediction": float(p.mean()),
        "roc_auc": roc_auc(labels, p),
        "log_loss": float(loss),
    }


def plot_results(rows, output):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    model_names = ["Train mean", "Duration only", "Duration + history"]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.9), layout="constrained")
    x = np.arange(len(model_names))
    axes[0].axhline(0.5, color="#777777", linestyle=":", linewidth=1, zorder=1.5)
    for split, offset, color in (
        ("standard", -0.19, "#216a8a"),
        ("randomized", 0.19, "#d08d26"),
    ):
        values = [row["roc_auc"] for row in rows if row["exposure"] == split]
        bars = axes[0].bar(x + offset, values, 0.35, label=split.title(), color=color)
        axes[0].bar_label(bars, fmt="%.3f", padding=4, fontsize=8)
    axes[0].set_xticks(x, model_names, fontsize=9)
    axes[0].set_ylim(0, 1)
    axes[0].set_title("ROC AUC (higher is better)")
    axes[0].legend(loc="upper left", frameon=False)

    history_results = [row for row in rows if row["model"] == "duration_and_history"]
    history_results.sort(key=lambda row: row["exposure"] != "standard")
    x = np.arange(len(history_results))
    for key, offset, color, label in (
        ("long_view_rate", -0.19, "#567d3a", "Observed"),
        ("mean_prediction", 0.19, "#8053a2", "Predicted"),
    ):
        values = [100 * row[key] for row in history_results]
        bars = axes[1].bar(x + offset, values, 0.35, label=label, color=color)
        axes[1].bar_label(bars, labels=[f"{value:.1f}%" for value in values], padding=4, fontsize=8)
    axes[1].set_xticks(x, [row["exposure"].title() for row in history_results], fontsize=9)
    max_rate = max(max(row["long_view_rate"], row["mean_prediction"]) for row in history_results)
    axes[1].set_ylim(0, min(100, max(50, 125 * max_rate)))
    axes[1].set_ylabel("Percent of exposures")
    axes[1].set_title("Long-view rate for model with history")
    axes[1].legend(loc="upper left", frameon=False)
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
    fig.suptitle("KuaiRand-Pure · Long-view prediction on later exposures")
    fig.savefig(output, dpi=150)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path, default=HERE / "data" / "KuaiRand-Pure.tar.gz")
    parser.add_argument("--output-dir", type=Path, default=HERE / "results")
    parser.add_argument("--tab", type=int, default=1, help="Analyze a single exposure context")
    args = parser.parse_args()
    if not args.archive.is_file():
        parser.error(f"Missing {args.archive}. Download from {ARCHIVE_URL}")

    logs = read_logs(args.archive, args.tab)
    earlier = logs.pop("earlier_standard")
    history = earlier.loc[earlier.date < FIT_START]
    train = earlier.loc[(earlier.date >= FIT_START) & (earlier.date < TEST_START)]
    shared_dates = set(logs["standard"].date) & set(logs["randomized"].date)
    test_days = sorted(day for day in shared_dates if TEST_START <= day <= TEST_END)
    tests = {split: data.loc[data.date.isin(test_days)] for split, data in logs.items()}
    for name, data in {"history": history, "train": train, **tests}.items():
        if set(data.long_view.unique()) != {0, 1}:
            raise ValueError(f"{name} must contain both labels for tab {args.tab}")

    history_prior = float(history.long_view.mean())
    train_prior = float(train.long_view.mean())
    user_history = history_statistics(history, "user_id", history_prior)
    video_history = history_statistics(history, "video_id", history_prior)
    x_train = make_features(train, history_prior, user_history, video_history)
    y_train = train.long_view.to_numpy()
    duration_model = fit_logistic(x_train[:, :1], y_train)
    history_model = fit_logistic(x_train, y_train)

    rows = []
    test_diagnostics = {}
    for exposure, test in tests.items():
        x = make_features(test, history_prior, user_history, video_history)
        labels = test.long_view.to_numpy()
        test_diagnostics[exposure] = {
            "unique_users": int(test.user_id.nunique()),
            "unique_videos": int(test.video_id.nunique()),
            "median_duration_seconds": float(test.duration_ms.median() / 1000),
            "user_seen_in_history_fraction": float((x[:, 1] > 0).mean()),
            "video_seen_in_history_fraction": float((x[:, 3] > 0).mean()),
        }
        predictions = {
            "train_mean": np.full(len(test), train_prior),
            "duration_only": predict_logistic(duration_model, x[:, :1]),
            "duration_and_history": predict_logistic(history_model, x),
        }
        for model, probabilities in predictions.items():
            rows.append({"exposure": exposure, "model": model, **evaluate(labels, probabilities)})

    args.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = args.output_dir / "metrics.csv"
    with metrics_path.open("w", encoding="utf-8", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    center, scale, weights = history_model
    metadata = {
        "source": ARCHIVE_URL,
        "archive_md5": ARCHIVE_MD5,
        "tab": args.tab,
        "history_rows": len(history),
        "history_dates": [int(history.date.min()), int(history.date.max())],
        "training_rows": len(train),
        "training_dates": [int(train.date.min()), int(train.date.max())],
        "shared_test_dates": test_days,
        "test_diagnostics": test_diagnostics,
        "training_long_view_rate": train_prior,
        "history_model": {
            "intercept": float(weights[0]),
            "features": {
                name: {"center": float(m), "scale": float(s), "coefficient": float(w)}
                for name, m, s, w in zip(FEATURE_NAMES, center, scale, weights[1:])
            },
        },
    }
    (args.output_dir / "run.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    plot_results(rows, args.output_dir / "comparison.png")
    print(f"History: {len(history):,} rows; training: {len(train):,} rows")
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
