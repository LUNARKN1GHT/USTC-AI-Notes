"""Compare color, texture and convolutional features on EuroSAT RGB."""

import argparse
import csv
import json
import time
import zipfile
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn


ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"
SEED = 41
SPLITS = {"train": 200, "val": 40, "in_region_test": 50, "spatial_test": 50}


def stratify(file_names, samples_per_class, rng):
    by_class = defaultdict(list)
    for name in file_names:
        by_class[Path(name).stem.rsplit("_", 1)[0]].append(name)
    for label, names in sorted(by_class.items()):
        if len(names) < samples_per_class:
            raise ValueError(f"{label}: only {len(names)} images for {samples_per_class} requested")
        yield label, rng.permutation(sorted(names))[:samples_per_class]


def load_samples(data_dir):
    rng = np.random.default_rng(SEED)
    split_files = {}
    for part in ("train", "val", "test"):
        split_path = data_dir / f"eurosat-spatial-{part}.txt"
        names = split_path.read_text(encoding="utf-8").splitlines()
        split_files[part] = {Path(name.strip()).name for name in names if name.strip()}
    if sum(map(len, split_files.values())) != len(set.union(*split_files.values())):
        raise ValueError("TorchGeo split files overlap")

    rows = []
    for label, names in stratify(split_files["train"], SPLITS["train"] + SPLITS["in_region_test"], rng):
        for idx, name in enumerate(names):
            part = "train" if idx < SPLITS["train"] else "in_region_test"
            rows.append((part, label, name))
    for official_part, pilot_part in (("val", "val"), ("test", "spatial_test")):
        for label, names in stratify(split_files[official_part], SPLITS[pilot_part], rng):
            rows.extend((pilot_part, label, name) for name in names)

    class_names = sorted({label for part, label, name in rows if part == "train"})
    if len(class_names) != 10:
        raise ValueError(f"Expected 10 EuroSAT classes, found {len(class_names)}")
    label_index = {label: idx for idx, label in enumerate(class_names)}
    data = {}
    with zipfile.ZipFile(data_dir / "EuroSAT_RGB.zip") as archive:
        members = {Path(name).name: name for name in archive.namelist() if name.lower().endswith(".jpg")}
        if not set.union(*split_files.values()).issubset(members):
            raise ValueError("Some TorchGeo split entries are absent from the RGB archive")
        for part in SPLITS:
            images, labels, names = [], [], []
            for row_part, label, name in rows:
                if row_part != part:
                    continue
                with archive.open(members[name]) as raw, Image.open(raw) as image:
                    pixels = image.convert("RGB").resize((32, 32), Image.Resampling.BILINEAR)
                    images.append(np.asarray(pixels, dtype=np.uint8))
                labels.append(label_index[label])
                names.append(name)
            if len(labels) != len(class_names) * SPLITS[part] or len(set(labels)) != len(class_names):
                raise ValueError(f"Missing samples in {part}")
            data[part] = (np.stack(images), torch.tensor(labels), names)
    return data, class_names, rows


def image_features(images):
    colors, textures = [], []
    for image in images:
        colors.append(np.concatenate([
            np.bincount(channel.ravel() // 32, minlength=8) / channel.size
            for channel in image.transpose(2, 0, 1)
        ]))
        gray = image @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        dy, dx = np.gradient(gray)
        magnitude = np.hypot(dx, dy)
        orientation = (np.arctan2(dy, dx) % np.pi * (8 / np.pi)).astype(int).clip(0, 7)
        histogram = []
        for i in range(0, 32, 8):
            for j in range(0, 32, 8):
                region = np.s_[i:i + 8, j:j + 8]
                cell = np.bincount(orientation[region].ravel(), weights=magnitude[region].ravel(), minlength=8)
                histogram.append(cell / max(cell.sum(), 1.0))
        textures.append(np.concatenate(histogram))
    return np.stack(colors), np.stack(textures)


def standardize(features):
    mean = features["train"].mean(axis=0)
    scale = features["train"].std(axis=0).clip(min=1e-6)
    return {part: torch.tensor((values - mean) / scale, dtype=torch.float32)
            for part, values in features.items()}


def fit_logistic(inputs, targets, n_classes):
    model = nn.Linear(inputs["train"].shape[1], n_classes)
    optimizer = torch.optim.LBFGS(model.parameters(), max_iter=75, line_search_fn="strong_wolfe")

    def loss_and_gradient():
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(inputs["train"]), targets["train"])
        loss = loss + 0.01 * model.weight.square().sum()
        loss.backward()
        return loss

    optimizer.step(loss_and_gradient)
    return model, {}


def fit_cnn(inputs, targets, n_classes, epochs):
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
        nn.AdaptiveAvgPool2d((4, 4)), nn.Flatten(), nn.Linear(32 * 4 * 4, n_classes),
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    shuffle = torch.Generator().manual_seed(SEED)
    best_loss = float("inf")
    for epoch in range(1, epochs + 1):
        model.train()
        for indices in torch.randperm(len(targets["train"]), generator=shuffle).split(64):
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(model(inputs["train"][indices]), targets["train"][indices])
            loss.backward()
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_loss = nn.functional.cross_entropy(model(inputs["val"]), targets["val"]).item()
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_epoch = epoch
            best_state = {name: values.clone() for name, values in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model, {"best_epoch": best_epoch}


def metrics(y_true, y_pred, class_names):
    count = len(class_names)
    confusion = np.bincount(count * y_true + y_pred, minlength=count ** 2).reshape(count, count)
    correct = confusion.diagonal()
    f1 = 2 * correct / np.maximum(confusion.sum(axis=0) + confusion.sum(axis=1), 1)
    return {
        "accuracy": float(correct.sum() / len(y_true)),
        "macro_f1": float(f1.mean()),
        "class_f1": dict(zip(class_names, f1.tolist())),
        "confusion_matrix": confusion.tolist(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=DATA)
    parser.add_argument("--output-dir", type=Path, default=ROOT)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.epochs < 1 or args.threads < 1:
        parser.error("--epochs and --threads must be positive")

    start = time.perf_counter()
    torch.manual_seed(SEED)
    torch.set_num_threads(args.threads)
    data, classes, rows = load_samples(args.data_dir)
    features = {part: image_features(images) for part, (images, labels, names) in data.items()}
    targets = {part: labels for part, (images, labels, names) in data.items()}
    inputs_by_model = {
        "logistic_color": standardize({part: pair[0] for part, pair in features.items()}),
        "logistic_color_texture": standardize({part: np.concatenate(pair, axis=1) for part, pair in features.items()}),
        "cnn": {part: torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255
                for part, (images, labels, names) in data.items()},
    }

    results = {"seed": SEED, "sample_counts": {part: len(y) for part, y in targets.items()},
               "class_order": classes, "cnn_epochs": args.epochs, "models": {}}
    predictions = []
    print(f"Samples: {results['sample_counts']}", flush=True)
    for name, inputs in inputs_by_model.items():
        training_start = time.perf_counter()
        torch.manual_seed(SEED)
        if name == "cnn":
            model, training = fit_cnn(inputs, targets, len(classes), args.epochs)
        else:
            model, training = fit_logistic(inputs, targets, len(classes))
        training["fit_seconds"] = round(time.perf_counter() - training_start, 2)
        training["parameters"] = sum(parameter.numel() for parameter in model.parameters())
        model.eval()
        for part in ("val", "in_region_test", "spatial_test"):
            with torch.no_grad():
                pred = torch.cat([model(batch).argmax(dim=1) for batch in inputs[part].split(128)]).numpy()
            training[part] = metrics(targets[part].numpy(), pred, classes)
            predictions.extend((name, part, file_name, classes[actual], classes[predicted])
                               for file_name, actual, predicted in zip(data[part][2], targets[part], pred))
        results["models"][name] = training
        print(f"{name:23s} in-region F1 {training['in_region_test']['macro_f1']:.3f}  "
              f"held-out-region F1 {training['spatial_test']['macro_f1']:.3f}", flush=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "pilot_splits.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("split", "label", "file_name"))
        writer.writerows(rows)
    with (args.output_dir / "predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("model", "split", "file_name", "label", "prediction"))
        writer.writerows(predictions)
    results["elapsed_seconds"] = round(time.perf_counter() - start, 2)
    (args.output_dir / "results.json").write_text(json.dumps(results, indent=2) + "\n", encoding="utf-8")
    print(f"Finished in {results['elapsed_seconds']:.1f}s. Results: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
