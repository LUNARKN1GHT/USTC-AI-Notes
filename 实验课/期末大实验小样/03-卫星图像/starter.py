"""用六个颜色统计量建立 EuroSAT 分类基线，在 CPU 上运行。"""

import argparse
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
from PIL import Image
from torch import nn

HERE = Path(__file__).resolve().parent
CLASSES = (
    "AnnualCrop",
    "Forest",
    "HerbaceousVegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "PermanentCrop",
    "Residential",
    "River",
    "SeaLake",
)


def load_data(data_dir, counts, seed):
    """从三个既定地理区域分别抽样，每个类别取相同数量。"""
    rng = np.random.default_rng(seed)
    names_by_split = {}
    for split in counts:
        path = data_dir / f"eurosat-spatial-{split}.txt"
        names_by_split[split] = {
            Path(line.strip()).name
            for line in path.read_text().splitlines()
            if line.strip()
        }
    sets = list(names_by_split.values())
    if sum(map(len, sets)) != len(set.union(*sets)):
        raise ValueError("训练、验证、测试清单出现重复图像")

    data = {}
    with ZipFile(data_dir / "EuroSAT_RGB.zip") as archive:
        members = {
            Path(name).name: name
            for name in archive.namelist()
            if name.endswith(".jpg")
        }
        for split, per_class in counts.items():
            features, labels, selected = [], [], []
            for label, class_name in enumerate(CLASSES):
                candidates = sorted(
                    name
                    for name in names_by_split[split]
                    if name.startswith(class_name + "_")
                )
                if len(candidates) < per_class:
                    raise ValueError(
                        f"{split}/{class_name} 只有 {len(candidates)} 张图像"
                    )
                for name in rng.permutation(candidates)[:per_class]:
                    with (
                        archive.open(members[name]) as source,
                        Image.open(source) as image,
                    ):
                        pixels = (
                            np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
                        )
                    # H×W×3 -> 6 维特征，分别是 RGB 的均值和标准差。
                    features.append(
                        np.r_[pixels.mean(axis=(0, 1)), pixels.std(axis=(0, 1))]
                    )
                    labels.append(label)
                    selected.append(name)
            data[split] = (
                torch.tensor(np.stack(features), dtype=torch.float32),
                torch.tensor(labels, dtype=torch.long),
                selected,
            )
    return data


def metrics(labels, predictions):
    confusion = torch.bincount(
        len(CLASSES) * labels + predictions, minlength=len(CLASSES) ** 2
    ).reshape(len(CLASSES), len(CLASSES))
    correct = confusion.diag().float()
    denominator = confusion.sum(dim=0) + confusion.sum(dim=1)
    f1 = 2 * correct / denominator.clamp(min=1)
    return float(correct.sum() / len(labels)), float(f1.mean())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=HERE / "data")
    parser.add_argument("--train-per-class", type=int, default=100)
    parser.add_argument("--val-per-class", type=int, default=20)
    parser.add_argument("--test-per-class", type=int, default=30)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    counts = {
        "train": args.train_per_class,
        "val": args.val_per_class,
        "test": args.test_per_class,
    }
    if min(*counts.values(), args.epochs) < 1:
        parser.error("每类图像数和训练轮数必须大于零")

    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    data = load_data(args.data_dir, counts, args.seed)
    for split, (features, labels, _) in data.items():
        print(
            f"{split}: X={tuple(features.shape)}, y={tuple(labels.shape)}, "
            f"每类数量={dict(Counter(labels.tolist()))}"
        )
    print("类别编号:", dict(enumerate(CLASSES)))

    # 只从训练数据估计标准化参数，验证和测试沿用这组参数。
    center = data["train"][0].mean(dim=0)
    scale = data["train"][0].std(dim=0).clamp(min=1e-6)
    inputs = {
        split: (features - center) / scale for split, (features, _, _) in data.items()
    }
    targets = {split: labels for split, (_, labels, _) in data.items()}
    model = nn.Linear(6, len(CLASSES))
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.03, weight_decay=0.01)
    best_loss = float("inf")
    for epoch in range(1, args.epochs + 1):
        model.train()
        optimizer.zero_grad()
        loss = nn.functional.cross_entropy(model(inputs["train"]), targets["train"])
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss = nn.functional.cross_entropy(
                model(inputs["val"]), targets["val"]
            ).item()
        if val_loss < best_loss:
            best_loss = val_loss
            best_epoch = epoch
            best_state = {
                key: value.clone() for key, value in model.state_dict().items()
            }
    model.load_state_dict(best_state)
    print(f"按验证集损失选择第 {best_epoch} 轮，验证损失={best_loss:.4f}")

    with torch.no_grad():
        for split in ("val", "test"):
            predictions = model(inputs[split]).argmax(dim=1)
            accuracy, macro_f1 = metrics(targets[split], predictions)
            print(f"{split}: accuracy={accuracy:.3f}, macro_f1={macro_f1:.3f}")
        majority = int(targets["train"].bincount().argmax())
        baseline = torch.full_like(targets["test"], majority)
        accuracy, macro_f1 = metrics(targets["test"], baseline)
        print(f"训练集多数类基线: accuracy={accuracy:.3f}, macro_f1={macro_f1:.3f}")
        predictions = model(inputs["test"]).argmax(dim=1)
        wrong = (predictions != targets["test"]).nonzero(as_tuple=True)[0][:5]
        print("可继续检查的错误图像:")
        for index in wrong.tolist():
            print(
                f"  {data['test'][2][index]}: "
                f"真实={CLASSES[int(targets['test'][index])]} "
                f"预测={CLASSES[int(predictions[index])]}"
            )


if __name__ == "__main__":
    main()
