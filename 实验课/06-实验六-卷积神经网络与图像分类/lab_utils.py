"""实验六教师工具：复用训练流程，让学生集中研究图像与模型结构。"""

import os
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms

CLASSES = (
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
)
CLASS_NAMES_ZH = ("飞机", "汽车", "鸟", "猫", "鹿", "狗", "青蛙", "马", "船", "卡车")
MEAN, STD = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def data_root():
    return Path(os.environ.get("CIFAR10_ROOT", "data"))


def build_transform(augment=False):
    """随机增强只给训练视图；统一用固定常数将 [0,1] 映射到 [-1,1]。"""
    steps = []
    if augment:
        steps += [
            transforms.RandomCrop(32, padding=4, padding_mode="reflect"),
            transforms.RandomHorizontalFlip(p=0.5),
        ]
    return transforms.Compose(
        steps + [transforms.ToTensor(), transforms.Normalize(MEAN, STD)]
    )


class ImageView(Dataset):
    """共享原始图像，但各自持有索引与变换，防止增强污染验证集。"""

    def __init__(self, base, indices, transform):
        self.base = base
        self.indices = tuple(int(i) for i in indices)
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        image, label = self.base[self.indices[index]]
        return self.transform(image), label


def load_train_val(root=None, download=False, train_limit=12000, split_seed=42):
    root = data_root() if root is None else Path(root)
    if train_limit is not None and not 1 <= train_limit <= 45000:
        raise ValueError("train_limit 必须在 1–45000 之间，或为 None（使用全部训练池）")
    try:
        base = datasets.CIFAR10(str(root), train=True, download=download)
    except RuntimeError as error:
        raise RuntimeError(
            f"CIFAR-10 数据不可用：{root.resolve()}。"
            "首次联网时设置 DOWNLOAD=True；离线时复制 data/cifar-10-batches-py/。"
        ) from error
    permutation = torch.randperm(
        50000, generator=torch.Generator().manual_seed(split_seed)
    )
    pool_indices, val_indices = permutation[:45000], permutation[45000:]
    train_indices = pool_indices if train_limit is None else pool_indices[:train_limit]
    plain = build_transform(False)
    return {
        "train": ImageView(base, train_indices, plain),
        "train_eval": ImageView(base, train_indices, plain),
        "val": ImageView(base, val_indices, plain),
        "train_pool_indices": tuple(pool_indices.tolist()),
        "split_seed": split_seed,
    }


def with_augmentation(dataset):
    return ImageView(dataset.base, dataset.indices, build_transform(True))


def load_test(root=None, download=False):
    root = data_root() if root is None else Path(root)
    return datasets.CIFAR10(
        str(root), train=False, transform=build_transform(False), download=download
    )


def make_loader(dataset, batch_size=128, shuffle=False, seed=2026):
    # 单进程兼容 macOS/Windows notebook；独立 generator 控制顺序。
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        drop_last=False,
        generator=torch.Generator().manual_seed(seed),
    )


def snapshot(model):
    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }


def synchronize(device):
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def evaluate(model, loader, device, details=False, error_limit=12):
    """按样本数汇总；details=True 时在同一次遍历中收集混淆矩阵和错分图。"""
    model.eval()
    criterion = nn.CrossEntropyLoss(reduction="sum")
    loss_sum, correct, n = 0.0, 0, 0
    confusion = torch.zeros(10, 10, dtype=torch.long)
    errors = []
    with torch.no_grad():
        for x, y in loader:
            scores = model(x.to(device))
            targets = y.to(device)
            loss = criterion(scores, targets)
            if not torch.isfinite(loss):
                raise FloatingPointError("评估损失非有限，请检查模型、输入和学习率")
            pred = scores.argmax(dim=1)
            loss_sum += loss.item()
            correct += (pred == targets).sum().item()
            n += len(y)
            if details:
                pred_cpu = pred.cpu()
                confusion += torch.bincount(y * 10 + pred_cpu, minlength=100).reshape(
                    10, 10
                )
                if len(errors) < error_limit:
                    confidence = scores.softmax(dim=1).max(dim=1).values.cpu()
                    for i in (pred_cpu != y).nonzero(as_tuple=True)[0].tolist():
                        errors.append(
                            (
                                x[i].clone(),
                                int(y[i]),
                                int(pred_cpu[i]),
                                float(confidence[i]),
                            )
                        )
                        if len(errors) == error_limit:
                            break
    if n == 0:
        raise ValueError("不能评估空数据集")
    result = {"loss": loss_sum / n, "accuracy": correct / n, "n": n}
    if details:
        result.update(confusion=confusion, errors=errors)
    return result


def train_epoch(model, loader, optimizer, device):
    model.train()
    criterion = nn.CrossEntropyLoss()
    loss_sum, correct, n = 0.0, 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad(set_to_none=True)
        scores = model(x)
        loss = criterion(scores, y)
        if not torch.isfinite(loss):
            raise FloatingPointError("训练损失非有限，请检查输入与学习率")
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * len(y)
        correct += (scores.argmax(1) == y).sum().item()
        n += len(y)
    if n == 0:
        raise ValueError("不能训练空数据集")
    return {"loss": loss_sum / n, "accuracy": correct / n, "n": n}


def fit_model(
    model,
    train_set,
    train_eval_set,
    val_set,
    *,
    device="cpu",
    epochs=8,
    lr=0.001,
    batch_size=128,
    order_seed=2026,
    name="model",
):
    """Adam 训练；同一固定模型评估无增强训练集和验证集，按验证结果保存最佳轮次。"""
    if epochs < 1:
        raise ValueError("epochs 至少为 1")
    if len(train_set) != len(train_eval_set):
        raise ValueError("训练和训练评估视图必须包含相同数量的样本")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_loader = make_loader(train_set, batch_size, True, order_seed)
    train_eval_loader = make_loader(train_eval_set, batch_size)
    val_loader = make_loader(val_set, batch_size)
    history, best_key, best_state, best_epoch = [], None, None, None
    synchronize(device)
    started = time.perf_counter()
    for epoch in range(epochs + 1):
        online = (
            None if epoch == 0 else train_epoch(model, train_loader, optimizer, device)
        )
        train_metrics = evaluate(model, train_eval_loader, device)
        val_metrics = evaluate(model, val_loader, device)
        synchronize(device)
        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_accuracy": train_metrics["accuracy"],
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "seconds": time.perf_counter() - started,
            "online_loss": None if online is None else online["loss"],
        }
        history.append(row)
        key = (val_metrics["accuracy"], -val_metrics["loss"])
        if best_key is None or key > best_key:
            best_key, best_state, best_epoch = key, snapshot(model), epoch
        print(
            f"{name:12s} epoch {epoch:2d}: train={row['train_accuracy']:.2%}, "
            f"val={row['val_accuracy']:.2%}, val loss={row['val_loss']:.4f}, "
            f"elapsed={row['seconds']:.1f}s",
            flush=True,
        )
    model.load_state_dict(best_state)
    model.eval()
    synchronize(device)
    return {
        "name": name,
        "model": model,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_accuracy": best_key[0],
        "best_val_loss": -best_key[1],
        "parameters": sum(p.numel() for p in model.parameters()),
        "seconds": time.perf_counter() - started,
        "train_n": len(train_set),
        "val_n": len(val_set),
        "epochs": epochs,
        "lr": lr,
        "batch_size": batch_size,
        "order_seed": order_seed,
    }


def display_image(image):
    """将标准化后的单张图像转换为适合显示的 NumPy 数组。

    Args:
        image (torch.Tensor): 形状为 (C, H, W) 的浮点图像张量，
            不含 batch 维度。图像应已使用 MEAN 和 STD 进行标准化：
            image = (原始像素值 - MEAN) / STD。

    Returns:
        numpy.ndarray: 形状为 (H, W, C) 的图像数组。先通过
            image * STD + MEAN 反标准化，再将数值裁剪到 [0, 1]，
            可直接传给 matplotlib 的 imshow 显示。

    Notes:
        显示过程不记录梯度，并将数据移到 CPU。
        本函数不修改传入的图像张量。
    """
    mean = image.new_tensor(MEAN)[:, None, None]
    std = image.new_tensor(STD)[:, None, None]
    return (
        (image.detach().cpu() * std.cpu() + mean.cpu())
        .clamp(0, 1)
        .permute(1, 2, 0)
        .numpy()
    )


def plot_samples(
    images, labels, predictions=None, probabilities=None, title="CIFAR-10"
):
    n = min(len(images), 12)
    if n == 0:
        return None
    cols, rows = min(n, 6), (n + 5) // 6
    row_height = 3.3 if rows > 1 else 2.8
    fig, axes = plt.subplots(
        rows, cols, figsize=(2.3 * cols, row_height * rows), squeeze=False
    )
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i >= n:
            continue
        ax.imshow(display_image(images[i]))
        text = f"true: {CLASSES[int(labels[i])]}"
        if predictions is not None:
            text += f"\npred: {CLASSES[int(predictions[i])]}"
        if probabilities is not None:
            text += f" ({float(probabilities[i]):.2f})"
        ax.set_title(text, fontsize=9)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.94), h_pad=3.0 if rows > 1 else 1.08)
    plt.show()
    return fig


def plot_channels(image):
    rgb = display_image(image)
    fig, axes = plt.subplots(1, 4, figsize=(11, 3))
    axes[0].imshow(rgb)
    axes[0].set_title("RGB")
    for c, name in enumerate(("R", "G", "B")):
        axes[c + 1].imshow(rgb[:, :, c], cmap="gray", vmin=0, vmax=1)
        axes[c + 1].set_title(f"{name} intensity")
    for ax in axes:
        ax.axis("off")
    fig.tight_layout()
    plt.show()
    return fig


def plot_history(results):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for name, result in results.items():
        h = result["history"]
        epochs = [r["epoch"] for r in h]
        for ax, metric in zip(axes, ("loss", "accuracy")):
            ax.plot(
                epochs, [r[f"train_{metric}"] for r in h], "--", label=f"{name} train"
            )
            ax.plot(epochs, [r[f"val_{metric}"] for r in h], label=f"{name} val")
    for ax, name in zip(axes, ("cross entropy", "accuracy")):
        ax.set_xlabel("epoch (0 = before training)")
        ax.set_ylabel(name)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    plt.show()
    return fig


def plot_confusion(confusion):
    totals = confusion.sum(1, keepdim=True)
    rates = confusion.float() / totals.clamp_min(1)
    fig, ax = plt.subplots(figsize=(9, 7))
    im = ax.imshow(rates.numpy(), vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(range(10), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(10), CLASSES)
    ax.set_xlabel("predicted class")
    ax.set_ylabel("true class")
    ax.set_title("Confusion matrix (each row normalized)")
    for i in range(10):
        for j in range(10):
            ax.text(
                j,
                i,
                f"{rates[i, j]:.0%}",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if rates[i, j] > 0.5 else "black",
            )
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    plt.show()
    return fig


def print_comparison(results):
    print(
        f"{'name':14s} {'params':>9s} {'epoch':>6s} {'val acc':>9s} {'val loss':>9s} {'seconds':>9s}"
    )
    for name, r in results.items():
        print(
            f"{name:14s} {r['parameters']:9,d} {r['best_epoch']:6d} "
            f"{r['best_val_accuracy']:9.2%} {r['best_val_loss']:9.4f} {r['seconds']:9.1f}"
        )
