"""实验五教师提供工具：数据、参数快照与绘图；训练核心留在 notebook。"""

import os
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def seed_everything(seed=42):
    """给实验中每个位置设置随机种子

    Args:
        seed: 设置的随机种子
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def data_root():
    """设置实验数据的下载目录"""
    return Path(os.environ.get("MNIST_ROOT", "data"))


def load_train_val(root=None, download=False, split_seed=42):
    """加载训练数据

    Args:
        root (_type_, optional): 设置的数据存放目录. Defaults to None.
        download (bool, optional): 是否已下载数据. Defaults to False.
        split_seed (int, optional): 随机划分数据集使用的种子. Defaults to 42.

    Raises:
        RuntimeError: _description_
    """
    root = data_root() if root is None else Path(root)

    try:
        full = datasets.MNIST(
            root=str(root),
            train=True,
            transform=transforms.ToTensor(),
            download=download,
        )

    except RuntimeError as error:
        raise RuntimeError(
            f"MNIST 训练数据不可用，目录为 {root.resolve()}。"
            "联网时先将 DOWNLOAD=True 运行一次，或复制教师提供的 data/MNIST/raw 缓存。"
        ) from error

    train, val = random_split(
        full, [50000, 10000], generator=torch.Generator().manual_seed(split_seed)
    )

    return train, val


def load_test(root=None, download=False):
    """加载测试数据集"""
    root = data_root() if root is None else Path(root)
    return datasets.MNIST(
        str(root), train=False, transform=transforms.ToTensor(), download=download
    )


def make_loader(dataset, batch_size=128, shuffle=False, seed=42):
    # 单进程加载兼容 Windows/macOS 的 notebook；独立 generator 控制批次顺序。
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
        drop_last=False,
    )


def snapshot(model):
    """独立复制：后续 optimizer.step 不会修改这份状态。"""
    return {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }


def plot_samples(images, labels, predictions=None, probabilities=None, title="MNIST"):
    """数据集中的抽样"""

    n = min(len(images), 12)
    if n == 0:
        print("没有可展示的样本。")
        return None

    cols = min(6, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2 * cols, 2.3 * rows), squeeze=False)
    for i, ax in enumerate(axes.flat):
        ax.axis("off")
        if i >= n:
            continue
        ax.imshow(images[i].detach().cpu().squeeze(0), cmap="gray", vmin=0, vmax=1)
        text = f"true={int(labels[i])}"
        if predictions is not None:
            text += f", pred={int(predictions[i])}"
        if probabilities is not None:
            text += f"\np(pred)={float(probabilities[i]):.2f}"
        ax.set_title(text, fontsize=9)

    fig.suptitle(title)
    fig.tight_layout()
    plt.show()
    return fig


def plot_history(histories):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5))
    for name, history in histories.items():
        epochs = [row["epoch"] for row in history]
        axes[0].plot(
            epochs, [r["train_loss"] for r in history], "--", label=f"{name} train"
        )
        axes[0].plot(epochs, [r["val_loss"] for r in history], label=f"{name} val")
        axes[1].plot(
            epochs, [r["train_accuracy"] for r in history], "--", label=f"{name} train"
        )
        axes[1].plot(epochs, [r["val_accuracy"] for r in history], label=f"{name} val")
    for ax, ylabel in zip(axes, ["cross entropy", "accuracy"]):
        ax.set_xlabel("epoch (0 = before training)")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
        ax.legend(fontsize=8)
    fig.tight_layout()
    plt.show()
    return fig


def collect_errors(model, loader, device, limit=12):
    """教师提供：按固定顺序收集前 limit 张错分图像，不挑选最夸张的样本。"""
    model.eval()
    images, labels, predictions, probs = [], [], [], []
    with torch.no_grad():
        for x, y in loader:
            scores = model(x.to(device))
            p, pred = scores.softmax(dim=1).max(dim=1)
            pred, p = pred.cpu(), p.cpu()
            for i in (pred != y).nonzero(as_tuple=True)[0].tolist():
                images.append(x[i])
                labels.append(y[i])
                predictions.append(pred[i])
                probs.append(p[i])
                if len(images) == limit:
                    return tuple(
                        torch.stack(v) for v in (images, labels, predictions, probs)
                    )
    if not images:
        return None
    return tuple(torch.stack(v) for v in (images, labels, predictions, probs))
