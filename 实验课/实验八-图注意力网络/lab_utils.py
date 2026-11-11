"""实验八公共工具。边索引的第 0 行为接收节点，第 1 行为发送节点。"""

from __future__ import annotations

import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_raw_cora(root=None):
    """读取实验七沿用的 Cora 原始文件，保留字符串 ID。"""
    root = Path(root) if root is not None else ROOT / "cora"
    contents = [
        line.split()
        for line in (root / "cora.content").read_text().splitlines()
        if line.strip()
    ]
    citations = [
        tuple(line.split())
        for line in (root / "cora.cites").read_text().splitlines()
        if line.strip()
    ]
    assert all(len(row) == 1435 for row in contents)
    assert all(len(row) == 2 for row in citations)
    return contents, citations


def normalize_features(x):
    return x / x.sum(dim=1, keepdim=True).clamp_min(1)


def stratified_split(y, seed=42, per_class=20, val_size=500):
    """按剩余类别比例分配验证配额；最大余数法保证总数恰好为 500。"""
    y = y.cpu()
    generator = torch.Generator().manual_seed(seed)
    classes = torch.unique(y, sorted=True)
    pools, train = [], []
    for label in classes:
        indices = torch.where(y == label)[0]
        assert len(indices) > per_class
        indices = indices[torch.randperm(len(indices), generator=generator)]
        train.append(indices[:per_class])
        pools.append(indices[per_class:])
    sizes = np.array([len(pool) for pool in pools])
    assert 0 < val_size < sizes.sum()
    quotas = sizes * val_size / sizes.sum()
    allocation = np.floor(quotas).astype(int)
    order = np.argsort(-(quotas - allocation), kind="stable")
    allocation[order[: val_size - int(allocation.sum())]] += 1
    val = [pool[:count] for pool, count in zip(pools, allocation)]
    test = [pool[count:] for pool, count in zip(pools, allocation)]
    return {
        name: torch.cat(parts).sort().values
        for name, parts in [("train", train), ("val", val), ("test", test)]
    }


def load_cora(root=None):
    """沿用实验七的节点顺序、特征归一化、双向建图和固定划分。"""
    contents, citations = read_raw_cora(root)
    paper_ids = [row[0] for row in contents]
    assert len(set(paper_ids)) == len(paper_ids), "论文 ID 应唯一"
    id_to_idx = {paper_id: i for i, paper_id in enumerate(paper_ids)}

    class_names = sorted({row[-1] for row in contents})
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    y = torch.tensor([class_to_idx[row[-1]] for row in contents], dtype=torch.long)
    word_columns = [row[1:-1] for row in contents]
    x = torch.from_numpy(np.asarray(word_columns, dtype=np.float32))

    pairs = {(i, i) for i in range(len(paper_ids))}
    for cited, citing in citations:
        receiver, sender = id_to_idx[cited], id_to_idx[citing]
        pairs.add((receiver, sender))
        pairs.add((sender, receiver))
    edges = torch.tensor(sorted(pairs), dtype=torch.long).T.contiguous()
    return {
        "x": normalize_features(x),
        "y": y,
        "edges": edges,
        "split": stratified_split(y),
        "paper_ids": paper_ids,
        "class_names": class_names,
    }


def self_edges(n, device=None):
    nodes = torch.arange(n, device=device)
    return torch.stack((nodes, nodes))


def edge_softmax(scores, receiver, n):
    """对每个接收节点、每个头独立归一化；不构造 N×N×H 张量。"""
    assert scores.ndim == 2 and scores.shape[0] == receiver.numel()
    index = receiver[:, None].expand_as(scores)
    maximum = scores.new_full((n, scores.shape[1]), -torch.inf)
    maximum.scatter_reduce_(0, index, scores.detach(), reduce="amax", include_self=True)
    exp_scores = (scores - maximum[receiver]).exp()
    denominator = scores.new_zeros((n, scores.shape[1]))
    denominator.scatter_add_(0, index, exp_scores)
    return exp_scores / denominator[receiver]


def uniform_attention(receiver, n, heads, dtype):
    degree = torch.bincount(receiver, minlength=n).to(dtype=dtype)
    return (1 / degree[receiver])[:, None].expand(-1, heads)


def gcn_attention(edges, n, heads, dtype):
    """对称归一化的每边系数 1/sqrt(d_i d_j)，图已去重并包含自环。"""
    receiver, sender = edges
    degree = torch.bincount(receiver, minlength=n).to(dtype=dtype)
    inv_sqrt = degree.clamp_min(1).rsqrt()
    coefficients = inv_sqrt[receiver] * inv_sqrt[sender]
    return coefficients[:, None].expand(-1, heads)


def sum_messages(messages, receiver, n):
    """消息形状 E×H×F，沿接收节点求和。"""
    output = messages.new_zeros((n,) + messages.shape[1:])
    return output.index_add_(0, receiver, messages)


def dense_attention(alpha, edges, n):
    """仅供小图检查：N×N×H，非边位置为零。"""
    result = alpha.new_zeros((n, n, alpha.shape[1]))
    result[edges[0], edges[1]] = alpha
    return result


def remove_undirected_edge(edges, u, v):
    assert u != v, "局部检查保留自环"
    keep = ~(((edges[0] == u) & (edges[1] == v)) | ((edges[0] == v) & (edges[1] == u)))
    assert int((~keep).sum()) == 2, "预期一条已去重的无向关系"
    return edges[:, keep]


def select_validation_nodes(edges, val_idx, limit=20):
    """在外部度数至少为 2 的验证节点中，按度数和节点下标排序并等距取样。"""
    if limit <= 0 or val_idx.numel() == 0:
        return []
    receiver, sender = edges.cpu()
    degree = torch.bincount(
        receiver[receiver != sender], minlength=int(val_idx.max()) + 1
    ).tolist()
    candidates = sorted(
        (node for node in val_idx.tolist() if degree[node] >= 2),
        key=lambda node: (degree[node], node),
    )
    count = min(limit, len(candidates))
    # 把候选排名均分成 count 段，从每段取中点对应的节点。
    ranks = [(2 * i + 1) * len(candidates) // (2 * count) for i in range(count)]
    return [candidates[rank] for rank in ranks]


def snapshot(model):
    return {
        key: value.detach().cpu().clone() for key, value in model.state_dict().items()
    }


def parameter_counts(model):
    return {
        "stored": sum(p.numel() for p in model.parameters()),
        "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad),
    }


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def classification_metrics(logits, labels, n_classes=7):
    predictions, labels = logits.argmax(dim=-1).cpu(), labels.cpu()
    confusion = torch.bincount(
        labels * n_classes + predictions, minlength=n_classes**2
    ).reshape(n_classes, n_classes)
    true_positive = confusion.diag().double()
    support = confusion.sum(1).double()
    predicted = confusion.sum(0).double()
    recall = true_positive / support.clamp_min(1)
    f1 = 2 * true_positive / (support + predicted).clamp_min(1)
    return {
        "accuracy": float(true_positive.sum() / support.sum()),
        "macro_f1": float(f1.mean()),
        "recall": recall.tolist(),
        "confusion": confusion.tolist(),
    }


def fit_model(
    model,
    x,
    operator,
    train_idx,
    train_y,
    val_idx,
    val_y,
    epochs=200,
    lr=0.01,
    weight_decay=0.0005,
    log_every=100,
):
    """接口只接收训练和验证标签；训练损失只包含 train_idx。

    每轮一次全图更新。所有曲线均在更新后的评估模式下重算。
    耗时包含训练、逐轮评估和最佳参数快照，不含绘图和最终测试。
    """
    assert epochs > 0
    device = x.device
    train_idx, val_idx = train_idx.to(device), val_idx.to(device)
    train_y, val_y = train_y.to(device), val_y.to(device)
    assert not torch.isin(train_idx, val_idx).any()
    optimizer = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad),
        lr=lr,
        weight_decay=weight_decay,
    )
    history, best_key, best_state, best_epoch = [], None, None, None
    synchronize(device)
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x, operator)
        loss = F.cross_entropy(logits[train_idx], train_y)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"第 {epoch} 轮训练损失不是有限值")
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            logits = model(x, operator)
            record = {
                "epoch": epoch,
                "train_loss": float(F.cross_entropy(logits[train_idx], train_y)),
                "val_loss": float(F.cross_entropy(logits[val_idx], val_y)),
                "train_acc": float(
                    (logits[train_idx].argmax(-1) == train_y).float().mean()
                ),
                "val_acc": float((logits[val_idx].argmax(-1) == val_y).float().mean()),
            }
        assert all(np.isfinite(value) for value in record.values())
        history.append(record)
        key = (record["val_acc"], -record["val_loss"])
        if best_key is None or key > best_key:
            best_key, best_state, best_epoch = key, snapshot(model), epoch
        if log_every and (epoch == 1 or epoch % log_every == 0 or epoch == epochs):
            print(
                f"epoch={epoch:3d} train={record['train_acc']:.3f} "
                f"val={record['val_acc']:.3f} val_loss={record['val_loss']:.3f}",
                flush=True,
            )
    synchronize(device)
    elapsed = time.perf_counter() - start
    model.load_state_dict(best_state)
    model.eval()
    best_record = history[best_epoch - 1]
    return {
        "model": model,
        "history": history,
        "best_epoch": best_epoch,
        "best_val_acc": best_record["val_acc"],
        "best_val_loss": best_record["val_loss"],
        "elapsed_seconds": elapsed,
        "parameters": parameter_counts(model),
    }


def plot_history(runs):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.7), constrained_layout=True)
    for group, run in runs.items():
        epochs = [row["epoch"] for row in run["history"]]
        for ax, suffix in zip(axes, ["loss", "acc"]):
            (line,) = ax.plot(
                epochs,
                [r[f"val_{suffix}"] for r in run["history"]],
                label=f"{group} val",
            )
            ax.plot(
                epochs,
                [r[f"train_{suffix}"] for r in run["history"]],
                color=line.get_color(),
                linestyle="--",
                alpha=0.7,
                label=f"{group} train",
            )
            ax.set(xlabel="Completed updates", ylabel=suffix)
            ax.grid(alpha=0.2)
    axes[0].legend(ncol=2, fontsize=8)
    return fig
