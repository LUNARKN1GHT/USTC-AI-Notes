"""实验七公共工具。edge_index[0] 为接收节点，edge_index[1] 为发送节点。"""
from __future__ import annotations

import copy
import hashlib
import json
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parent
GROUPS = {"A": "self", "B": "uniform", "C": "learned"}
SHORT_NAMES = ["Case", "Genetic", "Neural", "Prob.", "RL", "Rule", "Theory"]


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def read_raw_cora(root=None):
    """保留原始字符串 ID。数据编码与双向建图由学生完成。"""
    root = Path(root) if root is not None else ROOT / "cora"
    contents = [line.split() for line in (root / "cora.content").read_text().splitlines() if line.strip()]
    citations = [tuple(line.split()) for line in (root / "cora.cites").read_text().splitlines() if line.strip()]
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
    allocation[order[:val_size - int(allocation.sum())]] += 1
    val = [pool[:count] for pool, count in zip(pools, allocation)]
    test = [pool[count:] for pool, count in zip(pools, allocation)]
    return {name: torch.cat(parts).sort().values for name, parts in
            [("train", train), ("val", val), ("test", test)]}


def save_split(split, paper_ids, class_names, directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    payload = {"seed": 42, "classes": class_names, "indices": {}, "paper_ids": {}}
    for name, indices in split.items():
        payload["indices"][name] = indices.tolist()
        payload["paper_ids"][name] = [paper_ids[i] for i in indices.tolist()]
    payload["source_sha256"] = {
        name: hashlib.sha256((ROOT / "cora" / name).read_bytes()).hexdigest()
        for name in ["cora.content", "cora.cites"]
    }
    (directory / "split_seed42.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")


def self_edges(n, device=None):
    nodes = torch.arange(n, device=device)
    return torch.stack((nodes, nodes))


def edges_for_group(group, edges, n):
    assert group in GROUPS
    return self_edges(n, edges.device) if group == "A" else edges


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


def sum_messages(messages, receiver, n):
    """消息形状 E×H×F，沿接收节点求和。"""
    output = messages.new_zeros((n,) + messages.shape[1:])
    return output.index_add_(0, receiver, messages)


def dense_attention(alpha, edges, n):
    """仅供小图检查：N×N×H，非边位置为零。"""
    result = alpha.new_zeros((n, n, alpha.shape[1]))
    result[edges[0], edges[1]] = alpha
    return result


def snapshot(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def parameter_counts(model):
    return {"stored": sum(p.numel() for p in model.parameters()),
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}


def synchronize(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def classification_metrics(logits, labels, n_classes=7):
    predictions, labels = logits.argmax(dim=-1).cpu(), labels.cpu()
    confusion = torch.bincount(labels * n_classes + predictions,
                               minlength=n_classes ** 2).reshape(n_classes, n_classes)
    true_positive = confusion.diag().double()
    support = confusion.sum(1).double()
    predicted = confusion.sum(0).double()
    recall = true_positive / support.clamp_min(1)
    f1 = 2 * true_positive / (support + predicted).clamp_min(1)
    return {"accuracy": float(true_positive.sum() / support.sum()),
            "macro_f1": float(f1.mean()), "recall": recall.tolist(),
            "confusion": confusion.tolist()}


def fit_model(model, x, edges, train_idx, train_y, val_idx, val_y,
              epochs=300, lr=0.005, weight_decay=0.0005, log_every=100):
    """接口只接收训练和验证标签；训练损失只包含 train_idx。

    每轮一次全图更新。所有曲线均在更新后的评估模式下重算。
    耗时包含训练、逐轮评估和最佳参数快照，不含绘图和最终测试。
    """
    assert epochs > 0
    device = x.device
    train_idx, val_idx = train_idx.to(device), val_idx.to(device)
    train_y, val_y = train_y.to(device), val_y.to(device)
    assert not torch.isin(train_idx, val_idx).any()
    optimizer = torch.optim.Adam((p for p in model.parameters() if p.requires_grad),
                                 lr=lr, weight_decay=weight_decay)
    history, best_key, best_state, best_epoch = [], None, None, None
    synchronize(device)
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        logits = model(x, edges)
        loss = F.cross_entropy(logits[train_idx], train_y)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"第 {epoch} 轮训练损失不是有限值")
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            logits = model(x, edges)
            record = {"epoch": epoch,
                      "train_loss": float(F.cross_entropy(logits[train_idx], train_y)),
                      "val_loss": float(F.cross_entropy(logits[val_idx], val_y)),
                      "train_acc": float((logits[train_idx].argmax(-1) == train_y).float().mean()),
                      "val_acc": float((logits[val_idx].argmax(-1) == val_y).float().mean())}
        assert all(np.isfinite(value) for value in record.values())
        history.append(record)
        key = (record["val_acc"], -record["val_loss"])
        if best_key is None or key > best_key:
            best_key, best_state, best_epoch = key, snapshot(model), epoch
        if log_every and (epoch == 1 or epoch % log_every == 0 or epoch == epochs):
            print(f"epoch={epoch:3d} train={record['train_acc']:.3f} "
                  f"val={record['val_acc']:.3f} val_loss={record['val_loss']:.3f}", flush=True)
    synchronize(device)
    elapsed = time.perf_counter() - start
    model.load_state_dict(best_state)
    model.eval()
    best_record = history[best_epoch - 1]
    return {"model": model, "history": history, "best_epoch": best_epoch,
            "best_val_acc": best_record["val_acc"], "best_val_loss": best_record["val_loss"],
            "elapsed_seconds": elapsed, "parameters": parameter_counts(model)}


def print_comparison(runs):
    print("group  best_epoch  val_acc  val_loss  trainable/stored  seconds")
    for group, run in runs.items():
        p = run["parameters"]
        print(f"{group:5s}  {run['best_epoch']:10d}  {run['best_val_acc']:.4f}   "
              f"{run['best_val_loss']:.4f}   {p['trainable']}/{p['stored']}  {run['elapsed_seconds']:.1f}")


def plot_history(runs):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.7), constrained_layout=True)
    for group, run in runs.items():
        epochs = [row["epoch"] for row in run["history"]]
        for ax, suffix in zip(axes, ["loss", "acc"]):
            line, = ax.plot(epochs, [r[f"val_{suffix}"] for r in run["history"]], label=f"{group} val")
            ax.plot(epochs, [r[f"train_{suffix}"] for r in run["history"]],
                    color=line.get_color(), linestyle="--", alpha=0.7, label=f"{group} train")
            ax.set(xlabel="Completed updates", ylabel=suffix)
            ax.grid(alpha=0.2)
    axes[0].legend(ncol=2, fontsize=8)
    return fig


def select_validation_nodes(edges, val_idx, targets=(2, 5, 10)):
    """只根据验证索引与度数选样本，不查看类别、预测或注意力。"""
    degree = torch.bincount(edges[0].cpu(), minlength=int(edges.max()) + 1) - 1
    selected = []
    candidates = val_idx.cpu().tolist()
    for target in targets:
        available = [i for i in candidates if i not in selected and 1 <= degree[i] <= 12]
        if available:
            selected.append(min(available, key=lambda i: (abs(int(degree[i]) - target), i)))
    return selected


def plot_attention_neighborhood(node, edges, alpha, paper_ids, known_labels):
    """展示接收节点的全部一跳输入；不画邻居之间的边。测试标签不传入此函数。"""
    edges, alpha = edges.cpu(), alpha.detach().cpu()
    mask = edges[0] == node
    senders, weights = edges[1, mask], alpha[mask]
    order = torch.argsort(senders)
    senders, weights = senders[order], weights[order]
    outside = senders[senders != node].tolist()
    positions = {node: np.array([0.0, 0.0])}
    for k, sender in enumerate(outside):
        angle = 2 * np.pi * k / max(1, len(outside))
        positions[sender] = np.array([np.cos(angle), np.sin(angle)])
    fig, (ax, heat) = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    mean_weight = {int(sender): float(row.mean()) for sender, row in zip(senders, weights)}
    cmap = plt.get_cmap("tab10")
    for sender, position in positions.items():
        label = known_labels.get(sender)
        color = "lightgray" if label is None else cmap(label)
        if sender != node:
            ax.plot([0, position[0]], [0, position[1]], color="gray",
                    linewidth=1 + 6 * mean_weight[sender], alpha=0.65, zorder=1)
        ax.scatter(*position, s=240 if sender == node else 120, color=color, edgecolors="black", zorder=2)
        text = f"{paper_ids[sender]}\n{SHORT_NAMES[label] if label is not None else '?'}"
        ax.annotate(text, position, xytext=(5, 6), textcoords="offset points", fontsize=8)
    ax.set(xlim=(-1.45, 1.65), ylim=(-1.4, 1.4), title=f"Receiver {paper_ids[node]}: incoming star")
    ax.set_aspect("equal")
    ax.axis("off")
    image = heat.imshow(weights.numpy(), aspect="auto", cmap="viridis", vmin=0)
    heat.set(xticks=range(weights.shape[1]), xticklabels=range(1, weights.shape[1] + 1),
             yticks=range(len(senders)),
             yticklabels=[f"{paper_ids[i]}{' (self)' if i == node else ''}" for i in senders.tolist()],
             xlabel="Head", ylabel="Sender paper ID", title="Layer 1 attention before dropout")
    fig.colorbar(image, ax=heat, label="Weight")
    return fig


def remove_undirected_edge(edges, u, v):
    assert u != v, "局部检查保留自环"
    keep = ~(((edges[0] == u) & (edges[1] == v)) |
             ((edges[0] == v) & (edges[1] == u)))
    assert int((~keep).sum()) == 2, "预期一条已去重的无向关系"
    return edges[:, keep]


@torch.no_grad()
def edge_sensitivity(model, x, edges, node, label):
    """固定模型，分别移除第一层平均权重最高与最低的外部邻居关系。"""
    model.eval()
    logits, attention = model(x, edges, return_attention=True)
    mask = (edges[0] == node) & (edges[1] != node)
    senders = edges[1, mask]
    weights = attention[0][mask].mean(-1)
    assert len(senders) >= 2
    order = torch.argsort(weights, stable=True)
    before = logits[node].softmax(-1)
    records = []
    for name, position in [("highest", order[-1]), ("lowest", order[0])]:
        sender = int(senders[position])
        changed_edges = remove_undirected_edge(edges, node, sender)
        after = model(x, changed_edges)[node].softmax(-1)
        records.append({"removed": name, "receiver": node, "sender": sender,
                        "layer1_mean_attention": float(weights[position]),
                        "prediction_before": int(before.argmax()), "prediction_after": int(after.argmax()),
                        "true_class_prob_before": float(before[label]),
                        "true_class_prob_after": float(after[label]),
                        "probability_L1_change": float((after - before).abs().sum())})
    return records


def plot_confusions(test_results):
    fig, axes = plt.subplots(1, len(test_results), figsize=(5 * len(test_results), 4.6),
                             squeeze=False, constrained_layout=True)
    for ax, (group, metrics) in zip(axes[0], test_results.items()):
        cm = np.asarray(metrics["confusion"])
        ax.imshow(cm, cmap="Blues")
        for i in range(7):
            for j in range(7):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set(xticks=range(7), yticks=range(7), xticklabels=SHORT_NAMES, yticklabels=SHORT_NAMES,
               xlabel="Predicted", ylabel="True", title=f"{group}: test confusion (counts)")
        ax.tick_params(axis="x", rotation=45)
    return fig


def save_run_report(path, runs, test_results, config, environment, split):
    payload = {"config": config, "environment": environment,
               "split_sizes": {key: len(value) for key, value in split.items()}, "runs": {}}
    for group, run in runs.items():
        payload["runs"][group] = {key: copy.deepcopy(value) for key, value in run.items() if key != "model"}
        payload["runs"][group]["test"] = test_results[group]
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload
