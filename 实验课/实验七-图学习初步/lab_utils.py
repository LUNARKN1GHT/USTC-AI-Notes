"""实验七 GCN 公共工具：矩阵行表示接收节点，列表示发送节点。"""
from __future__ import annotations

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
SHORT_NAMES = ["Case", "Genetic", "Neural", "Prob.", "RL", "Rule", "Theory"]


def sparse_operator(edges, values, n):
    return torch.sparse_coo_tensor(edges, values, (n, n), check_invariants=True).coalesce()


def propagate(operator, x):
    return operator @ x if operator.layout == torch.strided else torch.sparse.mm(operator, x)


def identity_operator(n):
    return sparse_operator(self_edges(n), torch.ones(n), n)


def graph_audit(edges, n):
    edges = edges.cpu()
    external = edges[:, edges[0] != edges[1]]
    degree = torch.bincount(external[0], minlength=n)
    adjacent = [set() for _ in range(n)]
    for receiver, sender in external.T.tolist():
        adjacent[receiver].add(sender)
    unseen, sizes = set(range(n)), []
    while unseen:
        stack, size = [unseen.pop()], 0
        while stack:
            node = stack.pop()
            size += 1
            for neighbor in adjacent[node]:
                if neighbor in unseen:
                    unseen.remove(neighbor)
                    stack.append(neighbor)
        sizes.append(size)
    return dict(undirected_edges=external.shape[1] // 2,
                isolates=int((degree == 0).sum()), components=len(sizes),
                largest_component=max(sizes), degree=degree.tolist(),
                component_sizes=sorted(sizes, reverse=True))


def plot_graph_audit(audit):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), constrained_layout=True)
    axes[0].hist(audit["degree"], bins=30, color="steelblue")
    axes[0].set(xlabel="External degree", ylabel="Node count (log)", yscale="log")
    sizes = audit["component_sizes"][:10]
    axes[1].bar(range(1, len(sizes) + 1), sizes, color="darkcyan")
    axes[1].set(xlabel="Component rank", ylabel="Size (log)", yscale="log")
    return fig


def retain_undirected_edges(edges, n, keep=1.0, seed=2026):
    """以无向关系为单位，按同一固定乱序保留前缀，保留所有节点与自环。"""
    assert 0 <= keep <= 1
    assert edges.device.type == "cpu", "先在 CPU 固定生成实验图"
    pairs = edges[:, edges[0] < edges[1]]
    order = torch.randperm(pairs.shape[1], generator=torch.Generator().manual_seed(seed))
    chosen = pairs[:, order[:int(pairs.shape[1] * keep)]]
    result = torch.cat([chosen, chosen.flip(0), self_edges(n)], dim=1)
    keys = (result[0] * n + result[1]).unique(sorted=True)
    return torch.stack((keys // n, keys % n))


def case_key(spec):
    return f"{spec['model']}_L{spec['labels']}_E{round(spec['keep'] * 100)}"


def summarize_cases(records, field="best_val_acc"):
    grouped = {}
    for run in records.values():
        value = run[field] if field in run else run["test"][field]
        grouped.setdefault(case_key(run["spec"]), []).append(float(value))
    return {key: {"mean": float(np.mean(values)),
                  "sample_std": float(np.std(values, ddof=1)) if len(values) > 1 else 0.0,
                  "n": len(values)} for key, values in grouped.items()}


def print_validation_table(records):
    summary = summarize_cases(records)
    print("case                 val accuracy (mean ± sample std)  seeds")
    for key, row in summary.items():
        print(f"{key:22s} {row['mean']:.4f} ± {row['sample_std']:.4f}              {row['n']}")


def plot_factor(records, factor, models=("MLP", "GCN")):
    rows = {}
    for run in records.values():
        spec = run["spec"]
        if spec["model"] not in models:
            continue
        if factor == "labels" and spec["keep"] != 1:
            continue
        if factor == "keep" and spec["labels"] != 20:
            continue
        rows.setdefault((spec["model"], spec[factor]), []).append(run["best_val_acc"])
    fig, ax = plt.subplots(figsize=(7, 3.7), constrained_layout=True)
    for model in models:
        levels = sorted(level for (name, level) in rows if name == model)
        means = [np.mean(rows[model, level]) for level in levels]
        stds = [np.std(rows[model, level], ddof=1) if len(rows[model, level]) > 1 else 0 for level in levels]
        ax.errorbar(levels, means, yerr=stds, marker="o", capsize=4, label=model)
    ax.set(xlabel="Labels per class" if factor == "labels" else "Fraction of citation edges retained",
           ylabel="Validation accuracy", title="Fixed split; mean ± sample std across model seeds")
    ax.grid(alpha=0.2)
    ax.legend()
    return fig


def save_report(path, records, config, environment, split, label_sets, selection):
    payload = {"version": "GCN-160min", "config": config, "environment": environment,
               "split_sizes": {k: len(v) for k, v in split.items()},
               "label_sets": {str(k): v.tolist() for k, v in label_sets.items()},
               "selected_model_by_validation": selection,
               "runs": {key: {k: v for k, v in run.items() if k != "model"}
                        for key, run in records.items()},
               "validation_summary": summarize_cases(records),
               "test_accuracy_summary": summarize_cases(records, "accuracy"),
               "test_macro_f1_summary": summarize_cases(records, "macro_f1")}
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return payload

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


def fit_model(model, x, operator, train_idx, train_y, val_idx, val_y,
              epochs=200, lr=0.01, weight_decay=0.0005, log_every=100):
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
        logits = model(x, operator)
        loss = F.cross_entropy(logits[train_idx], train_y)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"第 {epoch} 轮训练损失不是有限值")
        loss.backward()
        optimizer.step()
        model.eval()
        with torch.no_grad():
            logits = model(x, operator)
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
