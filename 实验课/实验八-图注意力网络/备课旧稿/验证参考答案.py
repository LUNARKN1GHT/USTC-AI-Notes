"""数学核验：python 验证参考答案.py

完整执行并保留输出：python 验证参考答案.py --execute
再运行预先规定的另外两个种子：python 验证参考答案.py --repeat
应使用包含 PyTorch、nbformat、nbclient、ipykernel 的 ai-lab 环境。
"""
from __future__ import annotations

import argparse
import ast
import copy
import json
import os
import platform
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "lab7-mpl"))
os.environ.setdefault("IPYTHONDIR", str(Path(tempfile.gettempdir()) / "lab7-ipython"))
os.environ.setdefault("XDG_CACHE_HOME", str(Path(tempfile.gettempdir()) / "lab7-cache"))

import nbformat
import numpy as np
import torch
from torch.nn import functional as F

import lab_utils as u


def definitions():
    notebook = nbformat.read(ROOT / "实验七_参考答案.ipynb", as_version=4)
    namespace = {"__name__": "lab7_reference"}
    for cell in notebook.cells:
        if cell.cell_type == "code" and set(cell.metadata.get("tags", [])) & {"setup", "definitions"}:
            exec(compile(cell.source, "reference_definitions", "exec"), namespace)
    return namespace


def check_notebooks():
    for name in ["实验七_学生版.ipynb", "实验七_参考答案.ipynb"]:
        notebook = nbformat.read(ROOT / name, as_version=4)
        nbformat.validate(notebook)
        for index, cell in enumerate(notebook.cells):
            assert not any(ord(c) < 32 and c not in "\n\r\t" for c in cell.source), (name, index)
            if cell.cell_type == "code":
                ast.parse(cell.source)
        if "学生版" in name:
            code = "\n".join(c.source for c in notebook.cells if c.cell_type == "code")
            assert code.count("raise NotImplementedError") == 4
            assert "FROZEN = False" in code
            assert all(not c.get("outputs") and c.get("execution_count") is None
                       for c in notebook.cells if c.cell_type == "code")
    print("PASS notebook 格式、代码语法、占位与控制字符检查")


def check_data(ns):
    raw, citations = u.read_raw_cora()
    paper_ids, classes, x, y, edges = ns["encode_cora"](raw, citations)
    assert len(raw) == 2708 and len(citations) == 5429
    assert x.shape == (2708, 1433) and edges.shape == (2, 13264)
    assert classes == sorted(set(row[-1] for row in raw))
    assert torch.equal(torch.bincount(y), torch.tensor([298, 418, 818, 426, 217, 180, 351]))
    pairs = set(map(tuple, edges.T.tolist()))
    assert len(pairs) == 13264 and all((j, i) in pairs for i, j in pairs)
    assert all((i, i) in pairs for i in range(2708))
    assert torch.isfinite(x).all()
    assert torch.allclose(x.sum(1)[x.sum(1) > 0], torch.ones((x.sum(1) > 0).sum()))
    assert torch.equal(u.normalize_features(torch.zeros(2, 3)), torch.zeros(2, 3))
    split = u.stratified_split(y)
    assert [len(split[k]) for k in ["train", "val", "test"]] == [140, 500, 2068]
    combined = torch.cat(list(split.values()))
    assert combined.numel() == combined.unique().numel() == 2708
    assert torch.equal(torch.bincount(y[split["train"]]), torch.full((7,), 20))
    assert all(torch.equal(split[k], u.stratified_split(y)[k]) for k in split)
    assert not torch.equal(split["train"], u.stratified_split(y, seed=43)["train"])
    print("PASS 本地数据、零行归一化、双向去重、自环及确定性划分")
    return paper_ids, classes, x, y, edges, split


def check_attention(ns):
    Layer = ns["GATLayer"]
    torch.manual_seed(13)
    n, heads = 5, 2
    adjacency = torch.eye(n, dtype=torch.bool)
    # 故意使用不对称小图，以暴露接收／发送方向错误。
    adjacency[0, 1] = adjacency[0, 4] = adjacency[2, 0] = adjacency[4, 2] = True
    edges = torch.stack(torch.where(adjacency))
    x = torch.randn(n, 3, dtype=torch.float64, requires_grad=True)
    layer = Layer(3, 2, heads=heads, concat=True, dropout=0).double().eval()
    dense_layer = copy.deepcopy(layer)
    dense_x = x.detach().clone().requires_grad_()
    actual, alpha = layer(x, edges, return_attention=True)
    z = dense_layer.linear(dense_x).reshape(n, heads, 2)
    score_receiver = (z * dense_layer.attn_receiver).sum(-1)
    score_sender = (z * dense_layer.attn_sender).sum(-1)
    scores = F.leaky_relu(score_receiver[:, None, :] + score_sender[None, :, :], 0.2)
    dense_alpha = scores.masked_fill(~adjacency[..., None], -torch.inf).softmax(dim=1)
    expected = torch.einsum("ijh,jhf->ihf", dense_alpha, z).reshape(n, -1)
    assert torch.allclose(actual, expected, atol=1e-10)
    assert torch.allclose(u.dense_attention(alpha, edges, n), dense_alpha, atol=1e-10)
    assert torch.allclose(dense_alpha.sum(1), torch.ones(n, heads, dtype=torch.float64))
    loss_weights = torch.randn_like(actual)
    (actual * loss_weights).sum().backward()
    (expected * loss_weights).sum().backward()
    assert torch.allclose(x.grad, dense_x.grad, atol=1e-10)
    for a, b in zip(layer.parameters(), dense_layer.parameters()):
        assert torch.allclose(a.grad, b.grad, atol=1e-10)
    scores_for_gradcheck = torch.randn(edges.shape[1], heads, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(lambda values: u.edge_softmax(values, edges[0], n),
                                    (scores_for_gradcheck,), atol=1e-5)
    permutation = torch.tensor([3, 0, 4, 1, 2])
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(n)
    permuted = layer(x.detach()[permutation], inverse[edges])
    assert torch.allclose(permuted, actual.detach()[permutation], atol=1e-10)
    uniform = Layer(3, 2, heads=2, concat=False, attention="uniform", dropout=0).double().eval()
    uniform.load_state_dict(layer.state_dict())
    uniform_z = uniform.linear(x).reshape(n, heads, 2)
    uniform_result = uniform(x, edges)
    assert torch.allclose(uniform_result[0], uniform_z[[0, 1, 4]].mean(0).mean(0))
    identity_result, identity_alpha = layer(x.detach(), u.self_edges(n), return_attention=True)
    assert torch.all(identity_alpha == 1)
    assert torch.allclose(identity_result, layer.linear(x.detach()))
    # 输入只有自身的节点 3，不应受其他节点扰动影响。
    changed = x.detach().clone()
    changed[0] += 100
    assert torch.allclose(layer(changed, edges)[3], actual.detach()[3])
    print("PASS 不对称图稠密对照、参数与输入梯度、有限差分、置换等变性、等权与自环")


def check_training(ns):
    torch.manual_seed(29)
    GAT = ns["GAT"]
    x = torch.randn(5, 3)
    edges = torch.tensor([[0, 0, 1, 1, 2, 2, 3, 3, 4, 4],
                          [0, 1, 1, 2, 2, 3, 3, 4, 4, 0]])
    train, val = torch.tensor([0, 2]), torch.tensor([1, 4])
    train_y, val_y = torch.tensor([1, 0]), torch.tensor([0, 1])
    model = GAT(3, 2, hidden=4, heads=2, dropout=0)
    manual, altered_validation = copy.deepcopy(model), copy.deepcopy(model)
    optimizer = torch.optim.Adam(manual.parameters(), lr=0.005, weight_decay=0.0005)
    optimizer.zero_grad(set_to_none=True)
    F.cross_entropy(manual(x, edges)[train], train_y).backward()
    optimizer.step()
    run = u.fit_model(model, x, edges, train, train_y, val, val_y, epochs=1, log_every=0)
    u.fit_model(altered_validation, x, edges, train, train_y, val, 1 - val_y, epochs=1, log_every=0)
    for key, value in model.state_dict().items():
        assert torch.equal(value, manual.state_dict()[key]), key
        assert torch.equal(value, altered_validation.state_dict()[key]), key
    frozen = u.snapshot(model)
    first = next(model.parameters())
    with torch.no_grad():
        first.add_(1)
    assert not torch.equal(frozen["layer1.linear.weight"], first)
    model.load_state_dict(frozen)
    longer = u.fit_model(model, x, edges, train, train_y, val, val_y, epochs=6, log_every=0)
    best = max(longer["history"], key=lambda r: (r["val_acc"], -r["val_loss"]))
    assert longer["best_epoch"] == best["epoch"]
    with torch.no_grad():
        logits = model(x, edges)
        assert abs(float(F.cross_entropy(logits[val], val_y)) - best["val_loss"]) < 1e-7
    groups = {group: ns["make_model"](group) for group in u.GROUPS}
    for name in ["layer1.linear.weight", "layer2.linear.weight"]:
        assert all(torch.equal(groups["A"].state_dict()[name], groups[g].state_dict()[name]) for g in "BC")
    assert u.parameter_counts(groups["A"]) == {"stored": 92302, "trainable": 92160}
    assert u.parameter_counts(groups["C"]) == {"stored": 92302, "trainable": 92302}
    metric = u.classification_metrics(F.one_hot(torch.tensor([0, 1, 1, 1]), 3).float(),
                                       torch.tensor([0, 0, 1, 2]), n_classes=3)
    assert metric["accuracy"] == 0.5 and abs(metric["macro_f1"] - 7 / 18) < 1e-12
    original = torch.tensor([[0, 0, 1, 1], [0, 1, 0, 1]])
    assert torch.equal(u.remove_undirected_edge(original, 0, 1), u.self_edges(2))
    assert original.shape == (2, 4)
    print("PASS 训练掩码、验证标签不参与更新、快照独立、最佳轮次恢复、共享初值与指标")


def execute_notebook():
    from nbclient import NotebookClient
    from jupyter_client import KernelManager

    notebook = nbformat.read(ROOT / "实验七_参考答案.ipynb", as_version=4)
    manager = KernelManager(kernel_name="python3")
    # 使用启动验证脚本的解释器，不依赖用户默认 kernel 的 Python 路径。
    manager.kernel_spec.argv[0] = sys.executable
    client = NotebookClient(notebook, km=manager, timeout=600,
                            resources={"metadata": {"path": str(ROOT)}})
    client.on_cell_start = lambda cell, cell_index, **kwargs: print(
        f"EXEC cell {cell_index + 1}/{len(notebook.cells)}: {cell.source.splitlines()[0][:70]}", flush=True)
    try:
        client.execute(env=dict(os.environ))
    finally:
        if manager.has_kernel:
            manager.shutdown_kernel(now=True)
        manager.cleanup_resources()
    assert all(c.execution_count is not None and not any(o.output_type == "error" for o in c.outputs)
               for c in notebook.cells if c.cell_type == "code")
    nbformat.write(notebook, ROOT / "实验七_参考答案.ipynb")
    report = json.loads((ROOT / "输出" / "运行记录.json").read_text())
    if report["config"]["epochs"] == 300 and report["config"]["seed"] == 42:
        (ROOT / "参考运行记录.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print("PASS 参考 notebook 已从头执行并保留全部输出")
    for group, run in report["runs"].items():
        print(group, "best_epoch", run["best_epoch"], "val", run["best_val_acc"],
              "test", run["test"]["accuracy"], "seconds", round(run["elapsed_seconds"], 1))


def repeat_experiments(ns, data):
    paper_ids, classes, x, y, edges, split = data
    assert ns["CONFIG"]["epochs"] == 300 and ns["DEVICE"].type == "cpu"
    main = json.loads((ROOT / "参考运行记录.json").read_text())
    assert main["config"] == ns["CONFIG"]
    assert main["environment"] == ns["ENVIRONMENT"], "重复记录需要同一运行环境"
    all_records = [{"seed": 42, "group": group, "best_epoch": run["best_epoch"],
                    "val_accuracy": run["best_val_acc"], "test_accuracy": run["test"]["accuracy"],
                    "test_macro_f1": run["test"]["macro_f1"], "seconds": run["elapsed_seconds"]}
                   for group, run in main["runs"].items()]
    # 种子事先固定，不根据结果增加或筛选种子，不变更配置。
    for seed in [2026, 3407]:
        for group in u.GROUPS:
            print(f"REPEAT seed={seed} group={group}", flush=True)
            model = ns["make_model"](group, seed=seed)
            group_edges = u.edges_for_group(group, edges, len(x))
            run = u.fit_model(model, x, group_edges, split["train"], y[split["train"]],
                              split["val"], y[split["val"]], epochs=300, log_every=100)
            with torch.no_grad():
                metrics = u.classification_metrics(model(x, group_edges)[split["test"]], y[split["test"]])
            all_records.append({"seed": seed, "group": group, "best_epoch": run["best_epoch"],
                                "val_accuracy": run["best_val_acc"], "test_accuracy": metrics["accuracy"],
                                "test_macro_f1": metrics["macro_f1"], "seconds": run["elapsed_seconds"]})
    summary = {}
    for group in u.GROUPS:
        rows = [r for r in all_records if r["group"] == group]
        summary[group] = {}
        for key in ["val_accuracy", "test_accuracy", "test_macro_f1", "seconds"]:
            values = [r[key] for r in rows]
            summary[group][key] = {"mean": float(np.mean(values)), "sample_std": float(np.std(values, ddof=1))}
    report = {"seeds": [42, 2026, 3407], "config": main["config"],
              "environment": main["environment"], "split": "split_seed=42, 140/500/2068",
              "note": "同一划分的初始化与训练随机性；标准差采用 ddof=1，不代表跨划分稳定性。",
              "records": all_records, "summary": summary}
    (ROOT / "重复运行记录.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--repeat", action="store_true")
    args = parser.parse_args()
    check_notebooks()
    ns = definitions()
    data = check_data(ns)
    check_attention(ns)
    check_training(ns)
    if args.execute:
        execute_notebook()
    if args.repeat:
        repeat_experiments(ns, data)
