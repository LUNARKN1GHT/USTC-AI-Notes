"""教师维护：python 验证参考答案.py。无需下载数据；读取 notebook 的真实核心定义。

完整 MNIST 验证另需在本目录重启 kernel 从头运行参考 notebook。
"""

import ast
import copy
import json
import math
from pathlib import Path

import torch
from lab_utils import make_loader, seed_everything, snapshot
from torch import nn
from torch.utils.data import TensorDataset

ROOT = Path(__file__).resolve().parent


def notebook_definitions():
    notebooks = {}
    for name in ["学生版", "参考答案"]:
        nb = json.loads((ROOT / f"实验五_{name}.ipynb").read_text())
        for c in nb["cells"]:
            if c["cell_type"] == "code":
                ast.parse("".join(c["source"]))
        notebooks[name] = nb
    student, reference = notebooks["学生版"], notebooks["参考答案"]
    student_code = {c["id"]: c for c in student["cells"] if c["cell_type"] == "code"}
    reference_code = {c["id"]: c for c in reference["cells"] if c["cell_type"] == "code"}
    assert list(student_code) == [cell_id for cell_id in reference_code if cell_id in student_code]

    exercise_cells = {
        "lab5-010": {"build_model"},
        "lab5-016": {"train_step"},
        "lab5-020": {"evaluate", "train_epoch"},
    }
    # 参考版单独调整错例图布局，并开启默认关闭的选做对照。
    reference_variants = {"lab5-033", "lab5-036"}
    assert (exercise_cells.keys() | reference_variants) <= student_code.keys()
    for cell_id, s in student_code.items():
        assert s["outputs"] == [] and s["execution_count"] is None
        r = reference_code[cell_id]
        student_source = "".join(s["source"])
        reference_source = "".join(r["source"])
        if cell_id in exercise_cells:
            assert "TODO" in student_source and "raise NotImplementedError" not in reference_source
            for cell, source in [(s, student_source), (r, reference_source)]:
                assert "lab-definition" in cell["metadata"].get("tags", [])
                names = {node.name for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
                assert names == exercise_cells[cell_id]
        elif cell_id not in reference_variants:
            assert ast.dump(ast.parse(student_source)) == ast.dump(ast.parse(reference_source)), cell_id
    namespace: dict = dict(
        torch=torch,
        nn=nn,
        seed_everything=seed_everything,
        DEVICE=torch.device("cpu"),
        make_loader=make_loader,
        snapshot=snapshot,
        time=__import__("time"),
    )
    for c in reference["cells"]:
        if "lab-definition" in c["metadata"].get("tags", []):
            exec("".join(c["source"]), namespace)
    print("[通过] 两版语法、四个 TODO、配对代码及无学生输出")
    return namespace


def check_steps(ns):
    model = ns["build_model"]("linear", 5)
    independent = copy.deepcopy(model)
    opt = torch.optim.SGD(model.parameters(), lr=0.03)
    generator = torch.Generator().manual_seed(31)
    criterion = nn.CrossEntropyLoss()
    for _ in range(2):
        x = torch.randn(3, 1, 28, 28, generator=generator)
        y = torch.tensor([0, 2, 4])
        scores = independent(x)
        loss = criterion(scores, y)
        grads = torch.autograd.grad(loss, tuple(independent.parameters()))
        expected_correct = (scores.argmax(1) == y).sum().item()
        with torch.no_grad():
            for p, g in zip(independent.parameters(), grads):
                p.add_(g, alpha=-0.03)
        row = ns["train_step"](model, x, y, opt, criterion, torch.device("cpu"))
        assert row["n"] == 3 and row["correct"] == expected_correct
        assert math.isclose(row["loss_sum"], loss.item() * 3, rel_tol=1e-6)
        for a, b in zip(model.parameters(), independent.parameters()):
            torch.testing.assert_close(a, b)
    # 第二批的独立梯度对照能发现未清零产生的累加。
    print("[通过] 连续两批与独立 autograd.grad 更新一致，梯度不意外累加")


def check_evaluation(ns):
    class ObservedModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.bias = nn.Parameter(torch.zeros(2))

        def forward(self, x):
            assert not self.training, "评估应切换 eval 模式"
            assert not torch.is_grad_enabled(), "评估应关闭梯度记录"
            return x + self.bias

    model = ObservedModel()
    model.bias.grad = torch.tensor([3.0, 4.0])
    old = snapshot(model)
    x = torch.tensor([[5.0, 0.0], [0.0, 5.0], [5.0, 0.0], [0.0, 5.0], [0.0, 5.0]])
    y = torch.tensor([0, 1, 0, 1, 0])
    criterion = nn.CrossEntropyLoss()
    loader = make_loader(TensorDataset(x, y), 2)
    row = ns["evaluate"](model, loader, criterion, torch.device("cpu"))
    assert row["n"] == 5 and row["accuracy"] == 0.8
    assert abs(row["loss"] - criterion(x, y).item()) < 1e-6
    for k, v in model.state_dict().items():
        assert torch.equal(old[k], v)
    torch.testing.assert_close(model.bias.grad, torch.tensor([3.0, 4.0]))
    saved = snapshot(model)
    with torch.no_grad():
        model.bias.add_(1)
    assert torch.equal(saved["bias"], torch.zeros(2))
    empty = make_loader(TensorDataset(x[:0], y[:0]), 2)
    for function, args in [
        (ns["evaluate"], (model, empty, criterion, torch.device("cpu"))),
        (ns["train_epoch"], (model, empty, None, criterion, torch.device("cpu"))),
    ]:
        try:
            function(*args)
        except ValueError:
            pass
        else:
            raise AssertionError("空 loader 应被拒绝")
    print("[通过] 不等批量加权、eval/no_grad、不改参数或已有梯度、空数据防护、快照独立")


def check_epoch(ns):
    x = torch.randn(5, 1, 28, 28)
    y = torch.tensor([0, 1, 2, 3, 4])
    model = ns["build_model"]("linear", 6)
    manual = copy.deepcopy(model)
    criterion = nn.CrossEntropyLoss()
    opt = torch.optim.SGD(model.parameters(), lr=0.01)
    other_opt = torch.optim.SGD(manual.parameters(), lr=0.01)
    loader = make_loader(TensorDataset(x, y), 2)
    rows = [
        ns["train_step"](manual, a, b, other_opt, criterion, torch.device("cpu"))
        for a, b in loader
    ]
    row = ns["train_epoch"](model, loader, opt, criterion, torch.device("cpu"))
    assert row["n"] == 5 and row["accuracy"] == sum(r["correct"] for r in rows) / 5
    assert abs(row["loss"] - sum(r["loss_sum"] for r in rows) / 5) < 1e-6
    for a, b in zip(model.parameters(), manual.parameters()):
        torch.testing.assert_close(a, b)
    print("[通过] train_epoch 覆盖末尾小批次，参数与逐批更新一致")


def check_selection(ns):
    # 独立的受控验证序列，检验恢复的是中间最佳轮次，而非最后一轮。
    local = ns.copy()
    tiny = TensorDataset(torch.zeros(3, 1, 28, 28), torch.zeros(3, dtype=torch.long))
    local.update(
        train_set=tiny,
        train_eval_loader="train",
        val_loader="validation",
        criterion=None,
        BATCH_SIZE=2,
        SPLIT_SEED=42,
        ORDER_SEED=2026,
    )
    tick = [0]

    def train_epoch(model, *args):
        tick[0] += 1
        with torch.no_grad():
            for p in model.parameters():
                p.fill_(tick[0])
        return dict(loss=1.0, accuracy=0.5, n=3)

    def evaluate(model, loader, *args):
        accuracy = [0.1, 0.9, 0.2][tick[0]] if loader == "validation" else 0.5
        return dict(loss=1.0, accuracy=accuracy, n=3)

    local.update(train_epoch=train_epoch, evaluate=evaluate)
    # fit 的 globals 必须指向注入后的受控空间。
    import types

    fit = types.FunctionType(ns["fit"].__code__, local, argdefs=ns["fit"].__defaults__)
    run = fit("linear", epochs=2)
    assert run["best_epoch"] == 1 and len(run["history"]) == 3
    assert run["split_seed"] == 42 and run["order_seed"] == 2026
    assert all(torch.equal(p, torch.ones_like(p)) for p in run["model"].parameters())
    print("[通过] 保存独立状态并恢复中间最佳轮次，epoch=0 与历史对齐")


def main():
    torch.set_num_threads(2)
    ns = notebook_definitions()
    assert sum(p.numel() for p in ns["build_model"]("mlp").parameters()) == 101770
    assert sum(p.numel() for p in ns["build_model"]("linear").parameters()) == 7850
    check_steps(ns)
    check_evaluation(ns)
    check_epoch(ns)
    check_selection(ns)
    print("核心验证全部通过；完整数据结果见已运行的参考 notebook。")


if __name__ == "__main__":
    main()
