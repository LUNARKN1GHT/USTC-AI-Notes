"""教师检查：模型结构、评估汇总、参数快照、验证选轮次与数据隔离。

运行：python 验证参考答案.py [--check-data]
不启动完整训练，不修改 notebook 或实验结果。
"""

import argparse
import ast
import contextlib
import io
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset

import lab_utils as utils


ROOT = Path(__file__).resolve().parent


def reference_namespace():
    notebook = json.loads((ROOT / "实验六_参考答案.ipynb").read_text())
    namespace = {"torch": torch, "nn": nn, "seed_everything": utils.seed_everything,
                 "DEVICE": torch.device("cpu")}
    found = False
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        tree = ast.parse("".join(cell["source"]))
        for node in tree.body:
            if isinstance(node, ast.FunctionDef) and node.name == "build_model":
                exec(compile(ast.Module(body=[node], type_ignores=[]), "reference:build_model", "exec"), namespace)
                found = True
    assert found, "参考答案中缺少 build_model"
    return namespace


def check_models():
    build = reference_namespace()["build_model"]
    for name, expected in (("mlp", 394634), ("cnn", 136874)):
        model = build(name)
        assert model(torch.zeros(3, 3, 32, 32)).shape == (3, 10)
        assert sum(p.numel() for p in model.parameters()) == expected
        a, b = build(name), build(name)
        assert all(torch.equal(a.state_dict()[k], v) for k, v in b.state_dict().items())
    try:
        build("unknown")
    except ValueError:
        pass
    else:
        raise AssertionError("未知模型必须报错")
    # 一个输出通道应合并全部 RGB 输入通道。
    conv = nn.Conv2d(3, 1, kernel_size=1, bias=False)
    with torch.no_grad():
        conv.weight.fill_(1)
    rgb = torch.tensor([1., 2., 3.]).reshape(1, 3, 1, 1)
    assert conv(rgb).item() == 6


def check_evaluation():
    scores = torch.zeros(5, 10)
    scores[range(5), [0, 1, 2, 3, 4]] = 3
    targets = torch.tensor([0, 1, 1, 3, 0])
    dataset = TensorDataset(scores, targets)
    model = nn.Linear(10, 10, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.eye(10))
    model.weight.grad = torch.ones_like(model.weight)
    before, grad_before = utils.snapshot(model), model.weight.grad.clone()
    expected_loss = nn.functional.cross_entropy(scores, targets).item()
    grad_flags = []
    hook = model.register_forward_pre_hook(lambda module, inputs: grad_flags.append(torch.is_grad_enabled()))
    for batch_size in (2, 3, 5):
        report = utils.evaluate(model, utils.make_loader(dataset, batch_size), "cpu", details=True)
        assert abs(report["loss"] - expected_loss) < 1e-6
        assert report["accuracy"] == 3 / 5
        assert report["n"] == 5 and report["confusion"].sum().item() == 5
        assert len(report["errors"]) == 2
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert torch.equal(grad_before, model.weight.grad)
    assert not model.training
    hook.remove()
    assert grad_flags and not any(grad_flags)
    frozen = utils.snapshot(model)
    assert frozen["weight"].data_ptr() != model.weight.data_ptr()
    with torch.no_grad():
        model.weight.add_(1)
    assert torch.equal(frozen["weight"], before["weight"])
    empty = TensorDataset(torch.empty(0, 10), torch.empty(0, dtype=torch.long))
    try:
        utils.evaluate(model, utils.make_loader(empty), "cpu")
    except ValueError:
        pass
    else:
        raise AssertionError("空评估集必须报错")


def check_best_epoch():
    # 人造验证序列：第 1 轮最好，后续参数继续改变；返回时必须恢复第 1 轮。
    dataset = TensorDataset(torch.zeros(3, 1), torch.zeros(3, dtype=torch.long))
    model = nn.Linear(1, 1)
    with torch.no_grad():
        model.weight.zero_()
        model.bias.zero_()
    epoch = [0]

    def fake_train(model, *args):
        epoch[0] += 1
        with torch.no_grad():
            model.bias.fill_(epoch[0])
        return {"loss": 1., "accuracy": 0., "n": 3}

    reports = []
    for accuracy in (0.1, 0.8, 0.4, 0.3):
        reports.extend([{"loss": 1., "accuracy": 0.5, "n": 3},
                        {"loss": 1., "accuracy": accuracy, "n": 3}])
    with patch.object(utils, "train_epoch", side_effect=fake_train), \
         patch.object(utils, "evaluate", side_effect=reports), \
         contextlib.redirect_stdout(io.StringIO()):
        result = utils.fit_model(model, dataset, dataset, dataset, epochs=3)
    assert result["best_epoch"] == 1 and result["model"].bias.item() == 1
    assert len(result["history"]) == 4


def check_data():
    data = utils.load_train_val(root=ROOT / "data", train_limit=12000)
    train, train_eval, val = (data[k] for k in ("train", "train_eval", "val"))
    assert len(train) == 12000 and len(val) == 5000
    assert len(set(data["train_pool_indices"])) == 45000
    assert set(data["train_pool_indices"]).isdisjoint(val.indices)
    assert set(data["train_pool_indices"]) | set(val.indices) == set(range(50000))
    aug = utils.with_augmentation(train)
    assert train.indices == train_eval.indices == aug.indices
    assert train.base is aug.base is val.base
    assert train.transform is not aug.transform
    val_image = val[0][0].clone()
    for _ in range(5):
        image, label = aug[0]
        assert image.shape == (3, 32, 32) and -1 <= image.min() <= image.max() <= 1
        assert label == train[0][1]
    assert torch.equal(val_image, val[0][0])
    assert np.allclose(utils.display_image(train[0][0]),
                       np.asarray(train.base[train.indices[0]][0], dtype=np.float32) / 255, atol=1e-6)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-data", action="store_true", help="另检查本地 CIFAR-10 划分与增强隔离")
    args = parser.parse_args()
    torch.set_num_threads(2)
    for check in (check_models, check_evaluation, check_best_epoch):
        check()
        print(check.__name__, "OK")
    if args.check_data:
        check_data()
        print("check_data OK")
