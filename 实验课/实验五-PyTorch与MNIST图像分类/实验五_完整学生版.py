# %% [markdown]
# # 实验五：使用 PyTorch 训练 MNIST 手写数字分类模型 · 学生版
#
# 承接实验四：我们已经手写过前向传播、BP 和梯度下降。现在使用框架自动求导，将注意力放到**数据、训练流程、验证与实验结论**。
#
# 主线：认识数据 → 建立分类模型 → 检查一次更新 → 小批量训练 → 验证选模 → 最终测试与错误分析。
#
# **Q1–Q7 必做；Q8 为选做学习率对照。** Q1 是教师演示。先写预测，再补全代码，最后用输出解释结果。允许使用 AI，需能解释采用的方案并独立核验。
#
# 带 TODO 的函数会抛出 `NotImplementedError`，表示需要作答。不要删除检查来绕过问题。完成后重启 kernel，从头运行，保留输出并提交。

# %% [markdown]
# ## 0 · 文件、环境和实验约定
#
# 将本 notebook 与 `lab_utils.py` 放在同一目录，在该目录启动。沿用 `ai-lab` 环境，选择正确的 Python kernel。需要：
#
# ```bash
# python -m pip install torch torchvision numpy matplotlib
# ```
#
# CPU 足够完成必做实验；CUDA 安装可参考 [PyTorch 安装说明](../实验四-神经网络与反向传播/PyTorch安装说明.md) 和 [官方安装选择器](https://pytorch.org/get-started/locally/)。不需要安装 d2l 包。
#
# 第一次联网运行时，将下一单元的 `DOWNLOAD` 改为 `True`；数据缓存就绪后改回 `False`。离线时复制教师已准备的 `data/MNIST/raw/` 目录；这里的下载开关与 pip 软件源无关。
#
# | 设置 | 本版约定 |
# | --- | --- |
# | 数据划分 | 官方训练集固定拆成 50,000 训练、10,000 验证；官方测试集 10,000 最终使用 |
# | 输入 | float32，单通道 28×28，像素范围 [0,1]；无额外标准化或增强 |
# | 标签 | long，类别下标 0–9，形状 (B,) |
# | 主模型 | Flatten → Linear(784,128) → ReLU → Linear(128,10) |
# | 输出与损失 | logits，不加 softmax；CrossEntropyLoss 默认 mean |
# | 优化 | SGD，无动量；lr=0.1，batch_size=128，epochs=5 |
# | 随机设置 | 划分、初始化、批次顺序分别固定；CPU 为默认验证环境 |
# | 选模规则 | 验证准确率优先，相同准确率时验证损失较小优先，完全相同取较早轮次 |
#
# 先预测：框架计算了梯度，是否意味着数据、损失和训练流程就一定正确？

# %%
from pathlib import Path
import os, sys, math, time, platform
import numpy as np
import torch
from torch import nn
from torch.utils.data import TensorDataset, Subset
import torchvision
import matplotlib.pyplot as plt

assert Path("lab_utils.py").exists(), "请在实验五目录启动 notebook"
from lab_utils import (
    seed_everything,
    data_root,
    load_train_val,
    load_test,
    make_loader,
    snapshot,
    plot_samples,
    plot_history,
    collect_errors,
)

SEED, SPLIT_SEED, ORDER_SEED = 42, 42, 2026
BATCH_SIZE, EPOCHS, LR = 128, 5, 0.1
DOWNLOAD = False  # 第一次联网下载时改为 True，缓存就绪后改回 False。
DEVICE = torch.device("cpu")  # 已有 CUDA 环境可自行改为 torch.device("cuda")。
# 小模型使用过多 CPU 线程可能更慢；限制线程数便于普通电脑复现。
torch.set_num_threads(min(4, os.cpu_count() or 1))
seed_everything(SEED)
print("Python:", sys.version.split()[0], sys.executable)
print("PyTorch:", torch.__version__, "torchvision:", torchvision.__version__)
print(
    "device:",
    DEVICE,
    "threads:",
    torch.get_num_threads(),
    "machine:",
    platform.machine(),
)
print("data root:", data_root().resolve())
print(
    dict(
        seed=SEED,
        split_seed=SPLIT_SEED,
        order_seed=ORDER_SEED,
        batch_size=BATCH_SIZE,
        lr=LR,
        epochs=EPOCHS,
    )
)

# %% [markdown]
# ## Q1 · 从手写 BP 到自动求导（必做，教师演示）
#
# 实验四：$Z_1=XW_1+b_1$，$A_1=\tanh(Z_1)$，$\hat y=A_1W_2+b_2$，$L=\mathrm{mean}((\hat y-y)^2)$。
#
# 下面在三个样本的小网络上比较手写梯度和自动求导。直接复制同一份参数，统一 float64、激活及损失，而不是假设 NumPy 与 PyTorch 的随机种子能产生相同数值。
#
# 运行前预测：`loss.backward()` 会不会改变参数？连续两次反向传播且不清零，梯度会怎样？

# %%
rng = np.random.default_rng(7)
x_np = np.array([[-0.7], [0.2], [0.9]], dtype=np.float64)
y_np = np.sin(x_np)
p_np = {
    "W1": rng.normal(size=(1, 3)),
    "b1": np.zeros((1, 3)),
    "W2": rng.normal(scale=1 / np.sqrt(3), size=(3, 1)),
    "b2": np.zeros((1, 1)),
}
a = np.tanh(x_np @ p_np["W1"] + p_np["b1"])
yhat = a @ p_np["W2"] + p_np["b2"]
dy = 2 * (yhat - y_np) / len(x_np)
dz = (dy @ p_np["W2"].T) * (1 - a**2)
manual = {
    "W1": x_np.T @ dz,
    "b1": dz.sum(0, keepdims=True),
    "W2": a.T @ dy,
    "b2": dy.sum(0, keepdims=True),
}
params = {
    k: nn.Parameter(torch.tensor(v, dtype=torch.float64)) for k, v in p_np.items()
}
x_demo, y_demo = torch.tensor(x_np), torch.tensor(y_np)


def demo_loss():
    pred = (
        torch.tanh(x_demo @ params["W1"] + params["b1"]) @ params["W2"] + params["b2"]
    )
    return ((pred - y_demo) ** 2).mean()


before = {k: p.detach().clone() for k, p in params.items()}
optimizer_demo = torch.optim.SGD(params.values(), lr=0.001)
optimizer_demo.zero_grad(set_to_none=True)
demo_loss().backward()
first_grad = {k: p.grad.clone() for k, p in params.items()}
for k, p in params.items():
    np.testing.assert_allclose(p.grad.numpy(), manual[k], rtol=1e-10, atol=1e-12)
    assert torch.equal(p, before[k])
    print(k, "max gradient error =", np.max(np.abs(p.grad.numpy() - manual[k])))
# 重新前向创建计算图，但不清零；验证梯度累加。
demo_loss().backward()
for k, p in params.items():
    torch.testing.assert_close(p.grad, 2 * first_grad[k])
print("backward did not update parameters; repeated backward accumulated gradients")
optimizer_demo.zero_grad(set_to_none=True)
demo_loss().backward()
optimizer_demo.step()
assert any(not torch.equal(p, before[k]) for k, p in params.items())
print("optimizer.step changed parameters")

# %% [markdown]
# **我的解释与证据：** 待填写。

# %% [markdown]
# ## Q2 · 数据、标签与小批量（必做）
#
# 教师提供固定划分和加载工具。此时只使用官方训练数据；测试集到 Q7 才加载和评估。
#
# 查看样本并补全形状表，说明每个维度的含义。为什么标签是整数类别下标，而不是待拟合的连续数值？
#
# | 量 | 形状 | dtype | 含义 |
# | --- | --- | --- | --- |
# | 一个 batch 的图像 | 待填 | 待填 | 待填 |
# | 一个 batch 的标签 | 待填 | 待填 | 待填 |
# | 展平后图像 | 待填 | 待填 | 待填 |
# | 模型输出 | 待填 | 待填 | 待填 |
#
# 若验证和测试图像都不参与反向传播，为什么还要将它们区分开？

# %%
train_set, val_set = load_train_val(download=DOWNLOAD, split_seed=SPLIT_SEED)
assert len(train_set) == 50000 and len(val_set) == 10000
assert not set(train_set.indices).intersection(val_set.indices)
assert len(set(train_set.indices) | set(val_set.indices)) == 60000
val_loader = make_loader(val_set, BATCH_SIZE)
train_eval_loader = make_loader(train_set, BATCH_SIZE)
# 探索样本使用独立 loader，不消耗正式训练的随机 generator。
x_batch, y_batch = next(
    iter(make_loader(train_set, BATCH_SIZE, shuffle=True, seed=ORDER_SEED))
)
assert x_batch.shape == (BATCH_SIZE, 1, 28, 28)
assert x_batch.dtype == torch.float32 and y_batch.dtype == torch.long
assert y_batch.shape == (BATCH_SIZE,) and 0 <= y_batch.min() <= y_batch.max() <= 9
assert 0 <= x_batch.min() <= x_batch.max() <= 1
print("images:", x_batch.shape, x_batch.dtype, "labels:", y_batch.shape, y_batch.dtype)
print(
    "batches/epoch:",
    math.ceil(len(train_set) / BATCH_SIZE),
    "last batch:",
    len(train_set) % BATCH_SIZE,
)
_ = plot_samples(x_batch[:12], y_batch[:12], title="Training samples")

# %% [markdown]
# **我的解释与证据：** 待填写。

# %% [markdown]
# ## Q3 · 建立分类模型（必做）
#
# 实现 `build_model(kind="mlp", seed=42)`，返回位于 DEVICE 的网络。保留给定的种子设置。
#
# - `mlp`：Flatten → Linear(784,128) → ReLU → Linear(128,10)。
# - `linear`：Flatten → Linear(784,10)，供 Q6 对照。
# - 未知 kind 应抛出 ValueError。
#
# 线性层计算为 $XW^T+b$，PyTorch 的 weight 保存为 `(out_features,in_features)`，与实验四手写 W 的排列不同。
#
# 输出十个 logits（未经归一化的分数），不添加 softmax。解释：为什么不能直接输出一个浮点数，四舍五入后当作数字标签？去掉 ReLU 后两次仿射变换能否表示更一般的函数？


# %%
def build_model(kind="mlp", seed=42):
    seed_everything(seed)
    # TODO：按题意构造两种网络，未知 kind 抛出 ValueError。
    # 最后将模型移动到 DEVICE；输出层直接返回十个 logits。
    raise NotImplementedError("Q3：完成模型定义")


# %%
model = build_model("mlp", SEED)
scores = model(x_batch.to(DEVICE))
assert scores.shape == (len(y_batch), 10)
assert torch.isfinite(scores).all()
assert sum(p.numel() for p in model.parameters()) == 101770
assert sum(p.numel() for p in build_model("linear", SEED).parameters()) == 7850
print(model)
print("MLP parameters:", sum(p.numel() for p in model.parameters()))
print("logits of first sample:", scores[0].detach().cpu())

# %% [markdown]
# **我的解释与证据：** 待填写。

# %% [markdown]
# ## Q4 · 分类损失与一次参数更新（必做）
#
# 设第 i 个样本的类别分数为 $z_{i,k}$，真实类别为 $y_i$，交叉熵为
#
# $$
# L=-\frac1B\sum_i\log\frac{\exp(z_{i,y_i})}{\sum_{k=0}^9\exp(z_{i,k})}.
# $$
#
# 实现中直接使用 `nn.CrossEntropyLoss()`，它接收 logits 和整数类别标签，并稳定地组合所需运算。不要先对输出调用 softmax 再交给该损失。
#
# 先运行下面的小例子：两个结果的预测类别相同，准确率相同，损失是否相同？为什么不能用 argmax 后的准确率直接反向传播？

# %%
criterion = nn.CrossEntropyLoss()
target_demo = torch.tensor([3])
for margin in [0.2, 5.0]:
    z = torch.zeros(1, 10)
    z[0, 3] = margin
    print(
        f"margin={margin}, predicted={z.argmax(1).item()}, "
        f"p(true)={z.softmax(1)[0, 3].item():.4f}, CE={criterion(z, target_demo).item():.4f}"
    )

# %% [markdown]
# 补全 `train_step`，返回本批**更新前**的损失总和、预测正确数和样本数。
#
# 固定接口为 `{"loss_sum": float, "correct": int, "n": int}`。损失函数采用默认 mean，因此乘以本批真实样本数得到 loss_sum。不要把带计算图的 loss 保存在训练历史中。
#
# 顺序：清除旧梯度 → 前向 → 损失 → backward → step。图像与标签均需移动到同一 device；出现非有限损失或梯度时停止并排查。


# %%
def train_step(model, x, y, optimizer, criterion, device):
    model.train()
    # TODO：移动数据、清梯度、前向、求损失、反向、更新。
    # 检查损失与梯度有限；返回更新前的 loss_sum、correct 和 n。
    raise NotImplementedError("Q4：完成单批训练")


# %%
# 用新模型做一次检查，之后正式训练再重新初始化。
step_model = build_model("mlp", SEED)
step_optimizer = torch.optim.SGD(step_model.parameters(), lr=LR)
params_before = snapshot(step_model)
step_metrics = train_step(
    step_model, x_batch, y_batch, step_optimizer, criterion, DEVICE
)
assert step_metrics["n"] == len(y_batch)
assert math.isfinite(step_metrics["loss_sum"])
assert 0 <= step_metrics["correct"] <= step_metrics["n"]
assert any(
    not torch.equal(params_before[k], v.cpu())
    for k, v in step_model.state_dict().items()
)
assert all(torch.isfinite(p).all() for p in step_model.parameters())
print("single batch before-update metrics:", step_metrics)
# 训练完后的同一批损失仅作观察；任意学习率下一步都下降不是必须满足的数学保证。
with torch.no_grad():
    print(
        "same-batch loss after update:",
        criterion(step_model(x_batch.to(DEVICE)), y_batch.to(DEVICE)).item(),
    )

# %% [markdown]
# **我的解释与证据：** 待填写。

# %% [markdown]
# ## Q5 · 从一批到完整训练（必做）
#
# 实现 `evaluate` 和 `train_epoch`，接口均返回 `{"loss": float, "accuracy": float, "n": int}`。
#
# - `evaluate`：eval 模式和 no_grad 上下文；不更新模型、不清理或改写已有梯度。
# - `train_epoch`：遍历 loader，调用 Q4 的 train_step 并汇总；每个 batch 更新一次。
# - 两者均按真实样本数加权：`sum(batch_mean_loss * batch_n) / total_n`，准确率为总正确数／总样本数。空 loader 应抛出 ValueError。
#
# `model.eval()` 设置 Dropout/BatchNorm 等模块的行为；`no_grad()` 关闭梯度记录。当前网络没有这些模块，仍应区分二者。
#
# 先用不等大小 batch 的已知分数检查汇总，再做小样本拟合，最后进入完整训练。


# %%
def evaluate(model, loader, criterion, device):
    # TODO：eval + no_grad；按实际样本数汇总，不修改参数或梯度。
    # 返回 loss、accuracy、n，空 loader 抛出 ValueError。
    raise NotImplementedError("Q5：完成评估")


def train_epoch(model, loader, optimizer, criterion, device):
    # TODO：遍历 batch，调用 train_step 并按样本数汇总。
    # 返回 loss、accuracy、n，空 loader 抛出 ValueError。
    raise NotImplementedError("Q5：完成一轮训练")


# %%
known_scores = torch.tensor(
    [[5.0, 0.0], [0.0, 5.0], [5.0, 0.0], [0.0, 5.0], [0.0, 5.0]]
)
known_y = torch.tensor([0, 1, 0, 1, 0])  # 最后一张故意错分且损失很大。
known_loader = make_loader(TensorDataset(known_scores, known_y), batch_size=2)
identity = nn.Identity().to(DEVICE)
metrics = evaluate(identity, known_loader, criterion, DEVICE)
expected_loss = criterion(known_scores, known_y).item()
assert abs(metrics["loss"] - expected_loss) < 1e-6
assert metrics["accuracy"] == 4 / 5 and metrics["n"] == 5
batch_means = [criterion(a, b).item() for a, b in known_loader]
wrong_mean = sum(batch_means) / len(batch_means)
assert abs(wrong_mean - expected_loss) > 0.1
print(
    "correct sample-weighted loss:",
    metrics["loss"],
    "wrong batch-weighted loss:",
    wrong_mean,
)

# %%
# 固定 32 张训练图像反复拟合，用新模型；不使用验证或测试数据。
tiny_x, tiny_y = next(iter(make_loader(Subset(train_set, range(32)), batch_size=32)))
tiny_model = build_model("mlp", SEED)
tiny_optimizer = torch.optim.SGD(tiny_model.parameters(), lr=LR)
tiny_loader = make_loader(TensorDataset(tiny_x, tiny_y), batch_size=32)
tiny_before = evaluate(tiny_model, tiny_loader, criterion, DEVICE)
for _ in range(100):
    train_step(tiny_model, tiny_x, tiny_y, tiny_optimizer, criterion, DEVICE)
tiny_after = evaluate(tiny_model, tiny_loader, criterion, DEVICE)
print("tiny set before:", tiny_before, "after:", tiny_after)
assert tiny_after["loss"] < tiny_before["loss"] * 0.2, (
    "先排查小样本训练，暂不进入完整训练"
)

# %% [markdown]
# **我的解释与证据：** 待填写。

# %% [markdown]
# ### 完整训练与验证选轮次
#
# 下面的 `fit` 是教师提供的组织代码，调用你实现的 train_epoch 和 evaluate。逐轮阅读并解释它如何保存最佳参数。
#
# 每个 epoch 结束后，用**同一组固定参数**重新评估训练集和验证集并绘图；在线训练指标另外记录为 online_loss/online_accuracy，二者不可混用。epoch=0 表示训练前。
#
# 训练 loader 使用独立随机 generator，每次完整实验重建。snapshot 复制参数，避免后续更新覆盖“最佳模型”。训练结束加载验证集选出的参数；返回模型可能来自最后一轮以前。


# %%
def fit(kind, lr=0.1, epochs=5, seed=42):
    model = build_model(kind, seed)
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    loader = make_loader(train_set, BATCH_SIZE, shuffle=True, seed=ORDER_SEED)
    history = []
    best_key, best_state, best_epoch = None, None, None
    started = time.perf_counter()
    for epoch in range(epochs + 1):
        online = None
        if epoch > 0:
            online = train_epoch(model, loader, optimizer, criterion, DEVICE)
        train_metrics = evaluate(model, train_eval_loader, criterion, DEVICE)
        val_metrics = evaluate(model, val_loader, criterion, DEVICE)
        row = dict(
            epoch=epoch,
            train_loss=train_metrics["loss"],
            train_accuracy=train_metrics["accuracy"],
            val_loss=val_metrics["loss"],
            val_accuracy=val_metrics["accuracy"],
            seconds=time.perf_counter() - started,
            online_loss=None if online is None else online["loss"],
            online_accuracy=None if online is None else online["accuracy"],
        )
        history.append(row)
        key = (val_metrics["accuracy"], -val_metrics["loss"])
        if best_key is None or key > best_key:
            best_key, best_state, best_epoch = key, snapshot(model), epoch
        print(
            f"{kind} epoch={epoch}: train loss={row['train_loss']:.4f}, "
            f"acc={row['train_accuracy']:.2%}; val loss={row['val_loss']:.4f}, "
            f"acc={row['val_accuracy']:.2%}; elapsed={row['seconds']:.1f}s",
            flush=True,
        )
    model.load_state_dict(best_state)
    model.eval()
    return dict(
        model=model,
        history=history,
        best_epoch=best_epoch,
        best_val_accuracy=best_key[0],
        best_val_loss=-best_key[1],
        parameters=sum(p.numel() for p in model.parameters()),
        seconds=time.perf_counter() - started,
        kind=kind,
        lr=lr,
        epochs=epochs,
        seed=seed,
        order_seed=ORDER_SEED,
        split_seed=SPLIT_SEED,
    )


# %%
mlp_run = fit("mlp", lr=LR, epochs=EPOCHS, seed=SEED)
assert len(mlp_run["history"]) == EPOCHS + 1
restored_val = evaluate(mlp_run["model"], val_loader, criterion, DEVICE)
assert abs(restored_val["accuracy"] - mlp_run["best_val_accuracy"]) < 1e-12
assert abs(restored_val["loss"] - mlp_run["best_val_loss"]) < 1e-6
_ = plot_history({"mlp": mlp_run["history"]})
print("selected MLP epoch:", mlp_run["best_epoch"])

# %% [markdown]
# **我的解释与证据：** 待填写。

# %% [markdown]
# ## Q6 · 线性基线与模型选择（必做）
#
# **运行前写预测：** 与单隐藏层网络相比，线性分类器的训练速度、参数量、训练和验证准确率会怎样？说明依据。
#
# 对照固定划分、预处理、学习率、优化器、batch size、epoch 数和批次顺序。每次独立初始化；不同形状的模型不具有“相同初始参数”。主模型与基线同时改变了非线性与参数量，只能讨论这两种结构在当前训练预算下的表现。
#
# 补全结果表，并根据验证集选择最终模型。测试集在这里仍不使用。

# %% [markdown]
# **我的解释与证据：** 待填写。

# %%
linear_run = fit("linear", lr=LR, epochs=EPOCHS, seed=SEED)
runs = {"linear": linear_run, "mlp": mlp_run}
_ = plot_history({name: run["history"] for name, run in runs.items()})
print("model | parameters | selected epoch | train acc | val acc | val loss | seconds")
for name, run in runs.items():
    best_row = run["history"][run["best_epoch"]]
    print(
        f"{name} | {run['parameters']} | {run['best_epoch']} | "
        f"{best_row['train_accuracy']:.4f} | {run['best_val_accuracy']:.4f} | "
        f"{run['best_val_loss']:.4f} | {run['seconds']:.1f}"
    )
# 同一规则在模型之间选择；完全相同按字典顺序优先选更简单的 linear。
selected_name = max(
    runs,
    key=lambda name: (runs[name]["best_val_accuracy"], -runs[name]["best_val_loss"]),
)
selected_run = runs[selected_name]
final_model = selected_run["model"]
print("FROZEN choice:", selected_name, "epoch:", selected_run["best_epoch"])

# %% [markdown]
# **我的对照记录：**
#
# | 模型 | 参数量 | 选定轮次 | 训练准确率 | 验证准确率 | 验证损失 | 耗时 |
# | --- | --- | --- | --- | --- | --- | --- |
# | linear | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
# | mlp | 待填 | 待填 | 待填 | 待填 | 待填 | 待填 |
#
# 根据实际证据解释是否值得增加隐藏层，并写明结论的适用范围：待填写。

# %% [markdown]
# ## Q7 · 最终测试与错分样本（必做）
#
# 先确认 Q6 已冻结模型、轮次和训练方案，再打开这一部分。只对最终所选模型报告测试指标，不根据测试集再修改学习率或结构。
#
# 教师提供按顺序收集前 12 张错分图像的工具。观察哪些数字容易混淆，区分“看到了什么”和“猜测为什么”。p(pred) 是模型给预测类别的 softmax 概率，不是经过校准的正确率保证。
#
# 思考：高准确率能否证明模型对任意手写数字都可靠？查看错分图像后提出改进，应该在什么数据上验证？

# %%
test_set = load_test(download=DOWNLOAD)
assert len(test_set) == 10000
test_loader = make_loader(test_set, BATCH_SIZE)
final_train = evaluate(final_model, train_eval_loader, criterion, DEVICE)
final_val = evaluate(final_model, val_loader, criterion, DEVICE)
final_test = evaluate(final_model, test_loader, criterion, DEVICE)
print("final model:", selected_name, "epoch:", selected_run["best_epoch"])
print("train:", final_train, "validation:", final_val, "TEST:", final_test)
print("test errors:", round((1 - final_test["accuracy"]) * final_test["n"]))
errors = collect_errors(final_model, test_loader, DEVICE, limit=12)
if errors is not None:
    error_images, error_labels, error_preds, error_probs = errors
    _ = plot_samples(
        error_images,
        error_labels,
        error_preds,
        error_probs,
        title="First test errors (fixed order)",
    )
else:
    print("本次测试没有错分样本；不构造虚假的错误图。")

# %% [markdown]
# **我的解释与证据：** 待填写。

# %% [markdown]
# ## Q8 · 学习率对照（选做）
#
# 固定 MLP、同样初始化、数据划分、批次顺序和 epoch 数，仅改变学习率为 0.01、0.1、0.5。已有 0.1 的结果可直接复用；其他值从新模型开始，不能在 final_model 上继续训练。
#
# 把 RUN_EXTENSION 改为 True 前，写下对速度与稳定性的预测。只使用训练／验证指标，不再次查看测试集。若出现非有限值，记录学习率、出错轮次和现象并停止该次实验，不将中断曲线称为收敛。

# %%
RUN_EXTENSION = False
if RUN_EXTENSION:
    lr_runs = {"lr=0.1": mlp_run}
    for trial_lr in [0.01, 0.5]:
        try:
            lr_runs[f"lr={trial_lr}"] = fit(
                "mlp", lr=trial_lr, epochs=EPOCHS, seed=SEED
            )
        except FloatingPointError as error:
            print("stopped learning rate:", trial_lr, "reason:", error)
    _ = plot_history({name: run["history"] for name, run in lr_runs.items()})
else:
    print("Q8 skipped (optional); Q1–Q7 are unaffected")

# %% [markdown]
# **我的解释与证据：** 待填写。

# %% [markdown]
# ## 排错记录与提交（必做）
#
# 记录一次实际问题；若未遇到自然错误，可以使用 Q5 的“等权平均每批损失”作为**主动错误注入**练习，不能编造经历。
#
# ```text
# 现象：具体输出或报错是什么？
# 假设：怀疑哪个环节，依据是什么？
# 最小检查：怎样缩小数据或设计已知答案？
# 修改：改变了哪一步？
# 验证：修复后哪些独立检查通过？
# AI 辅助（如使用）：采用了什么建议，如何核验？
# ```
#
# 提交前检查：
#
# - Q1–Q7 解释与输出完整；Q8 完成或注明跳过。
# - 已完成模型、单步训练、训练一轮与评估的核心 TODO。
# - 数据划分检查、单次更新、小样本拟合、不等批量指标检查均通过。
# - 训练和验证曲线、基线对照、选模依据、最终测试及错分分析齐全。
# - notebook 与 lab_utils.py 同级；数据路径采用相对目录，不提交本机绝对路径。
# - 重启 kernel 从头运行并保留输出。提交学生 notebook 与 lab_utils.py；数据缓存按教师要求单独提供。

# %% [markdown]
# **我的解释与证据：** 待填写。

# %% [markdown]
# ## 参考资料
#
# 本实验的题目、检查和文字解释为课程自拟，参考下列公开材料的教学结构：
#
# - [d2l 3.5：图像分类数据集](https://zh.d2l.ai/chapter_linear-networks/image-classification-dataset.html)（原书使用 Fashion-MNIST，本实验采用 MNIST）。
# - [d2l 3.7：softmax 回归的简洁实现](https://zh.d2l.ai/chapter_linear-networks/softmax-regression-concise.html)。
# - [d2l 4.3：多层感知机的简洁实现](https://zh.d2l.ai/chapter_multilayer-perceptrons/mlp-concise.html)。
# - [PyTorch：优化模型参数](https://docs.pytorch.org/tutorials/beginner/basics/optimization_tutorial.html)。
# - [torchvision：MNIST](https://docs.pytorch.org/vision/stable/generated/torchvision.datasets.MNIST.html)。
#
# 本实验另设验证集，训练与评估代码显式展开，不依赖 d2l 的训练封装。
