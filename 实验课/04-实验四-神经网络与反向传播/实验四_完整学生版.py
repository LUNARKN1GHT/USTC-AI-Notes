"""实验四：神经网络与反向传播——单文件学生版。

只需本文件；判题依赖 NumPy，绘图另需 Matplotlib，PyTorch 演示可选。
用法：
    python 实验四_完整学生版.py              # 分项公开自测
    python 实验四_完整学生版.py --run        # 自测通过后运行 Q7/Q8 并绘图
    python 实验四_完整学生版.py --run --with-lr  # 加做 Q9
    python 实验四_完整学生版.py --torch      # 加做 Q10

学生填写 6 个 TODO：损失、前向、反向、更新、训练中的计算、训练中的更新。
填写时删除对应的 raise NotImplementedError；这些占位语句让未完成项明确可见。
函数版练习的核心计算已归入 TwoLayerNet，不需要重复实现另一套公式。
教师提供初始化、数据、绘图、有限差分、训练异常保护及判题代码。

统一约定：样本按行，float64，tanh 隐藏层 / 线性输出；
MSE 不含 1/2；全批量梯度下降；H=16、seed=42、lr=0.02、10000 次更新。
Q1–Q8 必做，Q9 进阶，Q10 可跳过。公开自测提供诊断，不代替文字分析评分。
允许使用 AI，但请解释公式、独立核验结果并记录一次真实排错。
"""

import argparse
import numpy as np

# ==================== 学生填写区：Q2–Q6 ====================

def mse_loss(y_pred, y):
    """Q3/Q7：返回 mean((y_pred-y)**2)，形状均为 (N,1)，不含 1/2。"""
    if y_pred.shape != y.shape or y.ndim != 2 or y.shape[1] != 1:
        raise ValueError("y_pred 和 y 必须具有相同的 (N, 1) 形状")
    if len(y) == 0:
        raise ValueError("至少需要一个样本")
    # TODO 1 / Q3：在此填写 MSE 计算。
    raise NotImplementedError("请完成 Q3/Q7：均方误差")


class TwoLayerNet:
    """固定的单隐藏层网络，初始化由教师提供。"""

    def __init__(self, hidden_dim=16, seed=42, activation="tanh"):
        if not isinstance(hidden_dim, (int, np.integer)) or hidden_dim <= 0:
            raise ValueError("hidden_dim 必须为正整数")
        if activation not in ("tanh", "linear"):
            raise ValueError("activation 只能为 'tanh' 或 'linear'")
        self.activation = activation
        rng = np.random.default_rng(seed)
        self.params = {
            "W1": rng.normal(0.0, 1.0, size=(1, hidden_dim)),
            "b1": np.zeros((1, hidden_dim), dtype=np.float64),
            "W2": rng.normal(0.0, 1.0 / np.sqrt(hidden_dim), size=(hidden_dim, 1)),
            "b2": np.zeros((1, 1), dtype=np.float64),
        }

    def forward(self, X):
        """Q2/Q7：返回 (y_pred,cache)，cache 包含 X、Z1、A1。"""
        if X.ndim != 2 or X.shape[1] != 1 or len(X) == 0:
            raise ValueError("X 必须为非空 (N, 1) 数组")
        # TODO 2 / Q2：在此填写前向传播和缓存。
        # self.activation == 'linear' 时，令 A1 = Z1。
        raise NotImplementedError("请完成 Q2/Q7：前向传播")

    def backward(self, y_pred, y, cache):
        """Q4/Q7：返回 W1、b1、W2、b2 梯度；不能更新参数。"""
        # TODO 3 / Q4：在此填写反向传播。偏置沿样本轴求和并保留维度。
        # linear 隐藏层的局部导数为 1；各梯度必须与参数形状相同。
        raise NotImplementedError("请完成 Q4/Q7：反向传播")

    def step(self, grads, lr):
        """Q6/Q7：原地更新 self.params，返回 None，不修改 grads。"""
        # TODO 4 / Q6：在此填写参数更新。
        raise NotImplementedError("请完成 Q6/Q7：参数更新")


# ==================== 教师提供区：工具 ====================

def make_sine_data(n_samples=64, noise_std=0.0, seed=42):
    """返回 float64 的 (N,1) 数据；基础任务不加噪声。"""
    if not isinstance(n_samples, (int, np.integer)) or n_samples < 2:
        raise ValueError("n_samples 必须为不小于 2 的整数")
    if not np.isfinite(noise_std) or noise_std < 0:
        raise ValueError("noise_std 必须为有限非负数")
    X = np.linspace(-np.pi, np.pi, n_samples, dtype=np.float64).reshape(-1, 1)
    y = np.sin(X)
    if noise_std > 0:
        y = y + np.random.default_rng(seed).normal(0.0, noise_std, size=y.shape)
    return X, y


def gradient_check(net, X, y, epsilon=1e-5, atol=1e-7, rtol=1e-4):
    """逐元素中心差分检查；无论成功或异常，均恢复被扰动的参数。

    每项判据：|analytic-numeric| <= atol + rtol*(|analytic|+|numeric|)。
    max_rel_error 仅作诊断；接近零的梯度应同时看绝对误差及 passed。
    数值损失在这里独立计算，避免错误的学生损失与错误梯度互相印证。
    """
    if not np.isfinite(epsilon) or epsilon <= 0:
        raise ValueError("epsilon 必须为有限正数")
    if not np.isfinite(atol) or not np.isfinite(rtol) or atol < 0 or rtol < 0:
        raise ValueError("容差必须为有限非负数")

    def numerical_loss():
        pred, _ = net.forward(X)
        if pred.shape != y.shape or y.ndim != 2 or y.shape[1] != 1 or len(y) == 0:
            raise ValueError("预测和目标必须具有相同的非空 (N, 1) 形状")
        return float(np.mean((pred - y) ** 2))

    original = {name: value.copy() for name, value in net.params.items()}
    results = {}
    try:
        pred, cache = net.forward(X)
        analytic = net.backward(pred, y, cache)
        for name, param in net.params.items():
            grad = np.asarray(analytic[name]).copy()
            if grad.shape != param.shape:
                raise ValueError(f"{name} 的梯度形状 {grad.shape} 与参数 {param.shape} 不一致")
            numeric = np.empty_like(param, dtype=np.float64)
            for index in np.ndindex(param.shape):
                saved = param[index].copy()
                try:
                    param[index] = saved + epsilon
                    plus = numerical_loss()
                    param[index] = saved - epsilon
                    minus = numerical_loss()
                    numeric[index] = (plus - minus) / (2.0 * epsilon)
                finally:
                    param[index] = saved
            abs_error = np.abs(grad - numeric)
            scale = np.abs(grad) + np.abs(numeric)
            finite = np.all(np.isfinite(grad)) and np.all(np.isfinite(numeric))
            results[name] = {
                "max_abs_error": float(np.max(abs_error)),
                "max_rel_error": float(np.max(abs_error / np.maximum(scale, 1e-12))),
                "passed": bool(finite and np.all(abs_error <= atol + rtol * scale)),
            }
    finally:
        for name, value in original.items():
            net.params[name][...] = value
    return results


def plot_results(X, y, grid, y_grid_pred, history, title=""):
    """返回 (fig,axes)：损失与区间内拟合双图；history 含初始损失。"""
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(np.arange(len(history)), history)
    axes[0].set(xlabel="Completed updates", ylabel="MSE", title="Loss")
    axes[0].grid(alpha=0.3)
    axes[1].scatter(X.ravel(), y.ravel(), s=18, label="Training samples")
    axes[1].plot(grid.ravel(), np.sin(grid).ravel(), "--", label="sin(x)")
    axes[1].plot(grid.ravel(), y_grid_pred.ravel(), label="Prediction")
    axes[1].set(xlabel="x", ylabel="y", title="Fit inside training interval")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig, axes


# ==================== 学生填写区：Q7 训练骨架 ====================

def train(net, X, y, lr=0.02, epochs=10000):
    """全批量训练；history 只包含成功完成且数值有限的更新。"""
    first_loss = mse_loss(net.forward(X)[0], y)
    if not np.isfinite(first_loss):
        raise ValueError("初始损失非有限，请检查数据与初始化")
    history = [first_loss]
    status = "completed"
    attempted_step = 0
    for attempted_step in range(1, epochs + 1):
        saved = {k: v.copy() for k, v in net.params.items()}
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                # TODO 5 / Q7：在此填写前向、损失、反向，得到 grads。
                raise NotImplementedError("Q7：补全训练循环中的前向与反向")
                if not all(np.all(np.isfinite(g)) for g in grads.values()):
                    raise FloatingPointError("非有限梯度")
                # TODO 6 / Q7：在此使用类的 step 完成一次更新。
                raise NotImplementedError("Q7：补全训练循环中的参数更新")
                if not all(np.all(np.isfinite(p)) for p in net.params.values()):
                    raise FloatingPointError("非有限参数")
                next_loss = mse_loss(net.forward(X)[0], y)
                if not np.isfinite(next_loss):
                    raise FloatingPointError("非有限损失")
            history.append(next_loss)
        except FloatingPointError as exc:
            for name in saved:
                net.params[name][...] = saved[name]
            status = f"stopped at attempted update {attempted_step}: {exc}"
            break
    info = {"status": status, "completed_updates": len(history) - 1,
            "attempted_update": attempted_step, "lr": lr}
    return np.asarray(history), info


# ==================== 教师提供区：公开判题 ====================

KEYS = ("W1", "b1", "W2", "b2")


def _snapshot(net):
    return {name: value.copy() for name, value in net.params.items()}


def _assert_unchanged(net, saved):
    for name in KEYS:
        np.testing.assert_array_equal(net.params[name], saved[name])


def _check_loss():
    pred, y = np.array([[1.], [3.]]), np.array([[0.], [1.]])
    result = mse_loss(pred, y)
    assert np.ndim(result) == 0, "损失必须为标量"
    np.testing.assert_allclose(result, 2.5)
    assert mse_loss(y, y) == 0.0
    np.testing.assert_allclose(mse_loss(np.vstack([pred, pred]), np.vstack([y, y])), result)
    try:
        mse_loss(pred, y.ravel())
    except ValueError:
        pass
    else:
        raise AssertionError("必须拒绝 (N,) 标签，避免意外广播")


def _check_forward(activation):
    net = TwoLayerNet(hidden_dim=3, seed=7, activation=activation)
    # 非零偏置避免只在零偏置特例下检查。
    net.params["b1"][...] = [[0.1, -0.2, 0.3]]
    net.params["b2"][...] = 0.15
    saved = _snapshot(net)
    X = np.array([[-0.7], [0.2], [0.9]])
    pred, cache = net.forward(X)
    p = net.params
    z = X @ p["W1"] + p["b1"]
    a = np.tanh(z) if activation == "tanh" else z
    assert pred.shape == (3, 1), "预测形状应为 (N,1)"
    assert set(cache) >= {"X", "Z1", "A1"}, "缓存缺少必要变量"
    np.testing.assert_allclose(cache["X"], X)
    np.testing.assert_allclose(cache["Z1"], z)
    np.testing.assert_allclose(cache["A1"], a)
    np.testing.assert_allclose(pred, a @ p["W2"] + p["b2"])
    assert np.isfinite(pred).all()
    _assert_unchanged(net, saved)


def _check_backward(activation):
    net = TwoLayerNet(hidden_dim=3, seed=7, activation=activation)
    X = np.array([[-0.7], [0.2], [0.9]])
    y = np.sin(X)
    saved = _snapshot(net)
    pred, cache = net.forward(X)
    grads = net.backward(pred, y, cache)
    assert set(grads) == set(KEYS), "梯度字典必须有四组参数"
    for name in KEYS:
        assert grads[name].shape == net.params[name].shape, f"{name} 梯度形状错误"
        assert np.isfinite(grads[name]).all(), f"{name} 梯度非有限"
    _assert_unchanged(net, saved)
    # 复制数据不应改变平均损失的梯度：检查重复除以 N 等错误。
    doubled_X, doubled_y = np.vstack([X, X]), np.vstack([y, y])
    doubled_pred, doubled_cache = net.forward(doubled_X)
    doubled_grads = net.backward(doubled_pred, doubled_y, doubled_cache)
    for name in KEYS:
        np.testing.assert_allclose(grads[name], doubled_grads[name], atol=1e-12)
    report = gradient_check(net, X, y)
    for name, item in report.items():
        print(f"    {name}: abs={item['max_abs_error']:.3e}, "
              f"rel={item['max_rel_error']:.3e}, passed={item['passed']}")
    assert all(item["passed"] for item in report.values()), "有限差分检查失败"
    _assert_unchanged(net, saved)


def _check_step():
    # 独立构造梯度，即使前向/反向尚未完成，也可以单独检查更新。
    net = TwoLayerNet(hidden_dim=3, seed=7)
    saved = _snapshot(net)
    ids = {name: id(value) for name, value in net.params.items()}
    grads = {name: np.full_like(value, 0.25) for name, value in net.params.items()}
    grad_saved = {name: value.copy() for name, value in grads.items()}
    assert net.step(grads, 0.01) is None, "step 应返回 None"
    for name in KEYS:
        np.testing.assert_allclose(net.params[name], saved[name] - 0.01 * grad_saved[name])
        np.testing.assert_array_equal(grads[name], grad_saved[name])
        assert id(net.params[name]) == ids[name], "应原地更新参数数组"


def _check_training():
    X, y = make_sine_data()
    net = TwoLayerNet()
    history, info = train(net, X, y, lr=0.02, epochs=20)
    assert info["status"] == "completed", info
    assert info["completed_updates"] == 20
    assert len(history) == 21, "必须记录初始损失与每次更新后的损失"
    assert np.isfinite(history).all()
    np.testing.assert_allclose(history[-1], mse_loss(net.forward(X)[0], y))
    assert history[-1] < history[0], "该固定设置的训练应整体下降"
    # 独立执行相同的 20 次更新，检查训练循环是否真的使用了每轮的新梯度。
    manual = TwoLayerNet()
    for _ in range(20):
        pred, cache = manual.forward(X)
        manual.step(manual.backward(pred, y, cache), 0.02)
    for name in KEYS:
        np.testing.assert_allclose(net.params[name], manual.params[name])


def run_checks():
    """逐项报告；未完成不隐藏其他独立项，全部通过返回 True。"""
    cases = [
        ("Q3 损失（TODO 1）", _check_loss),
        ("Q2 tanh 前向（TODO 2）", lambda: _check_forward("tanh")),
        ("Q2 linear 前向（TODO 2）", lambda: _check_forward("linear")),
        ("Q4/Q5 tanh 梯度（依赖 TODO 2、3）", lambda: _check_backward("tanh")),
        ("Q4/Q5 linear 梯度（依赖 TODO 2、3）", lambda: _check_backward("linear")),
        ("Q6 更新（TODO 4）", _check_step),
        ("Q7 训练（依赖 TODO 1–6）", _check_training),
    ]
    counts = {"通过": 0, "未完成": 0, "失败": 0}
    for title, check in cases:
        try:
            check()
        except NotImplementedError as error:
            counts["未完成"] += 1
            print(f"[未完成] {title}：{error}")
        except Exception as error:
            counts["失败"] += 1
            print(f"[失败] {title}：{type(error).__name__}: {error}")
        else:
            counts["通过"] += 1
            print(f"[通过] {title}")
    print("\n公开自测汇总：", counts)
    passed = counts["通过"] == len(cases)
    if not passed:
        print("先完成或修复上游题目，再检查依赖它的后续项。")
    print("通过自测后仍需完成 Q1、Q8 分析与真实排错记录。")
    return passed


# ==================== 教师提供区：完整实验与选做演示 ====================


def run_experiments(with_lr=False):
    import matplotlib.pyplot as plt

    X, y = make_sine_data()
    grid = np.linspace(-np.pi, np.pi, 400).reshape(-1, 1)
    records = {}
    for activation in ("tanh", "linear"):
        net = TwoLayerNet(hidden_dim=16, seed=42, activation=activation)
        history, info = train(net, X, y, lr=0.02, epochs=10000)
        pred, _ = net.forward(grid)
        records[activation] = (net, history, info)
        print(activation, info, "末次有限训练 MSE:", history[-1],
              "区间内网格 MSE:", mse_loss(pred, np.sin(grid)))
        plot_results(X, y, grid, pred, history, title=f"Q7/Q8: {activation}")
    if with_lr:
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
        for ax, lr in zip(axes, (0.002, 0.02, 0.2)):
            # 每次从相同初始参数重新训练，仅改变学习率。
            net = TwoLayerNet(hidden_dim=16, seed=42, activation="tanh")
            history, info = train(net, X, y, lr=lr, epochs=10000)
            print("Q9", info, "末次有限 MSE:", history[-1])
            ax.plot(np.arange(len(history)), np.log10(np.maximum(history, np.finfo(float).tiny)))
            ax.set(xlabel="Completed updates", ylabel="log10(finite MSE)", title=f"lr={lr}")
            ax.grid(alpha=0.3)
        fig.tight_layout()
    print("网格仅检查训练区间内拟合；停止并回滚不表示模型收敛。")
    plt.show()
    return records


def torch_demo():
    try:
        import torch
    except ImportError:
        print("未安装 PyTorch，跳过 Q10。")
        return
    X = np.array([[-0.7], [0.2], [0.9]])
    y = np.sin(X)
    net = TwoLayerNet(hidden_dim=3, seed=7)
    pred, cache = net.forward(X)
    grads = net.backward(pred, y, cache)
    tx, ty = torch.tensor(X, dtype=torch.float64), torch.tensor(y, dtype=torch.float64)
    params = {name: torch.tensor(value.copy(), dtype=torch.float64, requires_grad=True)
              for name, value in net.params.items()}
    result = torch.tanh(tx @ params["W1"] + params["b1"]) @ params["W2"] + params["b2"]
    loss = ((result - ty) ** 2).mean()
    np.testing.assert_allclose(result.detach().numpy(), pred)
    np.testing.assert_allclose(loss.item(), mse_loss(pred, y))
    loss.backward()
    for name, value in params.items():
        np.testing.assert_allclose(value.grad.numpy(), grads[name], rtol=1e-10, atol=1e-12)
        np.testing.assert_array_equal(value.detach().numpy(), net.params[name])
    print("[通过] Q10 自动求导与手写 BP 对齐；参数没有被 backward 更新。")


# Q1/Q8/Q9/排错的文字回答可写在此，也可在单独实验报告中提交。
ANALYSIS = {
    "Q1 形状、计算图、广播风险": "",
    "Q8 事前预测、控制变量、实际结果与仿射函数解释": "",
    "Q9 事前预测与实际结果（进阶，未做写跳过）": "",
    "真实排错：现象、依据、修改、复验、AI 建议核验": "",
}


def main():
    parser = argparse.ArgumentParser(description="实验四单文件：学生 TODO 与公开自测")
    parser.add_argument("--run", action="store_true", help="通过自测后执行 Q7/Q8 并绘图")
    parser.add_argument("--with-lr", action="store_true", help="与 --run 一起使用，执行 Q9")
    parser.add_argument("--torch", action="store_true", help="通过自测后演示 Q10")
    args = parser.parse_args()
    if args.with_lr and not args.run:
        parser.error("--with-lr 需要同时指定 --run")
    passed = run_checks()
    if passed:
        if args.run:
            run_experiments(args.with_lr)
        if args.torch:
            torch_demo()
    elif args.run or args.torch:
        print("请先完成 TODO 并通过自测，再运行完整实验或 PyTorch 演示。")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
