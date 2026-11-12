"""教师验收：在本目录运行 python 验证参考答案.py，仅依赖 NumPy。

这是材料维护脚本，不是学生的训练循环；学生训练与实验记录仍在 notebook。
"""

import numpy as np

from lab_utils import gradient_check, make_sine_data
from nn_numpy_参考答案 import TwoLayerNet, mse_loss


KEYS = ("W1", "b1", "W2", "b2")


def snapshots(net):
    return {name: value.copy() for name, value in net.params.items()}


def assert_same(net, before):
    for name in KEYS:
        np.testing.assert_array_equal(net.params[name], before[name])


def check_math(activation):
    X = np.array([[-0.7], [0.2], [0.9]], dtype=np.float64)
    y = np.sin(X)
    net = TwoLayerNet(hidden_dim=3, seed=7, activation=activation)
    before = snapshots(net)
    pred, cache = net.forward(X)
    assert pred.shape == (3, 1)
    assert set(cache) >= {"X", "Z1", "A1"}
    assert cache["Z1"].shape == cache["A1"].shape == (3, 3)
    assert tuple(net.params) == KEYS
    assert [p.shape for p in net.params.values()] == [(1, 3), (1, 3), (3, 1), (1, 1)]
    assert all(p.dtype == np.float64 for p in net.params.values())
    grads = net.backward(pred, y, cache)
    assert tuple(grads) == KEYS
    assert all(grads[k].shape == net.params[k].shape for k in KEYS)
    assert_same(net, before)

    # 独立检查偏置求和与 mean 的归一化：复制样本不应改变梯度。
    dY = 2.0 * (pred - y) / len(y)
    dZ = dY @ net.params["W2"].T
    if activation == "tanh":
        dZ = dZ * (1.0 - cache["A1"] ** 2)
    np.testing.assert_allclose(grads["b1"], dZ.sum(axis=0, keepdims=True))
    np.testing.assert_allclose(grads["b2"], dY.sum(axis=0, keepdims=True))
    doubled_X, doubled_y = np.vstack([X, X]), np.vstack([y, y])
    doubled_pred, doubled_cache = net.forward(doubled_X)
    doubled_grads = net.backward(doubled_pred, doubled_y, doubled_cache)
    for name in KEYS:
        np.testing.assert_allclose(grads[name], doubled_grads[name], atol=1e-14)

    report = gradient_check(net, X, y)
    assert all(item["passed"] for item in report.values()), report
    assert_same(net, before)
    largest_error = max(item["max_abs_error"] for item in report.values())

    # step 必须原地更新参数，且不能修改调用方持有的梯度。
    grad_copies = {k: v.copy() for k, v in grads.items()}
    parameter_ids = {k: id(v) for k, v in net.params.items()}
    loss_before = mse_loss(pred, y)
    assert net.step(grads, 1e-3) is None
    for name in KEYS:
        np.testing.assert_array_equal(grads[name], grad_copies[name])
        np.testing.assert_allclose(net.params[name], before[name] - 1e-3 * grads[name])
        assert id(net.params[name]) == parameter_ids[name]
    assert mse_loss(net.forward(X)[0], y) < loss_before
    if activation == "linear":
        p = net.params
        affine = X @ (p["W1"] @ p["W2"]) + p["b1"] @ p["W2"] + p["b2"]
        np.testing.assert_allclose(net.forward(X)[0], affine, atol=1e-14)
    print(f"[通过] {activation}: 形状、偏置、平均、有限差分、单次更新；最大绝对误差 {largest_error:.3e}")


def check_checker():
    X = np.array([[-0.7], [0.2], [0.9]])
    y = np.sin(X)

    class WrongGradient(TwoLayerNet):
        def backward(self, y_pred, y, cache):
            grads = super().backward(y_pred, y, cache)
            grads["W1"] *= -1.0
            return grads

    wrong = WrongGradient(hidden_dim=3, seed=7)
    report = gradient_check(wrong, X, y)
    assert not report["W1"]["passed"], "工具应能发现故意反转的梯度"

    class InterruptedForward(TwoLayerNet):
        def forward(self, X):
            self.calls = getattr(self, "calls", 0) + 1
            if self.calls == 3:
                raise RuntimeError("模拟有限差分中途异常")
            return super().forward(X)

    interrupted = InterruptedForward(hidden_dim=3, seed=7)
    before = snapshots(interrupted)
    try:
        gradient_check(interrupted, X, y)
    except RuntimeError as error:
        assert "模拟" in str(error)
    else:
        raise AssertionError("预期工具传播模拟异常")
    assert_same(interrupted, before)
    print("[通过] 梯度检查能发现错误梯度，且异常后精确恢复参数")


def check_data_and_loss():
    X, y = make_sine_data()
    assert X.shape == y.shape == (64, 1)
    assert X.dtype == y.dtype == np.float64
    np.testing.assert_array_equal(y, np.sin(X))
    assert X[0, 0] == -np.pi and X[-1, 0] == np.pi
    np.testing.assert_array_equal(make_sine_data(noise_std=0.1)[1], make_sine_data(noise_std=0.1)[1])
    assert mse_loss(np.array([[1.0], [3.0]]), np.array([[0.0], [1.0]])) == 2.5
    try:
        mse_loss(np.ones((3, 1)), np.ones(3))
    except ValueError:
        pass
    else:
        raise AssertionError("不应允许目标静默广播")
    print("[通过] 数据、种子、float64、MSE 数值与广播防护")


def train_for_validation(activation, lr, epochs=10000):
    """与 notebook 相同：异常时回滚，仅记录有限且成功的更新。"""
    X, y = make_sine_data()
    net = TwoLayerNet(activation=activation)
    history = [mse_loss(net.forward(X)[0], y)]
    reason = "completed"
    for attempted_step in range(1, epochs + 1):
        saved = snapshots(net)
        try:
            with np.errstate(over="raise", invalid="raise", divide="raise"):
                pred, cache = net.forward(X)
                grads = net.backward(pred, y, cache)
                if not all(np.isfinite(g).all() for g in grads.values()):
                    raise FloatingPointError("非有限梯度")
                net.step(grads, lr)
                if not all(np.isfinite(p).all() for p in net.params.values()):
                    raise FloatingPointError("非有限参数")
                loss = mse_loss(net.forward(X)[0], y)
                if not np.isfinite(loss):
                    raise FloatingPointError("非有限损失")
            history.append(loss)
        except FloatingPointError as error:
            for name in saved:
                net.params[name][...] = saved[name]
            reason = f"stopped at attempted update {attempted_step}: {error}"
            break
    return net, np.asarray(history), reason


def check_training():
    records = {}
    for activation, lr in [("tanh", 0.02), ("linear", 0.02), ("tanh", 0.002), ("tanh", 0.2)]:
        net, history, reason = train_for_validation(activation, lr)
        records[activation, lr] = net, history, reason
        print(f"[训练] activation={activation:6s}, lr={lr:<5}, 更新次数={len(history)-1}, "
              f"初始 MSE={history[0]:.8f}, 末次 MSE={history[-1]:.8g}, 状态={reason}")
        if reason == "completed":
            assert len(history) == 10001 and np.isfinite(history).all()
            X, y = make_sine_data()
            np.testing.assert_allclose(history[-1], mse_loss(net.forward(X)[0], y))
    base, base_history, base_reason = records["tanh", 0.02]
    _, linear_history, linear_reason = records["linear", 0.02]
    _, slow_history, slow_reason = records["tanh", 0.002]
    assert base_reason == linear_reason == slow_reason == "completed"
    # 宽容阈值只用于维护材料；课程评分不应只看某个最终损失。
    assert base_history[-1] < 0.005
    assert 0.20 < linear_history[-1] < 0.22
    assert base_history[-1] < slow_history[-1] < 0.04
    grid = np.linspace(-np.pi, np.pi, 400).reshape(-1, 1)
    grid_mse = mse_loss(base.forward(grid)[0], np.sin(grid))
    assert grid_mse < 0.005
    print(f"[通过] 基础与对照训练；基础区间内网格 MSE={grid_mse:.8f}（不是外推测试）")


def main():
    check_data_and_loss()
    check_math("tanh")
    check_math("linear")
    check_checker()
    check_training()
    print("全部验证通过。")


if __name__ == "__main__":
    main()
