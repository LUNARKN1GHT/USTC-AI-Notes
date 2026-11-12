"""实验四教师提供的工具；不导入参考答案，也不代替学生的训练循环。"""

import numpy as np


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
