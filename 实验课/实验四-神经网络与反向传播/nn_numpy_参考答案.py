"""实验四教师参考：单隐藏层网络；训练循环由 notebook 负责。"""

import numpy as np


def mse_loss(y_pred, y):
    """单输出均方误差，不含 1/2；禁止静默广播成 (N, N)。"""
    if y_pred.shape != y.shape or y.ndim != 2 or y.shape[1] != 1:
        raise ValueError("y_pred 和 y 必须具有相同的 (N, 1) 形状")
    if len(y) == 0:
        raise ValueError("至少需要一个样本")
    return float(np.mean((y_pred - y) ** 2))


class TwoLayerNet:
    """X (N,1) → 隐藏层 (N,H) → 线性输出 (N,1)。"""

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
        """返回预测和反向传播所需的缓存，不改变参数。"""
        if X.ndim != 2 or X.shape[1] != 1 or len(X) == 0:
            raise ValueError("X 必须为非空 (N, 1) 数组")
        p = self.params
        Z1 = X @ p["W1"] + p["b1"]
        A1 = np.tanh(Z1) if self.activation == "tanh" else Z1
        y_pred = A1 @ p["W2"] + p["b2"]
        return y_pred, {"X": X, "Z1": Z1, "A1": A1}

    def backward(self, y_pred, y, cache):
        """链式法则只计算梯度；此处不更新任何参数。"""
        if y_pred.shape != y.shape or y.ndim != 2 or y.shape[1] != 1 or len(y) == 0:
            raise ValueError("y_pred 和 y 必须具有相同的非空 (N, 1) 形状")
        X, A1 = cache["X"], cache["A1"]
        dY = 2.0 * (y_pred - y) / len(y)
        dW2 = A1.T @ dY
        db2 = np.sum(dY, axis=0, keepdims=True)
        dA1 = dY @ self.params["W2"].T
        dZ1 = dA1 * (1.0 - A1 ** 2) if self.activation == "tanh" else dA1
        dW1 = X.T @ dZ1
        db1 = np.sum(dZ1, axis=0, keepdims=True)
        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    def step(self, grads, lr):
        """原地执行梯度下降，不改变传入的梯度。"""
        # 更新前先验证全部形状，避免一部分参数已更新才发现错误。
        for name, value in self.params.items():
            if grads[name].shape != value.shape:
                raise ValueError(f"{name} 的梯度形状与参数不一致")
        for name in self.params:
            self.params[name] -= lr * grads[name]
