"""实验四学生模块：Q7 将 notebook 的 Q2–Q6 实现整理到本文件。"""

import numpy as np


def mse_loss(y_pred, y):
    """Q3/Q7：返回 mean((y_pred-y)**2)，形状均为 (N,1)，不含 1/2。"""
    if y_pred.shape != y.shape or y.ndim != 2 or y.shape[1] != 1:
        raise ValueError("y_pred 和 y 必须具有相同的 (N, 1) 形状")
    if len(y) == 0:
        raise ValueError("至少需要一个样本")
    # TODO：把 Q3 的 loss_mse 整理到这里。
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
        # TODO：由 self.params 计算隐藏层与线性输出。
        # self.activation == 'linear' 时，令 A1 = Z1。
        raise NotImplementedError("请完成 Q2/Q7：前向传播")

    def backward(self, y_pred, y, cache):
        """Q4/Q7：返回 W1、b1、W2、b2 梯度；不能更新参数。"""
        # TODO：从 dY 开始应用链式法则，偏置沿样本轴求和并保留维度。
        # linear 隐藏层的局部导数为 1；各梯度必须与参数形状相同。
        raise NotImplementedError("请完成 Q4/Q7：反向传播")

    def step(self, grads, lr):
        """Q6/Q7：原地更新 self.params，返回 None，不修改 grads。"""
        # TODO：每个参数沿负梯度方向移动 lr * grads[name]。
        raise NotImplementedError("请完成 Q6/Q7：参数更新")
