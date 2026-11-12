---
tags:
  - 类型/模型
aliases:
  - 长短时记忆网络
---
# LSTM

长短时记忆网络（LSTM）在[[RNN]]的隐藏状态 $h_t$ 之外维护单元状态 $c_t$。门控网络根据当前输入 $x_t$ 与上一时刻的隐藏状态 $h_{t-1}$，逐维决定保留、写入和读出多少信息。记 $[x_t;h_{t-1}]$ 为向量拼接，$\sigma$ 为 [[Sigmoid函数|Sigmoid]]，$\odot$ 为逐元素乘法。

#### 遗忘门

$$
f_t = \sigma(W_f \cdot [x_t; h_{t - 1}] + b_f), \quad \tilde{c}_{t - 1} = f_t \odot c_{t - 1}
$$

其中 $f_t \in {[0, 1]}^{d_h}$ 为遗忘门激活向量，决定上一时刻的单元状态 $c_{t-1}$ 保留多少。

#### 输入门及候选状态

$$
\begin{align}
i_t &= \sigma(W_i \cdot [x_t; h_{t - 1}] + b_i) \\
\tilde{c}_t &= \tanh(W_c \cdot [x_t; h_{t - 1}] + b_c) \\
c_t &= \tilde{c}_{t - 1} + i_t \odot \tilde{c}_t
\end{align}
$$

其中：

- $i_t \in [0, 1]^{d_h}$ 为输入门激活向量，控制新信息写入单元状态的比例；
- $\tilde{c}_t \in [-1, 1]^{d_h}$ 为候选状态，提供可能写入的新信息。

#### 输出门及隐藏状态

$$
o_t = \sigma(W_o \cdot [x_t; h_{t - 1}] + b_o), \quad h_t = o_t \odot \tanh(c_t)
$$

其中 $o_t \in [0, 1]^{d_h}$ 为输出门激活向量，控制隐藏状态输出的比例；$h_t \in \mathbb{R}^{d_h}$ 为时间步 $t$ 的隐藏状态，用于下一时刻及网络输出。

#### 常见优化方式：

- 堆叠多层（Multi-layer LSTM)
- 双向 (BiLSTM)
- 层归一化 (Layer-normalized LSTM)
