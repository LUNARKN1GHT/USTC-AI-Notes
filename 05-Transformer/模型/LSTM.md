---
tags:
  - 类型/模型
aliases:
  - 长短时记忆网络
---

直观来讲，在一个时间序列模型中，距离较远的内容对当下的影响，肯定比较近的内容更小。因此，我们可以对时间序列模型进行优化，让它可以“遗忘”较远的内容，“记住”较近的内容。

#### 遗忘门

$$
f_t = \sigma(W_f \cdot [x_t; h_{t - 1}] + b_f), \quad \tilde{c}_{t - 1} = f_t \odot c_{t - 1}
$$

其中 $f_t \in {[0, 1]}^{d_h}$ 为遗忘门激活向量。其余元素内容与上面的定义相同。

#### 输入门及候选状态

$$
\begin{align}
i_t &= \sigma(W_i \cdot [x_t; h_{t - 1} + b_i]) \\
\tilde{c}_t &= \tanh(W_c \cdot [x_t; h_{t - 1} + b_c] \\
c_t &= \tilde{c}_{t - 1} + i_t \odot \tilde{c}_t
\end{align}
$$

其中：

- $i_t \in [0, 1]^{d_h}$ 为输入门激活向量，控制新信息写入单元状态的比例；
- $\tilde{c}_t \in [-1, 1]^{d_h}$ 为候选状态，提供可能写入的新信息。

#### 输入门及隐藏状态[^1]更新

$$
o_t = \sigma(W_o \cdot [x_t; h_{t - 1}] + b_0), \quad h_t = o_t \odot \tanh(c_t)
$$

其中 $o_t \in [0, 1]^{d_h}$ 为输出门激活向量，控制隐藏状态输出的比例；$h_t \in \mathbb{R}^{d_h}$ 为当前时间步的 $t$ 的隐藏状态，用于下一时刻及网络输出。

#### 常见优化方式：

- 堆叠多层（Multi-layer LSTM)
- 双向 (BiLSTM)
- 层归一化 (Layer-normalized LSTM)
