---
tags:
  - 类型/算法
---

# Q 学习

Q 学习是无模型、异策略的时序差分控制算法，更新规则为：

$$
Q(s_t,a_t)\leftarrow Q(s_t,a_t)+\alpha\left[r_{t+1}+\gamma\max_{a'}Q(s_{t+1},a')-Q(s_t,a_t)\right].
$$

表格形式适合有限离散状态；状态空间较大时可用神经网络近似 Q 函数，得到 DQN。
