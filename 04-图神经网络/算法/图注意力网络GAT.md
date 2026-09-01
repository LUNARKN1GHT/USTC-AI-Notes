---
tags:
  - 类型/算法
aliases:
  - GAT
---

# 图注意力网络 GAT

图注意力网络为不同邻居学习不同的聚合权重。对边 $(i,j)$，先计算注意力分数 $e_{ij}$，再在节点 $i$ 的邻域内归一化：

$$
\alpha_{ij}=\operatorname{softmax}_j(e_{ij}),\qquad
h_i'=\sigma\!\left(\sum_{j\in\mathcal{N}(i)}\alpha_{ij}Wh_j\right).
$$

多头注意力可以提升训练稳定性，并允许模型从多个子空间聚合邻居信息。
