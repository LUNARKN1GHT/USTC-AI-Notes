---
tags:
  - 类型/算法
aliases:
  - Softmax
---

# Softmax函数

Softmax 函数是多分类问题中非常常用的一种[[激活函数]]，它可以将一个实数向量 **转换为概率分布**，使得输出的每个分量都在 0 到 1 之间，并且所有分量之和为 1。

假设有一个 n 维向量 $\mathbf{z} = [z_1, z_2, \dots, z_n]$，Softmax 函数定义为：

$$\text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^{n} e^{z_j}}, \quad i = 1,2,\dots,n$$
