---
tags:
  - 类型/算法
aliases:
  - Sigmoid
---

Sigmoid函数可以讲任何实数值压缩到 $(0,1)$ 之间，正好是概率的定义域：

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

> [!note]
> 小贴士，Sigmoid函数求导形式很好看结果就是 $\sigma’{z} = \sigma(z) \left(1 - \sigma(z) \right)$
> 这个结果也反映了另一个事实，Sigmoid函数的导数的最大值只有 0.25，在两侧函数更加平滑，这也导致在深度神经网络里 Sigmoid 作为激活函数表现不佳。
