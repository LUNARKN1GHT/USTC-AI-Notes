---
tags:
  - 类型/定义
aliases:
  - LReLU
  - PReLU
  - ELU
---
# ReLU

ReLU：Rectified Linear Unit，通过非饱和性设计改善梯度传播。标准的 ReLU 定义为

$$
\text{ReLU}(z) = \max(0, z)
$$

就是直接把负数部分直接截断。

相比 [[Sigmoid函数|Sigmoid]]，[[ReLU]]解决的问题是，0 和 1 附近的饱和问题。观察 [[Sigmoid函数|Sigmoid]]函数不难发现，这个函数在 0 和 1 附近非常平滑，存在梯度消失的问题，不方便训练。

## LReLU

Leaky ReLU：标准的 ReLU 有点过于激进了，我们可以折中，负区间用系数压住。能有效缓解神经元死亡现象。

$$
\text{LReLU}(z) = 

\left\{\begin{matrix}
 z,  & z > 0\\
 0.01 z, & z \le 0
\end{matrix}\right.
$$

## PReLU

Parameterize ReLU：上述的 [[ReLU|LReLU]] 似乎有点死板，没事，我们把这梯度变成可变参数：

$$
\text{LReLU}(z) = 

\left\{\begin{matrix}
 z,  & z > 0\\
 \alpha z, & z \le 0
\end{matrix}\right. \quad \alpha \in (0, 1)
$$

这里的 $\alpha$ 是可训练的参数，适用于各层数据分布显著的情况，具有自适应调节能力。

## ELU

Exponential Linear Unit: 用指数函数来做，其衰减性会更好一些

$$
\text{ELU}(z) = 

\left\{\begin{matrix}
 z & z > 0 \\
 \alpha \left(e^z - 1 \right) & z \le 0
\end{matrix}\right.
$$

优势包括：负区间连续可导；输出均值接近零且加速收敛；保持正区间梯度优势。但计算成本略高。
