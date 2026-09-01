---
tags:
  - 类型/算法
---

# Skip-gram

Skip-gram 根据中心词预测其上下文词。给定中心词 $w_t$ 和窗口内上下文，训练目标是最大化：

$$
\sum_t\sum_{-c\le j\le c,\,j\ne0}\log P(w_{t+j}\mid w_t).
$$

完整 Softmax 在词表很大时计算昂贵，实践中常使用负采样或层次 Softmax。
