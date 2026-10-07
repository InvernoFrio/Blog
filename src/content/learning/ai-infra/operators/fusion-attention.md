---
title: 算子融合与在线 Attention
description: 从 Bias 加 ReLU 的中间张量，到在线 Softmax 的状态更新，理解怎样减少计算过程中的数据移动。
course: ai-infra
topic: operators
module: 组合与验证
order: 6
updated: 2026-10-07
prerequisites: [归约与稳定 Softmax, 分块矩阵乘法]
---

## 中间张量也有成本

考虑 $Y=\operatorname{ReLU}(X+b)$，其中 $X$ 是 $M\times N$ 矩阵，$b$ 是长度为 $N$ 的偏置。先计算加法并写出临时矩阵，再执行 ReLU，需要把这个临时矩阵读回来。

忽略缓存与分配开销，按逻辑读写估算，分离执行涉及 $X$ 的读取、临时矩阵的写入和读取、最终输出的写入，再加偏置读取。融合后可以直接写出结果，省去两次临时矩阵访问。

这只是数据量分析。编译器可能已经做了融合，偏置也可能命中缓存，规模较小时还可能受启动开销主导。性能报告要说明比较的是 eager、编译后图，还是独立自定义 kernel。

## 一个简单的融合实现

对于连续矩阵，展平索引 `index` 对应的列是 `index % N`。于是同一个 program 可以加载输入和相应偏置，再计算并写回：

```python
values = tl.load(X + index, valid, other=0).to(tl.float32)
bias = tl.load(BIAS + index % COLS, valid, other=0).to(tl.float32)
tl.store(Y + index, tl.maximum(values + bias, 0), valid)
```

完整 `_bias_relu` 见 [kernels.py](../../../../learning/operators/kernels.py)。当前实验对比的是两个明确分离的 eager 操作，两边都预分配输出，临时矩阵也提前分配。若引入 `torch.compile` 基线，应重新记录编译成本与测量范围。

## 为什么 Attention 更值得研究

对于单个注意力头，经典表达式为

$$
S=\frac{QK^{\mathsf T}}{\sqrt{d}},\qquad
P=\operatorname{softmax}(S),\qquad O=PV.
$$

如果直接把 $S$ 或 $P$ 作为完整矩阵存储，长度为 $L$ 的序列会产生 $L\times L$ 的中间结果。问题是：是否能分块读取 $K$、$V$，逐步形成 $O$，而不把整个概率矩阵写到全局内存？

[FlashAttention 原始论文](https://arxiv.org/abs/2205.14135)研究了这种面向 IO 的精确注意力计算。它避免完整中间矩阵的存储，并不意味着标准稠密 Attention 的算术工作量从二次复杂度变成了线性。

## 在线 Softmax 的状态

先固定一个 query，只考虑它对应的一行 score。已经处理的部分维护三个量：最大值 $m$、归一化分母 $\ell$、未归一化的输出向量 $u$。

$$
\ell=\sum_{j\in\mathrm{seen}} e^{s_j-m},\qquad
u=\sum_{j\in\mathrm{seen}} e^{s_j-m}v_j.
$$

读取下一块 score 和 value 后，令

$$
m'=\max\left(m,\max_{j\in\mathrm{block}}s_j\right),\qquad
\alpha=e^{m-m'}.
$$

因为旧的状态以 $m$ 为基准，新状态以 $m'$ 为基准，所以必须把旧状态重新缩放：

$$
\ell'=\alpha\ell+\sum_{j\in\mathrm{block}}e^{s_j-m'},
$$

$$
u'=\alpha u+\sum_{j\in\mathrm{block}}e^{s_j-m'}v_j.
$$

最后输出 $o=u/\ell$。初始化可以取 $m=-\infty$、$\ell=0$、$u=0$；本篇的每个有效 block 都包含有限 score，第一次更新的修正系数为零。有关在线归一化的原始方法，见 [Online normalizer calculation for softmax](https://arxiv.org/abs/1805.02867)。

## 修正系数为什么不能省略

假设旧最大值为 `2`，下一块把最大值提高到了 `5`。旧分母和旧向量累计的是 `exp(score - 2)`，而新块使用 `exp(score - 5)`。直接相加混合了两个基准。

乘上 `exp(2 - 5)`，才把旧状态统一到新基准。这个推导提供了一个很好的测试用例：安排后面的 block 出现更大的 score，检查遗漏修正的实现是否失败。

CPU 脚本的 `online_attention` 实现了这组更新，并与先计算完整 score 行的参考实现比较。它只验证单个 query、非空 key/value、无 mask、无 dropout 的公式关系，不是 GPU FlashAttention 的完整实现。

## 从公式到真实 Kernel

GPU 实现还需要确定 query tile 与 key/value tile、资源占用、流水线、mask、反向传播和数值精度。这些工作决定了“公式正确”怎样变成“实现高效”。本专题先建立在线状态的直觉，再把完整 kernel 留作后续进阶。

融合也有边界。较大的融合区域可能增加资源需求；中间结果若有多个使用者，重新计算与保留之间还有取舍。因此，融合方案需要在整个计算图和实际工作负载中评估。

## 练习

用两个小 block 手算 $m$、$\ell$ 和 $u$ 的更新，并与直接 Softmax 比较。然后将 CPU 脚本的 block 大小改为 `1`、`3`、`8`：为什么分块方式改变，但数学目标仍然相同？

最后尝试加入因果 mask。先定义哪些 key 可见，再处理没有有效元素的 block；说明为什么全被 mask 的行需要明确规定行为，不能直接套用有限 score 的初始化论证。
