---
title: 因果 Attention：从前向到反向
description: 推导缩放点积 Attention 的矩阵梯度，检查未来信息泄漏，并理解训练时的并行。
course: llm
topic: from-scratch
module: 数据与模型
order: 4
updated: 2026-10-08
prerequisites: [线性层反向传播, Softmax 梯度]
---

## 一个头的计算过程

暂时省略 batch 和头数，$Q,K\in\mathbb R^{T\times d_h}$，$V\in\mathbb R^{T\times d_v}$。它们由输入分别经过三个可学习线性层得到。

$$
S=\frac{QK^T}{\sqrt{d_h}}+M,\qquad
P=\operatorname{softmax}_{\text{行}}(S),\qquad O=PV.
$$

$M_{ij}=0$ 当 $j\le i$，否则为 $-\infty$。每个 query 位置只读取当前与之前的 key/value。`softmax(dim=-1)` 在 key 轴归一化，不能在 query 轴替代。

若 query 和 key 各维近似独立、方差为 1，点积方差随 $d_h$ 增长。除以 $\sqrt{d_h}$ 让初始分数尺度较稳定，减轻 Softmax 过早饱和。这是初始化附近的启发式解释，不是所有训练阶段的分布定理。

## 多头实现的形状

| 张量 | 实验代码形状 |
| --- | --- |
| 输入 | $[B,T,d]$ |
| 一次投影后的 QKV | $[B,T,3d]$ |
| 拆头后的 Q、K、V | $[B,H,T,d_h]$，$d_h=d/H$ |
| 分数与概率 | $[B,H,T,T]$ |
| 每头输出 | $[B,H,T,d_h]$ |
| 拼接并输出投影 | $[B,T,d]$ |

转置后内存可能不连续，实验通过 `transpose(...).contiguous().view(...)` 拼头。轴的顺序错误有时仍能得到合法 shape，因此形状检查之外还需要与数学参考比较。

## Attention 的反向公式

先忽略 dropout，设上游梯度 $G=\partial L/\partial O$。由矩阵乘法：

$$
dV=P^TG,\qquad dP=GV^T.
$$

对第 $i$ 行，Softmax 的 Jacobian 是 $\operatorname{diag}(P_i)-P_iP_i^T$。不必显式构造这个矩阵：

$$
dS_{ij}=P_{ij}\left(dP_{ij}-\sum_kP_{ik}dP_{ik}\right).
$$

最后传到 Q 与 K：

$$
dQ=\frac{dSK}{\sqrt{d_h}},\qquad
dK=\frac{dS^TQ}{\sqrt{d_h}}.
$$

被遮住的 $P_{ij}$ 为零，因此对应 $dS_{ij}$ 也为零。对 Q/K/V 投影的参数继续应用上一章的线性层公式。`math_checks.py` 比较这三项与自动求导，并对一个 query 分量做中心差分。

训练代码还包含 attention dropout。它作用在 Softmax 之后，保留元素乘 $1/(1-p)$；反向要使用同一张随机遮罩。上面的推导是关闭 dropout 时的参考，不能直接把丢弃后的概率当成未归一化前的 Softmax Jacobian。

## 为什么训练可以一次算整段

真实文本的所有 token 已经知道，所以每个位置都能构造输入；下三角遮罩确保信息流只依赖前缀。这叫 teacher forcing。并行的是计算，不是允许模型访问未来。

本实验将 PAD 放在尾部，监督只覆盖有效位置。一般实现还要保证每个 query 至少有一个可访问的 key：整行都为 $-\infty$ 会让 Softmax 产生 NaN。

## 检验因果性

在 `model.eval()` 下输入 `[BOS,A,B,C,D]`，然后只把 C、D 改成其他 token。B 及之前的 logits 应保持一致。`smoke_test.py` 自动检查这个性质。

这个测试检查整张网络，包括位置编码、归一化和遮罩，而不仅是 Attention 函数。它能抓到“遮罩存在但方向反了”以及“其他模块读了未来”的错误。

## 从正确实现到高效实现

显式概率矩阵占 $O(BHT^2)$ 空间，长序列很快成为瓶颈。FlashAttention 等方法通过分块、重算和在线 Softmax 避免把完整矩阵写回高带宽内存，仍计算同一个稠密 Attention。稀疏 Attention 改变可访问位置，属于另一类改变。

进阶可对照[算子专题的在线 Attention](../../../ai-infra/operators/fusion-attention/)。先比较输出和梯度，再测量显存与速度；不能只根据 kernel 名字认定更快。理论起点见 [Attention Is All You Need](https://arxiv.org/abs/1706.03762)，高效实现见 [FlashAttention](https://arxiv.org/abs/2205.14135)。
