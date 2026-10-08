---
title: 概率模型、交叉熵与困惑度
description: 从序列的联合概率推导 next-token 目标，并手算 Softmax 交叉熵的梯度。
course: llm
topic: from-scratch
module: 数学基础
order: 1
updated: 2026-10-08
prerequisites: [条件概率, 对数, 链式法则]
---

## 把一句话写成概率

设 token 序列为 $x_1,\ldots,x_T$，模型参数为 $\theta$。概率的链式法则给出：

$$
p_\theta(x_{1:T})=\prod_{t=1}^{T}p_\theta(x_t\mid x_{<t}).
$$

这里的条件概率由神经网络计算。给定前缀，网络输出词表大小 $V$ 的分数向量 $z$，再转成概率：

$$
p_i=\frac{\exp z_i}{\sum_{j=1}^{V}\exp z_j}.
$$

同一模型、同一参数用于所有位置。训练时整段输入并行计算；生成时新 token 尚未存在，只能逐步采样。两种执行方式对应同一个条件分布。

## 最大似然为什么成为交叉熵

最大化训练文本的概率，等价于最小化负对数似然。令 $M$ 为所有参与监督的 token 数，目标为：

$$
L=-\frac1M\sum_{n,t:\,m_{n,t}=1}\log p_\theta(x_{n,t}\mid x_{n,<t}).
$$

$n$ 表示样本，$m$ 是监督遮罩。一个真实标签 $y$ 可以写成 one-hot 分布 $q_i=\mathbf1[i=y]$，于是单个位置的损失是 $-\sum_i q_i\log p_i$，即交叉熵。更一般地，$H(q,p)=H(q)+D_{KL}(q\|p)$；固定真实分布时，优化交叉熵等于优化 KL。

实际计算不要先求概率再取对数。用稳定形式：

$$
\ell=\log\sum_i\exp z_i-z_y
=a+\log\sum_i\exp(z_i-a)-z_y,\qquad a=\max_i z_i.
$$

减最大值避免指数溢出，数学结果不变。PyTorch 的 `cross_entropy` 接受原始 logits，内部处理这个问题。

## 手推每个分数的梯度

Softmax 的偏导是 $\partial p_i/\partial z_j=p_i(\mathbf1[i=j]-p_j)$。对 $\ell=-\log p_y$ 使用链式法则：

$$
\frac{\partial\ell}{\partial z_j}=p_j-\mathbf1[j=y].
$$

这条式子解释训练方向。正确标签的梯度非正，梯度下降会提高它的分数；其他分数的梯度非负。对 $M$ 个位置取平均时，每个位置还要乘 $1/M$。

例子：预测三个 token 的概率为 $(0.2,0.5,0.3)$，标签是第一个。损失为 $-\ln0.2\approx1.609$，对 logits 的梯度为 $(-0.8,0.5,0.3)$。分数整体加同一常数不改变概率，因此梯度之和恰为零。

## 困惑度怎样解释

当验证损失使用自然对数、并按有效 token 取平均时：

$$
\mathrm{PPL}=\exp L.
$$

它是正确 token 的逆条件概率的几何平均。均匀预测 $V$ 个 token 时 $L=\ln V$，PPL 为 $V$。本实验词表为 259，因此随机初始化附近的结果应与这个量级相近，但不要求恰好等于它。

不同分词器的 PPL 不能直接比较：一个中文词可能对应一个 token，也可能对应几个字节。跨分词器时可考虑每字节负对数似然或 bits-per-byte：$\mathrm{BPB}=\mathrm{NLL}/(\text{字节数}\cdot\ln2)$，并统一是否计入特殊 token。评估口径先固定，再谈改善。

## 动手验证

运行实验包中的 `python math_checks.py`，第一项比较手写 $p-q$ 与 float64 自动求导。再把例子中的概率改成 $(0.9,0.05,0.05)$：正确标签的梯度绝对值应变小。

检查 `minillm.py` 中的 `loss_sum / valid_tokens`。如果把分母改成 batch size，序列长度变化会改变梯度尺度。这会连带改变裁剪触发频率和学习率的有效作用。

练习：两个样本分别有 2 和 8 个有效 token，平均损失分别为 1 和 3。总体损失应为 $(2\times1+8\times3)/10=2.6$，而不是 2。下一章把这个监督集合落实到数据。
