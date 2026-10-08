---
title: 数学工具箱：读公式与查数值误差
description: 补习张量轴、矩阵微分、条件概率、期望梯度与浮点误差，连接教材中的常用记号。
course: llm
topic: from-scratch
module: 数学附录
order: 17
updated: 2026-10-08
prerequisites: [高中代数, Python]
---

## 本专题的统一记号

| 符号 | 含义 | 典型代码 |
| --- | --- | --- |
| $B,T,d$ | batch、序列长度、隐层宽度 | `batch, length, width` |
| $V,H,d_h$ | 词表、头数、每头宽度 | `VOCAB, heads, width // heads` |
| $L$ | 上下文决定是 loss 或层数 | `loss` 或 `config.layers` |
| $M$ | 有效监督 token 数；Attention 中也用作遮罩矩阵 | `valid_tokens` 或 `causal` |
| $\theta,\eta$ | 所有参数、学习率 | `model.parameters(), learning_rate` |
| $\odot$ | 逐元素乘法 | `a * b` |
| $\mathbb E$ | 对指定分布取期望 | 样本平均或枚举加权和 |

同一符号在不同论文中常有不同含义，读公式前先找定义。本教材在每章局部重新定义容易混淆的符号。

## 把张量理解为带轴的数组

一个 $[B,T,d]$ 张量有三条轴：样本、位置、特征。对 `dim=-1` 求和移除特征轴；使用 `keepdim=True` 保留长度 1 的轴，方便广播。

广播是按尾部维度对齐，维度相等或其中一个为 1 才能兼容。把 $[d]$ 的 bias 加到 $[B,T,d]$，等价于每个样本位置使用同一个向量。广播的反向会把重复使用处的梯度相加。

转置只改轴对应关系，通常不复制数据。`view` 重解释形状要求存储满足相应连续性；`reshape` 必要时可能复制。数学上的同一个张量布局，在硬件上可能有不同访问成本，详见[张量布局专题](../../../ai-infra/operators/tensor-layout/)。

## 链式法则不是逐元素乘到底

标量复合函数 $L=f(g(x))$ 的导数为 $f'(g(x))g'(x)$。向量输入输出时，导数是 Jacobian；反向传播计算上游梯度与 Jacobian 的乘积，而不是先存下完整矩阵。

若 $y=f(x)$ 且上游梯度为 $g_y$，列向量约定下 $g_x=J_f^Tg_y$。Softmax 的行内耦合、LayerNorm 的维度耦合都说明一个输出可能依赖多个输入。

## 用微分推导矩阵梯度

Frobenius 内积定义为 $\langle A,B\rangle=\sum_{ij}A_{ij}B_{ij}=\mathrm{tr}(A^TB)$。标量损失的微分写成 $dL=\langle\nabla_XL,dX\rangle$。

对于 $Y=XW$：

$$
dL=\mathrm{tr}(G^TdXW)+\mathrm{tr}(G^TXdW)
=\mathrm{tr}((GW^T)^TdX)+\mathrm{tr}((X^TG)^TdW).
$$

比较系数即得到 $\nabla_XL=GW^T$、$\nabla_WL=X^TG$。使用 trace 的循环性质时矩阵尺寸必须兼容；不能把一般矩阵乘法交换顺序。

## 条件概率与 teacher forcing

条件概率是 $p(a\mid b)=p(a,b)/p(b)$，不是 $p(b\mid a)$。自回归分解使用所有先前 token 作为条件。训练时已知完整答案，只能用它构造前缀与监督目标，不能把要预测的 token 当作输入信息提供给同一位置。

“独立同分布样本”常是分析假设，真实网页语料会重复、相关且随时间变化。固定验证集能控制一部分比较条件，但不能保证部署分布与验证分布相同。

## 期望、采样与策略梯度

离散变量的期望为 $\mathbb E[f(X)]=\sum_xp(x)f(x)$。从该分布独立采样并取平均，是它的 Monte Carlo 估计。估计值有波动，不能从一次运行宣布固定改善。

如果分布本身依赖参数：

$$
\nabla_\theta\mathbb E_{x\sim p_\theta}[f(x)]
=\sum_xf(x)\nabla p_\theta(x)
=\mathbb E[f(x)\nabla\log p_\theta(x)].
$$

这是策略梯度用到的 score-function 恒等式。若 $f$ 也显式依赖 $\theta$，还需加上 $\mathbb E[\nabla_\theta f]$；不能在所有目标上无条件省略这一项。对 RL 奖励，通常将验证器结果作为常数。

## 浮点计算的三个误差来源

有限范围会溢出，例如直接计算大 logits 的指数；有限精度会舍入；相近数相减会导致有效数字损失。浮点加法一般不满足严格结合律，所以改变梯度聚合顺序可能产生小差异。

float64 的推导检查有助于隔离公式错误，但生产训练还要测试实际精度。输出比较通常使用 $|a-b|\le\mathrm{atol}+\mathrm{rtol}|b|$；接近零时绝对误差尤其重要。

中心差分的截断误差通常随 $\delta^2$ 缩小，而抵消误差大致随机器误差除以 $\delta$ 增大。找稳定区间比无限缩小 $\delta$ 更合理。

## 一条公式的阅读步骤

先列输入输出的 shape，确认求和轴和归一化分母，再说明概率来自哪个分布、参数哪些被冻结、梯度经过哪些路径。最后检查极端情况：全遮罩、零有效 token、相同奖励、超长序列和异常值。

回到[导览](../overview/)时，可用这套步骤重新读每章。若一条公式无法落到具体数组和边界行为，它还没有真正进入可运行的训练系统。
