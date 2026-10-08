---
title: 线性层、Embedding 与反向传播
description: 用形状和链式法则推导矩阵梯度、共享权重与 LayerNorm 的反向计算。
course: llm
topic: from-scratch
module: 数学基础
order: 2
updated: 2026-10-08
prerequisites: [矩阵乘法, 交叉熵梯度]
---

## 一层线性变换的梯度

把 batch 和时间维展平，$X\in\mathbb R^{M\times d}$，$W\in\mathbb R^{d\times h}$，$b\in\mathbb R^h$。前向为 $Y=XW+b$，上游梯度 $G=\partial L/\partial Y$ 与 $Y$ 同形。

微分 $dY=dXW+XdW+db$，用 $dL=\mathrm{tr}(G^TdY)$ 收集各项，得到：

$$
\frac{\partial L}{\partial X}=GW^T,\quad
\frac{\partial L}{\partial W}=X^TG,\quad
\frac{\partial L}{\partial b}=\sum_{m=1}^{M}G_{m,:}.
$$

检查形状比背公式可靠。PyTorch `nn.Linear(d,h)` 存储的权重形状是 $[h,d]$，代码前向相当于 `X @ weight.T + bias`；因此参数梯度的存储形状也转置了。

## 查表层仍然能训练

Embedding 表 $E\in\mathbb R^{V\times d}$，输入 token ID 为 $i$ 时输出 $E_i$。整数 ID 不求导，表的每一行可求导。同一 ID 出现多次，其梯度相加：

$$
\frac{\partial L}{\partial E_i}=\sum_{m:\,x_m=i}\frac{\partial L}{\partial h_m}.
$$

这就是反向的 scatter-add。出现次数多的 token 收到更多梯度，并不意味着它自动更重要；数据频率、目标平均方式与优化器都会影响更新。

本实验输出 logits 为 $z=hE^T$，输入与输出共享同一个 $E$。因此 $E$ 的总梯度由查表路径和分类路径相加。实现用 `model.head.weight = model.token_embedding.weight`，让两处引用同一个 Parameter，而不是复制数值。

检查参数量时也只统计一次共享参数。若在优化器里手工把这个 Parameter 注册两次，会破坏更新语义。

## 残差把梯度怎样传回去

若 $y=x+f(x)$，则：

$$
\frac{\partial L}{\partial x}=G+J_f(x)^TG.
$$

第一项是一条不经过 $f$ 的直接梯度路径。它有利于深层网络传播信号，但不能保证所有层梯度稳定，也不能替代正确的初始化、归一化和学习率。

## LayerNorm 的完整反向

对单个 token 的 $d$ 维向量，定义：

$$
\mu=\frac1d\sum_i x_i,\quad
v=\frac1d\sum_i(x_i-\mu)^2,\quad
r=(v+\epsilon)^{-1/2},\quad
\hat x_i=(x_i-\mu)r,\quad y_i=\gamma_i\hat x_i+\beta_i.
$$

这里的方差分母为 $d$，不是统计学无偏估计的 $d-1$。设 $g_i=\partial L/\partial y_i$，$u_i=g_i\gamma_i$，平均符号表示对本 token 的维度取平均。链式法则展开并合并后：

$$
\frac{\partial L}{\partial x_i}
=r\left(u_i-\overline u-\hat x_i\overline{u\hat x}\right).
$$

对所有 batch、时间位置求和，得到 $\partial L/\partial\gamma_i=\sum g_i\hat x_i$ 与 $\partial L/\partial\beta_i=\sum g_i$。$\epsilon$ 已包含在 $r$ 和 $\hat x$ 中；验证时不要为了简化推导把它删掉。

LayerNorm 在每个 token 内归一化，不读取未来位置。若误把时间维也放进归一化轴，网络即使使用因果 Attention，仍可能从归一化统计量泄漏未来。

## 两种独立的数值检查

`math_checks.py` 用 float64 比较手写公式与自动求导。另一种检查是中心差分：

$$
\frac{\partial L}{\partial w_i}\approx\frac{L(w_i+\delta)-L(w_i-\delta)}{2\delta}.
$$

它不依赖自动求导路径，但有截断误差和浮点抵消误差，$\delta$ 不能无限缩小。检查时固定随机输入、关闭 dropout，避免两次前向其实不是同一个函数。

练习：把同一 token 输入两次，并对两次 embedding 输出求和。对应表行的梯度应为全 2，其他行全 0。然后打开共享输出头，观察其他行也可能获得梯度，并说明额外路径来自哪里。
