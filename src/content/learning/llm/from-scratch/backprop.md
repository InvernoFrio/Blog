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

## 反向传播在回答什么问题

前向传播从输入算出预测，再算损失。反向传播问的是：某个参数稍微增大一点，损失会怎样改变？这个局部变化率就是梯度。优化器随后利用梯度调整参数。求梯度与更新参数是两个独立步骤，`.backward()` 不会直接改变模型权重。

为了看清链式法则，先用一个只有两个输入的线性模型：$y=w_1x_1+w_2x_2+b$。临时选择平方损失 $L=\frac12(y-y_*)^2$；这里只是容易手算的教学例子，实际语言模型仍使用交叉熵。

设 $x=(3,4)$、$w=(2,-1)$、$b=0$、目标 $y_*=1$。前向结果为 $y=2\times3-1\times4=2$，损失为 0.5。先计算离损失最近的导数：$\partial L/\partial y=y-y_*=1$。再通过线性函数：

$$
\frac{\partial L}{\partial w_1}=1\times3=3,\quad
\frac{\partial L}{\partial w_2}=1\times4=4,\quad
\frac{\partial L}{\partial b}=1,\quad
\frac{\partial L}{\partial x}=(2,-1).
$$

例如 $w_1$ 增加 $\delta$，预测近似增加 $3\delta$，损失近似增加 $1\times3\delta$。链式法则就是把沿途这些局部影响相乘。一个变量影响损失的路径有多条时，各条路径的贡献还要相加。

取 SGD 学习率 0.01，更新后 $w=(1.97,-1.04)$、$b=-0.01$。同一个输入得到 $y=1.74$，损失为 0.2738。梯度只提供当前位置附近的方向；若学习率太大，更新可能越过较好的位置，损失并不保证下降。

## 上游梯度与局部导数为什么要分开

真实网络由许多层组成。一层通常不知道后面怎样计算 loss，但会收到“本层输出如何影响最终 loss”的梯度，称为上游梯度。它只需把这个梯度乘以自己的局部导数，再传给前一层。

上一章已经得到输出 logits 的梯度 $p-q$。如果 logits 来自 $z=hW+b$，线性层收到的上游梯度就是 $p-q$；它再求出隐藏向量 $h$ 和参数 $W,b$ 的梯度。重复这个过程，就不需要为整个网络从头展开一个巨大表达式。

不要把上游梯度与本层输入混淆。梯度与它对应的变量具有相同形状：$\partial L/\partial z$ 和 $z$ 同形，$\partial L/\partial W$ 和 $W$ 同形。这是检查实现错误的第一条规则。

## 从一个样本推到矩阵形式

把 batch 与时间维展平，令 $X\in\mathbb R^{M\times d}$：$M$ 行是不同 token 位置，$d$ 列是输入特征。权重 $W\in\mathbb R^{d\times h}$、bias $b\in\mathbb R^h$，前向为 $Y=XW+b$，输出形状为 $[M,h]$。bias 对每行广播，并不是为每个位置单独训练一份。

先看一个元素：

$$
Y_{mj}=\sum_{i=1}^{d}X_{mi}W_{ij}+b_j.
$$

令 $G_{mj}=\partial L/\partial Y_{mj}$。一个 $W_{ij}$ 出现在所有行的计算里，因此要把这些路径相加；一个 $X_{mi}$ 则影响同一行的全部输出特征：

$$
\frac{\partial L}{\partial W_{ij}}=\sum_m X_{mi}G_{mj},\qquad
\frac{\partial L}{\partial X_{mi}}=\sum_j G_{mj}W_{ij},\qquad
\frac{\partial L}{\partial b_j}=\sum_m G_{mj}.
$$

这些求和恰好就是矩阵乘法：

$$
\frac{\partial L}{\partial W}=X^TG,\qquad
\frac{\partial L}{\partial X}=GW^T,\qquad
\frac{\partial L}{\partial b}=\sum_mG_{m,:}.
$$

检查形状：$X^T$ 是 $[d,M]$，乘 $[M,h]$ 得到 $[d,h]$，与 $W$ 一致。$G$ 乘 $W^T$ 得到 $[M,d]$，与 $X$ 一致。转置来自被求和的轴，并不是背下来的装饰。

PyTorch 的 `nn.Linear(d,h)` 把权重存为 $[h,d]$，前向写成 `X @ weight.T + bias`。因此它的参数梯度存为 `G.T @ X`。数学上是同一个线性变换，只是存储约定不同。平均 loss 的 $1/M$ 已经包含在 $G$ 中，不应在每一层再除一次。

## Embedding 查表为什么可以训练

Embedding 表 $E\in\mathbb R^{V\times d}$ 的一行对应一个 token。输入 ID 为 $i$ 时，输出 $h=E_i$。整数 ID 只是索引，我们不求它的导数；被取出的表行是浮点参数，可以求导。

等价地，用 one-hot 行向量 $a$ 选择表行：$h=aE$。由线性层反向，$dE=a^Tdh$。因为 $a$ 只有一项为 1，只有被选中的行收到这个查表路径的梯度。

假设 ID 序列为 `[2, 1, 2]`，三个位置的上游梯度分别为 $(1,2)$、$(3,4)$、$(5,6)$。表第 2 行收到 $(1,2)+(5,6)=(6,8)$，第 1 行收到 $(3,4)$，其他行在这条路径上收到零。这就是 scatter-add：按索引把梯度加回去。

注意“同一行重复出现时相加”。模型跨位置共享权重，梯度也必须综合所有使用位置。出现得多不等于一定更新得大，因为不同位置的梯度可能部分抵消，优化器还会改变更新尺度。

## 共享输出头带来的第二条梯度路径

本实验输出 logits 为 $z=hE^T$，输入查表和输出分类共用同一份 $E$。在输出位置，所有词表行都参与计算：$z_i=\langle h,E_i\rangle$。即使某个 token 没在本次输入出现，它作为输出候选项，也可能收到分类梯度。

因此 $E$ 的总梯度是“输入查表贡献”加“输出分类贡献”。共享权重不会丢失其中一条路径，自动求导会相加。实现用 `model.head.weight = model.token_embedding.weight`，让两处引用同一个 Parameter。复制两份相同数值只会得到初始相等的两个参数，下一步就可能不同。

参数量统计只计算这份共享权重一次。优化器也应只注册一次；共享的是参数对象，不是要求我们手工更新两遍。

## 残差为何能提供直接路径

残差结构为 $y=x+f(x)$。输入 $x$ 一方面直接进入加法，另一方面经过 $f$ 再进入加法。上游梯度 $G$ 在加法处送入两条路径，最后合并：

$$
\frac{\partial L}{\partial x}=G+J_f(x)^TG.
$$

$J_f$ 是 $f$ 对输入的 Jacobian，记录每个输出对每个输入的导数。第一项不经过 $f$，即使模块分支的局部梯度很小，这条路径仍然存在。这有助于深层训练，但不保证总梯度永远稳定：第二项仍可能很大，或与第一项抵消。

## LayerNorm 前向：每个 token 内部怎样归一化

隐藏向量的各维数值尺度会随层数和训练变化。LayerNorm 在单个 token 的特征轴上计算均值与方差，再进行可学习的缩放和平移：

$$
\mu=\frac1d\sum_ix_i,\quad v=\frac1d\sum_i(x_i-\mu)^2,\quad
r=(v+\epsilon)^{-1/2},\quad \hat x_i=(x_i-\mu)r,\quad
y_i=\gamma_i\hat x_i+\beta_i.
$$

例如 $x=(1,2,3)$，均值为 2，方差为 $2/3$。忽略仅用于本次数值说明的小 $\epsilon$，归一化结果约为 $(-1.2247,0,1.2247)$。然后 $\gamma,\beta$ 可以重新学习需要的各维尺度与偏移。方差分母为 $d$，不是估计总体方差时常见的 $d-1$。

“每个 token 内部”非常重要。输入形状 $[B,T,d]$ 时，归一化轴是最后一维 $d$。如果把时间轴 $T$ 也纳入统计，当前位置的输出可能随未来 token 改变，因果 Attention 就不能保证整个网络因果。

## LayerNorm 反向：把被省略的链式法则展开

下面保留 $\epsilon$。令 $g_i=\partial L/\partial y_i$，先经过仿射变换得到 $u_i=g_i\gamma_i$。再令 $c_i=x_i-\mu$，于是 $\hat x_i=c_ir$。我们暂时将 $c$ 和 $r$ 看成两条计算路径：

$$
\frac{\partial L}{\partial r}=\sum_i u_ic_i,\qquad
\left.\frac{\partial L}{\partial c_i}\right|_{r\text{固定}}=u_ir.
$$

因为 $r=(v+\epsilon)^{-1/2}$，$\partial r/\partial v=-\frac12r^3$；又因为 $v=\frac1d\sum_i c_i^2$，$\partial v/\partial c_i=2c_i/d$。两条路径合起来，得到对中心化向量的梯度：

$$
a_i=\frac{\partial L}{\partial c_i}
=ru_i-\frac{r^3c_i}{d}\sum_j u_jc_j.
$$

最后 $c_i=x_i-\mu$、$\mu=\frac1d\sum_jx_j$。改变一个 $x_j$ 不只改变自己的 $c_j$，也改变所有项减去的均值。因此：

$$
\frac{\partial L}{\partial x_i}=a_i-\frac1d\sum_j a_j.
$$

利用 $\sum_i c_i=0$，以及 $\hat x_i=rc_i$，化简为：

$$
\frac{\partial L}{\partial x_i}
=r\left(u_i-\overline u-\hat x_i\overline{u\hat x}\right).
$$

横线表示本 token 特征维度的平均。第一项直接传播上游梯度，第二项来自均值依赖，第三项来自方差依赖。归一化把各维耦合在一起，不能只把它当成“除以一个固定标准差”。$\epsilon$ 已保留在 $r$ 中，不要在实现或检查时删掉。

可学习参数的梯度较简单：$d\gamma_i=\sum_{b,t}g_{b,t,i}\hat x_{b,t,i}$，$d\beta_i=\sum_{b,t}g_{b,t,i}$。求和跨 batch 和位置，因为所有 token 共用一份 $\gamma,\beta$。

## 自动求导、梯度累积与计算图

PyTorch 在前向时记录可求导运算，反向按依赖关系应用链式法则。参数通常是叶子张量，默认把梯度保存在 `.grad`；中间张量默认不保留 `.grad`，但这不表示梯度没有经过它。需要观察中间梯度时可用 `.retain_grad()`。

`.backward()` 是累加梯度，不是覆盖。如果两个损失来自两次新前向，分别反向而不清零，参数 `.grad` 就是两个梯度之和。这是梯度累积的基础；忘记清零则会意外累积上一轮更新的梯度。对已经反向并释放的同一计算图再次反向，通常会报错，不能与多次新前向混淆。

`.detach()` 会切断一条梯度路径。记录日志时对 loss 使用 `.item()` 没问题；但如果用 detach 后的隐藏向量计算训练 loss，上游模块可能收不到应有的梯度。反向之前要确认 loss 仍然连接到需要训练的参数。

## 两种检查与有答案的练习

实验 `math_checks.py` 用 float64 比较手写梯度与自动求导。还可以用中心差分，改变一个参数，再分别重新计算损失：

$$
\frac{\partial L}{\partial w_i}\approx
\frac{L(w_i+\delta)-L(w_i-\delta)}{2\delta}.
$$

中心差分独立于反向实现，但有近似误差。$\delta$ 太大偏离局部，太小又可能让两次近似相等的浮点数相减而丢失精度。使用 float64，尝试几个步长，固定输入并关闭 dropout，确认每次计算的是同一个确定函数。

**练习一。** 对两次相同 ID 的 embedding 输出求和，再对所有特征求和。每次输出的上游梯度是全 1，因此对应表行梯度为全 2，其他行是零。若再加入共享分类头，这个结论只描述查表贡献，总梯度还要加分类贡献。

**练习二。** 为什么 bias 梯度要沿样本位置求和？同一个 $b_j$ 加到了每一个 $Y_{mj}$，每一处都提供一条通向 loss 的路径，因此导数相加；广播的反向通常会对被广播的轴求和。

**练习三。** 为什么 LayerNorm 输入梯度各维之和为零？给所有 $x_i$ 加同一个常数，中心化向量和输出都不变，因此沿“全 1”方向的方向导数为零。也能从 $dx_i=a_i-\overline a$ 直接求和验证。这是一个比只检查 shape 更有解释力的性质测试。
