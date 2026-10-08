---
title: 把模块组装成 Decoder Transformer
description: 对齐实验模型的 Pre-LN、残差、GELU、位置编码与共享输出头，再推导 RoPE 和 RMSNorm。
course: llm
topic: from-scratch
module: 数据与模型
order: 5
updated: 2026-10-08
prerequisites: [因果 Attention, LayerNorm]
---

## 从整数序列到 logits

实验网络是：token embedding 加位置 embedding → dropout → 两个 Transformer block → 最终 LayerNorm → 共享输出头。输入形状 $[B,T]$，输出 $[B,T,259]$。

每个 Pre-LN block 计算：

$$
u=x+\operatorname{Attention}(\operatorname{LN}_1(x)),\qquad
y=u+\operatorname{MLP}(\operatorname{LN}_2(u)).
$$

MLP 为 $\operatorname{GELU}(hW_1+b_1)W_2+b_2$，中间宽度为 $4d$。GELU 定义为 $x\Phi(x)$，用输入值平滑控制通过比例；它没有像 ReLU 那样在负半轴全部置零。

Attention 混合不同位置的信息，MLP 对各位置分别变换特征。两个模块共享同一套参数跨位置使用。这里选择 Pre-LN 便于小模型训练；实际深层训练还可能调整残差尺度和初始化。

## MLP 怎样把梯度传回去

将位置展平，写成 $U=XW_1+b_1$、$A=\operatorname{GELU}(U)$、$Y=AW_2+b_2$。设 $G=\partial L/\partial Y$，先经过第二个线性层：

$$
dW_2=A^TG,\quad dA=GW_2^T.
$$

标准正态密度为 $\phi(u)=e^{-u^2/2}/\sqrt{2\pi}$，累计概率为 $\Phi(u)$。因为 $\operatorname{GELU}(u)=u\Phi(u)$，乘积法则给出：

$$
\operatorname{GELU}'(u)=\Phi(u)+u\phi(u),\qquad
dU=dA\odot\left[\Phi(U)+U\odot\phi(U)\right].
$$

再得到 $dW_1=X^TdU$、$dX=dUW_1^T$，两个 bias 梯度沿样本位置求和。实验 `nn.GELU()` 使用这个非 tanh 近似版本，数学检查脚本对导数做独立对照。

对于残差 $Y=X+f(\operatorname{LN}(X))$，输入梯度还有绕过模块的直接项 $G$。另一条路径按 $f$、LayerNorm 的反向依次传播，再相加。把本节、上一章 Attention 和前面的 LayerNorm 公式串起来，就得到一个 block 的完整反向结构。

## 位置编码给了什么信息

没有位置机制时，Attention 对 key/value 的排列缺少显式距离信息。实验用可学习表 $P\in\mathbb R^{T_{\max}\times d}$，输入第 $t$ 个位置为 $E_{x_t}+P_t$。

生成超过窗口长度时，只保留最近 $T_{\max}$ 个输入并把位置重新从零编号。这能运行，但不是原始长上下文推理，也不是 KV cache。训练只见过 64 字节上下文，不能因此宣称支持百万 token。

## RoPE 为什么包含相对位置

对 query/key 的每一对维度做旋转：

$$
R(\phi)=\begin{bmatrix}\cos\phi&-\sin\phi\\\sin\phi&\cos\phi\end{bmatrix},\qquad
q'_m=R(m\omega)q_m,\quad k'_n=R(n\omega)k_n.
$$

由于 $R(a)^TR(b)=R(b-a)$：

$$
\langle q'_m,k'_n\rangle=q_m^TR((n-m)\omega)k_n.
$$

不同维度对使用不同频率，Attention 的分数由此获得相对距离结构。位置缩放、频率修改与长上下文训练还会影响外推；“换成 RoPE 就支持无限长度”并不成立。[RoFormer 原始论文](https://arxiv.org/abs/2104.09864)给出这条路线的原始设计。

实验主线仍用位置 embedding，`math_checks.py` 单独验证旋转的相对位置恒等式。动手升级时应删除旧位置表，在 Q/K 拆头后应用 RoPE，再重新训练与检查因果性。

## RMSNorm 与 SwiGLU 的升级路线

RMSNorm 不减均值，单 token 计算：

$$
y_i=\gamma_i\frac{x_i}{\sqrt{\frac1d\sum_jx_j^2+\epsilon}}.
$$

其梯度仍包含跨维度的归一化耦合。令 $r=(\overline{x^2}+\epsilon)^{-1/2}$、$u=g\odot\gamma$，则 $dx=r u-r^3x\,\overline{ux}$。这与 LayerNorm 的“去均值”项不同。背景见 [RMSNorm 论文](https://arxiv.org/abs/1910.07467)。

SwiGLU 则是 $\operatorname{SiLU}(xW_g)\odot(xW_u)$ 再乘 $W_d$。它有三个矩阵；若要与两个矩阵的 $4d$ GELU MLP 比较参数量，中间维度应约为 $8d/3$，再按硬件友好的倍数取整。不能保持同样中间宽度却宣称是等参数对比。

## 对照代码与做一次消融

阅读 `Config`、`Block` 和 `MiniLM`，画出每次残差相加的两条路径。运行：

```bash
python minillm.py train --layers 1 --out runs/one-layer --steps 200
python minillm.py train --layers 2 --out runs/two-layer --steps 200
```

固定数据、种子、更新次数，比较验证 loss、参数量、已监督 token 与耗时。两层模型更多计算且随机初始化路径不同；单次结果只能作为初步观察。若要归因，应再跑多个种子，并区分等 token 与等计算预算的实验。
