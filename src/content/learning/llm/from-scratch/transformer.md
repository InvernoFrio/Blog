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

## 跟着一个 batch 走完所有模块

采用默认宽度 64、头数 4、层数 2，给网络一批形状为 $[2,5]$ 的 token ID。查表得到 $[2,5,64]$，位置表取出前 5 行，与两个样本分别相加。位置 0 的向量在样本之间共享，但不同位置的向量不同。

进入第一个 block，LayerNorm 只处理每个位置自己的 64 个特征，shape 不变。Attention 内部临时产生 $[2,4,5,5]$ 的概率矩阵，最终输出回到 $[2,5,64]$，才能与原输入逐元素相加。第二个 LayerNorm 后，MLP 将最后一维从 64 扩到 256，经过 GELU 再投影回 64，接上第二次残差。

第二个 block 重复相同结构，但参数独立。最终 LayerNorm 仍输出 $[2,5,64]$。共享输出头将每个 64 维向量与词表的 259 行 embedding 做内积，输出 $[2,5,259]$。此时最后一轴才是候选 token，前面两轴分别是样本和预测位置。

这条路径可以概括成“离散 ID → 连续表示 → 结合上下文的表示 → 候选分数”。中间 hidden state 不必归一化成概率，MLP 中间维度 256 也没有对应 256 个字节的分类含义；它只是特征通道数量恰好相近。

## 为什么 Attention 后还需要逐位置 MLP

Attention 负责从其他位置读取内容，输出投影负责混合多个头。MLP 则在当前已经聚合的表示上做非线性特征变换。它对每个位置使用同一套参数，但不在这一步直接读取其他位置。

两层线性变换之间若没有非线性，可以合并为一个线性变换，因此扩大到 $4d$ 再缩回去并不会单独带来相同的表达能力。GELU 打破这个线性合并：正值通常保留较多，负值也不是一刀切全部为零。

GELU 中的 $\Phi(x)$ 是标准正态累计分布函数。虽然公式带有概率函数，这个激活本身是确定运算，并没有随机采样“是否通过”。例如 $x=1$ 时约为 0.8413，$x=-1$ 时约为 -0.1587。Dropout 才是在训练中随机丢弃部分激活；二者目的与反向处理不同。

Pre-LN 把归一化放在模块分支入口，残差主路径保持 $x+\cdots$；Post-LN 则在相加之后归一化。改变位置会改变梯度经过的路径，不是同一个函数的两种写法。这里使用 Pre-LN 是基线选择，不能仅凭名称判定所有深度、初始化和任务上的优劣。

## 手算 120,768 个参数来自哪里

“两层、宽度 64”还不能确定精确参数量，需要把 bias、归一化、位置表和共享关系算进去：

| 部分 | 计算 | 参数数 |
| --- | --- | --- |
| token embedding | $259\times64$ | 16,576 |
| 位置 embedding | $64\times64$ | 4,096 |
| 每层 QKV 与输出投影 | $(64\times192+192)+(64\times64+64)$ | 16,640 |
| 每层 MLP | $(64\times256+256)+(256\times64+64)$ | 33,088 |
| 每层两个 LayerNorm | $2\times(64+64)$ | 256 |
| 最终 LayerNorm | $64+64$ | 128 |

合计 $16576+4096+2\times(16640+33088+256)+128=120768$。共享输出头不再增加一份 $259\times64$；dropout 和 GELU 没有可学习参数；因果遮罩也不属于参数。

如果你的统计比这多 16,576，优先检查是否把输出头另算了一份；如果改了词表或位置窗口，embedding 部分也要更新。粗估 $12Ld^2$ 有助于规模预算，但这张表用于检查实际代码。

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

## 怎样理解现代替换，而不只记名字

RoPE 不把位置向量加到输入，而是旋转 query 和 key。二维向量 $(1,0)$ 旋转 $90^\circ$ 后变成 $(0,1)$，长度不变；对 Q、K 施加不同位置对应的旋转，点积会包含它们的旋转角差。上面的相对位置恒等式说明位置因素怎样进入分数，不表示两个不同内容的 token 只要距离相同就有相同分数。

LayerNorm 与 RMSNorm 也不等价。例如输入 $(1,1,1)$，LayerNorm 中心化后为零，输出由 $\beta$ 决定；RMSNorm 不减均值，在尺度为 1、忽略小 $\epsilon$ 时仍输出 $(1,1,1)$。两者对整体平移的性质不同，不能无训练地替换后就期待输出完全一致。

SwiGLU 的 gate 在特征轴逐元素乘另一条投影，不是 Attention 的位置权重。一个决定“当前特征保留多少”，另一个决定“从哪些位置读取多少”。参数公平比较时，忽略 bias，GELU MLP 为 $8d^2$，SwiGLU 三矩阵为 $3dh$；令两者相等得到 $h=8d/3$，这才是中间宽度调整的来源。

**练习。** 如果输入是 $[2,5,64]$，MLP 的第一层输出是 $[2,5,256]$，第二层恢复 $[2,5,64]$。若第二层仍输出 256，残差加法为什么失败？因为残差需要逐元素相加，两条路径最后一维不匹配。增加中间宽度可以，模块出口必须回到主路径宽度。

## 对照代码与做一次消融

阅读 `Config`、`Block` 和 `MiniLM`，画出每次残差相加的两条路径。运行：

```bash
python minillm.py train --layers 1 --out runs/one-layer --steps 200
python minillm.py train --layers 2 --out runs/two-layer --steps 200
```

固定数据、种子、更新次数，比较验证 loss、参数量、已监督 token 与耗时。两层模型更多计算且随机初始化路径不同；单次结果只能作为初步观察。若要归因，应再跑多个种子，并区分等 token 与等计算预算的实验。
