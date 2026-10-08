---
title: 训练循环、梯度累积与 AdamW
description: 推导自适应更新，按有效 token 累积梯度，并把调度、数值精度和裁剪放回完整训练顺序。
course: llm
topic: from-scratch
module: 训练实践
order: 6
updated: 2026-10-08
prerequisites: [反向传播, 有效 token 遮罩]
---

## 一次参数更新的顺序

一个 optimizer step 先清梯度，采样一个或多个 microbatch，前向并计算损失，反向累积，检查数值，再裁剪梯度、设置本步学习率、更新参数。验证和保存检查点发生在完整更新结束后。

PyTorch 的 `.backward()` 将梯度加到 `.grad`，不会自动清空。`optimizer.zero_grad(set_to_none=True)` 放在一轮累积之前；如果每个 microbatch 都清空，前面的梯度就丢了。

## 有效 token 数才是正确分母

第 $k$ 个 microbatch 的损失和为 $S_k$，有效目标数为 $M_k$。一轮累积应优化：

$$
L=\frac{\sum_{k=1}^{K}S_k}{\sum_{k=1}^{K}M_k},\quad
\nabla L=\sum_k\frac{\nabla S_k}{M_{\mathrm{total}}}.
$$

因此实验先采样所有 microbatch 并计算 $M_{\mathrm{total}}$，再分别对 `loss_sum / valid_tokens` 反向。常见的 `mean_loss / accum_steps` 只有在各 microbatch 的有效 token 数相同时才等价。

数值例子：第一批 1 个 token、损失为 4；第二批 9 个 token、平均损失为 1。总体平均为 1.3，批平均的平均却为 2.5。这不只是日志不同，梯度方向和幅度也可能不同。数学检查脚本构造了不等 token 数的反例。

比较大 batch 与梯度累积时要关闭 dropout，或明确随机遮罩差异。即使目标归一化完全相同，随机层与浮点加法顺序也可能导致逐元素不相等。

## Adam 的状态从哪里来

令 $g_t$ 为第 $t$ 步梯度，逐元素计算：

$$
m_t=\beta_1m_{t-1}+(1-\beta_1)g_t,\qquad
v_t=\beta_2v_{t-1}+(1-\beta_2)g_t^2.
$$

若从零状态开始，在早期这两个指数平均被低估。用 $\hat m_t=m_t/(1-\beta_1^t)$、$\hat v_t=v_t/(1-\beta_2^t)$ 修正，再按 $\hat m_t/(\sqrt{\hat v_t}+\epsilon)$ 更新。$v$ 为每个参数提供不同尺度，不能把 Adam 理解为固定学习率的普通 SGD。

AdamW 把权重衰减从自适应梯度中分离：

$$
\theta_t=(1-\eta_t\lambda)\theta_{t-1}
-\eta_t\frac{\hat m_t}{\sqrt{\hat v_t}+\epsilon}.
$$

若把 $\lambda\theta$ 加进梯度，它会进入 $m$、$v$，通常不等价于这条式子。依据见 [Decoupled Weight Decay Regularization](https://arxiv.org/abs/1711.05101)。

实验对二维及以上权重使用衰减，对 bias 和归一化尺度不衰减；这是一项实验约定，并非数学上唯一正确的分组。默认 $\beta_1=0.9,\beta_2=0.95$，`math_checks.py` 检查第一步更新。

## 裁剪怎样改变梯度

全局梯度范数 $\|g\|_2$ 超过阈值 $c$ 时，使用：

$$
g'=g\min\left(1,\frac c{\|g\|_2+\epsilon}\right).
$$

所有参数合成一个范数，统一缩放；这不同于逐元素截断。日志中的 `grad_norm_before_clip` 是裁剪前的值。应在全部 microbatch 反向之后裁剪，否则累积结果一般不同。

FP16 配合 loss scaling 时，裁剪之前要先 unscale；本实验只支持 CPU FP32 和可用 CUDA 上的 BF16 autocast，不实现 FP16 GradScaler。logits 的交叉熵计算转回 FP32，模型参数仍是 FP32。BF16 能减少部分计算与激活开销，但不等于整个模型状态减半。

## 学习率与总训练步数

实验前 20 步线性 warmup，之后余弦衰减，至第 1000 步到基础学习率的 10%。若 $u=(t-w)/(D-w)$ 并截到 $[0,1]$，则：

$$
\eta_t=\eta_{\max}\left[0.1+0.9\frac{1+\cos(\pi u)}2\right].
$$

`--steps` 是本次要达到的总步数；`--decay-steps` 是调度的固定终点，两者分开，才能先停在 100 步、再继续到 200 步而不改变前 100 步的更新轨迹。续训不能悄悄更换学习率调度。

## 排查训练异常

loss 非有限时先检查标签、空监督批次和全遮罩行，再检查学习率与精度。验证变差而训练变好时，检查过拟合与数据分布；两者都几乎不变时，检查参数是否进入优化器、梯度是否清错、有效目标是否存在。

练习：在数学检查脚本中把正确的 token 加权改成批均值平均，确认测试失败。再解释为什么这个错误在全长文本上可能隐藏，在 SFT 的变长回答上更容易暴露。
