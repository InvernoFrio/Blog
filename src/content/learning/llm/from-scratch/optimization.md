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

## microbatch、累积轮数与 step 怎样对应

设一次只能装入 2 条样本，希望每次更新使用 8 条样本，可以连续计算 4 个 microbatch，再做一次 optimizer step。前四次 `.backward()` 改变的是 `.grad`，参数一直不动；最后 `.step()` 才改变参数。

这相当于把一个较大的批次拆开计算，但要保持同一个损失分母。若每个小批都先更新参数，后面小批使用了新参数，就不再是同一个大批次梯度，而是四次不同位置上的更新。

累积降低的是同时保留的激活开销，不会自动降低参数、梯度和 Adam 状态的存储。在有 dropout 时，各个小批还可能使用不同随机遮罩，因此与一次大批前向不必位级一致。比较时应先用确定性设置检查数学，再讨论随机层和硬件差异。

`step` 是参数更新计数，`microbatch` 是前向与反向计算单位，`epoch` 是遍历一遍数据的概念。本脚本有放回随机抽样窗口，通常没有严格的整遍历 epoch 边界。训练进度因此同时记录 step 与累计有效目标 token。

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

## Adam 的指数平均与偏差修正，具体算一次

将一阶矩递推展开，得到 $m_t=(1-\beta_1)\sum_{k=1}^{t}\beta_1^{t-k}g_k$。最近梯度权重大，较早梯度的影响指数衰减。若所有历史梯度恰好都是同一个 $g$，几何级数给出 $m_t=(1-\beta_1^t)g$；早期系数小于 1，就是零初始化带来的低估。

除以 $1-\beta_1^t$ 后恢复 $g$。二阶矩同理。这解释了偏差修正的来源，但不表示实际变化梯度下修正后的状态等于当前梯度。$v_t$ 是平方梯度的指数平均，即二阶原点矩，不是减过均值平方的统计方差。

取单个参数 $\theta_0=2$、梯度 $g_1=3$、$\beta_1=0.9,\beta_2=0.95$。从零状态得到 $m_1=0.3$、$v_1=0.45$。偏差修正后 $\hat m_1=3$、$\hat v_1=9$。暂时忽略小 $\epsilon$，自适应项为 $3/\sqrt9=1$。

若学习率为 0.01、权重衰减为 0.1，AdamW 更新给出 $\theta_1=(1-0.001)\times2-0.01=1.988$。其中 0.002 来自权重衰减，0.01 来自自适应梯度步。负梯度的第一步方向相反，但以后还会受积累状态影响。

这也解释了为何仅保存参数无法精确续训：两个模型当前 $\theta$ 相同，但历史 $m,v$ 不同，下一步就可能不同。Adam 的历史不是日志附属品，而是算法状态。

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

## 把学习率、裁剪和更新串成一个可检查的顺序

梯度向量为 $(3,4)$ 时，二范数为 5。阈值为 1 的全局裁剪将其缩放为 $(0.6,0.8)$，方向保持不变；逐元素夹到区间 $[-1,1]$ 则变为 $(1,1)$，方向改变。这里的梯度可以理解成把所有参数梯度展平成一条长向量。

对 AdamW 而言，裁剪约束输入 optimizer 的梯度范数，并不直接保证最终参数位移小于某个值，因为自适应状态与权重衰减仍参与更新。日志应保留裁剪前范数，才能判断多少 step 实际触发了裁剪。

默认基础学习率 0.003，前 20 步线性 warmup，因此第 1 步为 0.00015，第 20 步达到 0.003，第 1000 步为 0.0003。余弦衰减终点固定为 1000，训练只做到 200 步时还没有走完整条曲线。把 `--steps` 从 100 改为 200 是延长运行，不应重新规划早期调度。

下面是结构示意，`microbatches` 要在反向之前准备好，`valid_total` 是这轮所有有效目标数：

```python
optimizer.zero_grad(set_to_none=True)
valid_total = sum(int((y != -100).sum()) for x, y in microbatches)
assert valid_total > 0
for x, y in microbatches:
    logits = model(x)
    loss_sum = F.cross_entropy(
        logits.flatten(0, 1), y.flatten(),
        ignore_index=-100, reduction="sum",
    )
    (loss_sum / valid_total).backward()
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
for group in optimizer.param_groups:
    group["lr"] = learning_rate_for_this_step
optimizer.step()
```

这里省略数据采样、非有限值检查和精度处理，只展示依赖顺序。特别留意：清零在整轮之前，裁剪在所有反向之后，step 只有一次。源码是完整执行版本，不要把示意片段直接当成另一个完整训练器。

**练习。** 先训练第一批并 `.step()`，再训练第二批，与先累积两批再 `.step()` 为什么不同？前者第二批的梯度在已经更新的参数处计算，而且 Adam 状态更新两次；后者两批共用同一参数点和一个 step。即使有效 token 相同，它们也是不同算法。

## 排查训练异常

loss 非有限时先检查标签、空监督批次和全遮罩行，再检查学习率与精度。验证变差而训练变好时，检查过拟合与数据分布；两者都几乎不变时，检查参数是否进入优化器、梯度是否清错、有效目标是否存在。

练习：在数学检查脚本中把正确的 token 加权改成批均值平均，确认测试失败。再解释为什么这个错误在全长文本上可能隐藏，在 SFT 的变长回答上更容易暴露。
