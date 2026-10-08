---
title: 从偏好学习到 RLVR：DPO、PPO 与 GRPO
description: 推导偏好损失与策略梯度，理解可验证奖励、重要性采样、长度偏差和实际 rollout 系统。
course: llm
topic: from-scratch
module: 后训练
order: 13
updated: 2026-10-08
prerequisites: [条件概率, 反向传播, SFT]
---

## 三种数据提供三种信号

SFT 提供示范回答；偏好数据提供同一问题下两个回答的排序；强化学习让当前策略生成回答并接收奖励。三者可以组合，但不能把一个固定的高分答案数据集称为在线 RL。

设策略 $\pi_\theta(y\mid x)$ 是完整回答的概率，等于所有回答 token 条件概率的乘积。因此序列 log-prob 是 token log-prob 的和，需排除 prompt、padding，并明确是否含 EOS。

## DPO 的公式和梯度方向

对偏好回答 $y^+$、非偏好回答 $y^-$，以及冻结参考策略 $\pi_{\mathrm{ref}}$，定义：

$$
\Delta=\left[\log\pi_\theta(y^+\mid x)-\log\pi_\theta(y^-\mid x)\right]
-\left[\log\pi_{\mathrm{ref}}(y^+\mid x)-\log\pi_{\mathrm{ref}}(y^-\mid x)\right],
$$

$$
L_{\mathrm{DPO}}=-\log\sigma(\beta\Delta).
$$

参考模型冻结，不反传。若 $a=\log\pi_\theta(y^+\mid x)$、$b=\log\pi_\theta(y^-\mid x)$，则：

$$
\frac{\partial L}{\partial a}=-\beta\sigma(-\beta\Delta),\quad
\frac{\partial L}{\partial b}=\beta\sigma(-\beta\Delta).
$$

这推动偏好回答相对概率提高。`math_checks.py` 检查这两个方向。参考 log-prob 防止只追求两者绝对差值；$\beta$ 的作用应结合 [DPO 原始目标与推导](https://arxiv.org/abs/2305.18290)理解，而不是与学习率混同。

序列概率连乘会下溢，必须相加 log-prob；随意改成每 token 平均就改变目标，尤其在回答长度不同时。DPO 可以不做在线 rollout，但仍需要高质量偏好、参考模型和独立评估。

## 完整回答的 log-prob 怎样从模型输出取得

训练仍使用 teacher forcing：给定提示和真实回答，模型输出每个位置的 logits。先对词表轴做 `log_softmax`，再按移位标签取出真实下一项的 log-prob。最后只对回答区域求和，得到每条回答的 $\log\pi_\theta(y\mid x)$。

```python
import torch.nn.functional as F

def response_logprob(logits, labels):
    valid = labels != -100
    safe_labels = labels.masked_fill(~valid, 0)
    token_logp = F.log_softmax(logits, dim=-1).gather(
        -1, safe_labels.unsqueeze(-1)
    ).squeeze(-1)
    return token_logp.masked_fill(~valid, 0.0).sum(dim=-1)
```

形状是 logits `[B,T,V]`、labels `[B,T]`、结果 `[B]`。`-100` 不能拿来作为 gather 索引，所以先用合法索引占位，再把无监督项置零。这个函数要求标签已经正确移位，并且提示区域已设为 `-100`，它本身不会替你处理这些边界。

如果回答两个 token 的条件概率为 0.5、0.25，序列概率为 0.125，log-prob 为 $\ln0.5+\ln0.25=\ln0.125\approx-2.0794$。换成 token 平均得到 -1.0397，已经不是该完整序列的 log 概率。长度不同的回答上，两种做法会改变偏好比较。

冻结参考模型时不仅不要注册其参数到 optimizer，还要阻止参考前向建立可反传的训练路径；通常用 eval 模式和 no-grad。策略模型的 log-prob 必须保留计算图，否则 DPO loss 无法更新它。

## DPO 的参考项怎样改变比较基准

取当前策略 chosen/rejected log-prob 为 $(-2,-3)$，参考策略为 $(-2.5,-2.7)$。当前相对差为 1，参考相对差为 0.2，$\Delta=0.8$。若 $\beta=0.2$，损失为 $-\log\sigma(0.16)\approx0.6163$，chosen 梯度约 -0.0920，rejected 梯度约 0.0920。

如果当前策略与参考策略完全相同，$\Delta=0$，损失是 $\ln2$，无论参考本来更喜欢哪个回答。这时学习信号要求策略相对参考进一步倾向数据标注的 chosen。反过来，单看当前 chosen 的概率高于 rejected，并不保证 margin 已经大，因为参考也可能更强地偏向 chosen。

这里没有一个逐样本硬性限制，把当前分布锁在参考分布附近。原始推导从带 KL 正则的奖励目标出发；在固定奖励尺度下，$\beta$ 对应这种权衡，但在写出的偏好损失中它同时缩放 margin 和梯度。不能简单说“把 beta 加倍等于把学习率加倍”，也不能把参考项当成绝对不会偏离的保证。

## 奖励目标的梯度怎么来

暂时只有奖励 $R(y)$，目标 $J(\theta)=\mathbb E_{y\sim\pi_\theta}[R(y)]$。利用 $\nabla\pi=\pi\nabla\log\pi$：

$$
\nabla J=\mathbb E\left[R(y)\nabla\log\pi_\theta(y\mid x)\right].
$$

减去与动作无关的 baseline $b(x)$ 不改变期望，因为 $\mathbb E[\nabla\log\pi]=0$；它可减小方差。实现中 reward 与 baseline 通常作为常数处理，不通过采样 token ID 直接求导。

token 级梯度可展开为 $\sum_t\nabla\log\pi_\theta(y_t\mid x,y_{<t})$。一个最终成功奖励会影响整条轨迹，信用分配困难，这就是长程 agent 训练的一个核心问题。

实践先运行 `python rl_bandit.py`：它对四个离散动作训练 Softmax 策略，用已知奖励比较精确期望梯度与采样策略梯度，再观察策略概率变化。它隔离了 RL 数学，**没有运行 LLM 的 PPO 或 GRPO 训练**。

## 策略梯度如何让采样概率发生变化

离散动作可以先枚举求期望：$J=\sum_i p_iR_i$。对 logits 用 Softmax 导数，有：

$$
\frac{\partial J}{\partial z_j}
=\sum_iR_ip_i(\mathbf1[i=j]-p_j)
=p_j(R_j-\mathbb E[R]).
$$

奖励高于平均值的动作得到正的目标梯度；因为现在最大化 $J$，梯度上升会提高它的分数。实现若使用最小化 loss，通常取 $-R\log p$，不要忘记这个符号变化。

四动作奖励为 $(0,0,0,1)$，均匀概率 0.25 时，平均奖励为 0.25，梯度是 $(-0.0625,-0.0625,-0.0625,0.1875)$。一个抽样可能选中零奖励动作，未经 baseline 的这次估计就全为零；另一次选中奖励为 1 的动作，估计明显非零。单次估计不等于精确梯度，平均足够多样本才接近期望。

baseline 为何不改变期望？对于固定问题且不依赖所选动作的 $b$，$\sum_i b\nabla p_i=b\nabla\sum_i p_i=b\nabla1=0$。因此可以减去它帮助降低方差。但若 baseline 任意依赖当前动作，就不能直接用这条证明。

语言模型把动作扩成一串 token。完整回答得到成功奖励后，序列 log-prob 的梯度是每个生成 token 的 log-prob 梯度之和。奖励告诉我们整段成功了，却没有明确指出哪一步最关键，这就是信用分配问题。验证器的 0/1 结果本身不需要可微；我们对“产生这条结果的概率”求导。

## PPO 为什么保存旧策略概率

回答由旧策略 $\pi_{\mathrm{old}}$ 生成，更新时使用比率 $r_t=\pi_\theta(y_t\mid s_t)/\pi_{\mathrm{old}}(y_t\mid s_t)$。一种常见 surrogate 为：

$$
J_{\mathrm{clip}}=\mathbb E\left[\min\left(r_tA_t,
\operatorname{clip}(r_t,1-\epsilon,1+\epsilon)A_t\right)\right].
$$

它限制某些方向的过大改动，但不是严格的 KL 约束。实际算法还可包含 value model、优势估计与 KL 惩罚。详见 [PPO 论文](https://arxiv.org/abs/1707.06347)。代码中的 token、序列归一化选择同样会影响训练。

## GRPO 和可验证奖励

对同一问题生成一组回答，用组内奖励构造相对优势，例如 $A_i=(R_i-\overline R)/(\operatorname{std}(R)+\epsilon)$，再配合策略比率与约束优化。不同 GRPO 变体的归一化、KL 和长度处理不同，不应把一个公式说成所有企业的统一配方。

若整组奖励相同，这个优势为零，通常没有该组的奖励梯度。题目过难或过易，都可能降低有效信号；题目采样与奖励设计因此属于训练算法的一部分。参考 [DeepSeekMath](https://arxiv.org/abs/2402.03300)。

RLVR 的奖励来自可验证结果，例如数学答案、程序单元测试或环境状态。验证器不完善时，模型可能学会通过检查却未完成任务。设计时应使用隐藏测试、覆盖边界和独立评估，区分执行成功、答案正确与推理过程正确。

## PPO 的 clipping 和 GRPO 的组内优势，分别手算

先看 PPO。设 $\epsilon=0.2$、优势 $A=2$、新旧概率比 $r=1.5$。未裁剪项为 3，裁剪项为 $1.2\times2=2.4$，取较小值 2.4。在这一区域继续增加概率不会继续提高这个 surrogate，避免同一批样本推动过大的有利方向更新。

负优势不能照搬正优势直觉。若 $A=-2,r=0.5$，未裁剪项为 -1，裁剪项为 $0.8\times(-2)=-1.6$，最小值为 -1.6，继续降低这个坏动作的概率不再获得更大 surrogate。若概率朝相反的不利方向变化，目标仍可能施加惩罚。因此 clipping 不等于把所有更新强行限制在一个固定区间。

旧策略与参考策略也要分开。旧策略是实际生成这批 rollout 的行为策略，用于概率比；参考策略通常是约束漂移的固定基准。二者可能起初相同，之后职责和更新时间不同。

再看 GRPO。一个问题生成四条回答，奖励为 $(0,0,1,1)$。均值为 0.5，使用总体标准差、分母为组大小 4，标准差为 0.5。忽略很小的稳定项，优势为 $(-1,-1,1,1)$，只表达同组相对高低。

若奖励为 $(0,0,0,0)$，每条优势都是零，奖励部分没有更新信号；若全为 1，也同样没有组内比较信号。若算法另含 KL 正则，它仍可能产生梯度，所以不能说整个训练目标一定为零。是否使用总体或样本标准差、是否按回答长度归一化，也需和具体实现对应。

题目全部太难，生成都失败；题目全部太易，生成都成功；两种情况都可能缺少有用比较。因此要同时观察任务成功率、组内奖励差异和有效训练样本数，而不是只看奖励均值上升。

## 为什么企业还需要复杂的 rollout 系统

长回答和工具任务完成时间不一致，同步批次会等待最慢样本；异步生成提高利用率，却引入旧策略样本和先完成短回答的偏差。训练必须记录样本来自哪个策略版本、旧 log-prob、奖励与有效 token，并处理过期轨迹。

KL、奖励归一化、长度惩罚、任务混合与采样策略都会共同影响结果。下一章把这些概念映射到具体公开报告，避免从一个排行榜猜测训练配方。

练习：四个动作奖励为 $(0,0,0,1)$，均匀策略的期望奖励为 0.25。精确目标对 logits 的梯度应为 $(-0.0625,-0.0625,-0.0625,0.1875)$。用实验脚本验证，并解释为何单次采样梯度不必等于它。
