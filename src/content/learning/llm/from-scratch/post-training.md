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

## 奖励目标的梯度怎么来

暂时只有奖励 $R(y)$，目标 $J(\theta)=\mathbb E_{y\sim\pi_\theta}[R(y)]$。利用 $\nabla\pi=\pi\nabla\log\pi$：

$$
\nabla J=\mathbb E\left[R(y)\nabla\log\pi_\theta(y\mid x)\right].
$$

减去与动作无关的 baseline $b(x)$ 不改变期望，因为 $\mathbb E[\nabla\log\pi]=0$；它可减小方差。实现中 reward 与 baseline 通常作为常数处理，不通过采样 token ID 直接求导。

token 级梯度可展开为 $\sum_t\nabla\log\pi_\theta(y_t\mid x,y_{<t})$。一个最终成功奖励会影响整条轨迹，信用分配困难，这就是长程 agent 训练的一个核心问题。

实践先运行 `python rl_bandit.py`：它对四个离散动作训练 Softmax 策略，用已知奖励比较精确期望梯度与采样策略梯度，再观察策略概率变化。它隔离了 RL 数学，**没有运行 LLM 的 PPO 或 GRPO 训练**。

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

## 为什么企业还需要复杂的 rollout 系统

长回答和工具任务完成时间不一致，同步批次会等待最慢样本；异步生成提高利用率，却引入旧策略样本和先完成短回答的偏差。训练必须记录样本来自哪个策略版本、旧 log-prob、奖励与有效 token，并处理过期轨迹。

KL、奖励归一化、长度惩罚、任务混合与采样策略都会共同影响结果。下一章把这些概念映射到具体公开报告，避免从一个排行榜猜测训练配方。

练习：四个动作奖励为 $(0,0,0,1)$，均匀策略的期望奖励为 0.25。精确目标对 logits 的梯度应为 $(-0.0625,-0.0625,-0.0625,0.1875)$。用实验脚本验证，并解释为何单次采样梯度不必等于它。
