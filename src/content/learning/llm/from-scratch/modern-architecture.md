---
title: 现代 LLM 的结构与数值升级
description: 解释 GQA、MoE 路由、低精度训练与多 token 目标，区分总参数、激活参数和真实系统成本。
course: llm
topic: from-scratch
module: 规模与系统
order: 11
updated: 2026-10-08
prerequisites: [Transformer, 分布式训练]
---

## 升级要先指出瓶颈

小模型的实现便于检查，但不是大规模系统的最优实现。升级通常针对特定成本：GQA 减少 KV cache，MoE 增加可用参数而限制每 token 的专家计算，低精度降低传输和矩阵运算成本，融合算子减少中间访存。

每项改变都要重新做输出、梯度、数值范围与性能验证。不能把多个新名词加进网络后，只凭模型能跑就判断设计成功。

## GQA 为什么减少缓存

普通 MHA 每个 query 头对应自己的 K/V 头；GQA 让一组 query 头共享同一个 K/V 头。若 query 头数 $H_q$、KV 头数 $H_{kv}$ 且 $H_q/H_{kv}=g$ 为整数，每 $g$ 个 query 头读同一 K/V。

其缓存与 $H_{kv}$ 成正比，因此在固定头维度下，相对 MHA 的缓存比例为 $H_{kv}/H_q$。query 头的分数计算仍存在，不能据此宣称整个 Attention 计算同比例减少。原始设计参见 [GQA 论文](https://arxiv.org/abs/2305.13245)。

## MoE 把一个 MLP 换成专家集合

设 $E$ 个专家 $f_e(x)$，路由分数 $a=W_rx$，从中选 top-k 集合 $\mathcal K(x)$。一种实现对选中分数归一化：

$$
w_e=\frac{\exp a_e}{\sum_{j\in\mathcal K}\exp a_j},\qquad
y=\sum_{e\in\mathcal K}w_ef_e(x).
$$

不同模型的 gate、共享专家和归一化规则可能不同，这只是一个可分析的版本。在 top-k 集合固定的小区域内，设上游梯度 $G$，则：

$$
\frac{\partial L}{\partial a_e}
=w_e\left\langle G,f_e(x)-y\right\rangle,\quad e\in\mathcal K.
$$

集合选择本身离散，边界处不能按普通平滑函数求导。特别是 top-1 后只对这一项重新归一化，权重恒为 1，路由分数没有这个任务梯度；真实 top-1 设计常保留未归一化 gate 或其他训练机制，不能照抄此式。

每层总专家参数约随 $E$ 增长，每 token 的专家计算约随 $k$ 增长。但 Attention、共享专家、路由、通信仍有成本；“总参数 500B、激活 10B”也不等价于一个 10B 稠密模型的速度或显存。

## 为什么需要负载平衡

若多数 token 路由到同一个专家，其他专家学得少，热门专家排队或溢出。令 $f_e$ 为实际路由比例、$p_e$ 为平均 gate 概率，一种辅助目标为 $E\sum_ef_ep_e$，并乘可调系数。

这种可微惩罚与任务目标可能相互影响；其他方案使用路由 bias 调整负载，仍要处理局部失衡和专家容量。容量不足时丢弃 token 与动态扩容，会产生不同训练语义。负载平衡不是单纯的推理优化。

## 低精度是误差预算

简化对称量化 $q=\operatorname{clip}(\operatorname{round}(x/s),-Q,Q)$，反量化 $\tilde x=sq$。在不饱和时误差上界约 $s/2$；遇到异常值，统一 scale 会使小值精度下降。因此现代实现会使用更细粒度缩放、高精度累加和特定高精度状态。

FP8 训练、BF16 激活、FP32 optimizer state 与 FP4 推理 KV cache 对应不同对象和误差传播路径。看到“FP4 cache”不能写成“整个模型以 FP4 训练”。

[DeepSeek-V3 技术报告](https://arxiv.org/abs/2412.19437)是学习 FP8 训练和 MoE 工程的一个历史案例；这里引用它解释已公开设计，不把它当作 2026 年最新模型。当前案例另见[企业报告章节](../frontier/)。

## 多 token 预测怎样改变目标

一个抽象的辅助目标为：

$$
L=L_{\mathrm{next}}+\sum_{j=2}^{J}\lambda_jL_{t+j}.
$$

辅助模块让当前或后续组合表示预测更远的 token，实际结构决定它能读取哪些信息。每个 $L_{t+j}$ 都需要正确移位和边界遮罩；不能给模块真实未来 token 却把测试结果说成无条件预测。

MTP 有时还支持推理时的候选提出与验证，但推理算法和训练目标是两层设计。拿掉辅助头是否保留质量、加速是否需要特殊部署，应从具体报告中确认。

## 动手扩展的优先级

先给小模型加 RoPE 与 RMSNorm，分别做梯度和因果性测试；再加 GQA 比较缓存形状。MoE 则先在单设备上实现 4 专家 top-2，打印路由分布和每专家梯度，确认专家真正参与学习。

练习：$H_q=8,H_{kv}=2$ 时缓存约是 MHA 的四分之一。解释为什么这不意味着训练 FLOPs、权重显存和总生成延迟都变为四分之一。
