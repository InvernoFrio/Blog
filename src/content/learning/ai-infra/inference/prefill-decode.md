---
title: 模型装进显存之后：Prefill、Decode 与 KV Cache
description: 从一次生成的计算过程出发，推导 KV Cache 的作用与容量，并用 CPU 小实验验证缓存和完整计算的一致性。
course: ai-infra
topic: inference
module: 推理基础
order: 0
updated: 2026-10-10
prerequisites: [矩阵乘法, 因果 Attention]
---

## 一次请求先做什么，再做什么

还没有形成系统地图时，先读[AI Infra 总体概览](../../foundations/overview/)。这篇进入推理与服务这一层，先研究一个请求的计算和状态，再看多个请求怎样共同使用设备。

考虑一个自回归语言模型：输入 4 个 token，希望继续生成 3 个 token。以下流程假定没有已有前缀缓存、没有投机解码，也没有提前停止。

```text
输入：A B C D
Prefill：处理 A B C D，建立各层历史 K/V，用最后位置的 logits 选出 E
Decode 1：输入 E，读取 A..D 的 K/V 并追加 E 的 K/V，选出 F
Decode 2：输入 F，读取 A..E 的 K/V 并追加 F 的 K/V，选出 G
输出：E F G
```

**Prefill 处理已知的输入序列，Decode 随生成过程逐步处理新 token。** 第一个生成 token 通常由 Prefill 最后位置的输出选出；生成 3 个 token 的这个例子，只需要再执行 2 次单 token Decode。最后输出的 G 尚未作为输入送进下一次前向，因此此时缓存保存到 F。

同一个 Transformer 会在两个阶段运行，变化的是输入规模、已有状态与计算组织方式。

## 为什么两阶段的性能特征不同

忽略 batch，设输入长度为 $T$、隐藏维度为 $D$。一个线性层可以写作

$$
X_{T\times D}W_{D\times D'}.
$$

Prefill 有许多已知位置，可以同时处理它们。一个权重块能够服务多个位置，这给矩阵计算提供了较多复用机会。因果遮罩保证每个位置只看自己和此前的 token；它不要求按位置依次执行整个 prompt 的所有层。

单请求、每步单 token 的 Decode 则更接近

$$
x_{1\times D}W_{D\times D'}.
$$

这一轮只产生少量输出，但仍需要访问权重，并读取历史 K/V 来完成 Attention。在这样的负载下，数据移动与小任务的启动成本可能占主要部分。连续生成又存在依赖：下一步的输入要等当前 token 选出。

所以常说 Prefill 更容易发挥计算吞吐，低 batch 的 Decode 更容易受内存带宽限制。但判断必须带上 batch、上下文长度、模型结构与硬件条件。增加并发可以让多个请求共享一批权重读取；长上下文又会增加 Attention 对历史缓存的访问。可以用[算子专题的性能模型](../../operators/performance/)估算，再用测量确认。

## KV Cache 保存了什么

先看某一层、某一个 Attention head。由该层输入计算

$$
q_t=x_tW_Q,\qquad k_t=x_tW_K,\qquad v_t=x_tW_V.
$$

当前位置的 Attention 输出为

$$
o_t=\operatorname{softmax}\left(\frac{q_tK_{\leq t}^{\mathsf T}}{\sqrt{d_h}}\right)V_{\leq t}.
$$

未来 token 需要历史位置的 $k$ 和 $v$，所以将它们保存下来。历史 $q$ 已完成对应位置的输出，标准自回归 Attention 不需要用它计算未来位置的输出。

在确定的模型参数、位置和因果上下文下，新追加的 token 不会改变此前位置的隐藏状态，因此此前各层的 K/V 可以复用。真实模型要分别保存每一层的状态，并保持位置编码、遮罩与缓存索引一致；训练时的随机行为或修改既有上下文，需要另外处理。

**缓存省掉历史位置的重复前向计算，但当前 query 仍要访问它需要的历史 K/V。** 常规完整 Attention 的缓存随上下文增长，每步 Attention 的工作也会随历史长度增长。KV Cache 不会让长上下文的 Decode 变成固定成本。

前缀缓存进一步允许符合条件的多个请求复用相同前缀的状态，需要检查模型、输入 token、位置及其他相关配置。它与“一个请求内部保存历史 K/V”是相关但不同的管理问题。

## 算清楚缓存容量

对各层结构相同、采用常规完整 Attention 的模型，忽略分页、量化辅助数据与存储对齐开销，逻辑缓存字节数为

$$
M_{\mathrm{KV}}=2L H_{\mathrm{KV}}d_h s\sum_{i=1}^{B}T_i.
$$

其中 $L$ 是层数，$H_{\mathrm{KV}}$ 是每层 K/V head 数，$d_h$ 是每个 head 的维度，$s$ 是每个元素的字节数，$T_i$ 是第 $i$ 个请求已缓存的位置数。系数 2 来自 K 和 V 两份状态。若所有请求长度相同，最后一项就是 $BT$。

**用 KV head 数，而不是直接用 query head 数。** MHA 通常两者相等；GQA 让一组 query heads 共享 K/V heads；MQA 只用一个 K/V head。它们之间的结构关系见 [GQA 原论文](https://arxiv.org/abs/2305.13245)。

做一个假设算例：32 层、32 个 query heads、head 维度 128，以每元素 2 字节保存缓存。

| 缓存方式与请求数 | 每个请求的缓存长度 | 逻辑缓存容量 |
| --- | --- | --- |
| MHA：32 个 KV heads，1 个请求 | 8192 | 4 GiB |
| GQA：8 个 KV heads，1 个请求 | 8192 | 1 GiB |
| GQA：8 个 KV heads，1 个请求 | 32768 | 4 GiB |
| GQA：8 个 KV heads，16 个请求 | 每个 8192 | 16 GiB |

例如 GQA 第一行的计算是 $2\times32\times8\times128\times8192\times2=1{,}073{,}741{,}824$ 字节，也就是 1 GiB。这里用 $1\ \mathrm{GiB}=2^{30}$ 字节。

这些是公式估算，不是显卡实测。权重、工作区、中间激活和运行时开销还需要另外计入；服务引擎也可能预先分配一个缓存池，所以已占用的设备显存不一定随某次请求逐 token 增加。共享前缀、滑动窗口、MLA、混合线性 Attention 或缓存量化，需要按各自状态表示重新计算。

## 从缓存容量走向并发与响应

多个请求共同运行时，可以在每轮推进不同请求。一个请求结束，调度器释放相应状态，再加入新的请求，这构成连续批处理的基本想法。

更多并发可能提高设备上的总吞吐，也会占更多缓存。到达速度超过处理能力时，请求开始排队；如果插入一个很长的 Prefill，还可能延迟正在 Decode 的请求。

因此要区分这些测量：

| 指标 | 回答什么问题 | 注意计时边界 |
| --- | --- | --- |
| TTFT：首 token 延迟 | 从请求开始到第一个生成 token，需要等多久？ | 客户端值通常包含网络、排队与 Prefill，服务端指标范围可能不同 |
| ITL：token 间隔 | 后续生成是否流畅？ | token 的设备生成时间和流式响应到达时间可能不同；一次响应也可能携带多个 token |
| 吞吐 | 单位时间完成多少请求或 token？ | 分开说明输入、输出 token 与测量区间 |
| 满足目标的有效吞吐 | 多少工作在指定延迟和质量条件下完成？ | 明确 TTFT、生成速度或总延迟目标，以及失败请求的处理方式 |

均值还应配合 p50、p95 或 p99 等分位数。指标定义可对照 [vLLM Metrics](https://docs.vllm.ai/en/stable/design/metrics/)；实际比较要确认服务端与客户端采用的计时起点是否相同。

Chunked Prefill 将长 prompt 的计算拆成较小块，允许和 Decode 交错安排。块的大小与调度策略会影响首 token 延迟和生成间隔；具体配置应按负载测量，见 [vLLM 优化文档](https://docs.vllm.ai/en/stable/configuration/optimization/)。这一篇先建立原因，后续再做服务压测。

## CPU 实验：缓存是否改变计算结果

下载 [kv_cache_lab.py](../../../../learning/inference/kv_cache_lab.py)，只需要 Python 3 标准库。

```bash
python kv_cache_lab.py --check-only
python kv_cache_lab.py --lengths 16 64 128 --tokens 16 --repeats 3 > result.json
```

脚本使用固定随机权重的单层、单头因果 Attention，加上输入 embedding、位置相关项和输出投影。它是展示计算关系的小模型，没有训练，也没有完整 Transformer 的归一化、残差与 MLP；生成的 token ID 不代表语言能力。

有两条执行路径：

1. 完整参考路径：每次重新处理整个已知序列，计算所有位置的因果 Attention，然后取最后位置的输出。
2. 缓存路径：Prefill 建立 prompt 的 K/V 并计算最后位置输出，后面每步只追加新 token 的 K/V，再计算新位置的 Attention。

正确性检查会逐位置比较 logits，验证分块边界处的位置与缓存状态，检查贪心生成序列一致，并验证上一节的显存手算。它也覆盖单 token prompt 和空输入的处理。

实验输出包含 Python 与平台信息、计时中位数和范围、缓存位置数、实际缓存元素数，以及假设这些元素存成紧凑 FP32 张量时的逻辑字节数。Python 实际使用列表与浮点对象，**这个字节估算不是 Python 进程的真实内存占用**。

计时范围包括 Python 函数调用、列表分配、Prefill 和生成循环，先预热，再重复测量。这台 CPU 上的教学比较同时包含“只求最后位置”和“复用 K/V”的收益：缓存路径的 Prefill 只计算最后 query，完整参考计算所有 query。因此不能把测得的比例归因于某一个优化，也不能把它当作 GPU 或生产服务的加速倍数。

### 读结果时检查三个地方

先检查 `checks.passed` 和 `max_logit_error`。结果一致后，再比较时间。不同执行顺序可能引入浮点误差，因此采用容差，不承诺所有平台逐位相等。

其次，输入长度为 $T$、输出数量为 $N$ 时，最后一次前向后的缓存长度应为 $T+N-1$，因为最后选出的 token 尚未送进下一步。单层单头的 K/V 元素数应为 $2d_h(T+N-1)$。

最后，比较 16、64、128 个输入位置下的缓存规模。它应线性增长，但总运行时间包含投影、Attention 和 Python 开销，不需要按同一比例增长。正文不预填性能数字，结果由自己的执行环境生成。

## 再向真实服务走一步

分页缓存关注怎样为变长请求分配、释放和共享 K/V 存储，后续可以对照 [vLLM 的 Paged Attention 设计](https://docs.vllm.ai/en/stable/design/paged_attention/)研究索引与 kernel。连续批处理、前缀缓存和 Chunked Prefill 则进一步影响请求怎样共同运行。

PD 分离将 Prefill 与 Decode 分配给不同工作池，让两类资源可以独立配置，但它还需要传输 KV Cache。传输和额外调度有成本，必须比较同一负载下的整体收益。小模型、短输入或低并发时，合并部署可能更合适，见 [NVIDIA Dynamo 部署说明](https://docs.nvidia.com/dynamo/v1.5.0/kubernetes/disaggregated-serving/overview)。

截至 **2026-10-10** 核对的近期进展中，SGLang 10 月 2 日的 v0.5.21 更新列出了 PD 实例运行中切换角色，以及默认采用 Rust 核心的前缀缓存。这些变化可以作为后续研究动态资源分配和缓存管理的切入口，见 [SGLang 官方更新](https://www.sglang.io/)。

## 练习

假设一个常规 GQA 模型有 24 层、4 个 KV heads、head 维度 128，缓存每个元素用 2 字节。8 个请求各缓存 4096 个位置，需要多少逻辑 KV 空间？

代入得到 $2\times24\times4\times128\times2\times8\times4096=805{,}306{,}368$ 字节，即 768 MiB。再想一想：为什么即使剩余显存超过 768 MiB，也不能保证这 8 个请求一定可以被同时服务？运行时还需要工作区与中间结果，缓存池可能有分配粒度或容量限制，调度和延迟目标也会约束可接受的并发。
