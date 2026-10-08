---
title: 分布式训练：从正确梯度到集群效率
description: 推导多卡有效 token 加权，区分 DDP、FSDP、张量与流水并行，并讨论恢复、通信和数据状态。
course: llm
topic: from-scratch
module: 规模与系统
order: 10
updated: 2026-10-08
prerequisites: [梯度累积, 计算与显存预算]
---

## 多张卡训练的第一种方式

数据并行在每个 rank 放同一个模型，让不同 rank 读取不同数据。各自计算梯度，聚合后进行相同的参数更新。模型副本初始相同、聚合与优化顺序相同，才能保持同步。

单卡原型最常见的第一个升级是 DDP。它减少每张卡的数据计算量，但不分片模型和 Adam 状态；一个模型本来装不进单卡，直接加 DDP 通常没有解决这个问题。

## DDP 默认梯度平均与变长数据

假设有 $W$ 个 rank，第 $r$ 个 rank 的损失和为 $S_r$，有效 token 数为 $M_r$。我们需要：

$$
L=\frac{\sum_rS_r}{M_{\mathrm{global}}},\qquad M_{\mathrm{global}}=\sum_rM_r.
$$

按照 [PyTorch 2.8 DDP 文档](https://docs.pytorch.org/docs/2.8/generated/torch.nn.parallel.DistributedDataParallel.html)的梯度平均语义，本地反向应使用：

$$
L_r^{\mathrm{backward}}=\frac{W S_r}{M_{\mathrm{global}}}.
$$

这样聚合结果为 $\frac1W\sum_r\nabla L_r=\nabla L$。如果每个 rank 直接使用自己的平均损失 $S_r/M_r$，再平均梯度，就把 token 少的 rank 放大了。

例如两卡分别有 1、9 个有效 token。正确全局目标按 1:9 加权；rank 平均却按 1:1 加权。SFT、变长 packing 与最后一个 batch 都可能触发这个问题。

随附 `distributed_check.py` 启动两个真实 CPU 进程和不等 token 数。默认通过临时文件交换梯度，比较全局 token 加权后的结果与单进程合并批次的梯度，并确认错误的 rank 均值会失败。运行：

```bash
python distributed_check.py
```

默认实验检查数据并行的数学语义，文件交换没有调用 DDP 或 AllReduce。具备可用 Gloo transport 的环境可运行 `python distributed_check.py --backend gloo`，用真实 DDP 做相同对照。本机 Windows PyTorch 2.8.0 报告 `unsupported gloo device`，因此这里实际验证的是默认文件交换路径，不宣称 Gloo 路径已在本机通过。

实际训练还要考虑梯度累积：整轮更新的全局有效 token 数应先确定；使用 `no_sync()` 跳过前几个 microbatch 的通信时，最后一次同步必须包含累计梯度。两种实验都不是多卡吞吐测试。

## 几种并行策略切分不同对象

| 方式 | 切分对象 | 主要代价 |
| --- | --- | --- |
| DDP 数据并行 | 输入样本，复制状态 | 梯度 AllReduce |
| ZeRO / FSDP | 参数、梯度或优化器状态 | 参数 AllGather、梯度 ReduceScatter 等 |
| Tensor Parallel | 一个层内部矩阵与头 | 层内通信，对互联敏感 |
| Pipeline Parallel | 网络的不同层 | 流水气泡、阶段负载不均 |
| Context Parallel | 长序列的不同位置 | 跨分片获取 Attention 信息 |
| Expert Parallel | MoE 专家 | token dispatch、All-to-All、负载不均 |

FSDP 的具体行为取决于分片策略与实现版本，不能把所有参数都永久视为单卡上的 $1/W$。前向可能临时聚集完整参数，峰值还包含预取与工作区。概念可参照 [ZeRO 论文](https://arxiv.org/abs/1910.02054)与 [Megatron-LM](https://arxiv.org/abs/1909.08053)。

## 通信量和通信时间

Ring AllReduce 对每个 rank 的发送量近似为 $2(W-1)/W$ 倍梯度字节数。设带宽为 $b$、启动延迟为 $\alpha$，粗估包含传输项和若干轮延迟项；这个模型只描述特定算法的量级。

重叠反向计算与通信能隐藏一部分时间，但最后的尾部 bucket 通常不能完全隐藏。小模型、很小的 microbatch、跨节点慢互联可能让更多卡反而降低每卡效率。

衡量扩展应同时报告全局 tokens/s、每卡 tokens/s、有效监督比例和验证结果。若为了提高吞吐改变 batch 与学习率，模型质量比较也变了。

## 大规模恢复是分布式一致性问题

检查点要对应同一个已完成的 optimizer step。所有 rank 的参数分片、优化器分片、随机状态和数据位置需要完整；一张卡写完、另一张卡失败，不应把目录标记为可恢复。

通常先写临时分片与 manifest，校验完整性后发布完成标记。恢复时验证版本、拓扑和分片格式；改变 world size 还需要支持重分片与数据状态重映射。仅恢复权重能够继续优化，但无法证明恢复原来的训练轨迹。

## 实验与思考

运行双进程检查，观察脚本同时验证正确聚合与错误聚合。再把两个进程的 token 数改为相同，错误的均值平均也可能得到正确结果；这说明等长数据掩盖了错误。

进阶项目：用 DDP 包装 `MiniLM`，固定全局 batch 和调度，比较 1 卡与 2 卡的验证曲线。保存时让各 rank 记录自己的 RNG 与读取位置，并在一次中断后检查监督 token 是否重复或遗漏。
