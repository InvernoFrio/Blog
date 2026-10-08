---
title: 从随机参数训练，并精确恢复训练
description: 完整运行一个小模型，理解检查点的模型、优化器、调度和随机状态，并验证续训等价性。
course: llm
topic: from-scratch
module: 训练实践
order: 7
updated: 2026-10-08
prerequisites: [训练循环与 AdamW]
---

## 建立一条可以复查的基线

在实验包目录运行数学与集成检查，再进行 200 次参数更新：

```bash
python math_checks.py
python smoke_test.py
python minillm.py train --data demo.jsonl --out runs/base --steps 200
```

默认上下文 64 字节、宽度 64、4 个头、2 层、batch size 8、dropout 0.1、种子 17，CPU 单线程。每 50 步输出一条 JSON，记录本批 loss、验证 loss、学习率、裁剪前梯度范数和累计监督 token。

训练采样有放回，所以“200 步”不等于“200 个 epoch”。有效 token 数随尾部 padding 变化，预算比较应使用 `seen_tokens`。验证始终遍历固定的所有验证窗口，关闭 dropout，按有效 token 数加权。

随附的真实 CPU 实测中，固定验证 loss 从 5.5502 降至 2.5876，累计监督 84,712 个 token。数学、因果性与恢复检查也通过；这证明该小实验的训练链路成立，不证明通用语言能力。曲线与逐步记录见[实验档案](../lab/)。

## 读懂两个输出文件

`runs/base/report.json` 包含初始与最终验证指标、数据哈希、独立文档数、窗口数、模型配置和时间。它是实验证据，不是模型权重。

`runs/base/last.pt` 保存恢复训练所需状态：

| 状态 | 如果缺失会发生什么 |
| --- | --- |
| 模型参数与配置 | 无法恢复同一个网络 |
| AdamW 状态 | 动量与二阶矩从零开始，更新轨迹改变 |
| optimizer step 与已监督 token | 调度位置与预算日志错误 |
| 批采样 Generator 状态 | 下一批数据改变 |
| Torch、Python、CUDA 随机状态 | dropout 或随机操作改变 |
| 数据哈希与训练约定 | 不易发现数据或调度被更换 |

本脚本在完整 step 后原子替换检查点，不保存未完成的梯度累积；因此恢复点就是最近已完成的更新。大集群还需要保存分布式分片、数据读取位置、动态精度状态和版本清单。

## 连续训练和中断训练怎样比较

```bash
python minillm.py train --out runs/full --steps 200
python minillm.py train --out runs/part --steps 100
python minillm.py train --out runs/resumed --steps 200 --resume runs/part/last.pt
```

`--steps 200` 是达到第 200 步，恢复后再做 100 步。不能把 `--decay-steps` 从 1000 改成 200；其他影响优化与采样的参数也必须保持一致。脚本检查这些约定，不匹配就报错。

`smoke_test.py` 做了一个更快的 6 步对比：连续 6 步，与 3 步后恢复至 6 步，逐项比较参数、Adam 状态、随机状态与监督 token 数。启用 dropout 让随机状态缺失确实能被发现。

相同环境上的 CPU 测试可以要求逐元素一致；换 PyTorch 版本、设备、并行策略或 kernel 后，不能保证位级一致。统计可复现与位级可复现需要不同验收标准。

## 继续训练和开始新阶段

`--resume` 恢复全部状态，表示同一个实验继续。`--init-from` 只加载模型与结构、重新创建优化器和采样状态，表示新训练阶段。后面的 SFT 使用后者，两个参数不能同时指定。

改变语料继续优化可能合理，但应把它登记为新阶段。强行绕过恢复检查会让日志看似连续，而实验条件其实已经变了。

## 扩大实验前先做两个检查

先尝试在很小的训练集上过拟合：关闭 dropout，检查训练 loss 能否显著下降。这是优化路径的诊断，不是泛化评估。再增加文档多样性并独立留出验证集，检查改善是否保留。

GPU 支持需先安装 CUDA 版 PyTorch，再显式运行 `--device cuda`；设备支持时可用 `--amp bf16`。本专题发布的实测以 CPU 为准，不能从 CPU 结果推导 GPU 吞吐或集群成本。详细实测档案见[最后一章](../lab/)。

练习：复制语料并修改一行，尝试 `--resume`。预期恢复检查失败。解释这是为什么，以及何时应改用 `--init-from`。
