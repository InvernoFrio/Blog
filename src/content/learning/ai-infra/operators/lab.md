---
title: 验证与实验档案
description: 把公式、代码、正确性检查和性能数据放进同一份可复查的算子实验记录。
course: ai-infra
topic: operators
module: 组合与验证
order: 7
updated: 2026-10-07
prerequisites: [前六篇正文]
---

## 先保存一个完整实验目录

下载这三个文件，保持在同一目录：

- [cpu_checks.py](../../../../learning/operators/cpu_checks.py)：Python 标准库参考验证。
- [kernels.py](../../../../learning/operators/kernels.py)：向量加法、Softmax、GEMM 与 Bias-ReLU 教学 kernel。
- [gpu_lab.py](../../../../learning/operators/gpu_lab.py)：GPU 正确性检查和测量报告。

CPU 入口无需安装 PyTorch。GPU 入口需要匹配的 NVIDIA CUDA、PyTorch 与 Triton 环境；这两个入口各自验证不同层次的内容。

```bash
python cpu_checks.py
python gpu_lab.py --check-only
python gpu_lab.py > result.json
```

先执行正确性检查，确认通过后再比较时间。GPU 实验输出是运行时生成的记录，本页不提供预设性能数字。

## 第一组：验证地址与边界

CPU 脚本检查连续矩阵、转置、切片、广播的地址关系，并枚举长度 `0, 1, 255, 256, 257, 1025` 的分块覆盖。正确性标准是有效元素恰好覆盖一次。

GPU 向量加法覆盖这组长度和 FP16、FP32 两种 dtype。它同时检查尾部与空输出，帮助发现“整除 tile 时一切正常”的索引错误。

这些用例不会自动覆盖所有布局：本专题的 GPU wrapper 明确拒绝非连续输入。若修改接口支持任意 stride，需要增加切片、转置和广播的 GPU 用例。

## 第二组：验证数值计算

Softmax 检查框架参考和行和，包含列数为 `1`、`17`、`1025` 的矩阵。CPU 参考另检查大幅值输入与平移不变性。GEMM 使用 `(1,1,1)`、`(33,47,65)` 和 `(64,96,128)`，其中奇数尺寸负责覆盖尾部。

当前 GPU 用例设置的是教学容差，不是通用数值契约。输出接近零时要看绝对误差，输出较大时也要看相对误差；不能用一项平均误差掩盖少数明显错误。可以用 [torch.testing.assert_close](https://docs.pytorch.org/docs/stable/testing.html#torch.testing.assert_close)建立参考检查。

另一个独立检查是在线 Attention：CPU 参考比较不同 block 大小和输入尺度，检查分母与输出状态的修正是否正确。这并没有验证 GPU Attention 的线程映射、mask 或反向传播。

## 第三组：生成真实计时

GPU 脚本先预热，再采用 CUDA events 重复测量，报告中位数、最小值和最大值。输出与临时空间预分配，使用默认 stream 和重复的常驻输入。

| 算子 | 基线 | 当前实验范围 |
| --- | --- | --- |
| 向量加法 | `torch.add`，指定输出 | 长度 4096 和 1048576，FP32 |
| Softmax | `torch.softmax`，指定输出 | 32 或 256 行，1025 列，FP32 |
| Bias-ReLU | 两个独立 eager 操作 | 与 Softmax 同尺寸，FP32 |
| GEMM | `torch.mm`，指定输出 | 256 × 256 × 256，FP16 |

这个范围不包含 CPU 到 GPU 传输、首次 JIT 或模型端到端执行。向量加法的 `logical_GB_per_s` 根据逻辑数据量计算，不是 HBM 硬件计数。框架版本、GPU 型号和 CUDA runtime 会一并保存。

## 写一份有解释的报告

建议保存下面的记录，不只留下“快了多少”一句话：

| 字段 | 需要记录的内容 |
| --- | --- |
| 实验问题 | 本次要检验哪一个性能假设 |
| 环境 | GPU、驱动、PyTorch、Triton、CUDA runtime |
| 工作负载 | shape、dtype、stride、数据分布 |
| 正确性 | 参考实现、容差、边界测试、错误统计 |
| 测量边界 | 预热、输出分配、同步、重复次数、缓存条件 |
| 结果 | 原始时间、波动范围、使用的数据量估算 |
| 解释 | 支持假设的证据、尚未解释的现象 |

如果实验不支持原先的判断，就记录它。比如增大 tile 后变慢，可能意味着单个 program 的资源压力超过了增加复用的收益；下一步应当采集相应指标，而不是继续随机调整参数。

## 从教学实现走向框架算子

目前 wrapper 只是显式调用 GPU kernel。真正接入训练或编译流程，还要定义算子 schema、设备实现、输出形状推断，以及需要时的自动求导行为，并检查与编译器的兼容性。[PyTorch Custom Operators](https://docs.pytorch.org/tutorials/advanced/custom_ops_landing_page.html)是进入这一步的官方入口。

基础实现也没有承诺别名输入输出、并发 stream、所有特殊浮点值等行为。扩大接口时，先写清这些约定，再补测试，而不是让当前的有限用例代替完整验证。

## 专题验收任务

选择向量加法或 Bias-ReLU，提出一个可以检验的优化假设。完成正确性测试，在至少三种规模下比较两种实现，保存环境和原始数据，再写一段解释。

进阶任务是为 GEMM 改变 tile 划分，或给在线 Attention 参考加入明确的因果 mask。验收看的是：计算语义是否清楚、证据是否完整、结论是否与测量范围一致。
