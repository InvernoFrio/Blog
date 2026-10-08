---
title: 算子学习导览
description: 从一行张量表达式出发，建立理解数据布局、GPU 执行和性能优化的学习路径。
course: ai-infra
topic: operators
module: 学习导览
order: 0
updated: 2026-10-08
prerequisites: [Python 数组与循环, 矩阵乘法]
---

## 为什么从算子开始

写下 `y = x + bias` 时，我们描述了输入和输出之间的关系。真正执行它，还要决定哪些线程读取哪些元素、结果写到哪里、需要几次访问内存。算子学习把这些问题放到同一张桌上：先弄清计算语义，再研究硬件怎样完成计算。

**算子**是一个计算接口及其语义，例如加法、矩阵乘法和归一化。**Kernel** 是设备端执行代码。同一个算子可能根据 shape、dtype、布局和设备选择不同 kernel；几个算子也可能被编译器融合到一次执行中。因此，框架表达式的数量和 GPU kernel 的数量不能直接画等号。

这个专题围绕四个问题展开：数据在哪里，谁来计算，结果是否正确，时间花在了哪里。学完后，应当能读懂一个基础 Triton kernel，并为一次性能判断提供可复查的依据。

## 从一句表达式追到一次硬件执行

先看 `y = relu(x + bias)`。数学上，它要求每个位置先加对应偏置，再把负数变成零。框架首先检查输入的 shape、dtype、device 和广播关系，再根据这些信息选择实现。设备端实现还要决定，一次执行处理哪些位置、如何计算地址、什么时候加载和写回。

如果加法和 ReLU 分开执行，加法先把结果写入临时张量，ReLU 再读取它。如果融合成一个 kernel，某个位置的和可以直接参与 ReLU，再写到最终输出。数学结果可以相同，数据移动过程却不同。这就是算子优化的一个入口。

不过“少一个中间张量”还不等于“用户程序一定更快”。数据传输、首次编译、框架提交和其他算子都可能占时间。我们必须先定义优化的是哪一段：设备执行、一个算子调用，还是完整任务。之后才能说某个改动解决了什么问题。

可以把本专题的对象分成三层：算子语义定义结果，算法决定计算与复用方式，kernel 把算法映射到设备。先不要试图同时学完所有硬件细节；每到一层，都用小例子检查自己仍然在计算同一个结果。

## 一个贯穿全篇的例子：513 个元素的加法

输入是两个长度 513 的连续 FP32 向量，输出也有 513 项。设每个 program 负责 256 个逻辑位置，那么前两个 program 分别处理 0–255、256–511，第三个 program 只写第 512 项。其余逻辑位置必须禁止访问输入和输出。

这一例子会贯穿布局、执行、mask 与测量。布局回答 `x[512]` 在哪里；执行划分回答谁负责它；mask 回答为何第三个 program 不越界；测量回答这么小的任务是否已主要受启动影响。

等这些问题都能说清楚，再转向 Softmax。Softmax 的输出需要一整行共同决定分母，独立位置的划分不再够用。矩阵乘法又进一步要求多个输出共享输入。在线 Attention 最后把矩阵计算和归约结合起来，说明为什么系统优化必须同时理解数学与数据流。

## 学习时先写四句话

每个算子先写：输入有哪些，输出是什么，哪些输出依赖哪些输入，一个执行单元负责哪个区域。例如 Softmax 不是简单“每个数取指数”，还必须按同一行共享最大值和分母；GEMM 不是逐元素乘法，而是沿 K 轴归约。

接着写边界：长度为零怎样处理，维度不整除 tile 怎么办，是否接受非连续输入，是否允许输入输出指向同一存储。边界没有定义清楚，性能优化就可能悄悄改变算子的含义。

最后再提出性能假设。比如“融合将减少中间矩阵的两次逻辑访问”，是可以验证的假设；“用了 Triton 所以应该快”，没有指出成本来自哪里。后面的正文会把这一步落实到数字、代码和实验记录。

## 推荐学习顺序

| 顺序 | 文章 | 完成后应该能做什么 |
| --- | --- | --- |
| 01 | [张量布局与地址计算](../tensor-layout/) | 根据 shape、stride 和 offset 算出元素位置 |
| 02 | [GPU 执行与性能测量](../performance/) | 区分计算、访存与启动开销，设计计时范围 |
| 03 | [第一个 Triton 算子：向量加法](../vector-add/) | 分配 program 的工作，处理尾部 mask |
| 04 | [归约与稳定 Softmax](../softmax/) | 写清归约轴，解释数值稳定与填充值 |
| 05 | [分块矩阵乘法](../matmul/) | 理解 tile 复用、累加和边界处理 |
| 06 | [算子融合与在线 Attention](../fusion-attention/) | 分析中间结果的代价，推导分块归一化 |
| 07 | [验证与实验档案](../lab/) | 建立正确性测试和可复查的实验报告 |

建议先按顺序走一遍，再用左侧目录查阅具体问题。每篇末尾都有练习，练习的目标是检验理解，不需要追求同一台机器上的某个分数。

## 两种实验入口

### CPU：验证数学与索引

下载 [cpu_checks.py](../../../../learning/operators/cpu_checks.py)，使用 Python 标准库即可运行：

```bash
python cpu_checks.py
```

脚本检查地址计算、分块边界、稳定 Softmax、分块 GEMM，以及在线 Attention 与完整参考计算的一致性。它验证的是公式和算法关系，不模拟 GPU 的真实调度，也不能用于估计 GPU 性能。

### GPU：实现与测量

下载 [kernels.py](../../../../learning/operators/kernels.py) 和 [gpu_lab.py](../../../../learning/operators/gpu_lab.py)，放在同一目录。实验以兼容的 NVIDIA CUDA、PyTorch 和 Triton 环境为前提。

```bash
python gpu_lab.py --check-only
python gpu_lab.py > result.json
```

GPU 脚本先检查结果，再生成实际的计时数据。它会记录 GPU 型号、框架版本与 CUDA runtime，输出采用 CUDA events 的测量。安装 PyTorch 时按[官方安装选择器](https://pytorch.org/get-started/locally/)匹配环境；Triton 平台支持与依赖以[官方项目说明](https://github.com/triton-lang/triton)为准。

本专题提供的 kernel 是教学实现：连续输入、有限浮点值、明确的 dtype 和尺寸约束。它们没有实现自动求导、完整算子注册或生产级调优。GPU 部分的运行结果需要在你的实验环境中生成，正文没有预填加速倍数。

## 怎样判断自己学会了

读完某一篇后，尝试先关掉代码，回答三个问题：这个 kernel 的一个执行单元负责什么？任意一个输出元素怎样找到它的输入？当尺寸不是 tile 的整数倍时，为什么仍然正确？

完成专题的验收任务是：为一个基础算子建立参考实现、边界测试与性能报告，并解释至少一次“原以为会更快，测量却没有提升”的原因。解释可以是启动开销、缓存、资源占用，也可以是原先的模型不适用于当前规模。

## 资料怎样使用

先阅读正文并完成小实验，再对照原始资料查具体问题。推荐从 [Triton 向量加法](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)、[PyTorch Tensor Views](https://docs.pytorch.org/docs/stable/tensor_view.html) 和 [CUDA Best Practices Guide](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html) 的相关章节开始。有关框架算子注册的工程工作，留到后续阅读 [PyTorch Custom Operators](https://docs.pytorch.org/tutorials/advanced/custom_ops_landing_page.html)。
