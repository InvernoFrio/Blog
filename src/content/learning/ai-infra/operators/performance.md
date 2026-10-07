---
title: GPU 执行与性能测量
description: 用计算量、数据移动和计时边界建立性能模型，避免把提交任务的时间当成 GPU 执行时间。
course: ai-infra
topic: operators
module: 基础模型
order: 2
updated: 2026-10-07
prerequisites: [张量布局与地址计算]
---

## 先理解谁在执行

CPU 侧准备参数并提交 kernel。GPU 侧由许多执行线程完成工作。在 NVIDIA CUDA 模型中，线程组成 block，block 的线程按 warp 组织；一个 warp 包含 32 个线程。Block 被调度到 SM 上执行，线程使用寄存器，同一 block 可以使用共享内存协作。具体资源容量取决于 GPU 型号。[CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/index.html)是这些术语的原始说明。

Triton 将一块张量计算描述为一个 program instance，再由编译器完成更底层的映射。后面的示例会为一个 program 指定 tile，但 tile 中的一个逻辑元素不等于永久绑定一个 CUDA 线程。读代码时先明确 program 的工作范围，再研究底层资源使用。

## 三种常见限制

**计算限制**：所需运算很多，执行管线的吞吐可能成为主要约束。**数据移动限制**：每次搬来的数据只做少量计算，访存带宽或延迟可能成为主要约束。**固定开销限制**：任务非常小，kernel 启动和框架调度的成本可能超过有用计算。

这三种描述可以随着输入规模变化。同一个向量加法，在很短的向量上可能主要受固定开销影响，在长向量上才逐渐表现出数据移动的特征。不能只给算子贴一个永不变化的标签。

## 从算术强度开始估算

设运算量为 $F$ FLOPs，指定内存层次的数据移动量为 $Q$ bytes，算术强度为

$$
I = \frac{F}{Q}.
$$

假定计算吞吐上限为 $P$，带宽上限为 $B$，一个理想化的下界是

$$
T \geq \max\left(\frac{F}{P},\frac{Q}{B}\right).
$$

这是帮助判断方向的模型。实际 kernel 还受启动、同步、访存延迟、资源占用和实现效率影响；$Q$ 也必须对应明确的内存层次，不能把逻辑读写量直接当作实测 HBM 流量。

对长度为 $N$ 的 FP32 向量加法，逻辑上读取两个向量、写出一个向量，得到 $F=N$、$Q=12N$，因而 $I=1/12$ FLOP/byte。这是我们的数据量估算，并非 profiler 采集到的硬件计数。

## GPU 为什么不能直接这样计时

```python
start = time.perf_counter()
y = x + 1
elapsed = time.perf_counter() - start
```

CUDA 操作通常异步提交。上面的区间可能主要覆盖 CPU 的提交工作，而 GPU 尚未完成。若要使用 CPU 时钟衡量完整等待时间，需要在计时区间前后同步；若要衡量设备 stream 上的执行区间，可以使用 CUDA events。见 [PyTorch CUDA 异步执行说明](https://docs.pytorch.org/docs/stable/notes/cuda.html#asynchronous-execution)。

一个基本的设备计时过程如下：

```python
# function 只操作已经在 GPU 上的输入，输出提前分配。
for _ in range(10):
    function()
torch.cuda.synchronize()

start = torch.cuda.Event(enable_timing=True)
end = torch.cuda.Event(enable_timing=True)
start.record()
for _ in range(30):
    function()
end.record()
end.synchronize()
ms_per_call = start.elapsed_time(end) / 30
```

这段代码测量默认 stream 上重复执行的平均时间。它排除了首次 JIT 编译，使用常驻输入，也没有覆盖输入传输、首次初始化和完整应用响应。CPU 提交速度如果无法及时喂满 GPU，event 区间仍可能包含设备等待的空隙；它不等同于 profiler 给出的纯 kernel 指令执行时间。

## 公平比较与结果记录

一次比较至少固定 shape、dtype、布局、设备、数值容差和分配策略。若自定义 kernel 提前分配输出，框架参考也应提前分配；若测量包含拷贝，两边都要明确是否计入。

首次编译与预热分开记录。重复多组测量，保留中位数与范围。反复使用同一输入可能命中缓存，应说明这是常驻输入实验；若要研究接近冷数据的场景，另做超过缓存工作集、轮换缓冲区的实验。

本专题的 [gpu_lab.py](../../../../learning/operators/gpu_lab.py)按这个范围输出真实测量，既没有屏蔽可能出现的慢结果，也没有把理论上限写成实测性能。

## 怎样从测量走向分析

如果小尺寸没有提升，先检查启动与调度开销。如果大尺寸没有达到预期带宽，检查地址模式和工作分配。扩大 tile 后变慢，则需要关注寄存器、占用率及可并行 program 的数量。

进一步使用 [PyTorch Profiler](https://docs.pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)查看算子和 kernel 的时间线，再使用 [NVIDIA Nsight Compute](https://docs.nvidia.com/nsight-compute/)分析设备指标。工具负责提供证据，性能模型负责提出待验证的解释。

## 练习

给 FP32 向量加法写出逻辑 GB/s 的计算式，并检查毫秒到秒的换算。然后为“把输入从 CPU 拷到 GPU，执行加法，再拷回 CPU”设计另一种计时范围，解释为什么不能直接与上面的常驻输入结果比较。
