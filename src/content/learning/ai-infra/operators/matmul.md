---
title: 分块矩阵乘法
description: 从一个输出元素到一个输出 tile，理解数据复用、归约维度、边界处理与累加精度。
course: ai-infra
topic: operators
module: 动手实现
order: 5
updated: 2026-10-07
prerequisites: [矩阵乘法, 归约与稳定 Softmax]
---

## 一个输出元素需要什么

设 $A\in\mathbb{R}^{M\times K}$，$B\in\mathbb{R}^{K\times N}$，则 $C=AB$ 满足

$$
C_{ij}=\sum_{r=0}^{K-1}A_{ir}B_{rj}.
$$

最直接的实现为每个 $(i,j)$ 单独计算长度为 $K$ 的点积。观察相邻输出，会发现同一行的 $A$ 和同一列的 $B$ 被重复使用。分块算法把输出组织成一个区域，让一组输入共同服务于多个输出。

## 为一个 tile 分配工作

一个 program 负责 $B_M\times B_N$ 的输出 tile，每一步沿归约维度读取 $B_K$ 个元素。第 $t$ 步执行

$$
C_{\mathrm{tile}} \mathrel{+}=
A_{\mathrm{tile}}^{(t)} B_{\mathrm{tile}}^{(t)},
$$

其中两块输入的形状分别是 $B_M\times B_K$ 和 $B_K\times B_N$。

同一块 $A$ 的每个值可以参与 $B_N$ 个输出列的计算，同一块 $B$ 的每个值可以参与 $B_M$ 个输出行的计算。这是 tile 提供数据复用的逻辑基础；具体数据经过哪些寄存器、共享内存和硬件指令，则由实现与编译器决定。

## 把二维位置变成指针

对于连续矩阵，`A[row, reduction]` 的偏移是 `row * K + reduction`；`B[reduction, column]` 的偏移是 `reduction * N + column`。分别扩展行、列向量，可以形成整个 tile 的地址矩阵。

```python
rows = tl.program_id(0) * BM + tl.arange(0, BM)
columns = tl.program_id(1) * BN + tl.arange(0, BN)
inner = tl.arange(0, BK)

# 某一步的归约位置
reduction = step * BK + inner
a_addresses = A + rows[:, None] * K + reduction[None, :]
b_addresses = B + reduction[:, None] * N + columns[None, :]
```

加载 $A$ 时同时检查行边界与 $K$ 边界，加载 $B$ 时同时检查 $K$ 边界与列边界。越界的输入填零，因为零不改变点积；输出写回则检查 $M$、$N$ 两个边界。

只检查输出边界仍然不够：最后一次 $K$ 分块也可能不完整。测试 `(M,N,K)=(33,47,65)` 可以同时覆盖三个方向的尾部。

## 教学实现与累加精度

完整实现见 [kernels.py](../../../../learning/operators/kernels.py) 的 `_gemm` 与 `matmul_into`。它使用 `16 × 16` 输出 tile、长度为 `32` 的归约分块，输入和输出均为 FP16，累加器为 FP32。

```python
accumulator = tl.zeros((BM, BN), dtype=tl.float32)
for step in range(tl.cdiv(K, BK)):
    # 此处加载带边界保护的 a、b 两个 tile。
    accumulator = tl.dot(a, b, accumulator)
```

`tl.dot` 描述块矩阵的点积，最终指令选择与硬件、dtype 和编译设置有关。不能把使用了 `tl.dot` 直接等同于“获得峰值性能”。[Triton GEMM 教程](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)进一步讨论了 tile、调优和 program 排列。

FP32 累加有助于减少累加阶段的误差，但输入已经按 FP16 表示，输出还会再次舍入。浮点加法次序也可能不同，因此与框架比较要使用容差，而非逐位相等。后续研究 FP32 输入时，还要记录 TF32 或其他乘法精度设置，见 [PyTorch CUDA 精度说明](https://docs.pytorch.org/docs/stable/notes/cuda.html)。

## 怎样建立性能模型

若将乘法和加法各计一次，常用的近似运算量为 $2MNK$ FLOPs。假设 $A$、$B$ 各读取一次，$C$ 写入一次，理想数据量为

$$
Q_{\mathrm{ideal}}=s(MK+KN+MN),
$$

其中 $s$ 是元素字节数。这是一个理想复用模型。实际不同输出 tile 可能重复读取输入，缓存也可能减少到达 HBM 的访问，所以它不是硬件流量的直接测量。

增大输出 tile，可能增加复用，同时提高累加器资源需求；减小 tile，可能增加 program 数量，却减少每个 program 的复用。最终要结合矩阵尺寸与设备测量。我们的基础实现没有加入 autotune、异步流水线和复杂 program 排序，不以超过框架 GEMM 为验收条件。

## 练习

用小矩阵在纸上计算一个 `2 × 2` 输出 tile，列出每一步共享的输入。接着运行 CPU 脚本中的 `tiled_matmul`，改变 tile 大小并验证结果一致。

最后解释：当 $M$、$N$ 都很小而 $K$ 很大时，这种“一个 program 负责一个输出 tile”的划分，为什么可能无法充分提供并行工作？这会引出后续的 split-K、部分结果合并与额外开销问题。
