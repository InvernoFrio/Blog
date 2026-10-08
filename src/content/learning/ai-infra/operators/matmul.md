---
title: 分块矩阵乘法
description: 从一个输出元素到一个输出 tile，理解数据复用、归约维度、边界处理与累加精度。
course: ai-infra
topic: operators
module: 动手实现
order: 5
updated: 2026-10-08
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

## 一次两步归约，手算整个输出 tile

取 $A=\begin{bmatrix}1&2&3\\4&5&6\end{bmatrix}$、$B=\begin{bmatrix}7&8\\9&10\\11&12\end{bmatrix}$。输出是 $2\times2$，归约维 K 为 3。设教学分块 $B_K=2$，第一步处理 K 索引 0、1：

$$
C^{(0)}=\begin{bmatrix}1&2\\4&5\end{bmatrix}
\begin{bmatrix}7&8\\9&10\end{bmatrix}
=\begin{bmatrix}25&28\\73&82\end{bmatrix}.
$$

第二步只剩 K 索引 2，另一项越界，填零：

$$
C^{(1)}=\begin{bmatrix}3&0\\6&0\end{bmatrix}
\begin{bmatrix}11&12\\0&0\end{bmatrix}
=\begin{bmatrix}33&36\\66&72\end{bmatrix}.
$$

累加得到 $C=\begin{bmatrix}58&64\\139&154\end{bmatrix}$。这里第二步的零只补足逻辑块形状，不改变真实点积。上面的 $2\times2$ 用于手算；GPU 实验实际使用适合其实现约束的更大 tile，不能直接将所有教学尺寸作为 `tl.dot` 配置。

注意第一步 A 的值 1 同时服务输出第一行两列，B 的值 7 同时服务输出第一列两行。我们不是为四个输出分别加载四套独立输入，而是希望让这些重复使用发生在片上。这正是矩阵乘法能比逐元素算子有更高算术强度的原因。

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

## 三个维度有三种不同的边界

M 决定输出有多少行，N 决定输出有多少列，K 决定每个输出要加多少项。M、N 尾部影响“哪些输出应写”，K 尾部影响“每个输出实际累加哪些输入”。这两个问题不能使用一张输出 mask 一并解决。

例如 `(M,N,K)=(33,47,65)`，输出 tile 为 $16\times16$、归约块为 32。grid 为 $3\times3$，共 9 个 program；每个 program 沿 K 做 3 步，覆盖 0–31、32–63、64–95。最后一步只有索引 64 有效。

最右输出 tile 的列索引为 32–47，真实列只到 46；最下输出 tile 行索引为 32–47，真实行只到 32。加载 A 时检查 row 与 K，加载 B 时检查 K 与 column，写 C 时检查 row 与 column。只有把这三个轴的职责分开，mask 才容易写对。

二维 grid 中相邻的 program 写不同输出区域，所以普通方案不需要跨 program 共享一个累加器。每个 program 内部则必须把所有 K 分块累加完，不能把每一步结果覆盖到同一输出而丢失旧值。

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

## 复用和累加器占用，怎样一起估算

一个 $B_M\times B_N$ 输出 tile 的单次 K 分块，做约 $2B_MB_NB_K$ FLOPs，读取 $B_MB_K+B_KB_N$ 个输入元素。暂时忽略输出写回，其算术强度为：

$$
I_{\mathrm{tile}}\approx\frac{2B_MB_N}{s(B_M+B_N)}.
$$

若输出 tile 为正方形 $b\times b$，得到约 $b/s$ FLOP/byte，说明增大输出边长在这个模型中提高输入复用。但累加器有 $b^2$ 个元素，边长翻倍使其元素数变为四倍，资源需求也在增长。这个关系解释了调 tile 的取舍，而非给出普适最优值。

比如整个矩阵都是 $256\times256$、FP16，粗略 FLOPs 为 $2\times256^3=33554432$，理想数据量为 $2\times3\times256^2=393216$ 字节。两者相除约为 85.33 FLOP/byte。它假设输入只读一次，并不等于实际 HBM 计数。

如果 M、N 都很小，输出 tile 数可能只有一个，但 K 很长，这个 program 要循环很多次，其他 SM 没有相应工作。split-K 把归约区间分给多个 program，可以增加并行，却必须合并部分和，引入额外写回、同步或原子操作，浮点加法顺序也会变化。

**练习解答。** 把归约块从 32 改为 64，只改变遍历 K 的分组，不改变点积目标；但迭代次数、资源占用和累加顺序可能改变。先用小矩阵确认输出，再测大矩阵，不要仅凭循环更少判断更快。

## 练习

用小矩阵在纸上计算一个 `2 × 2` 输出 tile，列出每一步共享的输入。接着运行 CPU 脚本中的 `tiled_matmul`，改变 tile 大小并验证结果一致。

最后解释：当 $M$、$N$ 都很小而 $K$ 很大时，这种“一个 program 负责一个输出 tile”的划分，为什么可能无法充分提供并行工作？这会引出后续的 split-K、部分结果合并与额外开销问题。
