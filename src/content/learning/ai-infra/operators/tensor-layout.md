---
title: 张量布局与地址计算
description: Shape 描述逻辑结构，stride 决定怎样找到数据。用地址计算理解转置、切片与广播。
course: ai-infra
topic: operators
module: 基础模型
order: 1
updated: 2026-10-07
prerequisites: [数组索引]
---

## Shape 不等于数据布局

一个 `3 × 4` 的张量告诉我们有三行四列，但还没有告诉我们相邻两行在内存里相隔多远。对普通 strided tensor，可以用 storage、shape、stride 和 storage offset 描述元素位置。

假设底层连续存着 `0, 1, …, 11`，把它解释为三行四列的行优先矩阵。每跨一行移动四个元素，每跨一列移动一个元素，所以 stride 是 `(4, 1)`。这里 stride 的单位是**元素**，换成字节时再乘 `element_size`。

这些概念对应 PyTorch 的张量视图机制；转置可能共享原有存储，却改变索引到存储的映射。见 [PyTorch Tensor Views](https://docs.pytorch.org/docs/stable/tensor_view.html)。

## 把索引变成地址

对索引 $(i_0,\ldots,i_{d-1})$，相对于底层 storage 起点的元素偏移是

$$
o = o_{\mathrm{storage}} + \sum_{j=0}^{d-1} i_j s_j.
$$

这里 $s_j$ 是第 $j$ 维的 stride。字节地址为 $\mathrm{base} + o \times \mathrm{element\_size}$。例如，原矩阵的元素 `(2, 3)` 对应 `2 × 4 + 3 × 1 = 11`。

有一个容易混淆的边界：**底层 storage 的起点和一个视图的第一个元素不是同一个概念。** PyTorch 将张量传给 Triton 时，指针已经指向该张量的第一个元素。以这个指针为基准，kernel 只需要计算 stride 偏移，不应再次加 storage offset。

## 转置、切片和广播

### 转置：改变解释方式

原张量 shape 为 `(3, 4)`，stride 为 `(4, 1)`。转置后 shape 为 `(4, 3)`，stride 为 `(1, 4)`。新张量的 `(3, 2)` 仍然指向元素 `3 × 1 + 2 × 4 = 11`。

这说明创建转置视图可以很轻，但使用该视图的后续计算可能面对不同访存模式。是否更慢取决于 kernel 怎样把元素分配给线程，不能仅凭“转置了”下结论。

### 切片：带步长地读取

取原矩阵的 `x[:, 1::2]`，得到 shape `(3, 2)`，stride `(4, 2)`，storage offset 为 `1`。其中 `(1, 1)` 相对于底层 storage 的偏移是 `1 + 1 × 4 + 1 × 2 = 7`。

### 广播：多个逻辑位置共享元素

把一个长度为四的向量扩展成三行，可以用 stride `(0, 1)` 表示：跨行没有移动，三行读的是同一组数据。这种零 stride 对读取很有用，但并发写入时，不同逻辑索引可能落到同一位置，需要另行定义行为。

```python
import torch

x = torch.arange(12).reshape(3, 4)
for name, value in (
    ("original", x),
    ("transpose", x.t()),
    ("slice", x[:, 1::2]),
    ("broadcast", torch.arange(4).expand(3, 4)),
):
    print(name, value.shape, value.stride(), value.storage_offset())
```

没有安装 PyTorch 也可以运行专题的 [CPU 验证脚本](../../../../learning/operators/cpu_checks.py)，检查这些地址关系。

## Contiguous、view 与 reshape

连续性和张量的存储格式有关。本专题先讨论常见的行优先连续张量，后续例子的 wrapper 会明确要求 `is_contiguous()`。

`view` 要求新的形状与原布局兼容。非连续张量并非一律不能 `view`，关键是所需的维度合并能否用现有 stride 表达。`reshape` 可以返回视图，也可能创建拷贝；`contiguous()` 在已经满足目标格式时直接返回原张量，否则生成对应的连续数据。代码不能依靠 `reshape` 一定零拷贝来估算性能。[官方视图说明](https://docs.pytorch.org/docs/stable/tensor_view.html)列出了这些区别。

在 kernel 外先调用 `contiguous()` 是一种实现选择。若要比较端到端性能，必须把它的拷贝成本算进去；如果测试只接收已连续的输入，应当在报告中写清楚这一前提。

## 布局怎样影响 GPU 访存

连续元素只有在相邻执行线程访问相邻地址时，才可能形成良好的合并访问。连续 tensor 本身不保证任意 kernel 都有好的访存模式。

以行优先矩阵为例，沿列方向相邻的元素地址差为 `1`，沿行方向相邻的元素地址差为 `N`。如果一个 warp 的线程依次读取同一列的不同行，会与读取同一行的相邻列产生不同的访存行为。实际事务数量还与对齐、访问宽度和硬件有关，详见 [CUDA 合并访问](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/index.html#coalesced-access-to-global-memory)。

## 练习与检查

对 shape 为 `(2, 3, 4)` 的连续张量，先写出 stride，再计算 `(1, 2, 3)` 的偏移。交换第一维与第三维后，重新计算同一底层元素在新视图中的位置。

接着设计一个只支持连续矩阵的 kernel 接口：哪些输入应该被拒绝，哪些可以直接使用？最后解释，为什么把隐含的拷贝放进 wrapper，会让“kernel 更快”和“整个算子更快”成为两个不同结论。
