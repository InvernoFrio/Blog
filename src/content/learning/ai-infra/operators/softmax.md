---
title: 归约与稳定 Softmax
description: 从一行数据的最大值与总和出发，理解归约、数值稳定和融合实现的边界。
course: ai-infra
topic: operators
module: 动手实现
order: 4
updated: 2026-10-07
prerequisites: [第一个 Triton 算子：向量加法, 指数函数]
---

## 从逐元素计算走向归约

加法的每个输出只依赖对应位置的输入。Softmax 的一个输出却依赖整行：它需要所有元素共同决定分母。这样的依赖要求我们把一组值归约为最大值或总和。

对一行 $x$，Softmax 定义为

$$
y_i = \frac{e^{x_i}}{\sum_j e^{x_j}}.
$$

它把输入转换成非负且总和为一的权重。本篇只处理非空、有限浮点输入，不把 NaN、正无穷或整行负无穷的行为混入基础实现。

## 为什么要减去最大值

直接计算大正数的指数可能溢出。令 $m=\max_j x_j$，可以等价地写为

$$
y_i = \frac{e^{x_i-m}}{\sum_j e^{x_j-m}}.
$$

分子与分母同时乘了 $e^{-m}$，所以在实数运算中结果不变；移位后的最大值为零，指数不超过一。它避免了正方向指数溢出，但仍然有浮点误差，极小权重也可能下溢。

```python
import math

def stable_softmax(row):
    maximum = max(row)
    weights = [math.exp(value - maximum) for value in row]
    denominator = sum(weights)
    return [value / denominator for value in weights]
```

用 `[10000, 9999, -10000]` 检查这个实现，再把整行加上同一个常数，比较输出是否基本不变。

## 一个 program 处理一行

我们给矩阵的每一行分配一个 program。行长度为 $N$，逻辑向量长度取不小于 $N$ 的最小二次幂 $B$。例如 $N=1025$ 时，$B=2048$，额外位置仅用于表达计算形状，不属于真实输入。

```python
column = tl.arange(0, TILE)
values = tl.load(
    X + row * COLS + column,
    column < COLS,
    other=-float("inf"),
).to(tl.float32)
shifted = values - tl.max(values, axis=0)
weights = tl.exp(shifted)
probabilities = weights / tl.sum(weights, axis=0)
tl.store(Y + row * COLS + column, probabilities, column < COLS)
```

无效位置为什么填负无穷？因为它不应改变最大值，并且指数之后应贡献零。若填零，一行真实输入全部为负数时，填充值就可能错误地成为最大值，并且额外的指数项会进入分母。

`axis=0` 指这一行的逻辑列向量。代码将输入转换为 FP32 进行归约，写回时再转为输出 dtype。完整的 `_softmax` 与 `softmax_into` 见 [kernels.py](../../../../learning/operators/kernels.py)。[Triton Softmax 教程](https://triton-lang.org/main/getting-started/tutorials/02-fused-softmax.html)提供了进一步的资源分析与实现方法。

## 融合节省了什么

按最大值、减法、指数、求和、除法逐步执行，会产生多个中间结果，并可能反复访问全局内存。将适合的计算放到一个 kernel 中，可以让中间数据留在片上计算过程中，减少显式中间张量的读写。

但不能由此推断一定比 `torch.softmax` 快。框架的原生 Softmax 本身就可能经过优化。实验应比较当前的实际实现，并关注矩阵形状，而不是把五个表达式的假想朴素实现作为唯一基线。

## 一行很长时会发生什么

一行放进单个 program 的逻辑向量越大，资源需求通常越高；跨过二次幂边界，还会出现较多无效位置。本专题 wrapper 把列数限制为 `1..4096`，这只是教学接口范围，不是所有硬件上的性能保证。

长行可以考虑分阶段归约、更多协作方式或在线归一化。进入下一步之前，要重新解决部分结果的组合、同步与数值稳定问题，不能只是无限增大 `TILE`。

## 正确性检查与练习

检查输出接近框架参考、每行总和接近一、权重非负，以及给整行加常数后的不变性。FP16 结果与 FP32 参考的差异需要用合适的绝对和相对容差解释；接近零的项尤其不能只看相对误差。

把尾部填充值分别改为 `0` 和负无穷，用一行全负数输入解释差别。再比较列数 `1024` 与 `1025`：数学工作量几乎相同，教学 kernel 的逻辑 tile 却不同，测量可能反映哪些影响？
