---
title: 第一个 Triton 算子：向量加法
description: 把一个向量分给多个 program，理解索引、mask、数据加载与输出写回。
course: ai-infra
topic: operators
module: 动手实现
order: 3
updated: 2026-10-08
prerequisites: [张量布局与地址计算, GPU 执行与性能测量]
---

## 先约定接口

我们计算 $z_i=x_i+y_i$。输入是同一 CUDA 设备上、相同 dtype、相同长度的连续向量，输出提前分配。教学 wrapper 支持 FP16 和 FP32，不支持隐式广播和非连续输入。

这些约束让我们先集中解决地址计算。之后扩展算子时，可以加入 stride 参数和广播规则，但每增加一种输入能力，都需要增加对应的正确性检查。

## 一个 program 负责一段向量

设 tile 大小为 $B$，program 编号为 $p$，它负责的逻辑索引是

$$
pB,\;pB+1,\;\ldots,\;pB+B-1.
$$

要覆盖长度为 $N$ 的向量，需要 $\lceil N/B\rceil$ 个 program。完整代码放在 [kernels.py](../../../../learning/operators/kernels.py)，核心计算如下：

```python
@triton.jit
def _add(X, Y, Z, N: tl.constexpr, TILE: tl.constexpr):
    index = tl.program_id(0) * TILE + tl.arange(0, TILE)
    valid = index < N
    left = tl.load(X + index, valid, other=0)
    right = tl.load(Y + index, valid, other=0)
    tl.store(Z + index, left + right, valid)
```

`tl.arange` 描述一组逻辑位置。`tl.load` 根据这些位置读取输入，`tl.store` 将计算结果写回。`TILE` 是编译时常量，因为它决定这一块计算的形状。Triton 的官方[向量加法教程](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)介绍了这些 API 的基本语义。

## 按执行顺序读一遍 kernel

`X`、`Y`、`Z` 是设备指针，不是 Python 数组；`N` 是真实向量长度，`TILE` 是 program 的逻辑工作块大小。`tl.program_id(0)` 给出当前 program 在第 0 个 grid 轴上的编号，`tl.arange(0,TILE)` 给出块内一组偏移。

假设 `TILE=4`，program 2 的索引向量是 $2\times4+[0,1,2,3]=[8,9,10,11]$。`X + index` 是四个输入地址，`tl.load` 对这些地址进行带 mask 的读取，然后对读取结果逐项相加，最后写到四个输出地址。

Triton 描述的是一块向量运算，编译器把它映射到线程和指令；这里没有一个逐项 Python `for` 循环在设备上解释执行。也不要认为 TILE=256 必然代表 256 个 CUDA 线程：逻辑元素与物理线程的分配属于更低层映射。

同一个输出 $z_i$ 必须且只应被一个 program 写入。用 $p=\lfloor i/B\rfloor$、块内偏移 $i-pB$，可以从任意合法索引 $i$ 反推出负责它的 program。这是证明划分既不遗漏也不重叠的方法，比只看 grid 大小更完整。

## Mask 解决什么问题

若 $N=513$、$B=256$，三个 program 的起点依次是 `0`、`256` 和 `512`。最后一个 program 只有索引 `512` 有效，后面的逻辑位置超出了数组范围。

Mask 要同时保护加载和写回。只给输出写回加 mask，无法阻止输入的越界读取；只保护读取，也无法阻止写到输出范围之外。无效位置加载零是这里的实现选择，因为这些位置的结果不会写回。

CPU 验证脚本会枚举有效索引，检查所有元素恰好被覆盖一次。这个检查能验证分块公式，却不能发现真实 GPU 上所有潜在的内存错误。

## 一个能逐项核对的尾部例子

设 $N=10,B=4$，需要三个 program。其索引和 mask 是：

| program | 逻辑索引 | 有效 mask |
| --- | --- | --- |
| 0 | 0, 1, 2, 3 | 真, 真, 真, 真 |
| 1 | 4, 5, 6, 7 | 真, 真, 真, 真 |
| 2 | 8, 9, 10, 11 | 真, 真, 假, 假 |

第三个 program 无效加载位置返回 `other=0`，计算结果不写回。这些零不属于用户输出，只是定义内部计算张量的无效项。对独立逐元素加法，填成其他安全常数通常也不影响有效输出，因为有效位置不会读取无效位置的值。

后面的归约则不同：无效项会参加总和或最大值，填充值必须符合归约的中性行为。不能把向量加法里的 `other=0` 当成所有 kernel 的通用模板。

若误用 `index <= N`，索引 10 会被当成有效，恰好多访问一个元素。若 grid 误写为 `N // B`，尾部 8、9 又完全不会写。测试 `B+1` 与非整倍数长度能很快抓住这两类错误。

## Host wrapper 的职责

wrapper 首先检查 shape、dtype、设备与连续性，然后计算 grid 并启动 kernel。长度为零时直接返回空输出，避免无意义的启动。实现提供 `add_into(x, y, out, tile=256)`，输出由调用方管理。

```python
import torch
from kernels import add_into

x = torch.randn(513, device="cuda", dtype=torch.float32)
y = torch.randn_like(x)
out = torch.empty_like(x)
add_into(x, y, out)
torch.testing.assert_close(out, x + y, rtol=1e-3, atol=1e-3)
```

测试长度应覆盖 `0`、`1`、`B-1`、`B`、`B+1` 和更大的非整倍数。只测试恰好整除 tile 的长度，可能让尾部错误一直隐藏下去。

## Tile 更大一定更好吗

增大 tile 可以减少 program 数量，却也改变单个 program 的工作和资源需求。小向量可能没有足够的并行工作；过大的 tile 可能增加资源压力。选择 `256` 只是本专题的教学起点。

先验证，再用相同输入比较 `128`、`256`、`512` 等配置。GPU 型号、编译器版本与向量长度都是结论的一部分，不能把一组最佳配置推广到所有环境。

## 怎样解释带宽

对 FP32 长度 $N$ 的加法，逻辑数据量为 $12N$ bytes。如果测得时间为 $t$ 毫秒，逻辑吞吐可以写为

$$
B_{\mathrm{logical}}=\frac{12N}{t\times 10^6}\;\mathrm{GB/s}.
$$

这里的 GB 采用十进制。该指标使用逻辑读写量；输入或输出命中缓存时，不能把它当作实际 HBM 带宽。与框架比较时，使用 `torch.add(x, y, out=out)`，保持输出分配策略一致。

## 从向量加法走向稍复杂的算子

把目标改成 $z_i=2x_i+y_i$，地址与 mask 都不变，只增加一次乘法。按通常计数约定，每项变为 2 FLOPs，逻辑数据量仍是读两项、写一项的 12 字节，因此算术强度从 $1/12$ 增到 $1/6$ FLOP/byte。

如果任务仍由数据移动或启动主导，运算增加一倍并不意味着时间增加一倍。性能模型不是用来保证加速，而是帮助提出可解释的测量预期。请先验证数值，再按相同范围比较时间。

进一步支持 $z_i=x_i+b$ 的标量广播，需要明确 b 是 host 参数还是设备上的一个元素；支持非连续向量则需要传入 stride，地址变为 `X + index * stride_x`。这些扩展都保留同一个分工思路，但改变了接口和访存模式。

**练习解答。** $N=1025,B=256$ 时，前四个 program 处理 0–255、256–511、512–767、768–1023，第五个只写 1024。有效元素总数为 $4\times256+1=1025$；最后一块剩余 255 个逻辑位置均被 mask。

## 练习

把 $N=1025$、$B=256$ 的每个 program 工作范围写出来。再尝试将计算改为 $z_i=2x_i+y_i$，重新估算运算量与逻辑数据移动量，并解释为什么运算量增加不一定让总时间同比增加。
