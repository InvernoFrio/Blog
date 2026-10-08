---
title: 从零训练 LLM：学习地图
description: 把概率模型、反向传播、可复现实验与企业训练系统连成一条完整的学习路线。
course: llm
topic: from-scratch
module: 起点
order: 0
updated: 2026-10-08
prerequisites: [Python, 矩阵乘法, 导数与链式法则]
---

## 这次要亲手完成什么

从随机参数出发，训练一个能预测下一个 token 的小型语言模型。自己准备数据，计算损失，更新参数，检查验证集，保存状态，继续训练，最后生成文本。完成这条链后，再研究大模型怎样扩展它。

实践模型是两层 Decoder Transformer，默认只有 **120,768 个独立参数**。它使用 UTF-8 字节词表、Pre-LayerNorm、手写因果 Attention、GELU 和共享输入输出权重。这个规模适合普通 CPU 上检查数学与工程关系；随附的 32 篇原创短文用于实验，不足以训练出可用的通用助手。

“从零”指预训练不加载已有权重。后面的 SFT 实验加载我们自己训练的检查点，演示训练阶段的衔接。企业案例则研究公开报告，无法据此复现一家公司的完整数据和集群。

## 建议阅读顺序

| 阶段 | 文章 | 要回答的问题 |
| --- | --- | --- |
| 建立目标 | [概率与交叉熵](../probability/) | 模型究竟优化什么？ |
| 求出梯度 | [线性层与反向传播](../backprop/) | 每个参数为什么这样更新？ |
| 准备输入 | [数据与分词](../data-tokenizer/) | 预测标签为什么要移动一位？ |
| 核心计算 | [因果 Attention](../attention/) | 为什么模型不能看未来？ |
| 组装网络 | [Transformer 结构](../transformer/) | 残差、归一化、位置各负责什么？ |
| 优化参数 | [训练循环与 AdamW](../optimization/) | 累积、裁剪和学习率怎样配合？ |
| 实际运行 | [训练与精确续训](../training/) | 如何证明训练真的可复现？ |
| 检查能力 | [评估与生成](../evaluation/) | loss 下降为什么还会胡说？ |
| 扩展规模 | [数据、算力与显存](../scaling/) | 增大模型之前应估算什么？ |
| 扩展系统 | [分布式训练](../distributed/) | 多卡怎样得到正确的梯度？ |
| 现代结构 | [MoE 与高效模型](../modern-architecture/) | 参数量与计算量为什么分开？ |
| 指令学习 | [SFT 与监督遮罩](../sft/) | 只学回答意味着什么？ |
| 偏好与反馈 | [DPO、RL 与可验证奖励](../post-training/) | 交叉熵之外怎样训练？ |
| 前沿案例 | [企业训练报告怎么读](../frontier/) | 公开信息能支持哪些判断？ |
| 完整实践 | [实验档案与进阶项目](../lab/) | 如何留下别人能复查的证据？ |
| 查阅附录 | [逐步算例与习题详解](../worked-examples/) | 能否亲手算完一个更新？ |
| 补习数学 | [数学工具箱](../math-toolbox/) | 如何读懂期望、矩阵微分与数值误差？ |

数学较生疏时，先读概率、反向传播和 Attention。已经做过微调但没从头训练时，先运行实践，再回到公式检查自己的理解。AI Infra 的[算子专题](../../../ai-infra/operators/overview/)可以作为张量布局、Softmax 与性能测量的补充。

## 下载与第一轮实验

下载 [完整实验包](../../../../learning/llm-from-scratch/llm-from-scratch.zip)，解压并进入其中的目录。Python 建议 3.10–3.13；CPU 环境固定 PyTorch 2.8.0，便于复查。终端中的 `python` 应指向你创建的实验环境：

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1
# macOS / Linux:
# source .venv/bin/activate
python -m pip install -r requirements-cpu.txt
python math_checks.py
python smoke_test.py
python minillm.py train --out runs/base --steps 200
python minillm.py generate --checkpoint runs/base/last.pt --prompt "Learning "
```

若 PowerShell 的脚本策略不允许激活，可以直接用 `.venv\Scripts\python.exe` 替换命令中的 `python`。CPU 依赖文件仅供 CPU 实验；GPU 环境按 [PyTorch 官方安装入口](https://pytorch.org/get-started/locally/)选择与驱动兼容的版本。

## 三种成果要分别检查

数学成果是手推梯度与自动求导一致；工程成果是训练、验证和恢复状态正确；模型成果是对未见数据的能力改善。前两项通过，不自动意味着第三项达到实用水平。

每章都有一个可检查的问题，最后的实验页提供完整验收标准。读论文时也沿用这三个维度：公式改变了什么，系统怎样实现，哪些评估证明它有效。
