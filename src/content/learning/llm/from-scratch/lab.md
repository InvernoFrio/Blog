---
title: 实验档案、练习验收与进阶项目
description: 下载全部代码和原创数据，建立数学、训练、续训、SFT 与双进程梯度的完整验收流程。
course: llm
topic: from-scratch
module: 实验与复习
order: 15
updated: 2026-10-08
prerequisites: [完成前面的训练实践]
---

## 全部实验材料

[下载完整实验包](../../../../learning/llm-from-scratch/llm-from-scratch.zip)，或按需下载下表。下载脚本后放在同一目录；无需 API 密钥、外部数据服务或预训练权重。

| 文件 | 用途 |
| --- | --- |
| [minillm.py](../../../../learning/llm-from-scratch/minillm.py) | 字节模型、预训练、评估、恢复、生成与 SFT |
| [math_checks.py](../../../../learning/llm-from-scratch/math_checks.py) | CE、线性层、GELU、Attention、LayerNorm、AdamW、RoPE、DPO 的数值检查 |
| [smoke_test.py](../../../../learning/llm-from-scratch/smoke_test.py) | 因果性、标签边界、共享参数、精确恢复及阶段衔接 |
| [distributed_check.py](../../../../learning/llm-from-scratch/distributed_check.py) | 双 CPU 进程的全局梯度验证；默认文件交换，可选 Gloo DDP |
| [rl_bandit.py](../../../../learning/llm-from-scratch/rl_bandit.py) | 四动作策略梯度与精确期望对照 |
| [demo.jsonl](../../../../learning/llm-from-scratch/demo.jsonl) | 32 篇原创短文，文档级训练与验证切分 |
| [demo-sft.jsonl](../../../../learning/llm-from-scratch/demo-sft.jsonl) | 16 组原创短问答，回答区域监督 |
| [requirements-cpu.txt](../../../../learning/llm-from-scratch/requirements-cpu.txt) | 固定版本的 CPU 安装入口 |
| [README.md](../../../../learning/llm-from-scratch/README.md) | 命令、运行条件与实验限制 |
| [experiment_results.json](../../../../learning/llm-from-scratch/experiment_results.json) | 实际 CPU 运行的指标、配置、代码哈希和固定生成样本 |

代码与数据的使用约定随实验包提供。语料是为数学与工程教学编写的短样本，不是高质量预训练语料集的替代品。

## 一次完整验收

```bash
python math_checks.py
python smoke_test.py
python distributed_check.py
python rl_bandit.py
python minillm.py train --out runs/base --steps 200
python minillm.py generate --checkpoint runs/base/last.pt --prompt "Learning "
python minillm.py train --sft --data demo-sft.jsonl --init-from runs/base/last.pt --out runs/sft --steps 200 --lr 0.001
```

数学检查失败时先修公式或轴；因果性失败时不应继续用 loss 判断模型；恢复失败时先查 RNG、优化器与调度。所有检查通过后，再谈更大模型与更长训练。

## 已完成的 CPU 实测

2026-10-08 在 Windows、Python 3.13.11、PyTorch 2.8.0+cpu 上运行。模型参数与计算均为 FP32，CPU 单线程、种子 17。预训练和 SFT 各完成 200 次参数更新；后者只载入前者权重，重新开始优化器。

| 指标 | 随机初始化后的预训练 | 接续的小语料 SFT |
| --- | --- | --- |
| 独立训练 / 验证文档 | 26 / 6 | 13 / 3 |
| 固定验证目标 token 数 | 1,251 | 38 |
| 初始验证 loss | 5.5502 | 3.0263 |
| 最终验证 loss | 2.5876 | 4.3940 |
| 初始 → 最终验证 PPL | 257.29 → 13.30 | 20.62 → 80.96 |
| 累计监督 token | 84,712 | 24,091 |

<img src="../../../../learning/llm-from-scratch/validation-curves.svg" alt="实际 CPU 训练曲线：预训练验证损失下降，极小问答集的 SFT 出现训练拟合与验证退化" width="1700" height="714" loading="lazy" />

蓝线遍历固定验证集；橙线只是该更新所采样训练批次的平均损失，不是完整训练集评估。两阶段使用不同数据与监督区域，不能直接把两幅图的 PPL 当成同一任务指标。图中只展示实测点，连接线辅助阅读，未补造中间记录。可下载 [PNG](../../../../learning/llm-from-scratch/validation-curves.png) 或 [SVG](../../../../learning/llm-from-scratch/validation-curves.svg)。

预训练的概率预测有改善，SFT 则明显过拟合。两种结果都保留，固定提示词输出也原样保存在 JSON 中，包括可能出现的非法 UTF-8 替代字符。这个模型没有因此获得可靠的问答能力。

数学推导检查、整网因果性、共享权重、SFT 标签边界、生成、阶段切换、6 步连续训练与 3+3 步恢复的逐元素对照均通过。双 CPU 进程的文件交换梯度验证通过；本机 Gloo transport 不可用，未把它记为 DDP 通过。四动作策略实验的期望奖励从 0.25 提升至约 0.9885，它验证 RL 数学，不是 LLM 强化学习结果。

## 留下可比较的记录

为每个实验保存代码版本、完整命令、数据哈希、tokenizer 版本、模型配置、硬件、框架与精度。训练指标按有效 token 加权，验证集和生成提示词提前固定。

原型只记录一台 CPU 的运行耗时，不提供 GPU 吞吐结论。双进程实验只验证聚合语义，RL 实验只验证离散策略梯度；它们都不能当作企业 LLM 训练复现。

## 四个可逐级完成的项目

### 项目一：重现并解释

完成基线与 SFT，写出标签移动、attention 遮罩和 loss 分母。提交训练报告、固定提示词生成，以及连续训练与恢复训练的参数比较。

验收：不仅脚本退出成功，还要解释为何验证 loss 和本批训练 loss 不直接可比，为什么不同 tokenizer 的 PPL 不应横向比较。

### 项目二：等预算结构消融

把位置 embedding 换为 RoPE，或把 LayerNorm 换成 RMSNorm，先逐个修改。至少用三种种子，固定训练 token 预算，记录参数、耗时、loss 和输出行为。修改 SwiGLU 时明确是等宽度还是等参数量比较。

验收：有原始基线、单项改变和数学/因果性测试，报告失败结果；不能只保留最好的一次。

### 项目三：数据与规模实验

使用有权使用的更大语料，先去重和文档切分，再训练 tokenizer。固定数据清单，比较三个模型规模与两个数据预算，绘制 validation loss 对累计 token 的曲线。

验收：来源和过滤可复查，验证集没有参与 tokenizer 训练；分词变化后同步修改 embedding、输出头和指标口径。

### 项目四：一个可验证的后训练任务

先在简短算术任务上定义严格解析和正确性验证，收集 SFT 与偏好样本。实现每回答 log-prob、冻结参考策略，再运行小规模 DPO。在线 RL 则作为下一阶段，需增加 rollout、旧策略 log-prob 和奖励记录。

验收：测试题从训练题生成规则或取值范围中留出，记录正确率、无效输出率与长度；不要把格式合规当成答案正确。

## 常见失败的排查顺序

| 现象 | 优先检查 |
| --- | --- |
| loss 很低却不会续写 | 输入是否等于标签、是否看到了未来、是否数据泄漏 |
| loss 为 NaN | 全遮罩行、空监督、数值溢出、学习率、精度 |
| 恢复后曲线不同 | optimizer、step、batch RNG、dropout RNG、调度与文件哈希 |
| SFT 不学开头 | 回答起点是否在移位后错了一位 |
| 多卡梯度不同 | 有效 token 全局分母、DDP 平均规则、累积同步边界 |
| 吞吐升高但效果下降 | 数据混合、全局 batch、精度误差、被截断或丢弃的 token |

## 继续读论文的路径

先读 [Transformer](https://arxiv.org/abs/1706.03762)、[AdamW](https://arxiv.org/abs/1711.05101)、[RoPE](https://arxiv.org/abs/2104.09864)；能够对应到代码后，再读 [Chinchilla](https://arxiv.org/abs/2203.15556)、[ZeRO](https://arxiv.org/abs/1910.02054)与 [DPO](https://arxiv.org/abs/2305.18290)。最后用[企业报告阅读表](../frontier/)拆解近期系统。

完整教材应当让你能从目标推到梯度，从梯度落实到代码，再用证据解释结果。复习用的逐步解答见[推导与练习详解](../worked-examples/)。
