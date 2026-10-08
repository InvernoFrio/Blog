---
title: SFT：让模型学会回答，而非复述提示
description: 推导回答区域的监督目标，处理移位后的标签边界，并实际运行第二个训练阶段。
course: llm
topic: from-scratch
module: 后训练
order: 12
updated: 2026-10-08
prerequisites: [移位标签, 有效 token 累积, 检查点]
---

## 指令微调改变的是数据目标

预训练通常监督完整文本。SFT 给定问题或对话上下文 $c$，最大化示范回答 $y$ 的条件概率：

$$
L_{\mathrm{SFT}}=-\frac1M\sum_{n,t\in\mathrm{回答区域}}
\log p_\theta(y_{n,t}\mid c_n,y_{n,<t}).
$$

架构和反向传播没有必要改变；改变的是样本格式、监督遮罩与数据分布。它能训练模型遵循某种交互方式，但示范答案若有错误，交叉熵会奖励模仿这些错误。

实验的 `demo-sft.jsonl` 每行有 `prompt` 和 `response`，拼成：

```text
[BOS]User: <prompt>\nAssistant: <response>[EOS]
```

`User:`、换行和 `Assistant:` 都是普通字节，不是生产模型的标准 chat template。换用已有 tokenizer 或多轮对话时必须定义自己的角色、结束符和监督策略。

## 最容易错的那个位置

设原序列 `sequence` 中第一个回答 token 的索引为 $a$，从零开始。输入为 `sequence[:-1]`，标签为 `sequence[1:]`。标签位置 $t$ 实际监督原序列 $t+1$，因此：

$$
m_t=\mathbf1[t+1\ge a].
$$

第一个回答 token 的监督位置是 $a-1$，对应最后一个提示 token 的隐状态。若从 $a$ 才开始启用 label，就漏掉了回答的第一个 token。

数值例子：原序列为 `[BOS, U, :, A, B, EOS]`，A 是回答起点，索引 3。移位标签为 `[U,:,A,B,EOS]`，监督遮罩为 `[0,0,1,1,1]`。有效目标有 3 个。

prompt token 的标签被忽略，但它们仍是输入，回答 loss 的梯度仍可能传入 prompt 的 embedding 和前序 hidden state。标签遮罩不等于阻断上下文梯度。

## 运行第二阶段

先完成 200 步预训练，再开始新的 SFT optimizer：

```bash
python minillm.py train --out runs/base --steps 200
python minillm.py train --sft --data demo-sft.jsonl --init-from runs/base/last.pt --out runs/sft --steps 200 --lr 0.001
python minillm.py generate --checkpoint runs/sft/last.pt --prompt-file demo-prompt.txt
```

`demo-prompt.txt` 包含真实换行，最后停在 `Assistant: ` 的空格后。使用提示词文件避免终端把 `\n` 当成两个普通字符；这是训练模板与生成模板需要一致的一个小细节。

`--init-from` 载入预训练结构，所以这里的 block size 仍为 64。实验拒绝超过上下文的 SFT 样本，避免静默截掉回答。若需要更长样本，应先用更大的窗口重新做预训练；改变可学习位置表大小不能靠修改一个 CLI 参数完成恢复。

## 多轮、packing 与加权

多轮对话可监督所有 assistant 回答，或只监督最后一轮；两种目标不同。工具调用也需要明确监督哪些角色、哪些 JSON 字段和哪些结束标记。若回答被截断，不能照常添加代表“回答完整结束”的 EOS。

按 token 平均会让长回答贡献更多监督；按样本平均会让每个样本权重接近一致。都可能有合理用途，但代码和描述必须一致。本实验选择按有效回答 token 平均。

多个对话 packing 到同一序列时，需要决定是否允许跨对话 Attention。不能只把 prompt labels 设为 `-100` 就认为独立样本已经隔离。

## 实测：会模仿不等于会泛化

默认小模型预训练 200 步后，以 `--lr 0.001` 在 13 组训练问答上继续 SFT 200 步，另留 3 组问答验证。固定验证集的平均损失从约 3.026 上升到 4.394，而第 200 步当前训练批次的平均损失约为 0.436。

这是过拟合的具体例子：优化器成功拟合了非常小的训练集，但未见回答的概率预测变差了。验证集也只有 38 个回答目标 token，数字对样本选择敏感，不能作为通用 SFT 方法的质量比较。

可将较早的检查点、更多示范、更小学习率或正则化作为后续控制实验，但要使用新的独立测试集确认选择后的泛化。不要通过挑最好的一条生成隐藏这条验证曲线。完整数据与曲线见[实验档案](../lab/)。

## 检查与练习

集成测试用 `Hi → Hello`，确认只监督 `Hello` 的 5 个字节和 EOS，且第一项标签正好是 `H`。再检查 prompt 之前的 label 全为 `-100`，尾部 padding 同样不参与损失。

练习：将一个空回答样本加入数据。当前读取器应拒绝它；如果产品需要允许空回答，就必须明确是否只监督 EOS，以及如何评估这种样本。实现决定与数据约定应一同修改。
