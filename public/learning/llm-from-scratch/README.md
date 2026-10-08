# 从零训练 LLM：实验包

从随机参数训练的字节级 Decoder Transformer。对应博客“学习 / LLM / 从零训练 LLM”，包括数学检查、训练、验证、检查点、精确续训、生成、回答区域 SFT、双进程梯度和策略梯度实验。

## 安装

Python 3.10–3.13，建议新建隔离环境。`requirements-cpu.txt` 固定 PyTorch 2.8.0 与 NumPy 2.2.6，使用 PyPI 与官方 CPU 索引：

```bash
python -m venv .venv
# PowerShell：.venv\Scripts\Activate.ps1
# macOS/Linux：source .venv/bin/activate
python -m pip install -r requirements-cpu.txt
```

PowerShell 无法激活时，可直接用 `.venv\Scripts\python.exe` 执行。若官方 CPU 索引下载失败，可用 `python -m pip install torch==2.8.0 numpy==2.2.6 --index-url https://pypi.org/simple` 安装同版本 PyPI 发行包，再检查 `torch.cuda.is_available()`。安装包依赖和体积可能不同，实验始终用 `--device cpu` 明确选择设备。GPU 安装请使用 https://pytorch.org/get-started/locally/ 匹配驱动；不要把 CPU 依赖文件用于 GPU 环境。

## 全部检查

```bash
python math_checks.py
python smoke_test.py
python distributed_check.py
python rl_bandit.py
```

- 数学：float64 CE、线性层、精确 GELU 导数、Attention 前后向与有限差分、LayerNorm、token 加权累积、AdamW、RoPE、DPO。
- 集成：未来扰动不影响先前 logits、共享参数、SFT 起点、6 步连续训练与 3+3 恢复的参数/优化器/RNG 一致、错误恢复约定被拒绝、验证改善、生成和新 SFT 阶段。
- 分布式：默认启动两个真实 CPU 进程，以临时文件交换梯度并验证全局 token 平均，同时证明 rank 均值平均的错误；默认路径没有使用 DDP/AllReduce。可选 `--backend gloo` 需要可用 Gloo transport。本机 Windows 发行包的 Gloo 初始化失败，实际通过的路径为文件交换；不是多卡 benchmark。
- RL：四动作 REINFORCE，与精确期望梯度对照；不是 LLM PPO/GRPO。

## 预训练、生成与 SFT

```bash
python minillm.py train --data demo.jsonl --out runs/base --steps 200
python minillm.py generate --checkpoint runs/base/last.pt --prompt "Learning " --max-tokens 80
python minillm.py train --sft --data demo-sft.jsonl --init-from runs/base/last.pt --out runs/sft --steps 200 --lr 0.001
python minillm.py generate --checkpoint runs/sft/last.pt --prompt-file demo-prompt.txt --max-tokens 40
```

`demo-prompt.txt` 保留真实换行，末行以 `Assistant: ` 的空格结束，没有额外末尾换行。新阶段只加载模型权重和结构，optimizer/step 重新开始。随附语料是 32 篇原创短文与 16 组短问答，只用于教学，不能训练出实用助手。

每次训练输出 `last.pt` 和 `report.json`。默认配置：上下文 64、宽度 64、2 层、4 头、dropout 0.1、batch size 8、种子 17、CPU 单线程、AdamW、20 步 warmup、1000 步余弦终点、按有效 token 平均。词表为 256 字节加 BOS/EOS/PAD；共享输入输出表，独立参数 120,768。

`experiment_results.json` 是真实 CPU 运行的档案：Windows、Python 3.13.11、PyTorch 2.8.0+cpu，200 步预训练验证 loss 5.5502 → 2.5876；随后 200 步 SFT 验证 loss 3.0263 → 4.3940，显示小问答集过拟合。固定生成样本原样保留，不宣称语言流畅或回答可靠。`validation-curves.svg` / `.png` 展示原始指标；训练线是当前批次，验证线是固定完整验证集。

## 恢复相同实验

```bash
python minillm.py train --out runs/full --steps 200
python minillm.py train --out runs/part --steps 100
python minillm.py train --out runs/resumed --steps 200 --resume runs/part/last.pt
```

`--steps` 为总目标，不是额外步数。恢复时保留数据、batch、累积、种子、学习率、衰减、裁剪与调度。不要修改 `--decay-steps`。相同 CPU/框架/代码环境上的集成测试要求逐元素一致；跨设备与版本不承诺位级一致。

只在完整 optimizer step 后保存；未完成更新的梯度不保留。修改文件导致 SHA-256 改变会拒绝恢复。要换数据开始新阶段，应使用 `--init-from`。

## 限制与扩展

没有 BPE、RoPE、RMSNorm、SwiGLU、FlashAttention、KV cache、MoE 或多 GPU 模型训练；教材逐项解释升级路线。生成重算最近窗口并重新编号位置，UTF-8 字节采样可能出现替代字符，不能据此宣称长上下文或中文能力。

SFT 样本必须完整放进上下文，超长就报错；prompt、padding 不监督，回答和 EOS 监督。默认验证约占独立文档的 20%，精确去重不等于近似去重或评测污染审计。

CUDA 路径可用 `--device cuda`；支持 BF16 的设备可加 `--amp bf16`。模型和 optimizer 状态仍为 FP32；不提供 FP16 GradScaler。发布实测以 CPU 为准，GPU 速度与显存需自行测量。

## 使用约定

本目录 Python 教学代码遵循随附 MIT 许可；`demo.jsonl`、`demo-sft.jsonl` 与 `demo-prompt.txt` 是本专题原创的教学数据，按 CC0 1.0 提供：https://creativecommons.org/publicdomain/zero/1.0/ 。下载包不包含用户数据、预训练权重或 Python 环境。
