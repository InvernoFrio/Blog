"""A small, inspectable language-model training lab. No pretrained weights.

Python 3.10+, PyTorch 2.8.0 tested on CPU. See the accompanying README.
Byte tokens, document-level split, manual causal attention, token-weighted
accumulation, AdamW, evaluation, checkpoints, resume, generation, and masked SFT.
"""
import argparse
from contextlib import nullcontext
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import platform
import random
import sys
import time
import unicodedata

import torch
from torch import nn
from torch.nn import functional as F

BOS, EOS, PAD, VOCAB = 256, 257, 258, 259


@dataclass
class Config:
    block_size: int = 64
    width: int = 64
    heads: int = 4
    layers: int = 2
    dropout: float = 0.1

    def validate(self):
        if min(self.block_size, self.width, self.heads, self.layers) < 1 or self.width % self.heads:
            raise ValueError("Positive sizes and width divisible by heads are required")
        if not 0 <= self.dropout < 1:
            raise ValueError("Dropout must be in [0, 1)")


class Attention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.heads
        self.qkv = nn.Linear(config.width, 3 * config.width)
        self.projection = nn.Linear(config.width, config.width)
        self.attention_dropout = nn.Dropout(config.dropout)
        self.output_dropout = nn.Dropout(config.dropout)
        self.register_buffer("causal", torch.ones(config.block_size, config.block_size, dtype=torch.bool).tril(), persistent=False)

    def forward(self, x):
        batch, length, width = x.shape
        qkv = self.qkv(x).view(batch, length, 3, self.heads, width // self.heads)
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        scores = (q @ k.transpose(-2, -1)) / math.sqrt(width // self.heads)
        scores = scores.masked_fill(~self.causal[:length, :length], -float("inf"))
        weights = self.attention_dropout(torch.softmax(scores, dim=-1))
        context = (weights @ v).transpose(1, 2).contiguous().view(batch, length, width)
        return self.output_dropout(self.projection(context))


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm_attention = nn.LayerNorm(config.width)
        self.attention = Attention(config)
        self.norm_mlp = nn.LayerNorm(config.width)
        self.mlp = nn.Sequential(nn.Linear(config.width, 4 * config.width), nn.GELU(),
                                 nn.Linear(4 * config.width, config.width), nn.Dropout(config.dropout))

    def forward(self, x):
        x = x + self.attention(self.norm_attention(x))
        return x + self.mlp(self.norm_mlp(x))


class MiniLM(nn.Module):
    def __init__(self, config):
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(VOCAB, config.width)
        self.position_embedding = nn.Embedding(config.block_size, config.width)
        self.dropout = nn.Dropout(config.dropout)
        self.blocks = nn.Sequential(*(Block(config) for _ in range(config.layers)))
        self.final_norm = nn.LayerNorm(config.width)
        self.head = nn.Linear(config.width, VOCAB, bias=False)
        self.apply(self._initialize)
        self.head.weight = self.token_embedding.weight

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, ids):
        if ids.ndim != 2 or not 0 < ids.shape[1] <= self.config.block_size:
            raise ValueError("Expected token IDs shaped [batch, 1..block_size]")
        positions = torch.arange(ids.shape[1], device=ids.device)
        x = self.dropout(self.token_embedding(ids) + self.position_embedding(positions))
        return self.head(self.final_norm(self.blocks(x)))


def encode(text):
    return list(text.encode("utf-8"))


def load_documents(filename, sft=False):
    raw = Path(filename).read_bytes()
    documents, seen = [], set()
    for line_number, line in enumerate(raw.decode("utf-8").splitlines(), 1):
        if not line.strip():
            continue
        record = json.loads(line)
        fields = ("prompt", "response") if sft else ("text",)
        if any(not isinstance(record.get(field), str) or not record[field].strip() for field in fields):
            raise ValueError(f"Line {line_number}: nonempty string fields {fields} required")
        normalized = {field: unicodedata.normalize("NFC", record[field]).strip() for field in fields}
        key = json.dumps(normalized, sort_keys=True, ensure_ascii=False)
        if key not in seen:
            documents.append(normalized)
            seen.add(key)
    if len(documents) < 3:
        raise ValueError("At least three unique documents are required")
    return documents, hashlib.sha256(raw).hexdigest()


def split_documents(documents, seed):
    shuffled = list(documents)
    random.Random(seed).shuffle(shuffled)
    validation_size = max(1, len(shuffled) // 5)
    return shuffled[validation_size:], shuffled[:validation_size]


def make_windows(documents, block_size, sft=False):
    inputs, targets = [], []
    for document in documents:
        if sft:
            prefix = [BOS] + encode("User: " + document["prompt"] + "\nAssistant: ")
            sequence = prefix + encode(document["response"]) + [EOS]
            response_start = len(prefix)
            if len(sequence) > block_size + 1:
                raise ValueError("SFT example exceeds context: shorten it or increase block_size")
        else:
            sequence = [BOS] + encode(document["text"]) + [EOS]
            response_start = 1
        for start in range(0, len(sequence) - 1, block_size):
            chunk = sequence[start:start + block_size + 1]
            x = chunk[:-1] + [PAD] * (block_size - len(chunk) + 1)
            y = [token if start + offset + 1 >= response_start else -100
                 for offset, token in enumerate(chunk[1:])]
            y += [-100] * (block_size - len(y))
            if any(token != -100 for token in y):
                inputs.append(x)
                targets.append(y)
    if not inputs:
        raise ValueError("No supervised targets remain")
    return torch.tensor(inputs, dtype=torch.long), torch.tensor(targets, dtype=torch.long)


def optimizer_for(model, learning_rate, weight_decay):
    decay, no_decay = [], []
    for parameter in model.parameters():
        (decay if parameter.ndim >= 2 else no_decay).append(parameter)
    return torch.optim.AdamW([{"params": decay, "weight_decay": weight_decay},
                             {"params": no_decay, "weight_decay": 0.0}],
                            lr=learning_rate, betas=(0.9, 0.95), eps=1e-8, foreach=False)


def learning_rate_at(step, base, warmup, decay_steps):
    if warmup and step <= warmup:
        return base * step / warmup
    fraction = min(1.0, max(0.0, (step - warmup) / max(1, decay_steps - warmup)))
    return base * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * fraction)))


@torch.no_grad()
def evaluate(model, dataset, device, batch_size=8):
    was_training = model.training
    model.eval()
    total_loss, total_tokens = 0.0, 0
    for start in range(0, dataset[0].shape[0], batch_size):
        x, y = (tensor[start:start + batch_size].to(device) for tensor in dataset)
        total_loss += F.cross_entropy(model(x).float().flatten(0, 1), y.flatten(), ignore_index=-100, reduction="sum").item()
        total_tokens += (y != -100).sum().item()
    model.train(was_training)
    loss = total_loss / total_tokens
    return {"loss": loss, "perplexity": math.exp(loss), "tokens": total_tokens}


def atomic_save(state, destination):
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    torch.save(state, temporary)
    os.replace(temporary, destination)


def train(args):
    if min(args.steps, args.batch_size, args.accum_steps, args.threads, args.eval_every, args.save_every) < 1:
        raise ValueError("Steps, batch size, accumulation, thread count and intervals must be positive")
    if args.lr <= 0 or args.weight_decay < 0 or args.clip <= 0 or args.warmup < 0 or args.decay_steps <= args.warmup:
        raise ValueError("Invalid learning rate, decay, clipping, or schedule")
    if args.resume and args.init_from:
        raise ValueError("Use resume for continuation or init-from for a new stage, not both")
    torch.set_num_threads(args.threads)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")
    if args.amp == "bf16" and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise ValueError("BF16 autocast requires a supporting CUDA GPU")
    checkpoint_path = args.resume or args.init_from
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True) if checkpoint_path else None
    config = Config(**checkpoint["config"]) if checkpoint else Config(args.block_size, args.width, args.heads, args.layers, args.dropout)
    documents, digest = load_documents(args.data, args.sft)
    training_docs, validation_docs = split_documents(documents, args.seed)
    training = make_windows(training_docs, config.block_size, args.sft)
    validation = make_windows(validation_docs, config.block_size, args.sft)
    contract = {"data_sha256": digest, "sft": args.sft, "batch_size": args.batch_size,
                "accum_steps": args.accum_steps, "seed": args.seed, "lr": args.lr,
                "weight_decay": args.weight_decay, "warmup": args.warmup,
                "decay_steps": args.decay_steps, "clip": args.clip, "amp": args.amp}
    if args.resume and checkpoint["contract"] != contract:
        raise ValueError("Resume contract differs: preserve data, sampling and optimizer/schedule settings")
    model = MiniLM(config).to(device)
    optimizer = optimizer_for(model, args.lr, args.weight_decay)
    batch_rng = torch.Generator().manual_seed(args.seed + 1)
    step, seen_tokens = 0, 0
    if checkpoint:
        model.load_state_dict(checkpoint["model"])
    if args.resume:
        optimizer.load_state_dict(checkpoint["optimizer"])
        batch_rng.set_state(checkpoint["batch_rng"])
        torch.set_rng_state(checkpoint["torch_rng"])
        random.setstate(checkpoint["python_rng"])
        if device.type == "cuda" and checkpoint["cuda_rng"] is not None:
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng"])
        step, seen_tokens = checkpoint["step"], checkpoint["seen_tokens"]
    if step >= args.steps:
        raise ValueError("steps is the total target and must exceed the restored step")
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    initial_validation = evaluate(model, validation, device, args.batch_size)
    history = []
    started = time.perf_counter()
    model.train()
    while step < args.steps:
        optimizer.zero_grad(set_to_none=True)
        microbatches = []
        for _ in range(args.accum_steps):
            indices = torch.randint(training[0].shape[0], (args.batch_size,), generator=batch_rng)
            microbatches.append((training[0][indices], training[1][indices]))
        valid_tokens = sum((y != -100).sum().item() for _, y in microbatches)
        total_loss = 0.0
        for x_cpu, y_cpu in microbatches:
            x, y = x_cpu.to(device), y_cpu.to(device)
            context = torch.autocast("cuda", dtype=torch.bfloat16) if args.amp == "bf16" else nullcontext()
            with context:
                loss_sum = F.cross_entropy(model(x).float().flatten(0, 1), y.flatten(), ignore_index=-100, reduction="sum")
            if not torch.isfinite(loss_sum):
                raise FloatingPointError("Nonfinite loss: inspect data and numerical settings")
            (loss_sum / valid_tokens).backward()
            total_loss += loss_sum.item()
        gradient_norm = nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("Nonfinite gradient: the optimizer has not been updated")
        step += 1
        learning_rate = learning_rate_at(step, args.lr, args.warmup, args.decay_steps)
        for group in optimizer.param_groups:
            group["lr"] = learning_rate
        optimizer.step()
        seen_tokens += valid_tokens
        if step % args.eval_every == 0 or step == args.steps:
            record = {"step": step, "train_batch_loss": total_loss / valid_tokens,
                      "lr": learning_rate, "grad_norm_before_clip": gradient_norm.item(),
                      "seen_tokens": seen_tokens, "validation": evaluate(model, validation, device, args.batch_size)}
            history.append(record)
            print(json.dumps(record), flush=True)
        if step % args.save_every == 0 or step == args.steps:
            atomic_save({"schema": 1, "config": asdict(config), "contract": contract,
                         "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                         "step": step, "seen_tokens": seen_tokens, "batch_rng": batch_rng.get_state(),
                         "torch_rng": torch.get_rng_state(), "python_rng": random.getstate(),
                         "cuda_rng": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}, output / "last.pt")
    final_validation = evaluate(model, validation, device, args.batch_size)
    report = {"torch": str(torch.__version__), "python": platform.python_version(),
              "device": str(device), "config": asdict(config), "training_contract": contract,
              "code_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "parameters": sum(p.numel() for p in model.parameters()), "data_sha256": digest,
              "training_documents": len(training_docs), "validation_documents": len(validation_docs),
              "training_windows": training[0].shape[0], "validation_windows": validation[0].shape[0],
              "initial_validation": initial_validation, "final_validation": final_validation,
              "history": history, "total_step": step, "seen_tokens": seen_tokens,
              "elapsed_seconds": time.perf_counter() - started, "sft": args.sft}
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return model, report


@torch.no_grad()
def generate(model, prompt, max_tokens=80, temperature=0.8, top_k=40, seed=17):
    if max_tokens < 0 or temperature <= 0 or not 1 <= top_k <= 256:
        raise ValueError("Invalid generation settings")
    model.eval()
    device = next(model.parameters()).device
    ids = [BOS] + encode(prompt)
    generated = []
    generator = torch.Generator(device=device).manual_seed(seed)
    for _ in range(max_tokens):
        context = torch.tensor([ids[-model.config.block_size:]], device=device)
        logits = model(context)[0, -1].float() / temperature
        logits[BOS] = logits[PAD] = -float("inf")
        threshold = torch.topk(logits, top_k).values[-1]
        logits[logits < threshold] = -float("inf")
        token = torch.multinomial(torch.softmax(logits, dim=-1), 1, generator=generator).item()
        ids.append(token)
        if token == EOS:
            break
        generated.append(token)
    return prompt + bytes(generated).decode("utf-8", errors="replace")


def parser():
    root = argparse.ArgumentParser(description=__doc__)
    commands = root.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train")
    training.add_argument("--data", default=str(Path(__file__).with_name("demo.jsonl")))
    training.add_argument("--out", default="runs/minillm")
    training.add_argument("--steps", type=int, default=200)
    training.add_argument("--block-size", type=int, default=64)
    training.add_argument("--width", type=int, default=64)
    training.add_argument("--heads", type=int, default=4)
    training.add_argument("--layers", type=int, default=2)
    training.add_argument("--dropout", type=float, default=0.1)
    training.add_argument("--batch-size", type=int, default=8)
    training.add_argument("--accum-steps", type=int, default=1)
    training.add_argument("--lr", type=float, default=0.003)
    training.add_argument("--weight-decay", type=float, default=0.01)
    training.add_argument("--clip", type=float, default=1.0)
    training.add_argument("--warmup", type=int, default=20)
    training.add_argument("--decay-steps", type=int, default=1000)
    training.add_argument("--eval-every", type=int, default=50)
    training.add_argument("--save-every", type=int, default=50)
    training.add_argument("--seed", type=int, default=17)
    training.add_argument("--threads", type=int, default=1)
    training.add_argument("--device", default="cpu")
    training.add_argument("--amp", choices=("none", "bf16"), default="none")
    training.add_argument("--resume")
    training.add_argument("--init-from")
    training.add_argument("--sft", action="store_true")
    sampling = commands.add_parser("generate")
    sampling.add_argument("--checkpoint", required=True)
    prompts = sampling.add_mutually_exclusive_group()
    prompts.add_argument("--prompt")
    prompts.add_argument("--prompt-file", help="UTF-8 text file; preserves real newlines")
    sampling.add_argument("--max-tokens", type=int, default=80)
    sampling.add_argument("--temperature", type=float, default=0.8)
    sampling.add_argument("--top-k", type=int, default=40)
    sampling.add_argument("--seed", type=int, default=17)
    sampling.add_argument("--device", default="cpu")
    sampling.add_argument("--threads", type=int, default=1)
    return root


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    args = parser().parse_args()
    if args.command == "train":
        train(args)
    else:
        if args.threads < 1:
            raise ValueError("Thread count must be positive")
        torch.set_num_threads(args.threads)
        checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
        model = MiniLM(Config(**checkpoint["config"])).to(args.device)
        model.load_state_dict(checkpoint["model"])
        prompt = Path(args.prompt_file).read_text(encoding="utf-8") if args.prompt_file else (args.prompt if args.prompt is not None else "Learning ")
        print(generate(model, prompt, args.max_tokens, args.temperature, args.top_k, args.seed))


if __name__ == "__main__":
    main()
