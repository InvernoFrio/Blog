"""Causal attention and KV reuse lab; Python standard library, no GPU needed.

This is a random single-layer, single-head attention toy, not a trained LLM.
CPU wall-clock timings include Python overhead. Logical cache byte estimates
describe hypothetical packed tensors, not the memory used by Python lists.
"""

import argparse
import json
import math
import platform
import random
import statistics
import time


def project(row, weight):
    return [sum(x * w for x, w in zip(row, col)) for col in zip(*weight)]


def attend(query, keys, values):
    scores = [sum(q * k for q, k in zip(query, key)) / math.sqrt(len(query))
              for key in keys]
    peak = max(scores)
    exp_scores = [math.exp(score - peak) for score in scores]
    total = sum(exp_scores)
    probabilities = [score / total for score in exp_scores]
    return [sum(p * value[j] for p, value in zip(probabilities, values))
            for j in range(len(query))]


class AttentionToy:
    def __init__(self, width=16, vocab=32, seed=7):
        self.width = width
        self.vocab = vocab
        rng = random.Random(seed)

        def matrix(rows, cols):
            return [[rng.uniform(-0.5, 0.5) for _ in range(cols)]
                    for _ in range(rows)]

        self.embedding = matrix(vocab, width)
        self.wq = matrix(width, width)
        self.wk = matrix(width, width)
        self.wv = matrix(width, width)
        self.head = matrix(width, vocab)

    def qkv(self, token, position):
        # Keep absolute positions identical in full and incremental execution.
        row = [x + 0.1 * math.sin(position / (j + 1))
               for j, x in enumerate(self.embedding[token])]
        return tuple(project(row, weight)
                     for weight in (self.wq, self.wk, self.wv))

    def full(self, tokens):
        if not tokens:
            raise ValueError("A nonempty prompt is required")
        triples = [self.qkv(token, i) for i, token in enumerate(tokens)]
        queries, keys, values = zip(*triples)
        # Reference computes all causal rows, including historical queries.
        return [project(attend(query, keys[:i + 1], values[:i + 1]), self.head)
                for i, query in enumerate(queries)]

    def append(self, token, cache):
        keys, values = cache
        query, key, value = self.qkv(token, len(keys))
        keys.append(key)
        values.append(value)
        return project(attend(query, keys, values), self.head)

    def prefill(self, tokens):
        if not tokens:
            raise ValueError("A nonempty prompt is required")
        # Build prompt KV once. The toy only needs the last query's logits.
        triples = [self.qkv(token, i) for i, token in enumerate(tokens)]
        queries, keys, values = zip(*triples)
        cache = (list(keys), list(values))
        return project(attend(queries[-1], keys, values), self.head), cache


def greedy(logits):
    return max(range(len(logits)), key=logits.__getitem__)


def generate(model, prompt, count, cached):
    tokens = list(prompt)
    if cached:
        logits, cache = model.prefill(tokens)
    for i in range(count):
        if not cached:
            logits = model.full(tokens)[-1]
        token = greedy(logits)
        tokens.append(token)
        if cached and i + 1 < count:
            logits = model.append(token, cache)
    return tokens[len(prompt):]


def kv_bytes(layers, kv_heads, head_dim, lengths, bytes_per_element):
    if min(layers, kv_heads, head_dim, bytes_per_element) <= 0:
        raise ValueError("Dimensions and element size must be positive")
    if any(length < 0 for length in lengths):
        raise ValueError("Sequence lengths must be nonnegative")
    return 2 * layers * kv_heads * head_dim * sum(lengths) * bytes_per_element


def check(model):
    error = 0.0
    # Full causal attention vs append-only KV at every position.
    for size in (1, 2, 7, 17):
        tokens = [(i * 11 + 3) % model.vocab for i in range(size)]
        expected = model.full(tokens)
        cache = ([], [])
        for token, reference in zip(tokens, expected):
            observed = model.append(token, cache)
            error = max(error, max(abs(a - b)
                                   for a, b in zip(observed, reference)))
        # Chunk boundary must preserve both cache contents and positions.
        split = max(1, size // 2)
        logits, chunk_cache = model.prefill(tokens[:split])
        for token in tokens[split:]:
            logits = model.append(token, chunk_cache)
        error = max(error, max(abs(a - b)
                               for a, b in zip(logits, expected[-1])))
        if len(cache[0]) != size or len(chunk_cache[0]) != size:
            raise AssertionError("KV cache must grow by exactly one row per token")
        if generate(model, tokens, 8, True) != generate(model, tokens, 8, False):
            raise AssertionError("Cached and full greedy generation disagree")
    if error > 1e-10:
        raise AssertionError(f"Logits differ: {error}")
    # Hand calculations from the lesson: GQA and MHA at 8192 tokens.
    if kv_bytes(32, 8, 128, [8192], 2) != 2**30:
        raise AssertionError("GQA memory example is incorrect")
    if kv_bytes(32, 32, 128, [8192], 2) != 4 * 2**30:
        raise AssertionError("MHA memory example is incorrect")
    if kv_bytes(32, 8, 128, [8192] * 16, 2) != 16 * 2**30:
        raise AssertionError("Concurrency memory example is incorrect")
    if kv_bytes(1, 1, model.width, [], 4) != 0:
        raise AssertionError("An empty cache must have zero logical bytes")
    try:
        model.prefill([])
    except ValueError:
        pass
    else:
        raise AssertionError("An empty prompt must be rejected")
    return {"passed": True, "max_logit_error": error,
            "cases": "per-position causal equivalence, chunk boundary, greedy generation, capacity examples, empty prompt"}


def measure(function, repeats):
    function()  # Warm-up outside the reported interval.
    samples = []
    for _ in range(repeats):
        start = time.perf_counter()
        function()
        samples.append((time.perf_counter() - start) * 1000)
    return {"median_ms": statistics.median(samples),
            "min_ms": min(samples), "max_ms": max(samples)}


def run(args):
    model = AttentionToy()
    report = {
        "environment": {"python": platform.python_version(),
                        "platform": platform.platform()},
        "scope": "CPU Python attention toy; wall-clock including list allocation; no trained model, GPU, scheduler or network",
        "model": {"layers": 1, "query_heads": 1, "kv_heads": 1,
                  "head_dim": model.width, "vocab": model.vocab, "seed": 7},
        "checks": check(model),
    }
    if not args.check_only:
        report["experiments"] = []
        for length in args.lengths:
            prompt = [(i * 11 + 3) % model.vocab for i in range(length)]
            full = measure(lambda: generate(model, prompt, args.tokens, False),
                           args.repeats)
            cached = measure(lambda: generate(model, prompt, args.tokens, True),
                             args.repeats)
            _, cache = model.prefill(prompt)
            for i in range(args.tokens - 1):
                model.append(i % model.vocab, cache)
            final_length = len(cache[0])
            logical_elements = sum(len(row) for rows in cache for row in rows)
            if logical_elements != 2 * model.width * final_length:
                raise AssertionError("Actual KV element count disagrees with formula")
            report["experiments"].append({
                "prompt_tokens": length, "output_tokens": args.tokens,
                "full_recompute": full, "cached": cached,
                "note": "Full reference recomputes all causal rows; toy prefill evaluates only the last query. Ratio is not a GPU or production speedup.",
                "cache_tokens_after_last_forward": final_length,
                "kv_elements": logical_elements,
                "logical_kv_bytes_if_fp32": kv_bytes(1, 1, model.width,
                                                    [final_length], 4),
            })
    return report


def positive(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return number


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--lengths", nargs="+", type=positive, default=[16, 64, 128])
    parser.add_argument("--tokens", type=positive, default=16)
    parser.add_argument("--repeats", type=positive, default=3)
    print(json.dumps(run(parser.parse_args()), indent=2, ensure_ascii=False))
