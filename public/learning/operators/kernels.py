"""Teaching kernels: contiguous tensors, finite inputs, NVIDIA CUDA + Triton.

These examples expose indexing and masks. They are not tuned production kernels.
"""
import torch
import triton
import triton.language as tl


def check_tensors(*tensors):
    first = tensors[0]
    if not first.is_cuda:
        raise ValueError("CUDA tensors are required")
    if any(t.device != first.device or t.dtype != first.dtype or not t.is_contiguous()
           for t in tensors):
        raise ValueError("Tensors must be contiguous and have the same CUDA device and dtype")


@triton.jit
def _add(X, Y, Z, N: tl.constexpr, TILE: tl.constexpr):
    index = tl.program_id(0) * TILE + tl.arange(0, TILE)
    valid = index < N
    left = tl.load(X + index, valid, other=0)
    right = tl.load(Y + index, valid, other=0)
    tl.store(Z + index, left + right, valid)


def add_into(x, y, out, tile=256):
    check_tensors(x, y, out)
    if x.ndim != 1 or x.shape != y.shape or x.shape != out.shape:
        raise ValueError("Expected vectors with identical shapes")
    if x.dtype not in (torch.float16, torch.float32):
        raise ValueError("Expected float16 or float32")
    if tile < 32 or tile > 4096 or tile & (tile - 1):
        raise ValueError("Tile must be a power of two between 32 and 4096")
    if x.numel():
        with torch.cuda.device(x.device):
            _add[(triton.cdiv(x.numel(), tile),)](x, y, out, x.numel(), tile)
    return out


@triton.jit
def _softmax(X, Y, COLS: tl.constexpr, TILE: tl.constexpr):
    row = tl.program_id(0)
    column = tl.arange(0, TILE)
    values = tl.load(X + row * COLS + column, column < COLS, other=-float("inf")).to(tl.float32)
    shifted = values - tl.max(values, axis=0)
    weights = tl.exp(shifted)
    probabilities = weights / tl.sum(weights, axis=0)
    tl.store(Y + row * COLS + column, probabilities, column < COLS)


def softmax_into(x, out):
    check_tensors(x, out)
    if x.ndim != 2 or out.shape != x.shape or not 0 < x.shape[1] <= 4096:
        raise ValueError("Expected matching matrices with 1..4096 columns")
    if x.dtype not in (torch.float16, torch.float32):
        raise ValueError("Expected float16 or float32")
    if x.shape[0]:
        with torch.cuda.device(x.device):
            _softmax[(x.shape[0],)](x, out, x.shape[1], triton.next_power_of_2(x.shape[1]), num_warps=4)
    return out


@triton.jit
def _gemm(A, B, C, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    rows = tl.program_id(0) * BM + tl.arange(0, BM)
    columns = tl.program_id(1) * BN + tl.arange(0, BN)
    inner = tl.arange(0, BK)
    accumulator = tl.zeros((BM, BN), dtype=tl.float32)
    for step in range(tl.cdiv(K, BK)):
        reduction = step * BK + inner
        a = tl.load(A + rows[:, None] * K + reduction[None, :],
                    (rows[:, None] < M) & (reduction[None, :] < K), other=0)
        b = tl.load(B + reduction[:, None] * N + columns[None, :],
                    (reduction[:, None] < K) & (columns[None, :] < N), other=0)
        accumulator = tl.dot(a, b, accumulator)
    tl.store(C + rows[:, None] * N + columns[None, :], accumulator,
             (rows[:, None] < M) & (columns[None, :] < N))


def matmul_into(a, b, out):
    check_tensors(a, b, out)
    if a.ndim != 2 or b.ndim != 2 or a.shape[1] != b.shape[0] or a.shape[1] == 0:
        raise ValueError("Expected A[M,K] and B[K,N] with K > 0")
    m, k, n = a.shape[0], a.shape[1], b.shape[1]
    if out.shape != (m, n) or a.dtype != torch.float16:
        raise ValueError("Expected float16 inputs and output[M,N]")
    if m and n:
        with torch.cuda.device(a.device):
            _gemm[(triton.cdiv(m, 16), triton.cdiv(n, 16))](a, b, out, m, n, k, 16, 16, 32, num_warps=4)
    return out


@triton.jit
def _bias_relu(X, BIAS, Y, SIZE: tl.constexpr, COLS: tl.constexpr, TILE: tl.constexpr):
    index = tl.program_id(0) * TILE + tl.arange(0, TILE)
    valid = index < SIZE
    values = tl.load(X + index, valid, other=0).to(tl.float32)
    bias = tl.load(BIAS + index % COLS, valid, other=0).to(tl.float32)
    tl.store(Y + index, tl.maximum(values + bias, 0), valid)


def bias_relu_into(x, bias, out):
    check_tensors(x, bias, out)
    if x.ndim != 2 or bias.ndim != 1 or bias.shape[0] != x.shape[1] or out.shape != x.shape:
        raise ValueError("Expected X[M,N], bias[N], and output[M,N]")
    if x.dtype not in (torch.float16, torch.float32):
        raise ValueError("Expected float16 or float32")
    if x.numel():
        with torch.cuda.device(x.device):
            _bias_relu[(triton.cdiv(x.numel(), 256),)](x, bias, out, x.numel(), x.shape[1], 256)
    return out
