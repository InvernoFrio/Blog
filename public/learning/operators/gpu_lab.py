"""Run correctness checks, then print real CUDA-event measurements as JSON.

Usage: python gpu_lab.py --check-only
       python gpu_lab.py > result.json
Run beside kernels.py in a compatible PyTorch + Triton environment.
"""
import argparse
import datetime
import json
import statistics
import torch
import triton
from kernels import add_into, softmax_into, matmul_into, bias_relu_into


def latency_ms(function, warmup=10, samples=15, iterations=30):
    for _ in range(warmup):
        function()
    torch.cuda.synchronize()
    timings = []
    for _ in range(samples):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            function()
        end.record()
        end.synchronize()
        timings.append(start.elapsed_time(end) / iterations)
    return {"median_ms": statistics.median(timings), "min_ms": min(timings), "max_ms": max(timings)}


def correctness():
    count = 0
    for dtype in (torch.float16, torch.float32):
        for n in (0, 1, 255, 256, 257, 1025):
            x, y = (torch.randn(n, device="cuda", dtype=dtype) for _ in range(2))
            out = torch.empty_like(x)
            add_into(x, y, out)
            torch.testing.assert_close(out, x + y, rtol=1e-3, atol=1e-3)
            count += 1
        for rows, cols in ((1, 1), (3, 17), (7, 1025)):
            x = torch.randn(rows, cols, device="cuda", dtype=dtype)
            out = torch.empty_like(x)
            softmax_into(x, out)
            torch.testing.assert_close(out, torch.softmax(x, dim=1), rtol=2e-3, atol=2e-3)
            torch.testing.assert_close(out.float().sum(1), torch.ones(rows, device="cuda"), rtol=2e-3, atol=2e-3)
            bias = torch.randn(cols, device="cuda", dtype=dtype)
            bias_relu_into(x, bias, out)
            torch.testing.assert_close(out, torch.relu(x + bias), rtol=1e-3, atol=1e-3)
            count += 2
    for m, n, k in ((1, 1, 1), (33, 47, 65), (64, 96, 128)):
        a = torch.randn(m, k, device="cuda", dtype=torch.float16)
        b = torch.randn(k, n, device="cuda", dtype=torch.float16)
        out = torch.empty((m, n), device="cuda", dtype=torch.float16)
        matmul_into(a, b, out)
        torch.testing.assert_close(out, a @ b, rtol=2e-2, atol=2e-2)
        count += 1
    return count


def benchmarks():
    measurements = []
    for n in (4096, 1048576):
        x, y = (torch.randn(n, device="cuda") for _ in range(2))
        out = torch.empty_like(x)
        for name, function in (("triton_add", lambda: add_into(x, y, out)),
                               ("torch_add", lambda: torch.add(x, y, out=out))):
            timing = latency_ms(function)
            measurements.append({"operator": name, "shape": [n], "dtype": "float32",
                                 **timing, "logical_GB_per_s": 3 * n * 4 / (timing["median_ms"] * 1e6)})
    for rows, cols in ((32, 1025), (256, 1025)):
        x = torch.randn(rows, cols, device="cuda")
        out, temporary = torch.empty_like(x), torch.empty_like(x)
        bias = torch.randn(cols, device="cuda")
        def unfused():
            torch.add(x, bias, out=temporary)
            torch.clamp_min(temporary, 0, out=out)
        for name, function in (("triton_softmax", lambda: softmax_into(x, out)),
                               ("torch_softmax", lambda: torch.softmax(x, dim=1, out=out)),
                               ("triton_bias_relu", lambda: bias_relu_into(x, bias, out)),
                               ("torch_bias_relu_unfused", unfused)):
            measurements.append({"operator": name, "shape": [rows, cols], "dtype": "float32", **latency_ms(function)})
    a = torch.randn(256, 256, device="cuda", dtype=torch.float16)
    b, out = torch.randn_like(a), torch.empty_like(a)
    for name, function in (("triton_gemm", lambda: matmul_into(a, b, out)),
                           ("torch_mm", lambda: torch.mm(a, b, out=out))):
        measurements.append({"operator": name, "shape": [256, 256, 256], "dtype": "float16", **latency_ms(function)})
    return measurements


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available() or torch.version.hip is not None:
        raise SystemExit("This lab targets an NVIDIA CUDA GPU with compatible PyTorch and Triton.")
    torch.manual_seed(17)
    report = {
        "date_utc": datetime.datetime.now(datetime.timezone.utc).isoformat(),
        "gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
        "triton": triton.__version__, "cuda_runtime": torch.version.cuda,
        "measurement": "CUDA events, default stream, warmed JIT, preallocated outputs, repeated resident inputs",
        "correctness_cases": correctness(),
    }
    report["measurements"] = [] if args.check_only else benchmarks()
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
