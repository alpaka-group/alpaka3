#!/usr/bin/env python3
# -----------------------------------------------------------------------------
# PyTorch Conv2D Micro-Benchmark
#
# Quick start (GPU, fresh repo clone):
#   1) Create & activate venv
#        python3 -m venv .venv
#        source .venv/bin/activate
#   2) Install deps (torch pulls in CUDA build if a GPU + drivers are present)
#        pip install --upgrade pip
#        pip install torch numpy
#   3) Run a suite (results printed + JSON written)
#        # LeNet-sized layers (small shapes)
#        python benchmark/tensor/Conv2DBenchmark/python/pytorch_conv2d_benchmark.py \
#               --suite lenet --batch 4 --warmup 5 --iters 30 --output torch_lenet_cuda.json
#
#        # Common mid-size shapes
#        python benchmark/tensor/Conv2DBenchmark/python/pytorch_conv2d_benchmark.py \
#               --suite common --batch 4 --warmup 5 --iters 30 --output torch_common_cuda.json
#
#        # Large (compute-heavy) shapes matching C++ conv2d_bench large suite
#        python benchmark/tensor/Conv2DBenchmark/python/pytorch_conv2d_benchmark.py \
#               --suite large --batch 4 --warmup 5 --iters 30 --output torch_large_cuda.json
#
#        # All shapes (lenet + common + large)
#        python benchmark/tensor/Conv2DBenchmark/python/pytorch_conv2d_benchmark.py \
#               --suite all --batch 4 --warmup 5 --iters 30 --output torch_all_cuda.json
#
#   4) Force CPU (for reference / correctness checks)
#        python benchmark/tensor/Conv2DBenchmark/python/pytorch_conv2d_benchmark.py \
#               --suite large --batch 4 --iters 10 --warmup 3 --cpu --output torch_large_cpu.json
#
# Notes:
#   * --suite choices: lenet | common | large | all
#   * Increase --iters for more stable averages (default 200; reduced in examples for speed).
#   * TFLOPS is computed as:  (2 * Cin * K * K * N * Cout * H_out * W_out) / mean_ms / 1e9
#   * To compare with the C++ Alpaka benchmark, use the SAME batch, suite and shapes.
#   * This script sets cudnn.benchmark = True to allow cuDNN autotuning (similar to PyTorch default
#     when input sizes are stable). First few runs can be slower.
#   * For reproducibility you can disable autotune by exporting:  export CUDNN_BENCHMARK=0
#   * If torch selects CPU (e.g., no GPU), you will see 'Device: cpu'. Install a CUDA-enabled
#     PyTorch wheel or ensure driver/toolkit is present for GPU numbers.
#   * JSON schema: { "results": [ { name, batch, in_channels, out_channels, height, width,
#       kernel, stride, padding, mean_ms, min_ms, max_ms, tflops } ... ] }
#
# Optional deterministic mode (slower, disables some fast kernels):
#   export CUBLAS_WORKSPACE_CONFIG=:16:8  (for CUDA < 12 compatibility) and in code set
#   torch.use_deterministic_algorithms(True)
#
# -----------------------------------------------------------------------------
import torch, torch.nn as nn, torch.backends.cudnn as cudnn
import time, json, argparse, statistics, os
from dataclasses import dataclass, asdict

@dataclass
class Conv2DConfig:
    batch: int
    in_channels: int
    out_channels: int
    height: int
    width: int
    kernel_size: int
    stride: int = 1
    padding: int = 0
    name: str = ""

def lenet_configs(batch):
    return [
        Conv2DConfig(batch, 1, 6, 32, 32, 5, 1, 0, "LeNet_Conv1"),
        Conv2DConfig(batch, 6, 16, 14, 14, 5, 1, 0, "LeNet_Conv2"),
    ]

def common_configs(batch):
    return [
        Conv2DConfig(batch, 64, 64, 56, 56, 1, 1, 0, "Conv1x1_56x56"),
        Conv2DConfig(batch, 256, 256, 14, 14, 1, 1, 0, "Conv1x1_14x14"),
        Conv2DConfig(batch, 64, 64, 56, 56, 3, 1, 1, "Conv3x3_56x56"),
        Conv2DConfig(batch, 128, 128, 28, 28, 3, 1, 1, "Conv3x3_28x28"),
        Conv2DConfig(batch, 256, 256, 14, 14, 3, 1, 1, "Conv3x3_14x14"),
        Conv2DConfig(batch, 32, 64, 32, 32, 5, 1, 2, "Conv5x5_32x32"),
        Conv2DConfig(batch, 3, 64, 224, 224, 7, 2, 3, "Conv7x7_224x224"),
    ]

# Larger, more compute-heavy shapes mirroring C++ largeConfigs (names kept identical for easier comparison)
def large_configs(batch):
    return [
        Conv2DConfig(batch, 64, 128, 112, 112, 3, 1, 1, "Large_Conv3x3_112_C64_128"),
        Conv2DConfig(batch, 128, 256, 56, 56, 3, 1, 1, "Large_Conv3x3_56_C128_256"),
        Conv2DConfig(batch, 256, 512, 28, 28, 3, 1, 1, "Large_Conv3x3_28_C256_512"),
        Conv2DConfig(batch, 512, 512, 14, 14, 3, 1, 1, "Large_Conv3x3_14_C512_512"),
        Conv2DConfig(batch, 512, 1024, 14, 14, 3, 1, 1, "Large_Conv3x3_14_C512_1024"),
        Conv2DConfig(batch, 1024, 1024, 7, 7, 3, 1, 1, "Large_Conv3x3_7_C1024_1024"),
        Conv2DConfig(batch, 256, 512, 56, 56, 1, 1, 0, "Large_Conv1x1_56_C256_512"),
        Conv2DConfig(batch, 64, 128, 224, 224, 7, 2, 3, "Large_Conv7x7_224_C64_128"),
    ]

def calc_flops(cfg: Conv2DConfig):
    out_h = (cfg.height + 2*cfg.padding - cfg.kernel_size)//cfg.stride + 1
    out_w = (cfg.width + 2*cfg.padding - cfg.kernel_size)//cfg.stride + 1
    macs_per_out = cfg.kernel_size * cfg.kernel_size * cfg.in_channels
    total_out = cfg.batch * cfg.out_channels * out_h * out_w
    return 2.0 * macs_per_out * total_out

def run(cfg: Conv2DConfig, warmup: int, iters: int, device: torch.device):
    conv = nn.Conv2d(cfg.in_channels, cfg.out_channels, cfg.kernel_size,
                     stride=cfg.stride, padding=cfg.padding, bias=False).to(device)
    x = torch.randn(cfg.batch, cfg.in_channels, cfg.height, cfg.width, device=device)
    cudnn.benchmark = True

    # warmup
    for _ in range(warmup):
        with torch.no_grad():
            _ = conv(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        if device.type == 'cuda':
            torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = conv(x)
        if device.type == 'cuda':
            torch.cuda.synchronize()
        end = time.perf_counter()
        times.append((end - start)*1000.0)

    mean_ms = statistics.mean(times)
    flops = calc_flops(cfg)
    tflops = (flops / mean_ms) / 1e9
    return {
        'name': cfg.name,
        'batch': cfg.batch,
        'in_channels': cfg.in_channels,
        'out_channels': cfg.out_channels,
        'height': cfg.height,
        'width': cfg.width,
        'kernel': cfg.kernel_size,
        'stride': cfg.stride,
        'padding': cfg.padding,
        'mean_ms': mean_ms,
        'min_ms': min(times),
        'max_ms': max(times),
        'tflops': tflops
    }

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--batch', type=int, default=1)
    p.add_argument('--warmup', type=int, default=20)
    p.add_argument('--iters', type=int, default=200)
    p.add_argument('--suite', choices=['lenet','common','large','all'], default='lenet')
    p.add_argument('--output', '--out', dest='output', default='pytorch_conv2d.json')
    p.add_argument('--cpu', action='store_true')
    args = p.parse_args()

    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available() else 'cuda')
    print(f"Device: {device}")

    cfgs = []
    if args.suite in ('lenet','all'):
        cfgs += lenet_configs(args.batch)
    if args.suite in ('common','all'):
        cfgs += common_configs(args.batch)
    if args.suite in ('large','all'):
        cfgs += large_configs(args.batch)

    results = []
    for c in cfgs:
        print(f"Running {c.name} ...")
        r = run(c, args.warmup, args.iters, device)
        print(f"  Mean {r['mean_ms']:.3f} ms  TFLOPS {r['tflops']:.3f}")
        results.append(r)

    with open(args.output, 'w') as f:
        json.dump({'results': results}, f, indent=2)
    print(f"Saved to {args.output}")
