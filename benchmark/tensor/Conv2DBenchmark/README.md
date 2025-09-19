# Conv2D Benchmark Suite

This benchmark compares convolution performance across:

1. PyTorch CUDA backend (reference)
2. Alpaka TensorOpContext with provider (e.g. cuDNN/cuBLAS where available)
3. Alpaka fallback kernels (provider disabled)

## Structure
```
benchmark/tensor/Conv2DBenchmark/
  cpp/        # C++ Alpaka conv2d benchmark executable
  python/     # PyTorch reference scripts
  results/    # (ignored) store JSON benchmark outputs
```

## C++ Benchmark Features
- Runs specified Conv2D configurations
- Supports enabling/disabling provider (e.g. via `ALPAKA_DISABLE_CUDNN`)
- Reports: mean ms, TFLOPS, bandwidth estimate, output shape
- JSON output for comparison

## Python Benchmark Features
- Benchmarks equivalent conv2d layers using PyTorch
- Produces JSON matching C++ schema for easy diff

## Build C++ Benchmark
From project root (after normal build configuration):
```
cmake --build build --target conv2d_bench
```

## Run Examples
```
# PyTorch reference (batch 32 LeNet + common set)
python python/pytorch_conv2d_benchmark.py --batch 32 --suite all --iters 300 --print-summary \
  --output pytorch_all_b32.json

# Alpaka with provider
./conv2d_bench --suite lenet --batch 32 --iters 300 --json alpaka_provider_lenet_b32.json

# Alpaka fallback
ALPAKA_DISABLE_CUDNN=1 ./conv2d_bench --suite lenet --batch 32 --iters 300 --json alpaka_fallback_lenet_b32.json
```

## Compare
A comparison helper script will be added later (planned: `python/compare_conv2d.py`).

## Notes
- Ensure GPU clock stability (disable auto boost fluctuations when measuring)
- Run enough iterations for stable timing (>=200 typical)
- Warmup iterations default: 20
