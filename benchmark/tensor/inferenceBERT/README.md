# BERT Encoder Inference Benchmark

A minimal BERT encoder block benchmark using Alpaka tensor ops and provider-backed kernels (cuBLAS/cuDNN on CUDA).

## Features
- Self-contained C++ runner with CLI
- Auto-discovery of input data (.npy) under `cpp/data/`
- Sample data generation flag
- Provider summary and environment toggles printed at startup
- cuBLAS GEMM and cuDNN Activation (GELU) via CleanTensorOpContext
- attention_mask is applied during attention when provided

## Build
- Configure and build (Release):
  - cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
  - cmake --build build -j 8 --target inferenceBERT

## Run
- Quick GPU run:
  - ./build/benchmark/tensor/inferenceBERT/cpp/inferenceBERT --batch 2 --seq 8 --layers 1 --iters 5 --warmup 1 --timing --only-gpu --profile-layers
- Auto-detected inputs (if files exist under `cpp/data/`):
  - ./build/benchmark/tensor/inferenceBERT/cpp/inferenceBERT --iters 10 --timing
- Force CPU serial:
  - ./build/benchmark/tensor/inferenceBERT/cpp/inferenceBERT --only-serial --iters 5 --timing

## Data
Place `.npy` files in `benchmark/tensor/inferenceBERT/cpp/data/`:
- input_ids.npy: shape [batch, seq], int64 or int32
- attention_mask.npy: shape [batch, seq], int64 or int32 (optional; applied if provided)
- token_type_ids.npy: shape [batch, seq], int64 or int32 (optional; unused)

Generate sample data:
- ./build/benchmark/tensor/inferenceBERT/cpp/inferenceBERT --generate-sample-data --batch 2 --seq 8

## Environment toggles
- ALPAKA_DISABLE_CUBLAS=1, ALPAKA_DISABLE_CUDNN=1, ALPAKA_OPS_VERBOSE=1, ALPAKA_CONV_LOG=1, ALPAKA_BATCHNORM_LOG=1, ALPAKA_CONV_FIND=1, ALPAKA_DISABLE_TENSOR_CORES=1, ALPAKA_USE_FP16=1, ALPAKA_SOFTMAX_HOST=1, ALPAKA_LINEAR_INIT=0

## Notes
- The benchmark runs a simplified BERT encoder block: Embedding + PosEnc -> LN -> Attention -> Proj+Res -> LN -> FFN (GELU) -> Res
- Masked attention uses a large negative score bias on masked key positions before softmax; if the entire row is masked, the output is zero.
- For correctness/perf experiments, tune batch/seq/hidden/heads/layers via CLI.
