# Vendor Library Usage Verification (cuBLAS / cuDNN)

Date: 2025-09-24

This document explains how to rigorously verify that Alpaka tensor benchmarks and inference executables (e.g. `inferenceLeNet`, `inferenceResNet`, `inferenceBERT`) actually *use* NVIDIA vendor libraries (cuBLAS, cuBLASLt, cuDNN) rather than only linking against them.

---
## 1. Levels of Evidence
| Level | Goal | Tools / Method | Strength |
|-------|------|----------------|----------|
| L1 | Binary links to vendor libs | `ldd`, `nm -D`, `readelf -d` | Basic (potential, not proof of execution) |
| L2 | Symbols referenced in code paths | `nm -D | grep cudnnConvolutionForward` | Moderate |
| L3 | Runtime provider selection logs | Built-in verbose logging (single unified tracing) | Strong (dispatch chosen) |
| L4 | GPU activity correlated with call site | `nvidia-smi dmon`, Nsight Systems trace | Strong |
| L5 | API-level profiling / kernel attribution | Nsight Systems / Nsight Compute / CUPTI counters | Definitive |

Use at least L1 + L3 for quick checks; add L4 or L5 when auditing performance regressions.

---
## 2. L1: Dynamic Linkage Check
Verify dynamic dependencies:
```
ldd build/benchmark/tensor/inferenceLeNet/cpp/inferenceLeNet | egrep "cublas|cudnn"
ldd build/benchmark/tensor/inferenceResNet/cpp/inferenceResNet | egrep "cublas|cudnn"
```
Expected lines (examples):
```
libcublas.so.12 => /usr/local/cuda/.../libcublas.so.12
libcublasLt.so.12 => /usr/local/cuda/.../libcublasLt.so.12
libcudnn.so.9 => /lib/x86_64-linux-gnu/libcudnn.so.9
```
Optional deeper inspection:
```
readelf -d build/benchmark/tensor/inferenceLeNet/cpp/inferenceLeNet | grep NEEDED
nm -D build/benchmark/tensor/inferenceLeNet/cpp/inferenceLeNet | grep cudnnConvolutionForward
```

---
## 3. L2: Symbol Presence
List imported vendor API symbols actually referenced:
```
nm -D build/benchmark/tensor/inferenceResNet/cpp/inferenceResNet | grep -E 'cublas(Sgemm|GemmEx|Lt)|cudnnConvolutionForward'
```
If no symbols appear, either they are resolved via lazy loading wrappers or the code path is not compiled in; re-check CMake options.

---
## 4. L3: Runtime Provider Dispatch Logging
Run executables in GPU-only mode with verbose tracing (example flags may differ; replace `--only-gpu` / `--verbose` with your actual CLI):
```
ALPAKA_TENSOR_TRACE=1 build/benchmark/tensor/inferenceLeNet/cpp/inferenceLeNet --only-gpu --batch 2 --iters 1 --verbose
```
Sample (captured earlier):
```
Active providers: GEMM: cuBLAS (CUDA) Conv2D: cuDNN BatchNorm: cuDNN Activation: cuDNN Pooling: cuDNN
GEMM: Using cuBLAS backend (M=2 N=120 K=400)
Conv2D: Using alpaka 4D kernel for backend acceleration (heuristic small-size path)
```
Interpretation:
- Provider registry selected cuDNN & cuBLAS.
- A shape-specific heuristic still used the internal tiled kernel for this small conv (not a fallback failure).

For ResNet (larger spatial dims) you should observe a mix; if still internal only, raise the FLOP threshold forcing cuDNN usage.

### Recommended Enhancements (future)
- Add per-provider counters: `cudnn_conv_calls`, `generic_conv_calls` printed at program end.
- Distinguish: `Conv2D: cuDNN forward (algo=IMPLICIT_PRECOMP_GEMM, time=0.72 ms)` vs `Conv2D: internal_tiled`.

---
## 5. L4: Correlating GPU Activity (nvidia-smi)
While a run is active:
```
watch -n 0.5 nvidia-smi
```
Or capture utilization & memory timeline:
```
nvidia-smi dmon -s pucvmt -d 1 -o DT > dmon.log &
PID=$!
ALPAKA_TENSOR_TRACE=1 build/benchmark/tensor/inferenceResNet/cpp/inferenceResNet --only-gpu --batch 8 --iters 5
kill $PID
```
Review `dmon.log` for spikes in SM (`p` column) and memory throughput aligning with logged cuBLAS/cuDNN calls.

---
## 6. L5: Nsight Systems / Compute
### Systems (API timeline)
```
nsys profile -o resnet_profile \
  build/benchmark/tensor/inferenceResNet/cpp/inferenceResNet --only-gpu --batch 8 --iters 10
```
Inspect timeline for:
- cuBLAS API calls (e.g., `cublasLtMatmul`, `cublasGemmEx`).
- cuDNN API calls (e.g., `cudnnConvolutionForward`, `cudnnBatchNormalizationForwardInference`).
- Kernel names containing `void cudnn::` or `cublasLtMatmulKernel`.

### Compute (kernel-level performance)
Focus on one GEMM or convolution:
```
nsys launch --trace=cuda,osrt \
  build/benchmark/tensor/inferenceLeNet/cpp/inferenceLeNet --only-gpu --batch 32 --iters 10
# Or:
nsight-cu-cli --target-processes all --kernel-name-base demangled \
  build/benchmark/tensor/inferenceResNet/cpp/inferenceResNet --only-gpu --batch 8 --iters 5
```
Look for kernel attribution to cuDNN or cuBLAS in the CLI summary.

---
## 7. Distinguishing Link vs Use
| Symptom | Link? | Use? | Action |
|---------|-------|------|--------|
| ldd shows libs, logs show only generic kernels | Yes | Probably no | Raise batch/size; inspect heuristics; add counters |
| ldd + provider logs show cuBLAS/cuDNN lines | Yes | Likely yes | Optionally confirm with Nsight (spot check) |
| Nsight shows cuDNN kernels & API events | Yes | Confirmed | Done |
| No ldd entries | No | No | Reconfigure CMake with CUDA + cuDNN dev packages |

---
## 8. Forcing / Testing Fallback Paths (Clean Approach)
Instead of many environment flags, prefer a centralized runtime config (planned refactor). Until then, a temporary approach:
- Large batch & spatial dims trigger cuDNN automatically.
- Small shapes: expect internal kernels (documented).

Post-refactor policies (illustrative):
```
RuntimeConfig cfg;
cfg.conv = ConvProviderPolicy::ForceCuDNN;
initializeRuntimeConfig(cfg);
```
Then re-run and verify counts.

---
## 9. Adding Lightweight In-Code Proof (Optional)
Insert a single log site in each vendor dispatch wrapper (guarded by a tracing macro):
```
ALPAKA_TENSOR_TRACE("cuDNN conv forward: N=%d C=%d H=%d W=%d k=%d stride=%d", N,C,H,W,k,stride);
```
Aggregate counters at end:
```
[provider-summary] cudnn_conv_calls=12 generic_conv_calls=3 cublas_gemm_calls=27
```
This is minimally invasive and unambiguously shows usage without external tools.

---
## 10. Common Pitfalls
| Issue | Cause | Fix |
|-------|-------|-----|
| Linked but never used | Shape heuristic threshold too high | Lower FLOP threshold / force policy |
| cuDNN missing at runtime | Library path not exported | Add to `LD_LIBRARY_PATH` or install dev package |
| Nsight shows no vendor APIs | Optimized small test (too fast) | Increase problem size / batch |
| Mixed kernels expected but only generic appear | Unsupported stride/padding/data type | Add alternate cuDNN algorithm path or fallback explanation |

---
## 11. Minimal Verification Recipe (Fast Path)
1. `ldd` each binary → confirm vendor libs present.
2. Run with large-enough shapes & tracing → observe `GEMM: Using cuBLAS backend`, `Conv2D: cuDNN forward ...` (after enhancements).
3. (Optional) One Nsight Systems run on a representative workload.

If all three pass, vendor library usage is validated.

---
## 12. Next Enhancements (Action Items)
- Implement centralized runtime config (replace scattered env flags).
- Add provider usage counters + summary line.
- Provide a `--dump-config` CLI that prints current selection policies.
- Add a benchmark mode that runs same layer twice: generic vs provider, and reports speedup.

---
*End of VendorLibraryVerification.md*
