# Alpaka Tensor Library: ATen Compatibility and PyTorch Backend Roadmap

## Executive Summary

This document summarizes the analysis of Alpaka's tensor library against PyTorch's ATen operator set and proposes a roadmap for evolving Alpaka into a fully-capable ATen implementation and Python backend.

## Current State Analysis

### ATen Operator Coverage

#### ✅ **Well-Covered Operations** (Strong Foundation)
- **Linear Algebra**: GEMM, matrix multiplication via cuBLAS/rocBLAS with generic fallbacks
- **Convolution**: Conv2D forward/backward with cuDNN/MIOpen acceleration
- **Pooling**: Max/Avg pool2d operations
- **Normalization**: BatchNorm, LayerNorm implementations
- **Activations**: ReLU, GELU, Softmax with vendor-accelerated paths
- **Training Ops**: Backward passes for Conv2D, Linear, activation gradients

#### ⚠️ **Partially Covered Operations**
- **Element-wise**: Add implemented, but missing sub, mul, div, pow, exp, log, sqrt, trigonometric
- **Reductions**: Mean/var for LayerNorm exist, but no general reduction framework
- **Tensor Manipulation**: Flatten/copy helpers exist, but no general reshape/view/transpose

#### ❌ **Missing Operations**
- **Comparison Ops**: eq, ne, lt, gt, le, ge
- **Logical Ops**: and, or, xor, not
- **Indexing**: gather, scatter, index_select, masked operations
- **Advanced Linear Algebra**: SVD, eigenvalue, Cholesky decomposition
- **Random Generation**: rand, randn, uniform, normal distributions
- **Creation Ops**: zeros, ones, arange, linspace

### Architectural Comparison

| Feature | PyTorch Tensor | Alpaka Tensor | Gap Analysis |
|---------|---------------|---------------|--------------|
| **Rank** | Runtime dynamic | Compile-time fixed | Major difference |
| **Data Type** | Runtime (dtype) | Compile-time template | Major difference |
| **Device** | Runtime selection | Template parameter | Similar concept |
| **Views/Strides** | Full support | Partial | Needs enhancement |
| **Broadcasting** | Automatic | Manual | Missing feature |
| **Memory** | Shared storage with ref-counting | Direct ownership | Different model |
| **Shape Operations** | O(1) reshaping via views | Requires copy | Performance gap |

## Proposed Hybrid Architecture

### Core Design: Three-Layer Approach

```cpp
// Layer 1: Static Core (Current - Keep for Performance)
namespace alpaka::tensor::static_core {
    template<typename T, typename Device, int Rank>
    class StaticTensor {
        // Current efficient implementation
        // - Compile-time optimizations
        // - Direct memory access
        // - Zero-overhead abstractions
    };
}

// Layer 2: Dynamic Wrapper (New - For Flexibility)
namespace alpaka::tensor {
    class DynamicTensor {
    private:
        // Type-erased storage
        std::shared_ptr<Storage> storage_;
        Shape shape_;
        Stride strides_;
        ScalarType dtype_;
        Device device_;
        size_t offset_;
        
    public:
        // PyTorch-like API
        DynamicTensor view(IntArrayRef new_shape) const;
        DynamicTensor transpose(int64_t dim0, int64_t dim1) const;
        DynamicTensor squeeze(int64_t dim) const;
        DynamicTensor unsqueeze(int64_t dim) const;
        
        // Operator overloads with broadcasting
        DynamicTensor operator+(const DynamicTensor& other) const;
        DynamicTensor operator*(const DynamicTensor& other) const;
        
        // Device transfers
        DynamicTensor to(Device device) const;
        DynamicTensor to(ScalarType dtype) const;
    };
}

// Layer 3: ATen Operations (New - Standard Interface)
namespace alpaka::aten {
    // Standard ATen-compatible operations
    DynamicTensor add(const DynamicTensor& a, const DynamicTensor& b);
    DynamicTensor matmul(const DynamicTensor& a, const DynamicTensor& b);
    DynamicTensor conv2d(const DynamicTensor& input, const DynamicTensor& weight,
                         const DynamicTensor& bias, IntArrayRef stride,
                         IntArrayRef padding);
    
    // Dispatch to static kernels internally
    template<typename Dispatcher>
    void dispatch(const DynamicTensor& tensor, Dispatcher fn) {
        // Runtime to compile-time dispatch
        switch(tensor.dtype()) {
            case ScalarType::Float:
                dispatchRank<float>(tensor, fn);
                break;
            case ScalarType::Double:
                dispatchRank<double>(tensor, fn);
                break;
            // ...
        }
    }
}
```

### Storage and Memory Management

```cpp
class Storage {
private:
    std::shared_ptr<void> data_;
    size_t size_bytes_;
    Device device_;
    Allocator* allocator_;
    
public:
    // Reference-counted memory management
    Storage(size_t bytes, Device device);
    Storage share() const { return *this; }  // Shallow copy
    Storage clone() const;                   // Deep copy
    
    // Zero-copy views
    Storage slice(size_t offset, size_t size) const;
};
```

### Broadcasting System

```cpp
class BroadcastSpec {
    static BroadcastSpec compute(const Shape& a, const Shape& b);
    Shape output_shape;
    std::vector<int64_t> a_broadcast_dims;
    std::vector<int64_t> b_broadcast_dims;
};

template<typename Op>
DynamicTensor broadcast_binary_op(const DynamicTensor& a, 
                                  const DynamicTensor& b,
                                  Op operation) {
    auto spec = BroadcastSpec::compute(a.shape(), b.shape());
    // Dispatch to kernel with broadcasting iteration
}
```

## Implementation Roadmap (stabilization-first)

### Phase 0: Stabilization and Core Runtime (2–3 weeks)
1. Cleanup and DRY
   - Deduplicate kernels; keep a single canonical implementation per op family.
   - Enforce provider-first dispatch with clear fallbacks to Alpaka kernels.
   - Keep facades (InferenceOps.hpp, TrainingOps.hpp) thin and well-documented.
2. Shape/stride/view semantics (minimal)
   - Reshape, transpose, squeeze/unsqueeze with validation; metadata-only when possible.
   - Contiguity checks and helpers; no general broadcasting yet.
3. Reduction scaffolding
   - Define a minimal reduction API surface (sum/mean signatures) without broad kernel coverage.
4. Tests and docs
   - Ensure all existing examples/benchmarks/tests build and run; add smoke tests for views/contiguity.
   - Document error handling and unsupported-path behavior.

### Phase 1: Minimal ATen Bridge (add + matmul) (2–3 weeks)
1. Dynamic shim (narrow scope)
   - Introduce a tiny DynamicTensor wrapper sufficient to support add and matmul.
   - Runtime dtype/device and limited rank (1–4D) mapping to current static tensors.
2. ATen operator registration
   - Register aten.add and aten.matmul only, for float32/float64 on CPU/GPU where available.
   - Use a backend key (e.g., PrivateUse1 or custom) and provide meta kernels for shape inference.
3. Dispatch wiring
   - Route to provider-first paths (BLAS/cuBLAS/rocBLAS) with Alpaka fallbacks.
4. Validation
   - Reference tests against PyTorch for representative shapes/dtypes; CI for CPU and one GPU.
   - Package a minimal lib (e.g., libalpaka_aten.so) and document usage.

### Phase 2: Core Ops Expansion (3–5 weeks)
1. Elementwise family
   - sub, mul, div, pow, and unary exp, log, sqrt; error on unsupported dtypes/devices.
2. Broadcasting engine (initial)
   - Implement NumPy-style broadcasting rules for binary elementwise ops and add tests.
3. Reductions (initial)
   - sum/mean over dims with keepdim; contiguous fast paths first.
4. Views and type promotion
   - Improve transpose/view coverage; add basic type promotion rules aligned with ATen.

### Phase 3: Indexing and Performance (optional/incremental)
1. Indexing
   - gather/scatter, masked ops (limited dtypes/devices initially).
2. Strided iteration
   - Efficient kernels honoring arbitrary strides; micro-benchmarks and perf guards.
3. Perf infrastructure
   - Memory pooling, temporary buffer reuse, and a handful of fused patterns for hot paths.

### Phase 4: DNN Operator Registration (deferred)
1. Wire conv2d/pooling/batchnorm into the ATen bridge with provider-first acceleration.
2. Backward ops as needed by examples/benchmarks; maintain stable APIs.

Python integration is tracked separately in Pybind11IntegrationRoadmap.md. Early milestones (CPU tensor + NumPy zero-copy, then GPU + DLPack) can progress in parallel with Phase 0/1 as they reuse the same facades and dispatch.

## Performance Considerations

### Maintaining Efficiency
1. **Keep static kernels for hot paths**
   - Dynamic tensor dispatches to static implementations
   - Template specialization for common cases (float32, 2D/4D)

2. **Lazy allocation and execution**
   - Defer memory allocation until needed
   - Batch operations where possible

3. **Provider-first strategy**
   - Continue using cuDNN/cuBLAS/MIOpen when available
   - Fall back to generic Alpaka kernels

### Optimization Opportunities
1. **Kernel fusion**
   - Detect patterns like `add(relu(conv2d()))`
   - Generate fused kernels

2. **Memory pooling**
   - Reuse temporary buffers
   - Reduce allocation overhead

3. **Graph optimization**
   - Build computation graphs
   - Optimize before execution

## Testing Strategy

### Compatibility Testing
```python
# Test against PyTorch reference
import torch
import alpaka_tensor as at

def test_operation(op_name, *args):
    torch_result = getattr(torch, op_name)(*args)
    alpaka_args = [at.from_torch(a) for a in args]
    alpaka_result = getattr(at, op_name)(*alpaka_args)
    assert torch.allclose(torch_result, alpaka_result.to_torch())
```

### Performance Benchmarking
- Compare against PyTorch CPU/CUDA
- Profile against cuDNN/MKL direct calls
- Memory usage analysis

## Success Metrics

Phase-gated targets:
1. Phase 1: add/matmul parity with PyTorch numerics for float32/float64; provider-first paths exercised; CI green on CPU + one GPU.
2. Phase 2: Elementwise + broadcasting + basic reductions stable for common shapes; no correctness regressions.
3. Ongoing: Performance within ~10% of PyTorch for add/matmul; memory usage comparable; expand coverage pragmatically.

## Conclusion

The hybrid approach keeps Alpaka’s static, high-performance kernels while layering just enough dynamic machinery to integrate with ATen. With a stabilization-first plan, we will:

- ✅ Deliver meaningful ATen compatibility early (add/matmul), expanding incrementally
- ✅ Reuse provider-first dispatch to preserve performance on hot paths
- ✅ Keep APIs stable for existing examples/benchmarks while adding dynamic entry points
- ✅ Maintain backend portability (CPU/CUDA/HIP/SYCL)

This positions Alpaka to serve current users immediately and grow toward deeper PyTorch interoperability over time, without overcommitting to full operator coverage up front.
