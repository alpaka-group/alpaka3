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

## Implementation Roadmap

### Phase 1: Foundation (1-2 months)
1. **Implement DynamicTensor wrapper**
   - Type-erased base class
   - Runtime shape/dtype/device tracking
   - Basic view operations (reshape, transpose)

2. **Create Storage abstraction**
   - Reference-counted memory
   - Device-aware allocation
   - Slice/view support

3. **Build dispatch infrastructure**
   - Runtime to compile-time type dispatch
   - Rank dispatch (1D to 6D typical)
   - Device routing

### Phase 2: Core Operations (2-3 months)
1. **Extend element-wise operations**
   ```cpp
   // Add to ElementwiseGeneric.hpp
   DEFINE_UNARY_OP(exp, expf)
   DEFINE_UNARY_OP(log, logf)
   DEFINE_UNARY_OP(sqrt, sqrtf)
   DEFINE_UNARY_OP(sin, sinf)
   DEFINE_UNARY_OP(cos, cosf)
   DEFINE_BINARY_OP(pow, powf)
   ```

2. **Implement reduction framework**
   ```cpp
   template<typename ReduceOp>
   DynamicTensor reduce(const DynamicTensor& input,
                       IntArrayRef dims,
                       bool keepdim,
                       ReduceOp op);
   ```

3. **Add comparison operations**
   ```cpp
   DynamicTensor compare(const DynamicTensor& a,
                        const DynamicTensor& b,
                        ComparisonType cmp);
   ```

### Phase 3: Advanced Features (2-3 months)
1. **Indexing and slicing**
   - gather/scatter kernels
   - Advanced indexing with tensors
   - Masked operations

2. **Broadcasting engine**
   - NumPy-compatible broadcasting rules
   - Efficient strided iteration

3. **Autograd integration hooks**
   - Gradient tracking metadata
   - Backward function registration

### Phase 4: Python Integration (1-2 months)
1. **Python bindings via pybind11**
   ```python
   import alpaka_tensor as at
   
   x = at.tensor([[1, 2], [3, 4]], device='cuda:0')
   y = at.tensor([[5, 6]], device='cuda:0')
   z = at.matmul(x, y.T)  # With broadcasting
   ```

2. **PyTorch custom backend**
   ```python
   import torch
   torch.ops.load_library("alpaka_ops.so")
   
   x = torch.randn(100, 100, device='alpaka:0')
   ```

3. **Operator registration**
   ```cpp
   TORCH_LIBRARY(alpaka, m) {
       m.def("add(Tensor a, Tensor b) -> Tensor");
       m.def("conv2d(Tensor input, Tensor weight, ...) -> Tensor");
   }
   ```

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

1. **Operator Coverage**: Support 90% of core ATen ops (top 200)
2. **Performance**: Within 10% of PyTorch for common operations
3. **Memory Efficiency**: Comparable memory usage to PyTorch
4. **API Compatibility**: Drop-in replacement for common workflows
5. **Backend Integration**: Functional PyTorch custom backend

## Conclusion

The proposed hybrid architecture preserves Alpaka's performance advantages while adding the flexibility needed for ATen compatibility. By maintaining the efficient static core and adding a dynamic layer, we can achieve:

- ✅ Full ATen operator compatibility
- ✅ Python/PyTorch integration
- ✅ Maintained performance for specialized use cases
- ✅ Gradual migration path
- ✅ Backend portability (CPU/CUDA/HIP/SYCL)

This positions Alpaka as both a high-performance compute library and a viable PyTorch backend, opening opportunities for:
- Custom accelerator support
- Research into new hardware backends
- Specialized optimizations for specific workloads
- Educational framework for tensor computation

The modular approach allows teams to adopt components incrementally, from using just the static kernels to full PyTorch integration.
