#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/tensor/TensorCore.hpp>
#include <alpaka/tensor/ops/Activations.hpp>
#include <alpaka/tensor/ops/Conv2D.hpp>
#include <alpaka/tensor/ops/Gemm.hpp>

namespace alpaka::tensor::highlevel
{

    // High-level GEMM wrapper - simplified interface
    template<typename Exec, typename Device, typename Queue>
    void gemm(
        Exec const& exec,
        Device& device,
        Queue& queue,
        std::size_t M,
        std::size_t N,
        std::size_t K,
        float alpha,
        tensor::Tensor1D<float, Device>& A,
        tensor::Tensor1D<float, Device>& B,
        float beta,
        tensor::Tensor1D<float, Device>& C)
    {
        std::cout << "HighLevel API: GEMM called (M=" << M << ", N=" << N << ", K=" << K << ")" << std::endl;
        ops::gemm(exec, device, queue, 'N', 'N', M, N, K, alpha, A, B, beta, C);
    }

    // High-level Conv2D wrapper - simplified interface
    template<typename Exec, typename Device, typename Queue>
    auto conv2d(
        Exec const& exec,
        Device& device,
        Queue& queue,
        tensor::Tensor4D<float, Device> const& input,
        tensor::Tensor4D<float, Device> const& weight,
        ops::Conv2DParams const& params = {})
    {
        std::cout << "HighLevel API: Conv2D called" << std::endl;
        return ops::conv2d(exec, device, queue, input, weight, params);
    }

    // Convenience function for creating Conv2D parameters (non-constexpr due to non-constexpr ctor)
    inline ops::Conv2DParams make_conv2d_params(
        std::size_t stride_h = 1,
        std::size_t stride_w = 1,
        std::size_t pad_h = 0,
        std::size_t pad_w = 0,
        std::size_t dilation_h = 1,
        std::size_t dilation_w = 1)
    {
        return ops::Conv2DParams{stride_h, stride_w, pad_h, pad_w, dilation_h, dilation_w};
    }

    // High-level ReLU activation wrapper
    template<typename Exec, typename Device, typename Queue, typename Tensor>
    void relu(
        Exec const& exec,
        Device& device,
        Queue& queue,
        Tensor& input, // Remove const
        Tensor& output)
    {
        std::cout << "HighLevel API: ReLU called" << std::endl;
        ops::relu(exec, device, queue, input, output);
    }

    // High-level in-place ReLU activation wrapper
    template<typename Exec, typename Device, typename Queue, typename Tensor>
    void relu_inplace(Exec const& exec, Device& device, Queue& queue, Tensor& tensor)
    {
        std::cout << "HighLevel API: ReLU in-place called" << std::endl;
        ops::relu_inplace(exec, device, queue, tensor);
    }

} // namespace alpaka::tensor::highlevel
