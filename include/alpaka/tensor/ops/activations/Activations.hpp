/* Activations - Neural Network Activation Functions
 * Implements ReLU, Sigmoid, Tanh and other activation functions for inference
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/tensor/core/TensorCore.hpp>
#include <alpaka/tensor/core/TensorDebugMacros.hpp>
#include <alpaka/tensor/kernels/ElementwiseKernels.hpp>
#include <alpaka/tensor/kernels/GeluKernels.hpp>
#include <alpaka/tensor/ops/elementwise/ElementwiseGeneric.hpp>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <stdexcept>
#include <type_traits>

namespace alpaka::tensor::ops
{

    // SIMD-based ReLU kernel following alpaka3 best practices (like babelstream)
    struct SimdReLUKernel
    {
        ALPAKA_FN_ACC void operator()(
            auto const& acc,
            auto const& operation,
            alpaka::concepts::MdSpan auto const& input,
            alpaka::concepts::MdSpan auto output) const
        {
            auto simdGrid = onAcc::SimdAlgo{onAcc::worker::threadsInGrid};
            simdGrid.concurrent(acc, input.getExtents(), operation, input, output);
        }
    };

    // Simple ReLU kernel - Non-SIMD approach using proper multi-dimensional indexing
    struct ReLUKernel
    {
        template<typename T_Acc>
        ALPAKA_FN_ACC void operator()(
            T_Acc const& acc,
            alpaka::concepts::MdSpan auto input,
            alpaka::concepts::MdSpan auto output) const
        {
            // Use proper multi-dimensional indexing with tensor extents
            for(auto tensorIdx : onAcc::makeIdxMap(acc, onAcc::worker::threadsInGrid, IdxRange(input.getExtents())))
            {
                auto val = input[tensorIdx];
                output[tensorIdx] = (val > static_cast<decltype(val)>(0)) ? val : static_cast<decltype(val)>(0);
            }
        }
    };

    // Fallback now uses generic UnaryKernel with ReluOp from ElementwiseGeneric

    // ReLU operation for SIMD processing
    struct SimdReLUOp
    {
        constexpr void operator()(auto const&, auto const input, auto output) const
        {
            using SimdType = ALPAKA_TYPEOF(input.load());
            using ValueType = alpaka::trait::GetValueType_t<SimdType>;
            output = alpaka::math::max(input.load(), SimdType::all(ValueType{0}));
        }
    };

    // ReLU activation function - applies ReLU elementwise to separate input/output tensors
    template<typename Exec, typename Device, typename Queue, typename TensorIn, typename TensorOut>
    void relu(Exec const& exec, Device& device, Queue& queue, TensorIn& input, TensorOut& output)
    {
        // Generic element count check
        if(input.size() != output.size())
            throw std::runtime_error("ReLU: size mismatch");

        // Fast path for 4D tensors (keep existing debug output & indexing)
        if constexpr(TensorIn::rank == 4 && TensorOut::rank == 4)
        {
            bool const verbose = false;
            if(verbose)
                std::cout << "ReLU: 4D fast path" << std::endl;
            auto n = input.size();
            if(n == 0)
                return;
            auto extents = input.deviceBuffer(device, queue).getExtents();
            if(verbose)
                std::cout << "ReLU: Tensor extents = [" << extents[0] << ", " << extents[1] << ", " << extents[2]
                          << ", " << extents[3] << "]" << std::endl;
            input.ensureOnDevice(device, queue);
            output.ensureOnDevice(device, queue);
            auto frameExtent = alpaka::Vec{std::size_t{1}, std::size_t{1}, std::size_t{1}, extents[3]};
            auto numFrames = alpaka::Vec{extents[0], extents[1], extents[2], std::size_t{1}};
            auto frameSpec = alpaka::onHost::FrameSpec{numFrames, frameExtent};
            queue.enqueue(
                exec,
                frameSpec,
                ReLUKernel{},
                input.deviceBuffer(device, queue),
                output.deviceBuffer(device, queue));
            // Removed unconditional wait (can be re-enabled with ALPAKA_DEBUG_SYNC)
            // ::alpaka::onHost::wait(queue);
            output.markDeviceModified(device, queue);
            if(detail::eagerHostEnabled())
                output.toHost(device, queue);
            if constexpr(std::is_same_v<Exec, alpaka::exec::CpuOmpBlocks>)
            {
                // omit extra wait; host copy already synchronizes
            }
        }
        else
        {
            bool const verbose = false;
            // Fallback: use generic unary kernel (works for 1D, 2D, etc.)
            if(verbose)
                std::cout << "ReLU: generic fallback path (rank=" << TensorIn::rank << ")" << std::endl;
            input.ensureOnDevice(device, queue);
            output.ensureOnDevice(device, queue);
            auto n = input.size();
            if(n == 0)
                return;
            // Use shared generic UnaryKernel with ReluOp functor
            unsigned threadsPerBlock = 256u;
            unsigned blocks = static_cast<unsigned>((n + threadsPerBlock - 1) / threadsPerBlock);
            if(blocks == 0)
                blocks = 1;
            auto frame = alpaka::onHost::FrameSpec{
                alpaka::Vec<unsigned int, 1u>{blocks},
                alpaka::Vec<unsigned int, 1u>{threadsPerBlock}};
            queue.enqueue(
                exec,
                frame,
                UnaryKernel{},
                input.deviceBuffer(device, queue),
                output.deviceBuffer(device, queue),
                n,
                ReluOp{});
            // Removed unconditional wait (can be re-enabled with ALPAKA_DEBUG_SYNC)
            output.markDeviceModified(device, queue);
            if(detail::eagerHostEnabled())
                output.toHost(device, queue);
        }
    }

    // In-place ReLU activation - modifies input tensor directly
    template<typename Exec, typename Device, typename Queue, typename Tensor>
    void relu_inplace(Exec const& exec, Device& device, Queue& queue, Tensor& tensor)
    {
        relu(exec, device, queue, tensor, tensor);
    }

    template<typename T, std::size_t Rank, typename Exec, typename Device, typename Queue>
    void relu_inplace_async(Exec const& exec, Device const& device, Queue& queue, tensor::Tensor<T, Rank, Device>& t)
    {
        t.ensureOnDevice(device, queue);
        auto n = t.size();
        auto frame = ops::detail::makeFrame<Exec, Queue>(n);
        queue.enqueue(exec, frame, kernels::ReluInplaceKernel<T>{}, t.deviceBuffer(device, queue).data(), n);
        t.markDeviceModified(device, queue);
    }

    template<typename T, std::size_t Rank, typename Exec, typename Device, typename Queue>
    void gelu(Exec const& exec, Device const& device, Queue& queue, tensor::Tensor<T, Rank, Device>& t)
    {
        t.ensureOnDevice(device, queue);
        auto total = t.size();
        auto frame = ops::detail::makeFrame<Exec, Queue>(total);
        queue.enqueue(
            exec,
            frame,
            kernels::GeluKernel<T>{},
            t.deviceBuffer(device, queue).data(),
            t.deviceBuffer(device, queue).data(),
            total);
        t.markDeviceModified(device, queue);
    }

    template<typename T, typename Exec, typename Device, typename Queue>
    void gelu(Exec const& exec, Device const& device, Queue& queue, tensor::Tensor2D<T, Device>& t)
    {
        t.ensureOnDevice(device, queue);
        auto shape = t.shape();
        std::size_t M = shape[0];
        std::size_t D = shape[1];
        auto frame = ops::detail::makeFrame<Exec, Queue>(M * D);
        queue.enqueue(
            exec,
            frame,
            kernels::Gelu2DViewKernel<T>{},
            t.deviceBuffer(device, queue),
            t.deviceBuffer(device, queue),
            M,
            D);
        t.markDeviceModified(device, queue);
    }

} // namespace alpaka::tensor::ops
