/* Activations - Neural Network Activation Functions
 * Implements ReLU, Sigmoid, Tanh and other activation functions for inference
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/tensor/TensorCore.hpp>
#include <alpaka/tensor/ops/ElementwiseGeneric.hpp>
#include <iostream>
#include <stdexcept>
#include <type_traits>
#include <algorithm>
#include <cmath>

namespace alpaka::tensor::ops {

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

    // Generic 1D fallback kernel for elementwise ReLU (rank-agnostic flattened launch)
    struct SimpleUnaryReluKernel {
        template<typename Acc, typename InBuf, typename OutBuf>
        ALPAKA_FN_ACC void operator()(Acc const& acc, InBuf in, OutBuf out, std::size_t n) const {
            for(auto [i] : alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{n})) {
                auto v = in[i];
                out[i] = v > decltype(v){} ? v : decltype(v){};
            }
        }
    };

    // ReLU operation for SIMD processing
    struct SimdReLUOp
    {
        constexpr void operator()(auto const&, auto const input, auto output) const
        {
            using SimdType = ALPAKA_TYPEOF(input.load());
            using ValueType = trait::GetValueType_t<SimdType>;
            output = alpaka::math::max(input.load(), SimdType::all(ValueType{0}));
        }
    };

    // ReLU activation function - applies ReLU elementwise to separate input/output tensors
    template<typename Exec, typename Device, typename Queue, typename TensorIn, typename TensorOut>
    void relu(
        Exec const& exec,
        Device& device,
        Queue& queue,
        TensorIn& input,
        TensorOut& output)
    {
        // Generic element count check
        if(input.size() != output.size())
            throw std::runtime_error("ReLU: size mismatch");

        // Fast path for 4D tensors (keep existing debug output & indexing)
        if constexpr (TensorIn::rank == 4 && TensorOut::rank == 4) {
            std::cout << "ReLU: 4D fast path" << std::endl;
            auto n = input.size(); if(n==0) return;
            auto extents = input.deviceBuffer(device, queue).getExtents();
            std::cout << "ReLU: Tensor extents = [" << extents[0] << ", " << extents[1] << ", " << extents[2] << ", " << extents[3] << "]" << std::endl;
            input.ensureOnDevice(device, queue);
            output.ensureOnDevice(device, queue);
            auto frameExtent = alpaka::Vec{std::size_t{1}, std::size_t{1}, std::size_t{1}, extents[3]};
            auto numFrames   = alpaka::Vec{extents[0], extents[1], extents[2], std::size_t{1}};
            auto frameSpec   = alpaka::onHost::FrameSpec{numFrames, frameExtent};
            queue.enqueue(exec, frameSpec, ReLUKernel{},
                input.deviceBuffer(device, queue),
                output.deviceBuffer(device, queue));
            ::alpaka::onHost::wait(queue);
            output.markDeviceModified(device, queue);
            output.toHost(device, queue);
            if constexpr (std::is_same_v<Exec, alpaka::exec::CpuOmpBlocks>) ::alpaka::onHost::wait(queue);
        } else {
            // Fallback: use generic unary kernel (works for 1D, 2D, etc.)
            std::cout << "ReLU: generic fallback path (rank=" << TensorIn::rank << ")" << std::endl;
            input.ensureOnDevice(device, queue);
            output.ensureOnDevice(device, queue);
            auto n = input.size(); if(n==0) return;
            // Reuse elementwise generic launcher (local minimal inline implementation)
            unsigned threadsPerBlock = 256u;
            unsigned blocks = static_cast<unsigned>((n + threadsPerBlock - 1) / threadsPerBlock);
            if(blocks==0) blocks=1;
            auto frame = alpaka::onHost::FrameSpec{alpaka::Vec<unsigned int,1u>{blocks}, alpaka::Vec<unsigned int,1u>{threadsPerBlock}};
            queue.enqueue(exec, frame, SimpleUnaryReluKernel{}, input.deviceBuffer(device, queue), output.deviceBuffer(device, queue), n);
            ::alpaka::onHost::wait(queue);
            output.markDeviceModified(device, queue);
            output.toHost(device, queue);
        }
    }

    // In-place ReLU activation - modifies input tensor directly
    template<typename Exec, typename Device, typename Queue, typename Tensor>
    void relu_inplace(
        Exec const& exec,
        Device& device,
        Queue& queue,
        Tensor& tensor)
    {
        relu(exec, device, queue, tensor, tensor);
    }

}
