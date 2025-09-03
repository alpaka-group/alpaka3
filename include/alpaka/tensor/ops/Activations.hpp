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
        TensorIn& input,  // Remove const to allow device operations
        TensorOut& output)
    {
        std::cout << "ReLU: Checking backend type" << std::endl;
        
        // Ensure input and output have same size
        if(input.size() != output.size()) {
            throw std::runtime_error("ReLU: Input and output tensors must have same size");
        }
        // Additional debug safety: extents equality if rank matches
        if constexpr (requires { input.extents(); output.extents(); }) {
            auto inExt = input.extents();
            auto outExt = output.extents();
            bool same=true; for(std::size_t i=0;i<input.rank;i++){ if(inExt[i]!=outExt[i]) { same=false; break; } }
            assert(same && "ReLU: input/output extents mismatch");
        }
        
        std::size_t n = input.size();
        if(n == 0) return;
        
        std::cout << "ReLU: Processing " << n << " elements" << std::endl;
        
        // Get tensor extents for proper multi-dimensional access
    auto extents = input.deviceBuffer(device, queue).getExtents();
        std::cout << "ReLU: Tensor extents = [" << extents[0] << ", " << extents[1] << ", " << extents[2] << ", " << extents[3] << "]" << std::endl;
        
        // Ensure data is on device
    input.ensureOnDevice(device, queue);
    output.ensureOnDevice(device, queue);
        
        // Setup kernel launch parameters using proper multi-dimensional frame configuration        
        // For 4D tensors, we can configure the frames to match the tensor structure
        // Example: for tensor [2,3,4,4], we could use frames of [1,1,4,4] with [2,3,1,1] frames
        // Or use a simpler approach: frames of [1,1,1,extents[3]] with [extents[0],extents[1],extents[2],1] frames
        
        auto frameExtent = alpaka::Vec{std::size_t{1}, std::size_t{1}, std::size_t{1}, extents[3]};
        auto numFrames = alpaka::Vec{extents[0], extents[1], extents[2], std::size_t{1}};
        
        auto frameSpec = alpaka::onHost::FrameSpec{numFrames, frameExtent};
        
        std::cout << "ReLU: Using frame spec with numFrames=[" << numFrames[0] << "," << numFrames[1] << "," << numFrames[2] << "," << numFrames[3] 
                  << "] frameExtent=[" << frameExtent[0] << "," << frameExtent[1] << "," << frameExtent[2] << "," << frameExtent[3] << "]" << std::endl;
        
        // Launch ReLU kernel using proper multi-dimensional frame configuration
        queue.enqueue(exec, frameSpec, ReLUKernel{}, 
            input.deviceBuffer(device, queue),
            output.deviceBuffer(device, queue));
        
        // Wait for completion and sync to host
        ::alpaka::onHost::wait(queue);
    output.markDeviceModified(device, queue);
    output.toHost(device, queue);
        
        // Backend-specific synchronization workaround
        // Some backends need additional synchronization for proper data consistency
        if constexpr (std::is_same_v<Exec, alpaka::exec::CpuOmpBlocks>) {
            ::alpaka::onHost::wait(queue);  // Extra wait for OpenMP blocks
        }
        
        std::cout << "ReLU: Operation completed" << std::endl;
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
