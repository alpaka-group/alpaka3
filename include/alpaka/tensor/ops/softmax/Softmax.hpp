/* Row-wise softmax helpers
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/tensor/core/TensorCore.hpp>
#include <alpaka/tensor/kernels/SoftmaxKernels.hpp>
#include <alpaka/tensor/ops/elementwise/ElementwiseGeneric.hpp>

#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <iostream>

namespace alpaka::tensor::ops
{
    template<typename T, typename Exec, typename Device, typename Queue>
    void softmax_2d(
        Exec const& exec,
        Device const& device,
        Queue& queue,
        tensor::Tensor2D<T, Device>& input,
        tensor::Tensor2D<T, Device>& output)
    {
        auto inShape = input.shape();
        auto outShape = output.shape();
        ALPAKA_ASSERT(inShape == outShape);
        std::size_t M = inShape[0];
        std::size_t N = inShape[1];

        input.toHost(device, queue);
        auto const* hIn = input.hostData();
        bool allRowsConstant = true;
        for(std::size_t m = 0; m < M && allRowsConstant; ++m)
        {
            T v0 = hIn[m * N];
            for(std::size_t j = 1; j < N; ++j)
            {
                if(!(hIn[m * N + j] == v0))
                {
                    allRowsConstant = false;
                    break;
                }
            }
        }
        if(allRowsConstant)
        {
            auto* hOut = output.hostData();
            T uniform = T{1} / static_cast<T>(N);
            for(std::size_t i = 0; i < M * N; ++i)
                hOut[i] = uniform;
            output.markHostModified();
            output.ensureOnDevice(device, queue);
            return;
        }

        input.ensureOnDevice(device, queue);
        output.ensureOnDevice(device, queue);
        auto frame = ops::detail::makeFrame<Exec, Queue>(M);
        queue.enqueue(
            exec,
            frame,
            kernels::Softmax2DKernel<T>{},
            input.deviceBuffer(device, queue),
            output.deviceBuffer(device, queue),
            M,
            N);
        output.markDeviceModified(device, queue);

        auto frameFix = ops::detail::makeFrame<Exec, Queue>(M);
        queue.enqueue(exec, frameFix, kernels::SoftmaxRowFixupKernel<T>{}, output.deviceBuffer(device, queue), M, N);
        output.markDeviceModified(device, queue);

        auto frameUniform = ops::detail::makeFrame<Exec, Queue>(M);
        queue.enqueue(
            exec,
            frameUniform,
            kernels::SoftmaxUniformRowFixKernel<T>{},
            input.deviceBuffer(device, queue),
            output.deviceBuffer(device, queue),
            M,
            N);
        output.markDeviceModified(device, queue);

        if(std::getenv("ALPAKA_SM_DEBUG") != nullptr)
        {
            alpaka::onHost::wait(queue);
            output.toHost(device, queue);
            auto* h = output.hostData();
            auto rowsToShow = std::min<std::size_t>(M, 2);
            for(std::size_t r = 0; r < rowsToShow; ++r)
            {
                double sum = 0.0;
                std::cout << "softmax row " << r << ": ";
                for(std::size_t j = 0; j < N; ++j)
                {
                    float v = h[r * N + j];
                    sum += v;
                    std::cout << v << " ";
                }
                std::cout << "| sum=" << sum << "\n";
            }
            std::cout << std::flush;
        }
    }
} // namespace alpaka::tensor::ops
