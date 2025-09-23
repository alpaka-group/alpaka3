/* Simple reduction scaffolding for Phase 0
 * - sum_all: device reduction to a scalar (host-returned)
 * - mean_all: device mean using sum_all/size
 * Uses existing Alpaka primitives (MdSpan, onHost::reduce) and TensorCore buffers.
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/mem/View.hpp>
#include <alpaka/tensor/TensorCore.hpp>

#include <functional>

namespace alpaka::tensor::ops
{
    namespace detail
    {
        // Create a 1-element device buffer for reductions
        template<typename T, typename Device>
        inline auto makeDeviceScalar(Device const& dev)
        {
            using Ext1 = alpaka::Vec<std::size_t, 1>;
            return alpaka::onHost::alloc<T>(dev, Ext1{1});
        }

        // Create a 1-element host buffer
        template<typename T>
        inline auto makeHostScalar()
        {
            using Ext1 = alpaka::Vec<std::size_t, 1>;
            return alpaka::onHost::allocHost<T>(Ext1{1});
        }
    } // namespace detail

    // Reduce all elements to a scalar on device then return the value on host.
    template<typename T, std::size_t Rank, typename Exec, typename Device, typename Queue>
    inline T sum_all(Exec const& exec, Device const& device, Queue& queue, tensor::Tensor<T, Rank, Device>& in)
    {
        // Ensure input is resident on device
        in.ensureOnDevice(device, queue);

        // Create device output (1 element) and corresponding views
        auto d_out = detail::makeDeviceScalar<T>(device);
        auto outView = alpaka::makeView(d_out);

        auto& d_in = in.deviceBuffer(device, queue);
        auto inView = alpaka::makeView(d_in);

        // Perform reduction: neutral = 0, op = plus
        alpaka::onHost::reduce(queue, exec, T{0}, outView, std::plus<>{}, inView);

        // Copy result to host and return
        auto h_out = detail::makeHostScalar<T>();
        alpaka::onHost::memcpy(queue, h_out, d_out);
        alpaka::onHost::wait(queue);
        return h_out.data()[0];
    }

    template<typename T, std::size_t Rank, typename Exec, typename Device, typename Queue>
    inline T mean_all(Exec const& exec, Device const& device, Queue& queue, tensor::Tensor<T, Rank, Device>& in)
    {
        auto s = sum_all<T, Rank>(exec, device, queue, in);
        return static_cast<T>(s / static_cast<double>(in.size()));
    }

} // namespace alpaka::tensor::ops
