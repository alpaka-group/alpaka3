/* Copyright 2025 Alpaka Tensor Library Contributors
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <alpaka/tensor/core/TensorCore.hpp>
#include <alpaka/tensor/core/TensorDebugMacros.hpp>
#include <alpaka/tensor/core/TensorView.hpp>
#include <alpaka/tensor/ops/elementwise/ElementwiseGeneric.hpp>

namespace alpaka::tensor::ops
{

    // View-based operations that don't auto-sync to host
    // These operations stay on device until explicitly synced

    // out = f(a, b) - lazy version
    template<typename T, std::size_t Rank, typename Exec, typename Device, typename Queue, typename Functor>
    tensor::Tensor<T, Rank, Device> binaryLazy(
        Exec const& exec,
        Device const& device,
        Queue& queue,
        tensor::Tensor<T, Rank, Device>& a,
        tensor::Tensor<T, Rank, Device>& b,
        Functor f,
        char const* name = "binary")
    {
        a.ensureOnDevice(device, queue);
        b.ensureOnDevice(device, queue);
        tensor::Tensor<T, Rank, Device> out(device, a.shape(), name);
        out.ensureOnDevice(device, queue);
        auto n = a.size();
        auto frame = detail::makeFrame<Exec, Queue>(n);
        queue.enqueue(
            exec,
            frame,
            BinaryKernel{},
            a.deviceBuffer(device, queue),
            b.deviceBuffer(device, queue),
            out.deviceBuffer(device, queue),
            n,
            f);
        // Removed forced wait (ALPAKA_DEBUG_SYNC can restore)
        out.markDeviceModified(device, queue);
        // Note: NO toHost() call - stays on device!
        return out;
    }

    // out = f(in) - lazy version
    template<typename T, std::size_t Rank, typename Exec, typename Device, typename Queue, typename Functor>
    tensor::Tensor<T, Rank, Device> unaryLazy(
        Exec const& exec,
        Device const& device,
        Queue& queue,
        tensor::Tensor<T, Rank, Device>& in,
        Functor f,
        char const* name = "unary")
    {
        in.ensureOnDevice(device, queue);
        tensor::Tensor<T, Rank, Device> out(device, in.shape(), name);
        out.ensureOnDevice(device, queue);
        auto n = in.size();
        auto frame = detail::makeFrame<Exec, Queue>(n);
        queue.enqueue(
            exec,
            frame,
            UnaryKernel{},
            in.deviceBuffer(device, queue),
            out.deviceBuffer(device, queue),
            n,
            f);
        // Removed forced wait (ALPAKA_DEBUG_SYNC can restore)
        out.markDeviceModified(device, queue);
        // Note: NO toHost() call - stays on device!
        return out;
    }

    // Lazy versions of common operations
    template<typename T, std::size_t Rank, typename Exec, typename Device, typename Queue>
    tensor::Tensor<T, Rank, Device> addLazy(
        Exec const& exec,
        Device const& device,
        Queue& queue,
        tensor::Tensor<T, Rank, Device>& a,
        tensor::Tensor<T, Rank, Device>& b)
    {
        return binaryLazy<T, Rank>(exec, device, queue, a, b, AddOp{}, "add_lazy");
    }

    template<typename T, std::size_t Rank, typename Exec, typename Device, typename Queue>
    tensor::Tensor<T, Rank, Device> mulLazy(
        Exec const& exec,
        Device const& device,
        Queue& queue,
        tensor::Tensor<T, Rank, Device>& a,
        tensor::Tensor<T, Rank, Device>& b)
    {
        return binaryLazy<T, Rank>(exec, device, queue, a, b, MulOp{}, "mul_lazy");
    }

    template<typename T, std::size_t Rank, typename Exec, typename Device, typename Queue, typename S>
    tensor::Tensor<T, Rank, Device> mul_scalarLazy(
        Exec const& exec,
        Device const& device,
        Queue& queue,
        tensor::Tensor<T, Rank, Device>& in,
        S scalar)
    {
        return unaryLazy<T, Rank>(exec, device, queue, in, MulScalarOp<S>{scalar}, "mul_scalar_lazy");
    }

    template<typename T, std::size_t Rank, typename Exec, typename Device, typename Queue>
    tensor::Tensor<T, Rank, Device> reluLazy(
        Exec const& exec,
        Device const& device,
        Queue& queue,
        tensor::Tensor<T, Rank, Device>& in)
    {
        return unaryLazy<T, Rank>(exec, device, queue, in, ReluOp{}, "relu_lazy");
    }

    // =============================================================================
    // VIEW-BASED OPERATIONS - These work with TensorView for method chaining
    // =============================================================================

    // Operations on TensorView that return TensorView for chaining
    template<typename T, std::size_t Rank, typename Exec, typename Device, typename Queue, typename S>
    TensorView<T, Rank, Device> mul_scalar(
        TensorView<T, Rank, Device> view,
        Exec const& exec,
        Device const& device,
        Queue& queue,
        S scalar)
    {
        auto& tensor = view.getTensor();
        auto result = mul_scalarLazy<T, Rank>(exec, device, queue, tensor, scalar);
        tensor = std::move(result); // Update the view's tensor
        return view; // Return the same view for chaining
    }

    template<typename T, std::size_t Rank, typename Exec, typename Device, typename Queue>
    TensorView<T, Rank, Device> add(
        TensorView<T, Rank, Device> view,
        Exec const& exec,
        Device const& device,
        Queue& queue,
        tensor::Tensor<T, Rank, Device>& other)
    {
        auto& tensor = view.getTensor();
        auto result = addLazy<T, Rank>(exec, device, queue, tensor, other);
        tensor = std::move(result);
        return view;
    }

    // Final operation that syncs and returns tensor
    template<typename T, std::size_t Rank, typename Exec, typename Device, typename Queue>
    tensor::Tensor<T, Rank, Device> relu(
        TensorView<T, Rank, Device> view,
        Exec const& exec,
        Device const& device,
        Queue& queue)
    {
        auto& tensor = view.getTensor();
        auto result = reluLazy<T, Rank>(exec, device, queue, tensor);
        if(detail::eagerHostEnabled())
            result.toHost(device, queue); // Optional host sync at end of chain
        return result;
    }

    // =============================================================================
    // SYNTAX SUGAR - Allow method-like chaining on tensors
    // =============================================================================

    // Free functions that enable: tensor.view().mul_scalar(...).relu(...)
    template<typename T, std::size_t Rank, typename Exec, typename Device, typename Queue, typename S>
    auto operator|(TensorView<T, Rank, Device> view, std::tuple<Exec, Device&, Queue&, S>)
        -> TensorView<T, Rank, Device>
    {
        // This is a placeholder - we'll implement proper chaining syntax later
        return view;
    }

} // namespace alpaka::tensor::ops
