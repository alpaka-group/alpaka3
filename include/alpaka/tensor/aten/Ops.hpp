// SPDX-License-Identifier: MPL-2.0
#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/tensor/aten/Broadcast.hpp>
#include <alpaka/tensor/aten/DynamicTensor.hpp>
#include <alpaka/tensor/ops/ElementwiseGeneric.hpp>
#include <alpaka/tensor/ops/Gemm.hpp>
#include <alpaka/tensor/ops/InferenceOps.hpp>

#include <stdexcept>

namespace alpaka::tensor::aten
{
    // add: a + b -> out (Float32 only). Supports 1D or 2D with exact shape match.
    template<typename Exec, typename Device, typename Queue>
    DynamicTensor<Device> add(
        Exec const& exec,
        Device const& device,
        Queue& queue,
        DynamicTensor<Device>& a,
        DynamicTensor<Device>& b)
    {
        if(a.dtype() != b.dtype())
            throw std::runtime_error("aten::add: dtype mismatch");
        if(a.rank() != b.rank())
            throw std::runtime_error("aten::add: rank mismatch");
        if(a.dtype() != ScalarType::Float32)
            throw std::runtime_error("aten::add: only Float32 supported");
        auto sa = a.shape();
        auto sb = b.shape();
        if(sa != sb)
        {
            // Try broadcasting for ranks 1 or 2 only (Float32)
            if(a.rank() == 1)
                return binary_broadcast_1d(exec, device, queue, a, b, ops::AddOp{}, "aten_add_1d_bc");
            else if(a.rank() == 2)
                return binary_broadcast_2d(exec, device, queue, a, b, ops::AddOp{}, "aten_add_2d_bc");
            else
                throw std::runtime_error("aten::add: shape mismatch");
        }
        if(a.rank() == 1)
        {
            auto& ta = a.template as<float, 1>();
            auto& tb = b.template as<float, 1>();
            auto out = ops::add<float, 1>(exec, device, queue, ta, tb);
            return DynamicTensor<Device>::template wrap<float, 1>(std::move(out));
        }
        else if(a.rank() == 2)
        {
            auto& ta = a.template as<float, 2>();
            auto& tb = b.template as<float, 2>();
            // Flatten both to 1D, add on device, then copy back to 2D on host (robust across pitched layouts)
            auto A1D = ops::flatten<float, 2>(exec, device, queue, ta);
            auto B1D = ops::flatten<float, 2>(exec, device, queue, tb);
            auto C1D = ops::add<float, 1>(exec, device, queue, A1D, B1D);
            C1D.toHost(device, queue);
            tensor::Tensor2D<float, Device> out(device, ta.shape(), "aten_add_2d");
            auto* outH = out.hostData();
            auto const* cH = C1D.hostData();
            for(std::size_t i = 0; i < ta.size(); ++i)
                outH[i] = cH[i];
            out.markHostModified();
            out.ensureOnDevice(device, queue);
            return DynamicTensor<Device>::template wrap<float, 2>(std::move(out));
        }
        else
        {
            throw std::runtime_error("aten::add: only ranks 1 or 2 supported");
        }
    }

    // sub: a - b -> out (Float32 only). Supports 1D or 2D with exact shape match.
    template<typename Exec, typename Device, typename Queue>
    DynamicTensor<Device> sub(
        Exec const& exec,
        Device const& device,
        Queue& queue,
        DynamicTensor<Device>& a,
        DynamicTensor<Device>& b)
    {
        if(a.dtype() != b.dtype())
            throw std::runtime_error("aten::sub: dtype mismatch");
        if(a.rank() != b.rank())
            throw std::runtime_error("aten::sub: rank mismatch");
        if(a.dtype() != ScalarType::Float32)
            throw std::runtime_error("aten::sub: only Float32 supported");
        auto sa = a.shape();
        auto sb = b.shape();
        if(sa != sb)
        {
            if(a.rank() == 1)
                return binary_broadcast_1d(exec, device, queue, a, b, ops::SubOp{}, "aten_sub_1d_bc");
            else if(a.rank() == 2)
                return binary_broadcast_2d(exec, device, queue, a, b, ops::SubOp{}, "aten_sub_2d_bc");
            else
                throw std::runtime_error("aten::sub: shape mismatch");
        }
        if(a.rank() == 1)
        {
            auto& ta = a.template as<float, 1>();
            auto& tb = b.template as<float, 1>();
            auto out = ops::sub<float, 1>(exec, device, queue, ta, tb);
            return DynamicTensor<Device>::template wrap<float, 1>(std::move(out));
        }
        else if(a.rank() == 2)
        {
            auto& ta = a.template as<float, 2>();
            auto& tb = b.template as<float, 2>();
            // Flatten both to 1D, sub on device, then copy back to 2D on host
            auto A1D = ops::flatten<float, 2>(exec, device, queue, ta);
            auto B1D = ops::flatten<float, 2>(exec, device, queue, tb);
            auto C1D = ops::sub<float, 1>(exec, device, queue, A1D, B1D);
            C1D.toHost(device, queue);
            tensor::Tensor2D<float, Device> out(device, ta.shape(), "aten_sub_2d");
            auto* outH = out.hostData();
            auto const* cH = C1D.hostData();
            for(std::size_t i = 0; i < ta.size(); ++i)
                outH[i] = cH[i];
            out.markHostModified();
            out.ensureOnDevice(device, queue);
            return DynamicTensor<Device>::template wrap<float, 2>(std::move(out));
        }
        else
        {
            throw std::runtime_error("aten::sub: only ranks 1 or 2 supported");
        }
    }

    // mul: a * b -> out (Float32 only). Supports 1D or 2D with exact shape match.
    template<typename Exec, typename Device, typename Queue>
    DynamicTensor<Device> mul(
        Exec const& exec,
        Device const& device,
        Queue& queue,
        DynamicTensor<Device>& a,
        DynamicTensor<Device>& b)
    {
        if(a.dtype() != b.dtype())
            throw std::runtime_error("aten::mul: dtype mismatch");
        if(a.rank() != b.rank())
            throw std::runtime_error("aten::mul: rank mismatch");
        if(a.dtype() != ScalarType::Float32)
            throw std::runtime_error("aten::mul: only Float32 supported");
        auto sa = a.shape();
        auto sb = b.shape();
        if(sa != sb)
        {
            if(a.rank() == 1)
                return binary_broadcast_1d(exec, device, queue, a, b, ops::MulOp{}, "aten_mul_1d_bc");
            else if(a.rank() == 2)
                return binary_broadcast_2d(exec, device, queue, a, b, ops::MulOp{}, "aten_mul_2d_bc");
            else
                throw std::runtime_error("aten::mul: shape mismatch");
        }
        if(a.rank() == 1)
        {
            auto& ta = a.template as<float, 1>();
            auto& tb = b.template as<float, 1>();
            auto out = ops::mul<float, 1>(exec, device, queue, ta, tb);
            return DynamicTensor<Device>::template wrap<float, 1>(std::move(out));
        }
        else if(a.rank() == 2)
        {
            auto& ta = a.template as<float, 2>();
            auto& tb = b.template as<float, 2>();
            auto A1D = ops::flatten<float, 2>(exec, device, queue, ta);
            auto B1D = ops::flatten<float, 2>(exec, device, queue, tb);
            auto C1D = ops::mul<float, 1>(exec, device, queue, A1D, B1D);
            C1D.toHost(device, queue);
            tensor::Tensor2D<float, Device> out(device, ta.shape(), "aten_mul_2d");
            auto* outH = out.hostData();
            auto const* cH = C1D.hostData();
            for(std::size_t i = 0; i < ta.size(); ++i)
                outH[i] = cH[i];
            out.markHostModified();
            out.ensureOnDevice(device, queue);
            return DynamicTensor<Device>::template wrap<float, 2>(std::move(out));
        }
        else
        {
            throw std::runtime_error("aten::mul: only ranks 1 or 2 supported");
        }
    }

    // div: a / b -> out (Float32 only). Supports 1D or 2D with exact shape match.
    template<typename Exec, typename Device, typename Queue>
    DynamicTensor<Device> div(
        Exec const& exec,
        Device const& device,
        Queue& queue,
        DynamicTensor<Device>& a,
        DynamicTensor<Device>& b)
    {
        if(a.dtype() != b.dtype())
            throw std::runtime_error("aten::div: dtype mismatch");
        if(a.rank() != b.rank())
            throw std::runtime_error("aten::div: rank mismatch");
        if(a.dtype() != ScalarType::Float32)
            throw std::runtime_error("aten::div: only Float32 supported");
        auto sa = a.shape();
        auto sb = b.shape();
        if(sa != sb)
        {
            if(a.rank() == 1)
                return binary_broadcast_1d(exec, device, queue, a, b, ops::DivOp{}, "aten_div_1d_bc");
            else if(a.rank() == 2)
                return binary_broadcast_2d(exec, device, queue, a, b, ops::DivOp{}, "aten_div_2d_bc");
            else
                throw std::runtime_error("aten::div: shape mismatch");
        }
        if(a.rank() == 1)
        {
            auto& ta = a.template as<float, 1>();
            auto& tb = b.template as<float, 1>();
            auto out = ops::div<float, 1>(exec, device, queue, ta, tb);
            return DynamicTensor<Device>::template wrap<float, 1>(std::move(out));
        }
        else if(a.rank() == 2)
        {
            auto& ta = a.template as<float, 2>();
            auto& tb = b.template as<float, 2>();
            auto A1D = ops::flatten<float, 2>(exec, device, queue, ta);
            auto B1D = ops::flatten<float, 2>(exec, device, queue, tb);
            auto C1D = ops::div<float, 1>(exec, device, queue, A1D, B1D);
            C1D.toHost(device, queue);
            tensor::Tensor2D<float, Device> out(device, ta.shape(), "aten_div_2d");
            auto* outH = out.hostData();
            auto const* cH = C1D.hostData();
            for(std::size_t i = 0; i < ta.size(); ++i)
                outH[i] = cH[i];
            out.markHostModified();
            out.ensureOnDevice(device, queue);
            return DynamicTensor<Device>::template wrap<float, 2>(std::move(out));
        }
        else
        {
            throw std::runtime_error("aten::div: only ranks 1 or 2 supported");
        }
    }

    // matmul: C = A @ B for 2D Float32 tensors
    template<typename Exec, typename Device, typename Queue>
    DynamicTensor<Device> matmul(
        Exec const& exec,
        Device const& device,
        Queue& queue,
        DynamicTensor<Device>& A,
        DynamicTensor<Device>& B)
    {
        if(A.dtype() != ScalarType::Float32 || B.dtype() != ScalarType::Float32)
            throw std::runtime_error("aten::matmul: only Float32 supported");
        if(A.rank() != 2 || B.rank() != 2)
            throw std::runtime_error("aten::matmul: only 2D tensors supported in Phase 1");
        auto& a2 = A.template as<float, 2>();
        auto& b2 = B.template as<float, 2>();
        std::size_t M = a2.shape()[0];
        std::size_t K = a2.shape()[1];
        if(b2.shape()[0] != K)
            throw std::runtime_error("aten::matmul: inner dimension mismatch");
        std::size_t N = b2.shape()[1];
        // Provider-first: flatten A,B to 1D, GEMM into flat C, then copy to 2D on host and sync back.
        // Rationale: Host copyback avoids backend-specific edge cases in 1D->2D device copies across
        // pitched/strided layouts. Once the device-side 1D->2D copy path is hardened, we can switch to it.
        auto A1D = ops::flatten<float, 2>(exec, device, queue, a2);
        auto B1D = ops::flatten<float, 2>(exec, device, queue, b2);
        tensor::Tensor1D<float, Device> C1D(device, {M * N}, "aten_mm_out_flat");
        ops::gemm(exec, device, queue, 'N', 'N', M, N, K, 1.0f, A1D, B1D, 0.0f, C1D);
        C1D.toHost(device, queue);
        tensor::Tensor2D<float, Device> c2(device, {M, N}, "aten_mm_out");
        auto* c2h = c2.hostData();
        auto const* c1h = C1D.hostData();
        for(std::size_t i = 0; i < M * N; ++i)
            c2h[i] = c1h[i];
        c2.markHostModified();
        c2.ensureOnDevice(device, queue);
        return DynamicTensor<Device>::template wrap<float, 2>(std::move(c2));

        // Debug-only manual host fallback (kept for reference, intentionally inactive):
        // a2.toHost(device, queue);
        // b2.toHost(device, queue);
        // tensor::Tensor2D<float, Device> c2h_only(device, {M, N}, "aten_mm_out_host");
        // auto const* Ah = a2.hostData();
        // auto const* Bh = b2.hostData();
        // auto* Ch = c2h_only.hostData();
        // for(std::size_t i = 0; i < M; ++i)
        //     for(std::size_t j = 0; j < N; ++j)
        //     {
        //         float s = 0.f;
        //         for(std::size_t k = 0; k < K; ++k)
        //             s += Ah[i * K + k] * Bh[k * N + j];
        //         Ch[i * N + j] = s;
        //     }
        // c2h_only.markHostModified();
        // c2h_only.ensureOnDevice(device, queue);
        // return DynamicTensor<Device>::template wrap<float, 2>(std::move(c2h_only));
    }
} // namespace alpaka::tensor::aten
