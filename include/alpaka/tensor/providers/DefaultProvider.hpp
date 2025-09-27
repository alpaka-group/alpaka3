/* Default Provider Implementation - Generic Alpaka Kernels
 * Fallback implementation when specialized libraries unavailable
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <alpaka/tensor/ops/fallback/FallbackOps.hpp>
#include <alpaka/tensor/providers/ProviderInterface.hpp>

namespace alpaka::tensor
{
    /**
     * Default provider using generic Alpaka kernels.
     * Used as fallback when specialized libraries (cuBLAS, cuDNN) unavailable.
     */
    class DefaultProvider : public IOpProvider
    {
    public:
        std::string getBackendName() const override
        {
            return "Default (Generic Alpaka Kernels)";
        }

        bool supportsOperation(OpType op) const override
        {
            // Default provider supports all operations via generic kernels
            switch(op)
            {
            case OpType::Conv2D:
            case OpType::GEMM:
            case OpType::BatchNorm:
            case OpType::Pooling:
            case OpType::Activation:
                return true;
            default:
                return false;
            }
        }

        bool isActive() const override
        {
            return true; // Always available
        }

    protected:
        OpStatus conv2d_impl(
            void const* exec_ptr,
            void const* device_ptr,
            void* queue_ptr,
            void const* inputDesc,
            void const* weightDesc,
            ops::Conv2DParams const& params,
            void* outDesc) override
        {
            // TODO: implement proper type-erased conv2d delegation
            return OpStatus::Unsupported; // signal caller to fallback to template path
        }

        OpStatus gemm_impl(
            void const* exec_ptr,
            void const* device_ptr,
            void* queue_ptr,
            std::size_t M,
            std::size_t N,
            std::size_t K,
            float alpha,
            void const* ADesc,
            void const* BDesc,
            float beta,
            void* CDesc) override
        {
            // Without runtime type info we cannot safely dispatch; request fallback path
            return OpStatus::Unsupported;
        }

        OpStatus batchnorm_impl(
            void const* /*exec_ptr*/,
            void const* /*device_ptr*/,
            void* /*queue_ptr*/,
            void const* /*inputDesc*/,
            void const* /*meanDesc*/,
            void const* /*varianceDesc*/,
            void const* /*gammaDesc*/,
            void const* /*betaDesc*/,
            float /*epsilon*/,
            void* /*outDesc*/) override
        {
            return OpStatus::Unsupported;
        }

    public:
        // Convenience methods that preserve type information and delegate to existing ops
        template<typename T, typename Exec, typename Device, typename Queue>
        auto conv2d(
            Exec const& exec,
            Device const& device,
            Queue& queue,
            tensor::Tensor4D<T, Device> const& input,
            tensor::Tensor4D<T, Device> const& weight,
            ops::Conv2DParams const& params) -> tensor::Tensor4D<T, Device>
        {
            return ops::fallback::conv2d<T>(exec, device, queue, input, weight, params);
        }

        template<typename Exec, typename Device, typename Queue>
        void gemm(
            Exec const& exec,
            Device const& device,
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
            ops::fallback::gemm(exec, device, queue, M, N, K, alpha, A, B, beta, C);
        }

        template<typename T, typename Exec, typename Device, typename Queue>
        auto max_pool2d(
            Exec const& exec,
            Device const& device,
            Queue& queue,
            tensor::Tensor4D<T, Device>& input,
            ops::Pool2DParams const& params) -> tensor::Tensor4D<T, Device>
        {
            return ops::fallback::max_pool2d<T>(exec, device, queue, input, params);
        }

        template<typename T, typename Exec, typename Device, typename Queue>
        auto avg_pool2d(
            Exec const& exec,
            Device const& device,
            Queue& queue,
            tensor::Tensor4D<T, Device>& input,
            ops::Pool2DParams const& params) -> tensor::Tensor4D<T, Device>
        {
            return ops::fallback::avg_pool2d<T>(exec, device, queue, input, params);
        }
    };

} // namespace alpaka::tensor
