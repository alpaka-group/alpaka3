/* MIOpen Provider Implementation (stub)
 * Clean separation of MIOpen-specific logic from generic operations
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <alpaka/tensor/providers/ProviderInterface.hpp>

#ifdef ALPAKA_HAS_MIOPEN
#    include <miopen/miopen.h>
#    include <hip/hip_runtime.h>
#    include <cstdlib>
#    include <iostream>
#    include <string>
#endif

namespace alpaka::tensor
{
    /**
     * MIOpen provider for HIP GPU deep-learning operations.
     * Minimal stub advertising supported ops; actual calls are not yet implemented.
     */
    class MIOpenProvider : public IOpProvider
    {
    private:
#ifdef ALPAKA_HAS_MIOPEN
        mutable miopenHandle_t handle_ = nullptr;
        mutable bool initialized_ = false;
#endif

    public:
        ~MIOpenProvider() override
        {
#ifdef ALPAKA_HAS_MIOPEN
            if(handle_)
            {
                miopenDestroy(handle_);
                handle_ = nullptr;
            }
#endif
        }

        std::string getBackendName() const override
        {
#ifdef ALPAKA_HAS_MIOPEN
            return "MIOpen (HIP)";
#else
            return "MIOpen (unavailable)";
#endif
        }

        bool supportsOperation(OpType op) const override
        {
#ifdef ALPAKA_HAS_MIOPEN
            return op == OpType::Conv2D || op == OpType::BatchNorm || op == OpType::Activation;
#else
            (void)op;
            return false;
#endif
        }

        bool isActive() const override
        {
#ifdef ALPAKA_HAS_MIOPEN
            const char* disableEnv = std::getenv("ALPAKA_DISABLE_MIOPEN");
            if(disableEnv)
            {
                std::string v(disableEnv);
                bool disable = (v == "1" || v == "ON" || v == "on" || v == "true" || v == "TRUE");
                if(disable)
                    return false;
            }
            return true;
#else
            return false;
#endif
        }

    protected:
        OpStatus conv2d_impl(
            void const*,
            void const*,
            void*,
            void const*,
            void const*,
            ops::Conv2DParams const&,
            void*) override
        {
            return OpStatus::Unsupported; // Not yet implemented
        }

        OpStatus gemm_impl(
            void const*,
            void const*,
            void*,
            std::size_t,
            std::size_t,
            std::size_t,
            float,
            void const*,
            void const*,
            float,
            void*) override
        {
            return OpStatus::Unsupported; // BLAS not provided by MIOpen
        }

        OpStatus batchnorm_impl(
            void const*,
            void const*,
            void*,
            void const*,
            void const*,
            void const*,
            void const*,
            void const*,
            float,
            void*) override
        {
            return OpStatus::Unsupported; // Not yet implemented
        }

    private:
#ifdef ALPAKA_HAS_MIOPEN
        void ensureInitialized() const
        {
            if(initialized_)
                return;
            auto st = miopenCreate(&handle_);
            if(st != miopenStatusSuccess)
            {
                std::cerr << "MIOpen create handle failed status=" << st << "\n";
                return;
            }
            initialized_ = true;
        }
#endif
    };
} // namespace alpaka::tensor
