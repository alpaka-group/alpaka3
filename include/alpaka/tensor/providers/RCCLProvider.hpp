/* RCCL Collective Provider
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <alpaka/tensor/ops/CollectiveOps.hpp>
#include <alpaka/tensor/providers/ICollectiveProvider.hpp>

#include <stdexcept>
#include <string>
#include <vector>

#ifdef ALPAKA_HAS_RCCL
#    include <hip/hip_runtime_api.h>
#    include <rccl/rccl.h>
#endif

namespace alpaka::tensor
{
    class RCCLProvider : public ICollectiveProvider
    {
    public:
        struct Diagnostics
        {
            int deviceCount = 0;
            int activeDevice = -1;
            std::string activeDeviceName;
            std::vector<std::string> deviceNames;
            std::vector<std::vector<bool>> peerAccess;
        };

        RCCLProvider() = default;
        ~RCCLProvider() override;

        std::string getBackendName() const override;
        bool supportsOperation(OpType op) const override;
        bool isActive() const override;

        Diagnostics diagnostics() const;

        bool supportsPattern(ops::CollectivePattern pattern) const override;
        bool supportsReduction(ops::CollectiveReduction reduction) const override;
        bool supportsDataType(ops::CollectiveDataType dtype) const override;

        std::size_t worldSize() const override;
        std::size_t worldRank() const override;

    protected:
        OpStatus allreduce_impl(
            CollectiveExecutionContext const& ctx,
            void const* sendBuffer,
            void* recvBuffer,
            std::size_t elementCount,
            ops::CollectiveDataType dtype,
            ops::CollectiveReduction reduction,
            bool async) override;

        OpStatus broadcast_impl(
            CollectiveExecutionContext const& ctx,
            void* buffer,
            std::size_t elementCount,
            ops::CollectiveDataType dtype,
            std::size_t rootRank,
            bool async) override;

        OpStatus barrier_impl(CollectiveExecutionContext const& ctx) override;

    private:
        void ensureInitialized() const;
        void finalize() noexcept;

#ifdef ALPAKA_HAS_RCCL
    ncclDataType_t mapDataType(ops::CollectiveDataType dtype) const;
    ncclRedOp_t mapReduction(ops::CollectiveReduction reduction) const;
#endif

    private:
#ifdef ALPAKA_HAS_RCCL
    mutable ncclComm_t comm_ = nullptr;
        mutable bool initialized_ = false;
        mutable bool active_ = false;
        mutable int worldSize_ = 1;
        mutable int worldRank_ = 0;
#else
        static constexpr bool active_ = false;
#endif
    };
} // namespace alpaka::tensor

#ifdef ALPAKA_HAS_RCCL
namespace alpaka::tensor
{
    inline RCCLProvider::~RCCLProvider()
    {
        finalize();
    }

    inline std::string RCCLProvider::getBackendName() const
    {
        return "RCCL (HIP Collective)";
    }

    inline bool RCCLProvider::supportsOperation(OpType op) const
    {
        return op == OpType::Collective;
    }

    inline bool RCCLProvider::isActive() const
    {
        ensureInitialized();
        return active_;
    }

    inline bool RCCLProvider::supportsPattern(ops::CollectivePattern pattern) const
    {
        return pattern == ops::CollectivePattern::AllReduce || pattern == ops::CollectivePattern::Broadcast
               || pattern == ops::CollectivePattern::Barrier;
    }

    inline bool RCCLProvider::supportsReduction(ops::CollectiveReduction reduction) const
    {
        switch(reduction)
        {
        case ops::CollectiveReduction::Sum:
        case ops::CollectiveReduction::Product:
        case ops::CollectiveReduction::Maximum:
        case ops::CollectiveReduction::Minimum:
            return true;
        default:
            return false;
        }
    }

    inline bool RCCLProvider::supportsDataType(ops::CollectiveDataType dtype) const
    {
        switch(dtype)
        {
        case ops::CollectiveDataType::Float16:
        case ops::CollectiveDataType::BFloat16:
        case ops::CollectiveDataType::Float32:
        case ops::CollectiveDataType::Float64:
        case ops::CollectiveDataType::Int8:
        case ops::CollectiveDataType::UInt8:
        case ops::CollectiveDataType::Int32:
        case ops::CollectiveDataType::UInt32:
        case ops::CollectiveDataType::Int64:
            return true;
        default:
            return false;
        }
    }

    inline std::size_t RCCLProvider::worldSize() const
    {
        ensureInitialized();
        return static_cast<std::size_t>(worldSize_);
    }

    inline std::size_t RCCLProvider::worldRank() const
    {
        ensureInitialized();
        return static_cast<std::size_t>(worldRank_);
    }

    inline RCCLProvider::Diagnostics RCCLProvider::diagnostics() const
    {
        Diagnostics diag;

        int deviceCount = 0;
        hipError_t hipStatus = hipGetDeviceCount(&deviceCount);
        if(hipStatus != hipSuccess || deviceCount <= 0)
        {
            return diag;
        }

        diag.deviceCount = deviceCount;

        int activeDevice = -1;
        hipStatus = hipGetDevice(&activeDevice);
        if(hipStatus == hipSuccess)
        {
            diag.activeDevice = activeDevice;
        }

        diag.deviceNames.resize(static_cast<std::size_t>(deviceCount));
        diag.peerAccess.assign(static_cast<std::size_t>(deviceCount), std::vector<bool>(static_cast<std::size_t>(deviceCount), false));

        for(int dev = 0; dev < deviceCount; ++dev)
        {
            hipDeviceProp_t props{};
            hipStatus = hipGetDeviceProperties(&props, dev);
            if(hipStatus == hipSuccess)
            {
                diag.deviceNames[static_cast<std::size_t>(dev)] = props.name;
                if(dev == diag.activeDevice)
                {
                    diag.activeDeviceName = props.name;
                }
            }
            else
            {
                diag.deviceNames[static_cast<std::size_t>(dev)] = "unknown";
            }
        }

        for(int src = 0; src < deviceCount; ++src)
        {
            for(int dst = 0; dst < deviceCount; ++dst)
            {
                if(src == dst)
                {
                    diag.peerAccess[static_cast<std::size_t>(src)][static_cast<std::size_t>(dst)] = true;
                    continue;
                }

                int canAccess = 0;
                hipStatus = hipDeviceCanAccessPeer(&canAccess, src, dst);
                if(hipStatus == hipSuccess)
                {
                    diag.peerAccess[static_cast<std::size_t>(src)][static_cast<std::size_t>(dst)] = (canAccess != 0);
                }
            }
        }

        return diag;
    }

    inline ncclDataType_t RCCLProvider::mapDataType(ops::CollectiveDataType dtype) const
    {
        switch(dtype)
        {
        case ops::CollectiveDataType::Float16:
            return ncclFloat16;
        case ops::CollectiveDataType::BFloat16:
            return ncclBfloat16;
        case ops::CollectiveDataType::Float32:
            return ncclFloat;
        case ops::CollectiveDataType::Float64:
            return ncclDouble;
        case ops::CollectiveDataType::Int8:
            return ncclInt8;
        case ops::CollectiveDataType::UInt8:
            return ncclUint8;
        case ops::CollectiveDataType::Int32:
            return ncclInt;
        case ops::CollectiveDataType::UInt32:
            return ncclUint32;
        case ops::CollectiveDataType::Int64:
            return ncclInt64;
        default:
            throw std::invalid_argument("RCCL does not support requested collective datatype");
        }
    }

    inline ncclRedOp_t RCCLProvider::mapReduction(ops::CollectiveReduction reduction) const
    {
        switch(reduction)
        {
        case ops::CollectiveReduction::Sum:
            return ncclSum;
        case ops::CollectiveReduction::Product:
            return ncclProd;
        case ops::CollectiveReduction::Maximum:
            return ncclMax;
        case ops::CollectiveReduction::Minimum:
            return ncclMin;
        default:
            throw std::invalid_argument("RCCL reduction not supported");
        }
    }

    inline void RCCLProvider::ensureInitialized() const
    {
        if(initialized_)
            return;

        int deviceId = 0;
        hipError_t hipStatus = hipGetDevice(&deviceId);
        if(hipStatus != hipSuccess)
        {
            initialized_ = true;
            active_ = false;
            return;
        }

        ncclResult_t ncclStatus = ncclCommInitAll(&comm_, 1, &deviceId);
        if(ncclStatus != ncclSuccess)
        {
            initialized_ = true;
            active_ = false;
            comm_ = nullptr;
            return;
        }

        worldSize_ = 1;
        worldRank_ = 0;
        active_ = true;
        initialized_ = true;
    }

    inline void RCCLProvider::finalize() noexcept
    {
        if(comm_ != nullptr)
        {
            ncclCommDestroy(comm_);
            comm_ = nullptr;
        }
        initialized_ = false;
        active_ = false;
    }

    inline OpStatus RCCLProvider::allreduce_impl(
        CollectiveExecutionContext const& ctx,
        void const* sendBuffer,
        void* recvBuffer,
        std::size_t elementCount,
        ops::CollectiveDataType dtype,
        ops::CollectiveReduction reduction,
        bool async)
    {
        ensureInitialized();
        if(!active_)
            return OpStatus::Unsupported;

        hipStream_t hipStream = ctx.nativeQueue ? static_cast<hipStream_t>(ctx.nativeQueue) : nullptr;
        ncclResult_t status = ncclAllReduce(
            sendBuffer,
            recvBuffer,
            elementCount,
            mapDataType(dtype),
            mapReduction(reduction),
            comm_,
            hipStream);
        if(status != ncclSuccess)
            return OpStatus::Error;

        if(!async)
        {
            if(hipStream != nullptr)
            {
                hipError_t hipStatus = hipStreamSynchronize(hipStream);
                if(hipStatus != hipSuccess)
                    return OpStatus::Error;
            }
            else
            {
                hipError_t hipStatus = hipDeviceSynchronize();
                if(hipStatus != hipSuccess)
                    return OpStatus::Error;
            }
        }
        return OpStatus::Success;
    }

    inline OpStatus RCCLProvider::broadcast_impl(
        CollectiveExecutionContext const& ctx,
        void* buffer,
        std::size_t elementCount,
        ops::CollectiveDataType dtype,
        std::size_t rootRank,
        bool async)
    {
        ensureInitialized();
        if(!active_)
            return OpStatus::Unsupported;

        hipStream_t hipStream = ctx.nativeQueue ? static_cast<hipStream_t>(ctx.nativeQueue) : nullptr;
        ncclResult_t status = ncclBroadcast(
            static_cast<void const*>(buffer),
            buffer,
            elementCount,
            mapDataType(dtype),
            static_cast<int>(rootRank),
            comm_,
            hipStream);
        if(status != ncclSuccess)
            return OpStatus::Error;

        if(!async)
        {
            if(hipStream != nullptr)
            {
                hipError_t hipStatus = hipStreamSynchronize(hipStream);
                if(hipStatus != hipSuccess)
                    return OpStatus::Error;
            }
            else
            {
                hipError_t hipStatus = hipDeviceSynchronize();
                if(hipStatus != hipSuccess)
                    return OpStatus::Error;
            }
        }
        return OpStatus::Success;
    }

    inline OpStatus RCCLProvider::barrier_impl(CollectiveExecutionContext const& ctx)
    {
        ensureInitialized();
        if(!active_)
            return OpStatus::Unsupported;

        hipStream_t hipStream = ctx.nativeQueue ? static_cast<hipStream_t>(ctx.nativeQueue) : nullptr;
        if(hipStream != nullptr)
        {
            hipError_t hipStatus = hipStreamSynchronize(hipStream);
            if(hipStatus != hipSuccess)
                return OpStatus::Error;
        }
        else
        {
            hipError_t hipStatus = hipDeviceSynchronize();
            if(hipStatus != hipSuccess)
                return OpStatus::Error;
        }
        return OpStatus::Success;
    }
} // namespace alpaka::tensor
#else
namespace alpaka::tensor
{
    inline RCCLProvider::~RCCLProvider() = default;

    inline std::string RCCLProvider::getBackendName() const
    {
        return "RCCL (unavailable)";
    }

    inline bool RCCLProvider::supportsOperation(OpType) const
    {
        return false;
    }

    inline bool RCCLProvider::isActive() const
    {
        return false;
    }

    inline bool RCCLProvider::supportsPattern(ops::CollectivePattern) const
    {
        return false;
    }

    inline bool RCCLProvider::supportsReduction(ops::CollectiveReduction) const
    {
        return false;
    }

    inline bool RCCLProvider::supportsDataType(ops::CollectiveDataType) const
    {
        return false;
    }

    inline std::size_t RCCLProvider::worldSize() const
    {
        return 1;
    }

    inline std::size_t RCCLProvider::worldRank() const
    {
        return 0;
    }

    inline RCCLProvider::Diagnostics RCCLProvider::diagnostics() const
    {
        return Diagnostics{};
    }

    inline void RCCLProvider::ensureInitialized() const
    {
    }

    inline void RCCLProvider::finalize() noexcept
    {
    }

    inline OpStatus RCCLProvider::allreduce_impl(
        CollectiveExecutionContext const&,
        void const*,
        void*,
        std::size_t,
        ops::CollectiveDataType,
        ops::CollectiveReduction,
        bool)
    {
        return OpStatus::Unsupported;
    }

    inline OpStatus RCCLProvider::broadcast_impl(
        CollectiveExecutionContext const&,
        void*,
        std::size_t,
        ops::CollectiveDataType,
        std::size_t,
        bool)
    {
        return OpStatus::Unsupported;
    }

    inline OpStatus RCCLProvider::barrier_impl(CollectiveExecutionContext const&)
    {
        return OpStatus::Unsupported;
    }
} // namespace alpaka::tensor
#endif
