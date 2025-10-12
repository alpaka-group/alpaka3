/* RCCL collective provider implementation for HIP backends.
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <alpaka/tensor/providers/EnabledVendorLibs.hpp>
#include <alpaka/tensor/providers/collective/CollectiveProviderInterface.hpp>

#include <optional>
#include <vector>

#ifdef ALPAKA_HAS_RCCL
#    include <hip/hip_runtime_api.h>
#    if __has_include(<rccl/rccl.h>)
#        include <rccl/rccl.h>
#    else
#        include <rccl.h>
#    endif
#endif

namespace alpaka::tensor::collective
{
    namespace detail
    {
#ifdef ALPAKA_HAS_RCCL
        [[nodiscard]] inline std::optional<ncclDataType_t> toRccl(DataType type)
        {
            switch(type)
            {
            case DataType::Float32:
                return ncclFloat;
            case DataType::Float64:
                return ncclDouble;
            case DataType::Int32:
                return ncclInt32;
            case DataType::Int64:
                return ncclInt64;
            case DataType::UInt8:
                return ncclUint8;
            }
            return std::nullopt;
        }

        [[nodiscard]] inline std::optional<ncclRedOp_t> toRccl(ReduceOp op)
        {
            switch(op)
            {
            case ReduceOp::Sum:
                return ncclSum;
            case ReduceOp::Prod:
                return ncclProd;
            case ReduceOp::Min:
                return ncclMin;
            case ReduceOp::Max:
                return ncclMax;
            }
            return std::nullopt;
        }
#endif
    } // namespace detail

    class RcclCollectiveProvider : public ICollectiveProvider
    {
    public:
        RcclCollectiveProvider() = default;

        ~RcclCollectiveProvider() override
        {
#ifdef ALPAKA_HAS_RCCL
            for(auto& comm : comms_)
            {
                if(comm != nullptr)
                {
                    ncclCommDestroy(comm);
                    comm = nullptr;
                }
            }
#endif
        }

        std::string_view name() const override
        {
            return "RCCL";
        }

        bool isActive() const override
        {
            return active_;
        }

        std::size_t worldSize() const override
        {
            return devices_.size();
        }

        [[nodiscard]] OpStatus initialize(GroupConfig const& config) override
        {
            reset();
            if(!EnabledVendorLibs::hasRCCL)
                return OpStatus::Unsupported;

#ifdef ALPAKA_HAS_RCCL
            auto deviceIds = config.deviceIds;
            if(deviceIds.empty())
            {
                int count = 0;
                if(hipGetDeviceCount(&count) != hipSuccess || count <= 0)
                {
                    return OpStatus::Error;
                }
                deviceIds.resize(static_cast<std::size_t>(count));
                for(int i = 0; i < count; ++i)
                {
                    deviceIds[static_cast<std::size_t>(i)] = i;
                }
            }
            devices_ = deviceIds;
            comms_.resize(devices_.size());
            auto result = ncclCommInitAll(comms_.data(), static_cast<int>(devices_.size()), devices_.data());
            if(result != ncclSuccess)
            {
                reset();
                return OpStatus::Error;
            }
            active_ = true;
            return OpStatus::Success;
#else
            static_cast<void>(config);
            return OpStatus::Unsupported;
#endif
        }

        [[nodiscard]] OpStatus allReduce(AllReduceRequest const& request) override
        {
            if(!EnabledVendorLibs::hasRCCL)
                return OpStatus::Unsupported;
#ifdef ALPAKA_HAS_RCCL
            if(!active_ || devices_.empty())
                return OpStatus::Error;

            auto dtype = detail::toRccl(request.dataType);
            auto rop = detail::toRccl(request.reduceOp);
            if(!dtype || !rop)
                return OpStatus::Unsupported;

            auto const deviceCount = devices_.size();
            if(request.buffers.recv.size() != deviceCount)
                return OpStatus::Error;
            if(!request.buffers.inPlace && request.buffers.send.size() != deviceCount)
                return OpStatus::Error;
            if(!request.buffers.streams.empty() && request.buffers.streams.size() != deviceCount)
                return OpStatus::Error;

            auto getStream = [&](std::size_t idx) -> hipStream_t
            {
                if(request.buffers.streams.empty() || request.buffers.streams[idx] == nullptr)
                    return nullptr;
                return reinterpret_cast<hipStream_t>(request.buffers.streams[idx]);
            };

            auto getSendPtr = [&](std::size_t idx) -> void const*
            {
                if(request.buffers.inPlace)
                    return request.buffers.recv[idx];
                if(request.buffers.send.empty() || request.buffers.send[idx] == nullptr)
                    return request.buffers.recv[idx];
                return request.buffers.send[idx];
            };

            auto getRecvPtr = [&](std::size_t idx) -> void* { return request.buffers.recv[idx]; };

            ncclResult_t launchStatus = ncclGroupStart();
            if(launchStatus != ncclSuccess)
                return OpStatus::Error;

            ncclResult_t opStatus = ncclSuccess;
            for(std::size_t idx = 0; idx < deviceCount; ++idx)
            {
                if(hipSetDevice(devices_[idx]) != hipSuccess)
                {
                    opStatus = ncclSystemError;
                    break;
                }

                auto comm = comms_[idx];
                auto sendPtr = getSendPtr(idx);
                auto recvPtr = getRecvPtr(idx);
                auto stream = getStream(idx);
                opStatus = ncclAllReduce(sendPtr, recvPtr, request.elementCount, *dtype, *rop, comm, stream);
                if(opStatus != ncclSuccess)
                    break;
            }

            auto groupStatus = ncclGroupEnd();
            if(opStatus != ncclSuccess || groupStatus != ncclSuccess)
                return OpStatus::Error;

            return OpStatus::Success;
#else
            static_cast<void>(request);
            return OpStatus::Unsupported;
#endif
        }

    private:
        void reset()
        {
#ifdef ALPAKA_HAS_RCCL
            for(auto& comm : comms_)
            {
                if(comm != nullptr)
                {
                    ncclCommDestroy(comm);
                    comm = nullptr;
                }
            }
            comms_.clear();
#endif
            devices_.clear();
            active_ = false;
        }

#ifdef ALPAKA_HAS_RCCL
        std::vector<ncclComm_t> comms_{};
#else
        std::vector<void*> comms_{};
#endif
        std::vector<int> devices_{};
        bool active_{false};
    };
} // namespace alpaka::tensor::collective
