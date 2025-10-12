/* Collective communication type system used by tensor providers.
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <cstddef>
#include <span>
#include <vector>

namespace alpaka::tensor::collective
{
    enum class DataType
    {
        Float32,
        Float64,
        Int32,
        Int64,
        UInt8
    };

    enum class ReduceOp
    {
        Sum,
        Prod,
        Min,
        Max
    };

    enum class Operation
    {
        AllReduce,
        Broadcast,
        AllGather,
        ReduceScatter,
        Barrier
    };

    struct GroupConfig
    {
        std::vector<int> deviceIds{};
        bool multiProcess = false;
        int worldRank = 0;
        int worldSize = 0;
    };

    struct MultiDeviceBuffers
    {
        std::span<void const*> send = {};
        std::span<void*> recv = {};
        std::span<void*> streams = {};
        bool inPlace = false;
    };

    struct AllReduceRequest
    {
        MultiDeviceBuffers buffers;
        std::size_t elementCount = 0;
        DataType dataType = DataType::Float32;
        ReduceOp reduceOp = ReduceOp::Sum;
    };

    struct BroadcastRequest
    {
        MultiDeviceBuffers buffers;
        std::size_t elementCount = 0;
        DataType dataType = DataType::Float32;
        int rootRank = 0;
    };

    struct BarrierRequest
    {
        std::span<void*> streams{};
    };
} // namespace alpaka::tensor::collective
