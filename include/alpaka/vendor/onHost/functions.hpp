/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

namespace alpaka::vendor::onHost
{
    struct Unimplemented
    {
    };

    struct ScanMode
    {
        struct Inclusive
        {
        };

        static constexpr Inclusive inclusive{};

        struct Exclusive
        {
        };

        static constexpr Exclusive exclusive{};
    };

    template<typename T_FnImpl = Unimplemented>
    struct Scan
        : ScanMode
        , T_FnImpl
    {
        constexpr size_t getBufferSize(
            auto& queue,
            auto scanType,
            alpaka::concepts::IMdSpan auto&& input,
            alpaka::concepts::IMdSpan auto&& output) const
        {
            return static_cast<T_FnImpl const&>(*this)
                .getBufferSize(queue, scanType, ALPAKA_FORWARD(input), ALPAKA_FORWARD(output));
        }

        constexpr decltype(auto) operator()(
            auto& queue,
            auto scanType,
            alpaka::concepts::IMdSpan auto& tmp,
            alpaka::concepts::IMdSpan auto&& input,
            alpaka::concepts::IMdSpan auto&& output) const
        {
            return static_cast<T_FnImpl const&>(
                *this)(queue, scanType, tmp, ALPAKA_FORWARD(input), ALPAKA_FORWARD(output));
        }
    };

} // namespace alpaka::vendor::onHost
