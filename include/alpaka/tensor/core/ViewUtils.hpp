/* View/shape utilities (Phase 0)
 * - Contiguity checks using TensorDescriptor
 * - Shape equality helper
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <alpaka/tensor/core/TensorDescriptor.hpp>

namespace alpaka::tensor::viewutils
{
    template<typename T, std::size_t Rank, typename Device>
    inline bool isContiguous(tensor::Tensor<T, Rank, Device> const& t) noexcept
    {
        auto d = tensor::makeDescriptor(t);
        return d.isContiguous();
    }

    template<typename T, std::size_t Rank, typename Device>
    inline bool sameShape(tensor::Tensor<T, Rank, Device> const& a, tensor::Tensor<T, Rank, Device> const& b) noexcept
    {
        auto const& sa = a.shape();
        auto const& sb = b.shape();
        for(std::size_t i = 0; i < Rank; ++i)
            if(sa[i] != sb[i])
                return false;
        return true;
    }
} // namespace alpaka::tensor::viewutils
