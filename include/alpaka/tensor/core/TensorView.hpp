/* Copyright 2025 Alpaka Tensor Library Contributors
 * SPDX-License-Identifier: MPL-2.0
 *
 */

#pragma once

#include <cstddef>

namespace alpaka::tensor
{

    // Forward declaration matching new Tensor signature
    template<typename T, std::size_t Rank, typename TDevice>
    class Tensor;

    template<typename T, std::size_t Rank, typename TDevice>
    class TensorView
    {
    private:
        Tensor<T, Rank, TDevice>* tensor_;

    public:
        // Constructor from tensor reference
        explicit TensorView(Tensor<T, Rank, TDevice>& tensor) : tensor_(&tensor)
        {
        }

        // Get the underlying tensor
        Tensor<T, Rank, TDevice>& getTensor()
        {
            return *tensor_;
        }

        Tensor<T, Rank, TDevice> const& getTensor() const
        {
            return *tensor_;
        }

        // Auto-convert to tensor when needed (for backward compatibility)
        operator Tensor<T, Rank, TDevice>&()
        {
            return *tensor_;
        }

        operator Tensor<T, Rank, TDevice> const&() const
        {
            return *tensor_;
        }

        // Method chaining will be implemented via free functions in ops namespace
        // This keeps the view lightweight and avoids circular dependencies
    };

} // namespace alpaka::tensor
