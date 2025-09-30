/* Forward declarations and convenience aliases for strongly typed tensors.
 * Provides lightweight access to Tensor<T, Rank, Device> without pulling in
 * the full TensorCore implementation. Include this header when only the
 * type signatures are required (e.g., provider interfaces, concepts).
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <cstddef>

namespace alpaka::tensor
{
    template<typename T, std::size_t Rank, typename Device>
    class Tensor;

    template<typename T, typename Device>
    using Tensor1D = Tensor<T, 1, Device>;

    template<typename T, typename Device>
    using Tensor2D = Tensor<T, 2, Device>;

    template<typename T, typename Device>
    using Tensor3D = Tensor<T, 3, Device>;

    template<typename T, typename Device>
    using Tensor4D = Tensor<T, 4, Device>;
} // namespace alpaka::tensor
