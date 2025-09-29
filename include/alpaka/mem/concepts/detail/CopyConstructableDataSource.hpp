/* Copyright 2025 Simeon Ehrig
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/mem/trait.hpp"

#include <concepts>

namespace alpaka::internal::concepts
{
    /** Check whether the copy constructor of the data source `T` respects the const correctness of the data type.
     *
     * @details
     * Data sources have a data type that can be mutable or constant (marked with const). The following copies or
     * assignments to a new object with the corresponding data type are possible:
     *  - mutable -> mutable
     *  - const -> const
     *  - mutable -> const
     **/
    template<typename T>
    concept CopyConstructableDataSource = requires {
        requires alpaka::internal::CopyConstructableDataSource<std::decay_t<T>>::value;
        /// copy constructor inner mutable -> inner mutable
        requires std::constructible_from<
            typename alpaka::internal::CopyConstructableDataSource<std::decay_t<T>>::InnerMutable,
            typename alpaka::internal::CopyConstructableDataSource<std::decay_t<T>>::InnerMutable&>;
        /// copy constructor inner const -> inner const
        requires std::constructible_from<
            typename alpaka::internal::CopyConstructableDataSource<std::decay_t<T>>::InnerConst,
            typename alpaka::internal::CopyConstructableDataSource<std::decay_t<T>>::InnerConst&>;
        /// copy constructor inner mutable -> inner const
        requires std::constructible_from<
            typename alpaka::internal::CopyConstructableDataSource<std::decay_t<T>>::InnerConst,
            typename alpaka::internal::CopyConstructableDataSource<std::decay_t<T>>::InnerMutable&>;
        /// not allowed: copy constructor inner const -> inner mutable
        requires !std::constructible_from<
            typename alpaka::internal::CopyConstructableDataSource<std::decay_t<T>>::InnerMutable,
            typename alpaka::internal::CopyConstructableDataSource<std::decay_t<T>>::InnerConst&>;
        /// copy assignment inner mutable -> inner mutable
        requires std::assignable_from<
            typename alpaka::internal::CopyConstructableDataSource<std::decay_t<T>>::InnerMutable&,
            typename alpaka::internal::CopyConstructableDataSource<std::decay_t<T>>::InnerMutable>;
        /// copy assignment inner const -> inner const
        requires std::assignable_from<
            typename alpaka::internal::CopyConstructableDataSource<std::decay_t<T>>::InnerConst&,
            typename alpaka::internal::CopyConstructableDataSource<std::decay_t<T>>::InnerConst>;
        /// copy assignment inner mutable -> inner const
        requires std::assignable_from<
            typename alpaka::internal::CopyConstructableDataSource<std::decay_t<T>>::InnerConst&,
            typename alpaka::internal::CopyConstructableDataSource<std::decay_t<T>>::InnerMutable>;
        /// not allowed: copy assignment inner const -> inner mutable
        requires !std::assignable_from<
            typename alpaka::internal::CopyConstructableDataSource<std::decay_t<T>>::InnerMutable&,
            typename alpaka::internal::CopyConstructableDataSource<std::decay_t<T>>::InnerConst>;
    };
} // namespace alpaka::internal::concepts
