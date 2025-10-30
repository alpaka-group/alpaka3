/* Copyright 2025 Simeon Ehrig
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/Vec.hpp"
#include "alpaka/mem/Alignment.hpp"
#include "alpaka/mem/concepts/ExpectedValueType.hpp"
#include "alpaka/trait.hpp"

#include <concepts>

namespace alpaka::concepts
{
    namespace impl
    {
        template<typename T, typename T_Mut, typename T_Const>
        concept IDataSource
            = requires(T t, T_Mut mut_t, T_Const const_t, alpaka::Vec<typename T::index_type, T::dim()> vec) {
                  typename T::value_type;
                  typename T::index_type;

                  requires std::movable<T_Mut>;

                  /// The bool operator returns true if the access operator returns valid values. For example, memory
                  /// access may be invalid after moving the DataSource.
                  static_cast<bool>(t);

                  // only if no reference is not defined, check if the access operator returns a value
                  requires requires { typename T::reference; } || requires {
                      { mut_t[vec] } -> std::same_as<typename T::value_type>;
                  };
                  requires requires { typename T::const_reference; } || requires {
                      { const_t[vec] } -> std::same_as<typename T::value_type>;
                  };

                  // only if no reference type is defined and for 1D, the access operator with an integral is available
                  requires requires { typename T::reference; } || requires { requires(T::dim() > 1); } || requires {
                      { mut_t[0] } -> std::same_as<typename T::value_type>;
                  };
                  requires requires { typename T::const_reference; } || requires { requires(T::dim() > 1); }
                               || requires {
                                      { const_t[0] } -> std::same_as<typename T::value_type>;
                                  };

                  // typically the alignment of the value_type.
                  { t.getAlignment() } -> alpaka::concepts::Alignment;
                  /** @todo implement concept alpaka::concepts::Extents and use it as return value
                   * @todo in general a generator is not required to have extents but our algorithm e.g. onHost::reduce
                   *will not work without extents
                   **/
                  t.getExtents();
                  /// @todo implement concept alpaka::concepts::Pitches and use it as return value
                  t.getPitches();
              };
    } // namespace impl

    template<typename T, typename T_ValueType = alpaka::NotRequired>
    concept IDataSource = requires {
        requires impl::IDataSource<
            std::remove_reference_t<T>,
            std::remove_const_t<std::remove_reference_t<T>>,
            std::add_const_t<std::remove_reference_t<T>>>;
        requires ExpectedValueType<trait::GetValueType_t<std::decay_t<T>>, T_ValueType>;
    };
} // namespace alpaka::concepts
