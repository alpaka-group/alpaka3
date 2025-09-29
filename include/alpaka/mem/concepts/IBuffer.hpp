/* Copyright 2025 Simeon Ehrig
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/mem/concepts/ExpectedValueType.hpp"
#include "alpaka/mem/concepts/IView.hpp"
#include "alpaka/trait.hpp"

#include <type_traits>

namespace alpaka::concepts
{
    /** Dummy function for concepts.
     *
     * Represent a callable without arguments and return value void. Required because nvcc could not handle empty
     * lambdas in concepts.
     */
    inline void empty_callable()
    {
    }

    namespace impl
    {
        /** Interface concept that describes how multidimensional memory can be accessed on the host and device side.
         * An `alpaka::buffer` like object contains information about the device(s) to which it is connected. The
         * `alpaka::buffer` like object has memory ownership and therefore manages memory lifetime according to the
         * RAII principle.
         *
         * The concept fulfills all requirements of the alpaka::concepts::impl::IView concept and therefore offers
         * the same interface.
         *
         * @section memberfunction member functions
         *
         * - <b>t.addDestructorAction</b>: Adds a destructor action to the shared buffer.
         * @code{.unparsed}
         *    The action will be executed when the buffer is destroyed.
         *    This can be used to add additional cleanup actions e.g. waiting on a specific queue.
         *    Actions are executed in FIFO order.
         * @endcode
         * - <b>t.destructorWaitFor</b>: Add an action to be executed when the shared_ptr is destroyed.
         **/
        template<typename T, typename T_Mut, typename T_Const>
        concept IBuffer = requires(T t) {
            requires IView<T, T_Mut, T_Const>;
            t.addDestructorAction(alpaka::concepts::empty_callable);
            t.destructorWaitFor(alpaka::concepts::empty_callable);
        };
    } // namespace impl

    /** Interface concept that describes how multidimensional memory can be accessed on the host and device side.
     * An `alpaka::buffer` like object contains information about the device(s) to which it is connected. The
     * `alpaka::buffer` like object has memory ownership and therefore manages memory lifetime according to the RAII
     * principle.
     *
     * @attention Use `alpaka::IBuffer` to restrict types in your code. The actual interface is described in
     * alpaka::concepts::impl::IBuffer.
     *
     **/
    template<typename T, typename T_ValueType = alpaka::NotRequired>
    concept IBuffer = requires(T t) {
        requires impl::IBuffer<
            std::remove_reference_t<T>,
            std::remove_const_t<std::remove_reference_t<T>>,
            std::add_const_t<std::remove_reference_t<T>>>;
        requires ExpectedValueType<trait::GetValueType_t<std::decay_t<T>>, T_ValueType>;
    };
} // namespace alpaka::concepts
