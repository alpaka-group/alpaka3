/* Copyright 2025 René Widera, Simeon Ehrig
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/Vec.hpp"
#include "alpaka/mem/Alignment.hpp"
#include "alpaka/trait.hpp"

#include <concepts>
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

    template<typename T, typename T_ValueType = alpaka::NotRequired>
    concept MdSpan = alpaka::isMdSpan_v<T>
                     && (std::same_as<T_ValueType, trait::GetValueType_t<std::decay_t<T>>>
                         || std::same_as<T_ValueType, alpaka::NotRequired>);

    template<typename T>
    concept Reference = std::is_reference_v<T>;

    template<typename TMdSpan, typename TExpected>
    concept ExpectedValueType = std::same_as<TExpected, TMdSpan> || std::same_as<TExpected, alpaka::NotRequired>;

    namespace impl
    {
        /** Interface concept that describes how multidimensional memory can be accessed on the host and device side.
         *
         * @details
         *
         * An object of type `alpaka::mdspan` does not store any information about the storage location, e.g., whether
         * the memory is located on a CPU or a GPU. The interface corresponds to that of a standard library container
         * with continuous memory, but has some differences to support multidimensional memory. For example, instead of
         * the member function `size()`, which returns the 1D size, `alpaka::mdspan` like objects provides the function
         *`getExtents()`, which returns the size of each dimension.
         *
         * @param t Object of type `alpaka::mdspan`. May or may not have a const modifier.
         * @param mut_t Mutable object of type `alpaka::mdspan`. Does not have a const modifier.
         * @param const_t Constant object of type `alpaka::mdspan`. Does have a const modifier.
         * @param vec Vector with the same number of elements as the dimension of the `alpaka::mdspan` like object.
         * Used to call the access operator.
         *
         *  @section components Components
         *
         * An `alpaka::mdspan` like object contains 4 components:
         * - A pointer to the actual memory.
         * - An extents object that describes the number of dimensions and their respective sizes.
         * - A pitch object that specifies how many bytes are required to jump to the next element in each dimension.
         * - An alignment object that describes how the elements are aligned in memory, see:
         * alpaka::concepts::Alignment
         *
         * @section membertypes Member types
         * - <b>T::value_type</b>: The element type. Can be const or not.
         * - <b>T::reference</b>: The element reference type is either const or non-const, depending on
         *`T::value_type`.
         * - <b>T::const_reference</b>: The constant reference type for an element. Always const.
         * - <b>T::pointer</b>: The element pointer type is either const or non-const, depending on
         *`T::value_type`.
         * - <b>T::const_pointer</b>: The constant pointer type for an element. Always const.
         * - <b>T::index_type</b>: The index type of the pitch.
         *
         * @note The access operator [] with an integral as an argument is only available if the dimension is one.
         **/
        template<typename T, typename MutT, typename ConstT>
        concept IMdSpan = requires(T t, MutT mut_t, ConstT const_t, alpaka::Vec<uint32_t, T::dim()> vec) {
            typename T::value_type;
            typename T::reference;
            typename T::const_reference;
            typename T::pointer;
            typename T::const_pointer;
            typename T::index_type;

            requires std::copy_constructible<T>;
            requires std::assignable_from<MutT&, MutT&>;
            // TODO(SimeonEhrig): check if assignment fulfills const correctness
            // requires std::assignable_from<decltype(const_t)&, decltype(const_t)&>;
            requires std::movable<MutT>;

            { T::dim() } -> std::same_as<uint32_t>;
            { *mut_t } -> std::same_as<typename T::reference>;
            { *const_t } -> std::same_as<typename T::const_reference>;
            { mut_t.data() } -> std::same_as<typename T::pointer>;
            { const_t.data() } -> std::same_as<typename T::const_pointer>;
            /// @todo check for a MDIterator concept
            t.begin();
            t.end();
            t.cbegin();
            t.cend();

            { mut_t[vec] } -> std::same_as<typename T::reference>;
            { const_t[vec] } -> std::same_as<typename T::const_reference>;
            // only if MdSpan like object is 1D, the access operator with an integral is available
            requires(T::dim() > 1) || requires {
                { mut_t[0] } -> std::same_as<typename T::reference>;
            };
            requires(T::dim() > 1) || requires {
                { const_t[0] } -> std::same_as<typename T::const_reference>;
            };

            /// @todo add getSlice, getConstSlice and getView, getConstView functions

            { t.getAlignment() } -> alpaka::concepts::Alignment;
            t.getExtents();
            t.getPitches();
        };

        /** Interface concept that describes how multidimensional memory can be accessed on the host and device side.
         * An `alpaka::view` like object contains information about the device(s) to which it is connected. The
         * `alpaka::view` like object has no memory ownership, and therefore it does not manage the memory lifetime.
         *
         * The concept fulfills all requirements of the alpaka::concepts::impl::IMdSpan concept and therefore offers
         * the same interface.
         *
         **/
        template<typename T, typename MutT, typename ConstT>
        concept IView = requires(T t) {
            requires IMdSpan<T, MutT, ConstT>;
            // TODO(SimeonEhrig): { t.getApi() } -> alpaka::concepts::Api
            // cannot find concept
            // looks like I have care about the correct ordering of concept includes
            t.getApi();
        };

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
        template<typename T, typename MutT, typename ConstT>
        concept IBuffer = requires(T t) {
            requires IView<T, MutT, ConstT>;
            t.addDestructorAction(alpaka::concepts::empty_callable);
            t.destructorWaitFor(alpaka::concepts::empty_callable);
        };
    } // namespace impl

    /** Interface concept that describes how multidimensional memory can be accessed on the host and device side.
     * An object of type `alpaka::mdspan` does not store any information about the storage location, e.g., whether
     * the memory is located on a CPU or a GPU.
     *
     * \attention Use `alpaka::IMdSpan` to restrict types in your code. The actual interface is described in
     * alpaka::concepts::impl::IMdSpan.
     *
     **/
    template<typename T, typename T_ValueType = alpaka::NotRequired>
    concept IMdSpan = requires {
        requires impl::IMdSpan<
            std::remove_reference_t<T>,
            std::remove_const_t<std::remove_reference_t<T>>,
            std::add_const_t<std::remove_reference_t<T>>>;
        requires ExpectedValueType<trait::GetValueType_t<std::decay_t<T>>, T_ValueType>;
    };


    /** Interface concept that describes how multidimensional memory can be accessed on the host and device side.
     * An `alpaka::view` like object contains information about the device(s) to which it is connected. The
     * `alpaka::view` like object has no memory ownership, and therefore it does not manage the memory lifetime.
     *
     * \attention Use `alpaka::IView` to restrict types in your code. The actual interface is described in
     * alpaka::concepts::impl::IView.
     *
     **/
    template<typename T, typename T_ValueType = alpaka::NotRequired>
    concept IView = requires(T t) {
        requires impl::IView<
            std::remove_reference_t<T>,
            std::remove_const_t<std::remove_reference_t<T>>,
            std::add_const_t<std::remove_reference_t<T>>>;
        requires ExpectedValueType<trait::GetValueType_t<std::decay_t<T>>, T_ValueType>;
    };

    /** Interface concept that describes how multidimensional memory can be accessed on the host and device side.
     * An `alpaka::buffer` like object contains information about the device(s) to which it is connected. The
     * `alpaka::buffer` like object has memory ownership and therefore manages memory lifetime according to the RAII
     * principle.
     *
     * \attention Use `alpaka::IBuffer` to restrict types in your code. The actual interface is described in
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
