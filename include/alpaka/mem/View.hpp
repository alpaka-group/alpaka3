/* Copyright 2024 Bernhard Manfred Gruber, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/config.hpp"
#include "alpaka/interface.hpp"
#include "alpaka/internal.hpp"
#include "alpaka/mem/MdSpan.hpp"
#include "alpaka/mem/concepts.hpp"
#include "alpaka/onHost.hpp"
#include "alpaka/onHost/Handle.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <sstream>

namespace alpaka
{

    /** Non owning view to data
     *
     * This view is only holding a points to real data, copying the view is cheap.
     * Const-ness of the view is NOT propagated to the data region.
     */
    template<
        typename T_Api,
        typename T_Type,
        alpaka::concepts::Vector T_Extents,
        alpaka::concepts::Alignment T_MemAlignment = Alignment<>>
    struct View;

    template<typename T_ValueType, concepts::Alignment T_MemAlignment = Alignment<>>
    inline constexpr auto makeView(
        auto&& anyWithApi,
        T_ValueType* pointer,
        concepts::Vector auto const& extents,
        T_MemAlignment const memAlignment = T_MemAlignment{})
    {
        auto pitchMd = alpaka::calculatePitchesFromExtents<T_ValueType>(extents);
        return View{getApi(ALPAKA_FORWARD(anyWithApi)), pointer, extents, pitchMd, memAlignment};
    }

    inline constexpr auto makeView(auto&& any)
    {
        return View{
            getApi(any),
            onHost::data(any),
            onHost::getExtents(any),
            onHost::getPitches(any),
            alpaka::getAlignment(any)};
    }

    template<
        typename T_Api,
        typename T_Type,
        alpaka::concepts::Vector T_Extents,
        alpaka::concepts::Alignment T_MemAlignment>
    struct View : MdSpan<T_Type, typename T_Extents::UniVec, typename T_Extents::UniVec, T_MemAlignment>
    {
    private:
        using BaseMdSpan = MdSpan<T_Type, typename T_Extents::UniVec, typename T_Extents::UniVec, T_MemAlignment>;

    public:
        /** creates a view
         *
         * @param data handle to the physical data
         * @param extents M-dimensional extents in elements of the view. Must be <= number of elements in the dat
         * handle.
         */
        template<
            alpaka::concepts::HasApi T_Any,
            alpaka::concepts::Vector T_UserExtents,
            alpaka::concepts::Vector T_UserPitches>
        constexpr View(
            T_Any const& any,
            T_Type* data,
            T_UserExtents const& extents,
            T_UserPitches const& pitches,
            T_MemAlignment const memAlignment = T_MemAlignment{})
            : BaseMdSpan{
                  data,
                  typename ALPAKA_TYPEOF(extents)::UniVec{extents},
                  typename ALPAKA_TYPEOF(pitches)::UniVec{pitches},
                  memAlignment}
        {
            static_assert(
                isLosslessConvertible_v<typename T_UserPitches::type, typename T_UserExtents::type>,
                "extent type and pitch type must be lossless convertible");
        }

        constexpr View(View const&) = default;
        constexpr View(View&&) = default;
        constexpr View& operator=(View const&) = default;
        constexpr View& operator=(View&&) = default;

        static consteval T_Api getApi()
        {
            return T_Api{};
        }

        constexpr alpaka::concepts::MdSpan auto getMdSpan() const
        {
            return BaseMdSpan{*this};
        }

        /** create a read only view */
        constexpr auto getConstView() const
        {
            using ConstValueType = std::add_const_t<typename BaseMdSpan::value_type>;
            return View<T_Api, ConstValueType, T_Extents, T_MemAlignment>{
                T_Api{},
                static_cast<ConstValueType*>(this->data()),
                this->getExtents(),
                this->getPitches(),
                T_MemAlignment{}};
        }

        /** Creates a sub view to a part of the memory.
         *
         * @param extents number of elements for each dimension
         * @return View which is pointing only to a part of the original view.
         */
        constexpr auto getSubView(alpaka::concepts::Vector auto const& extents) const
        {
            assert((extents <= this->getExtents()).reduce(std::logical_and{}));
            return View{T_Api{}, this->data(), extents, this->getPitches(), T_MemAlignment{}};
        }

        /** Creates a sub view to a part of the memory.
         *
         * @param offset offset in elements to the original view
         * @param extents number of elements for each dimension
         * @return View which is pointing only to a part of the original view with a shifted origin pointer.
         *         View which pointThe alignment of the sub view is reduced to the element alignment.
         */
        constexpr auto getSubView(
            alpaka::concepts::Vector auto const& offset,
            alpaka::concepts::Vector auto const& extents) const
        {
            assert((extents <= this->getExtents()).reduce(std::logical_and{}));
            auto shiftedPtr = &getMdSpan()[offset];
            return View{T_Api{}, shiftedPtr, extents, this->getPitches(), Alignment<>{}};
        }
    };

    template<
        alpaka::concepts::HasApi T_Any,
        typename T_Type,
        alpaka::concepts::Vector T_UserExtents,
        alpaka::concepts::Vector T_UserPitches,
        alpaka::concepts::Alignment T_MemAlignment>
    ALPAKA_FN_HOST_ACC View(
        T_Any const&,
        T_Type*,
        T_UserExtents const&,
        T_UserPitches const&,
        T_MemAlignment const memAlignment)
        -> View<ALPAKA_TYPEOF(getApi(std::declval<T_Any>())), T_Type, typename T_UserPitches::UniVec, T_MemAlignment>;

    template<
        alpaka::concepts::HasApi T_Any,
        typename T_Type,
        alpaka::concepts::Vector T_UserExtents,
        alpaka::concepts::Vector T_UserPitches>
    ALPAKA_FN_HOST_ACC View(T_Any, T_Type*, T_UserExtents const&, T_UserPitches const&)
        -> View<ALPAKA_TYPEOF(getApi(std::declval<T_Any>())), T_Type, typename T_UserPitches::UniVec, Alignment<>>;

    namespace trait
    {
        template<typename T>
        requires(isSpecializationOf_v<std::remove_cvref_t<T>, alpaka::View>)
        struct IsMdSpan<T> : std::true_type
        {
        };
    } // namespace trait
} // namespace alpaka

namespace alpaka::internal
{
    // external define the API trait to support constexpr evaluation
    template<
        typename T_Api,
        typename T_Type,
        alpaka::concepts::Vector T_Extents,
        alpaka::concepts::Alignment T_MemAlignment>
    struct GetApi::Op<alpaka::View<T_Api, T_Type, T_Extents, T_MemAlignment>>
    {
        inline constexpr auto operator()(auto&& data) const
        {
            return T_Api{};
        }
    };
} // namespace alpaka::internal
