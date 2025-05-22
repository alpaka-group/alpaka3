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

namespace alpaka::onHost
{
    /** Non owning view to data
     *
     * This view is only holding a points to real data, copying the view is cheap.
     * Const-ness of the view is propagated to the data region.
     */
    template<
        typename T_Api,
        typename T_Type,
        alpaka::concepts::Vector T_Extents,
        alpaka::concepts::Alignment T_MemAlignment = Alignment<>>
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
        View(
            T_Any const& any,
            T_Type* data,
            T_UserExtents const& extents,
            T_UserPitches const& pitches,
            T_MemAlignment const memAlignment = Alignment{})
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

        View(View const&) = default;
        View(View&&) = default;
        constexpr View& operator=(View const&) = default;
        constexpr View& operator=(View&&) = default;

        static consteval T_Api getApi()
        {
            return T_Api{};
        }

        /** pointer to data */
        decltype(auto) data()
        {
            return BaseMdSpan::data();
        }

        /** pointer to data */
        decltype(auto) data() const
        {
            using RetType = ConstPtr_t<ALPAKA_TYPEOF(BaseMdSpan::data())>;
            return static_cast<RetType>(BaseMdSpan::data());
        }

        alpaka::concepts::MdSpan auto getMdSpan() const
        {
            return this->getConstMdSpan();
        }

        alpaka::concepts::MdSpan auto getMdSpan()
        {
            return BaseMdSpan{*this};
        }

        /** create a read only view */
        auto getConstView() const
        {
            return View<T_Api, std::remove_pointer_t<decltype(this->data())>, T_Extents, T_MemAlignment>{
                T_Api{},
                this->data(),
                this->getExtents(),
                this->getPitches(),
                T_MemAlignment{}};
        }

        /** Creates a sub view to a part of the memory.
         *
         * @param extents number of elements for each dimension
         * @return View which is pointing only to a part of the original view.
         */
        auto getSubView(alpaka::concepts::Vector auto const& extents) const
        {
            assert(extents <= this->getExtents());
            return View{T_Api{}, data(), extents, this->getPitches(), T_MemAlignment{}};
        }

        auto getSubView(alpaka::concepts::Vector auto const& extents)
        {
            assert(extents <= this->getExtents());
            return View{T_Api{}, data(), extents, this->getPitches(), T_MemAlignment{}};
        }

        /** Creates a sub view to a part of the memory.
         *
         * @param offset offset in elements to the original view
         * @param extents number of elements for each dimension
         * @return View which is pointing only to a part of the original view with a shifted origin pointer.
         *         View which pointThe alignment of the sub view is reduced to the element alignment.
         */
        auto getSubView(alpaka::concepts::Vector auto const& offset, alpaka::concepts::Vector auto const& extents)
            const
        {
            assert(offset + extents <= this->getExtents());
            auto shiftedPtr = &getMdSpan()[offset];
            return View{T_Api{}, shiftedPtr, extents, this->getPitches(), Alignment<>{}};
        }

        auto getSubView(alpaka::concepts::Vector auto const& offset, alpaka::concepts::Vector auto const& extents)
        {
            assert(offset + extents <= this->getExtents());
            auto shiftedPtr = &getMdSpan()[offset];
            return View{T_Api{}, shiftedPtr, extents - offset, this->getPitches(), Alignment<>{}};
        }

    private:
        /** @todo move this to trais or somewhere else that it can be used everywhere */
        template<alpaka::concepts::IsPointer T>
        using ConstPtr_t = std::add_pointer_t<std::add_const_t<std::remove_pointer_t<T>>>;

        void _()
        {
            //                static_assert(concepts::Device<Device>);
        }
    };

    template<
        alpaka::concepts::HasApi T_Any,
        typename T_Type,
        alpaka::concepts::Vector T_UserExtents,
        alpaka::concepts::Vector T_UserPitches,
        alpaka::concepts::Alignment T_MemAlignment>
    View(T_Any const&, T_Type*, T_UserExtents const&, T_UserPitches const&, T_MemAlignment const memAlignment)
        -> View<ALPAKA_TYPEOF(getApi(std::declval<T_Any>())), T_Type, typename T_UserPitches::UniVec, T_MemAlignment>;

    template<
        alpaka::concepts::HasApi T_Any,
        typename T_Type,
        alpaka::concepts::Vector T_UserExtents,
        alpaka::concepts::Vector T_UserPitches>
    View(T_Any, T_Type*, T_UserExtents const&, T_UserPitches const&)
        -> View<ALPAKA_TYPEOF(getApi(std::declval<T_Any>())), T_Type, typename T_UserPitches::UniVec, Alignment<>>;
} // namespace alpaka::onHost

namespace alpaka::internal
{
    // external define the API trait to support constexpr evaluation
    template<
        typename T_Api,
        typename T_Type,
        alpaka::concepts::Vector T_Extents,
        alpaka::concepts::Alignment T_MemAlignment>
    struct GetApi::Op<onHost::View<T_Api, T_Type, T_Extents, T_MemAlignment>>
    {
        inline constexpr auto operator()(auto&& data) const
        {
            return T_Api{};
        }
    };
} // namespace alpaka::internal
