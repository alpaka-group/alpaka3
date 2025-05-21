/* Copyright 2024 René Widera, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */


#pragma once

#include "alpaka/Vec.hpp"
#include "alpaka/core/config.hpp"
#include "alpaka/internal.hpp"
#include "alpaka/onHost.hpp"
#include "alpaka/onHost/Device.hpp"
#include "alpaka/onHost/Handle.hpp"
#include "alpaka/onHost/mem/MangedDealloc.hpp"
#include "alpaka/onHost/mem/View.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <sstream>

namespace alpaka::onHost
{

    namespace mem
    {
        //! Calculate the pitches purely from the extents.
        template<typename T_Elem, alpaka::concepts::Vector T_Vec>
        constexpr auto calculatePitchesFromExtents(T_Vec const& extent)
        {
            constexpr auto dim = T_Vec::dim();
            using type = typename T_Vec::type;
            auto pitchBytes = typename T_Vec::UniVec{};
            if constexpr(dim > 0)
                pitchBytes.back() = static_cast<type>(sizeof(T_Elem));
            if constexpr(dim > 1)
                for(type i = dim - 1; i > 0; i--)
                    pitchBytes[i - 1] = extent[i] * pitchBytes[i];
            return pitchBytes;
        }

        //! Calculate the pitches purely from the extents.
        template<typename T_Elem, alpaka::concepts::Vector T_Vec>
        requires(T_Vec::dim() >= 2)
        constexpr auto calculatePitches(T_Vec const& extent, typename T_Vec::type const& rowPitchBytes)
        {
            constexpr auto dim = T_Vec::dim();
            using type = typename T_Vec::type;
            auto pitchBytes = typename T_Vec::UniVec{};
            pitchBytes.back() = static_cast<type>(sizeof(T_Elem));
            if constexpr(dim > 1)
                pitchBytes[dim - 2u] = rowPitchBytes;
            if constexpr(dim > 2)
                for(type i = dim - 2; i > 0; i--)
                    pitchBytes[i - 1] = extent[i] * pitchBytes[i];
            return pitchBytes;
        }
    } // namespace mem

    /** Container with contiguous data
     *
     * The container owns the data and will deallocate it when the buffer is destroyed.
     * Const-ness of the buffer is propagated to the data region.
     */
    template<
        alpaka::concepts::Api T_Api,
        typename T_Type,
        alpaka::concepts::Vector T_Extents,
        alpaka::concepts::Alignment T_MemAlignment = Alignment<>>
    struct Buffer : View<T_Api, T_Type, T_Extents, T_MemAlignment>
    {
    private:
        using BaseView = View<T_Api, T_Type, T_Extents, T_MemAlignment>;

        /** Constructor with existing managed deleter */
        Buffer(
            T_Api const api,
            T_Type* data,
            T_Extents const& extents,
            T_Extents const& pitches,
            std::shared_ptr<mem::ManagedDealloc> managedDeleter,
            T_MemAlignment const memAlignment)
            : BaseView{api, data, extents, pitches, memAlignment}
            , m_deleter{std::move(managedDeleter)}
        {
        }

        // friend declaration is required that a non const buffer can access the private constructor
        friend struct Buffer<T_Api, std::remove_const_t<T_Type>, T_Extents, T_MemAlignment>;

    public:
        template<
            concepts::HasApi T_Any,
            alpaka::concepts::Vector T_UserExtents,
            alpaka::concepts::Vector T_UserPitches>
        Buffer(
            T_Any const& any,
            T_Type* data,
            T_UserExtents const& extents,
            T_UserPitches const& pitches,
            std::invocable<> auto deleter,
            T_MemAlignment const memAlignment = Alignment{})
            : BaseView{any, data, extents, pitches, memAlignment}
            , m_deleter{std::make_shared<mem::ManagedDealloc>(deleter)}
        {
            static_assert(
                isLosslessConvertible_v<typename T_UserPitches::type, typename T_UserExtents::type>,
                "extent type and pitch type must be lossless convertible");
        }

        auto getView() const

        {
            return this->getConstView();
        }

        BaseView getView()
        {
            return *this;
        }

        /** create a read only buffer */
        auto getConstBuffer() const
        {
            auto const* ptr = this->data();
            return Buffer<T_Api, std::remove_pointer_t<decltype(ptr)>, T_Extents, T_MemAlignment>(
                T_Api{},
                ptr,
                this->getExtents(),
                this->getPitches(),
                m_deleter,
                T_MemAlignment{});
        }

        /** Creates a sub buffer to a part of the memory.
         *
         * @param extents number of elements for each dimension
         * @return View which is pointing only to a part of the original buffer.
         */
        auto getSubBuffer(alpaka::concepts::Vector auto const& extents) const
        {
            assert(extents <= this->getExtents());
            return Buffer{T_Api{}, this->data(), extents, this->getPitches(), m_deleter, T_MemAlignment{}};
        }

        auto getSubBuffer(alpaka::concepts::Vector auto const& extents)
        {
            assert(extents <= this->getExtents());
            return Buffer{T_Api{}, this->data(), extents, this->getPitches(), m_deleter, T_MemAlignment{}};
        }

        /** Creates a sub buffer to a part of the memory.
         *
         * @param offset offset in elements to the original buffer
         * @param extents number of elements for each dimension
         * @return View which is pointing only to a part of the original buffer with a shifted origin pointer.
         *         View which pointThe alignment of the sub view is reduced to the element alignment.
         */
        auto getSubBuffer(alpaka::concepts::Vector auto const& offset, alpaka::concepts::Vector auto const& extents)
            const
        {
            assert(offset + extents <= this->getExtents());
            auto shiftedPtr = &this->getMdSpan()[offset];
            return Buffer{T_Api{}, shiftedPtr, extents, this->getPitches(), Alignment<>{}};
        }

        auto getSubBuffer(alpaka::concepts::Vector auto const& offset, alpaka::concepts::Vector auto const& extents)
        {
            assert(offset + extents <= this->getExtents());
            auto shiftedPtr = &this->getMdSpan()[offset];
            return Buffer{T_Api{}, shiftedPtr, extents - offset, this->getPitches(), Alignment<>{}};
        }

    private:
        std::shared_ptr<mem::ManagedDealloc> m_deleter;
    }; // namespace alpaka::onHost

    template<
        concepts::HasApi T_Any,
        typename T_Type,
        alpaka::concepts::Vector T_UserExtents,
        alpaka::concepts::Vector T_UserPitches,
        alpaka::concepts::Alignment T_MemAlignment>
    Buffer(
        T_Any const&,
        T_Type*,
        T_UserExtents const&,
        T_UserPitches const&,
        std::invocable<> auto,
        T_MemAlignment const)
        -> Buffer<
            ALPAKA_TYPEOF(onHost::getApi(std::declval<T_Any>())),
            T_Type,
            typename T_UserPitches::UniVec,
            T_MemAlignment>;

    template<
        concepts::HasApi T_Any,
        typename T_Type,
        alpaka::concepts::Vector T_UserExtents,
        alpaka::concepts::Vector T_UserPitches>
    Buffer(T_Any const&, T_Type*, T_UserExtents const&, T_UserPitches const&, std::invocable<> auto) -> Buffer<
        ALPAKA_TYPEOF(onHost::getApi(std::declval<T_Any>())),
        T_Type,
        typename T_UserPitches::UniVec,
        Alignment<>>;
} // namespace alpaka::onHost

namespace alpaka::internal
{
    // external define the API trait to support constexpr evaluation
    template<
        alpaka::concepts::Api T_Api,
        typename T_Type,
        alpaka::concepts::Vector T_Extents,
        alpaka::concepts::Alignment T_MemAlignment>
    struct GetApi::Op<onHost::Buffer<T_Api, T_Type, T_Extents, T_MemAlignment>>
    {
        inline constexpr auto operator()(auto&& data) const
        {
            return T_Api{};
        }
    };
} // namespace alpaka::internal
