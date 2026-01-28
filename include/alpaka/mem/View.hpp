/* Copyright 2024 Bernhard Manfred Gruber, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/api/concepts/api.hpp"
#include "alpaka/interface.hpp"
#include "alpaka/internal/interface.hpp"
#include "alpaka/mem/BoundaryIter.hpp"
#include "alpaka/mem/MdSpan.hpp"
#include "alpaka/mem/concepts/IMdSpan.hpp"
#include "alpaka/mem/concepts/detail/InnerTypeAllowedCast.hpp"
#include "alpaka/mem/trait.hpp"
#include "alpaka/onHost/interface.hpp"

#include <cstdint>
#include <functional>

namespace alpaka
{
    /** @brief Non owning view to data
     *
     * This view is only holding a pointer to real data, copying the view is cheap.
     * Const-ness of the view instance is propagated to the data region.
     *
     * This satisfies the alpaka::concepts::IView concept and, therefore, also the alpaka::concepts::IMdSpan concept.
     */
    template<
        alpaka::concepts::Api T_Api,
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

    template<typename T_ValueType, concepts::Alignment T_MemAlignment = Alignment<>>
    inline constexpr auto makeView(
        auto&& anyWithApi,
        T_ValueType* pointer,
        concepts::Vector auto const& extents,
        concepts::Vector auto const& pitches,
        T_MemAlignment const memAlignment = T_MemAlignment{})
    {
        static_assert(std::is_same_v<ALPAKA_TYPEOF(extents), ALPAKA_TYPEOF(pitches)>);
        return View{getApi(ALPAKA_FORWARD(anyWithApi)), pointer, extents, pitches, memAlignment};
    }

    inline constexpr auto makeView(auto&& any)
    {
        return View{
            getApi(ALPAKA_FORWARD(any)),
            onHost::data(ALPAKA_FORWARD(any)),
            onHost::getExtents(ALPAKA_FORWARD(any)),
            onHost::getPitches(ALPAKA_FORWARD(any)),
            alpaka::getAlignment(ALPAKA_FORWARD(any))};
    }

    template<
        alpaka::concepts::Api T_Api,
        typename T_Type,
        alpaka::concepts::Vector T_Extents,
        alpaka::concepts::Alignment T_MemAlignment>
    struct View : MdSpan<T_Type, typename T_Extents::UniVec, typename T_Extents::UniVec, T_MemAlignment>
    {
    private:
        using BaseMdSpan = MdSpan<T_Type, typename T_Extents::UniVec, typename T_Extents::UniVec, T_MemAlignment>;

    public:
        /** Creates a view
         *
         * @param data handle to the physical data
         * @param extents n-dimensional extents in elements of the view. Must satisfy `n <= number_of_elements` in the
         * data handle.
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
                  typename T_UserExtents::UniVec{extents},
                  typename T_UserPitches::UniVec{pitches},
                  memAlignment}
        {
            alpaka::unused(any);
            static_assert(
                isLosslesslyConvertible_v<typename T_UserPitches::type, typename T_UserExtents::type>,
                "extent type and pitch type must be lossless convertible");
        }

        template<typename T_Type_Other>
        requires alpaka::internal::concepts::InnerTypeAllowedCast<T_Type, T_Type_Other>
        constexpr View(View<T_Api, T_Type_Other, T_Extents, T_MemAlignment> const& other)
            : BaseMdSpan{static_cast<BaseMdSpan>(other)}
        {
        }

        constexpr View(View const&) = default;

        template<typename T_Type_Other>
        requires alpaka::internal::concepts::InnerTypeAllowedCast<T_Type, T_Type_Other>
        constexpr View(View<T_Api, T_Type_Other, T_Extents, T_MemAlignment>&& other)
            : BaseMdSpan{std::move(static_cast<BaseMdSpan>(other))}
        {
        }

        constexpr View(View&&) = default;

        constexpr View& operator=(View const&) = default;
        constexpr View& operator=(View&&) = default;

        static consteval T_Api getApi()
        {
            return T_Api{};
        }

        constexpr alpaka::concepts::IMdSpan auto getMdSpan() const
        {
            return BaseMdSpan::getConstMdSpan();
        }

        constexpr alpaka::concepts::IMdSpan auto getMdSpan()
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
        constexpr auto getSubView(alpaka::concepts::VectorOrScalar auto const& extents) const
        {
            Vec extentMd = extents;
            assert((extentMd <= this->getExtents()).reduce(std::logical_and{}));
            return makeView(T_Api{}, this->data(), extentMd, this->getPitches(), T_MemAlignment{});
        }

        constexpr auto getSubView(alpaka::concepts::VectorOrScalar auto const& extents)
        {
            Vec extentMd = extents;
            assert((extentMd <= this->getExtents()).reduce(std::logical_and{}));
            return makeView(T_Api{}, this->data(), extentMd, this->getPitches(), T_MemAlignment{});
        }

        /** Creates a sub view to a part of the memory.
         *
         * @param offset offset in elements to the original view
         * @param extents number of elements for each dimension
         * @return View which is pointing only to a part of the original view with a shifted origin pointer.
         *         View which pointThe alignment of the sub view is reduced to the element alignment.
         */
        constexpr auto getSubView(
            alpaka::concepts::VectorOrScalar auto const& offset,
            alpaka::concepts::VectorOrScalar auto const& extents) const
        {
            Vec offsetMd = offset;
            Vec extentMd = extents;
            assert((offsetMd + extentMd <= this->getExtents()).reduce(std::logical_and{}));
            auto shiftedPtr = &(*this)[offsetMd];
            return makeView(T_Api{}, shiftedPtr, extentMd, this->getPitches(), Alignment<>{});
        }

        constexpr auto getSubView(
            alpaka::concepts::VectorOrScalar auto const& offset,
            alpaka::concepts::VectorOrScalar auto const& extents)
        {
            Vec offsetMd = offset;
            Vec extentMd = extents;
            assert((offsetMd + extentMd <= this->getExtents()).reduce(std::logical_and{}));
            auto shiftedPtr = &(*this)[offsetMd];
            return makeView(T_Api{}, shiftedPtr, extentMd, this->getPitches(), Alignment<>{});
        }

        template<alpaka::concepts::Vector LowHaloVecType, alpaka::concepts::Vector UpHaloVecType>
        constexpr auto getSubView(
            alpaka::BoundaryDirection<View::dim(), LowHaloVecType, UpHaloVecType> boundaryDir) const
        {
            constexpr uint32_t dim = View::dim();
            auto offset = alpaka::Vec<uint32_t, dim>{};
            auto extents = alpaka::Vec<uint32_t, dim>{};

            for(uint32_t i = 0; i < dim; ++i)
            {
                switch(boundaryDir.data[i])
                {
                case BoundaryType::LOWER:
                    offset[i] = 0;
                    extents[i] = boundaryDir.lowerHaloSize[i];
                    break;
                case BoundaryType::UPPER:
                    offset[i] = this->getExtents()[i] - boundaryDir.upperHaloSize[i];
                    extents[i] = boundaryDir.upperHaloSize[i];
                    break;
                case BoundaryType::MIDDLE:
                    offset[i] = boundaryDir.lowerHaloSize[i];
                    extents[i] = this->getExtents()[i] - boundaryDir.lowerHaloSize[i] - boundaryDir.upperHaloSize[i];
                    break;
                default:
                    throw std::invalid_argument("invalid direction");
                }
            }
            return getSubView(offset, extents);
        }
    };

    template<typename T_Api, typename T_Type, concepts::Vector T_Extents, concepts::Alignment T_MemAlignment>
    std::ostream& operator<<(std::ostream& s, View<T_Api, T_Type, T_Extents, T_MemAlignment> const& view)
    {
        return s << "View{ dim=" << ALPAKA_TYPEOF(view)::dim() << ", api= " << onHost::getName(T_Api{})
                 << ", extents=" << view.getExtents().toString() << ", pitches=" << view.getPitches().toString()
                 << " , alignment=" << T_MemAlignment::template get<T_Type>() << " }";
    }

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
} // namespace alpaka

namespace alpaka::internal
{
    // externally define the API trait to support constexpr evaluation
    template<
        alpaka::concepts::Api T_Api,
        typename T_Type,
        alpaka::concepts::Vector T_Extents,
        alpaka::concepts::Alignment T_MemAlignment>
    struct GetApi::Op<alpaka::View<T_Api, T_Type, T_Extents, T_MemAlignment>>
    {
        inline constexpr auto operator()(auto&& view) const
        {
            alpaka::unused(view);
            return T_Api{};
        }
    };

    template<
        alpaka::concepts::Api T_Api,
        typename T_Type,
        alpaka::concepts::Vector T_Extents,
        alpaka::concepts::Alignment T_MemAlignment>
    struct CopyConstructableDataSource<View<T_Api, T_Type, T_Extents, T_MemAlignment>> : std::true_type
    {
        using InnerMutable = View<T_Api, std::remove_const_t<T_Type>, T_Extents, T_MemAlignment>;
        using InnerConst = View<T_Api, std::add_const_t<T_Type>, T_Extents, T_MemAlignment>;
    };
} // namespace alpaka::internal
