/* Copyright 2024 René Widera, Bernhard Manfred Gruber
 * SPDX-License-Identifier: MPL-2.0
 */


#pragma once

#include "alpaka/Vec.hpp"
#include "alpaka/core/config.hpp"
#include "alpaka/internal.hpp"
#include "alpaka/mem/View.hpp"
#include "alpaka/onHost.hpp"
#include "alpaka/onHost/Device.hpp"
#include "alpaka/onHost/Handle.hpp"
#include "alpaka/onHost/concepts.hpp"
#include "alpaka/onHost/mem/MangedDealloc.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <sstream>

namespace alpaka::onHost
{
    /** Life time managed view with contiguous data
     *
     * This managed view owns the data and will deallocate it when the view is destroyed.
     * Const-ness of the managed view is NOT propagated to the data region.
     */
    template<
        alpaka::concepts::Api T_Api,
        typename T_Type,
        alpaka::concepts::Vector T_Extents,
        alpaka::concepts::Alignment T_MemAlignment = Alignment<>>
    struct ManagedView : View<T_Api, T_Type, T_Extents, T_MemAlignment>
    {
    private:
        using BaseView = View<T_Api, T_Type, T_Extents, T_MemAlignment>;

        /** Constructor with existing managed deleter */
        ManagedView(
            T_Api const api,
            T_Type* data,
            T_Extents const& extents,
            T_Extents const& pitches,
            std::shared_ptr<internal::ManagedDealloc> managedDeleter,
            T_MemAlignment const memAlignment)
            : BaseView{api, data, extents, pitches, memAlignment}
            , m_deleter{std::move(managedDeleter)}
        {
        }

        // friend declaration is required that a non const managed view can access the private constructor
        friend struct ManagedView<T_Api, std::remove_const_t<T_Type>, T_Extents, T_MemAlignment>;


    public:
        template<
            alpaka::concepts::HasApi T_Any,
            alpaka::concepts::Vector T_UserExtents,
            alpaka::concepts::Vector T_UserPitches>
        ManagedView(
            T_Any const& any,
            T_Type* data,
            T_UserExtents const& extents,
            T_UserPitches const& pitches,
            std::invocable<> auto deleter,
            T_MemAlignment const memAlignment = Alignment{})
            : BaseView{any, data, extents, pitches, memAlignment}
            , m_deleter{std::make_shared<internal::ManagedDealloc>(deleter)}
        {
            static_assert(
                isLosslessConvertible_v<typename T_UserPitches::type, typename T_UserExtents::type>,
                "extent type and pitch type must be lossless convertible");
        }

        auto getView() const
        {
            return static_cast<BaseView>(*this);
        }

        /** create a read only managed view */
        auto getConstManagedView() const
        {
            using ConstValueType = std::add_const_t<typename BaseView::value_type>;
            return ManagedView<T_Api, ConstValueType, T_Extents, T_MemAlignment>(
                T_Api{},
                static_cast<ConstValueType*>(this->data()),
                this->getExtents(),
                this->getPitches(),
                m_deleter,
                T_MemAlignment{});
        }

        /** Creates a sub managed view to a part of the memory.
         *
         * @param extents number of elements for each dimension
         * @return View which is pointing only to a part of the original managed view.
         */
        auto getManagedSubView(alpaka::concepts::Vector auto const& extents) const
        {
            assert(extents <= this->getExtents());
            return ManagedView{T_Api{}, this->data(), extents, this->getPitches(), m_deleter, T_MemAlignment{}};
        }

        /** Creates a sub managed view to a part of the memory.
         *
         * @param offset offset in elements to the original managed view
         * @param extents number of elements for each dimension
         * @return View which is pointing only to a part of the original managed view with a shifted origin pointer.
         *         View which pointThe alignment of the sub view is reduced to the element alignment.
         */
        auto getManagedSubView(
            alpaka::concepts::Vector auto const& offset,
            alpaka::concepts::Vector auto const& extents) const
        {
            assert(offset + extents <= this->getExtents());
            auto shiftedPtr = &this->getMdSpan()[offset];
            return ManagedView{T_Api{}, shiftedPtr, extents, this->getPitches(), Alignment<>{}};
        }

        /** Adds a destructor action to the managed view
         *
         * The action will be executed when the managed view is destroyed.
         * This can be used to add additional cleanup actions e.g. waiting on a specific queue.
         * Actions are executed in FIFO order.
         *
         * @param action callable to execute on destruction
         */
        void addDestructorAction(std::function<void()>&& action)
        {
            m_deleter->addAction(ALPAKA_FORWARD(action));
        }

        /** Add an action to be executed when the shared_ptr is destroyed.
         *
         * @param action Callable to execute on destruction.
         */
        void destructorWaitFor(auto const& any)
        {
            addDestructorAction([any]() { onHost::wait(any); });
        }

    private:
        /** @todo move this to trais or somewhere else that it can be used everywhere */
        template<alpaka::concepts::IsPointer T>
        using ConstPtr_t = std::add_pointer_t<std::add_const_t<std::remove_pointer_t<T>>>;

        std::shared_ptr<internal::ManagedDealloc> m_deleter;
    }; // namespace alpaka::onHost

    template<
        alpaka::concepts::HasApi T_Any,
        typename T_Type,
        alpaka::concepts::Vector T_UserExtents,
        alpaka::concepts::Vector T_UserPitches,
        alpaka::concepts::Alignment T_MemAlignment>
    ManagedView(
        T_Any const&,
        T_Type*,
        T_UserExtents const&,
        T_UserPitches const&,
        std::invocable<> auto,
        T_MemAlignment const)
        -> ManagedView<
            ALPAKA_TYPEOF(getApi(std::declval<T_Any>())),
            T_Type,
            typename T_UserPitches::UniVec,
            T_MemAlignment>;

    template<
        alpaka::concepts::HasApi T_Any,
        typename T_Type,
        alpaka::concepts::Vector T_UserExtents,
        alpaka::concepts::Vector T_UserPitches>
    ManagedView(T_Any const&, T_Type*, T_UserExtents const&, T_UserPitches const&, std::invocable<> auto)
        -> ManagedView<
            ALPAKA_TYPEOF(getApi(std::declval<T_Any>())),
            T_Type,
            typename T_UserPitches::UniVec,
            Alignment<>>;

    template<
        alpaka::concepts::Api T_Api,
        typename T_Type,
        alpaka::concepts::Vector T_Extents,
        alpaka::concepts::Alignment T_MemAlignment>
    struct MakeAccessibleOnAcc::Op<ManagedView<T_Api, T_Type, T_Extents, T_MemAlignment>>
    {
        auto operator()(auto&& any) const
        {
            return any.getView();
        }
    };

} // namespace alpaka::onHost

namespace alpaka::internal
{
    // external define the API trait to support constexpr evaluation
    template<
        alpaka::concepts::Api T_Api,
        typename T_Type,
        alpaka::concepts::Vector T_Extents,
        alpaka::concepts::Alignment T_MemAlignment>
    struct GetApi::Op<onHost::ManagedView<T_Api, T_Type, T_Extents, T_MemAlignment>>
    {
        inline constexpr auto operator()(auto&& data) const
        {
            return T_Api{};
        }
    };
} // namespace alpaka::internal

namespace alpaka::trait
{
    template<typename T>
    requires(isSpecializationOf_v<std::remove_cvref_t<T>, alpaka::onHost::ManagedView>)
    struct IsMdSpan<T> : std::true_type
    {
    };
} // namespace alpaka::trait
