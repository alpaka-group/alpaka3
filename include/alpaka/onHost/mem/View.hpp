/* Copyright 2024 Bernhard Manfred Gruber, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/config.hpp"
#include "alpaka/internal.hpp"
#include "alpaka/mem/MdSpan.hpp"
#include "alpaka/onHost.hpp"
#include "alpaka/onHost/Handle.hpp"

#include <cstdint>
#include <functional>
#include <memory>
#include <sstream>

namespace alpaka::onHost
{
    /** memory owning view
     *
     * This view is only holding a handle to the real data, copying the view is cheap.
     * Owning mean that it is guaranteed that the lifetime of the data is defined by the lifetime of the view.
     */
    template<typename T_Datahandle, typename T_Extents>
    struct View
    {
    public:
        /** creates a view
         *
         * @param data handle to the physical data
         * @param extents M-dimensional extents in elements of the view. Must be <= number of elements in the dat
         * handle.
         */
        View(T_Datahandle data, T_Extents const& extents) : m_data(std::move(data)), m_extents(extents)
        {
        }

        /** creates a view
         *
         * Extents will be dirived from the data handle.
         *
         * @param data handle to the physical data
         */
        View(T_Datahandle data) : m_data(std::move(data)), m_extents(m_data->m_extents)
        {
        }

        View(View const&) = default;
        View(View&&) = default;

        /** shallow copy the handle
         *
         * It is not copying the data the view is pointing to.
         */
        View& operator=(View const&) = default;

        static consteval uint32_t dim()
        {
            return T_Extents::dim();
        }

        using type = typename T_Datahandle::element_type::type;
        using index_type = typename T_Extents::type;

        /** get the number of elements for each dimension */
        auto getExtents() const
        {
            return m_extents;
        }

        /** Get the distance in bytes to move to the next element in the corresponding dimension. */
        auto getPitches() const
        {
            return m_data->getPitches();
        }

        /** pointer to data */
        decltype(auto) data()
        {
            return onHost::data(m_data);
        }

        /** pointer to data */
        decltype(auto) data() const
        {
            return onHost::data(m_data);
        }

        auto getMdSpan() const
        {
            auto* ptr = onHost::data(m_data);
            return makeMdSpan(
                ptr,
                m_data->getExtents(),
                m_data->getPitches(),
                ALPAKA_TYPEOF(m_data->getAlignment()){});
        }

        /** access 1-dimensional data with a scalar index
         *
         * @{
         */
        decltype(auto) operator[](std::integral auto idx) const requires(dim() == 1u)
        {
            return data()[idx];
        }

        decltype(auto) operator[](std::integral auto idx) requires(dim() == 1u)
        {
            return data()[idx];
        }

        /** @} */

        /** access M-dimensional data with a vector index
         *
         * @{
         */
        decltype(auto) operator[](alpaka::concepts::Vector auto idx) const
        {
            return getMdSpan()[idx];
        }

        decltype(auto) operator[](alpaka::concepts::Vector auto idx)
        {
            return getMdSpan()[idx];
        }

        /** @} */

    private:
        void _()
        {
            //                static_assert(concepts::Device<Device>);
        }

        friend struct internal::Memcpy;

        T_Datahandle m_data;
        T_Extents m_extents;

        friend struct internal::Data;
        friend struct alpaka::internal::GetApi;
    };

    template<typename T_Datahandle>
    ALPAKA_FN_HOST_ACC View(T_Datahandle) -> View<T_Datahandle, typename T_Datahandle::element_type::ExtentType>;

    namespace internal
    {
        template<typename... T_Args>
        struct Data::Op<View<T_Args...>>
        {
            decltype(auto) operator()(auto&& buffer) const
            {
                return onHost::data(buffer.m_data);
            }
        };
    } // namespace internal
} // namespace alpaka::onHost

namespace alpaka::internal
{
    template<typename... T_Args>
    struct GetApi::Op<onHost::View<T_Args...>>
    {
        decltype(auto) operator()(auto&& buffer) const
        {
            return onHost::getApi(buffer.m_data);
        }
    };
} // namespace alpaka::internal
