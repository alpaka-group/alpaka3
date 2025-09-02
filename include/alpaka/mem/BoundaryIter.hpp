/* Copyright 2025 Anton Reinhard
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/CVec.hpp"
#include "alpaka/Vec.hpp"
#include "alpaka/api/api.hpp"
#include "alpaka/api/host/Api.hpp"
#include "alpaka/concepts.hpp"
#include "alpaka/core/Assert.hpp"
#include "alpaka/utility.hpp"

#include <ostream>

namespace alpaka
{

    /**
     * @brief An enum representing the different types of boundary, with LOWER, MIDDLE, and UPPER being valid states,
     * and OOB being invalid (out-of-bounds).
     */
    enum class BoundaryType : uint32_t
    {
        LOWER,
        MIDDLE,
        UPPER,
        OOB
    };

    /**
     * @brief An n-dimensional boundary direction. Encodes a single unique boundary of an nD volume, e.g., a specific
     * corner of a 2D plane or a side of a 3D cube.
     *
     * @tparam Dim The dimensionality of the volume that this is a boundary direction for.
     * @tparam LowHaloVecType The vector type used for the lower halo sizes.
     * @tparam UpHaloVecType The vector type used for the upper halo sizes.
     */
    template<uint32_t Dim, alpaka::concepts::Vector LowHaloVecType, alpaka::concepts::Vector UpHaloVecType>
    struct BoundaryDirection
    {
        using BoundaryVecType = alpaka::Vec<BoundaryType, Dim>;

        BoundaryVecType data;
        LowHaloVecType lowerHaloSize;
        UpHaloVecType upperHaloSize;

        constexpr BoundaryDirection(
            alpaka::concepts::Vector auto const& boundaries,
            LowHaloVecType const& lower_halo_sizes,
            UpHaloVecType const& upper_halo_sizes)
            : data(boundaries)
            , lowerHaloSize(lower_halo_sizes)
            , upperHaloSize(upper_halo_sizes)
        {
        }

        [[nodiscard]] static constexpr uint32_t dim()
        {
            return Dim;
        }

        [[nodiscard]] constexpr uint32_t boundaryDimensionality() const
        {
            uint32_t c = 0;
            for(auto i = 0; i < Dim; ++i)
            {
                if(data[i] == BoundaryType::MIDDLE)
                    ++c;
            }
            return c;
        }

        [[nodiscard]] constexpr bool isPoint() const
        {
            return boundaryDimensionality() == 0;
        }

        [[nodiscard]] constexpr bool isLine() const
        {
            return boundaryDimensionality() == 1;
        }

        [[nodiscard]] constexpr bool isPlane() const
        {
            return boundaryDimensionality() == 2;
        }

        [[nodiscard]] constexpr bool isVolume() const
        {
            return boundaryDimensionality() == 3;
        }

        [[nodiscard]] constexpr auto operator<=>(BoundaryDirection const&) const = default;
    };

    /**
     * @brief The iterator type for [`BoundaryDirectionsContainer`](@ref)
     *
     * @tparam Dim The dimensionality of the volume that this is a boundary direction iterator for.
     * @tparam LowHaloVecType The vector type used for the lower halo sizes.
     * @tparam UpHaloVecType The vector type used for the upper halo sizes.
     */
    template<uint32_t Dim, alpaka::concepts::Vector LowHaloVecType, alpaka::concepts::Vector UpHaloVecType>
    struct BoundaryDirectionIter
    {
        using BoundaryVecType = alpaka::Vec<BoundaryType, Dim>;

        using difference_type = std::ptrdiff_t;
        using value_type = BoundaryDirection<Dim, LowHaloVecType, UpHaloVecType>;
        using reference = value_type&;
        using const_reference = value_type const&;
        using pointer = value_type*;
        using const_pointer = value_type const*;

        constexpr BoundaryDirectionIter(
            BoundaryVecType const& boundaries,
            LowHaloVecType const& lower_halo_sizes,
            UpHaloVecType const& upper_halo_sizes)
            : boundaries(boundaries, lower_halo_sizes, upper_halo_sizes)
            , lowerHaloSizes(lower_halo_sizes)
            , upperHaloSizes(upper_halo_sizes)
        {
        }

        [[nodiscard]] constexpr const_reference& operator*() const
        {
            return boundaries;
        }

        [[nodiscard]] constexpr reference& operator*()
        {
            return boundaries;
        }

        constexpr auto& operator++()
        {
            uint32_t i = Dim - 1;
            bool oob = true;
            while(i != static_cast<uint32_t>(-1))
            {
                switch(boundaries.data[i])
                {
                case BoundaryType::LOWER:
                    boundaries.data[i] = BoundaryType::MIDDLE;
                    i = static_cast<uint32_t>(-1);
                    oob = false;
                    break;
                case BoundaryType::MIDDLE:
                    boundaries.data[i] = BoundaryType::UPPER;
                    i = static_cast<uint32_t>(-1);
                    oob = false;
                    break;
                case BoundaryType::UPPER:
                    boundaries.data[i] = BoundaryType::LOWER;
                    --i;
                    break;
                case BoundaryType::OOB:
                    [[fallthrough]];
                default:
                    constexpr bool onHost = std::is_same_v<api::Host, ALPAKA_TYPEOF(thisApi())>;
                    if constexpr(onHost)
                        assert(false);
                    else
                        ALPAKA_ASSERT_ACC(false);
                }
            }
            if(oob)
            {
                boundaries
                    = {Vec<BoundaryType, Dim>([](int) { return BoundaryType::OOB; }), lowerHaloSizes, upperHaloSizes};
            }
            return *this;
        }

        [[nodiscard]] static constexpr auto dim()
        {
            return Dim;
        }

        [[nodiscard]] constexpr auto operator<=>(BoundaryDirectionIter const&) const = default;

    private:
        BoundaryDirection<Dim, LowHaloVecType, UpHaloVecType> boundaries;

        LowHaloVecType lowerHaloSizes;
        UpHaloVecType upperHaloSizes;
    };

    /**
     * @brief A container for boundary directions of an n-dimensional volume.
     * For example, a 1-dimensional (1D) volume has two 0D ends and a 1D center. A 2D volume has 4 0D corners, 4 1D
     * edges, and one 2D center. In general, there are 3^n boundaries for an nD volume. This class implements begin(),
     * end(), and length(), and can be iterated over.
     *
     * @tparam Dim The dimensionality of the volume that this contains boundaries for.
     * @tparam LowHaloVecType The vector type used for the lower halo sizes.
     * @tparam UpHaloVecType The vector type used for the upper halo sizes.
     */
    template<uint32_t Dim, alpaka::concepts::Vector LowHaloVecType, alpaka::concepts::Vector UpHaloVecType>
    struct BoundaryDirectionsContainer
    {
        static_assert(Dim > 0, "0 Dimension Boundary Direction Container is not defined");

        constexpr BoundaryDirectionsContainer(
            LowHaloVecType const& lowerHaloSizes,
            UpHaloVecType const& upperHaloSizes)
            : lowerHaloSizes(lowerHaloSizes)
            , upperHaloSizes(upperHaloSizes)
        {
        }

        [[nodiscard]] constexpr BoundaryDirectionIter<Dim, LowHaloVecType, UpHaloVecType> begin() const
        {
            return BoundaryDirectionIter<Dim, LowHaloVecType, UpHaloVecType>{
                Vec<BoundaryType, Dim>([](int) { return BoundaryType::LOWER; }),
                lowerHaloSizes,
                upperHaloSizes};
        }

        [[nodiscard]] constexpr BoundaryDirectionIter<Dim, LowHaloVecType, UpHaloVecType> end() const
        {
            return BoundaryDirectionIter<Dim, LowHaloVecType, UpHaloVecType>{
                Vec<BoundaryType, Dim>([](int) { return BoundaryType::OOB; }),
                lowerHaloSizes,
                upperHaloSizes};
        }

        [[nodiscard]] static consteval uint32_t length()
        {
            return ipow(3u, Dim);
        }

        [[nodiscard]] static constexpr auto dim()
        {
            return Dim;
        }

    private:
        LowHaloVecType const lowerHaloSizes;
        UpHaloVecType const upperHaloSizes;
    };

    template<alpaka::concepts::Vector LowHaloVecType, alpaka::concepts::Vector UpHaloVecType>
    BoundaryDirectionsContainer(LowHaloVecType const& lowerHalos, UpHaloVecType const& upperHalos)
        -> BoundaryDirectionsContainer<LowHaloVecType::dim(), LowHaloVecType, UpHaloVecType>;

    /** @brief Construct and return a single boundary direction specifying the middle of a volume.
     */
    template<uint32_t Dims>
    [[nodiscard]] constexpr auto makeCoreBoundaryDirection(
        alpaka::concepts::Vector auto const& lowerHalos,
        alpaka::concepts::Vector auto const& upperHalos)
    {
        return BoundaryDirection<Dims, ALPAKA_TYPEOF(lowerHalos), ALPAKA_TYPEOF(upperHalos)>{
            fillCVec<BoundaryType, Dims, BoundaryType::MIDDLE>(),
            lowerHalos,
            upperHalos};
    }

    /** @brief Construct and return a single boundary direction specifying the middle of a volume with symmetric halos.
     */
    template<uint32_t Dims>
    [[nodiscard]] constexpr auto makeCoreBoundaryDirection(alpaka::concepts::Vector auto const& halos)
    {
        return BoundaryDirection<Dims, ALPAKA_TYPEOF(halos), ALPAKA_TYPEOF(halos)>{
            fillCVec<BoundaryType, Dims, BoundaryType::MIDDLE>(),
            halos,
            halos};
    }

    /**
     * @brief Construct and return a single boundary direction specifying the middle of a volume with all halo sizes
     * set to 1.
     */
    template<uint32_t Dims>
    consteval auto makeCoreBoundaryDirection()
    {
        return makeCoreBoundaryDirection<Dims>(fillCVec<uint32_t, Dims, 1u>());
    }

    /** @brief Construct and return a boundary direction container. This container can be iterated over. See
     * BoundaryDirectionsContainer.
     * This constructor uses a default halo size of 1 everywhere.
     *
     * @tparam Dims The dimensionality of the container.
     */
    template<uint32_t Dims>
    [[nodiscard]] constexpr auto makeBoundaryDirIterator()
    {
        auto lowerHalos = alpaka::fillCVec<uint32_t, Dims, static_cast<uint32_t>(1)>();
        auto upperHalos = alpaka::fillCVec<uint32_t, Dims, static_cast<uint32_t>(1)>();
        return BoundaryDirectionsContainer{lowerHalos, upperHalos};
    }

    /** @brief Construct and return a boundary direction container with the given halo sizes.
     * This container can be iterated over. See BoundaryDirectionsContainer.
     * The dimensionality is inferred from the given haloSizes.
     *
     * @param haloSizes The halo sizes per dimension. The halos are used for both "ends" of each dimension
     * symmetrically.
     */
    [[nodiscard]] constexpr auto makeBoundaryDirIterator(alpaka::concepts::Vector auto const& haloSizes)
    {
        return BoundaryDirectionsContainer{haloSizes, haloSizes};
    }

    /** @brief Construct and return a boundary direction container with the given halo sizes.
     * This container can be iterated over. See BoundaryDirectionsContainer.
     * The dimensionality is inferred from the given halo sizes, which are asserted to be identical.
     *
     * @param lowerHaloSizes The lower end halo sizes per dimension. These are the halos from 0 in each dimension.
     * @param upperHaloSizes The upper end halo sizes per dimension. These are the halos to `size()` in each dimension.
     */
    [[nodiscard]] constexpr auto makeBoundaryDirIterator(
        alpaka::concepts::Vector auto const& lowerHaloSizes,
        alpaka::concepts::Vector auto const& upperHaloSizes)
    {
        static_assert(
            ALPAKA_TYPEOF(lowerHaloSizes)::dim() == ALPAKA_TYPEOF(upperHaloSizes)::dim(),
            "Dimension mismatch");
        return BoundaryDirectionsContainer{lowerHaloSizes, upperHaloSizes};
    }

    /** @brief Construct and return a boundary direction container for the given view with default (size 1) halo
     * sizes. This container can be iterated over. See BoundaryDirectionsContainer.
     * For custom halo sizes, use one of the other overloads.
     *
     * @param view The given view; only the dimension of the view matters.
     */
    [[nodiscard]] constexpr auto makeBoundaryDirIterator(alpaka::concepts::View auto const& view)
    {
        return makeBoundaryDirIterator<static_cast<uint32_t>(ALPAKA_TYPEOF(view)::dim())>();
    }

    namespace trait
    {
        template<typename T>
        struct IsBoundaryDirection : std::false_type
        {
        };

        template<uint32_t Dim, alpaka::concepts::Vector LowHaloVecType, alpaka::concepts::Vector UpHaloVecType>
        requires(Dim == LowHaloVecType::dim() && Dim == UpHaloVecType::dim())
        struct IsBoundaryDirection<BoundaryDirection<Dim, LowHaloVecType, UpHaloVecType>> : std::true_type
        {
        };
    } // namespace trait

    template<typename T>
    constexpr bool isBoundaryDirection_v = trait::IsBoundaryDirection<T>::value;

    namespace concepts
    {
        /** @brief Concept checking whether T is a boundary direction.
         */
        template<typename T>
        concept BoundaryDirection = isBoundaryDirection_v<T>;
    } // namespace concepts

    std::ostream& operator<<(std::ostream& os, concepts::BoundaryDirection auto const& bd)
    {
        for(auto i = 0; i < bd.dim(); ++i)
        {
            switch(bd.data[i])
            {
            case BoundaryType::LOWER:
                os << 'v';
                break;
            case BoundaryType::MIDDLE:
                os << '-';
                break;
            case BoundaryType::UPPER:
                os << '^';
                break;
            case BoundaryType::OOB:
                [[fallthrough]];
            default:
                os << 'x';
                break;
            }
        }

        if(bd.isPoint())
            os << " (point) ";
        if(bd.isLine())
            os << " (line)  ";
        if(bd.isPlane())
            os << " (plane) ";
        if(bd.isVolume())
            os << " (volume)";
        if(bd.boundaryDimensionality() >= 4)
            os << " (" << bd.boundaryDimensionality() << "D volume)";

        return os;
    }
} // namespace alpaka
