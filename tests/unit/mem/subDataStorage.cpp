/* Copyright 2026 Simeon Ehrig
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>

#include <alpakaTest/testMacros.hpp>
#include <catch2/catch_test_macros.hpp>

TEST_CASE("1D alpaka::View::getSubView function tests", "[mem][view][SubDataStorage]")
{
    constexpr int x = 10;
    auto buffer0 = alpaka::onHost::allocHost<int>(x);
    for(auto i = 0; i < x; ++i)
    {
        buffer0[i] = i;
    }

    auto view0 = buffer0.getView();

    SECTION("getSubView, single integer, define extent")
    {
        constexpr int sub_size = 8;
        auto sub_view0 = view0.getSubView(sub_size);
        REQUIRE(sub_view0.getExtents() == alpaka::Vec{sub_size});
        for(auto i = 0; i < sub_size; ++i)
        {
            REQUIRE_MESSAGE(sub_view0[i] == i, "i=" << i);
        }
    }

    SECTION("getSubView, single integer, define offset and extent")
    {
        constexpr int offset = 2;
        constexpr int end = 7;
        auto sub_view0 = view0.getSubView(offset, end);

        REQUIRE(sub_view0.getExtents() == alpaka::Vec{end});
        for(auto i = 0; i < sub_view0.getExtents()[0]; ++i)
        {
            REQUIRE_MESSAGE(sub_view0[i] == i + offset, "i=" << i);
        }
    }
}

TEST_CASE("3D alpaka::View::getSubView function tests", "[mem][view][SubDataStorage]")
{
    constexpr int z = 3;
    constexpr int y = 5;
    constexpr int x = 4;
    alpaka::Vec total_extents{z, y, x};
    REQUIRE(total_extents.z() == z);
    REQUIRE(total_extents.y() == y);
    REQUIRE(total_extents.x() == x);

    auto buffer0 = alpaka::onHost::allocHost<int>(total_extents);

    for(auto vec : alpaka::IdxRange{total_extents})
    {
        buffer0[vec] = vec.z() * 100 + vec.y() * 10 + vec.x();
    }

    auto view0 = buffer0.getView();

    SECTION("getSubView, define extent")
    {
        alpaka::Vec extents_subview0{z - 1, y - 1, x - 1};
        auto sub_view0 = view0.getSubView(extents_subview0);
        REQUIRE(sub_view0.getApi() == view0.getApi());

        STATIC_REQUIRE(sub_view0.dim() == extents_subview0.dim());
        REQUIRE(sub_view0.getExtents() == extents_subview0);

        for(auto vec : alpaka::IdxRange{extents_subview0})
        {
            REQUIRE_MESSAGE(sub_view0[vec] == vec.z() * 100 + vec.y() * 10 + vec.x(), "vec=" << vec);
        }
    }

    SECTION("getSubView, define offset and extent")
    {
        alpaka::Vec offset_subview0{1, 2, 1};
        alpaka::Vec extents_subview0{z - 1, y - 3, x - 1};

        auto sub_view0 = view0.getSubView(offset_subview0, extents_subview0);
        REQUIRE(sub_view0.getApi() == view0.getApi());

        STATIC_REQUIRE(sub_view0.dim() == extents_subview0.dim());
        REQUIRE(sub_view0.getExtents() == extents_subview0);

        REQUIRE(sub_view0[alpaka::Vec{0, 0, 0}] == view0[offset_subview0]);
        REQUIRE(
            sub_view0[alpaka::Vec{0, 1, 0}]
            == view0[alpaka::Vec{offset_subview0.z(), offset_subview0.y() + 1, offset_subview0.x()}]);
        REQUIRE(
            sub_view0[extents_subview0 - alpaka::Vec{1, 1, 1}]
            == view0[offset_subview0 + extents_subview0 - alpaka::Vec{1, 1, 1}]);

        REQUIRE(offset_subview0.x() + extents_subview0.x() <= view0.getExtents().x());
        REQUIRE(offset_subview0.y() + extents_subview0.y() <= view0.getExtents().y());
        REQUIRE(offset_subview0.z() + extents_subview0.z() <= view0.getExtents().z());

        auto counter = offset_subview0;
        for(auto vec : alpaka::IdxRange{sub_view0.getExtents()})
        {
            REQUIRE_MESSAGE(sub_view0[vec] == counter.z() * 100 + counter.y() * 10 + counter.x(), "vec=" << vec);
            counter.x()++;
            if(counter.x() == offset_subview0.x() + extents_subview0.x())
            {
                counter.y()++;
                counter.x() = offset_subview0.x();
                if(counter.y() == offset_subview0.y() + extents_subview0.y())
                {
                    counter.z()++;
                    counter.y() = offset_subview0.y();
                }
            }
        }
    }
}
