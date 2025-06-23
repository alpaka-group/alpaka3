/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/example/executeForEach.hpp>
#include <alpaka/example/executors.hpp>

#include <catch2/catch_test_macros.hpp>


using namespace alpaka;
using namespace alpaka::onHost;

TEST_CASE("mdIterator", "")
{
    constexpr auto numElements = Vec<size_t, 2u>{17, 31};
    auto span = onHost::allocHost<uint32_t>(numElements);

    size_t counter = 0u;
    for(auto& v : span)
        v = counter++;

    // validate by using the forward iterator
    size_t refence = 0u;
    for(auto const& v : span)
    {
        CHECK(v == refence);
        ++refence;
    }

    // validate without using the forward iterator
    meta::ndLoopIncIdx(numElements, [&](auto idx) { CHECK(span[idx] == linearize(numElements, idx)); });
}
