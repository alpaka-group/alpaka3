/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/onHost/example/executors.hpp>
#include <alpaka/onHost/executeForEach.hpp>

#include <catch2/catch_test_macros.hpp>


using namespace alpaka;
using namespace alpaka::onHost;

TEST_CASE("mdIterator", "")
{
    constexpr auto numElements = CVec<size_t, 17, 31>{};
    alpaka::concepts::IMdSpan auto span = onHost::allocHost<uint32_t>(numElements);

    size_t counter = 0u;
    for(uint32_t& v : span)
        v = counter++;

    // validate by using the forward iterator
    size_t refence = 0u;
    for(uint32_t v : span)
    {
        CHECK(v == refence);
        ++refence;
    }

    // validate without using the forward iterator
    meta::ndLoopIncIdx(
        numElements,
        [&](alpaka::concepts::Vector<size_t, 2> auto idx) { CHECK(span[idx] == linearize(numElements, idx)); });
}
