/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/core/Dict.hpp>
#include <alpaka/core/Tag.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <functional>

using namespace alpaka;
#if 1
TEST_CASE("dict mutate entry", "")
{
    ALPAKA_TAG(a_);
    ALPAKA_TAG(b_);
    ALPAKA_TAG(c_);

    static_assert(std::is_same<decltype(a_), decltype(a_)>::value);
    static_assert(!std::is_same<decltype(a_), decltype(b_)>::value);

    int aValue = 42;
    int bValue = 43;
    Dict dictionary = Dict{std::make_tuple(DictEntry(a_, std::ref(aValue)), DictEntry(b_, bValue))};

    static_assert(hasTag(dictionary, a_));
    static_assert(hasTag(dictionary, b_));
    static_assert(!hasTag(dictionary, c_));

    static_assert(idx(dictionary, a_) == 0);
    CHECK(getTag(dictionary, a_) == 42);
    CHECK(getTag(dictionary, b_) == 43);

    aValue++;
    bValue += 2;

    CHECK(getTag(dictionary, a_) == 43);
    CHECK(getTag(dictionary, b_) == 43);
    CHECK(dictionary[a_] == 43);
    CHECK(dictionary[b_] == 43);

    DictEntry entry0{a_, std::ref(aValue)};
    Dict dictionary2 = {entry0, DictEntry(b_, bValue)};
    CHECK(getTag(dictionary2, a_) == 43);

    Dict dictionary4 = Dict{std::make_tuple(DictEntry(a_, std::ref(aValue)), DictEntry(b_, bValue))};
    dictionary4[a_] = 1;
    CHECK(dictionary4[a_] == 1);

    Dict dictionary5 = Dict{std::make_tuple(DictEntry(c_, 5))};
    auto joinedDict = joinDict(dictionary4, dictionary5);
    CHECK(joinedDict[a_] == 1);
    CHECK(joinedDict[c_] == 5);
}
#endif
