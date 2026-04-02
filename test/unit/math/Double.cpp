/* Copyright 2022 Jakob Krude, Benjamin Worpitz, Bernhard Manfred Gruber, Sergei Bastrakov, Jan Stephans
 * SPDX-License-Identifier: MPL-2.0
 */

#include "Functor.hpp"
#include "TestTemplate.hpp"

#include <alpaka/alpaka.hpp>
#include <alpaka/meta/meta.hpp>

#include <catch2/catch_template_test_macros.hpp>

#include <tuple>

using namespace alpaka;

using TestBackends = std::decay_t<decltype(onHost::allBackends(onHost::enabledDeviceSpecs, exec::enabledExecutors))>;

// This file only has unit tests for real numbers in order to split the tests between object files
using FunctorsReal = alpaka::meta::
    Concatenate<mathtest::UnaryFunctorsReal, mathtest::BinaryFunctorsReal, mathtest::TernaryFunctorsReal>;
using TestAccFunctorTuplesReal = alpaka::meta::CartesianProduct<std::tuple, TestBackends, FunctorsReal>;

TEMPLATE_LIST_TEST_CASE("mathOpsDouble", "[math] [operator]", TestAccFunctorTuplesReal)
{
    using Backend = std::tuple_element_t<0u, TestType>;
    using Functor = std::tuple_element_t<1u, TestType>;
    auto cfg = Backend::makeDict();
    auto testTemplate = mathtest::TestTemplate<Functor>{};
    testTemplate.template operator()<double>(cfg);
}
