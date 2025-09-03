/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/onHost/example/executors.hpp>
#include <alpaka/onHost/executeForEach.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <iostream>

using namespace alpaka;

/** @file test the behaviour for constness of buffers/views/MdSpan
 *
 * We run two different test, one for access with 1D memory and the usage of scalars to access the memory and one for
 * multidimensional memory, because there are specializations for 1D in our implementations.
 */

/** validate that the buffer is mutable
 *
 * If we use non-outer const buffers/views in the function signature it should be possible to call the function even if
 * the buffer/view had an outer const before. The reason why we can not transform the outer const into an inner const
 * is that we requires this within @c KernelBundle to forward the kernel user arguments in a tuple to the user kernel.
 */
void requiresMutableBufferMd(concepts::MdSpan auto buffer)
{
    static_assert(!std::is_const_v<std::remove_pointer_t<decltype(buffer.data())>>);
    static_assert(!std::is_const_v<std::remove_reference_t<decltype(buffer[Vec{0, 0}])>>);
}

TEST_CASE("buffer const correctness 1D", "")
{
    auto buffer0 = onHost::allocHost<int>(10);
    static_assert(!std::is_const_v<std::remove_pointer_t<decltype(buffer0.data())>>);
    // mutable views
    {
        [[maybe_unused]] auto subBuffer0 = buffer0.getManagedSubView(2);
        static_assert(!std::is_const_v<std::remove_pointer_t<decltype(subBuffer0.data())>>);
        static_assert(!std::is_const_v<std::remove_reference_t<decltype(subBuffer0[0])>>);

        [[maybe_unused]] auto subOffsetBuffer0 = buffer0.getManagedSubView(1, 2);
        static_assert(!std::is_const_v<std::remove_pointer_t<decltype(subOffsetBuffer0.data())>>);
        static_assert(!std::is_const_v<std::remove_reference_t<decltype(subOffsetBuffer0[0])>>);

        [[maybe_unused]] auto subView0 = buffer0.getSubView(2);
        static_assert(!std::is_const_v<std::remove_pointer_t<decltype(subView0.data())>>);
        static_assert(!std::is_const_v<std::remove_reference_t<decltype(subView0[0])>>);

        [[maybe_unused]] auto subOffsetView0 = buffer0.getSubView(1, 2);
        static_assert(!std::is_const_v<std::remove_pointer_t<decltype(subOffsetView0.data())>>);
        static_assert(!std::is_const_v<std::remove_reference_t<decltype(subOffsetView0[0])>>);

        [[maybe_unused]] View view = buffer0.getView();
        static_assert(!std::is_const_v<std::remove_pointer_t<decltype(view.data())>>);
        static_assert(!std::is_const_v<std::remove_reference_t<decltype(view[0])>>);

        [[maybe_unused]] MdSpan mdSpan = buffer0.getMdSpan();
        static_assert(!std::is_const_v<std::remove_pointer_t<decltype(mdSpan.data())>>);
        static_assert(!std::is_const_v<std::remove_reference_t<decltype(mdSpan[0])>>);
    }
    // non-mutable views
    {
        onHost::ManagedView innerConstBuffer0 = buffer0.getConstManagedView();
        [[maybe_unused]] auto subBuffer0 = innerConstBuffer0.getManagedSubView(2);
        static_assert(std::is_const_v<std::remove_pointer_t<decltype(subBuffer0.data())>>);
        static_assert(std::is_const_v<std::remove_reference_t<decltype(subBuffer0[0])>>);

        [[maybe_unused]] auto subOffsetBuffer0 = innerConstBuffer0.getManagedSubView(1, 2);
        static_assert(std::is_const_v<std::remove_pointer_t<decltype(subOffsetBuffer0.data())>>);
        static_assert(std::is_const_v<std::remove_reference_t<decltype(subOffsetBuffer0[0])>>);

        [[maybe_unused]] auto subView0 = innerConstBuffer0.getSubView(2);
        static_assert(std::is_const_v<std::remove_pointer_t<decltype(subView0.data())>>);
        static_assert(std::is_const_v<std::remove_reference_t<decltype(subView0[0])>>);

        [[maybe_unused]] auto subOffsetView0 = innerConstBuffer0.getSubView(1, 2);
        static_assert(std::is_const_v<std::remove_pointer_t<decltype(subOffsetView0.data())>>);
        static_assert(std::is_const_v<std::remove_reference_t<decltype(subOffsetView0[0])>>);

        [[maybe_unused]] View view = innerConstBuffer0.getView();
        static_assert(std::is_const_v<std::remove_pointer_t<decltype(view.data())>>);
        static_assert(std::is_const_v<std::remove_reference_t<decltype(view[0])>>);

        [[maybe_unused]] MdSpan mdSpan = innerConstBuffer0.getMdSpan();
        static_assert(std::is_const_v<std::remove_pointer_t<decltype(mdSpan.data())>>);
        static_assert(std::is_const_v<std::remove_reference_t<decltype(mdSpan[0])>>);
    }
    // non-mutable views (outer const)
    {
        onHost::ManagedView const outerConstBuffer0 = buffer0;
        [[maybe_unused]] auto subBuffer0 = outerConstBuffer0.getManagedSubView(2);
        static_assert(std::is_const_v<std::remove_pointer_t<decltype(subBuffer0.data())>>);
        static_assert(std::is_const_v<std::remove_reference_t<decltype(subBuffer0[0])>>);

        [[maybe_unused]] auto subOffsetBuffer0 = outerConstBuffer0.getManagedSubView(1, 2);
        static_assert(std::is_const_v<std::remove_pointer_t<decltype(subOffsetBuffer0.data())>>);
        static_assert(std::is_const_v<std::remove_reference_t<decltype(subOffsetBuffer0[0])>>);

        [[maybe_unused]] auto subView0 = outerConstBuffer0.getSubView(2);
        static_assert(std::is_const_v<std::remove_pointer_t<decltype(subView0.data())>>);
        static_assert(std::is_const_v<std::remove_reference_t<decltype(subView0[0])>>);

        [[maybe_unused]] auto subOffsetView0 = outerConstBuffer0.getSubView(1, 2);
        static_assert(std::is_const_v<std::remove_pointer_t<decltype(subOffsetView0.data())>>);
        static_assert(std::is_const_v<std::remove_reference_t<decltype(subOffsetView0[0])>>);

        [[maybe_unused]] View view = outerConstBuffer0.getView();
        static_assert(std::is_const_v<std::remove_pointer_t<decltype(view.data())>>);
        static_assert(std::is_const_v<std::remove_reference_t<decltype(view[0])>>);

        [[maybe_unused]] MdSpan mdSpan = outerConstBuffer0.getMdSpan();
        static_assert(std::is_const_v<std::remove_pointer_t<decltype(mdSpan.data())>>);
        static_assert(std::is_const_v<std::remove_reference_t<decltype(mdSpan[0])>>);
    }
}

TEST_CASE("buffer const correctness MD", "")
{
    auto buffer0 = onHost::allocHost<int>(Vec{10, 10});
    requiresMutableBufferMd(buffer0);
    static_assert(!std::is_const_v<std::remove_pointer_t<decltype(buffer0.data())>>);
    // mutable views
    {
        [[maybe_unused]] auto subBuffer0 = buffer0.getManagedSubView(Vec{2, 2});
        static_assert(!std::is_const_v<std::remove_pointer_t<decltype(subBuffer0.data())>>);
        static_assert(!std::is_const_v<std::remove_reference_t<decltype(subBuffer0[Vec{0, 0}])>>);

        [[maybe_unused]] auto subOffsetBuffer0 = buffer0.getManagedSubView(Vec{1, 1}, Vec{2, 2});
        static_assert(!std::is_const_v<std::remove_pointer_t<decltype(subOffsetBuffer0.data())>>);
        static_assert(!std::is_const_v<std::remove_reference_t<decltype(subOffsetBuffer0[Vec{0, 0}])>>);

        [[maybe_unused]] auto subView0 = buffer0.getSubView(Vec{2, 2});
        static_assert(!std::is_const_v<std::remove_pointer_t<decltype(subView0.data())>>);
        static_assert(!std::is_const_v<std::remove_reference_t<decltype(subView0[Vec{0, 0}])>>);

        [[maybe_unused]] auto subOffsetView0 = buffer0.getSubView(Vec{1, 1}, Vec{2, 2});
        static_assert(!std::is_const_v<std::remove_pointer_t<decltype(subOffsetView0.data())>>);
        static_assert(!std::is_const_v<std::remove_reference_t<decltype(subOffsetView0[Vec{0, 0}])>>);

        [[maybe_unused]] View view = buffer0.getView();
        static_assert(!std::is_const_v<std::remove_pointer_t<decltype(view.data())>>);
        static_assert(!std::is_const_v<std::remove_reference_t<decltype(view[Vec{0, 0}])>>);

        [[maybe_unused]] MdSpan mdSpan = buffer0.getMdSpan();
        static_assert(!std::is_const_v<std::remove_pointer_t<decltype(mdSpan.data())>>);
        static_assert(!std::is_const_v<std::remove_reference_t<decltype(mdSpan[Vec{0, 0}])>>);
    }
    // non-mutable views
    {
        onHost::ManagedView innerConstBuffer0 = buffer0.getConstManagedView();

        [[maybe_unused]] auto subBuffer0 = innerConstBuffer0.getManagedSubView(Vec{2, 2});
        static_assert(std::is_const_v<std::remove_pointer_t<decltype(subBuffer0.data())>>);
        static_assert(std::is_const_v<std::remove_reference_t<decltype(subBuffer0[Vec{0, 0}])>>);

        [[maybe_unused]] auto subOffsetBuffer0 = innerConstBuffer0.getManagedSubView(Vec{1, 1}, Vec{2, 2});
        static_assert(std::is_const_v<std::remove_pointer_t<decltype(subOffsetBuffer0.data())>>);
        static_assert(std::is_const_v<std::remove_reference_t<decltype(subOffsetBuffer0[Vec{0, 0}])>>);

        [[maybe_unused]] auto subView0 = innerConstBuffer0.getSubView(Vec{2, 2});
        static_assert(std::is_const_v<std::remove_pointer_t<decltype(subView0.data())>>);
        static_assert(std::is_const_v<std::remove_reference_t<decltype(subView0[Vec{0, 0}])>>);

        [[maybe_unused]] auto subOffsetView0 = innerConstBuffer0.getSubView(Vec{1, 1}, Vec{2, 2});
        static_assert(std::is_const_v<std::remove_pointer_t<decltype(subOffsetView0.data())>>);
        static_assert(std::is_const_v<std::remove_reference_t<decltype(subOffsetView0[Vec{0, 0}])>>);

        [[maybe_unused]] View view = innerConstBuffer0.getView();
        static_assert(std::is_const_v<std::remove_pointer_t<decltype(view.data())>>);
        static_assert(std::is_const_v<std::remove_reference_t<decltype(view[Vec{0, 0}])>>);

        [[maybe_unused]] MdSpan mdSpan = innerConstBuffer0.getMdSpan();
        static_assert(std::is_const_v<std::remove_pointer_t<decltype(mdSpan.data())>>);
        static_assert(std::is_const_v<std::remove_reference_t<decltype(mdSpan[Vec{0, 0}])>>);
    }
    // non-mutable views (outer const)
    {
        onHost::ManagedView const outerConstBuffer0 = buffer0;
        requiresMutableBufferMd(outerConstBuffer0);

        [[maybe_unused]] auto subBuffer0 = outerConstBuffer0.getManagedSubView(Vec{2, 2});
        static_assert(std::is_const_v<std::remove_pointer_t<decltype(subBuffer0.data())>>);
        static_assert(std::is_const_v<std::remove_reference_t<decltype(subBuffer0[Vec{0, 0}])>>);

        [[maybe_unused]] auto subOffsetBuffer0 = outerConstBuffer0.getManagedSubView(Vec{1, 1}, Vec{2, 2});
        static_assert(std::is_const_v<std::remove_pointer_t<decltype(subOffsetBuffer0.data())>>);
        static_assert(std::is_const_v<std::remove_reference_t<decltype(subOffsetBuffer0[Vec{0, 0}])>>);

        [[maybe_unused]] auto subView0 = outerConstBuffer0.getSubView(Vec{2, 2});
        static_assert(std::is_const_v<std::remove_pointer_t<decltype(subView0.data())>>);
        static_assert(std::is_const_v<std::remove_reference_t<decltype(subView0[Vec{0, 0}])>>);

        [[maybe_unused]] auto subOffsetView0 = outerConstBuffer0.getSubView(Vec{1, 1}, Vec{2, 2});
        static_assert(std::is_const_v<std::remove_pointer_t<decltype(subOffsetView0.data())>>);
        static_assert(std::is_const_v<std::remove_reference_t<decltype(subOffsetView0[Vec{0, 0}])>>);

        [[maybe_unused]] View view = outerConstBuffer0.getView();
        static_assert(std::is_const_v<std::remove_pointer_t<decltype(view.data())>>);
        static_assert(std::is_const_v<std::remove_reference_t<decltype(view[Vec{0, 0}])>>);

        [[maybe_unused]] MdSpan mdSpan = outerConstBuffer0.getMdSpan();
        static_assert(std::is_const_v<std::remove_pointer_t<decltype(mdSpan.data())>>);
        static_assert(std::is_const_v<std::remove_reference_t<decltype(mdSpan[Vec{0, 0}])>>);
    }
}
