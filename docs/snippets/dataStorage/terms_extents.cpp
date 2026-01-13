#include <alpaka/alpaka.hpp>

int main()
{
    // BEGIN-DATASTORAGE-termExtents
    auto extents = alpaka::Vec<uint32_t, 2u>{3u, 5u};
    auto buffer = alpaka::onHost::allocHost<int>(extents);
    // access element column 3 and row 1
    int value = buffer[alpaka::Vec{1u, 3u}];
    // END-DATASTORAGE-termExtents

    alpaka::unused(value);

    return 0;
}
