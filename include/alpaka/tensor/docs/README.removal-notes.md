This repository previously contained `ElementwiseFixed.hpp` with minimal tutorial-style kernels.

As part of DRY and cleanup:
- The file was removed.
- Production code should use `ops/ElementwiseGeneric.hpp` for elementwise unary/binary operations.
- Examples and layers have been adjusted to use generic or canonical kernels.

If you had code including `ElementwiseFixed.hpp`, switch to:
- `#include <alpaka/tensor/ops/ElementwiseGeneric.hpp>`
- Use `ops::binary(...)`, `ops::unary(...)`, `ops::add(...)`, or `ops::relu(...)` as appropriate.
