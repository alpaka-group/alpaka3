/* OpStatus - Operation status enumeration for tensor operations
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

namespace alpaka::tensor
{
    // Operation status enumeration for provider delegation
    enum class OpStatus
    {
        Success, // Operation completed successfully
        Unsupported, // Operation not supported by this provider
        Error // Operation failed due to error
    };

} // namespace alpaka::tensor
