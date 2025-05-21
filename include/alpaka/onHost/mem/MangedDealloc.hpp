/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <functional>
#include <memory>

namespace alpaka::onHost::mem
{
    /** Manage the deallocation of memory
     *
     * This class is used to manage the deallocation of memory in a shared_ptr.
     * It takes a function that will be called when the shared_ptr is destroyed.
     * This is useful for managing memory that needs to be deallocated
     * when the shared_ptr goes out of scope.
     */
    struct ManagedDealloc : std::enable_shared_from_this<ManagedDealloc>
    {
        std::function<void()> freeOp;

        /**
         * Constructor
         * @param freeOp Function to be called when the shared_ptr is destroyed.
         *               All dependencies required to deallocate the memory must be holed by freeOp.
         */
        ManagedDealloc(std::function<void()> freeOp) : freeOp{std::move(freeOp)}
        {
        }

        ~ManagedDealloc()
        {
            freeOp();
        }

        std::shared_ptr<ManagedDealloc> getSharedPtr()
        {
            return this->shared_from_this();
        }
    };
} // namespace alpaka::onHost::mem
