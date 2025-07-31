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
        /**
         * Constructor
         * @param freeOp Function to be called when the shared_ptr is destroyed after all actions are executed.
         *               All dependencies required to deallocate the memory must be holed by freeOp.
         */
        ManagedDealloc(std::function<void()> freeOp) : freeOp{std::move(freeOp)}
        {
        }

        ~ManagedDealloc()
        {
            // Execute all actions before freeing the memory
            {
                std::lock_guard<std::mutex> lock{actionGuard};
                for(auto& action : actions)
                {
                    action();
                }
            }
            freeOp();
        }

        /** Add an action to be executed when the shared_ptr is destroyed.
         *
         * @param action Callable to execute on destruction.
         */
        void addAction(std::function<void()> action)
        {
            std::lock_guard<std::mutex> lock{actionGuard};
            actions.push_back(std::move(action));
        }

        std::shared_ptr<ManagedDealloc> getSharedPtr()
        {
            return this->shared_from_this();
        }

    private:
        std::function<void()> freeOp;
        std::mutex actionGuard;
        std::vector<std::function<void()>> actions;
    };
} // namespace alpaka::onHost::mem
