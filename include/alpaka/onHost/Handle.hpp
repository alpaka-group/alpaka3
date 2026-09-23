/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <memory>
#include <mutex>
#include <type_traits>

namespace alpaka::onHost
{
    template<typename T_Object, typename... T_Args>
    inline auto makeSharedSingleton(T_Args&&... args)
    {
        static std::mutex mutex;
        static std::weak_ptr<T_Object> platform;

        std::lock_guard<std::mutex> lk(mutex);
        if(auto sharedPtr = platform.lock())
        {
            return sharedPtr;
        }
        auto newPlatform = std::make_shared<T_Object>(std::forward<T_Args>(args)...);
        platform = newPlatform;
        return newPlatform;
    }

    template<typename T>
    using Handle = std::shared_ptr<T>;
} // namespace alpaka::onHost
