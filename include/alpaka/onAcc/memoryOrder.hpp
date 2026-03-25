/* Copyright 2025 Mehmet Yusufoglu, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <string>

/**
 * @brief Provides scopes for atomic and memory fence operations, analogous to NVIDIA CUDA's atomic and fence scopes.
 *
 * This namespace defines the visibility scopes for atomic operations and memory fences,
 * which control the visibility of memory operations across threads, blocks, and devices.
 * The provided scopes are:
 * - Block: Visibility within a thread block.
 * - Device: Visibility across all thread blocks on the same device.
 * - System: System-wide visibility, mapped to the strongest available atomic/fence by the backend.
 *
 * @see alpaka::onAcc::atomicAdd, alpaka::onAcc::memFence
 */
namespace alpaka::onAcc
{
    namespace order
    {

        /**
         * @brief Base tag for memory order types.
         *
         * This tag can be used to constrain APIs that accept only valid memory orders.
         */
        struct MemoryOrderTag
        {
        };

        /**
         * @brief Sequentially consistent memory ordering.
         *
         * This is the strongest memory ordering and provides a single global order
         * for all sequentially consistent operations.
         */
        struct SeqCst : MemoryOrderTag
        {
            static std::string getName()
            {
                return "SeqCst";
            }
        };

        inline constexpr SeqCst seq_cst{};

        /**
         * @brief Acquire-release memory ordering.
         *
         * Ensures both acquire and release semantics. This ordering is typically
         * used for read-modify-write operations.
         */
        struct AcqRel : MemoryOrderTag
        {
            static std::string getName()
            {
                return "AcqRel";
            }
        };

        inline constexpr AcqRel acq_rel{};

        /**
         * @brief Release memory ordering.
         *
         * Ensures that all writes before the operation become visible before the
         * release operation itself becomes visible.
         */
        struct Release : MemoryOrderTag
        {
            static std::string getName()
            {
                return "Release";
            }
        };

        inline constexpr Release release{};

        /**
         * @brief Acquire memory ordering.
         *
         * Ensures that all reads and writes after the operation observe effects
         * that became visible before the acquire operation.
         */
        struct Acquire : MemoryOrderTag
        {
            static std::string getName()
            {
                return "Acquire";
            }
        };

        inline constexpr Acquire acquire{};

        /**
         * @brief Relaxed memory ordering.
         *
         * Provides atomicity without additional ordering guarantees.
         */
        struct Relaxed : MemoryOrderTag
        {
            static std::string getName()
            {
                return "Relaxed";
            }
        };

        inline constexpr Relaxed relaxed{};
    } // namespace order

    namespace concepts
    {
        template<typename T>
        concept MemoryOrder = std::derived_from<T, order::MemoryOrderTag>;
    } // namespace concepts

} // namespace alpaka::onAcc
