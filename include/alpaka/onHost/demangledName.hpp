/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/config.hpp"

#include <regex>
#include <source_location>
#include <string>
#include <string_view>

/** This type is required to be in the global namespace to avoid invalid offsets during demangling */
struct AlpakaDemangleReferenceType
{
};

namespace alpaka::onHost
{
    /// \file
    /// use source_location to derive the demangled type name
    /// based on:
    /// https://www.reddit.com/r/cpp/comments/lfi6jt/finally_a_possibly_portable_way_to_convert_types/?utm_source=share&utm_medium=web3x&utm_name=web3xcss&utm_term=1&utm_content=share_button

    template<typename T>
    constexpr auto EmbedTypeIntoSignature()
    {
        return std::string_view{std::source_location::current().function_name()};
    }

    template<typename T>
    struct Demangled
    {
        static constexpr auto name()
        {
            constexpr size_t testSignatureLength = sizeof("AlpakaDemangleReferenceType") - 1;
            auto const DummySignature = EmbedTypeIntoSignature<AlpakaDemangleReferenceType>();
            // count char's until the type name starts
            auto const startPosition = DummySignature.find("AlpakaDemangleReferenceType");
            // count char's after the type information by removing type name information and pre information
            auto const tailLength = DummySignature.size() - startPosition - testSignatureLength;
            auto const EmbeddingSignature = EmbedTypeIntoSignature<T>();
            auto const typeLength = EmbeddingSignature.size() - startPosition - tailLength;
            return EmbeddingSignature.substr(startPosition, typeLength);
        }
    };

    template<typename T>
    constexpr auto demangledName()
    {
        return std::string(Demangled<T>::name());
    }

    template<typename T>
    constexpr auto demangledName(T const&)
    {
        return std::string(Demangled<T>::name());
    }

    /** Simplify the C++ signature of a function
     *
     *  Template parameters will be left out and the alpaka namespace will be removed.
     */
    inline std::string simplifyFunctionSignature(std::string const& deName)
    {
        // 1. Remove the type assignments in template parameters (e.g., T_DeviceKind = ...)
        std::string simplified = std::regex_replace(deName, std::regex("<[^>]*=\\s*[^>]*>"), "<...>");

        // 2. Remove redundant occurrences of "alpaka::" within the template arguments (keep it once)
        simplified = std::regex_replace(simplified, std::regex("alpaka::(?![A-Za-z0-9_]+<)"), "");

        // 3. Simplify nested templates by removing template arguments, e.g., <...>
        simplified = std::regex_replace(simplified, std::regex("<[^>]*>"), "<...>");

        // 4. Optionally remove remaining `alpaka::` namespaces if desired
        simplified = std::regex_replace(simplified, std::regex("^(alpaka::)+"), "");

        return simplified;
    }

    /** Get a simplified demangled name of an object
     *
     * Template parameters will be left out and the alpaka namespace will be removed.
     */
    inline std::string demangledNameShort(auto const& demangledName)
    {
        return simplifyDemangedName(demangledName(demangledName));
    }

} // namespace alpaka::onHost
