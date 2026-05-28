/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/config.hpp"

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
        std::string simplified;
        simplified.reserve(deName.size());

        int templateDepth = 0;
        // Simplify nested templates by removing template arguments, e.g., <...>
        for(char const c : deName)
        {
            if(c == '<')
            {
                if(templateDepth++ == 0)
                    simplified += "<...>";
                continue;
            }
            if(c == '>')
            {
                if(templateDepth > 0)
                {
                    --templateDepth;
                    continue;
                }
            }
            if(templateDepth > 0)
                continue;
            simplified += c;
        }

        // Remove "alpaka::" from the signatures
        std::string withoutAlpaka;
        withoutAlpaka.reserve(simplified.size());
        constexpr std::string_view alpakaNamespace = "alpaka::";
        for(size_t i = 0; i < simplified.size();)
        {
            if(simplified.compare(i, alpakaNamespace.size(), alpakaNamespace) == 0)
            {
                i += alpakaNamespace.size();
                continue;
            }
            withoutAlpaka += simplified[i++];
        }
        simplified = std::move(withoutAlpaka);
        return simplified;
    }

} // namespace alpaka::onHost
