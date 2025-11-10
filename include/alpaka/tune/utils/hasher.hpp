//
// Created by tim on 03.11.25.
//

#ifndef HASHER_H
#define HASHER_H
#include <alpaka/tune/utils/serialize.hpp>

#include <cstddef>

namespace alpaka::tune::detail
{
    /**
     * @brief Computes a byte-level hash of any trivially copyable value ( this must be ensured by the caller)
     *
     * Implements the 64-bit FNV-1a hash algorithm for hashing raw object bytes.
     * Fast, constexpr-friendly, and suitable for deterministic non-cryptographic hashing.
     *
     * @see https://datatracker.ietf.org/doc/html/draft-eastlake-fnv
     *
     * @tparam T  Type of the value to hash (should be trivially copyable).
     * @param[in] value  Object whose bytes will be hashed.
     * @return 64-bit FNV-1a hash of the object's memory representation.
     */
    template<typename T>
    constexpr std::size_t hashBytes(T const& value)
    {
        auto const* ptr = reinterpret_cast<unsigned char const*>(&value);
        std::size_t hash = 1'469'598'103'934'665'603ULL;
        for(std::size_t i = 0; i < sizeof(T); ++i)
            hash = (hash ^ ptr[i]) * 1'099'511'628'211ULL;
        return hash;
    }

    template<typename T>
    struct Hasher
    {
        constexpr Hasher() = default;

        std::size_t operator()(T const& val) const
        {
            if constexpr(Hashable<T>)
            {
                return std::hash<T>{}(val);
            }
            if constexpr(concepts::Serializable<T>)
            {
                return std::hash<std::string>{}(detail::toStringGeneric(val));
            }
            else
            {
                static_assert(
                    !std::is_same_v<T, T>,
                    " Tunable types must be hashable: either specialize std::hash or provide a "
                    "serialization schema for your type @see alpaka::tune::Serializable");
                return hashBytes(val);
            }
        }
    };

    /**
     * @brief Combines a hash value into an existing hash seed.
     *
     * Implements the Boost `hash_combine` pattern to mix the hash of a value
     * into an accumulated hash seed.
     *
     * @see https://www.boost.org/doc/libs/1_55_0/doc/html/hash/reference.html#boost.hash_combine
     *
     * @tparam T  Type of the value to hash.
     * @param[in,out] seed  Current hash seed to be updated.
     * @param[in] value Value to incorporate into the hash.
     */
    template<typename T>
    constexpr void hash_combine(std::size_t& seed, T const& value)
    {
        Hasher<T> hasher;
        seed ^= hasher(value) + 0x9e37'79b9 + (seed << 6) + (seed >> 2);
    }

    // overload to perform boost hash combine on a std::vector given a seed
    template<typename T>
    constexpr void hash_combine(std::size_t& seed, std::vector<T> const& values)
    {
        for(auto const& value : values)
        {
            hash_combine(seed, value);
        }
    }

    template<typename T>
    std::size_t hashType(std::size_t& seed)
    {
        std::string name = onHost::demangledName<T>();
        hash_combine(seed, name);
        return seed;
    }
} // namespace alpaka::tune::detail
#endif // HASHER_H
