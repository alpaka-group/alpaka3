//
// Created by tim on 03.11.25.
//

#ifndef SERIALIZE_H
#define SERIALIZE_H
#include <alpaka/tune/concepts.hpp>

namespace alpaka::tune::detail
{
    template<tune::concepts::Serializable T>
    std::string toStringGeneric(T const& value)
    {
        if constexpr(concepts::serialize::HasTraitSerializer<T>)
        {
            return ::alpaka::tune::trait::Serialize<T>{}(value);
        }
        else if constexpr(std::is_convertible_v<T, std::string>)
        {
            return std::string(value);
        }
        else if constexpr(std::is_arithmetic_v<T> || concepts::serialize::HasStdToString<T>)
        {
            return std::to_string(value);
        }
        else if constexpr(concepts::serialize::HasToStringMethod<T>)
        {
            return value.toString();
        }
        else if constexpr(concepts::serialize::HasStreamOperator<T>)
        {
            std::ostringstream oss;
            oss << value;
            return oss.str();
        }
        else
        {
            static_assert(!std::is_same_v<T, T>, "Could not convert type to string! ");
        }
        return "";
    }
} // namespace alpaka::tune::detail
#endif // SERIALIZE_H
