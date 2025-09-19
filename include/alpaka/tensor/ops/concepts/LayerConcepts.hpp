#pragma once
#include <type_traits>

namespace alpaka::tensor::ops
{

    // Concept to identify valid layers that can be added to MultiSequential
    template<typename L, typename Device, typename Exec, typename Queue>
    concept Layer = requires {
        // Layer must have input and output type definitions
        typename L::input_type;
        typename L::output_type;
    } && requires(L layer, Exec const& exec, Device& device, Queue& queue, typename L::input_type& input) {
        // Layer must be callable with exec, device, queue, and input
        {
            layer.template operator()<Exec, Queue>(exec, device, queue, input)
        } -> std::same_as<typename L::output_type>;
    };

    // Helper trait to check if a type is a valid layer
    template<typename L, typename Device, typename Exec, typename Queue>
    constexpr bool is_layer_v = Layer<L, Device, Exec, Queue>;

} // namespace alpaka::tensor::ops
