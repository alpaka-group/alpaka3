#pragma once
#include <alpaka/alpaka.hpp>
#include <alpaka/tensor/core/TensorCore.hpp>
#include <alpaka/tensor/layers/base/LayerConcepts.hpp>

#include <cstddef>
#include <utility>

namespace alpaka::tensor::ops::layers
{
    // Simple Embedding lookup layer: indices [N] -> output [N, D]
    // Host-side reference implementation for now (copies to device at the end)
    template<typename Device>
    struct EmbeddingLayer
    {
        using input_type = tensor::Tensor1D<std::size_t, Device>; // token ids
        using output_type = tensor::Tensor2D<float, Device>; // [N, D]

        tensor::Tensor2D<float, Device> weights; // [V, D]

        explicit EmbeddingLayer(tensor::Tensor2D<float, Device> W) : weights(std::move(W))
        {
        }

        template<typename Exec, typename Queue>
        tensor::Tensor2D<float, Device> operator()(
            Exec const& exec,
            Device& device,
            Queue& queue,
            tensor::Tensor1D<std::size_t, Device>& indices)
        {
            auto V = weights.shape()[0];
            auto D = weights.shape()[1];
            auto N = indices.shape()[0];
            tensor::Tensor2D<float, Device> out(device, {N, D}, "embedding_out");

            // Host fallback gather
            indices.toHost(device, queue);
            weights.toHost(device, queue);
            float const* W = weights.hostData();
            std::size_t const* ids = indices.hostData();
            float* O = out.hostData();
            for(std::size_t n = 0; n < N; ++n)
            {
                auto id = ids[n];
                if(id >= V)
                    id = 0; // clamp unknown to 0 for now
                auto src = &W[id * D];
                auto dst = &O[n * D];
                for(std::size_t d = 0; d < D; ++d)
                    dst[d] = src[d];
            }
            out.markHostModified();
            out.ensureOnDevice(device, queue);
            (void) exec;
            return out;
        }
    };

    template<typename Device>
    EmbeddingLayer<Device> embedding(tensor::Tensor2D<float, Device> weights)
    {
        return EmbeddingLayer<Device>{std::move(weights)};
    }
} // namespace alpaka::tensor::ops::layers
