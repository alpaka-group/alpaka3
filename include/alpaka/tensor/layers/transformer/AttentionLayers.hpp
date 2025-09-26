#pragma once
#include <alpaka/alpaka.hpp>
#include <alpaka/tensor/core/TensorCore.hpp>
#include <alpaka/tensor/ops/linear/LinearOps.hpp>
#include <alpaka/tensor/ops/softmax/Softmax.hpp>

#include <cmath>
#include <utility>

namespace alpaka::tensor::ops::layers
{
    // Minimal scaled dot-product attention (single-head) scaffold
    // Input: Q,K,V as [N, D] each. Output: [N, D]
    // This is a simple host-reference implementation using existing ops for softmax when possible
    template<typename Device>
    struct ScaledDotAttentionLayer
    {
        using input_type = tensor::Tensor2D<float, Device>; // we'll take Q as input; K,V bound in layer
        using output_type = tensor::Tensor2D<float, Device>;

        tensor::Tensor2D<float, Device> K; // [N, D]
        tensor::Tensor2D<float, Device> V; // [N, D]

        explicit ScaledDotAttentionLayer(tensor::Tensor2D<float, Device> k, tensor::Tensor2D<float, Device> v)
            : K(std::move(k))
            , V(std::move(v))
        {
        }

        template<typename Exec, typename Queue>
        output_type operator()(Exec const& exec, Device& device, Queue& queue, input_type& Q)
        {
            auto N = Q.shape()[0];
            auto D = Q.shape()[1];
            // Compute scores S = (Q * K^T) / sqrt(D) on host for now
            tensor::Tensor2D<float, Device> S(device, {N, N}, "attn_scores");
            float const* q = Q.hostData();
            float const* k = K.hostData();
            float const* v = V.hostData();
            float* s = S.hostData();
            float scale = 1.0f / std::sqrt(float(D));
            for(std::size_t i = 0; i < N; ++i)
            {
                for(std::size_t j = 0; j < N; ++j)
                {
                    double acc = 0.0;
                    for(std::size_t d = 0; d < D; ++d)
                        acc += double(q[i * D + d]) * double(k[j * D + d]);
                    s[i * N + j] = float(acc) * scale;
                }
            }
            S.markHostModified();
            S.ensureOnDevice(device, queue);

            // softmax over rows of S -> P
            tensor::Tensor2D<float, Device> P(device, {N, N}, "attn_prob");
            ops::softmax_2d<float>(exec, device, queue, S, P);

            // Output O = P * V
            char const* devFlag = std::getenv("ALPAKA_ATTENTION_DEVICE_GEMM");
            if(devFlag
               && (std::string(devFlag) == "1" || std::string(devFlag) == "ON" || std::string(devFlag) == "on"))
            {
                tensor::Tensor2D<float, Device> O(device, {N, D}, "attn_out");
                ops::matmul_2d(exec, device, queue, P, V, O);
                return O;
            }
            else
            {
                // Host reference path (default): ensures numerical parity
                P.toHost(device, queue);
                tensor::Tensor2D<float, Device> O(device, {N, D}, "attn_out");
                float const* p = P.hostData();
                float const* v = V.hostData();
                float* oh = O.hostData();
                for(std::size_t i = 0; i < N; ++i)
                    for(std::size_t d = 0; d < D; ++d)
                    {
                        double acc = 0.0;
                        for(std::size_t j = 0; j < N; ++j)
                            acc += double(p[i * N + j]) * double(v[j * D + d]);
                        oh[i * D + d] = float(acc);
                    }
                O.markHostModified();
                O.ensureOnDevice(device, queue);
                return O;
            }
        }
    };

    template<typename Device>
    ScaledDotAttentionLayer<Device> attention(tensor::Tensor2D<float, Device> K, tensor::Tensor2D<float, Device> V)
    {
        return ScaledDotAttentionLayer<Device>{std::move(K), std::move(V)};
    }
} // namespace alpaka::tensor::ops::layers
