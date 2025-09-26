#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/tensor/context/CleanTensorOpContext.hpp>
#include <alpaka/tensor/core/TensorCore.hpp>
#include <alpaka/tensor/layers/normalization/NormalizationLayers.hpp>
#include <alpaka/tensor/layers/transformer/AttentionLayers.hpp>
#include <alpaka/tensor/ops/bias/BiasAdd.hpp>
#include <alpaka/tensor/ops/elementwise/ActivationOps.hpp>
#include <alpaka/tensor/ops/linear/LinearOps.hpp>
#include <alpaka/tensor/ops/normalization/LayerNorm.hpp>

#include <cmath>
#include <optional>
#include <stdexcept>
#include <utility>

namespace alpaka::tensor::ops::layers
{

    // Self-Attention over [M, D] using optional learned projections (single-head)
    // If Wq/Wk/Wv are not provided, defaults to identity projections (Q=K=V=input)
    template<typename Device>
    struct SelfAttention2DLayer
    {
        using input_type = tensor::Tensor2D<float, Device>;
        using output_type = tensor::Tensor2D<float, Device>;

        // Optional projections: [D,D]
        mutable std::optional<tensor::Tensor2D<float, Device>> Wq;
        mutable std::optional<tensor::Tensor2D<float, Device>> Wk;
        mutable std::optional<tensor::Tensor2D<float, Device>> Wv;
        // Optional output projection: [D,D]
        mutable std::optional<tensor::Tensor2D<float, Device>> Wo;

        // If true, prefer device GEMM path for attention output (P*V)
        bool preferDeviceGemm{false};

        template<typename Exec, typename Queue>
        output_type operator()(Exec const& exec, Device& device, Queue& queue, input_type& in) const
        {
            auto M = in.shape()[0];
            auto D = in.shape()[1];
            if(M == 0 || D == 0)
            {
                return tensor::Tensor2D<float, Device>(device, {M, D}, "self_attn_out");
            }

            // Compute Q, K, V
            tensor::Tensor2D<float, Device> Q(device, {M, D}, "Q");
            tensor::Tensor2D<float, Device> K(device, {M, D}, "K");
            tensor::Tensor2D<float, Device> V(device, {M, D}, "V");

            if(Wq && Wk && Wv)
            {
                ops::matmul_2d(exec, device, queue, in, const_cast<tensor::Tensor2D<float, Device>&>(*Wq), Q);
                ops::matmul_2d(exec, device, queue, in, const_cast<tensor::Tensor2D<float, Device>&>(*Wk), K);
                ops::matmul_2d(exec, device, queue, in, const_cast<tensor::Tensor2D<float, Device>&>(*Wv), V);
            }
            else
            {
                // Identity projections: Q=K=V=in
                in.toHost(device, queue);
                std::memcpy(Q.hostData(), in.hostData(), sizeof(float) * Q.size());
                std::memcpy(K.hostData(), in.hostData(), sizeof(float) * K.size());
                std::memcpy(V.hostData(), in.hostData(), sizeof(float) * V.size());
                Q.markHostModified();
                K.markHostModified();
                V.markHostModified();
                Q.ensureOnDevice(device, queue);
                K.ensureOnDevice(device, queue);
                V.ensureOnDevice(device, queue);
            }

            // Scaled dot-product attention using existing layer scaffold
            auto attn = layers::attention<Device>(std::move(K), std::move(V));
            auto O = attn(exec, device, queue, Q);

            if(Wo)
            {
                tensor::Tensor2D<float, Device> P(device, {M, D}, "attn_proj");
                ops::matmul_2d(exec, device, queue, O, const_cast<tensor::Tensor2D<float, Device>&>(*Wo), P);
                return P;
            }
            return O;
        }
    };

    template<typename Device>
    SelfAttention2DLayer<Device> selfAttention2d(
        std::optional<tensor::Tensor2D<float, Device>> Wq = std::nullopt,
        std::optional<tensor::Tensor2D<float, Device>> Wk = std::nullopt,
        std::optional<tensor::Tensor2D<float, Device>> Wv = std::nullopt,
        std::optional<tensor::Tensor2D<float, Device>> Wo = std::nullopt,
        bool preferDeviceGemm = false)
    {
        SelfAttention2DLayer<Device> l;
        l.Wq = std::move(Wq);
        l.Wk = std::move(Wk);
        l.Wv = std::move(Wv);
        l.Wo = std::move(Wo);
        l.preferDeviceGemm = preferDeviceGemm;
        return l;
    }

    // Position-wise FeedForward: Z = GELU(X * W1) * W2
    template<typename Device>
    struct FeedForward2DLayer
    {
        using input_type = tensor::Tensor2D<float, Device>;
        using output_type = tensor::Tensor2D<float, Device>;

        mutable tensor::Tensor2D<float, Device> W1; // [D, 4D]
        mutable tensor::Tensor2D<float, Device> W2; // [4D, D]

        // Optional clean context for provider-accelerated GELU
        mutable void* context{nullptr};

        template<typename Exec, typename Queue>
        output_type operator()(Exec const& exec, Device& device, Queue& queue, input_type& in) const
        {
            auto M = in.shape()[0];
            auto D = in.shape()[1];
            if(M == 0 || D == 0)
                return tensor::Tensor2D<float, Device>(device, {M, D}, "ffn_out");

            tensor::Tensor2D<float, Device> Z(device, {M, static_cast<std::size_t>(4 * D)}, "ffn_Z");
            ops::matmul_2d(exec, device, queue, in, const_cast<tensor::Tensor2D<float, Device>&>(W1), Z);

            if(context)
            {
                using CleanCtx = tensor::CleanTensorOpContext<Exec, Device, Queue>;
                auto* ctx = static_cast<CleanCtx*>(context);
                ctx->template gelu<float, 2>(Z);
            }
            else
            {
                // Fallback GELU approximation on host if no context is provided
                Z.toHost(device, queue);
                auto* zh = Z.hostData();
                auto total = Z.size();
                for(std::size_t i = 0; i < total; ++i)
                {
                    float x = zh[i];
                    // tanh approximation of GELU
                    float c = 0.044715f;
                    float y = 0.5f * x * (1.0f + std::tanh(0.7978845608f * (x + c * x * x * x)));
                    zh[i] = y;
                }
                Z.markHostModified();
                Z.ensureOnDevice(device, queue);
            }

            tensor::Tensor2D<float, Device> Out(device, {M, D}, "ffn_out");
            ops::matmul_2d(exec, device, queue, Z, const_cast<tensor::Tensor2D<float, Device>&>(W2), Out);
            return Out;
        }
    };

    template<typename Device>
    FeedForward2DLayer<Device> feedForward2d(tensor::Tensor2D<float, Device> W1, tensor::Tensor2D<float, Device> W2)
    {
        return FeedForward2DLayer<Device>{std::move(W1), std::move(W2)};
    }

    // BERT Encoder Block (Pre-LN):
    // LN1 -> SelfAttention -> Residual -> LN2 -> FFN -> Residual
    template<typename Device>
    struct BertEncoderBlock2D
    {
        using input_type = tensor::Tensor2D<float, Device>;
        using output_type = tensor::Tensor2D<float, Device>;

        // LayerNorm params
        tensor::Tensor1D<float, Device> ln1_gamma; // [D]
        tensor::Tensor1D<float, Device> ln1_beta; // [D]
        tensor::Tensor1D<float, Device> ln2_gamma; // [D]
        tensor::Tensor1D<float, Device> ln2_beta; // [D]
        float eps{1e-5f};

        // Sub-layers
        SelfAttention2DLayer<Device> attn;
        FeedForward2DLayer<Device> ffn;

        // Optional clean context for provider delegation
        mutable void* context{nullptr};

        template<typename Exec, typename Queue>
        output_type operator()(Exec const& exec, Device& device, Queue& queue, input_type& in) const
        {
            auto M = in.shape()[0];
            auto D = in.shape()[1];
            if(M == 0 || D == 0)
                return tensor::Tensor2D<float, Device>(device, {M, D}, "bert_block_out");

            // LN1
            tensor::Tensor2D<float, Device> Xln(device, {M, D}, "X_ln1");
            ops::layer_norm_2d<float>(
                exec,
                device,
                queue,
                in,
                const_cast<tensor::Tensor1D<float, Device>&>(ln1_gamma),
                const_cast<tensor::Tensor1D<float, Device>&>(ln1_beta),
                eps,
                Xln);

            // Self-Attention (+optional Wo inside)
            auto Aout = attn(exec, device, queue, Xln);

            // Residual 1: Xin + Aout (use shared safe helper)
            if(std::getenv("ALPAKA_BERT_DEBUG"))
                std::cerr << "[bert-layers-debug] entering residual 1" << std::endl;
            tensor::Tensor2D<float, Device> Xres1(device, {M, D}, "X_res1");
            ops::residual_add_2d<float>(exec, device, queue, in, Aout, Xres1);
            if(std::getenv("ALPAKA_BERT_DEBUG"))
                std::cerr << "[bert-layers-debug] after residual 1" << std::endl;

            // LN2
            tensor::Tensor2D<float, Device> Y(device, {M, D}, "X_ln2");
            ops::layer_norm_2d<float>(
                exec,
                device,
                queue,
                Xres1,
                const_cast<tensor::Tensor1D<float, Device>&>(ln2_gamma),
                const_cast<tensor::Tensor1D<float, Device>&>(ln2_beta),
                eps,
                Y);

            // FFN
            if(context)
                ffn.context = context; // propagate context for GELU
            auto U = ffn(exec, device, queue, Y);

            // Residual 2: Xres1 + U
            if(std::getenv("ALPAKA_BERT_DEBUG"))
                std::cerr << "[bert-layers-debug] entering residual 2" << std::endl;
            tensor::Tensor2D<float, Device> Out(device, {M, D}, "bert_block_out");
            ops::residual_add_2d<float>(exec, device, queue, Xres1, U, Out);
            if(std::getenv("ALPAKA_BERT_DEBUG"))
                std::cerr << "[bert-layers-debug] after residual 2" << std::endl;

            return Out;
        }
    };

    template<typename Device>
    BertEncoderBlock2D<Device> bertEncoderBlock2d(
        tensor::Tensor1D<float, Device> ln1_gamma,
        tensor::Tensor1D<float, Device> ln1_beta,
        tensor::Tensor1D<float, Device> ln2_gamma,
        tensor::Tensor1D<float, Device> ln2_beta,
        SelfAttention2DLayer<Device> attn,
        FeedForward2DLayer<Device> ffn,
        float eps = 1e-5f)
    {
        return BertEncoderBlock2D<Device>{
            std::move(ln1_gamma),
            std::move(ln1_beta),
            std::move(ln2_gamma),
            std::move(ln2_beta),
            eps,
            std::move(attn),
            std::move(ffn)};
    }

} // namespace alpaka::tensor::ops::layers
