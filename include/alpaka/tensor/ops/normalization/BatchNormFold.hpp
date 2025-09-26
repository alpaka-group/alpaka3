/* BatchNormFold.hpp - BatchNorm inference parameter folding into Conv2D
 * Moved from ops/batchnorm to ops/normalization (normalization ops grouping)
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <alpaka/tensor/core/TensorCore.hpp>
#include <alpaka/tensor/layers/base/Layer.hpp>

#include <cassert>
#include <cmath>
#include <iostream>
#include <optional>
#include <utility>

namespace alpaka::tensor::ops
{
    template<typename Exec, typename Device, typename Queue>
    bool fold_batch_norm_into_conv(
        Exec const& exec,
        Device& device,
        Queue& queue,
        tensor::Tensor4D<float, Device>& weights,
        std::optional<tensor::Tensor1D<float, Device>>& bias,
        tensor::Tensor1D<float, Device>& gamma,
        tensor::Tensor1D<float, Device>& beta,
        tensor::Tensor1D<float, Device>& running_mean,
        tensor::Tensor1D<float, Device>& running_var,
        float epsilon,
        bool& alreadyFolded)
    {
        if(alreadyFolded)
        {
            std::cout << "BN fold: skipped (already folded)" << std::endl;
            return false;
        }
        (void) exec;
        auto wShape = weights.shape();
        std::size_t C_out = wShape[0];
        assert(
            gamma.shape()[0] == C_out && beta.shape()[0] == C_out && running_mean.shape()[0] == C_out
            && running_var.shape()[0] == C_out && "BN fold: parameter size mismatch");
        weights.toHost(device, queue);
        gamma.toHost(device, queue);
        beta.toHost(device, queue);
        running_mean.toHost(device, queue);
        running_var.toHost(device, queue);
        if(!bias)
            bias.emplace(device, std::array<std::size_t, 1>{C_out}, "conv_bias");
        bias->toHost(device, queue);
        auto* wHost = weights.hostData();
        auto* bHost = bias->hostData();
        auto* gHost = gamma.hostData();
        auto* beHost = beta.hostData();
        auto* mHost = running_mean.hostData();
        auto* vHost = running_var.hostData();
        std::size_t C_in = wShape[1];
        std::size_t K_h = wShape[2];
        std::size_t K_w = wShape[3];
        std::size_t perChannel = C_in * K_h * K_w;
        for(std::size_t c = 0; c < C_out; ++c)
        {
            float scale = gHost[c] / std::sqrt(vHost[c] + epsilon);
            float* wChan = wHost + c * perChannel;
            for(std::size_t i = 0; i < perChannel; ++i)
                wChan[i] *= scale;
            float bOrig = bHost[c];
            bHost[c] = beHost[c] + scale * (bOrig - mHost[c]);
        }
        weights.markHostModified();
        bias->markHostModified();
        weights.ensureOnDevice(device, queue);
        bias->ensureOnDevice(device, queue);
        alreadyFolded = true;
        std::cout << "BN fold: applied (C_out=" << C_out << ")" << std::endl;
        return true;
    }

    template<typename Exec, typename Device, typename Queue>
    std::pair<tensor::Tensor4D<float, Device>, tensor::Tensor1D<float, Device>> clone_and_fold_batch_norm(
        Exec const& exec,
        Device& device,
        Queue& queue,
        tensor::Tensor4D<float, Device> const& weightsSrc,
        std::optional<tensor::Tensor1D<float, Device>> const& biasSrc,
        tensor::Tensor1D<float, Device> const& gamma,
        tensor::Tensor1D<float, Device> const& beta,
        tensor::Tensor1D<float, Device> const& running_mean,
        tensor::Tensor1D<float, Device> const& running_var,
        float epsilon)
    {
        auto weights = weightsSrc;
        tensor::Tensor1D<float, Device> bias(device, {weights.shape()[0]}, "folded_bias");
        if(biasSrc)
        {
            auto* dst = bias.hostData();
            auto* src = biasSrc->hostData();
            for(std::size_t i = 0; i < weights.shape()[0]; ++i)
                dst[i] = src[i];
            bias.markHostModified();
        }
        else
        {
            bias.fill(0.f);
        }
        bool folded = false;
        auto gammaCopy = gamma;
        auto betaCopy = beta;
        auto meanCopy = running_mean;
        auto varCopy = running_var;
        std::optional<tensor::Tensor1D<float, Device>> biasOpt = bias;
        fold_batch_norm_into_conv(
            exec,
            device,
            queue,
            weights,
            biasOpt,
            gammaCopy,
            betaCopy,
            meanCopy,
            varCopy,
            epsilon,
            folded);
        return {std::move(weights), std::move(*biasOpt)};
    }

    template<typename Exec, typename Device, typename Queue>
    Conv2DLayerStruct<Device> make_conv2d_layer_with_bn_folding(
        Exec const& exec,
        Device& device,
        Queue& queue,
        tensor::Tensor4D<float, Device> const& weightsSrc,
        std::optional<tensor::Tensor1D<float, Device>> const& biasSrc,
        tensor::Tensor1D<float, Device> const& gamma,
        tensor::Tensor1D<float, Device> const& beta,
        tensor::Tensor1D<float, Device> const& running_mean,
        tensor::Tensor1D<float, Device> const& running_var,
        float epsilon,
        Conv2DParams params = {})
    {
        auto [wFold, bFold] = clone_and_fold_batch_norm(
            exec,
            device,
            queue,
            weightsSrc,
            biasSrc,
            gamma,
            beta,
            running_mean,
            running_var,
            epsilon);
        Conv2DLayerStruct<Device> layer{std::move(wFold), std::optional{std::move(bFold)}, params};
        return layer;
    }

} // namespace alpaka::tensor::ops
