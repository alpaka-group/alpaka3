#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/tensor/core/TensorCore.hpp>
#include <alpaka/tensor/layers/mlp/ReLULayer.hpp>
#include <alpaka/tensor/layers/vision/Conv2DLayer.hpp>
#include <alpaka/tensor/ops/elementwise/ElementwiseGeneric.hpp>
#include <alpaka/tensor/ops/normalization/BatchNorm.hpp>
#include <alpaka/tensor/providers/CleanTensorOpContext.hpp>

#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <stdexcept>
#include <utility>

namespace alpaka::tensor::ops
{
    using alpaka::tensor::ops::layers::ReLULayer;

    template<typename Device>
    struct BatchNorm2DLayerStruct
    {
        tensor::Tensor1D<float, Device> runningMean;
        tensor::Tensor1D<float, Device> runningVar;
        tensor::Tensor1D<float, Device> gamma;
        tensor::Tensor1D<float, Device> beta;
        float eps{1e-5f};
        mutable void* context{nullptr};

        BatchNorm2DLayerStruct() = default;

        BatchNorm2DLayerStruct(
            tensor::Tensor1D<float, Device> rm,
            tensor::Tensor1D<float, Device> rv,
            tensor::Tensor1D<float, Device> g,
            tensor::Tensor1D<float, Device> b,
            float e = 1e-5f)
            : runningMean(std::move(rm))
            , runningVar(std::move(rv))
            , gamma(std::move(g))
            , beta(std::move(b))
            , eps(e)
        {
        }

        template<typename Exec, typename Queue>
        tensor::Tensor4D<float, Device> operator()(
            Exec const& exec,
            Device& device,
            Queue& queue,
            tensor::Tensor4D<float, Device>& in)
        {
            auto s = in.shape();
            auto N = s[0];
            auto C = s[1];
            auto H = s[2];
            auto W = s[3];
            if(runningMean.size() != C || runningVar.size() != C || gamma.size() != C || beta.size() != C)
            {
                throw std::runtime_error(
                    "BatchNorm fallback: parameter size mismatch (mean/var/gamma/beta vs channels)");
            }
            if(in.size() != N * C * H * W)
            {
                throw std::runtime_error("BatchNorm fallback: input tensor size mismatch");
            }

            if(context)
            {
                try
                {
                    auto* cleanContext = static_cast<tensor::CleanTensorOpContext<Exec, Device, Queue>*>(context);
                    return cleanContext->batchnorm(in, runningMean, runningVar, gamma, beta, eps);
                }
                catch(std::runtime_error const&)
                {
                }
            }

            tensor::Tensor4D<float, Device> out(device, s, "bn_out");
            ops::batch_norm_inference<float>(exec, device, queue, in, gamma, beta, runningMean, runningVar, eps, out);
            return out;
        }
    };

    template<typename Device>
    struct BasicBlockLayerStruct
    {
        tensor::Tensor4D<float, Device> w1;
        tensor::Tensor4D<float, Device> w2;
        tensor::Tensor1D<float, Device> bn1Mean, bn1Var, bn1Gamma, bn1Beta;
        tensor::Tensor1D<float, Device> bn2Mean, bn2Var, bn2Gamma, bn2Beta;
        bool hasProj{false};
        tensor::Tensor4D<float, Device> wProj;
        tensor::Tensor1D<float, Device> projMean, projVar, projGamma, projBeta;
        Conv2DParams conv1Params{};
        Conv2DParams conv2Params{};
        Conv2DParams projParams{};
        float bnEps{1e-5f};
        bool initialized{false};

        template<typename Tensor4D>
        static void initKaimingFanInUniformHost(Tensor4D& t, std::size_t fanIn)
        {
            float limit = std::sqrt(6.f / fanIn);
            unsigned seed = 1337u;
            auto lcg = [&]()
            {
                seed = seed * 1'664'525u + 1'013'904'223u;
                return seed;
            };
            float* h = t.hostData();
            for(std::size_t i = 0; i < t.size(); ++i)
            {
                float r = static_cast<float>(lcg() & 0xFF'FFFF) / static_cast<float>(0xFF'FFFF);
                h[i] = -limit + 2.f * limit * r;
            }
            t.markHostModified();
        }

        template<typename Exec, typename Queue>
        void ensureInit(Exec const&, Device& dev, Queue&, std::size_t C_in, std::size_t C_out, bool downsample)
        {
            if(initialized)
                return;
            conv1Params.pad_h = conv1Params.pad_w = 1;
            conv2Params.pad_h = conv2Params.pad_w = 1;
            if(downsample)
                conv1Params.stride_h = conv1Params.stride_w = 2;
            w1 = tensor::Tensor4D<float, Device>(dev, {C_out, C_in, 3, 3}, "bb_w1");
            w2 = tensor::Tensor4D<float, Device>(dev, {C_out, C_out, 3, 3}, "bb_w2");
            initKaimingFanInUniformHost(w1, 3 * 3 * C_in);
            initKaimingFanInUniformHost(w2, 3 * 3 * C_out);
            bn1Mean = tensor::Tensor1D<float, Device>(dev, {C_out}, "bn1Mean");
            bn1Var = tensor::Tensor1D<float, Device>(dev, {C_out}, "bn1Var");
            bn1Gamma = tensor::Tensor1D<float, Device>(dev, {C_out}, "bn1Gamma");
            bn1Beta = tensor::Tensor1D<float, Device>(dev, {C_out}, "bn1Beta");
            bn2Mean = tensor::Tensor1D<float, Device>(dev, {C_out}, "bn2Mean");
            bn2Var = tensor::Tensor1D<float, Device>(dev, {C_out}, "bn2Var");
            bn2Gamma = tensor::Tensor1D<float, Device>(dev, {C_out}, "bn2Gamma");
            bn2Beta = tensor::Tensor1D<float, Device>(dev, {C_out}, "bn2Beta");
            for(auto* t : {&bn1Mean, &bn1Var, &bn1Gamma, &bn1Beta, &bn2Mean, &bn2Var, &bn2Gamma, &bn2Beta})
            {
                float* h = t->hostData();
                bool isVar = (t == &bn1Var) || (t == &bn2Var);
                bool isGamma = (t == &bn1Gamma) || (t == &bn2Gamma);
                for(std::size_t i = 0; i < t->size(); ++i)
                    h[i] = isVar ? 1.f : (isGamma ? 1.f : 0.f);
                t->markHostModified();
            }
            if(C_in != C_out || downsample)
            {
                hasProj = true;
                projParams.stride_h = projParams.stride_w = downsample ? 2 : 1;
                wProj = tensor::Tensor4D<float, Device>(dev, {C_out, C_in, 1, 1}, "bb_wProj");
                initKaimingFanInUniformHost(wProj, C_in);
                projMean = tensor::Tensor1D<float, Device>(dev, {C_out}, "projMean");
                projVar = tensor::Tensor1D<float, Device>(dev, {C_out}, "projVar");
                projGamma = tensor::Tensor1D<float, Device>(dev, {C_out}, "projGamma");
                projBeta = tensor::Tensor1D<float, Device>(dev, {C_out}, "projBeta");
                for(auto* t : {&projMean, &projVar, &projGamma, &projBeta})
                {
                    float* h = t->hostData();
                    bool isVar = (t == &projVar);
                    bool isGamma = (t == &projGamma);
                    for(std::size_t i = 0; i < t->size(); ++i)
                        h[i] = isVar ? 1.f : (isGamma ? 1.f : 0.f);
                    t->markHostModified();
                }
            }
            initialized = true;
        }

        template<typename Exec, typename Queue>
        tensor::Tensor4D<float, Device> operator()(
            Exec const& exec,
            Device& dev,
            Queue& q,
            tensor::Tensor4D<float, Device>& in,
            std::size_t C_in,
            std::size_t C_out,
            bool downsample)
        {
            auto dbg = std::getenv("ALPAKA_RESNET_DEBUG") != nullptr;
            auto printShape = [&](char const* name, auto const& t)
            {
                if(!dbg)
                    return;
                auto s = t.shape();
                std::cerr << name << " shape=[" << s[0] << "," << s[1] << "," << s[2] << "," << s[3] << "]\n";
            };
            ensureInit(exec, dev, q, C_in, C_out, downsample);
            Conv2DLayerStruct<Device> c1{w1, std::nullopt, conv1Params};
            printShape("[BB] in", in);
            auto x = c1(exec, dev, q, in);
            printShape("[BB] after conv1", x);
            BatchNorm2DLayerStruct<Device> bn1{bn1Mean, bn1Var, bn1Gamma, bn1Beta, bnEps};
            x = bn1(exec, dev, q, x);
            printShape("[BB] after bn1", x);
            ReLULayer<Device> relu{};
            relu.inPlace = true;
            relu(exec, dev, q, x);
            printShape("[BB] after relu1", x);
            Conv2DLayerStruct<Device> c2{w2, std::nullopt, conv2Params};
            x = c2(exec, dev, q, x);
            printShape("[BB] after conv2", x);
            BatchNorm2DLayerStruct<Device> bn2{bn2Mean, bn2Var, bn2Gamma, bn2Beta, bnEps};
            x = bn2(exec, dev, q, x);
            printShape("[BB] after bn2", x);
            tensor::Tensor4D<float, Device> identity = in;
            if(hasProj)
            {
                Conv2DLayerStruct<Device> cp{wProj, std::nullopt, projParams};
                identity = cp(exec, dev, q, in);
                BatchNorm2DLayerStruct<Device> bnp{projMean, projVar, projGamma, projBeta, bnEps};
                identity = bnp(exec, dev, q, identity);
            }
            printShape("[BB] identity", identity);
            auto y = tensor::ops::add<float, 4>(exec, dev, q, x, identity);
            if(dbg)
                std::cerr << "[BB] after add()" << "\n";
            ::alpaka::onHost::wait(q);
            x = std::move(y);
            relu(exec, dev, q, x);
            if(dbg)
                printShape("[BB] after relu2", x);
            return x;
        }
    };

    template<typename Device>
    struct AddLayerStruct
    {
        template<typename Exec, typename Queue>
        tensor::Tensor4D<float, Device> operator()(
            Exec const& exec,
            Device& device,
            Queue& queue,
            tensor::Tensor4D<float, Device>& input1,
            tensor::Tensor4D<float, Device>& input2) const
        {
            auto shape1 = input1.shape();
            auto shape2 = input2.shape();
            assert(shape1 == shape2 && "AddLayer: input tensors must have same shape");
            return ops::add<float, 4>(exec, device, queue, input1, input2);
        }
    };

    template<typename Device>
    struct ResidualBlockStruct
    {
        Conv2DLayerStruct<Device> conv1;
        ReLULayer<Device> relu1;
        Conv2DLayerStruct<Device> conv2;
        std::optional<Conv2DLayerStruct<Device>> projection;
        ReLULayer<Device> final_relu;

        template<typename Exec, typename Queue>
        tensor::Tensor4D<float, Device> operator()(
            Exec const& exec,
            Device& device,
            Queue& queue,
            tensor::Tensor4D<float, Device>& input) const
        {
            auto skip = input;
            auto x = conv1(exec, device, queue, input);
            x = relu1(exec, device, queue, x);
            x = conv2(exec, device, queue, x);

            if(projection)
            {
                skip = (*projection)(exec, device, queue, skip);
            }

            AddLayerStruct<Device> add_layer;
            auto output = add_layer(exec, device, queue, x, skip);
            return final_relu(exec, device, queue, output);
        }
    };

    namespace residualhelpers
    {
        struct ResidualDefaults
        {
            static constexpr std::size_t kernel_size = 3;
            static constexpr std::size_t stride = 1;
            static constexpr std::size_t padding = 1;
            static constexpr bool use_bias = false;
        };

        template<typename Device, typename Exec, typename Queue>
        ResidualBlockStruct<Device> createBasicBlock(
            Exec const& exec,
            Queue& queue,
            Device& device,
            std::size_t in_channels,
            std::size_t out_channels,
            std::size_t stride = 1)
        {
            ResidualBlockStruct<Device> block;

            Conv2DParams params1{};
            params1.stride_h = stride;
            params1.stride_w = stride;
            params1.pad_h = ResidualDefaults::padding;
            params1.pad_w = ResidualDefaults::padding;

            tensor::Tensor4D<float, Device> weights1(device, {out_channels, in_channels, 3, 3}, "resblock_conv1_w");
            auto* w1_data = weights1.hostData();
            for(std::size_t i = 0; i < weights1.size(); ++i)
            {
                w1_data[i] = 0.01f;
            }
            weights1.markHostModified();

            block.conv1 = Conv2DLayerStruct<Device>{std::move(weights1), std::nullopt, params1};
            block.relu1 = ReLULayer<Device>{true};

            Conv2DParams params2{};
            params2.stride_h = 1;
            params2.stride_w = 1;
            params2.pad_h = ResidualDefaults::padding;
            params2.pad_w = ResidualDefaults::padding;

            tensor::Tensor4D<float, Device> weights2(device, {out_channels, out_channels, 3, 3}, "resblock_conv2_w");
            auto* w2_data = weights2.hostData();
            for(std::size_t i = 0; i < weights2.size(); ++i)
            {
                w2_data[i] = 0.01f;
            }
            weights2.markHostModified();

            block.conv2 = Conv2DLayerStruct<Device>{std::move(weights2), std::nullopt, params2};

            if(in_channels != out_channels || stride != 1)
            {
                Conv2DParams proj_params{};
                proj_params.stride_h = stride;
                proj_params.stride_w = stride;
                proj_params.pad_h = 0;
                proj_params.pad_w = 0;

                tensor::Tensor4D<float, Device> proj_weights(
                    device,
                    {out_channels, in_channels, 1, 1},
                    "resblock_proj_w");
                auto* proj_data = proj_weights.hostData();
                for(std::size_t i = 0; i < proj_weights.size(); ++i)
                {
                    proj_data[i] = 0.01f;
                }
                proj_weights.markHostModified();

                block.projection = Conv2DLayerStruct<Device>{std::move(proj_weights), std::nullopt, proj_params};
            }

            block.final_relu = ReLULayer<Device>{true};

            return block;
        }

    } // namespace residualhelpers

} // namespace alpaka::tensor::ops
