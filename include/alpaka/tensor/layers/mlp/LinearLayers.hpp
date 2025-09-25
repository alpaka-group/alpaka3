#pragma once
#include <alpaka/alpaka.hpp>
#include <alpaka/tensor/context/CleanTensorOpContext.hpp>
#include <alpaka/tensor/core/TensorCore.hpp>
#include <alpaka/tensor/kernels/ElementwiseKernels.hpp>
#include <alpaka/tensor/layers/base/LayerConcepts.hpp>
#include <alpaka/tensor/ops/linear/LinearOps.hpp>
#include <alpaka/tensor/ops/reshape/Reshape.hpp>

#include <array>
#include <cstdlib>
#include <optional>
#include <string>

namespace alpaka::tensor::ops::layers
{

    // Multi-rank layers
    template<typename Device>
    struct FlattenLayer
    {
        using input_type = tensor::Tensor4D<float, Device>;
        using output_type = tensor::Tensor1D<float, Device>;

        template<typename Exec, typename Queue>
        tensor::Tensor1D<float, Device> operator()(
            Exec const& exec,
            Device& device,
            Queue& queue,
            tensor::Tensor4D<float, Device>& in) const
        {
            return flatten_4d_to_2d<float>(exec, device, queue, in);
        }
    };

    template<typename Device>
    struct LinearLayer
    {
        using input_type = tensor::Tensor1D<float, Device>;
        using output_type = tensor::Tensor1D<float, Device>;

        std::size_t batch{1};
        std::size_t outFeatures{1};
        mutable std::optional<tensor::Tensor1D<float, Device>> weights;
        mutable std::optional<tensor::Tensor1D<float, Device>> bias;

        // Non-owning pointer to clean context for provider delegation
        mutable void* context{nullptr};

        template<typename Exec, typename Queue>
        tensor::Tensor1D<float, Device> operator()(
            Exec const& exec,
            Device& device,
            Queue& queue,
            tensor::Tensor1D<float, Device>& in) const
        {
            auto total = in.size();
            auto K = total / batch;
            if(!weights)
            {
                weights.emplace(device, std::array<std::size_t, 1>{K * outFeatures}, "linearW");
                auto* pw = weights->hostData();
                // He or Xavier style init depending on activation expectation: assume ReLU -> He
                float fanIn = static_cast<float>(K);
                float fanOut = static_cast<float>(outFeatures);
                // Default: He uniform
                float limit = std::sqrt(6.f / fanIn); // Xavier uniform would be sqrt(6/(fanIn+fanOut))
                char const* modeEnv = std::getenv("ALPAKA_LINEAR_INIT");
                if(modeEnv)
                {
                    std::string m(modeEnv);
                    if(m == "xavier" || m == "XAVIER")
                        limit = std::sqrt(6.f / (fanIn + fanOut));
                }
                // Simple LCG for deterministic reproducible init (no <random> dependency differences across compilers)
                unsigned seed = 1337u;
                auto lcg = [&]()
                {
                    seed = seed * 1'664'525u + 1'013'904'223u;
                    return seed;
                };
                for(std::size_t i = 0; i < weights->size(); ++i)
                {
                    float r = (float) (lcg() & 0xFF'FFFF) / float(0xFF'FFFF); // [0,1)
                    float val = -limit + 2.f * limit * r; // [-limit, limit]
                    pw[i] = val;
                }
                weights->markHostModified();
                if(!bias)
                {
                    bias.emplace(device, std::array<std::size_t, 1>{outFeatures}, "linearB");
                    auto* pb = bias->hostData();
                    for(std::size_t j = 0; j < outFeatures; ++j)
                        pb[j] = 0.f; // zero bias
                    bias->markHostModified();
                }
            }
            auto& W = *weights;
            tensor::Tensor1D<float, Device> out(device, {batch * outFeatures}, "linearOut");
            auto* bptr = bias ? &*bias : nullptr;

            if(context)
            {
                // Use GEMM provider delegation through clean context
                auto* cleanContext = static_cast<tensor::CleanTensorOpContext<Exec, Device, Queue>*>(context);

                // Try to use provider delegation first
                try
                {
                    // Linear operation is essentially a GEMM: out = in * W + b
                    // For linear: A=[batch, K], W=[K, outFeatures], out=[batch, outFeatures]
                    // GEMM: C = alpha * A * B + beta * C
                    // We want: out = in * W + b, so alpha=1.0, beta=0.0 (overwrites output)

                    // Perform GEMM operation through provider context
                    cleanContext->gemm(
                        batch, // M: rows of A and C
                        outFeatures, // N: columns of B and C
                        K, // K: columns of A, rows of B
                        1.0f, // alpha
                        const_cast<tensor::Tensor1D<float, Device>&>(in), // A
                        const_cast<tensor::Tensor1D<float, Device>&>(W), // B
                        0.0f, // beta (overwrite output, no need to zero first)
                        out // C (output)
                    );

                    // Add bias if present using the same approach as existing linear function
                    if(bias)
                    {
                        auto& b = const_cast<tensor::Tensor1D<float, Device>&>(*bias);
                        // Broadcast add bias across rows (batch rows)
                        b.ensureOnDevice(device, queue);
                        out.ensureOnDevice(device, queue);
                        std::size_t total = batch * outFeatures;
                        auto frame = ops::detail::makeFrame<Exec, Queue>(total);
                        queue.enqueue(
                            exec,
                            frame,
                            ops::kernels::LinearBiasKernel{},
                            out.deviceBuffer(device, queue).data(),
                            b.deviceBuffer(device, queue).data(),
                            batch,
                            outFeatures,
                            total);
                        out.markDeviceModified(device, queue);
                    }
                }
                catch(...)
                {
                    // Fallback to direct linear implementation if provider fails
                    linear(
                        exec,
                        device,
                        queue,
                        batch,
                        outFeatures,
                        K,
                        in,
                        const_cast<tensor::Tensor1D<float, Device>&>(W),
                        bptr ? const_cast<tensor::Tensor1D<float, Device>*>(bptr) : nullptr,
                        out);
                }
            }
            else
            {
                // Fallback to existing linear implementation
                linear(
                    exec,
                    device,
                    queue,
                    batch,
                    outFeatures,
                    K,
                    in,
                    const_cast<tensor::Tensor1D<float, Device>&>(W),
                    bptr ? const_cast<tensor::Tensor1D<float, Device>*>(bptr) : nullptr,
                    out);
            }
            return out;
        }
    };

    template<typename Device>
    struct LinearReLULayer
    {
        using input_type = tensor::Tensor1D<float, Device>;
        using output_type = tensor::Tensor1D<float, Device>;

        std::size_t batch{1};
        std::size_t outFeatures{1};
        mutable std::optional<tensor::Tensor1D<float, Device>> weights;
        mutable std::optional<tensor::Tensor1D<float, Device>> bias;

        template<typename Exec, typename Queue>
        tensor::Tensor1D<float, Device> operator()(
            Exec const& exec,
            Device& device,
            Queue& queue,
            tensor::Tensor1D<float, Device>& in) const
        {
            auto total = in.size();
            auto K = total / batch;
            if(!weights)
            {
                weights.emplace(device, std::array<std::size_t, 1>{K * outFeatures}, "linearW");
                auto* pw = weights->hostData();
                // He initialization (optimal for ReLU)
                float fanIn = static_cast<float>(K);
                float limit = std::sqrt(6.f / fanIn);
                char const* modeEnv = std::getenv("ALPAKA_LINEAR_INIT");
                if(modeEnv)
                {
                    std::string m(modeEnv);
                    if(m == "xavier" || m == "XAVIER")
                    {
                        float fanOut = static_cast<float>(outFeatures);
                        limit = std::sqrt(6.f / (fanIn + fanOut));
                    }
                }
                // Simple LCG for deterministic reproducible init
                unsigned seed = 1337u;
                auto lcg = [&]()
                {
                    seed = seed * 1'664'525u + 1'013'904'223u;
                    return seed;
                };
                for(std::size_t i = 0; i < weights->size(); ++i)
                {
                    float r = (float) (lcg() & 0xFF'FFFF) / float(0xFF'FFFF); // [0,1)
                    float val = -limit + 2.f * limit * r; // [-limit, limit]
                    pw[i] = val;
                }
                weights->markHostModified();
                if(!bias)
                {
                    bias.emplace(device, std::array<std::size_t, 1>{outFeatures}, "linearB");
                    auto* pb = bias->hostData();
                    for(std::size_t j = 0; j < outFeatures; ++j)
                        pb[j] = 0.f; // zero bias
                    bias->markHostModified();
                }
            }
            auto& W = *weights;
            tensor::Tensor1D<float, Device> out(device, {batch * outFeatures}, "linearReluOut");
            auto* bptr = bias ? &*bias : nullptr;
            // Use fused linear + ReLU operation
            linear_relu(
                exec,
                device,
                queue,
                batch,
                outFeatures,
                K,
                in,
                const_cast<tensor::Tensor1D<float, Device>&>(W),
                bptr ? const_cast<tensor::Tensor1D<float, Device>*>(bptr) : nullptr,
                out);
            return out;
        }
    };

    // Factory functions with CamelCase naming
    template<typename Device>
    FlattenLayer<Device> flatten()
    {
        return FlattenLayer<Device>{};
    }

    template<typename Device>
    LinearLayer<Device> linear(std::size_t batch, std::size_t outFeatures)
    {
        return LinearLayer<Device>{batch, outFeatures};
    }

    template<typename Device>
    LinearReLULayer<Device> linearReLu(std::size_t batch, std::size_t outFeatures)
    {
        return LinearReLULayer<Device>{batch, outFeatures};
    }

} // namespace alpaka::tensor::ops::layers
