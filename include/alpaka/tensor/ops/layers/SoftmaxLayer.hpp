#pragma once
#include <alpaka/alpaka.hpp>
#include <alpaka/tensor/TensorCore.hpp>
#include <alpaka/tensor/ops/InferenceOps.hpp>
#include <alpaka/tensor/ops/concepts/LayerConcepts.hpp>

namespace alpaka::tensor::ops::layers
{

    template<typename Device>
    struct SoftmaxLayer
    {
        using input_type = tensor::Tensor1D<float, Device>;
        using output_type = tensor::Tensor2D<float, Device>;

        std::size_t batch{1};
        std::size_t features{1};

        template<typename Exec, typename Queue>
        tensor::Tensor2D<float, Device> operator()(
            Exec const& exec,
            Device& device,
            Queue& queue,
            tensor::Tensor1D<float, Device>& in) const
        {
            auto logits2D = copy_flat_to_2d<float>(exec, device, queue, in, batch, features);
            tensor::Tensor2D<float, Device> probs(device, {batch, features}, "softmaxOut");
            softmax_2d<float>(exec, device, queue, logits2D, probs);
            // Ensure the softmax kernel that reads logits2D has finished before
            // logits2D goes out of scope and frees its buffers. The queue is
            // non-blocking on host, so we must wait here to prevent UAF.
            alpaka::onHost::wait(queue);
            return probs;
        }
    };

    // Factory function with CamelCase naming
    template<typename Device>
    SoftmaxLayer<Device> softmax(std::size_t batch, std::size_t features)
    {
        return SoftmaxLayer<Device>{batch, features};
    }

} // namespace alpaka::tensor::ops::layers
