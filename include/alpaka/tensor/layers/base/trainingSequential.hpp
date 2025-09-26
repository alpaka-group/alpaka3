#pragma once

#include <alpaka/tensor/ops/training/TrainingSequential.hpp>

namespace alpaka::tensor::ops
{

    template<typename Device, typename Exec, typename Queue, typename... Layers>
    using TrainingSequential [[deprecated(
        "TrainingSequential has been replaced by TrainingSequentialCT; include "
        "<alpaka/tensor/ops/training/TrainingSequential.hpp> and use the CT version")]]
    = TrainingSequentialCT<Device, Exec, Queue, Layers...>;

} // namespace alpaka::tensor::ops
