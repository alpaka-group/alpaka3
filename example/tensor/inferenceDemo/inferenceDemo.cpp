/* Inference Demo Example
 * Pipeline: Conv -> ReLU -> MaxPool -> Flatten -> Linear -> Softmax
 * Validates: probabilities sum ~1 per sample
 */
#include <alpaka/alpaka.hpp>
#include <alpaka/onHost/executeForEach.hpp>
#include <alpaka/onHost/example/executors.hpp>
#include <iostream>
#include <random>

// Avoid global using-directives to prevent ambiguous lookups after rebase changes.
namespace tt = alpaka::tensor;
namespace ops = alpaka::tensor::ops;

template<typename Tag>
int runInference(Tag const& tag){
    auto deviceSpec = tag[alpaka::object::deviceSpec];
    auto exec = tag[alpaka::object::exec];
    auto sel = alpaka::onHost::makeDeviceSelector(deviceSpec);
    alpaka::onHost::Device device = sel.makeDevice(0);
    alpaka::onHost::Queue queue = device.makeQueue();

    using Device = decltype(device);
    using Exec = decltype(exec);
    using Tensor4D = tt::Tensor4D<float,Device>;

    std::cout << "=== Inference Demo ===\nDevice: " << deviceSpec.getApi().getName()
                  << "\nExecutor: " << alpaka::onHost::demangledName(exec) << std::endl;

    // Dimensions
    std::size_t N=1, C_in=3, H=32, W=32;
    std::size_t C_out1=8, K=3; // conv1
    std::size_t FC_OUT=10;     // linear classes

    // Input
    Tensor4D input(device, {N,C_in,H,W}, "input");
    {
        std::mt19937 rng(42); std::uniform_real_distribution<float> dist(-1.f,1.f);
        auto *p = input.hostData(); for(std::size_t i=0;i<input.size();++i) p[i]=dist(rng); input.markHostModified();
    }

    // Conv weights
    tt::Tensor4D<float,Device> conv1W(device, {C_out1,C_in,K,K}, "conv1W");
    { auto *p = conv1W.hostData(); for(std::size_t i=0;i<conv1W.size();++i) p[i]=0.05f; conv1W.markHostModified(); }
    std::optional< tt::Tensor1D<float,Device> > conv1Bias;

    // Build multi-rank pipeline
    ops::MultiSequential<Device> pipe;
    pipe.addConv2D(exec, queue, ops::Conv2DLayerStruct<Device>{ std::move(conv1W), conv1Bias, ops::Conv2DParams{1,1,1,1,1,1} });
    pipe.addReLU(exec, queue, ops::ReLULayerStruct<Device>{ true });
    pipe.addMaxPool(exec, queue, ops::MaxPool2DLayerStruct<Device>{ ops::Pool2DParams{2,2,2,2,0,0} });
    pipe.addFlatten(exec, queue, ops::FlattenLayerStruct<Device>{});
    pipe.addLinear(exec, queue, ops::LinearLayerStruct<Device>{ N, FC_OUT, std::nullopt, std::nullopt });
    pipe.addSoftmax(exec, queue, ops::SoftmaxLayerStruct<Device>{ N, FC_OUT });

    // Run pipeline
    typename ops::MultiSequential<Device>::Any inputAny = std::move(input);
    auto resultAny = pipe.forward(exec, device, queue, std::move(inputAny));
    auto* probsPtr = std::get_if<tt::Tensor2D<float,Device>>(&resultAny);
    assert(probsPtr && "Final output should be 2D probs");
    probsPtr->toHost(device, queue);
    auto *ph = probsPtr->hostData();
    for(std::size_t m=0;m<N;++m){
        double sum=0; for(std::size_t j=0;j<FC_OUT;++j) sum += ph[m*FC_OUT + j];
        std::cout << "Sample " << m << " prob sum=" << sum << std::endl;
        if(std::abs(sum-1.0) > 1e-3) return 1;
    }
    return 0;
}

int main(){
    auto result = alpaka::onHost::executeForEachIfHasDevice(
        [](auto const& tag){ return runInference(tag); },
        alpaka::onHost::allBackends(alpaka::onHost::enabledApis, alpaka::onHost::example::enabledExecutors));
    return result;
}
