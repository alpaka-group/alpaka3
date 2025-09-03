/* Lightweight helper utilities for the new strongly typed tensor API.
 * Goal: reduce verbosity in user code while keeping compile-time safety.
 * SPDX-License-Identifier: MPL-2.0
 */
#pragma once

#include <alpaka/tensor/TensorCore.hpp>
#include <utility>
#include <string>
#include <type_traits>
#include <functional>

namespace alpaka::tensor::helpers {

// Context bundles a device & queue so high-level calls can accept a single arg.
// Template parameters are deduced via makeContext.
// Example:
//   auto ctx = makeContext(device, queue);
//   gemm(ctx, M,N,K, 1.f, A,B, 0.f, C);  // future wrapper

template<typename TDevice, typename TQueue>
struct Context {
    TDevice const& device;
    TQueue& queue;
};

template<typename TDevice, typename TQueue>
Context(TDevice const&, TQueue&) -> Context<TDevice, TQueue>;

template<typename TDevice, typename TQueue>
constexpr Context<TDevice,TQueue> makeContext(TDevice const& dev, TQueue& q){
    return {dev,q};
}

// Convenience type aliases for common float tensors.
// Usage: helpers::F32T4<decltype(device)> A(device,{n,c,h,w});

template<typename D> using F32T1 = Tensor1D<float,D>;
template<typename D> using F32T2 = Tensor2D<float,D>;
template<typename D> using F32T3 = Tensor3D<float,D>;
template<typename D> using F32T4 = Tensor4D<float,D>;

template<typename D> using F64T1 = Tensor1D<double,D>;

// Factory helpers remove the need to spell the Device in the template argument.
// Example: auto A = makeTensor2D<float>(device, M, N, "A");

template<typename T, typename Device>
auto makeTensor1D(Device const& dev, std::size_t d0, std::string name="") -> Tensor<T,1,Device> {
    return Tensor<T,1,Device>(dev, {d0}, std::move(name));
}

template<typename T, typename Device>
auto makeTensor2D(Device const& dev, std::size_t d0, std::size_t d1, std::string name="") -> Tensor<T,2,Device> {
    return Tensor<T,2,Device>(dev, {d0,d1}, std::move(name));
}

template<typename T, typename Device>
auto makeTensor3D(Device const& dev, std::size_t d0, std::size_t d1, std::size_t d2, std::string name="") -> Tensor<T,3,Device> {
    return Tensor<T,3,Device>(dev, {d0,d1,d2}, std::move(name));
}

template<typename T, typename Device>
auto makeTensor4D(Device const& dev, std::size_t d0, std::size_t d1, std::size_t d2, std::size_t d3, std::string name="") -> Tensor<T,4,Device> {
    return Tensor<T,4,Device>(dev, {d0,d1,d2,d3}, std::move(name));
}

// Generic N-D factory from std::array shape (Rank inferred from array size via template parameter R).
// Example: auto X = makeTensor<float>(device, std::array{M,K});

template<typename T, typename Device, std::size_t R>
auto makeTensor(Device const& dev, std::array<std::size_t,R> shape, std::string name="") -> Tensor<T,R,Device> {
    return Tensor<T,R,Device>(dev, shape, std::move(name));
}

// Host fill helpers that also mark the tensor host-modified.
// Overload 1: constant value

template<typename T, std::size_t Rank, typename Device>
void fillHost(Tensor<T,Rank,Device>& t, T value){
    auto* p = t.hostData();
    for(std::size_t i=0;i<t.size();++i) p[i]=value;
    t.markHostModified();
}

// Overload 2: generator functor/lambda taking (index) -> T

template<typename T, std::size_t Rank, typename Device, typename F>
void fillHost(Tensor<T,Rank,Device>& t, F&& gen){
    auto* p = t.hostData();
    for(std::size_t i=0;i<t.size();++i) p[i]=static_cast<T>(gen(i));
    t.markHostModified();
}

// Overload 3: generator with multi-dimensional indices (passes array of indices)

namespace detail {
    template<std::size_t Rank>
    struct IndexIterator {
        std::array<std::size_t,Rank> shape;
        std::array<std::size_t,Rank> idx{};
        bool first=true;
        bool next(){
            if(first){ first=false; return true; }
            for(std::size_t r=Rank; r-- > 0;){
                if(++idx[r] < shape[r]) return true;
                idx[r]=0;
            }
            return false; // exhausted
        }
    };
}

template<typename T, std::size_t Rank, typename Device, typename F>
void fillHostNd(Tensor<T,Rank,Device>& t, F&& gen){
    auto* p = t.hostData();
    detail::IndexIterator<Rank> it{t.shape()};
    std::size_t linear=0;
    while(it.next()){
        p[linear++] = static_cast<T>(gen(it.idx));
    }
    t.markHostModified();
}

// Convenience access wrappers (explicit naming)

template<typename T, std::size_t Rank, typename Device>
T* writableHostData(Tensor<T,Rank,Device>& t){ return t.hostData(); }

template<typename T, std::size_t Rank, typename Device>
T const* readableHostData(Tensor<T,Rank,Device> const& t){ return t.hostData(); }

// Future: context-aware wrappers for gemm/conv/relu could go here (thin forwarders).

} // namespace alpaka::tensor::helpers
