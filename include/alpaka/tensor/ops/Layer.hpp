// Clean minimal inference layer abstractions (Conv2D, ReLU, MaxPool, Flatten, Linear, Softmax)
// SPDX-License-Identifier: MPL-2.0

#pragma once
#include <alpaka/alpaka.hpp>
#include <alpaka/tensor/TensorCore.hpp>
#include <alpaka/tensor/ops/Conv2D.hpp>
#include <alpaka/tensor/ops/InferenceOps.hpp>
#include <alpaka/tensor/ops/Activations.hpp>
#include <cassert>
#include <optional>
#include <variant>
#include <vector>
#include <functional>
#include <array>

namespace alpaka::tensor::ops {

template<typename Device>
struct Conv2DLayerStruct { tensor::Tensor4D<float,Device> weights; std::optional<tensor::Tensor1D<float,Device>> bias; Conv2DParams params{}; template<typename Exec, typename Queue> tensor::Tensor4D<float,Device> operator()(Exec const& exec, Device& device, Queue& queue, tensor::Tensor4D<float,Device>& in) const { auto out=conv2d<float>(exec,device,queue,in,weights,params); if(bias){ tensor::Tensor4D<float,Device> tmp(device,out.shape(),"conv_bias_tmp"); auto &b = const_cast<tensor::Tensor1D<float,Device>&>(*bias); bias_add_4d(exec,device,queue,out,b,tmp); out=std::move(tmp);} return out; } };

template<typename Device>
struct ReLULayerStruct { bool inPlace{true}; template<typename Exec, typename Queue> tensor::Tensor4D<float,Device> operator()(Exec const& exec, Device& device, Queue& queue, tensor::Tensor4D<float,Device>& in) const { if(inPlace){ relu_inplace(exec,device,queue,in); return in;} tensor::Tensor4D<float,Device> out(device,in.shape(),"relu_out"); relu(exec,device,queue,in,out); return out; } };

// 1D variant for post-linear activations
template<typename Device>
struct ReLU1DLayerStruct { bool inPlace{true}; template<typename Exec, typename Queue> tensor::Tensor1D<float,Device> operator()(Exec const& exec, Device& device, Queue& queue, tensor::Tensor1D<float,Device>& in) const { if(inPlace){ relu_inplace(exec,device,queue,in); return in;} tensor::Tensor1D<float,Device> out(device,in.shape(),"relu1d_out"); relu(exec,device,queue,in,out); return out; } };

template<typename Device>
struct MaxPool2DLayerStruct { Pool2DParams params{}; template<typename Exec, typename Queue> tensor::Tensor4D<float,Device> operator()(Exec const& exec, Device& device, Queue& queue, tensor::Tensor4D<float,Device>& in) const { return max_pool2d<float>(exec,device,queue,in,params);} };

template<typename Device>
class Sequential { public: using Tensor4D=tensor::Tensor4D<float,Device>; using Fn=std::function<Tensor4D(void const*,void*,void*,Tensor4D&)>; template<typename Exec, typename Queue> void addConv2D(Exec const&, Queue&, Conv2DLayerStruct<Device> l){ nodes_.emplace_back([l=std::move(l)](void const* e, void* d, void* q, Tensor4D& in){ auto& exec=*static_cast<Exec const*>(e); auto& dev=*static_cast<Device*>(d); auto& qu=*static_cast<Queue*>(q); return l(exec,dev,qu,in); }); } template<typename Exec, typename Queue> void addReLU(Exec const&, Queue&, ReLULayerStruct<Device> l){ nodes_.emplace_back([l=std::move(l)](void const* e, void* d, void* q, Tensor4D& in){ auto& exec=*static_cast<Exec const*>(e); auto& dev=*static_cast<Device*>(d); auto& qu=*static_cast<Queue*>(q); return l(exec,dev,qu,in); }); } template<typename Exec, typename Queue> void addMaxPool(Exec const&, Queue&, MaxPool2DLayerStruct<Device> l){ nodes_.emplace_back([l=std::move(l)](void const* e, void* d, void* q, Tensor4D& in){ auto& exec=*static_cast<Exec const*>(e); auto& dev=*static_cast<Device*>(d); auto& qu=*static_cast<Queue*>(q); return l(exec,dev,qu,in); }); } template<typename Exec, typename Queue> Tensor4D forward(Exec const& exec, Device& dev, Queue& q, Tensor4D in){ for(auto& f:nodes_) in=f(&exec,&dev,&q,in); return in;} private: std::vector<Fn> nodes_{}; };

// Multi-rank layers
template<typename Device>
struct FlattenLayerStruct { template<typename Exec, typename Queue> tensor::Tensor1D<float,Device> operator()(Exec const& exec, Device& device, Queue& queue, tensor::Tensor4D<float,Device>& in) const { return flatten_4d_to_2d<float>(exec,device,queue,in);} };

template<typename Device>
struct LinearLayerStruct { std::size_t batch{1}; std::size_t outFeatures{1}; mutable std::optional<tensor::Tensor1D<float,Device>> weights; mutable std::optional<tensor::Tensor1D<float,Device>> bias; template<typename Exec, typename Queue> tensor::Tensor1D<float,Device> operator()(Exec const& exec, Device& device, Queue& queue, tensor::Tensor1D<float,Device>& in) const { auto total=in.size(); auto K=total/batch; if(!weights){ weights.emplace(device,std::array<std::size_t,1>{K*outFeatures},"linearW"); auto* pw=weights->hostData(); for(std::size_t i=0;i<weights->size();++i) pw[i]=0.01f; weights->markHostModified(); if(!bias){ bias.emplace(device,std::array<std::size_t,1>{outFeatures},"linearB"); auto* pb=bias->hostData(); for(std::size_t j=0;j<outFeatures;++j) pb[j]=0.f; bias->markHostModified(); } } auto& W=*weights; tensor::Tensor1D<float,Device> out(device,{batch*outFeatures},"linearOut"); auto* bptr=bias?&*bias:nullptr; linear(exec,device,queue,batch,outFeatures,K,in,const_cast<tensor::Tensor1D<float,Device>&>(W), bptr? const_cast<tensor::Tensor1D<float,Device>*>(bptr):nullptr, out); return out; } };

template<typename Device>
struct SoftmaxLayerStruct { std::size_t batch{1}; std::size_t features{1}; template<typename Exec, typename Queue> tensor::Tensor2D<float,Device> operator()(Exec const& exec, Device& device, Queue& queue, tensor::Tensor1D<float,Device>& in) const { auto logits2D=copy_flat_to_2d<float>(exec,device,queue,in,batch,features); tensor::Tensor2D<float,Device> probs(device,{batch,features},"softmaxOut"); softmax_2d<float>(exec,device,queue,logits2D,probs); return probs; } };

template<typename Device>
class MultiSequential { public: using T4=tensor::Tensor4D<float,Device>; using T1=tensor::Tensor1D<float,Device>; using T2=tensor::Tensor2D<float,Device>; using Any=std::variant<T4,T1,T2>; using Fn=std::function<Any(void const*,void*,void*,Any&&)>; template<typename Exec, typename Queue> void addConv2D(Exec const&, Queue&, Conv2DLayerStruct<Device> l){ nodes_.emplace_back([l=std::move(l)](void const* e, void* d, void* q, Any&& in){ auto& exec=*static_cast<Exec const*>(e); auto& dev=*static_cast<Device*>(d); auto& qu=*static_cast<Queue*>(q); auto* t4=std::get_if<T4>(&in); assert(t4); auto out=l(exec,dev,qu,*t4); return Any{std::move(out)}; }); } template<typename Exec, typename Queue> void addReLU(Exec const&, Queue&, ReLULayerStruct<Device> l){ nodes_.emplace_back([l=std::move(l)](void const* e, void* d, void* q, Any&& in){ auto& exec=*static_cast<Exec const*>(e); auto& dev=*static_cast<Device*>(d); auto& qu=*static_cast<Queue*>(q); if(auto* t4=std::get_if<T4>(&in)){ auto out=l(exec,dev,qu,*t4); return Any{std::move(out)}; } assert(false && "ReLU layer expects 4D tensor"); return in; }); } template<typename Exec, typename Queue> void addReLU1D(Exec const&, Queue&, ReLU1DLayerStruct<Device> l){ nodes_.emplace_back([l=std::move(l)](void const* e, void* d, void* q, Any&& in){ auto& exec=*static_cast<Exec const*>(e); auto& dev=*static_cast<Device*>(d); auto& qu=*static_cast<Queue*>(q); if(auto* t1=std::get_if<T1>(&in)){ auto out=l(exec,dev,qu,*t1); return Any{std::move(out)}; } assert(false && "ReLU1D layer expects 1D tensor"); return in; }); } template<typename Exec, typename Queue> void addMaxPool(Exec const&, Queue&, MaxPool2DLayerStruct<Device> l){ nodes_.emplace_back([l=std::move(l)](void const* e, void* d, void* q, Any&& in){ auto& exec=*static_cast<Exec const*>(e); auto& dev=*static_cast<Device*>(d); auto& qu=*static_cast<Queue*>(q); auto* t4=std::get_if<T4>(&in); assert(t4); auto out=l(exec,dev,qu,*t4); return Any{std::move(out)}; }); } template<typename Exec, typename Queue> void addFlatten(Exec const&, Queue&, FlattenLayerStruct<Device> l){ nodes_.emplace_back([l](void const* e, void* d, void* q, Any&& in){ auto& exec=*static_cast<Exec const*>(e); auto& dev=*static_cast<Device*>(d); auto& qu=*static_cast<Queue*>(q); auto* t4=std::get_if<T4>(&in); assert(t4); auto out=l(exec,dev,qu,*t4); return Any{std::move(out)}; }); } template<typename Exec, typename Queue> void addLinear(Exec const&, Queue&, LinearLayerStruct<Device> l){ nodes_.emplace_back([l=std::move(l)](void const* e, void* d, void* q, Any&& in) mutable { auto& exec=*static_cast<Exec const*>(e); auto& dev=*static_cast<Device*>(d); auto& qu=*static_cast<Queue*>(q); auto* t1=std::get_if<T1>(&in); assert(t1); auto out=l(exec,dev,qu,*t1); return Any{std::move(out)}; }); } template<typename Exec, typename Queue> void addSoftmax(Exec const&, Queue&, SoftmaxLayerStruct<Device> l){ nodes_.emplace_back([l](void const* e, void* d, void* q, Any&& in){ auto& exec=*static_cast<Exec const*>(e); auto& dev=*static_cast<Device*>(d); auto& qu=*static_cast<Queue*>(q); auto* t1=std::get_if<T1>(&in); assert(t1); auto out=l(exec,dev,qu,*t1); return Any{std::move(out)}; }); } template<typename Exec, typename Queue> Any forward(Exec const& exec, Device& dev, Queue& q, Any in){ for(auto& f:nodes_) in=f(&exec,&dev,&q,std::move(in)); return in;} private: std::vector<Fn> nodes_{}; };

} // namespace alpaka::tensor::ops
