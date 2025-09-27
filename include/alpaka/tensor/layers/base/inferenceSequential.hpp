#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/tensor/core/TensorCore.hpp>
#include <alpaka/tensor/layers/base/ResidualHelpers.hpp>
#include <alpaka/tensor/layers/mlp/LinearLayers.hpp>
#include <alpaka/tensor/layers/mlp/ReLULayer.hpp>
#include <alpaka/tensor/layers/mlp/SoftmaxLayer.hpp>
#include <alpaka/tensor/layers/normalization/BatchNormLayer.hpp>
#include <alpaka/tensor/layers/transformer/BertLayers.hpp>
#include <alpaka/tensor/layers/vision/Conv2DLayer.hpp>
#include <alpaka/tensor/layers/vision/PoolingLayers.hpp>
#include <alpaka/tensor/ops/transform/Transform.hpp>
#include <alpaka/tensor/providers/CleanTensorOpContext.hpp>

#include <cassert>
#include <chrono>
#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <utility>
#include <variant>
#include <vector>

namespace alpaka::tensor::layers
{

    template<typename Device>
    class Sequential
    {
    public:
        using Tensor4D = alpaka::tensor::Tensor4D<float, Device>;
        using Fn = std::function<Tensor4D(void const*, void*, void*, Tensor4D&)>;

        template<typename Exec, typename Queue>
        void addConv2D(Exec const&, Queue&, Conv2DLayerStruct<Device> l)
        {
            nodes_.emplace_back(
                [l = std::move(l)](void const* e, void* d, void* q, Tensor4D& in)
                {
                    auto& exec = *static_cast<Exec const*>(e);
                    auto& dev = *static_cast<Device*>(d);
                    auto& qu = *static_cast<Queue*>(q);
                    return l(exec, dev, qu, in);
                });
        }

        template<typename Exec, typename Queue>
        void addReLU(Exec const&, Queue&, ReLULayer<Device> l)
        {
            nodes_.emplace_back(
                [l = std::move(l)](void const* e, void* d, void* q, Tensor4D& in)
                {
                    auto& exec = *static_cast<Exec const*>(e);
                    auto& dev = *static_cast<Device*>(d);
                    auto& qu = *static_cast<Queue*>(q);
                    return l(exec, dev, qu, in);
                });
        }

        template<typename Exec, typename Queue>
        void addMaxPool(Exec const&, Queue&, MaxPool2DLayer<Device> l)
        {
            nodes_.emplace_back(
                [l = std::move(l)](void const* e, void* d, void* q, Tensor4D& in)
                {
                    auto& exec = *static_cast<Exec const*>(e);
                    auto& dev = *static_cast<Device*>(d);
                    auto& qu = *static_cast<Queue*>(q);
                    return l(exec, dev, qu, in);
                });
        }

        template<typename Exec, typename Queue>
        void addAvgPool(Exec const&, Queue&, AvgPool2DLayer<Device> l)
        {
            nodes_.emplace_back(
                [l = std::move(l)](void const* e, void* d, void* q, Tensor4D& in)
                {
                    auto& exec = *static_cast<Exec const*>(e);
                    auto& dev = *static_cast<Device*>(d);
                    auto& qu = *static_cast<Queue*>(q);
                    return l(exec, dev, qu, in);
                });
        }

        template<typename Exec, typename Queue>
        void addGlobalAvgPool(Exec const&, Queue&, GlobalAveragePool2DLayer<Device> l)
        {
            nodes_.emplace_back(
                [l = std::move(l)](void const* e, void* d, void* q, Tensor4D& in)
                {
                    auto& exec = *static_cast<Exec const*>(e);
                    auto& dev = *static_cast<Device*>(d);
                    auto& qu = *static_cast<Queue*>(q);
                    return l(exec, dev, qu, in);
                });
        }

        // Run forward pass through all layers
        template<typename Exec, typename Queue>
        Tensor4D forward(Exec const& exec, Device& dev, Queue& q, Tensor4D in)
        {
            // Sequential execution: each layer transforms input and passes to next
            for(auto& f : nodes_)
            {
                in = f(&exec, &dev, &q, in);
                // Ensure kernels launched within the layer complete before the
                // previous tensor instance goes out of scope and gets destroyed.
                // This prevents use-after-free when layers enqueue async work
                // consuming the input tensor while the pipeline replaces it.
                alpaka::onHost::wait(q);
            }
            return in;
        }

        std::vector<Fn> nodes_{};
    };

    /**
     * @brief Neural network pipeline builder using type-erased sequential execution.
     *
     * MultiSequential chains heterogeneous layer operations (Conv2D, Linear, ReLU, pooling, transformers, ...)
     * while seamlessly handling tensors of different ranks via std::variant.
     */

    template<typename Device, typename Exec, typename Queue>
    class MultiSequential
    {
    public:
        using T4 = tensor::Tensor4D<float, Device>;
        using T1 = tensor::Tensor1D<float, Device>;
        using T2 = tensor::Tensor2D<float, Device>;
        using Any = std::variant<T4, T1, T2>;
        using Fn = std::function<void(Any&)>; // mutate variant in-place
        using CleanTensorOpContext = tensor::CleanTensorOpContext<Exec, Device, Queue>;

        // Legacy constructor (backward compatibility) - no Context
        MultiSequential(Exec const& exec, Device& dev, Queue& queue)
            : exec_(exec)
            , dev_(dev)
            , queue_(queue)
            , cleanTensorOpContext_(nullptr)
            , cleanTensorOpContextPtr_(nullptr)
        {
        }

        // Constructor with CleanTensorOpContext (move semantics)
        MultiSequential(Exec const& exec, Device& dev, Queue& queue, CleanTensorOpContext&& cleanCtx)
            : exec_(exec)
            , dev_(dev)
            , queue_(queue)
            , cleanTensorOpContext_(std::make_unique<CleanTensorOpContext>(std::move(cleanCtx)))
            , cleanTensorOpContextPtr_(nullptr)
        {
        }

        // Constructor with CleanTensorOpContext (pointer/reference)
        MultiSequential(Exec const& exec, Device& dev, Queue& queue, CleanTensorOpContext* cleanCtx)
            : exec_(exec)
            , dev_(dev)
            , queue_(queue)
            , cleanTensorOpContext_(nullptr)
            , cleanTensorOpContextPtr_(cleanCtx)
        {
        }

        CleanTensorOpContext* getCleanTensorOpContext() const
        {
            return cleanTensorOpContext_ ? cleanTensorOpContext_.get() : cleanTensorOpContextPtr_;
        }

        bool hasCleanTensorOpContext() const
        {
            return cleanTensorOpContext_ || cleanTensorOpContextPtr_;
        }

        // Enable/disable generic host-side per-layer profiling (chrono-based, device agnostic)
        void enableProfiling(bool enable = true)
        {
            profilingEnabled_ = enable;
            if(enable && lastDurations_.size() != nodes_.size())
                lastDurations_.assign(nodes_.size(), 0.0);
        }

        bool isProfilingEnabled() const
        {
            return profilingEnabled_;
        }

        // Returns names of layers in sequential order
        std::vector<std::string> const& layerNames() const
        {
            return layerNames_;
        }

        // Returns last forward() per-layer durations in milliseconds (same order as layerNames())
        std::vector<double> const& lastDurations() const
        {
            return lastDurations_;
        }

    private:
        // Generic helper for layers that consume one tensor type and produce a (possibly different) tensor
        template<typename InTensor, typename F>
        void addImpl(F&& layerInvoker, char const* debugName)
        {
            nodes_.emplace_back(
                [this, inv = std::forward<F>(layerInvoker), debugName](Any& a) mutable
                {
                    auto* inPtr = std::get_if<InTensor>(&a);
                    assert(inPtr && "Layer input tensor rank/type mismatch");

                    // Use move semantics to avoid expensive tensor copying
                    // The layer invoker is responsible for proper memory management
                    auto out = inv(*inPtr);

                    // Synchronize at the layer boundary to ensure that any kernels
                    // launched inside the layer that read from the input have finished
                    // before the input tensor is destroyed when the variant is replaced.
                    // This guards against heap-use-after-free detected by ASan under
                    // asynchronous backends (e.g., CpuOmpBlocks, GPU queues).
                    alpaka::onHost::wait(queue_);

                    a = Any{std::move(out)};
#ifdef ALPAKA_TENSOR_PIPELINE_DEBUG
                    (void) debugName; // could log here if desired
#endif
                });
            layerNames_.emplace_back(debugName ? debugName : "layer");
            if(profilingEnabled_)
                lastDurations_.push_back(0.0);
        }

        // Helper for in-place layers (e.g. in-place ReLU) returning same tensor reference
        template<typename InTensor, typename F>
        void addImplInPlace(F&& layerInvoker, char const* debugName)
        {
            nodes_.emplace_back(
                [this, inv = std::forward<F>(layerInvoker), debugName](Any& a) mutable
                {
                    auto* inPtr = std::get_if<InTensor>(&a);
                    assert(inPtr && "Layer input tensor rank/type mismatch (in-place)");
                    inv(*inPtr); // modifies in place

                    // Synchronize at the layer boundary to ensure in-place updates
                    // are visible to subsequent layers and to keep execution ordering
                    // consistent across backends.
                    alpaka::onHost::wait(queue_);
#ifdef ALPAKA_TENSOR_PIPELINE_DEBUG
                    (void) debugName;
#endif
                });
            layerNames_.emplace_back(debugName ? debugName : "layer");
            if(profilingEnabled_)
                lastDurations_.push_back(0.0);
        }

    public:
        void addConv2D(Conv2DLayerStruct<Device> l)
        {
            // Inject clean context if available
            if(hasCleanTensorOpContext())
                l.context = getCleanTensorOpContext();
            addImpl<T4>([this, l = std::move(l)](T4& in) mutable { return l(exec_, dev_, queue_, in); }, "Conv2D");
        }

        void addReLU(ReLULayer<Device> l)
        {
            if(l.inPlace)
            {
                addImplInPlace<T4>([this, l](T4& in) mutable { l(exec_, dev_, queue_, in); }, "ReLU_inplace");
            }
            else
            {
                addImpl<T4>([this, l = std::move(l)](T4& in) mutable { return l(exec_, dev_, queue_, in); }, "ReLU");
            }
        }

        void addReLU1D(ReLU1DLayer<Device> l)
        {
            if(l.inPlace)
            {
                addImplInPlace<T1>([this, l](T1& in) mutable { l(exec_, dev_, queue_, in); }, "ReLU1D_inplace");
            }
            else
            {
                addImpl<T1>([this, l = std::move(l)](T1& in) mutable { return l(exec_, dev_, queue_, in); }, "ReLU1D");
            }
        }

        void addMaxPool(MaxPool2DLayer<Device> l)
        {
            addImpl<T4>([this, l = std::move(l)](T4& in) mutable { return l(exec_, dev_, queue_, in); }, "MaxPool");
        }

        void addAvgPool(AvgPool2DLayer<Device> l)
        {
            addImpl<T4>([this, l = std::move(l)](T4& in) mutable { return l(exec_, dev_, queue_, in); }, "AvgPool");
        }

        void addFlatten()
        {
            addImpl<T4>(
                [this](T4& in) mutable
                { return alpaka::tensor::ops::flatten_4d_to_2d<float>(exec_, dev_, queue_, in); },
                "Flatten");
        }

        void addLinear(LinearLayer<Device> l)
        {
            // Inject clean context if available
            if(hasCleanTensorOpContext())
                l.context = getCleanTensorOpContext();
            addImpl<T1>([this, l = std::move(l)](T1& in) mutable { return l(exec_, dev_, queue_, in); }, "Linear");
        }

        void addLinearReLU(LinearReLULayer<Device> l)
        {
            addImpl<T1>([this, l = std::move(l)](T1& in) mutable { return l(exec_, dev_, queue_, in); }, "LinearReLU");
        }

        void addSoftmax(SoftmaxLayer<Device> l)
        {
            addImpl<T1>([this, l = std::move(l)](T1& in) mutable { return l(exec_, dev_, queue_, in); }, "Softmax");
        }

        void addGlobalAvgPool(GlobalAveragePool2DLayer<Device> l)
        {
            addImpl<T4>(
                [this, l = std::move(l)](T4& in) mutable { return l(exec_, dev_, queue_, in); },
                "GlobalAvgPool");
        }

        void addBatchNorm(BatchNorm2DLayer<Device> l)
        {
            // Inject clean context if available
            if(hasCleanTensorOpContext())
                l.context = getCleanTensorOpContext();
            addImpl<T4>(
                [this, l = std::move(l)](T4& in) mutable { return l(exec_, dev_, queue_, in); },
                "BatchNorm2D");
        }

        void addBasicBlock(
            BasicBlockLayerStruct<Device> l,
            std::size_t inChannels,
            std::size_t outChannels,
            bool downsample)
        {
            // Wrap block execution capturing shape metadata; we infer C_in from tensor
            addImpl<T4>(
                [this, l = std::move(l), inChannels, outChannels, downsample](T4& in) mutable
                {
                    auto s = in.shape();
                    std::size_t C_in = s[1];
                    // Prefer provided inChannels if non-zero, else use tensor shape
                    std::size_t CinUse = inChannels ? inChannels : C_in;
                    auto out = l(exec_, dev_, queue_, in, CinUse, outChannels, downsample);
                    return out;
                },
                "BasicBlock");
        }

        // ---- New Generic Add Method with Auto-Deduction ----
        template<typename LayerType>
        void add(LayerType layer)
        {
            using namespace layers;

            // Auto-deduce input/output tensor types based on layer type
            if constexpr(
                std::is_same_v<LayerType, Conv2DLayer<Device>> || std::is_same_v<LayerType, ReLULayer<Device>>
                || std::is_same_v<LayerType, BatchNorm2DLayer<Device>>
                || std::is_same_v<LayerType, MaxPool2DLayer<Device>>
                || std::is_same_v<LayerType, AvgPool2DLayer<Device>>
                || std::is_same_v<LayerType, GlobalAveragePool2DLayer<Device>>)
            {
                // 4D tensor layers
                if constexpr(
                    std::is_same_v<LayerType, Conv2DLayer<Device>>
                    || std::is_same_v<LayerType, BatchNorm2DLayer<Device>>)
                {
                    // Inject clean context if available
                    if(hasCleanTensorOpContext())
                        layer.context = getCleanTensorOpContext();
                }

                if constexpr(std::is_same_v<LayerType, ReLULayer<Device>>)
                {
                    if(layer.inPlace)
                    {
                        addImplInPlace<T4>(
                            [this, layer](T4& in) mutable { layer(exec_, dev_, queue_, in); },
                            "ReLU_inplace");
                    }
                    else
                    {
                        addImpl<T4>(
                            [this, layer = std::move(layer)](T4& in) mutable
                            { return layer(exec_, dev_, queue_, in); },
                            "ReLU");
                    }
                }
                else
                {
                    addImpl<T4>(
                        [this, layer = std::move(layer)](T4& in) mutable { return layer(exec_, dev_, queue_, in); },
                        typeid(LayerType).name());
                }
            }
            else if constexpr(std::is_same_v<LayerType, FlattenTo1DLayer<Device>>)
            {
                // 4D to 1D conversion
                addImpl<T4>(
                    [this, layer = std::move(layer)](T4& in) mutable { return layer(exec_, dev_, queue_, in); },
                    "Flatten");
            }
            else if constexpr(
                std::is_same_v<LayerType, LinearLayer<Device>> || std::is_same_v<LayerType, LinearReLULayer<Device>>
                || std::is_same_v<LayerType, ReLU1DLayer<Device>>)
            {
                // 1D tensor layers
                if constexpr(std::is_same_v<LayerType, LinearLayer<Device>>)
                {
                    // Inject clean context if available
                    if(hasCleanTensorOpContext())
                        layer.context = getCleanTensorOpContext();
                }

                if constexpr(std::is_same_v<LayerType, ReLU1DLayer<Device>>)
                {
                    if(layer.inPlace)
                    {
                        addImplInPlace<T1>(
                            [this, layer](T1& in) mutable { layer(exec_, dev_, queue_, in); },
                            "ReLU1D_inplace");
                    }
                    else
                    {
                        addImpl<T1>(
                            [this, layer = std::move(layer)](T1& in) mutable
                            { return layer(exec_, dev_, queue_, in); },
                            "ReLU1D");
                    }
                }
                else
                {
                    addImpl<T1>(
                        [this, layer = std::move(layer)](T1& in) mutable { return layer(exec_, dev_, queue_, in); },
                        typeid(LayerType).name());
                }
            }
            else if constexpr(std::is_same_v<LayerType, SoftmaxLayer<Device>>)
            {
                // 1D to 2D conversion
                addImpl<T1>(
                    [this, layer = std::move(layer)](T1& in) mutable { return layer(exec_, dev_, queue_, in); },
                    "Softmax");
            }
            else if constexpr(
                std::is_same_v<LayerType, layers::LayerNorm2DLayer<Device>>
                || std::is_same_v<LayerType, layers::SelfAttention2DLayer<Device>>
                || std::is_same_v<LayerType, layers::FeedForward2DLayer<Device>>
                || std::is_same_v<LayerType, layers::BertEncoderBlock2D<Device>>)
            {
                // 2D tensor layers used by BERT-style models
                addImpl<T2>(
                    [this, layer = std::move(layer)](T2& in) mutable { return layer(exec_, dev_, queue_, in); },
                    typeid(LayerType).name());
            }
            else
            {
                static_assert(sizeof(LayerType) == 0, "Unsupported layer type for generic add() method");
            }
        }

        Any forward(Any in)
        {
            if(!profilingEnabled_)
            {
                for(auto& f : nodes_)
                    f(in);
                return in;
            }
            // profiling path
            if(lastDurations_.size() != nodes_.size())
                lastDurations_.assign(nodes_.size(), 0.0);
            for(std::size_t i = 0; i < nodes_.size(); ++i)
            {
                auto t0 = std::chrono::high_resolution_clock::now();
                nodes_[i](in);
                auto t1 = std::chrono::high_resolution_clock::now();
                lastDurations_[i] = std::chrono::duration<double, std::milli>(t1 - t0).count();
            }
            return in;
        }

        Exec const& executor() const
        {
            return exec_;
        }

        Device& device() const
        {
            return dev_;
        }

        Queue& queue() const
        {
            return queue_;
        }

    private:
        Exec const& exec_;
        Device& dev_;
        Queue& queue_;
        std::unique_ptr<CleanTensorOpContext> cleanTensorOpContext_{nullptr}; // Owned clean tensor op context
        CleanTensorOpContext* cleanTensorOpContextPtr_{nullptr}; // Non-owned clean tensor op context pointer
        std::vector<Fn> nodes_{};
        std::vector<std::string> layerNames_{};
        bool profilingEnabled_{false};
        std::vector<double> lastDurations_{}; // ms
    };

} // namespace alpaka::tensor::layers
