// C++20 compile-time TrainingSequential using concepts and fold expressions
// SPDX-License-Identifier: MPL-2.0

#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/tensor/CleanTensorOpContext.hpp>
#include <alpaka/tensor/TensorCore.hpp>

#include <array>
#include <concepts>
#include <memory>
#include <tuple>
#include <type_traits>
#include <utility>
#include <variant>

namespace alpaka::tensor::ops
{
    // Detection idiom: detect presence of nested `Cache` type on a layer
    template<class, class = void>
    struct has_cache : std::false_type
    {
    };

    template<class T>
    struct has_cache<T, std::void_t<typename T::Cache>> : std::true_type
    {
    };

    // Concepts: detect training forward/backward signatures on a layer type
    template<typename L, typename Exec, typename Device, typename Queue, typename Any>
    concept TrainForward = requires(L l, Exec const& e, Device& d, Queue& q, Any x, void* c) {
        { l.forward(e, d, q, x, c) } -> std::same_as<Any>;
    };

    template<typename L, typename Exec, typename Device, typename Queue, typename Any>
    concept TrainBackward = requires(L l, Exec const& e, Device& d, Queue& q, Any dy, void const* c) {
        { l.backward(e, d, q, dy, c) } -> std::same_as<Any>;
    };

    template<typename L, typename CtxPtr>
    concept ContextInjectable = requires(L l, CtxPtr ctx) {
        l.context = ctx; // presence of a public context pointer member
    };

    // Compile-time sequential training pipeline storing layers in a tuple
    template<typename Device, typename Exec, typename Queue, typename... Layers>
    class TrainingSequentialCT
    {
    public:
        using T4 = tensor::Tensor4D<float, Device>;
        using T1 = tensor::Tensor1D<float, Device>;
        using T2 = tensor::Tensor2D<float, Device>;
        using Any = std::variant<T4, T1, T2>;
        using CleanTensorOpContext = tensor::CleanTensorOpContext<Exec, Device, Queue>;

        explicit TrainingSequentialCT(Exec const& exec, Device& dev, Queue& q, Layers... layers)
            : exec_(exec)
            , dev_(dev)
            , queue_(q)
            , layers_(std::move(layers)...)
        {
        }

        TrainingSequentialCT(Exec const& exec, Device& dev, Queue& q, CleanTensorOpContext* ctx, Layers... layers)
            : exec_(exec)
            , dev_(dev)
            , queue_(q)
            , layers_(std::move(layers)...)
            , cleanCtxPtr_(ctx)
        {
            injectContext();
        }

        TrainingSequentialCT(Exec const& exec, Device& dev, Queue& q, CleanTensorOpContext&& ctx, Layers... layers)
            : exec_(exec)
            , dev_(dev)
            , queue_(q)
            , layers_(std::move(layers)...)
            , cleanCtxOwned_(std::make_unique<CleanTensorOpContext>(std::move(ctx)))
        {
            injectContext();
        }

        CleanTensorOpContext* context() const
        {
            return cleanCtxOwned_ ? cleanCtxOwned_.get() : cleanCtxPtr_;
        }

        // Forward: left-to-right fold
        Any forward(Any x)
        {
            caches_ = CachesTuple{}; // default construct caches
            forwardImpl(x, std::make_index_sequence<sizeof...(Layers)>{});
            return x;
        }

        // Backward: right-to-left fold
        Any backward(Any dy)
        {
            backwardImpl(dy, std::make_index_sequence<sizeof...(Layers)>{});
            return dy;
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
        // Cache storage mirrors layers count (type-erased shared_ptr<void>)
        static constexpr std::size_t kNumLayers = sizeof...(Layers);
        using CachesTuple = std::array<std::shared_ptr<void>, kNumLayers>;

        // Initialize caches using the actual member tuple types to appease NVCC
        template<std::size_t Index>
        std::shared_ptr<void> makeCacheForMember()
        {
            auto& layer = std::get<Index>(layers_);
            using L = std::remove_cvref_t<decltype(layer)>;
            if constexpr(requires { typename L::Cache; })
            {
                using CacheT = typename L::Cache;
                return std::make_shared<CacheT>();
            }
            else
            {
                return {};
            }
        }

        template<std::size_t... Is>
        void initCaches(std::index_sequence<Is...>)
        {
            ((caches_[Is] = makeCacheForMember<Is>()), ...);
        }

        // Inject CleanTensorOpContext into layers if they expose a public .context
        void injectContext()
        {
            if(auto* ctx = context())
            {
                injectContextImpl(ctx, std::make_index_sequence<sizeof...(Layers)>{});
            }
        }

        template<std::size_t... Is>
        void injectContextImpl(CleanTensorOpContext* ctx, std::index_sequence<Is...>)
        {
            (injectIfContextMember(std::get<Is>(layers_), ctx), ...);
        }

        template<typename L>
        static void injectIfContextMember(L& l, CleanTensorOpContext* ctx)
        {
            if constexpr(ContextInjectable<L, CleanTensorOpContext*>)
            {
                l.context = ctx;
            }
        }

        // Forward one layer by index
        template<std::size_t Index>
        Any forwardOne(Any x)
        {
            auto& layer = std::get<Index>(layers_);
            auto& cache = caches_.at(Index);
            using L = std::remove_cvref_t<decltype(layer)>;
            if constexpr(TrainForward<L, Exec, Device, Queue, Any>)
            {
                return layer.forward(exec_, dev_, queue_, x, cache.get());
            }
            else
            {
                return x; // pass-through if not trainable
            }
        }

        // Backward one layer by index
        template<std::size_t Index>
        Any backwardOne(Any dy)
        {
            auto& layer = std::get<Index>(layers_);
            auto const& cache = caches_.at(Index);
            using L = std::remove_cvref_t<decltype(layer)>;
            if constexpr(TrainBackward<L, Exec, Device, Queue, Any>)
            {
                return layer.backward(exec_, dev_, queue_, dy, cache.get());
            }
            else
            {
                return dy; // identity gradient if no backward
            }
        }

        // Fold forward across indices 0..N-1
        template<std::size_t... Is>
        void forwardImpl(Any& x, std::index_sequence<Is...>)
        {
            initCaches(std::index_sequence<Is...>{});
            ((x = forwardOne<Is>(std::move(x))), ...);
        }

        // Fold backward across indices N-1..0
        template<std::size_t... Is>
        void backwardImpl(Any& dy, std::index_sequence<Is...>)
        {
            ((dy = backwardOne<sizeof...(Is) - 1U - Is>(std::move(dy))), ...);
        }

        Exec const& exec_;
        Device& dev_;
        Queue& queue_;
        std::tuple<Layers...> layers_;
        CachesTuple caches_{};
        std::unique_ptr<CleanTensorOpContext> cleanCtxOwned_{};
        CleanTensorOpContext* cleanCtxPtr_{nullptr};
    };
} // namespace alpaka::tensor::ops
