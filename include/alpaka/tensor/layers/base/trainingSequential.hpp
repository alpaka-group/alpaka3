#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/tensor/context/CleanTensorOpContext.hpp>
#include <alpaka/tensor/core/TensorCore.hpp>

#include <memory>
#include <variant>
#include <vector>

namespace alpaka::tensor::ops
{

    template<typename Device, typename Exec, typename Queue>
    class TrainingSequential
    {
    public:
        using T4 = tensor::Tensor4D<float, Device>;
        using T1 = tensor::Tensor1D<float, Device>;
        using T2 = tensor::Tensor2D<float, Device>;
        using Any = std::variant<T4, T1, T2>;
        using CleanTensorOpContext = tensor::CleanTensorOpContext<Exec, Device, Queue>;

        TrainingSequential(Exec const& exec, Device& dev, Queue& queue)
            : exec_(exec)
            , dev_(dev)
            , queue_(queue)
            , cleanTensorOpContext_(nullptr)
            , cleanTensorOpContextPtr_(nullptr)
        {
        }

        TrainingSequential(Exec const& exec, Device& dev, Queue& queue, CleanTensorOpContext&& cleanCtx)
            : exec_(exec)
            , dev_(dev)
            , queue_(queue)
            , cleanTensorOpContext_(std::make_unique<CleanTensorOpContext>(std::move(cleanCtx)))
            , cleanTensorOpContextPtr_(nullptr)
        {
        }

        TrainingSequential(Exec const& exec, Device& dev, Queue& queue, CleanTensorOpContext* cleanCtx)
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

        struct ILayer
        {
            virtual ~ILayer() = default;

            virtual std::shared_ptr<void> makeCache()
            {
                return {};
            }

            virtual Any forward(Exec const&, Device&, Queue&, Any, std::shared_ptr<void>& cache) = 0;
            virtual Any backward(Exec const&, Device&, Queue&, Any, std::shared_ptr<void> const& cache) = 0;
        };

        template<typename Layer>
        struct Model : ILayer
        {
            Layer layer;

            explicit Model(Layer l) : layer(std::move(l))
            {
            }

            std::shared_ptr<void> makeCache() override
            {
                if constexpr(requires { typename Layer::Cache; })
                {
                    return std::make_shared<typename Layer::Cache>();
                }
                else
                {
                    return {};
                }
            }

            Any forward(Exec const& e, Device& d, Queue& q, Any x, std::shared_ptr<void>& cache) override
            {
                if constexpr(requires(Layer L, Exec const& ee, Device& dd, Queue& qq, Any& xx, void* c) {
                                 L.forward(ee, dd, qq, xx, c);
                             })
                {
                    return layer.forward(e, d, q, x, cache.get());
                }
                else
                {
                    // If layer does not implement training forward, pass through
                    return x;
                }
            }

            Any backward(Exec const& e, Device& d, Queue& q, Any dy, std::shared_ptr<void> const& cache) override
            {
                if constexpr(requires(Layer L, Exec const& ee, Device& dd, Queue& qq, Any& ddy, void const* c) {
                                 L.backward(ee, dd, qq, ddy, c);
                             })
                {
                    return layer.backward(e, d, q, dy, cache.get());
                }
                else
                {
                    // If no backward, identity gradient
                    return dy;
                }
            }
        };

        template<typename Layer>
        void addTrainable(Layer layer)
        {
            // Inject clean context for provider-backed layers if available
            if(hasCleanTensorOpContext())
            {
                if constexpr(requires(Layer l) { l.context = getCleanTensorOpContext(); })
                {
                    layer.context = getCleanTensorOpContext();
                }
            }
            layers_.emplace_back(std::make_shared<Model<Layer>>(std::move(layer)));
        }

        void add(std::shared_ptr<ILayer> layer)
        {
            layers_.emplace_back(std::move(layer));
        }

        Any forward(Any x)
        {
            caches_.clear();
            caches_.reserve(layers_.size());
            for(auto& l : layers_)
            {
                auto cache = l->makeCache();
                x = l->forward(exec_, dev_, queue_, std::move(x), cache);
                caches_.push_back(std::move(cache));
            }
            return x;
        }

        Any backward(Any dy)
        {
            // Iterate in reverse order
            for(std::size_t i = layers_.size(); i-- > 0;)
            {
                dy = layers_[i]->backward(exec_, dev_, queue_, std::move(dy), caches_[i]);
            }
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
        Exec const& exec_;
        Device& dev_;
        Queue& queue_;
        std::unique_ptr<CleanTensorOpContext> cleanTensorOpContext_{nullptr};
        CleanTensorOpContext* cleanTensorOpContextPtr_{nullptr};
        std::vector<std::shared_ptr<ILayer>> layers_{};
        std::vector<std::shared_ptr<void>> caches_{};
    };

} // namespace alpaka::tensor::ops
