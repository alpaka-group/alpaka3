// SPDX-License-Identifier: MPL-2.0
#pragma once

#include <alpaka/alpaka.hpp>
#include <alpaka/tensor/TensorCore.hpp>
#include <alpaka/tensor/TensorGeneric.hpp>

#include <array>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace alpaka::tensor::aten
{
    enum class ScalarType
    {
        Float32,
        // Double64, // deferred until ops support double
    };

    // Minimal dynamic tensor wrapper (Phase 1):
    // - Runtime dtype (Float32 only for now)
    // - Runtime rank (1D/2D supported)
    // - Device type is a template parameter (keeps performance, simplifies integration)
    template<typename Device>
    class DynamicTensor
    {
        struct IHolder
        {
            virtual ~IHolder() = default;
            virtual std::size_t rank() const = 0;
            virtual ScalarType dtype() const = 0;
            virtual std::vector<std::size_t> shape() const = 0;
        };

        template<typename T, std::size_t Rank>
        struct Holder final : IHolder
        {
            tensor::Tensor<T, Rank, Device> t;

            explicit Holder(tensor::Tensor<T, Rank, Device> v) : t(std::move(v))
            {
            }

            std::size_t rank() const override
            {
                return Rank;
            }

            ScalarType dtype() const override
            {
                if constexpr(std::is_same_v<T, float>)
                    return ScalarType::Float32;
                else
                    throw std::runtime_error("Unsupported dtype in DynamicTensor holder");
            }

            std::vector<std::size_t> shape() const override
            {
                auto s = t.shape();
                return std::vector<std::size_t>(s.begin(), s.end());
            }
        };

    public:
        DynamicTensor() = default;

        template<typename T, std::size_t Rank>
        static DynamicTensor wrap(tensor::Tensor<T, Rank, Device> v)
        {
            DynamicTensor out;
            out.holder_ = std::make_unique<Holder<T, Rank>>(std::move(v));
            out.dtype_ = out.holder_->dtype();
            out.rank_ = Rank;
            return out;
        }

        template<std::size_t Rank>
        static DynamicTensor empty(Device const& device, std::array<std::size_t, Rank> const& shape, ScalarType dt)
        {
            if(dt != ScalarType::Float32)
                throw std::runtime_error("DynamicTensor::empty only supports Float32 for now");
            if constexpr(Rank == 1)
            {
                tensor::Tensor1D<float, Device> t(device, {shape[0]}, "aten_empty_1d");
                return DynamicTensor::template wrap<float, 1>(std::move(t));
            }
            else if constexpr(Rank == 2)
            {
                tensor::Tensor2D<float, Device> t(device, {shape[0], shape[1]}, "aten_empty_2d");
                return DynamicTensor::template wrap<float, 2>(std::move(t));
            }
            else
            {
                static_assert(Rank == 1 || Rank == 2, "Rank not supported in DynamicTensor::empty");
            }
        }

        bool defined() const
        {
            return holder_ != nullptr;
        }

        ScalarType dtype() const
        {
            return dtype_;
        }

        std::size_t rank() const
        {
            return rank_;
        }

        std::vector<std::size_t> shape() const
        {
            if(!holder_)
                return {};
            return holder_->shape();
        }

        template<typename T, std::size_t Rank>
        bool is() const
        {
            if(!holder_)
                return false;
            if(rank_ != Rank)
                return false;
            if constexpr(std::is_same_v<T, float>)
                return dtype_ == ScalarType::Float32;
            else
                return false;
        }

        template<typename T, std::size_t Rank>
        inline tensor::Tensor<T, Rank, Device>& as()
        {
            if(!is<T, Rank>())
                throw std::runtime_error("DynamicTensor::as<T,Rank> type mismatch");
            return static_cast<Holder<T, Rank>*>(holder_.get())->t;
        }

        template<typename T, std::size_t Rank>
        inline tensor::Tensor<T, Rank, Device> const& as() const
        {
            if(!is<T, Rank>())
                throw std::runtime_error("DynamicTensor::as<T,Rank> type mismatch");
            return static_cast<Holder<T, Rank> const*>(holder_.get())->t;
        }

    private:
        std::unique_ptr<IHolder> holder_{};
        ScalarType dtype_{ScalarType::Float32};
        std::size_t rank_{0};
    };
} // namespace alpaka::tensor::aten
