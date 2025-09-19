/* TensorCore - First version of strongly typed tensor abstraction.
 * Non backwards compatible rewrite (device type bound at compile time).
 * Features:
 *  - Static shape rank
 *  - Strongly typed host/device buffers (no std::any)
 *  - Explicit coherence state enum
 *  - Automatic host modification tracking on mutable access
 *  - Row-major contiguous layout (future: add layouts)
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include <alpaka/alpaka.hpp>

#include <algorithm>
#include <array>
#include <cassert>
#include <optional>
#include <string>

namespace alpaka::tensor
{

    // Data coherence state between host & device copies
    enum class CoherenceState
    {
        Unallocated, // No memory yet
        HostFresh, // Host has newest copy; device stale or absent
        DeviceFresh, // Device has newest copy; host stale
        BothFresh // Identical & in sync
    };

    // Forward declaration
    template<typename T, std::size_t Rank, typename TDevice>
    class Tensor;

    // Aliases for convenience
    template<typename T, typename D>
    using Tensor1D = Tensor<T, 1, D>;
    template<typename T, typename D>
    using Tensor2D = Tensor<T, 2, D>;
    template<typename T, typename D>
    using Tensor3D = Tensor<T, 3, D>;
    template<typename T, typename D>
    using Tensor4D = Tensor<T, 4, D>;

    template<typename T, std::size_t Rank, typename TDevice>
    class Tensor
    {
    public:
        using value_type = T;
        using Device = TDevice;
        using Shape = std::array<std::size_t, Rank>;
        using ExtentVec = alpaka::Vec<std::size_t, Rank>;
        static constexpr std::size_t rank = Rank;

    private:
        // shape_
        // - The logical, row-major extents of the tensor (one entry per rank).
        // - Immutable with respect to the memory layout; algorithms rely on a
        //   contiguous, row-major layout consistent with this shape.
        Shape shape_{};

        // extents_
        // - Cached Alpaka extent vector corresponding to `shape_`.
        // - Stored to avoid reconstructing an `alpaka::Vec` for every operation
        //   requiring extents (allocation, kernel launches, copies).
        ExtentVec extents_{};

        // HostBuffer
        // - Concrete host allocation type for elements of T with extents `extents_`.
        // - Derived from alpaka host allocator; encapsulates ownership and pointer
        //   to the host memory region.
        using HostBuffer = decltype(alpaka::onHost::allocHost<T>(ExtentVec{}));

        // DeviceBuffer
        // - Concrete device allocation type for elements of T on `Device` with
        //   extents `extents_`.
        // - The exact type depends on the backend/device; resolved via decltype
        //   using the device-aware alpaka allocator.
        using DeviceBuffer = decltype(alpaka::onHost::alloc<T>(std::declval<Device const&>(), ExtentVec{}));

        // host_
        // - Optional owning handle to the host-side storage.
        // - Emplaced in constructors (unless moved-from) and used as the source of
        //   truth when `state_ == CoherenceState::HostFresh`.
        // - Reset when the tensor is default-constructed or moved-from without
        //   host data.
        std::optional<HostBuffer> host_{};

        // device_
        // - Optional owning handle to the device-side storage.
        // - Allocated lazily on first need (e.g., ensureOnDevice) using `boundDevice_`.
        // - When `state_ == CoherenceState::DeviceFresh`, this buffer holds the
        //   newest data; host must be updated before host reads.
        std::optional<DeviceBuffer> device_{};

        // state_
        // - Tracks the synchronization status between host/device copies.
        //   * Unallocated: no memory currently owned.
        //   * HostFresh:   host has the newest contents; device (if any) is stale.
        //   * DeviceFresh: device has the newest contents; host (if any) is stale.
        //   * BothFresh:   host and device contain identical, up-to-date data.
        // - Updated by accessors and transfer helpers to enforce correct coherence.
        CoherenceState state_{CoherenceState::Unallocated};

        // name_
        // - Optional human-readable label carried with the tensor.
        // - Useful for debugging, profiling, and diagnostics (e.g., printing).
        std::string name_{};

        // boundDevice_
        // - Non-owning pointer to the device instance used to construct this tensor.
        // - Provides the default target for lazy device allocation and transfers.
        // - May be null for default-constructed or moved-from tensors; APIs that
        //   require a device will take one explicitly in that case.
        Device const* boundDevice_{nullptr};

        std::size_t calcSize() const noexcept
        {
            // Small ranks; manual multiply is fine
            std::size_t s = 1;
            for(auto d : shape_)
                s *= d;
            return s;
        }

        static ExtentVec toExtents(Shape const& shape)
        {
            ExtentVec e{};
            for(std::size_t i = 0; i < Rank; ++i)
                e[i] = shape[i];
            return e;
        }

    public:
        Tensor() = default;

        Tensor(Device const& dev, Shape shape, std::string name = "")
            : shape_(shape)
            , extents_(toExtents(shape))
            , name_(std::move(name))
            , boundDevice_(&dev)
        {
            host_.emplace(alpaka::onHost::allocHost<T>(extents_));
            std::fill(host_->data(), host_->data() + size(), T{});
            state_ = CoherenceState::HostFresh;
        }

        // Copy
        Tensor(Tensor const& other)
            : shape_(other.shape_)
            , extents_(other.extents_)
            , name_(other.name_ + "_copy")
            , boundDevice_(other.boundDevice_)
        {
            if(other.host_)
            {
                host_.emplace(alpaka::onHost::allocHost<T>(extents_));
                std::copy(other.host_->data(), other.host_->data() + other.size(), host_->data());
                state_ = CoherenceState::HostFresh; // device side not copied
            }
        }

        Tensor& operator=(Tensor const& other)
        {
            if(this != &other)
            {
                shape_ = other.shape_;
                extents_ = other.extents_;
                boundDevice_ = other.boundDevice_;
                name_ = other.name_ + "_assigned";
                if(other.host_)
                {
                    host_.emplace(alpaka::onHost::allocHost<T>(extents_));
                    std::copy(other.host_->data(), other.host_->data() + other.size(), host_->data());
                    state_ = CoherenceState::HostFresh;
                }
                else
                {
                    host_.reset();
                    device_.reset();
                    state_ = CoherenceState::Unallocated;
                }
            }
            return *this;
        }

        // Move
        Tensor(Tensor&& other) noexcept
            : shape_(other.shape_)
            , extents_(other.extents_)
            , host_(std::move(other.host_))
            , device_(std::move(other.device_))
            , state_(other.state_)
            , name_(std::move(other.name_))
            , boundDevice_(other.boundDevice_)
        {
            other.state_ = CoherenceState::Unallocated;
            other.boundDevice_ = nullptr;
        }

        Tensor& operator=(Tensor&& other) noexcept
        {
            if(this != &other)
            {
                shape_ = other.shape_;
                extents_ = other.extents_;
                host_ = std::move(other.host_);
                device_ = std::move(other.device_);
                state_ = other.state_;
                name_ = std::move(other.name_);
                boundDevice_ = other.boundDevice_;
                other.state_ = CoherenceState::Unallocated;
                other.boundDevice_ = nullptr;
            }
            return *this;
        }

        // Introspection
        Shape const& shape() const noexcept
        {
            return shape_;
        }

        std::size_t size() const noexcept
        {
            return calcSize();
        }

        std::string const& name() const noexcept
        {
            return name_;
        }

        ExtentVec const& extents() const noexcept
        {
            return extents_;
        }

        CoherenceState coherence() const noexcept
        {
            return state_;
        }

        // Host accessors
        T* hostData()
        {
            ensureHostAllocated();
            state_ = (state_ == CoherenceState::DeviceFresh)
                         ? CoherenceState::BothFresh
                         : state_; // If device had freshest, now both fresh after potential toHost
            // Mark upcoming mutation - return mutable pointer; call markHostModified explicitly for now
            return host_->data();
        }

        T const* hostData() const
        {
            assert(host_ && "Host buffer not allocated");
            return host_->data();
        }

        // Ensure host has up-to-date copy; if device fresh, copy back
        template<typename Queue>
        void toHost(Device const& dev, Queue& q)
        {
            if(state_ == CoherenceState::DeviceFresh)
            {
                ensureDevice(dev);
                ensureHostAllocated();
                alpaka::onHost::memcpy(q, *host_, *device_);
                ::alpaka::onHost::wait(q);
                state_ = CoherenceState::BothFresh;
            }
        }

        // Mark host modified after user writes
        void markHostModified() noexcept
        {
            // Any modification on host means host now freshest copy.
            switch(state_)
            {
            case CoherenceState::DeviceFresh:
            case CoherenceState::BothFresh:
            case CoherenceState::Unallocated:
                state_ = CoherenceState::HostFresh;
                break;
            case CoherenceState::HostFresh:
                break; // already
            }
        }

        // Device buffer management
        template<typename Queue>
        void ensureOnDevice(Device const& dev, Queue& q)
        {
            ensureDevice(dev);
            if(state_ == CoherenceState::HostFresh || state_ == CoherenceState::Unallocated)
            {
                ensureHostAllocated();
                alpaka::onHost::memcpy(q, *device_, *host_);
                ::alpaka::onHost::wait(q);
                state_ = CoherenceState::BothFresh;
            }
        }

        template<typename Queue>
        void markDeviceModified(Device const&, Queue&) noexcept
        {
            switch(state_)
            {
            case CoherenceState::HostFresh:
            case CoherenceState::BothFresh:
            case CoherenceState::Unallocated:
                state_ = CoherenceState::DeviceFresh;
                break;
            case CoherenceState::DeviceFresh:
                break;
            }
        }

        // Access device buffer (non-const creates & syncs if needed)
        template<typename Queue>
        DeviceBuffer& deviceBuffer(Device const& dev, Queue& q)
        {
            ensureOnDevice(dev, q);
            return *device_;
        }

        DeviceBuffer const& deviceBufferNoSync(Device const& dev) const
        {
            assert(device_ && boundDevice_ == &dev);
            return *device_;
        }

        // Fill host and mark HostFresh
        void fill(T value)
        {
            ensureHostAllocated();
            std::fill(host_->data(), host_->data() + size(), value);
            state_ = CoherenceState::HostFresh;
        }

    private:
        void ensureHostAllocated()
        {
            if(!host_)
                host_.emplace(alpaka::onHost::allocHost<T>(extents_));
        }

        void ensureDevice(Device const& dev)
        {
            if(!device_)
            {
                if(!boundDevice_)
                    boundDevice_ = &dev;
                assert(boundDevice_ == &dev && "Tensor bound to different device type instance");
                device_.emplace(alpaka::onHost::alloc<T>(dev, extents_));
            }
        }
    };

} // namespace alpaka::tensor
