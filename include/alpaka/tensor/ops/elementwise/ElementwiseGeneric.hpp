/* Elementwise tensor operations with Alpaka queues
 *
 * IMPORTANT: Understanding queue semantics is crucial for performance!
 * See include/alpaka/tensor/QueueSemantics.hpp for detailed documentation
 * on asynchronous execution, data dependencies, and synchronization patterns.
 */

#pragma once
#include <alpaka/alpaka.hpp>
#include <alpaka/tensor/core/SyncDebug.hpp>
#include <alpaka/tensor/core/TensorCore.hpp>

#include <type_traits>

namespace alpaka
{
    namespace tensor
    {
        namespace ops
        {

            // Generic unary elementwise kernel applying Functor: out[i] = f(in[i])
            class UnaryKernel
            {
            public:
                template<typename Acc, typename InBuf, typename OutBuf, typename Functor>
                ALPAKA_FN_ACC void operator()(Acc const& acc, InBuf in, OutBuf out, std::size_t n, Functor f) const
                {
                    // For rank > 1 views, operator[](integral) is not available.
                    // Use raw pointers for linearized access which is valid for contiguous buffers.
                    auto const* inPtr = in.data();
                    auto* outPtr = out.data();
                    for(auto [i] :
                        alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{n}))
                    {
                        outPtr[i] = f(inPtr[i]);
                    }
                }
            };

            // Generic binary elementwise kernel: out[i] = f(a[i], b[i])
            class BinaryKernel
            {
            public:
                template<typename Acc, typename ABuf, typename BBuf, typename OutBuf, typename Functor>
                ALPAKA_FN_ACC void operator()(Acc const& acc, ABuf a, BBuf b, OutBuf out, std::size_t n, Functor f)
                    const
                {
                    // For rank > 1 views, operator[](integral) is not available.
                    // Use raw pointers for linearized access which is valid for contiguous buffers.
                    auto const* aPtr = a.data();
                    auto const* bPtr = b.data();
                    auto* outPtr = out.data();
                    for(auto [i] :
                        alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{n}))
                    {
                        outPtr[i] = f(aPtr[i], bPtr[i]);
                    }
                }
            };

            // Helper launching functions
            namespace detail
            {
                template<typename Exec, typename Queue>
                inline auto makeFrame(std::size_t n)
                {
                    unsigned threadsPerBlock = 256u;
                    unsigned blocks = static_cast<unsigned>((n + threadsPerBlock - 1) / threadsPerBlock);

                    // Ensure we have enough total threads to cover all elements
                    // For GPU, make sure we have enough blocks to process all elements
                    if(blocks == 0)
                        blocks = 1;

                    return alpaka::onHost::FrameSpec{
                        alpaka::Vec<unsigned int, 1u>{blocks},
                        alpaka::Vec<unsigned int, 1u>{threadsPerBlock}};
                }
            } // namespace detail

            // Public APIs ---------------------------------------------------------

            // out = a (+,-,*,/) b using provided functor f
            // Functor must be __host__ __device__ compatible (ALPAKA_FN_HOST_ACC)

            template<typename T, std::size_t Rank, typename Exec, typename Device, typename Queue, typename Functor>
            tensor::Tensor<T, Rank, Device> binary(
                Exec const& exec,
                Device const& device,
                Queue& queue,
                tensor::Tensor<T, Rank, Device>& a,
                tensor::Tensor<T, Rank, Device>& b,
                Functor f,
                char const* name = "binary")
            {
                a.ensureOnDevice(device, queue);
                b.ensureOnDevice(device, queue);
                tensor::Tensor<T, Rank, Device> out(device, a.shape(), name);
                out.ensureOnDevice(device, queue);
                auto n = a.size();
                auto frame = detail::makeFrame<Exec, Queue>(n);
                queue.enqueue(
                    exec,
                    frame,
                    BinaryKernel{},
                    a.deviceBuffer(device, queue),
                    b.deviceBuffer(device, queue),
                    out.deviceBuffer(device, queue),
                    n,
                    f);
                // Removed forced wait (ALPAKA_DEBUG_SYNC can restore)
                out.markDeviceModified(device, queue);
                if(detail::eagerHostEnabled())
                    out.toHost(device, queue);
                return out;
            }

            // out = f(in)
            template<typename T, std::size_t Rank, typename Exec, typename Device, typename Queue, typename Functor>
            tensor::Tensor<T, Rank, Device> unary(
                Exec const& exec,
                Device const& device,
                Queue& queue,
                tensor::Tensor<T, Rank, Device>& in,
                Functor f,
                char const* name = "unary")
            {
                in.ensureOnDevice(device, queue);
                tensor::Tensor<T, Rank, Device> out(device, in.shape(), name);
                out.ensureOnDevice(device, queue);
                auto n = in.size();
                auto frame = detail::makeFrame<Exec, Queue>(n);
                queue.enqueue(
                    exec,
                    frame,
                    UnaryKernel{},
                    in.deviceBuffer(device, queue),
                    out.deviceBuffer(device, queue),
                    n,
                    f);
                // Removed forced wait (ALPAKA_DEBUG_SYNC can restore)
                out.markDeviceModified(device, queue);
                if(detail::eagerHostEnabled())
                    out.toHost(device, queue);
                return out;
            }

            // Convenience wrappers for common operations

            struct AddOp
            {
                template<typename V>
                ALPAKA_FN_HOST_ACC V operator()(V a, V b) const
                {
                    return a + b;
                }
            };

            struct SubOp
            {
                template<typename V>
                ALPAKA_FN_HOST_ACC V operator()(V a, V b) const
                {
                    return a - b;
                }
            };

            struct MulOp
            {
                template<typename V>
                ALPAKA_FN_HOST_ACC V operator()(V a, V b) const
                {
                    return a * b;
                }
            };

            struct DivOp
            {
                template<typename V>
                ALPAKA_FN_HOST_ACC V operator()(V a, V b) const
                {
                    return a / b;
                }
            };

            struct ReluOp
            {
                template<typename V>
                ALPAKA_FN_HOST_ACC V operator()(V v) const
                {
                    return v > V{} ? v : V{};
                }
            };

            // Scalar ops wrappers (functors capturing scalar value)
            template<typename S>
            struct AddScalarOp
            {
                S s;

                template<typename V>
                ALPAKA_FN_HOST_ACC V operator()(V a) const
                {
                    return a + static_cast<V>(s);
                }
            };

            template<typename S>
            struct MulScalarOp
            {
                S s;

                template<typename V>
                ALPAKA_FN_HOST_ACC V operator()(V a) const
                {
                    return a * static_cast<V>(s);
                }
            };

            // High-level wrappers

            template<typename T, std::size_t Rank, typename Exec, typename Device, typename Queue>
            tensor::Tensor<T, Rank, Device> add(
                Exec const& exec,
                Device const& device,
                Queue& queue,
                tensor::Tensor<T, Rank, Device>& a,
                tensor::Tensor<T, Rank, Device>& b)
            {
                return binary<T, Rank>(exec, device, queue, a, b, AddOp{}, "add");
            }

            template<typename T, std::size_t Rank, typename Exec, typename Device, typename Queue>
            tensor::Tensor<T, Rank, Device> sub(
                Exec const& exec,
                Device const& device,
                Queue& queue,
                tensor::Tensor<T, Rank, Device>& a,
                tensor::Tensor<T, Rank, Device>& b)
            {
                return binary<T, Rank>(exec, device, queue, a, b, SubOp{}, "sub");
            }

            template<typename T, std::size_t Rank, typename Exec, typename Device, typename Queue>
            tensor::Tensor<T, Rank, Device> mul(
                Exec const& exec,
                Device const& device,
                Queue& queue,
                tensor::Tensor<T, Rank, Device>& a,
                tensor::Tensor<T, Rank, Device>& b)
            {
                return binary<T, Rank>(exec, device, queue, a, b, MulOp{}, "mul");
            }

            template<typename T, std::size_t Rank, typename Exec, typename Device, typename Queue>
            tensor::Tensor<T, Rank, Device> div(
                Exec const& exec,
                Device const& device,
                Queue& queue,
                tensor::Tensor<T, Rank, Device>& a,
                tensor::Tensor<T, Rank, Device>& b)
            {
                return binary<T, Rank>(exec, device, queue, a, b, DivOp{}, "div");
            }

            template<typename T, std::size_t Rank, typename Exec, typename Device, typename Queue>
            tensor::Tensor<T, Rank, Device> relu(
                Exec const& exec,
                Device const& device,
                Queue& queue,
                tensor::Tensor<T, Rank, Device>& in)
            {
                return unary<T, Rank>(exec, device, queue, in, ReluOp{}, "relu");
            }

            // sub/mul/div already provided via generic wrappers above

            template<typename T, std::size_t Rank, typename Exec, typename Device, typename Queue, typename S>
            tensor::Tensor<T, Rank, Device> add_scalar(
                Exec const& exec,
                Device const& device,
                Queue& queue,
                tensor::Tensor<T, Rank, Device>& in,
                S scalar,
                char const* name = "add_scalar")
            {
                return unary<T, Rank>(exec, device, queue, in, AddScalarOp<S>{scalar}, name);
            }

            template<typename T, std::size_t Rank, typename Exec, typename Device, typename Queue, typename S>
            tensor::Tensor<T, Rank, Device> mul_scalar(
                Exec const& exec,
                Device const& device,
                Queue& queue,
                tensor::Tensor<T, Rank, Device>& in,
                S scalar,
                char const* name = "mul_scalar")
            {
                return unary<T, Rank>(exec, device, queue, in, MulScalarOp<S>{scalar}, name);
            }

            // In-place ReLU

            template<typename T, std::size_t Rank, typename Exec, typename Device, typename Queue>
            void relu_inplace(Exec const& exec, Device const& device, Queue& queue, tensor::Tensor<T, Rank, Device>& t)
            {
                t.ensureOnDevice(device, queue);
                auto n = t.size();
                auto frame = detail::makeFrame<Exec, Queue>(n);
                queue.enqueue(
                    exec,
                    frame,
                    UnaryKernel{},
                    t.deviceBuffer(device, queue),
                    t.deviceBuffer(device, queue),
                    n,
                    ReluOp{});
                // Removed forced wait (ALPAKA_DEBUG_SYNC can restore)
                t.markDeviceModified(device, queue);
                if(detail::eagerHostEnabled())
                    t.toHost(device, queue);
            }

        } // namespace ops
    } // namespace tensor
} // namespace alpaka
