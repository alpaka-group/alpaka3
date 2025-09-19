#include <array>
#include <cassert>
#include <iostream>
#include <memory>

// Minimal standalone test of the provider architecture
// This test uses simple mock tensor types to validate the design

// Mock minimal types for testing
struct MockDevice
{
};

struct MockQueue
{
};

template<typename T>
struct MockTensor
{
    std::array<size_t, 4> dims;

    size_t size() const
    {
        return dims[0] * dims[1] * dims[2] * dims[3];
    }
};

using Tensor4D = MockTensor<float>;

// Simple OpStatus enum for testing
enum class OpStatus
{
    Success,
    Failure,
    NotSupported
};

// Provider interface
enum class OpType
{
    Conv2D,
    GEMM,
    BatchNorm
};

class IOpProvider
{
public:
    virtual ~IOpProvider() = default;

    virtual bool supportsOperation(OpType type) const = 0;

    virtual OpStatus conv2d(
        MockDevice& device,
        MockQueue& queue,
        Tensor4D const& input,
        Tensor4D const& weights,
        Tensor4D& output)
        = 0;

    virtual OpStatus gemm(MockDevice& device, MockQueue& queue, Tensor4D const& a, Tensor4D const& b, Tensor4D& c) = 0;

    virtual OpStatus batchNorm(MockDevice& device, MockQueue& queue, Tensor4D const& input, Tensor4D& output) = 0;

    virtual std::string getName() const = 0;
};

// Mock CUDA provider
class MockCuBLASProvider : public IOpProvider
{
public:
    bool supportsOperation(OpType type) const override
    {
        return type == OpType::GEMM || type == OpType::Conv2D;
    }

    OpStatus conv2d(MockDevice&, MockQueue&, Tensor4D const&, Tensor4D const&, Tensor4D&) override
    {
        std::cout << "CuBLAS: Performing Conv2D operation\n";
        return OpStatus::Success;
    }

    OpStatus gemm(MockDevice&, MockQueue&, Tensor4D const&, Tensor4D const&, Tensor4D&) override
    {
        std::cout << "CuBLAS: Performing GEMM operation\n";
        return OpStatus::Success;
    }

    OpStatus batchNorm(MockDevice&, MockQueue&, Tensor4D const&, Tensor4D&) override
    {
        std::cout << "CuBLAS: BatchNorm not supported\n";
        return OpStatus::NotSupported;
    }

    std::string getName() const override
    {
        return "MockCuBLASProvider";
    }
};

// Mock CPU provider
class MockDefaultProvider : public IOpProvider
{
public:
    bool supportsOperation(OpType type) const override
    {
        return true; // Supports all operations
    }

    OpStatus conv2d(MockDevice&, MockQueue&, Tensor4D const&, Tensor4D const&, Tensor4D&) override
    {
        std::cout << "Default: Performing Conv2D operation\n";
        return OpStatus::Success;
    }

    OpStatus gemm(MockDevice&, MockQueue&, Tensor4D const&, Tensor4D const&, Tensor4D&) override
    {
        std::cout << "Default: Performing GEMM operation\n";
        return OpStatus::Success;
    }

    OpStatus batchNorm(MockDevice&, MockQueue&, Tensor4D const&, Tensor4D&) override
    {
        std::cout << "Default: Performing BatchNorm operation\n";
        return OpStatus::Success;
    }

    std::string getName() const override
    {
        return "MockDefaultProvider";
    }
};

// Factory for provider creation
enum class DeviceType
{
    CPU,
    CUDA,
    HIP
};

class ProviderFactory
{
public:
    static std::unique_ptr<IOpProvider> createProvider(DeviceType deviceType)
    {
        switch(deviceType)
        {
        case DeviceType::CUDA:
            return std::make_unique<MockCuBLASProvider>();
        case DeviceType::CPU:
        case DeviceType::HIP:
        default:
            return std::make_unique<MockDefaultProvider>();
        }
    }
};

// Clean context that coordinates providers
class CleanTensorOpContext
{
private:
    std::unique_ptr<IOpProvider> convProvider_;
    std::unique_ptr<IOpProvider> gemmProvider_;

public:
    CleanTensorOpContext(DeviceType deviceType)
    {
        convProvider_ = ProviderFactory::createProvider(deviceType);
        gemmProvider_ = ProviderFactory::createProvider(deviceType);
    }

    OpStatus conv2d(
        MockDevice& device,
        MockQueue& queue,
        Tensor4D const& input,
        Tensor4D const& weights,
        Tensor4D& output)
    {
        if(!convProvider_->supportsOperation(OpType::Conv2D))
        {
            std::cout << "Conv2D not supported by " << convProvider_->getName() << ", using fallback\n";
            auto fallback = ProviderFactory::createProvider(DeviceType::CPU);
            return fallback->conv2d(device, queue, input, weights, output);
        }
        return convProvider_->conv2d(device, queue, input, weights, output);
    }

    OpStatus gemm(MockDevice& device, MockQueue& queue, Tensor4D const& a, Tensor4D const& b, Tensor4D& c)
    {
        if(!gemmProvider_->supportsOperation(OpType::GEMM))
        {
            std::cout << "GEMM not supported by " << gemmProvider_->getName() << ", using fallback\n";
            auto fallback = ProviderFactory::createProvider(DeviceType::CPU);
            return fallback->gemm(device, queue, a, b, c);
        }
        return gemmProvider_->gemm(device, queue, a, b, c);
    }

    OpStatus batchNorm(MockDevice& device, MockQueue& queue, Tensor4D const& input, Tensor4D& output)
    {
        if(!convProvider_->supportsOperation(OpType::BatchNorm))
        {
            std::cout << "BatchNorm not supported by " << convProvider_->getName() << ", using fallback\n";
            auto fallback = ProviderFactory::createProvider(DeviceType::CPU);
            return fallback->batchNorm(device, queue, input, output);
        }
        return convProvider_->batchNorm(device, queue, input, output);
    }
};

// Test the clean architecture
int main()
{
    std::cout << "=== Testing Clean Provider Architecture ===\n\n";

    // Create mock objects
    MockDevice device;
    MockQueue queue;
    Tensor4D input{{1, 3, 224, 224}};
    Tensor4D weights{{64, 3, 7, 7}};
    Tensor4D output{{1, 64, 112, 112}};
    Tensor4D a{{1, 256}};
    Tensor4D b{{256, 1000}};
    Tensor4D c{{1, 1000}};

    // Test CUDA context
    std::cout << "1. Testing CUDA Context:\n";
    CleanTensorOpContext cudaContext(DeviceType::CUDA);

    auto status = cudaContext.conv2d(device, queue, input, weights, output);
    assert(status == OpStatus::Success);

    status = cudaContext.gemm(device, queue, a, b, c);
    assert(status == OpStatus::Success);

    status = cudaContext.batchNorm(device, queue, input, output);
    assert(status == OpStatus::Success); // Should fallback to default

    std::cout << "\n2. Testing CPU Context:\n";
    CleanTensorOpContext cpuContext(DeviceType::CPU);

    status = cpuContext.conv2d(device, queue, input, weights, output);
    assert(status == OpStatus::Success);

    status = cpuContext.gemm(device, queue, a, b, c);
    assert(status == OpStatus::Success);

    status = cpuContext.batchNorm(device, queue, input, output);
    assert(status == OpStatus::Success);

    std::cout << "\n=== All Tests Passed! ===\n";
    std::cout << "\nArchitectural Benefits Demonstrated:\n";
    std::cout << "✓ Single Responsibility: Each provider handles one backend\n";
    std::cout << "✓ Open/Closed: Easy to add new providers without changing existing code\n";
    std::cout << "✓ Dependency Inversion: Context depends on abstractions, not concrete providers\n";
    std::cout << "✓ Clean Separation: No backend-specific code in context\n";
    std::cout << "✓ Graceful Fallback: Automatic fallback when operations not supported\n";

    return 0;
}
