/* Simple Clean Provider Design Test
 * Demonstrates the architectural principles without full Alpaka integration
 * SPDX-License-Identifier: MPL-2.0
 */

#include <cstdlib>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

namespace clean_design_demo
{
    // Simplified operation types
    enum class OpType
    {
        GEMM,
        Conv2D,
        BatchNorm
    };

    enum class OpStatus
    {
        Success,
        Unsupported,
        Error
    };

    // Pure provider interface
    class IOpProvider
    {
    public:
        virtual ~IOpProvider() = default;
        virtual std::string getBackendName() const = 0;
        virtual bool supportsOperation(OpType op) const = 0;
        virtual bool isActive() const = 0;

        virtual OpStatus executeGemm(size_t M, size_t N, size_t K) = 0;
        virtual OpStatus executeConv2D(size_t batch, size_t channels) = 0;
    };

    // Default provider implementation
    class DefaultProvider : public IOpProvider
    {
    public:
        std::string getBackendName() const override
        {
            return "Default (Generic Kernels)";
        }

        bool supportsOperation(OpType op) const override
        {
            return true; // Supports all operations via fallback
        }

        bool isActive() const override
        {
            return true; // Always available
        }

        OpStatus executeGemm(size_t M, size_t N, size_t K) override
        {
            std::cout << "  DefaultProvider::GEMM(" << M << "x" << N << "x" << K << ") - using generic kernel"
                      << std::endl;
            return OpStatus::Success;
        }

        OpStatus executeConv2D(size_t batch, size_t channels) override
        {
            std::cout << "  DefaultProvider::Conv2D(batch=" << batch << ", channels=" << channels
                      << ") - using generic kernel" << std::endl;
            return OpStatus::Success;
        }
    };

    // Specialized CUDA provider
    class CudaProvider : public IOpProvider
    {
    private:
        bool cudaAvailable_;

    public:
        CudaProvider() : cudaAvailable_(!std::getenv("DISABLE_CUDA"))
        {
            if(cudaAvailable_)
            {
                std::cout << "CudaProvider: Initialized successfully" << std::endl;
            }
        }

        std::string getBackendName() const override
        {
            return "CUDA (cuBLAS + cuDNN)";
        }

        bool supportsOperation(OpType op) const override
        {
            return cudaAvailable_ && (op == OpType::GEMM || op == OpType::Conv2D);
        }

        bool isActive() const override
        {
            return cudaAvailable_;
        }

        OpStatus executeGemm(size_t M, size_t N, size_t K) override
        {
            if(!isActive())
                return OpStatus::Unsupported;

            std::cout << "  CudaProvider::GEMM(" << M << "x" << N << "x" << K << ") - using cuBLAS with Tensor Cores"
                      << std::endl;
            return OpStatus::Success;
        }

        OpStatus executeConv2D(size_t batch, size_t channels) override
        {
            if(!isActive())
                return OpStatus::Unsupported;

            std::cout << "  CudaProvider::Conv2D(batch=" << batch << ", channels=" << channels << ") - using cuDNN"
                      << std::endl;
            return OpStatus::Success;
        }
    };

    // Provider factory
    enum class DeviceType
    {
        CPU,
        GPU_CUDA,
        GPU_HIP
    };

    class ProviderFactory
    {
    public:
        static std::unique_ptr<IOpProvider> createProvider(DeviceType device)
        {
            switch(device)
            {
            case DeviceType::GPU_CUDA:
                {
                    auto cuda = std::make_unique<CudaProvider>();
                    if(cuda->isActive())
                        return std::move(cuda);
                    // Fall through to default
                }
            case DeviceType::CPU:
            case DeviceType::GPU_HIP:
            default:
                return std::make_unique<DefaultProvider>();
            }
        }
    };

    // Clean context that coordinates providers
    class CleanTensorContext
    {
    private:
        std::unique_ptr<IOpProvider> gemmProvider_;
        std::unique_ptr<IOpProvider> convProvider_;
        std::unique_ptr<IOpProvider> fallbackProvider_;
        DeviceType device_;

    public:
        CleanTensorContext(DeviceType device) : device_(device)
        {
            // Create specialized providers
            gemmProvider_ = ProviderFactory::createProvider(device);
            convProvider_ = ProviderFactory::createProvider(device);
            fallbackProvider_ = std::make_unique<DefaultProvider>();
        }

        void printInfo() const
        {
            std::cout << "=== Clean Tensor Context ===" << std::endl;
            std::cout << "Device: " << getDeviceName() << std::endl;
            std::cout << "GEMM Provider: " << gemmProvider_->getBackendName() << std::endl;
            std::cout << "Conv Provider: " << convProvider_->getBackendName() << std::endl;
            std::cout << "============================" << std::endl;
        }

        // Pure delegation - no backend-specific code
        void executeGemm(size_t M, size_t N, size_t K)
        {
            std::cout << "CleanTensorContext::GEMM - delegating to provider..." << std::endl;

            auto& provider = getGemmProvider();
            auto status = provider.executeGemm(M, N, K);

            if(status != OpStatus::Success && &provider != fallbackProvider_.get())
            {
                std::cout << "  Primary provider failed, falling back to default..." << std::endl;
                fallbackProvider_->executeGemm(M, N, K);
            }
        }

        void executeConv2D(size_t batch, size_t channels)
        {
            std::cout << "CleanTensorContext::Conv2D - delegating to provider..." << std::endl;

            auto& provider = getConvProvider();
            auto status = provider.executeConv2D(batch, channels);

            if(status != OpStatus::Success && &provider != fallbackProvider_.get())
            {
                std::cout << "  Primary provider failed, falling back to default..." << std::endl;
                fallbackProvider_->executeConv2D(batch, channels);
            }
        }

    private:
        IOpProvider& getGemmProvider()
        {
            if(gemmProvider_->isActive() && gemmProvider_->supportsOperation(OpType::GEMM))
                return *gemmProvider_;
            return *fallbackProvider_;
        }

        IOpProvider& getConvProvider()
        {
            if(convProvider_->isActive() && convProvider_->supportsOperation(OpType::Conv2D))
                return *convProvider_;
            return *fallbackProvider_;
        }

        std::string getDeviceName() const
        {
            switch(device_)
            {
            case DeviceType::CPU:
                return "CPU";
            case DeviceType::GPU_CUDA:
                return "CUDA GPU";
            case DeviceType::GPU_HIP:
                return "HIP GPU";
            default:
                return "Unknown";
            }
        }
    };
} // namespace clean_design_demo

int main()
{
    using namespace clean_design_demo;

    std::cout << "=== Clean Provider Design Demo ===" << std::endl;

    // Test 1: CPU context
    std::cout << "\n--- Testing CPU Context ---" << std::endl;
    {
        CleanTensorContext context(DeviceType::CPU);
        context.printInfo();
        context.executeGemm(128, 64, 32);
        context.executeConv2D(32, 256);
    }

    // Test 2: CUDA context (available)
    std::cout << "\n--- Testing CUDA Context (Available) ---" << std::endl;
    {
        CleanTensorContext context(DeviceType::GPU_CUDA);
        context.printInfo();
        context.executeGemm(256, 128, 64);
        context.executeConv2D(64, 512);
    }

    // Test 3: CUDA context (disabled)
    std::cout << "\n--- Testing CUDA Context (Disabled) ---" << std::endl;
    {
        setenv("DISABLE_CUDA", "1", 1);
        CleanTensorContext context(DeviceType::GPU_CUDA);
        context.printInfo();
        context.executeGemm(512, 256, 128);
        context.executeConv2D(128, 1024);
        unsetenv("DISABLE_CUDA");
    }

    std::cout << "\n=== Design Benefits Demonstrated ===" << std::endl;
    std::cout << "✓ Single Responsibility: Each provider handles one backend" << std::endl;
    std::cout << "✓ Open/Closed: Added providers without modifying existing code" << std::endl;
    std::cout << "✓ Pure Delegation: Context contains no backend-specific logic" << std::endl;
    std::cout << "✓ Graceful Fallback: Automatic fallback when specialized providers fail" << std::endl;
    std::cout << "✓ Easy Extension: Adding hipBLAS/MIOpen requires only new provider classes" << std::endl;

    return 0;
}
