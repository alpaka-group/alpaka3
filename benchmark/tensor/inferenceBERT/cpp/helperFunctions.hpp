#pragma once

#include <alpaka/alpaka.hpp>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <sstream>
#include <string>
#include <vector>

// Minimal .npy reader/writer utilities
namespace npy
{
    struct Array
    {
        std::string descr; // e.g. "<i8"
        bool fortran{false};
        std::vector<std::size_t> shape;
        std::vector<std::byte> raw;
    };

    inline std::string trim(std::string s)
    {
        auto issp = [](unsigned char c) { return std::isspace(c) != 0; };
        while(!s.empty() && issp(static_cast<unsigned char>(s.front())))
            s.erase(s.begin());
        while(!s.empty() && issp(static_cast<unsigned char>(s.back())))
            s.pop_back();
        return s;
    }

    inline Array load(std::string const& path)
    {
        std::ifstream f(path, std::ios::binary);
        if(!f)
            throw std::runtime_error("Failed to open npy file: " + path);
        unsigned char magic[6]{};
        f.read(reinterpret_cast<char*>(magic), 6);
        if(!(magic[0] == 0x93 && magic[1] == 'N' && magic[2] == 'U' && magic[3] == 'M' && magic[4] == 'P'
             && magic[5] == 'Y'))
            throw std::runtime_error("Not a .npy file: " + path);
        uint8_t verMajor = 0, verMinor = 0;
        f.read(reinterpret_cast<char*>(&verMajor), 1);
        f.read(reinterpret_cast<char*>(&verMinor), 1);
        uint16_t hlen16 = 0;
        uint32_t hlen32 = 0;
        std::size_t headerLen = 0;
        if(verMajor == 1)
        {
            f.read(reinterpret_cast<char*>(&hlen16), 2);
            headerLen = hlen16;
        }
        else
        {
            f.read(reinterpret_cast<char*>(&hlen32), 4);
            headerLen = hlen32;
        }
        std::string header(headerLen, '\0');
        f.read(header.data(), static_cast<std::streamsize>(headerLen));
        Array arr;
        auto findKV = [&](char const* key)
        {
            auto pos = header.find(key);
            if(pos == std::string::npos)
                return std::string("");
            pos = header.find(':', pos);
            if(pos == std::string::npos)
                return std::string("");
            auto end = header.find_first_of(",}\n", pos + 1);
            auto val = header.substr(pos + 1, end == std::string::npos ? std::string::npos : end - pos - 1);
            return trim(val);
        };
        // descr
        auto d = findKV("'descr'");
        if(!d.empty() && (d.front() == '\'' || d.front() == '"'))
            d = d.substr(1, d.size() > 1 ? d.size() - 2 : 0);
        arr.descr = d;
        // fortran_order
        auto fo = findKV("'fortran_order'");
        for(auto& c : fo)
            c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
        arr.fortran = (fo.find("true") != std::string::npos);
        // shape tuple
        auto sh = findKV("'shape'");
        arr.shape.clear();
        std::size_t i = 0;
        while(i < sh.size() && sh[i] != '(')
            ++i;
        if(i < sh.size())
            ++i; // skip '('
        std::string num;
        for(; i < sh.size(); ++i)
        {
            char c = sh[i];
            if((c >= '0' && c <= '9'))
                num.push_back(c);
            else if(c == ',' || c == ')')
            {
                if(!num.empty())
                {
                    arr.shape.push_back(static_cast<std::size_t>(std::stoll(num)));
                    num.clear();
                }
                if(c == ')')
                    break;
            }
        }
        // Read payload
        std::size_t itemSize = 0;
        if(arr.descr == "<i8" || arr.descr == "|i8")
            itemSize = 8;
        else if(arr.descr == "<i4" || arr.descr == "|i4")
            itemSize = 4;
        else if(arr.descr == "<f4" || arr.descr == "|f4")
            itemSize = 4;
        else
            throw std::runtime_error("Unsupported dtype in npy: " + arr.descr);
        std::size_t count = 1;
        for(auto s : arr.shape)
            count *= s;
        arr.raw.resize(count * itemSize);
        f.read(reinterpret_cast<char*>(arr.raw.data()), static_cast<std::streamsize>(arr.raw.size()));
        if(!f)
            throw std::runtime_error("Failed to read npy payload: " + path);
        return arr;
    }

    template<typename T>
    inline void toVector(Array const& a, std::vector<T>& out)
    {
        std::size_t count = 1;
        for(auto s : a.shape)
            count *= s;
        out.resize(count);
        if(a.descr == "<i8" || a.descr == "|i8")
        {
            auto p = reinterpret_cast<long long const*>(a.raw.data());
            for(std::size_t i = 0; i < count; ++i)
                out[i] = static_cast<T>(p[i]);
        }
        else if(a.descr == "<i4" || a.descr == "|i4")
        {
            auto p = reinterpret_cast<int32_t const*>(a.raw.data());
            for(std::size_t i = 0; i < count; ++i)
                out[i] = static_cast<T>(p[i]);
        }
        else if(a.descr == "<f4" || a.descr == "|f4")
        {
            auto p = reinterpret_cast<float const*>(a.raw.data());
            for(std::size_t i = 0; i < count; ++i)
                out[i] = static_cast<T>(p[i]);
        }
        else
        {
            throw std::runtime_error("Unsupported dtype conversion from: " + a.descr);
        }
    }

    // Minimal NumPy .npy writer for little-endian int64 arrays with shape metadata
    inline void save_i64(std::string const& path, int64_t const* data, std::vector<std::size_t> const& shape)
    {
        std::ofstream f(path, std::ios::binary);
        if(!f)
            throw std::runtime_error("Failed to open for write: " + path);
        unsigned char const magic[6] = {0x93, 'N', 'U', 'M', 'P', 'Y'};
        f.write(reinterpret_cast<char const*>(magic), 6);
        unsigned char ver[2] = {1, 0};
        f.write(reinterpret_cast<char const*>(ver), 2);
        // Build header dict
        std::ostringstream oss;
        oss << "{";
        oss << "'descr': '<i8', ";
        oss << "'fortran_order': False, ";
        oss << "'shape': (";
        for(std::size_t i = 0; i < shape.size(); ++i)
        {
            oss << static_cast<long long>(shape[i]);
            if(i + 1 < shape.size())
                oss << ", ";
        }
        if(shape.size() == 1)
            oss << ","; // ensure tuple for 1D
        oss << ")";
        oss << "}";
        std::string header = oss.str();
        // Pad header to 16-byte alignment and end with newline
        std::size_t headerLen = header.size() + 1; // +1 for \n
        std::size_t preamble = 10; // 6 magic + 2 ver + 2 header-len (v1.0)
        std::size_t total = preamble + headerLen;
        std::size_t pad = (16 - (total % 16)) % 16;
        header.append(pad, ' ');
        header.push_back('\n');
        uint16_t hlen16 = static_cast<uint16_t>(header.size());
        f.write(reinterpret_cast<char const*>(&hlen16), 2);
        f.write(header.data(), static_cast<std::streamsize>(header.size()));
        // Payload
        std::size_t count = 1;
        for(auto s : shape)
            count *= s;
        f.write(reinterpret_cast<char const*>(data), static_cast<std::streamsize>(count * sizeof(int64_t)));
        if(!f)
            throw std::runtime_error("Failed to write npy payload: " + path);
    }
} // namespace npy

// Generate sample BERT inputs into the default data directory under the repo
inline void generateSampleDataFiles(std::string const& argv0, int batch, int seq)
{
    namespace fs = std::filesystem;
    std::error_code ec;
    fs::path exeAbs = fs::absolute(fs::path(argv0), ec);
    fs::path exeDir = ec ? fs::path(argv0).parent_path() : exeAbs.parent_path();
    fs::path repoRoot = exeDir;
    // From build/.../benchmark/tensor/inferenceBERT/cpp -> go up to repo root
    for(int i = 0; i < 5; ++i)
        repoRoot = repoRoot.parent_path();
    fs::path outDir = repoRoot / "benchmark/tensor/inferenceBERT/cpp/data";
    fs::create_directories(outDir);
    std::vector<int64_t> ids(static_cast<std::size_t>(batch) * seq);
    std::vector<int64_t> mask(ids.size());
    std::vector<int64_t> types(ids.size());
    std::mt19937 rng(1234);
    std::uniform_int_distribution<int> dist(0, 30521);
    for(std::size_t i = 0; i < ids.size(); ++i)
    {
        ids[i] = static_cast<int64_t>(dist(rng));
        mask[i] = 1;
        types[i] = 0;
    }
    std::vector<std::size_t> shape{static_cast<std::size_t>(batch), static_cast<std::size_t>(seq)};
    npy::save_i64((outDir / "input_ids.npy").string(), ids.data(), shape);
    npy::save_i64((outDir / "attention_mask.npy").string(), mask.data(), shape);
    npy::save_i64((outDir / "token_type_ids.npy").string(), types.data(), shape);
    std::cout << "Sample data written to: " << outDir.string() << "\n";
}

// Initialize embedding weights with Normal(0,0.02)
template<typename Tensor2D>
void initEmbedding(Tensor2D& W, uint32_t seed)
{
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.f, 0.02f);
    auto* h = W.hostData();
    for(std::size_t i = 0; i < W.size(); ++i)
        h[i] = dist(rng);
    W.markHostModified();
}

// Simple sinusoidal positional encoding added on host
template<typename Tensor2D, typename Device, typename Queue>
void addPositionalEncoding(Tensor2D& X, std::size_t seqLen, Device& device, Queue& queue)
{
    X.toHost(device, queue);
    auto M = X.shape()[0];
    auto D = X.shape()[1];
    float* x = X.hostData();
    for(std::size_t n = 0; n < M; ++n)
    {
        std::size_t pos = n % seqLen;
        for(std::size_t d = 0; d < D; ++d)
        {
            double angle = (double) pos / std::pow(10000.0, (double) (d / 2) / (double) D);
            float pe = (d % 2 == 0) ? std::sin(angle) : std::cos(angle);
            x[n * D + d] += pe;
        }
    }
    X.markHostModified();
    X.ensureOnDevice(device, queue);
}

// Simple elementwise add for Tensor2D on device
namespace detail
{
    struct Add2DKernel
    {
        template<typename Acc, typename BufA, typename BufB, typename BufO>
        ALPAKA_FN_ACC void operator()(Acc const& acc, BufA A, BufB B, BufO O, std::size_t M, std::size_t D) const
        {
            // Defensive: zero-size early exit (buffers are assumed valid)
            if(M == 0 || D == 0)
                return;
            for(auto [idx] :
                alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{M * D}))
            {
                std::size_t m = idx / D;
                std::size_t d = idx % D;
                O[alpaka::Vec<std::size_t, 2>{m, d}]
                    = A[alpaka::Vec<std::size_t, 2>{m, d}] + B[alpaka::Vec<std::size_t, 2>{m, d}];
            }
        }
    };

    // Alternate kernel capturing shape in object instead of passing via params
    // (used when ALPAKA_USE_TRANSFORM_ADD is set)
    struct Add2DParamKernel
    {
        std::size_t M;
        std::size_t D;

        template<typename Acc, typename BufA_, typename BufB_, typename BufO_>
        ALPAKA_FN_ACC void operator()(Acc const& acc, BufA_ A_, BufB_ B_, BufO_ O_) const
        {
            for(auto [idx] :
                alpaka::onAcc::makeIdxMap(acc, alpaka::onAcc::worker::threadsInGrid, alpaka::IdxRange{M * D}))
            {
                std::size_t m = idx / D;
                std::size_t d = idx % D;
                O_[alpaka::Vec<std::size_t, 2>{m, d}]
                    = A_[alpaka::Vec<std::size_t, 2>{m, d}] + B_[alpaka::Vec<std::size_t, 2>{m, d}];
            }
        }
    };

    // Simple transform-based elementwise add replacing Add2DKernel when enabled.
    template<typename Exec, typename Queue, typename Device, typename TensorA, typename TensorB, typename TensorOut>
    inline void residualAdd2D(Exec const& exec, Device& device, Queue& queue, TensorA& A, TensorB& B, TensorOut& O)
    {
        std::size_t M = A.shape()[0];
        std::size_t D = A.shape()[1];
        // Host-side shape validation (throws on mismatch to surface logic errors early)
        if(B.shape()[0] != M || B.shape()[1] != D || O.shape()[0] != M || O.shape()[1] != D)
        {
            throw std::runtime_error(
                "residualAdd2D shape mismatch: expected (" + std::to_string(M) + "," + std::to_string(D)
                + ") but got B=(" + std::to_string(B.shape()[0]) + "," + std::to_string(B.shape()[1]) + "), O=("
                + std::to_string(O.shape()[0]) + "," + std::to_string(O.shape()[1]) + ")");
        }
        if(M == 0 || D == 0)
            return; // nothing to do
        A.ensureOnDevice(device, queue);
        B.ensureOnDevice(device, queue);
        O.ensureOnDevice(device, queue);
        // Guard with env var to allow fallback/debug comparison.
        bool useTransform = std::getenv("ALPAKA_USE_TRANSFORM_ADD") != nullptr;
        if(!useTransform)
        {
            auto frame = alpaka::tensor::ops::detail::makeFrame<Exec, Queue>(M * D);
            queue.enqueue(
                exec,
                frame,
                Add2DKernel{},
                A.deviceBuffer(device, queue),
                B.deviceBuffer(device, queue),
                O.deviceBuffer(device, queue),
                M,
                D);
            O.markDeviceModified(device, queue);
            return;
        }

        auto frame = alpaka::tensor::ops::detail::makeFrame<Exec, Queue>(M * D);
        queue.enqueue(
            exec,
            frame,
            Add2DParamKernel{M, D},
            A.deviceBuffer(device, queue),
            B.deviceBuffer(device, queue),
            O.deviceBuffer(device, queue));
        O.markDeviceModified(device, queue);
    }
} // namespace detail
