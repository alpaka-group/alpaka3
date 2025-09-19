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
        // Write payload
        std::size_t count = 1;
        for(auto s : shape)
            count *= s;
        f.write(reinterpret_cast<char const*>(data), static_cast<std::streamsize>(count * sizeof(int64_t)));
        if(!f)
            throw std::runtime_error("Failed to write npy payload: " + path);
    }
} // namespace npy

namespace tt = alpaka::tensor;

inline void generateSampleDataFiles(std::string const& argv0, int batch, int seq)
{
    try
    {
        namespace fs = std::filesystem;
        // Find repo root from executable path and write into the default BERT data folder
        std::error_code ec;
        fs::path exeAbs = fs::absolute(fs::path(argv0), ec);
        fs::path exeDir = ec ? fs::path(argv0).parent_path() : exeAbs.parent_path();
        fs::path repoRoot = exeDir;
        for(int i = 0; i < 5; ++i)
            repoRoot = repoRoot.parent_path();
        fs::path outDir = repoRoot / "benchmark/tensor/inferenceBERT/cpp/data";
        fs::create_directories(outDir);

        std::vector<std::size_t> input_ids(static_cast<std::size_t>(batch * seq), 1);
        std::vector<std::size_t> attention_mask(static_cast<std::size_t>(batch * seq), 1);
        std::vector<std::size_t> token_type_ids(static_cast<std::size_t>(batch * seq), 0);

        auto save_vec = [&](std::string const& name, std::vector<std::size_t> const& v)
        {
            std::vector<int64_t> tmp(v.begin(), v.end());
            npy::save_i64(
                (outDir / name).string(),
                tmp.data(),
                {static_cast<std::size_t>(batch), static_cast<std::size_t>(seq)});
        };
        save_vec("input_ids.npy", input_ids);
        save_vec("attention_mask.npy", attention_mask);
        save_vec("token_type_ids.npy", token_type_ids);

        std::cout << "Sample data written to: " << outDir << "\n";
    }
    catch(std::exception const& e)
    {
        std::cerr << "Failed to generate sample data: " << e.what() << "\n";
    }
}

// Initialize embedding weights with Normal(0, 0.02) in a backend-agnostic way
template<typename Tensor2D>
inline void initEmbedding(Tensor2D& embW, uint32_t seed)
{
    std::mt19937 rng(seed);
    std::normal_distribution<float> dist(0.f, 0.02f);
    auto* h = embW.hostData();
    for(std::size_t i = 0; i < embW.size(); ++i)
        h[i] = dist(rng);
    embW.markHostModified();
}

// Add simple sin/cos positional encoding on host, then sync to device
template<typename Tensor2D, typename Device, typename Queue>
inline void addPositionalEncoding(Tensor2D& X, std::size_t seqLen, Device& device, Queue& queue)
{
    X.toHost(device, queue);
    auto M = X.shape()[0];
    auto D = X.shape()[1];
    float* h = X.hostData();
    for(std::size_t i = 0; i < M; ++i)
    {
        std::size_t pos = i % seqLen;
        for(std::size_t d = 0; d < D; ++d)
        {
            double angle
                = static_cast<double>(pos) / std::pow(10000.0, static_cast<double>(d / 2) / static_cast<double>(D));
            float pe = (d % 2 == 0) ? std::sin(angle) : std::cos(angle);
            h[i * D + d] += pe * 0.01f;
        }
    }
    X.markHostModified();
    X.ensureOnDevice(device, queue);
}
