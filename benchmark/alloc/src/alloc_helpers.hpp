// Alpaka‑style micro‑benchmark helpers for memory allocation
// SPDX‑License‑Identifier: MPL‑2.0
// Authors: Ivan Andriievskyi, Jiří Vyskočil
// Work funded by US NAS and ONRG (IMPRESS-U).
#pragma once

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <limits>
#include <map>
#include <numeric>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

namespace alpaka::benchmark::alloc
{
    enum class BMInfoDataType
    {
        AcceleratorType,
        NumRuns,
        DataSize,
        DeviceName,
        KernelNames,
        KernelDataUsageValues,
        KernelBandwidths,
        KernelMinTimes,
        KernelMaxTimes,
        KernelAvgTimes,
        KernelColdTimes
    };

    template<typename T>
    inline bool FuzzyEqual(T a, T b)
    {
        if constexpr(std::is_floating_point_v<T>)
        {
            return std::fabs(a - b) < (std::numeric_limits<T>::epsilon() * static_cast<T>(100.0));
        }
        else
        {
            return a == b;
        }
    }

    template<typename T>
    inline std::string joinElements(std::vector<T> const& vec, std::string const& delim)
    {
        std::ostringstream oss;
        for(std::size_t i = 0; i < vec.size(); ++i)
        {
            if(i != 0)
            {
                oss << delim;
            }
            oss << std::setprecision(5) << vec[i];
        }
        return oss.str();
    }

    inline std::string toStr(BMInfoDataType value)
    {
        switch(value)
        {
        case BMInfoDataType::AcceleratorType:
            return "Accelerator";
        case BMInfoDataType::NumRuns:
            return "Runs";
        case BMInfoDataType::DataSize:
            return "Bytes";
        case BMInfoDataType::DeviceName:
            return "Device";
        case BMInfoDataType::KernelNames:
            return "Kernels";
        case BMInfoDataType::KernelDataUsageValues:
            return "Data(MB)";
        case BMInfoDataType::KernelBandwidths:
            return "BW(GB/s)";
        case BMInfoDataType::KernelMinTimes:
            return "Min(s)";
        case BMInfoDataType::KernelMaxTimes:
            return "Max(s)";
        case BMInfoDataType::KernelAvgTimes:
            return "Avg(s)";
        case BMInfoDataType::KernelColdTimes:
            return "Cold(s)";
        }
        return "";
    }

    class BenchmarkMetaData
    {
        std::map<BMInfoDataType, std::string> m_items;

    public:
        template<typename T_Value>
        void setItem(BMInfoDataType key, T_Value const& value)
        {
            std::ostringstream oss;
            oss << value;
            m_items[key] = oss.str();
        }

        std::string serializeAsTable() const
        {
            std::ostringstream ss;

            for(auto key :
                {BMInfoDataType::AcceleratorType,
                 BMInfoDataType::DeviceName,
                 BMInfoDataType::DataSize,
                 BMInfoDataType::NumRuns})
            {
                if(auto it = m_items.find(key); it != m_items.end())
                {
                    ss << toStr(key) << ':' << it->second << '\n';
                }
            }

            if(auto cold = m_items.find(BMInfoDataType::KernelColdTimes); cold != m_items.end())
            {
                ss << toStr(BMInfoDataType::KernelColdTimes) << ':' << cold->second << '\n';
            }

            auto const& namesSerialized = m_items.at(BMInfoDataType::KernelNames);
            auto const split = [](std::string const& str)
            {
                std::vector<std::string> values;
                std::stringstream stream(str);
                std::string token;
                while(std::getline(stream, token, ','))
                {
                    if(!token.empty() && token.front() == ' ')
                    {
                        token.erase(token.begin());
                    }
                    values.push_back(token);
                }
                return values;
            };

            auto const names = split(namesSerialized);
            auto const dataMegabytes = split(m_items.at(BMInfoDataType::KernelDataUsageValues));
            auto const bandwidths = split(m_items.at(BMInfoDataType::KernelBandwidths));
            auto const minTimes = split(m_items.at(BMInfoDataType::KernelMinTimes));
            auto const maxTimes = split(m_items.at(BMInfoDataType::KernelMaxTimes));
            auto const avgTimes = split(m_items.at(BMInfoDataType::KernelAvgTimes));

            ss << std::left << std::setw(12) << "Kernel" << ' ' << std::setw(10) << "BW" << ' ' << std::setw(8)
               << "Min" << ' ' << std::setw(8) << "Max" << ' ' << std::setw(8) << "Avg" << ' ' << "Data" << '\n';

            for(std::size_t i = 0; i < names.size(); ++i)
            {
                ss << std::left << std::setw(12) << names[i] << ' ' << std::setw(10) << bandwidths[i] << ' '
                   << std::setw(8) << minTimes[i] << ' ' << std::setw(8) << maxTimes[i] << ' ' << std::setw(8)
                   << avgTimes[i] << ' ' << dataMegabytes[i] << '\n';
            }

            return ss.str();
        }
    };

    struct AllocResults
    {
        struct Entry
        {
            std::vector<double> times;
            double bytesMB{0.0};
        };

        std::map<std::string, Entry> data;

        void addKernel(std::string const& name, double bytes)
        {
            data[name] = Entry{{}, bytes};
        }

        void setTimes(std::string const& name, std::vector<double> values)
        {
            data[name].times = std::move(values);
        }

        static double min(std::vector<double> const& values)
        {
            if(values.empty())
            {
                return 0.0;
            }
            if(values.size() == 1)
            {
                return values.front();
            }
            return *std::min_element(values.begin() + 1, values.end());
        }

        static double max(std::vector<double> const& values)
        {
            if(values.empty())
            {
                return 0.0;
            }
            if(values.size() == 1)
            {
                return values.front();
            }
            return *std::max_element(values.begin() + 1, values.end());
        }

        static double avg(std::vector<double> const& values)
        {
            if(values.empty())
            {
                return 0.0;
            }
            auto const* beginHot = values.size() > 1 ? values.data() + 1 : values.data();
            auto const count = static_cast<double>(values.size() > 1 ? values.size() - 1 : values.size());
            return count > 0.0 ? std::accumulate(beginHot, values.data() + values.size(), 0.0) / count : 0.0;
        }

        static double cold(std::vector<double> const& values)
        {
            return values.empty() ? 0.0 : values.front();
        }

        std::vector<double> bandwidths() const
        {
            std::vector<double> result;
            for(auto const& [_, entry] : data)
            {
                result.push_back(entry.bytesMB / 1.0e3 / min(entry.times));
            }
            return result;
        }

        std::vector<double> mins() const
        {
            std::vector<double> result;
            for(auto const& [_, entry] : data)
            {
                result.push_back(min(entry.times));
            }
            return result;
        }

        std::vector<double> maxs() const
        {
            std::vector<double> result;
            for(auto const& [_, entry] : data)
            {
                result.push_back(max(entry.times));
            }
            return result;
        }

        std::vector<double> avgs() const
        {
            std::vector<double> result;
            for(auto const& [_, entry] : data)
            {
                result.push_back(avg(entry.times));
            }
            return result;
        }

        std::vector<double> colds() const
        {
            std::vector<double> result;
            for(auto const& [_, entry] : data)
            {
                result.push_back(cold(entry.times));
            }
            return result;
        }

        std::vector<double> bytes() const
        {
            std::vector<double> result;
            for(auto const& [_, entry] : data)
            {
                result.push_back(entry.bytesMB);
            }
            return result;
        }

        std::string kernelNames() const
        {
            std::string names;
            for(auto const& [name, _] : data)
            {
                if(!names.empty())
                {
                    names += ", ";
                }
                names += name;
            }
            return names;
        }
    };
} // namespace alpaka::benchmark::alloc
