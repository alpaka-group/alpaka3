//
// Created by tim on 17.10.25.
//

#ifndef TUNEABLEHELPER_H
#define TUNEABLEHELPER_H
#include <alpaka/UniqueId.hpp>
#include <alpaka/Vec.hpp>

#include <string>

namespace alpaka::onHost::tune::internal
{
    enum class SpecialTuneableID : std::size_t
    {
        userDef = alpaka::uniqueId(),
        numBlocks = alpaka::uniqueId(),
        numThreads = alpaka::uniqueId(),
        numFrames = alpaka::uniqueId(),
        frameExtent = alpaka::uniqueId(),
        NoTune = alpaka::uniqueId(),
        DefaultCompileTune = alpaka::uniqueId(),
        Count
    };

    /**
     * @brief kind of tuneable to ease compile-time handling**/
    enum class TunableKind
    {
        TunableMD,
        Tunable,
        CTunable,
        Dummy
    };

    template<std::size_t N, TunableKind kind>
    std::string getDefaultName()
    {
        if constexpr(kind == TunableKind::CTunable)
        {
            static int numCompileTuneables = 0;
            return "C_Tunable " + std::to_string(numCompileTuneables++);
        }
        else
        {
            static int numTuneables = 0;
            static int numMDTunables = 0;
            int& nr = numTuneables;
            std::string bareName;

            if constexpr(kind == TunableKind::TunableMD)
            {
                nr = numMDTunables;
                bareName = "TunableMD ";
            }
            else if constexpr(kind == TunableKind::Tunable)
            {
                nr = numTuneables;
                bareName = "Tunable ";
            }

            switch(N)
            {
            case static_cast<std::size_t>(SpecialTuneableID::numBlocks):
                return "NumBlocksTune";
            case static_cast<std::size_t>(SpecialTuneableID::numThreads):
                return "ThreadBlockTune";
            case static_cast<std::size_t>(SpecialTuneableID::numFrames):
                return "NumFramesTune";
            case static_cast<std::size_t>(SpecialTuneableID::frameExtent):
                return "FrameExtentTune";
            default:
                break;
            }

            // Tunables do not include the ID in their name because the name acts
            // as a persistent identifier in history matching.
            // Using a generated unique ID (e.g., from alpaka::uniqueID) would
            // result in false-negative matches after code changes.
            return bareName + std::to_string(nr++);
        }
    }

    struct NoTune
    {
        [[nodiscard]] static NoTune copy()
        {
            return NoTune{};
        }

        [[nodiscard]] static std::string toHash()
        {
            return "";
        }
    };

    template<typename T>
    constexpr bool is_NoTune_v = std::is_same_v<T, NoTune>;

    inline constexpr NoTune noTune{};

    /*
     * basically holds and stores an ID -> used to "enable" frameTuneables without specific typing, while
     */
    template<auto ID>
    struct ShallowTunableDummy
    {
        ShallowTunableDummy() = default;
        static constexpr auto tag = ID;
        static constexpr auto dim = 1;
        static constexpr auto tuneableType = TunableKind::Dummy;

        auto getNumValues() const
        {
            return Vec<uint32_t, 1u>{1u};
        }
    };
} // namespace alpaka::onHost::tune::internal
#endif // TUNEABLEHELPER_H
