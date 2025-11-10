//
// Created by tim on 16.02.25.
//

#ifndef TUNEABLE_H
#define TUNEABLE_H
#include <alpaka/mem/IdxRange.hpp>
#include <alpaka/onHost/tune/tunable/tunableHelper.hpp>
#include <alpaka/onHost/tune/utils/hashUtils.hpp>
#include <alpaka/onHost/tune/utils/tupleHelper.hpp>
#include <alpaka/onHost/tune/utils/vecUtils.hpp>

#include <algorithm>
#include <string>
#include <utility>

namespace alpaka::onHost::tune
{
    namespace internal
    {
        // get the first argument from a parameter pack, where T is compliant with the tunable interface
        template<typename First, typename...>
        struct GetTFromArgs
        {
        private:
            // Match std::vector<T>
            template<typename T>
            static auto deduce(std::vector<T> const&) -> T;

            // Match std::initializer_list<T>
            template<typename T>
            static auto deduce(std::initializer_list<T>) -> T;

            // Match IdxRange<T>
            template<typename T>
            static auto deduce(IdxRange<T> const&) -> T;

            // Fallback triggers static_assert
            static void deduce(...);

        public:
            using type = decltype(deduce(std::declval<First>()));
        };

        template<typename... Args>
        using GetTunableTypeFromParameterPack = typename GetTFromArgs<Args...>::type;
        template<typename... Args>
        using constructTunableMD_FromParameterPack
            = TunableMD<alpaka::uniqueId(), GetTunableTypeFromParameterPack<Args...>>;

        template<auto ID, typename T, auto Dim, internal::TunableKind kind, typename T_Storage>
        struct BaseTunable
        {
            using ValueType = T;
            using StorageType = T_Storage;
            BaseTunable() = default;
            virtual ~BaseTunable() = default;
            virtual Vec<uint32_t, Dim> getNumValues() const = 0;
            [[nodiscard]] virtual std::string getName() const = 0;
            static constexpr auto tag = ID;
            static constexpr auto tuneableType = kind;
            std::optional<uint32_t> startingIndex;
        };
    } // namespace internal

    /**
     * @brief Used to define a compile-time tuneable.
     * All tuning values are encoded as template arguments and accessible during compilation.
     * Each `CTunable` defines a fixed, immutable configuration space, internally represented as a `std::tuple`
     * IMPORTANT: supply template types, such as std::integral_constant or alpaka::CVec to wrap arithmetic types and
     * use them in your kernel (all these types T... need to define trivially copyable objects)
     * CTunable is NOT constexpr because it uses a string name identifier -- to adhere to a common tunable interface.
     *
     * combination with a trait definition of your Kernel @see alpaka::tune::traits
     *
     *  Example usage:
     * @code
     * using namespace alpaka::tune;
     * auto tileSize = CTunable<
     *     alpaka::uniqueID(), // Identifier
     *     CVec<uint32_t, 20>,
     *     CVec<uint32_t, 40>,
     *     CVec<uint32_t, 80>,
     * >{};
     */

    template<auto ID = static_cast<uint32_t>(internal::SpecialTuneableID::DefaultCompileTune), typename... T>
    struct CTunable
    {
        static constexpr auto tag = ID;
        static constexpr auto tuneableType = internal::TunableKind::CTunable;
        static constexpr auto dim = 1u;
        // All CVecs must be compatible
        static_assert((std::is_trivially_copyable_v<T> && ...), "tunable types must be trivially copyable!");
        static_assert(sizeof...(T) > 0, "CTunable requires at least one Parameter");
        using Tuple = std::tuple<T...>;
        using Values = Tuple; // this is the constexpr compile-time payload
        using value_type = std::tuple_element_t<0, std::tuple<T...>>;
        std::string m_name = internal::getDefaultName<ID, tuneableType>();
        std::optional<uint32_t> startingIndex = std::nullopt;
        // Runtime storage (for BaseTunable interface)
        static constexpr std::array<Tuple, 1u> values = {Tuple{}};

        template<std::size_t I>
        static constexpr auto getValueByIndex()
        {
            static_assert(I < std::tuple_size_v<Values>, " Ctuneable index access, exceeds boundaries!");
            constexpr auto elem = std::get<I>(values[0]);
            return elem;
        }

        // we hash need to hash the types in addition to the values -- not static in order to have a common interface
        // for all tunables

        [[nodiscard]] std::size_t valuesToHash() const
        {
            std::size_t seed = 0;

            std::apply(
                [&]<typename... T0>(T0... v) { ((internal::hashType<std::decay_t<T0>>(seed)), ...); },
                Values{});

            return seed;
        }

        constexpr explicit CTunable(std::string const& name = "")
        {
            if(!name.empty())
                m_name = name;
        }

        constexpr ~CTunable() = default;

        // --- BaseTunable Interface Implementations ---

        [[nodiscard]] std::string getName() const
        {
            return m_name;
        }

        [[nodiscard]] static constexpr Vec<uint32_t, 1u> getNumValues()
        {
            return Vec<uint32_t, 1u>{sizeof...(T)};
        }
    };

    /**
     * @brief Used to define a Runtime tuneable.
     *specialization of CTuneable to expand a tuple of values into a parameter pack -> to ease generator syntax**/

    template<auto ID, typename... T>
    struct CTunable<ID, std::tuple<T...>> : CTunable<ID, T...>
    {
    };

    /**
     * @brief Used to define a Runtime tuneable.
     *
     * Tunable` allows defining runtime-resolved tuning parameters
     * it must be defined on the Host.
     * The provided values will be used on the device and therefore require to be trivially copyable.
     * the values are stored in a `std::vector`.
     * Runtime generators (e.g., `linSpace`, `logSpace`) can be used to generate tuning spaces.
     * (see namespace alpaka::tune::generate)
     *
     *
     * Example usage:
     * @code
     *
     * struct foo{};
     * struct bar{};
     * using namespace alpaka::tune;
     * auto tune = Tunable<ID>({foo{}, bar{}});
     * auto learningRateA = Tunable({0.005, 0.02, 0.1});
     * auto learningRateB = Tunable<alpaka::uniqueID()>(generate::logSpace(1e-6,0.1,2),0,"learningRate");
     * @endcode
     *
     * @tparam ID Identifier for the tuneable ->
     * must only be modified when using this tuneable in a constraint
     * ( unless using this for frameSpec tunes -> IDs for frameTuneables found under namespace: tune::frame)
     *
     *
     *
     * @tparam T Type of the values stored in the vector.
     */
    template<
        auto ID = static_cast<uint32_t>(internal::SpecialTuneableID::userDef),
        typename T = alpaka::Vec<uint32_t, 1u>>
    struct Tunable : public internal::BaseTunable<ID, T, 1u, internal::TunableKind::Tunable, std::vector<T>>
    {
        static_assert(std::is_trivially_copyable_v<T>, " tunable types must be trivially copyable!");
        using Base = internal::BaseTunable<ID, T, 1u, internal::TunableKind::Tunable, std::vector<T>>;
        using T_Storage = typename Base::StorageType;
        T_Storage values;
        using value_type = T;
        /// an index within the tuningSpace
        std::optional<uint32_t> startingIndex = std::nullopt;
        std::string m_name = internal::getDefaultName<ID, Base::tuneableType>();
        static constexpr auto dim = 1u;

        [[nodiscard]] std::string getName() const override
        {
            return m_name;
        }

        [[nodiscard]] alpaka::Vec<uint32_t, 1u> getNumValues() const override
        {
            return Vec<uint32_t, 1u>{values.size()};
        }

        T const& getValueByIndex(alpaka::Vec<uint32_t, 1u> const& idx) const
        {
            assert(idx[0u] < values.size());
            return values[idx[0u]];
        };

        // we dont need to hash the type as its included in the type of the kernelbundle
        [[nodiscard]] std::size_t valuesToHash() const
        {
            std::size_t seed = 0;
            internal::hash_combine(seed, values);
            return seed;
        }

        /**
         * @brief Construct from an initializer list of values.
         *
         * @param input List of values defining the tuning space for this parameter.
         * @param startingValue  Optional starting Value for runtime tuning.
         * @param name Optional name for this tuneable.
         */
        constexpr Tunable(
            std::initializer_list<T> input,
            std::optional<T> startingValue = std::nullopt,
            std::string const& name = "")
            : values(input)
        {
            if(!name.empty())
                m_name = name;
            if(startingValue.has_value())
            {
                this->startingIndex = assignStartingIndex(startingValue.value());
            }
        }

        /**
         * @brief Construct tuning space from an std::vector
         *
         * @param input List of values defining the tuning space for this parameter.
         * @param startingValue  Optional starting Value for runtime tuning.
         * @param name Optional name for this tuneable.
         */
        constexpr explicit Tunable(
            std::vector<T> input,
            std::optional<T> startingValue = std::nullopt,
            std::string const& name = "")
            : values(input)
        {
            if(!name.empty())
                m_name = name;
            if(startingValue.has_value())
            {
                this->startingIndex = assignStartingIndex(startingValue.value());
            }
        }

        /**
         * @brief Construct from an `IdxRange<T>` to generate a tuning space.
         *
         * might need to specify T since type deduction fails (Tunable<T>{}).
         * requires T to be  alpaka::Vec type or scalar type,
         *
         * @param input  IdxRange (m_begin<=val<=m_end (inclusive Range))
         * @param startingValue optional starting value contained within the space the range describes.
         * @param name optional string identifier (only used for history purposes)
         */
        template<alpaka::concepts::Vector U = T>
        constexpr explicit Tunable(
            alpaka::IdxRange<U, U, U> input,
            std::optional<T> startingValue = std::nullopt,
            std::string const& name = "")
        {
            for(T val = input.m_begin; internal::utils::allTrue(val <= input.m_end); val += input.m_stride)
            {
                values.push_back(val);
            }
            if(!name.empty())
                m_name = name;
            if(startingValue.has_value())
            {
                this->startingIndex = assignStartingIndex(startingValue.value());
            }
        }

    private:
        std::optional<uint32_t> assignStartingIndex(T const& startVal)
        {
            auto it = std::find(values.begin(), values.end(), startVal);
            if(it != values.end())
                return static_cast<uint32_t>(std::distance(values.begin(), it));

            std::cerr << "Warning: starting value " << startVal << " not found in tuneable '" << this->getName()
                      << "'\n";
            return std::nullopt;
        }
    };

    /**
     * @brief Used to define a Runtime tuneable.
     *
     * does expose the same interface as @see Tuneable, yet the type is restricted to the alpaka::concepts::vector
     * the alpaka vector has to have atleast 2 dimensions
     * furthermore this tuneable allows expressing tuning dimensions using a alpaka::Vector as independent tuning
     * knobs. Meaning that all dimensions are treated as a seperate parameter in the tuning process. If this behaviour
     * is unintended, simply use a normal tuneable, as its the more generic (even if you are dealing with
     * multi-dimensional vectors) Example usage:
     * @code
     * auto tune = Tunable({Vec{8,2}, Vec{6,8}, Vec{10,12}});
     *
     * auto tune1 = Tunable(IdxRange{Vec{8,2},Vec{100,20},Vec{1,1}});
     * @endcode
     *
     * @tparam ID Identifier for the tuneable ->
     * must only be modified when using this tuneable in a constraint
     * ( unless using this for frameSpec tunes -> IDs for frameTuneables found under namespace: tune::frame)
     * @tparam T Type of the values stored in the vector.
     * */
    template<
        auto ID = static_cast<uint32_t>(internal::SpecialTuneableID::userDef),
        alpaka::concepts::Vector T = alpaka::Vec<uint32_t, 2u>>
    struct TunableMD
        : public internal::BaseTunable<
              ID,
              T,
              alpaka::getDim(T{}),
              internal::TunableKind::TunableMD,
              std::array<std::vector<typename T::type>, alpaka::getDim(T{})>>
    {
        static_assert(std::is_trivially_copyable_v<T>, " tunable types must be trivially copyable!");
        static constexpr auto dim = alpaka::getDim(T{});
        using Base = internal::BaseTunable<
            ID,
            T,
            dim,
            internal::TunableKind::TunableMD,
            std::array<std::vector<typename T::type>, alpaka::getDim(T{})>>;
        using value_type = T;
        using T_Storage = typename Base::StorageType;
        using idxType = typename T::index_type;
        static_assert(dim > 0, "Given alpaka Vector - dimension must be higher then 0.");
        T_Storage values{};
        std::optional<alpaka::Vec<idxType, dim>> startingIndex = std::nullopt;
        std::string m_name = internal::getDefaultName<ID, Base::tuneableType>();

        // this is currently a value copy, coule be improved by using a reference Storage on the Vector Vec<T,dim,Ref>
        T getValueByIndex(alpaka::Vec<idxType, dim> const& vec) const
        {
            T ret{};
            for(idxType i = 0; i < dim; ++i)
            {
                assert(vec[i] < values[i].size());
                ret[i] = values[i][vec[i]];
            }
            return ret;
        };

        constexpr TunableMD() = default;

        [[nodiscard]] std::string getName() const override
        {
            return m_name;
        }

        // we dont need to hash the type as its included in the type of the kernelbundle.
        [[nodiscard]] std::size_t valuesToHash() const
        {
            std::size_t seed = 0;
            for(idxType i = 0; i < dim; ++i)
            {
                internal::hash_combine(seed, values[i]);
            }
            return seed;
        }

        /// Return the number of values in the tuning space
        Vec<idxType, dim> getNumValues() const override
        {
            Vec<idxType, dim> ret{};
            for(idxType i = 0; i < dim; ++i)
                ret[i] = values[i].size();
            return ret;
        }

        /**
         * @brief Construct from an initializer list of alpaka::Vec.
         *
         * @param input List of alpaka::Vec defining the tuning space.
         * @param startingValue  Optional starting Value for runtime tuning.
         * @param name Optional name for this tuneable.
         */
        constexpr TunableMD(
            std::initializer_list<T> input,
            std::optional<T> startingValue = std::nullopt,
            std::string const& name = "")
        {
            std::vector<T> inp(input);
            generateSortedSpaces(inp);

            if(!name.empty())
                m_name = name;
            if(startingValue.has_value())
            {
                findStartingIndex(startingValue.value());
            }
        }

        /**
         * @brief Construct from a std::vector of alpaka::Vec.
         *
         * @param input Vector of alaka::Vec defining the tuning space.
         * @param startingValue  Optional starting Value for runtime tuning.
         * @param name Optional name for this tuneable.
         */
        constexpr explicit TunableMD(
            std::vector<T> input,
            std::optional<uint32_t> startingValue = std::nullopt,
            std::string const& name = "")
        {
            generateSortedSpaces(input);
            if(!name.empty())
                m_name = name;
            if(startingValue.has_value())
            {
                findStartingIndex(startingValue.value());
            }
        }

        /**
         * @brief Construct from an `IdxRange<T>` to generate a tuning space.
         *
         * Each dimension is treated independently. For example, if
         * `start = {2,3}`, `end = {4,5}`, `stride = {3,2}`, then
         * `{2,5}` is included in the generated space.
         * If this behavior is not intended, use a normal `Tunable`.
         *
         * @param input Index range defining the start, end, and stride per dimension.
         * @param startingValue Optional starting value (has to be part of the constructed space in order to be
         * considered)
         * @param name Optional name for the tuneable.
         */
        constexpr explicit TunableMD(
            IdxRange<T> input,
            std::optional<uint32_t> startingValue = std::nullopt,
            std::string const& name = "")
        {
            // this is not a exhaustive extension. only a manhatten like traverse
            generateFromIdxRange(input);
            if(!name.empty())
                m_name = name;

            if(startingValue.has_value())
            {
                findStartingIndex(startingValue.value());
            }
        }

    private:
        void generateFromIdxRange(IdxRange<T> const& input)
        {
            for(uint32_t curDim = 0; curDim < dim; ++curDim)
            {
                for(auto cur = input.m_begin[curDim]; cur <= input.m_end[curDim]; cur += input.m_stride[curDim])
                {
                    values.at(curDim).push_back(cur);
                }
            }
        }

        void generateSortedSpaces(std::vector<T>& input)
        {
            for(uint32_t curDim = 0; curDim < dim; ++curDim)
            {
                for(auto const& vec : input)
                {
                    values.at(curDim).push_back(vec[curDim]);
                }
                sort(values.at(curDim).begin(), values.at(curDim).end());
            }
        }

        void findStartingIndex(T const& startingValue)
        {
            Vec<uint32_t, dim> startingIndex{};
            bool foundStart = true;
            for(uint32_t curDim = 0; curDim < dim && foundStart == true; ++curDim)
            {
                auto it = std::find(values.at(curDim).begin(), values.at(curDim).end(), startingValue[curDim]);

                std::optional<uint32_t> idx;
                if(it != values.at(curDim).end())
                {
                    idx = static_cast<uint32_t>(std::distance(values.at(curDim).begin(), it));
                }
                else
                {
                    idx = std::nullopt;
                }
                if(idx.has_value())
                {
                    startingIndex[curDim] = idx.value();
                }
                else
                {
                    foundStart = false;
                }
            }
            if(!foundStart)
            {
                std::cerr << " Warning: starting Value " << startingValue.toString()
                          << " for Tunable: " << this->getName() << " has to be part of its tuning space!"
                          << std::endl;
            }
            else
            {
                this->startingIndex = startingIndex;
            }
        }
    };

    /**
     * @namespace frame
     * @brief Contains predefined identifier constants for special frameSpec-related tuneables.
     * They can be used to define constraints between frame related tuneables,
     * without the need to manual define identfier.
     * @code
     * concepts::TuningSession auto session=TuningBuilder{}.with....
     * session.withConstraints<frameTune::FrameExtent,
     *  frame::ThreadBlock>([&](auto frameElems,auto threads)
     * { return frameElems%threads==0; };
     *
     */
    namespace frame
    {
        static constexpr std::size_t numBlocks(static_cast<std::size_t>(internal::SpecialTuneableID::numBlocks));
        static constexpr std::size_t numThreads(static_cast<std::size_t>(internal::SpecialTuneableID::numThreads));
        static constexpr std::size_t numFrames(static_cast<std::size_t>(internal::SpecialTuneableID::numFrames));
        static constexpr std::size_t frameExtent(static_cast<std::size_t>(internal::SpecialTuneableID::frameExtent));
    } // namespace frame

    // -----------------------------------------------------------------------------
    // NumFramesTune
    // -----------------------------------------------------------------------------
    /**
     * @brief Thin wrapper alias for TunableMD, to specify a FrameSpec tuning for the
     * "numFrames" parameter. Inherits all constructors from TunableMD.
     * Expects alpaka::Vec types.
     * Should be used in combination with CTAD.
     * Example:
     * @code
     * NumFramesTune numFramesTune{std::vector{V2u{1, 1}, V2u{2, 2}}, std::nullopt, "numFrames"};
     * NumFramesTune numFramesTune{std::vector{V2u{1, 1}, V2u{2, 2}}};
     * @endcode
     */
    template<auto ID = tune::frame::numFrames, alpaka::concepts::Vector T = alpaka::Vec<uint32_t, 2u>>
    struct NumFramesTune : TunableMD<ID, T>
    {
        using Base = TunableMD<ID, T>;
        using Base::Base; // inherit all constructors
    };
    template<class... Args>
    NumFramesTune(Args...)
        -> NumFramesTune<tune::frame::numFrames, internal::GetTunableTypeFromParameterPack<Args...>>;

    // -----------------------------------------------------------------------------
    // FrameExtentTune
    // -----------------------------------------------------------------------------
    /**
     * @brief Thin wrapper alias for TunableMD, to specify a FrameSpec tuning for the
     * "frameExtent" parameter. Inherits all constructors from TunableMD.
     * Expects alpaka::Vec types.
     * Should be used in combination with CTAD.
     * Example:
     * @code
     * FrameExtentTune frameExtentTune{std::vector{V2u{4, 4}, V2u{8, 8}}, std::nullopt, "chunkSize"};
     * FrameExtentTune frameExtentTune{std::vector{V2u{4, 4}, V2u{8, 8}}, V2u{4, 4}, "chunkSize"};
     * FrameExtentTune frameExtentTune{std::vector{V2u{4, 4}, V2u{8, 8}}};
     * @endcode
     */
    template<auto ID = tune::frame::frameExtent, alpaka::concepts::Vector T = alpaka::Vec<uint32_t, 2u>>
    struct FrameExtentTune : TunableMD<ID, T>
    {
        using Base = TunableMD<ID, T>;
        using Base::Base;
    };

    template<class... Args>
    FrameExtentTune(Args...)
        -> FrameExtentTune<tune::frame::frameExtent, internal::GetTunableTypeFromParameterPack<Args...>>;

    // -----------------------------------------------------------------------------
    // NumBlocksTune
    // -----------------------------------------------------------------------------
    /**
     * @brief Thin wrapper alias for TunableMD, to specify a FrameSpec tuning for the
     * "numBlocks" parameter. Inherits all constructors from TunableMD.
     * Expects alpaka::Vec types.
     * Should be used in combination with CTAD.
     * Example:
     * @code
     * NumBlocksTune numBlocksTune{std::vector{V2u{2, 2}, V2u{4, 4}}, std::nullopt, "numBlocks"};
     * NumBlocksTune numBlocksTune{std::vector{V2u{2, 2}, V2u{4, 4}}};
     * @endcode
     */
    template<auto ID = tune::frame::numBlocks, alpaka::concepts::Vector T = alpaka::Vec<uint32_t, 2u>>
    struct NumBlocksTune : TunableMD<ID, T>
    {
        using Base = TunableMD<ID, T>;
        using Base::Base;
    };

    template<class... Args>
    NumBlocksTune(Args...)
        -> NumBlocksTune<tune::frame::numBlocks, internal::GetTunableTypeFromParameterPack<Args...>>;

    // -----------------------------------------------------------------------------
    // NumThreadsTune
    // -----------------------------------------------------------------------------
    /**
     * @brief Thin wrapper alias for TunableMD, to specify a FrameSpec tuning for the
     * "numThreads" parameter. Inherits all constructors from TunableMD.
     * Expects alpaka::Vec types.
     * Should be used in combination with CTAD.
     * Example:
     * @code
     * NumThreadsTune numThreadsTune{std::vector{V2u{8, 8}, V2u{16, 16}}, std::nullopt, "numThreads"};
     * NumThreadsTune numThreadsTune{std::vector{V2u{8, 8}, V2u{16, 16}}};
     * @endcode
     */
    template<auto ID = tune::frame::numThreads, alpaka::concepts::Vector T = alpaka::Vec<uint32_t, 2u>>
    struct NumThreadsTune : TunableMD<ID, T>
    {
        using Base = TunableMD<ID, T>;
        using Base::Base;
    };

    template<class... Args>
    NumThreadsTune(Args...)
        -> NumThreadsTune<tune::frame::numThreads, internal::GetTunableTypeFromParameterPack<Args...>>;

} // namespace alpaka::onHost::tune

#endif // TUNEABLE_H
