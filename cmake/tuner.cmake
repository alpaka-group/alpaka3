# ==========================================================
# Tuner CMake configuration
# ==========================================================

## === JSON SUPPORT ===
option(ALPAKA_TUNE_DISABLE_JSON "Disable use of nlohmann/json in Alpaka Tuner" OFF)

if(NOT ALPAKA_TUNE_DISABLE_JSON)
    # Fetch and install nlohmann/json if not already fetched
    if(NOT DEFINED _NLOHMANN_JSON_FETCHED OR NOT _NLOHMANN_JSON_FETCHED)
        message(STATUS "Installing nlohmann/json library (this may take a few seconds)...")
    endif()

    include(FetchContent)

    FetchContent_Declare(
        nlohmann_json
        GIT_REPOSITORY https://github.com/nlohmann/json.git
        GIT_TAG
            v3.12.0 # or latest stable
        GIT_SHALLOW TRUE
    )
    FetchContent_MakeAvailable(nlohmann_json)

    if(NOT DEFINED _NLOHMANN_JSON_FETCHED OR NOT _NLOHMANN_JSON_FETCHED)
        message(STATUS "Successfully installed nlohmann/json library")
        set(_NLOHMANN_JSON_FETCHED ON CACHE INTERNAL "Flag to avoid fetching nlohmann/json multiple times")
    endif()

    target_link_libraries(alpaka_target_headers INTERFACE nlohmann_json::nlohmann_json)
    target_compile_definitions(alpaka_target_headers INTERFACE ALPAKA_TUNE_HAS_JSON=1)
else()
    message(STATUS "Building without JSON support (ALPAKA_TUNE_DISABLE_JSON=ON)")
    target_compile_definitions(alpaka_target_headers INTERFACE ALPAKA_TUNE_HAS_JSON=0 ALPAKA_TUNE_DISABLE_JSON)
endif()

# ==========================================================
# === STRATEGY SELECTION ===
# ==========================================================
# Allow user to set strategies using cmake options
set(alpaka_Tuner_Strategy "randomExplore" CACHE STRING "Tuning strategy strategy selection")
set_property(CACHE alpaka_Tuner_Strategy PROPERTY STRINGS "randomExplore, exhaustive,randomSample")
# Strategy selection macro
if(alpaka_Tuner_Strategy STREQUAL "randomExplore")
    set(DTUNER_STRATEGY_MACRO strategy_randomSearch)
elseif(DTuner_Strategy STREQUAL "exhaustive")
    set(DTUNER_STRATEGY_MACRO strategy_exhaustiveSearch)
elseif(DTuner_Strategy STREQUAL "simulatedAnnealing")
    set(DTUNER_STRATEGY_MACRO strategy_simulatedAnnealing)
    message(
        FATAL_ERROR
        "The 'simulated annealing' strategy is not yet supported in this build.\n"
        "Please use a different strategy (e.g. randomExplore, exhaustive  or randomSample)."
    )
elseif(DTuner_Strategy STREQUAL "randomSample")
    set(DTUNER_STRATEGY_MACRO strategy_randomSample)
elseif(DTuner_Strategy STREQUAL "bayesianOptimization")
    set(DTUNER_STRATEGY_MACRO strategy_bayesianOptimization)
    message(
        FATAL_ERROR
        "The 'bayesianOptimization' strategy is not yet supported in this build.\n"
        "Please use a different strategy (e.g. randomExplore, exhaustive  or randomSample)."
    )
    if(NOT DEFINED _EIGEN_FETCHED OR NOT _EIGEN_FETCHED)
        include(FetchContent)

        if(NOT eigen_POPULATED)
            message(STATUS "Fetching Eigen library for Bayesian Optimization strategy...")
            FetchContent_Declare(
                eigen
                GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
                GIT_TAG 3.4.0
                GIT_SHALLOW TRUE
            )
            FetchContent_MakeAvailable(eigen)
            set(_EIGEN_FETCHED ON CACHE INTERNAL "Flag to avoid fetching Eigen multiple times")
            message(STATUS "Successfully installed Eigen library")
        endif()
    endif()
else()
    message(FATAL_ERROR "Invalid DTuner_Strategy: ${DTuner_Strategy}")
endif()

target_compile_definitions(alpaka_target_headers INTERFACE ${DTUNER_STRATEGY_MACRO})
