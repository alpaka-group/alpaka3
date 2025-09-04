/* Copyright 2024 Tapish Narwal
 * SPDX-License-Identifier: ISC
 */

#pragma once

#include "analyticalSolution.hpp"
#include "helpers.hpp"

#include <alpaka/alpaka.hpp>

//! alpaka version of explicit finite-difference 1d heat equation solver
//!
//! Applies boundary conditions
//! forward difference in t and second-order central difference in x
//!
//! \param uBuf grid values of u for each x, y and the current value of t:
//!                 u(x, y, t)  | t = t_current
//! \param pitch
//! \param dx step in x
//! \param dy step in y
//! \param dt step in t
struct BoundaryKernel
{
    template<typename TAcc>
    ALPAKA_FN_ACC auto operator()(
        TAcc const& acc,
        alpaka::concepts::MdSpan auto uBuf,
        alpaka::concepts::Vector auto numNodes,
        uint32_t step,
        double const dx,
        double const dy,
        double const dt) const -> void
    {
        using Idx = uint32_t;

        using namespace alpaka;

        // move over X surfaces
        for(auto [x] : onAcc::makeIdxMap(acc, onAcc::worker::linearThreadsInGrid, IdxRange{Vec{numNodes.x()}}))
        {
            auto nodeIdxBegin = Vec{Idx{0u}, x};
            uBuf[nodeIdxBegin] = exactSolution(nodeIdxBegin.x() * dx, nodeIdxBegin.y() * dy, step * dt);
            auto nodeIdxEnd = Vec{numNodes.y() - Idx{1u}, x};
            uBuf[nodeIdxEnd] = exactSolution(nodeIdxEnd.x() * dx, nodeIdxEnd.y() * dy, step * dt);
        }

        // move over Y surfaces, skip first and last node
        for(auto [y] :
            onAcc::makeIdxMap(acc, onAcc::worker::linearThreadsInGrid, IdxRange{Vec{1u}, Vec{numNodes.y() - 1u}}))
        {
            auto nodeIdxBegin = Vec{y, Idx{0u}};
            uBuf[nodeIdxBegin] = exactSolution(nodeIdxBegin.x() * dx, nodeIdxBegin.y() * dy, step * dt);
            auto nodeIdxEnd = Vec{y, numNodes.y() - Idx{1u}};
            uBuf[nodeIdxEnd] = exactSolution(nodeIdxEnd.x() * dx, nodeIdxEnd.y() * dy, step * dt);
        }
    }
};
