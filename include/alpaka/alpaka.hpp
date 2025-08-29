/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/CVec.hpp"
#include "alpaka/Simd.hpp"
#include "alpaka/UniqueId.hpp"
#include "alpaka/Vec.hpp"
#include "alpaka/api/api.hpp"
#include "alpaka/api/cpu.hpp"
#include "alpaka/api/oneApi.hpp"
#include "alpaka/api/unifiedCudaHip.hpp"
#include "alpaka/apply.hpp"
#include "alpaka/core/Dict.hpp"
#include "alpaka/core/Tag.hpp"
#include "alpaka/core/common.hpp"
#include "alpaka/core/config.hpp"
#include "alpaka/interface.hpp"
#include "alpaka/internal.hpp"
#include "alpaka/math.hpp"
#include "alpaka/math/constants.hpp"
#include "alpaka/math/internal/Complex.hpp"
#include "alpaka/mem/Iter.hpp"
#include "alpaka/onAcc.hpp"
#include "alpaka/onAcc/Acc.hpp"
#include "alpaka/onAcc/GlobalMem.hpp"
#include "alpaka/onAcc/SimdAlgo.hpp"
#include "alpaka/onAcc/atomic.hpp"
#include "alpaka/onAcc/tag.hpp"
#include "alpaka/onHost.hpp"
#include "alpaka/onHost/Device.hpp"
#include "alpaka/onHost/DeviceSelector.hpp"
#include "alpaka/onHost/Queue.hpp"
#include "alpaka/onHost/algo/concurrent.hpp"
#include "alpaka/onHost/algo/iota.hpp"
#include "alpaka/onHost/algo/reduce.hpp"
#include "alpaka/onHost/algo/transform.hpp"
#include "alpaka/onHost/algo/transformReduce.hpp"
#include "alpaka/onHost/mem/stdContainer.hpp"
#include "alpaka/tag.hpp"
#include "utility.hpp"

#include <alpaka/onHost/demangledName.hpp>

/** main alpaka namespace.
 *
 * namespace onHost::* contains all functionality which is usable on the host CPU controller thread.
 * namespace onAcc::* contains all functionality which is usable on the accelerator compute device from within a
 * kernel. namespace alpaka contains all functionality which is generic and can be used from within the host controller
 * thread and within compute device kernels.
 */
namespace alpaka
{
}
