/* Copyright 2025 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#pragma once

#include "alpaka/core/PP.hpp"

#if ALPAKA_LANG_HIP

#    include <hip/hip_version.h>

// version numbers are only defined on the device side
#    if !defined(ALPAKA_AMDGPU_ARCH) && defined(__HIP__) && defined(__HIP_DEVICE_COMPILE__)                           \
        && __HIP_DEVICE_COMPILE__ == 1

/* Map AMDGPU arch macro -> ALPAKA_VRRR_TO_VERSION(wrapped code)
 *  Rules:
 *   - gfx9xx (numeric): 9xy -> 90xy  (e.g., 908->9008, 906->9006, 942->9042)
 *   - gfx10xx / gfx11xx: stxy -> st0xy (e.g., 1036->10036, 1103->11003)
 *   - Suffix: a->+100 (90a->9100), b->+200, c->+300
 */

#        if defined(__gfx1153__)
/* RDNA 3.5 iGPU (Medusa Point / Strix Halo successor) */
#            define ALPAKA_AMDGPU_ARCH ALPAKA_VRRR_TO_VERSION(11053)
#        elif defined(__gfx1152__)
/* RDNA 3.5 iGPU (Krackan Point) */
#            define ALPAKA_AMDGPU_ARCH ALPAKA_VRRR_TO_VERSION(11052)
#        elif defined(__gfx1151__)
/* RDNA 3.5 iGPU (Strix Halo) */
#            define ALPAKA_AMDGPU_ARCH ALPAKA_VRRR_TO_VERSION(11051)
#        elif defined(__gfx1150__)
/* RDNA 3.5 iGPU (Radeon 890M on Strix Point) */
#            define ALPAKA_AMDGPU_ARCH ALPAKA_VRRR_TO_VERSION(11050)

#        elif defined(__gfx1103__)
/* RDNA 3 APU (Radeon 780M, 760M, ROG Ally Extreme) */
#            define ALPAKA_AMDGPU_ARCH ALPAKA_VRRR_TO_VERSION(11003)
#        elif defined(__gfx1102__)
/* RDNA 3 Desktop (RX 7600 / 7600 XT) */
#            define ALPAKA_AMDGPU_ARCH ALPAKA_VRRR_TO_VERSION(11002)
#        elif defined(__gfx1101__)
/* RDNA 3 Desktop (RX 7700 / 7700 XT, Pro W7700 / V710) */
#            define ALPAKA_AMDGPU_ARCH ALPAKA_VRRR_TO_VERSION(11001)
#        elif defined(__gfx1100__)
/* RDNA 3 Desktop (RX 7900 XT, XTX, Pro W7900) */
#            define ALPAKA_AMDGPU_ARCH ALPAKA_VRRR_TO_VERSION(11000)

#        elif defined(__gfx1036__)
/* RDNA 2 APU (Radeon Graphics 128-SP iGPU) */
#            define ALPAKA_AMDGPU_ARCH ALPAKA_VRRR_TO_VERSION(10036)
#        elif defined(__gfx1035__)
/* RDNA 2 APU (Radeon 660M, 680M) */
#            define ALPAKA_AMDGPU_ARCH ALPAKA_VRRR_TO_VERSION(10035)
#        elif defined(__gfx1034__)
/* RDNA 2 Mobile (Pro W6300/W6400, RX 6400-6500) */
#            define ALPAKA_AMDGPU_ARCH ALPAKA_VRRR_TO_VERSION(10034)
#        elif defined(__gfx1033__)
/* RDNA 2 APU (Steam Deck) */
#            define ALPAKA_AMDGPU_ARCH ALPAKA_VRRR_TO_VERSION(10033)
#        elif defined(__gfx1032__)
/* RDNA 2 Desktop (RX 6600 XT, 6650 XT/S, 6700S) */
#            define ALPAKA_AMDGPU_ARCH ALPAKA_VRRR_TO_VERSION(10032)
#        elif defined(__gfx1031__)
/* RDNA 2 Desktop (RX 6700 series, 6750/6850M XT) */
#            define ALPAKA_AMDGPU_ARCH ALPAKA_VRRR_TO_VERSION(10031)
#        elif defined(__gfx1030__)
/* RDNA 2 Desktop (RX 6800 / 6900 XT, Pro W6800) */
#            define ALPAKA_AMDGPU_ARCH ALPAKA_VRRR_TO_VERSION(10030)

#        elif defined(__gfx1013__)
/* RDNA 1 Mobile (RX 5300M / 5500M) */
#            define ALPAKA_AMDGPU_ARCH ALPAKA_VRRR_TO_VERSION(10013)
#        elif defined(__gfx1012__)
/* RDNA 1 Desktop (RX 5500 / 5500 XT) */
#            define ALPAKA_AMDGPU_ARCH ALPAKA_VRRR_TO_VERSION(10012)
#        elif defined(__gfx1011__)
/* RDNA 1 Desktop (Pro V520) */
#            define ALPAKA_AMDGPU_ARCH ALPAKA_VRRR_TO_VERSION(10011)
#        elif defined(__gfx1010__)
/* RDNA 1 Desktop (RX 5700 / 5700 XT, Pro 5600 XT/M) */
#            define ALPAKA_AMDGPU_ARCH ALPAKA_VRRR_TO_VERSION(10010)

#        elif defined(__gfx942__)
/* CDNA 3 (Instinct MI300 series: MI300/MI300A/MI300X) */
#            define ALPAKA_AMDGPU_ARCH ALPAKA_VRRR_TO_VERSION(9042)
#        elif defined(__gfx941__)
/* CDNA 2/3 (Instinct MI210) */
#            define ALPAKA_AMDGPU_ARCH ALPAKA_VRRR_TO_VERSION(9041)
#        elif defined(__gfx940__)
/* CDNA 2 (Instinct MI200) */
#            define ALPAKA_AMDGPU_ARCH ALPAKA_VRRR_TO_VERSION(9040)

#        elif defined(__gfx90c__)
/* CDNA 1 (Renoir APUs), c -> +300 */
#            define ALPAKA_AMDGPU_ARCH ALPAKA_VRRR_TO_VERSION(9300)
#        elif defined(__gfx90b__)
/* (If present) b -> +200 */
#            define ALPAKA_AMDGPU_ARCH ALPAKA_VRRR_TO_VERSION(9200)
#        elif defined(__gfx90a__)
/* CDNA 2 (Instinct MI250 / MI250X), a -> +100 */
#            define ALPAKA_AMDGPU_ARCH ALPAKA_VRRR_TO_VERSION(9100)
#        elif defined(__gfx908__)
/* CDNA 1 (Instinct MI100) */
#            define ALPAKA_AMDGPU_ARCH ALPAKA_VRRR_TO_VERSION(9008)
#        elif defined(__gfx906__)
/* Vega 20 (Radeon VII, Instinct MI50/60) */
#            define ALPAKA_AMDGPU_ARCH ALPAKA_VRRR_TO_VERSION(9006)

#        else
#            warning                                                                                                  \
                "Unknown AMDGPU architecture, please define __gfxXXX__ macro for your target. Until alpaka is updated you can define the macro ALPAKA_AMDGPU_ARCH to avoid this warning."
#            define ALPAKA_AMDGPU_ARCH ALPAKA_VERSION_NUMBER_UNKNOWN
#        endif

#    endif
#endif
