/* Copyright 2023 Andrea Bocci, Aurora Perego, René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include "alpaka/core/config.hpp"

#if ALPAKA_LANG_SYCL

#    if defined(__SYCL_DEVICE_ONLY__)

// defines can be taken from
// https://github.com/llvm/llvm-project/blob/3cfe6aa46e06a8caa3f07057838d31c6ce840076/clang/include/clang/Basic/OffloadArch.h#L18-L28

/* Intel GPU and CPU subgroup size detection */
#        if (/* Broadwell Intel graphics architecture */ __SYCL_TARGET_INTEL_GPU_BDW__)                               \
            || (/* Skylake Intel graphics architecture */ __SYCL_TARGET_INTEL_GPU_SKL__)                              \
            || (/* Kaby Lake Intel graphics architecture */ __SYCL_TARGET_INTEL_GPU_KBL__)                            \
            || (/* Coffee Lake Intel graphics architecture */ __SYCL_TARGET_INTEL_GPU_CFL__)                          \
            || (/* Apollo Lake Intel graphics architecture */ __SYCL_TARGET_INTEL_GPU_APL__)                          \
            || (/* Gemini Lake Intel graphics architecture */ __SYCL_TARGET_INTEL_GPU_GLK__)                          \
            || (/* Whiskey Lake Intel graphics architecture */ __SYCL_TARGET_INTEL_GPU_WHL__)                         \
            || (/* Amber Lake Intel graphics architecture */ __SYCL_TARGET_INTEL_GPU_AML__)                           \
            || (/* Comet Lake Intel graphics architecture */ __SYCL_TARGET_INTEL_GPU_CML__)                           \
            || (/* Ice Lake Intel graphics architecture */ __SYCL_TARGET_INTEL_GPU_ICLLP__)                           \
            || (/* Elkhart Lake or Jasper Lake Intel graphics architecture */ __SYCL_TARGET_INTEL_GPU_EHL__)          \
            || (/* Tiger Lake Intel graphics architecture */ __SYCL_TARGET_INTEL_GPU_TGLLP__)                         \
            || (/* Rocket Lake Intel graphics architecture */ __SYCL_TARGET_INTEL_GPU_RKL__)                          \
            || (/* Alder Lake S or Raptor Lake S Intel graphics architecture */ __SYCL_TARGET_INTEL_GPU_ADL_S__)      \
            || (/* Alder Lake P Intel graphics architecture */ __SYCL_TARGET_INTEL_GPU_ADL_P__)                       \
            || (/* Alder Lake N Intel graphics architecture */ __SYCL_TARGET_INTEL_GPU_ADL_N__)                       \
            || (/* DG1 Intel graphics architecture */ __SYCL_TARGET_INTEL_GPU_DG1__)                                  \
            || (/* Alchemist G10 Intel graphics architecture */ __SYCL_TARGET_INTEL_GPU_ACM_G10__)                    \
            || (/* Alchemist G11 Intel graphics architecture */ __SYCL_TARGET_INTEL_GPU_ACM_G11__)                    \
            || (/* Alchemist G12 Intel graphics architecture */ __SYCL_TARGET_INTEL_GPU_ACM_G12__)                    \
            || (/* Meteor Lake U/S or Arrow Lake U/S Intel graphics architecture */ __SYCL_TARGET_INTEL_GPU_MTL_U__)  \
            || (/* Meteor Lake H Intel graphics architecture */ __SYCL_TARGET_INTEL_GPU_MTL_H__)                      \
            || (/* Arrow Lake H Intel graphics architecture */ __SYCL_TARGET_INTEL_GPU_ARL_H__)                       \
            || (/* Battlemage G21 Intel graphics architecture */ __SYCL_TARGET_INTEL_GPU_BMG_G21__)                   \
            || (/* Lunar Lake M Intel graphics architecture */ __SYCL_TARGET_INTEL_GPU_LNL_M__)
#            define SYCL_SUBGROUP_SIZE (8 | 16 | 32)

#        elif (/* Ponte Vecchio Intel graphics architecture */ __SYCL_TARGET_INTEL_GPU_PVC__)                         \
            || (/* Ponte Vecchio VG Intel graphics architecture */ __SYCL_TARGET_INTEL_GPU_PVC_VG__)
#            define SYCL_SUBGROUP_SIZE (16 | 32)

#        elif (/* x86_64 CPUs */ __SYCL_TARGET_INTEL_X86_64__)
// @attention ony CPU side detachment of SYCL kernel we limit the CPU currently to max warp group size of 32, therefore
// 64 is removed from this list
#            define SYCL_SUBGROUP_SIZE (1 | 2 | 4 | 8 | 16 | 32)

#        elif (/* NVIDIA Maxwell architecture (compute capability 5.0) */ __SYCL_TARGET_NVIDIA_GPU_SM_50__)           \
            || (/* NVIDIA Maxwell architecture (compute capability 5.2) */ __SYCL_TARGET_NVIDIA_GPU_SM_52__)          \
            || (/* NVIDIA Jetson TX1 / Nano (compute capability 5.3) */ __SYCL_TARGET_NVIDIA_GPU_SM_53__)             \
            || (/* NVIDIA Pascal architecture (compute capability 6.0) */ __SYCL_TARGET_NVIDIA_GPU_SM_60__)           \
            || (/* NVIDIA Pascal architecture (compute capability 6.1) */ __SYCL_TARGET_NVIDIA_GPU_SM_61__)           \
            || (/* NVIDIA Jetson TX2 (compute capability 6.2) */ __SYCL_TARGET_NVIDIA_GPU_SM_62__)                    \
            || (/* NVIDIA Volta architecture (compute capability 7.0) */ __SYCL_TARGET_NVIDIA_GPU_SM_70__)            \
            || (/* NVIDIA Jetson AGX Xavier (compute capability 7.2) */ __SYCL_TARGET_NVIDIA_GPU_SM_72__)             \
            || (/* NVIDIA Turing architecture (compute capability 7.5) */ __SYCL_TARGET_NVIDIA_GPU_SM_75__)           \
            || (/* NVIDIA Ampere architecture (compute capability 8.0) */ __SYCL_TARGET_NVIDIA_GPU_SM_80__)           \
            || (/* NVIDIA Ampere architecture (compute capability 8.6) */ __SYCL_TARGET_NVIDIA_GPU_SM_86__)           \
            || (/* NVIDIA Jetson/Drive AGX Orin (compute capability 8.7) */ __SYCL_TARGET_NVIDIA_GPU_SM_87__)         \
            || (/* NVIDIA Ada Lovelace architecture (compute capability 8.9) */ __SYCL_TARGET_NVIDIA_GPU_SM_89__)     \
            || (/* NVIDIA Hopper architecture (compute capability 9.0) */ __SYCL_TARGET_NVIDIA_GPU_SM_90__)           \
            || (/* NVIDIA Hopper architecture variant (compute capability 9.0a) */ __SYCL_TARGET_NVIDIA_GPU_SM_90a__) \
            || (/* NVIDIA Blackwell architecture (compute capability 10.0) */ __SYCL_TARGET_NVIDIA_GPU_SM_100__)      \
            || (/* NVIDIA Blackwell architecture variant (compute capability 10.0a) */                                \
                __SYCL_TARGET_NVIDIA_GPU_SM_100a__)                                                                   \
            || (/* NVIDIA Blackwell Next architecture (compute capability 10.1) */ __SYCL_TARGET_NVIDIA_GPU_SM_101__) \
            || (/* NVIDIA Blackwell Next architecture variant (compute capability 10.1a) */                           \
                __SYCL_TARGET_NVIDIA_GPU_SM_101a__)                                                                   \
            || (/* NVIDIA Next-generation architecture (compute capability 10.3) */                                   \
                __SYCL_TARGET_NVIDIA_GPU_SM_103__)                                                                    \
            || (/* NVIDIA Next-generation architecture variant (compute capability 10.3a) */                          \
                __SYCL_TARGET_NVIDIA_GPU_SM_103a__)                                                                   \
            || (/* NVIDIA Future architecture (compute capability 12.0) */ __SYCL_TARGET_NVIDIA_GPU_SM_120__)         \
            || (/* NVIDIA Future architecture variant (compute capability 12.0a) */                                   \
                __SYCL_TARGET_NVIDIA_GPU_SM_120a__)                                                                   \
            || (/* NVIDIA Future architecture (compute capability 12.1) */ __SYCL_TARGET_NVIDIA_GPU_SM_121__)         \
            || (/*NVIDIA Future architecture variant (compute capability 12.1a) */                                    \
                __SYCL_TARGET_NVIDIA_GPU_SM_121a__)

#            define SYCL_SUBGROUP_SIZE (32) /* CUDA supports warp size 32 */

#        elif (/* AMD GCN 5.0 Vega architecture (gfx 9.0) */ __SYCL_TARGET_AMD_GPU_GFX900__)                          \
            || (/* AMD GCN 5.0 Vega architecture (gfx 9.0) */ __SYCL_TARGET_AMD_GPU_GFX902__)                         \
            || (/* AMD GCN 5.0 Vega architecture (gfx 9.0) */ __SYCL_TARGET_AMD_GPU_GFX904__)                         \
            || (/* AMD GCN 5.1 Vega II architecture (gfx 9.0) */ __SYCL_TARGET_AMD_GPU_GFX906__)                      \
            || (/* AMD CDNA 1.0 Arcturus architecture (gfx 9.0) */ __SYCL_TARGET_AMD_GPU_GFX908__)                    \
            || (/* AMD GCN 5.0 Raven 2 architecture (gfx 9.0) */ __SYCL_TARGET_AMD_GPU_GFX909__)                      \
            || (/* AMD CDNA 2.0 Aldebaran architecture (gfx 9.0) */ __SYCL_TARGET_AMD_GPU_GFX90A__)                   \
            || (/* AMD GCN 5.1 Renoir architecture (gfx 9.0) */ __SYCL_TARGET_AMD_GPU_GFX90C__)                       \
            || (/* AMD CDNA 3.x generic architecture (gfx 9.4) */ __SYCL_TARGET_AMD_GPU_GFX9_4_GENERIC__)             \
            || (/* AMD CDNA 3.0 Aqua Vanjaram architecture (gfx 9.4) */ __SYCL_TARGET_AMD_GPU_GFX940__)               \
            || (/* AMD CDNA 3.0 Aqua Vanjaram architecture (gfx 9.4) */ __SYCL_TARGET_AMD_GPU_GFX941__)               \
            || (/* AMD CDNA 3.0 Aqua Vanjaram architecture (gfx 9.4) */ __SYCL_TARGET_AMD_GPU_GFX942__)               \
            || (/* AMD CDNA 3.5 derivative architecture (gfx 9.5) */ __SYCL_TARGET_AMD_GPU_GFX950__)                  \
            || (/* AMD GCN 5.x generic architecture (gfx 9.x) */ __SYCL_TARGET_AMD_GPU_GFX9_GENERIC__)

#            define SYCL_SUBGROUP_SIZE (64) /* up to gfx9, HIP supports wavefront size 64 */

#        elif (/* AMD RDNA 1.0 Navi 10 architecture (gfx 10.1) */ __SYCL_TARGET_AMD_GPU_GFX1010__)                    \
            || (/* AMD RDNA 1.0 Navi 12 architecture (gfx 10.1) */ __SYCL_TARGET_AMD_GPU_GFX1011__)                   \
            || (/* AMD RDNA 1.0 Navi 14 architecture (gfx 10.1) */ __SYCL_TARGET_AMD_GPU_GFX1012__)                   \
            || (/* AMD RDNA 2.0 Oberon architecture (gfx 10.1) */ __SYCL_TARGET_AMD_GPU_GFX1013__)                    \
            || (/* AMD RDNA 1.x generic architecture (gfx 10.1) */ __SYCL_TARGET_AMD_GPU_GFX10_1_GENERIC__)           \
            || (/* AMD RDNA 2.0 Navi 21 architecture (gfx 10.3) */ __SYCL_TARGET_AMD_GPU_GFX1030__)                   \
            || (/* AMD RDNA 2.0 Navi 22 architecture (gfx 10.3) */ __SYCL_TARGET_AMD_GPU_GFX1031__)                   \
            || (/* AMD RDNA 2.0 Navi 23 architecture (gfx 10.3) */ __SYCL_TARGET_AMD_GPU_GFX1032__)                   \
            || (/* AMD RDNA 2.0 Van Gogh architecture (gfx 10.3) */ __SYCL_TARGET_AMD_GPU_GFX1033__)                  \
            || (/* AMD RDNA 2.0 Navi 24 architecture (gfx 10.3) */ __SYCL_TARGET_AMD_GPU_GFX1034__)                   \
            || (/* AMD RDNA 2.0 Rembrandt Mobile architecture (gfx 10.3) */ __SYCL_TARGET_AMD_GPU_GFX1035__)          \
            || (/* AMD RDNA 2.0 Raphael architecture (gfx 10.3) */ __SYCL_TARGET_AMD_GPU_GFX1036__)                   \
            || (/* AMD RDNA 2.x generic architecture (gfx 10.3) */ __SYCL_TARGET_AMD_GPU_GFX10_3_GENERIC__)           \
            || (/* AMD RDNA 3.0 Navi 31 architecture (gfx 11.0) */ __SYCL_TARGET_AMD_GPU_GFX1100__)                   \
            || (/* AMD RDNA 3.0 Navi 32 architecture (gfx 11.0) */ __SYCL_TARGET_AMD_GPU_GFX1101__)                   \
            || (/* AMD RDNA 3.0 Navi 33 architecture (gfx 11.0) */ __SYCL_TARGET_AMD_GPU_GFX1102__)                   \
            || (/* AMD RDNA 3.0 Phoenix mobile architecture (gfx 11.0) */ __SYCL_TARGET_AMD_GPU_GFX1103__)            \
            || (/* AMD RDNA 3.x generic architecture (gfx 11.x) */ __SYCL_TARGET_AMD_GPU_GFX11_GENERIC__)             \
            || (/* AMD RDNA 3.5 Strix Point architecture (gfx 11.5) */ __SYCL_TARGET_AMD_GPU_GFX1150__)               \
            || (/* AMD RDNA 3.5 Strix Halo architecture (gfx 11.5) */ __SYCL_TARGET_AMD_GPU_GFX1151__)                \
            || (/* AMD RDNA 3.5 derivative architecture (gfx 11.5) */ __SYCL_TARGET_AMD_GPU_GFX1152__)                \
            || (/* AMD RDNA 3.5 derivative architecture (gfx 11.5) */ __SYCL_TARGET_AMD_GPU_GFX1153__)                \
            || (/* AMD RDNA 4.0 Navi 44 architecture (gfx 12.0) */ __SYCL_TARGET_AMD_GPU_GFX1200__)                   \
            || (/* AMD RDNA 4.0 Navi 48 architecture (gfx 12.0) */ __SYCL_TARGET_AMD_GPU_GFX1201__)                   \
            || (/* AMD RDNA 4.x generic architecture (gfx 12.x) */ __SYCL_TARGET_AMD_GPU_GFX12_GENERIC__)             \
            || (/* AMD RDNA 4.5 derivative architecture (gfx 12.5) */ __SYCL_TARGET_AMD_GPU_GFX1250__)                \
            || (/* AMD RDNA 4.5 derivative architecture (gfx 12.5) */ __SYCL_TARGET_AMD_GPU_GFX1251__)

#            define SYCL_SUBGROUP_SIZE (32) /* starting from gfx10, HIP supports wavefront size 32 */

#        else // __SYCL_TARGET_*

// if we do not compile ahead of time for a device and use e.g. -fsycl-targets=spir64 we need to accept all possible
// variants
#            define SYCL_SUBGROUP_SIZE (0xFFFF'FFFF) /* unknown target */

#        endif // __SYCL_TARGET_*

#    else

// ony the host side we need to allow all possible variants of a subgroup size else kernel will not be build
#        define SYCL_SUBGROUP_SIZE (0xFFFF'FFFF) /* host compilation */

#    endif // __SYCL_DEVICE_ONLY__

#endif
