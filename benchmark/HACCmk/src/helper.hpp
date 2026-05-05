/* Copyright (c) 2020 , Argonne National Laboratory   (Zheming Jin)
 * Copyright (c) 2020-, Oak Ridge National Laboratory   (Zheming Jin)
 * Copyright 2026 René Widera
 * SPDX-License-Identifier: BSD-3-Clause
 */


/** @file The file is providing some helpers for the HACCmk implementation shared between the original and the alpaka
 * implementation.
 */

#include <math.h>
#include <stdio.h>

namespace hacc
{
    void haccmk_gold(
        int count1,
        float xxi,
        float yyi,
        float zzi,
        float fsrrmax2,
        float mp_rsm2,
        auto xx1,
        auto yy1,
        auto zz1,
        auto mass1,
        auto dxi,
        auto dyi,
        auto dzi)
    {
        float const ma0 = 0.269327f, ma1 = -0.0750978f, ma2 = 0.0114808f, ma3 = -0.00109313f, ma4 = 0.0000605491f,
                    ma5 = -0.00000147177f;


        float dxc, dyc, dzc, m, r2, f, xi, yi, zi;

        xi = 0.f;
        yi = 0.f;
        zi = 0.f;

        for(int j = 0; j < count1; j++)
        {
            dxc = xx1[j] - xxi;
            dyc = yy1[j] - yyi;
            dzc = zz1[j] - zzi;

            r2 = dxc * dxc + dyc * dyc + dzc * dzc;

            if(r2 < fsrrmax2)
                m = mass1[j];
            else
                m = 0.f;

            f = r2 + mp_rsm2;
            f = m * (1.f / (f * sqrtf(f)) - (ma0 + r2 * (ma1 + r2 * (ma2 + r2 * (ma3 + r2 * (ma4 + r2 * ma5))))));

            xi = xi + f * dxc;
            yi = yi + f * dyc;
            zi = zi + f * dzc;
        }

        *dxi = xi;
        *dyi = yi;
        *dzi = zi;
    }

    int verify(int n2, auto vx2, auto vy2, auto vz2, auto vx2_hw, auto vy2_hw, auto vz2_hw)
    {
        int error = 0;
        float const eps = 1.0f;
        for(int i = 0; i < n2; i++)
        {
            if(fabsf(vx2[i] - vx2_hw[i]) > eps)
            {
                printf("error at vx2[%d] %f %f\n", i, vx2[i], vx2_hw[i]);
                error = 1;
                break;
            }
            if(fabsf(vy2[i] - vy2_hw[i]) > eps)
            {
                printf("error at vy2[%d]: %f %f\n", i, vy2[i], vy2_hw[i]);
                error = 1;
                break;
            }
            if(fabsf(vz2[i] - vz2_hw[i]) > eps)
            {
                printf("error at vz2[%d]: %f %f\n", i, vz2[i], vz2_hw[i]);
                error = 1;
                break;
            }
        }
        printf("%s\n", error ? "Verification FAILED" : "Verification PASSED");
        return error;
    }

} // namespace hacc
