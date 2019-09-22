/*
 * =====================================================================================
 *
 *        Created:  14/09/2019
 *       Revision:  none
 *       Compiler:  nvcc
 *
 *         Author:  Calum Chamberlain
 *   Organization:  EQcorrscan
 *      Copyright:  EQcorrscan developers.
 *        License:  GNU Lesser General Public License, Version 3
 *                  (https://www.gnu.org/copyleft/lesser.html)
 *
 * =====================================================================================
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#if (defined(_MSC_VER))
    #include <float.h>
    #define isnanf(x) _isnan(x)
    #define inline __inline
#endif
#if (defined(__APPLE__) && !isnanf)
    #define isnanf isnan
#endif
#if defined(__linux__) || defined(__linux) || defined(__APPLE__) || defined(__FreeBSD__) || defined(__OpenBSD__) || defined(__NetBSD__)
    #include <omp.h>
#endif
#ifndef OUTER_SAFE
    #if defined(__linux__) || defined(__linux)
        #define OUTER_SAFE 1
    #else
        #define OUTER_SAFE 0
    #endif
#else
    #define OUTER_SAFE 1
#endif
// Define minimum variance to compute correlations - requires some signal
#define ACCEPTED_DIFF 1e-10 //1e-15
// Define difference to warn user on
#define WARN_DIFF 1e-8 //1e-10

// multi_corr functions
int normxcorr_cuda_main(float*, long, long, float*, long, int, int, float*, long,
                        float*, float*, float*, fftwf_complex*, fftwf_complex*,
                        fftwf_complex*, fftwf_plan, fftwf_plan, fftwf_plan,
                        int*, int*, int, int*, int*, int);

int normxcorr_cuda_internal(
    long, long, float*, long, int, int, float*, long, long, float*, float*, float*,
    float*, fftwf_complex*, fftwf_complex*, fftwf_complex*, fftwf_plan,
    fftwf_plan, int*, int*, int, int*, int*, int, long);

int multi_normxcorr_cuda(
    float*, long, long, long, float*, long, float*, long, int*, int*, int,
    int*, int*, int);
