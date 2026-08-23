#pragma once

#include "common.cuh"

#define QR_TURBO3 1
#define QR_TURBO4 1

static __constant__ float TURBO3_CENTROIDS[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f,
};

static __constant__ float TURBO3_MIDPOINTS[7] = {
    -0.154259f, -0.091775f, -0.043589f, 0.0f,
     0.043589f,  0.091775f,  0.154259f,
};

static __constant__ float TURBO3_WHT_SIGNS_1[QK_TURBO3_GROUP] = {
    -1,1,1,-1,-1,1,-1,1,-1,-1,1,1,1,1,1,1,1,-1,1,-1,1,-1,-1,1,1,1,-1,1,1,-1,-1,-1,
    -1,1,1,-1,1,1,-1,1,-1,1,1,-1,-1,1,-1,1,1,1,1,-1,-1,-1,-1,-1,1,-1,1,1,1,1,-1,1,
    -1,-1,1,-1,-1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,1,-1,-1,1,1,1,-1,-1,1,1,-1,1,1,-1,1,-1,
    -1,1,1,-1,1,-1,1,-1,1,1,1,1,-1,1,-1,1,1,-1,1,1,-1,-1,-1,-1,-1,1,1,-1,1,1,-1,1,
};

static __constant__ float TURBO3_WHT_SIGNS_2[QK_TURBO3_GROUP] = {
    1,1,1,1,-1,1,1,-1,1,-1,-1,-1,1,-1,-1,-1,1,1,-1,-1,1,-1,1,-1,1,-1,-1,1,-1,1,1,1,
    1,1,-1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,1,-1,1,-1,1,1,1,-1,-1,1,-1,-1,-1,-1,-1,-1,1,1,
    1,-1,1,-1,-1,-1,-1,1,-1,1,-1,1,-1,-1,1,1,-1,1,-1,1,1,-1,1,-1,-1,-1,-1,1,-1,-1,1,-1,
    1,-1,1,1,1,-1,-1,1,-1,1,-1,1,1,-1,-1,1,-1,1,-1,1,1,-1,1,-1,1,-1,-1,-1,-1,-1,1,-1,
};

static __device__ __forceinline__ uint8_t turbo3_nearest(float x) {
    if (x < TURBO3_MIDPOINTS[0]) return 0;
    if (x < TURBO3_MIDPOINTS[1]) return 1;
    if (x < TURBO3_MIDPOINTS[2]) return 2;
    if (x < TURBO3_MIDPOINTS[3]) return 3;
    if (x < TURBO3_MIDPOINTS[4]) return 4;
    if (x < TURBO3_MIDPOINTS[5]) return 5;
    if (x < TURBO3_MIDPOINTS[6]) return 6;
    return 7;
}

static __device__ __forceinline__ float turbo3_dequant(
        const block_turbo3_0 * block, int index, float norm) {
    const uint8_t low = (block->qs[index/4] >> (2*(index % 4))) & 3;
    const uint8_t high = (block->signs[index/8] >> (index % 8)) & 1;
    return norm*TURBO3_CENTROIDS[low | (high << 2)];
}

static __constant__ float TURBO4_CENTROIDS[16] = {
    -0.173926f, -0.117195f, -0.089527f, -0.068756f,
    -0.051262f, -0.035597f, -0.020989f, -0.006938f,
     0.006938f,  0.020989f,  0.035597f,  0.051262f,
     0.068756f,  0.089527f,  0.117195f,  0.173926f,
};

static __constant__ float TURBO4_MIDPOINTS[15] = {
    -0.145561f, -0.103361f, -0.079142f, -0.060009f,
    -0.043430f, -0.028293f, -0.013964f,  0.0f,
     0.013964f,  0.028293f,  0.043430f,  0.060009f,
     0.079142f,  0.103361f,  0.145561f,
};

static __device__ __forceinline__ uint8_t turbo4_nearest(float x) {
    if (x < TURBO4_MIDPOINTS[ 0]) return  0;
    if (x < TURBO4_MIDPOINTS[ 1]) return  1;
    if (x < TURBO4_MIDPOINTS[ 2]) return  2;
    if (x < TURBO4_MIDPOINTS[ 3]) return  3;
    if (x < TURBO4_MIDPOINTS[ 4]) return  4;
    if (x < TURBO4_MIDPOINTS[ 5]) return  5;
    if (x < TURBO4_MIDPOINTS[ 6]) return  6;
    if (x < TURBO4_MIDPOINTS[ 7]) return  7;
    if (x < TURBO4_MIDPOINTS[ 8]) return  8;
    if (x < TURBO4_MIDPOINTS[ 9]) return  9;
    if (x < TURBO4_MIDPOINTS[10]) return 10;
    if (x < TURBO4_MIDPOINTS[11]) return 11;
    if (x < TURBO4_MIDPOINTS[12]) return 12;
    if (x < TURBO4_MIDPOINTS[13]) return 13;
    if (x < TURBO4_MIDPOINTS[14]) return 14;
    return 15;
}

static __device__ __forceinline__ float turbo4_dequant(
        const block_turbo4_0 * block, int index, float norm) {
    const uint8_t index_ = (block->qs[index/2] >> (4*(index % 2))) & 0xF;
    return norm*TURBO4_CENTROIDS[index_];
}
