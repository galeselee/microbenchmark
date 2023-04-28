// Incorrect result
// 64TB/s on 2060 :-)

#include <cuda_runtime.h>
#include <iostream>
#include <cstdio>

const int DATA_SIZE_IN_BYTE = (128*4*256) ;
const int N_LDG = 128*256*1024;

const int WARMUP_ITER = 20000;
const int BENCH_ITER = 20000;
const int UNROLL = 256;
const int BLOCK = 128; // there are 4 warp scheduler

__device__ __forceinline__
int ldg_cg(const void *ptr) {
    int ret;
    asm volatile (
        "ld.global.ca.b32 %0, [%1];"
        : "=r"(ret)
        : "l"(ptr)
    );

    return ret;
}

template <int BLOCK, int UNROLL, int N_DATA>
__global__ void kernel(const int *x, int *y) {
    int offset = threadIdx.x % N_DATA + blockIdx.x / 1024;
    const int *ldg_ptr = x + offset;
    int reg[UNROLL];

    #pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
        reg[i] = ldg_cg(ldg_ptr + BLOCK * i);
    }

    if (*ldg_ptr != 0) {
        #pragma unroll
        for (int i = 0; i < UNROLL; ++i) {
            *y += reg[i];
        }
    }
}

int main() {
    const int N_DATA = DATA_SIZE_IN_BYTE / sizeof(int);

    // 2048

    int *x, *y;
    cudaMalloc(&x, N_DATA * sizeof(int));
    cudaMalloc(&y, N_DATA * sizeof(int));
    cudaMemset(x, 0, N_DATA * sizeof(int));
    cudaMemset(y, 0, N_DATA * sizeof(int));

    int grid = 1024;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // warm up to cache data into L2
    for (int i = 0; i < WARMUP_ITER; ++i) {
        kernel<BLOCK, UNROLL, N_DATA><<<grid, BLOCK>>>(x, y);
    }

    cudaEventRecord(start);
    for (int i = 0; i < BENCH_ITER ; ++i) {
        kernel<BLOCK, UNROLL, N_DATA><<<grid, BLOCK>>>(x, y);
    }
    cudaDeviceSynchronize();
    cudaEventRecord(stop);
    std::cout << cudaGetLastError() << std::endl;

    float time_ms = 0.f;
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time_ms, start, stop);
    double gbps = ((double)(N_LDG * sizeof(int)) / 1e9) /
                  ((double)time_ms / BENCH_ITER / 1e3);
    printf("L1 cache bandwidth: %fGB/s\n", gbps);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    cudaFree(x);
    cudaFree(y);

    return 0;
}



