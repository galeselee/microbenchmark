// 3000 GB/s L1 cache
// 2060
#include <cuda_runtime.h>
#include <iostream>
#include <string>
#include <cstdio>

const int DATA_SIZE_IN_BYTE = (1lu << 12) ;
const int N_LDG = 64 * 256;

const int WARMUP_ITER = 200;
const int BENCH_ITER = 200;
const int UNROLL = 64;
const int BLOCK = 256; // there are 4 warp scheduler

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
__global__ void kernel(const int *x, int *y, uint32_t* clock_d) {
    int offset = threadIdx.x;
    const int *ldg_ptr = x + offset;
    int reg[UNROLL];

    #pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
        reg[i] = ldg_cg(ldg_ptr + BLOCK * i);
    }

    int sum = 0;
    #pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
        sum += reg[i];
    }
    
    uint32_t start, end;

    asm volatile(
        "bar.sync 0;\n"
        "mov.u32 %0, %%clock;\n"
        : "=r"(start)
        :
        : "memory"
    );
    #pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
        reg[i] = ldg_cg(ldg_ptr + BLOCK * i);
    }


    asm volatile(
        "bar.sync 0;\n"
        "mov.u32 %0, %%clock;\n"
        : "=r"(end)
        :
        : "memory"
    );

    #pragma unroll
    for (int i = 0; i < UNROLL; ++i) {
        sum += reg[i];
    }
    clock_d[offset] = end-start;


    if (sum != 0) {
        *y = sum;
    }
}

int main() {
    const int N_DATA = DATA_SIZE_IN_BYTE / sizeof(int);

    int *x,*y;
    uint32_t *clock_d;
    uint32_t *clock_h = (uint32_t*)malloc(128*4);
    cudaMalloc(&x, N_DATA * sizeof(int));
    cudaMalloc(&y, N_DATA * sizeof(int));
    cudaMalloc(&clock_d, 128 * sizeof(int));
    cudaMemset(x, 0, N_DATA * sizeof(int));

    int grid = 1;

    // warm up to cache data into L2
    for (int i = 0; i < WARMUP_ITER; ++i) {
        kernel<BLOCK, UNROLL, N_DATA><<<grid, BLOCK>>>(x, y, clock_d);
    }

    kernel<BLOCK, UNROLL, N_DATA><<<grid, BLOCK>>>(x, y, clock_d);

    cudaDeviceSynchronize();
    cudaMemcpy(clock_h, clock_d, sizeof(int)*128, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    uint32_t clk = prop.clockRate / 1000;
    uint32_t sm = prop.multiProcessorCount;
    std::cout << N_LDG * sizeof(int)<< " Byte" << std::endl;
    std::cout << clock_h[0] << " cycle" << std::endl;

    printf("standard clock frequency: %u MHz\n", clk);
    printf("SM: %u\n", sm);

    printf("time :%f s\n", (float)clock_h[0] * 1 / clk * 1e-6);
    std::cout << float(N_LDG)*sizeof(int) * sm *clk *1e-3 / (float)clock_h[0]  <<  std::endl;


    cudaFree(x);
    cudaFree(y);
    cudaFree(clock_d);

    return 0;
}



