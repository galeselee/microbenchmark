#include <cuda_runtime.h>
#include <iostream>
#include <string>

// len = 0.26 M 
const int arr_size = 1 << 18; 
const int BLOCK = 32;
const int GRID = 1;
const int unroll = arr_size / BLOCK;

__device__ __forceinline__ int ldg2l2cache(int *ptr) {
    int ret;
    asm volatile(
        "ld.global.cg.b32 %0, [%1];"
        : "=r"(ret)
        : "l"(ptr)
    );
    // printf("ret = %d\n",ret);
    return ret;
}

__device__ __forceinline__ int ticktime() {
    int time;
    asm volatile(
        "bar.sync 0;\n"
        "mov.u32 %0, %%clock;\n"
        : "=r"(time)
        :
    );
    return time;
}

__global__ void l2cache_latency(int *arr, int *clock, int UNROLL, int *ret) {
    int *arr_ptr = arr + threadIdx.x;

    uint32_t start, end;
    for (int ii = 0; ii < UNROLL; ii++) {
        arr_ptr += ldg2l2cache(arr_ptr);
    }
    arr_ptr -= 32 * 8192;

    start = ticktime();

    for (int ii = 0; ii < UNROLL; ii++) {
        arr_ptr += ldg2l2cache(arr_ptr);
    }

    end = ticktime();
    clock[threadIdx.x] = end-start;
    *ret = *arr_ptr;
}

int main() {
    int *arr_h = (int*)malloc(arr_size * 4);
    int *clock_h = (int *)malloc(32 * 4);
    int *arr_d;
    int *clock_d;
    int *ret_d;
    for (int ii = 0; ii < arr_size; ii++) {
        arr_h[ii] = 32;
    }
    cudaMalloc((void**)&arr_d, 4 * arr_size);
    cudaMalloc((void**)&clock_d, 4 * 32);
    cudaMalloc((void**)&ret_d, 4);
    cudaMemset(clock_d, 0, sizeof(int) * 32);
    cudaMemcpy(arr_d, arr_h, sizeof(int) * arr_size, cudaMemcpyHostToDevice);

    l2cache_latency<<<GRID, BLOCK>>>(arr_d, clock_d, unroll, ret_d);
    l2cache_latency<<<GRID, BLOCK>>>(arr_d, clock_d, unroll, ret_d);
    l2cache_latency<<<GRID, BLOCK>>>(arr_d, clock_d, unroll, ret_d);
    l2cache_latency<<<GRID, BLOCK>>>(arr_d, clock_d, unroll, ret_d);

    cudaDeviceSynchronize();
    cudaMemcpy(clock_h, clock_d, sizeof(int) * 32, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    std::cout << "error code = " << cudaGetLastError() << std::endl;

    float avg_latency = 0;
    for (int ii = 0; ii < 32; ii++) {
        avg_latency += clock_h[ii];
    }
    avg_latency /= 32 * 8192;

    std::cout << avg_latency << std::endl;

    
}
