// There are many thing need to do
// ptx O0 vs O3
// ptx *ret = *arr_ptr vs *ret = stride
// latency 220 cycle

#include <cuda_runtime.h>
#include <iostream>
#include <string>

const uint32_t ROUND = 5;//arr_size / BLOCK;
const uint32_t arr_size = 32 * ROUND;
const uint32_t BLOCK = 32;
const uint32_t GRID = 1;

// The cost of operation for char is much less than int

__global__ void l2cache_latency(uint32_t *arr, uint32_t *clock, uint32_t *ret) {
    const char *arr_ptr = reinterpret_cast<const char *>( arr + threadIdx.x);
    // int *arr_ptr = arr+threadIdx.x;

    uint32_t start, end;
    uint32_t stride;
    
#pragma unroll
    for (int ii = 0; ii < ROUND; ++ii) {
        asm volatile(
            "ld.global.ca.b32 %0, [%1];\n"
            : "=r"(stride)
            : "l"(arr_ptr)
            : "memory"
        );
        arr_ptr += stride;
    }
    arr_ptr -= stride * ROUND;

    asm volatile(
        "bar.sync 0;\n"
        "mov.u32 %0, %%clock;\n"
        : "=r"(start)
        :
        : "memory"
    );

    
    asm volatile(
        "ld.global.ca.b32 %0, [%1];\n"
        : "=r"(stride)
        : "l"(arr_ptr)
        : "memory"
    );
    arr_ptr = arr_ptr + stride;

    asm volatile(
        "bar.sync 0;\n"
        "mov.u32 %0, %%clock;\n"
        : "=r"(end)
        :
        : "memory"
    );

    auto gap = end - start;
    arr_ptr -= stride;

    asm volatile(
        "bar.sync 0;\n"
        "mov.u32 %0, %%clock;\n"
        : "=r"(start)
        :
        : "memory"
    );
    // end and start approximately 56 cycle


#pragma unroll
    for (int ii = 0; ii < ROUND; ++ii) {
        asm volatile(
            "ld.global.ca.b32 %0, [%1];\n"
            : "=r"(stride)
            : "l"(arr_ptr)
            : "memory"
        );
        arr_ptr = arr_ptr + stride;
    }
    // there is about external 10 cycle for + operation
    // use **ptr to improve


    asm volatile(
        "bar.sync 0;\n"
        "mov.u32 %0, %%clock;\n"
        : "=r"(end)
        :
        : "memory"
    );

    clock[threadIdx.x] = end-start-gap;
    if (stride != 0)
    *ret += stride;
    // *ret += *arr_ptr; // man 10cycle
}

int main() {
    uint32_t *arr_h = (uint32_t*)malloc(arr_size * 4);
    uint32_t *clock_h = (uint32_t *)malloc(32 * 4);
    uint32_t *arr_d;
    uint32_t *clock_d;
    uint32_t *ret_d;
    for (int ii = 0; ii < arr_size; ii++) {
        arr_h[ii] = 128;
    }
    cudaMalloc((void**)&arr_d, 4 * arr_size);
    cudaMalloc((void**)&clock_d, 4 * 32);
    cudaMalloc((void**)&ret_d, 4);

    cudaMemcpy(arr_d, arr_h, sizeof(int) * arr_size, cudaMemcpyHostToDevice);

    for (int ii = 0; ii < 10000; ii++) {
        l2cache_latency<<<GRID, BLOCK>>>(arr_d, clock_d, ret_d);
    }

    cudaDeviceSynchronize();
    l2cache_latency<<<GRID, BLOCK>>>(arr_d, clock_d, ret_d);

    cudaDeviceSynchronize();
    cudaMemcpy(clock_h, clock_d, sizeof(int) * 32, cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
    std::cout << "error code = " << cudaGetLastError() << std::endl;

    float avg_latency = clock_h[0] / (ROUND-1);

    std::cout << avg_latency << std::endl;

}
