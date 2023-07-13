#include "../config.h"

#ifdef HAVE_CUBLAS
#include "gpu_alloc.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "buffer.h"

void
NDArray_VMALLOC(void** target, unsigned int size) {
    MAIN_MEM_STACK.totalGPUAllocated++;
    cudaMalloc(target, size);
    cudaDeviceSynchronize();
}

void
NDArray_VMEMCPY_D2D(char* target, char* dst, unsigned int size) {
    cudaMemcpy(dst, target, size, cudaMemcpyDeviceToDevice);
    cudaDeviceSynchronize();
}

void
NDArray_VFREE(void* target) {
    MAIN_MEM_STACK.totalGPUAllocated--;
    cudaFree(target);
    cudaDeviceSynchronize();
}

void
NDArray_VCHECK() {
    if (MAIN_MEM_STACK.totalGPUAllocated != 0) {
        printf("\nVRAM MEMORY LEAK: Unallocated %d arrays\n", MAIN_MEM_STACK.totalGPUAllocated);
    }
}

#endif
