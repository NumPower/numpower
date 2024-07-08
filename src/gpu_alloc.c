#include "../config.h"
#include <Zend/zend.h>

#ifdef HAVE_CUBLAS
#include "gpu_alloc.h"
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "buffer.h"

void
vmalloc(void** target, unsigned int size) {
    MAIN_MEM_STACK.totalGPUAllocated++;
    cublasStatus_t stat = cudaMalloc(target, size);
    if (stat != cudaSuccess) {
        zend_throw_error(NULL, "device memory allocation failed");
    }
}

void
vmemcpyd2d(char* target, char* dst, unsigned int size) {
    cudaMemcpy(dst, target, size, cudaMemcpyDeviceToDevice);
}

void
vmemcpyh2d(char* target, char* dst, unsigned int size) {
    cudaMemcpy(dst, target, size, cudaMemcpyHostToDevice);
}

void
vfree(void* target) {
    MAIN_MEM_STACK.totalGPUAllocated--;
    cudaFree(target);
}

void
vmemcheck() {
    if (MAIN_MEM_STACK.totalGPUAllocated != 0) {
        printf("\nVRAM MEMORY LEAK: leaked %d array(s)\n", MAIN_MEM_STACK.totalGPUAllocated);
    }
}

float
NDArray_VFLOAT(char *target) {
    float value;
    cudaMemcpy(&value, target, sizeof(float), cudaMemcpyDeviceToHost);
    return value;
}

float
NDArray_VFLOATF_I(float *target, int index) {
    float value;
    cudaMemcpy(&value, &(target[index]), sizeof(float), cudaMemcpyDeviceToHost);
    return value;
}

#endif
