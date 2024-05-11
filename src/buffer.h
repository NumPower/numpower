#ifndef PHPSCI_NDARRAY_BUFFER_H
#define PHPSCI_NDARRAY_BUFFER_H
#include "ndarray.h"

/**
 * MemoryStack : The memory buffer of CArrays
 */
struct MemoryStack {
    NDArray** buffer;   // Dynamic array to store NDArray pointers
    int bufferSize;     // Current size of the buffer
    int numElements;
    int lastFreed;
    int totalGPUAllocated;
    int totalAllocated;
    int totalFreed;
};

extern struct MemoryStack MAIN_MEM_STACK;

void buffer_ndarray_free(int uuid);
void add_to_buffer(NDArray* array);
void buffer_init(int size);
NDArray* buffer_get(int uuid);
void buffer_free();
#endif //PHPSCI_NDARRAY_BUFFER_H
