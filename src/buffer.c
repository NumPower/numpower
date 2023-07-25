#include <stdlib.h>
#include <php.h>
#include "Zend/zend_alloc.h"
#include "Zend/zend_API.h"
#include "buffer.h"
#include "string.h"
#include "ndarray.h"
#include "debug.h"

/**
 * MEMORY STACK
 *
 * CArrays Memory Buffer
 */
struct MemoryStack MAIN_MEM_STACK;

void buffer_dump() {
    printf("\nMAIN_MEM_STACK.totalAllocated: %d", MAIN_MEM_STACK.totalAllocated);
    printf("\nMAIN_MEM_STACK.totalFreed: %d\n", MAIN_MEM_STACK.totalFreed);
}

/**
 * If CARRAY_GC_DEBUG env is True, CArray Garbage Collector
 * will print debug messages when destructing objects.
 */
static int
CArrayBuffer_ISDEBUGON()
{
    if (getenv("NDARRAY_BUFFER_DEBUG") == NULL) {
        return 0;
    }
    if (!strcmp(getenv("NDARRAY_BUFFER_DEBUG"), "0")) {
        return 0;
    }
    return 1;
}

/**
 * Initialize MemoryStack Buffer
 */
void buffer_init(int size) {
    MAIN_MEM_STACK.buffer = (NDArray**)emalloc(size * sizeof(NDArray*));
    MAIN_MEM_STACK.bufferSize = size;
    MAIN_MEM_STACK.numElements = 0;
    MAIN_MEM_STACK.lastFreed = -1;
    MAIN_MEM_STACK.totalGPUAllocated = 0;
    MAIN_MEM_STACK.totalAllocated = 0;
    MAIN_MEM_STACK.totalFreed = 0;
}

/**
 * Free the buffer
 */
void buffer_free() {
    if (MAIN_MEM_STACK.buffer != NULL) {
        efree(MAIN_MEM_STACK.buffer);
        MAIN_MEM_STACK.buffer = NULL;
    }
}

/**
 * @param uuid
 */
void buffer_ndarray_free(int uuid) {
    if (MAIN_MEM_STACK.buffer != NULL) {
        // @todo investigate double free problem
        if (MAIN_MEM_STACK.lastFreed == -1) {
            MAIN_MEM_STACK.lastFreed = uuid;
        }
        if (MAIN_MEM_STACK.buffer[uuid] != NULL) {
            NDArray_FREE(MAIN_MEM_STACK.buffer[uuid]);
            MAIN_MEM_STACK.buffer[uuid] = NULL;
            MAIN_MEM_STACK.totalFreed++;
        }
    }
}

/**
 * @param uuid
 */
NDArray* buffer_get(int uuid) {
    assert(MAIN_MEM_STACK.buffer[uuid] != NULL);
    return MAIN_MEM_STACK.buffer[uuid];
}


/**
 * Add CArray to MemoryStack (Buffer) and retrieve MemoryPointer
 *
 * @param array CArray CArray to add into the stack
 * @param size  size_t Size of CArray in bytes
 */
void add_to_buffer(NDArray* ndarray, size_t size) {
    if (MAIN_MEM_STACK.lastFreed > -1) {
        ndarray->uuid = MAIN_MEM_STACK.lastFreed;
        MAIN_MEM_STACK.buffer[MAIN_MEM_STACK.lastFreed] = ndarray;
        MAIN_MEM_STACK.lastFreed = -1;
        return;
    }

    // Increase the buffer size if necessary
    if (MAIN_MEM_STACK.numElements >= MAIN_MEM_STACK.bufferSize) {
        int newSize = (MAIN_MEM_STACK.bufferSize == 0) ? 1 : (MAIN_MEM_STACK.bufferSize * 2);
        NDArray** newBuffer = (NDArray**)erealloc(MAIN_MEM_STACK.buffer, newSize * sizeof(NDArray*));
        if (newBuffer == NULL) {
            // Error handling: Failed to allocate memory for the buffer
            php_printf("Failed to allocate memory for the buffer");
            return;
        }
        MAIN_MEM_STACK.buffer = newBuffer;
        MAIN_MEM_STACK.bufferSize = newSize;
    }

    // Set the NDArray's uuid to its position in the buffer
    ndarray->uuid = MAIN_MEM_STACK.numElements;

    // Add the NDArray to the buffer
    MAIN_MEM_STACK.buffer[MAIN_MEM_STACK.numElements] = ndarray;
    MAIN_MEM_STACK.numElements++;
    MAIN_MEM_STACK.totalAllocated++;
}