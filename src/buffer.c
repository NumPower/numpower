#include <stdlib.h>
#include <php.h>
#include "Zend/zend_alloc.h"
#include "buffer.h"
#include "string.h"
#include "ndarray.h"

#ifdef ZTS
#include "TSRM.h"
#endif

/**
 * MEMORY STACK
 *
 * CArrays Memory Buffer
 */
struct MemoryStack MAIN_MEM_STACK;

#ifdef ZTS
static void ndarray_init_globals(void) {
    MAIN_MEM_STACK.lock = tsrm_mutex_alloc();
}
#endif

/**
 * Initialize MemoryStack Buffer
 */
void buffer_init(int size) {
#ifdef ZTS
    if (!MAIN_MEM_STACK.lock) {
        ndarray_init_globals();
    }
#endif

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
#ifdef ZTS
    tsrm_mutex_lock(MAIN_MEM_STACK.lock);
#endif

    if (MAIN_MEM_STACK.buffer != NULL) {
        efree(MAIN_MEM_STACK.buffer);
        MAIN_MEM_STACK.buffer = NULL;
        MAIN_MEM_STACK.numElements = 0;
        MAIN_MEM_STACK.bufferSize = 0;
    }

#ifdef ZTS
    tsrm_mutex_unlock(MAIN_MEM_STACK.lock);
#endif
}

/**
 * Debug dump of buffer stats
 */
void buffer_dump() {
#ifdef ZTS
    tsrm_mutex_lock(MAIN_MEM_STACK.lock);
#endif

    printf("\nMAIN_MEM_STACK.totalAllocated: %d", MAIN_MEM_STACK.totalAllocated);
    printf("\nMAIN_MEM_STACK.totalFreed: %d\n", MAIN_MEM_STACK.totalFreed);

#ifdef ZTS
    tsrm_mutex_unlock(MAIN_MEM_STACK.lock);
#endif
}

/**
 * Free specific NDArray in buffer
 */
void buffer_ndarray_free(int uuid) {
#ifdef ZTS
    tsrm_mutex_lock(MAIN_MEM_STACK.lock);
#endif

    if (MAIN_MEM_STACK.buffer != NULL && 
        uuid >= 0 && 
        uuid < MAIN_MEM_STACK.numElements && 
        MAIN_MEM_STACK.buffer[uuid] != NULL) {
        
        NDArray_FREE(MAIN_MEM_STACK.buffer[uuid]);
        MAIN_MEM_STACK.buffer[uuid] = NULL;
        MAIN_MEM_STACK.lastFreed = uuid;
        MAIN_MEM_STACK.totalFreed++;
    }

#ifdef ZTS
    tsrm_mutex_unlock(MAIN_MEM_STACK.lock);
#endif
}

/**
 * Get NDArray from buffer by UUID
 */
NDArray* buffer_get(int uuid) {
#ifdef ZTS
    tsrm_mutex_lock(MAIN_MEM_STACK.lock);
#endif

    assert(uuid >= 0 && uuid < MAIN_MEM_STACK.numElements);
    assert(MAIN_MEM_STACK.buffer[uuid] != NULL);
    NDArray* result = MAIN_MEM_STACK.buffer[uuid];

#ifdef ZTS
    tsrm_mutex_unlock(MAIN_MEM_STACK.lock);
#endif

    return result;
}

/**
 * Add NDArray to global buffer
 */
void add_to_buffer(NDArray* ndarray) {
#ifdef ZTS
    tsrm_mutex_lock(MAIN_MEM_STACK.lock);
#endif

    if (MAIN_MEM_STACK.buffer == NULL) {
        buffer_init(1);
    }

    if (MAIN_MEM_STACK.lastFreed > -1) {
        ndarray->uuid = MAIN_MEM_STACK.lastFreed;
        MAIN_MEM_STACK.buffer[MAIN_MEM_STACK.lastFreed] = ndarray;
        MAIN_MEM_STACK.lastFreed = -1;
#ifdef ZTS
        tsrm_mutex_unlock(MAIN_MEM_STACK.lock);
#endif
        return;
    }

    if (MAIN_MEM_STACK.numElements >= MAIN_MEM_STACK.bufferSize) {
        int newSize = MAIN_MEM_STACK.bufferSize * 2;
        NDArray** newBuffer = (NDArray**)erealloc(MAIN_MEM_STACK.buffer, 
                                newSize * sizeof(NDArray*));
        if (!newBuffer) {
            php_error_docref(NULL, E_WARNING, 
                "Failed to allocate memory for the buffer");
#ifdef ZTS
            tsrm_mutex_unlock(MAIN_MEM_STACK.lock);
#endif
            return;
        }
        MAIN_MEM_STACK.buffer = newBuffer;
        MAIN_MEM_STACK.bufferSize = newSize;
    }

    ndarray->uuid = MAIN_MEM_STACK.numElements;
    MAIN_MEM_STACK.buffer[MAIN_MEM_STACK.numElements] = ndarray;
    MAIN_MEM_STACK.numElements++;
    MAIN_MEM_STACK.totalAllocated++;

#ifdef ZTS
    tsrm_mutex_unlock(MAIN_MEM_STACK.lock);
#endif
}
