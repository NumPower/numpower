#include "indexing.h"
#include <php.h>
#include "Zend/zend_alloc.h"
#include "Zend/zend_API.h"
#include "ndarray.h"
#include "initializers.h"
#include "types.h"
#include "../config.h"
#include "gpu_alloc.h"

/**
 * NDArray diagonal
 *
 * @param target
 * @param offset
 * @return
 */
NDArray*
NDArray_Diagonal(NDArray *target, int offset) {
    NDArray *rtn;
    int i;
    if (NDArray_NDIM(target) != 2) {
        zend_throw_error(NULL, "NDArray_Diagonal: Array must be 2-d.");
        return NULL;
    }

    if (NDArray_DEVICE(target) == NDARRAY_DEVICE_CPU) {
        int *new_shape = emalloc(sizeof(int));
        new_shape[0] = NDArray_SHAPE(target)[NDArray_NDIM(target) - 1];
        rtn = NDArray_Empty(new_shape, 1, NDARRAY_TYPE_FLOAT32, NDARRAY_DEVICE_CPU);
        for (i = 0; i < NDArray_SHAPE(target)[NDArray_NDIM(target) - 1]; i++) {
            NDArray_FDATA(rtn)[i] = ((float*)(NDArray_DATA(target) + (i * NDArray_STRIDES(target)[NDArray_NDIM(target) - 2]) + (i * NDArray_STRIDES(target)[NDArray_NDIM(target) - 1])))[0];
        }
    }
#ifdef HAVE_CUBLAS
    if (NDArray_DEVICE(target) == NDARRAY_DEVICE_GPU) {
        int *new_shape = emalloc(sizeof(int));
        new_shape[0] = NDArray_SHAPE(target)[NDArray_NDIM(target) - 1];
        rtn = NDArray_Empty(new_shape, 1, NDARRAY_TYPE_FLOAT32, NDARRAY_DEVICE_GPU);
        for (i = 0; i < NDArray_SHAPE(target)[NDArray_NDIM(target) - 1]; i++) {
            NDArray_VMEMCPY_D2D((NDArray_DATA(target) + (i * NDArray_STRIDES(target)[NDArray_NDIM(target) - 2]) + (i * NDArray_STRIDES(target)[NDArray_NDIM(target) - 1])), NDArray_DATA(rtn) + (i * sizeof(float)), sizeof(float));
        }
    }
#endif
    return rtn;
}
