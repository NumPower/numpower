#include "indexing.h"
#include <php.h>
#include "Zend/zend_alloc.h"
#include "Zend/zend_API.h"
#include "ndarray.h"
#include "initializers.h"
#include "types.h"
#include "../config.h"

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

/**
 * @param r
 * @param length
 * @param start
 * @param stop
 * @param step
 * @param slicelength
 * @return
 */
int
Slice_GetIndices(SliceObject *r, int length, int *start, int *stop, int *step, int *slicelength)
{
    int defstop;

    if (r->step == NULL) {
        *step = 1;
    } else {
        *step = r->step[0];
        if (*step == 0) {
            zend_throw_error(NULL,
                            "slice step cannot be zero");
            return -1;
        }
    }

    defstop = *step < 0 ? -1 : length;

    if (r->start == NULL) {
        *start = *step < 0 ? length-1 : 0;
    } else {
        *start = *(r->start);
        if (*start < 0) *start += length;
        if (*start < 0) *start = (*step < 0) ? -1 : 0;
        if (*start >= length) {
            *start = (*step < 0) ? length - 1 : length;
        }
    }

    if (r->stop == NULL) {
        *stop = defstop;
    } else {
        *stop = r->stop[0];
        if (*stop < 0) *stop += length;
        if (*stop < 0) *stop = -1;
        if (*stop > length) *stop = length;
    }

    if ((*step < 0 && *stop >= *start) || \
            (*step > 0 && *start >= *stop)) {
        *slicelength = 0;
    } else if (*step < 0) {
        *slicelength = (*stop - *start + 1) / (*step) + 1;
    } else {
        *slicelength = (*stop - *start - 1) / (*step) + 1;
    }

    return 0;
}
