#include "indexing.h"
#include <php.h>
#include "Zend/zend_alloc.h"
#include "Zend/zend_API.h"
#include "ndarray.h"
#include "initializers.h"
#include "types.h"

/**
 * Slice a buffer of type float
 *
 * @param buffer
 * @param ndims
 * @param shape
 * @param strides
 * @param start
 * @param stop
 * @param step
 * @return
 */
void*
slice_float(float* buffer, int ndims, int* shape, int* strides, int* start, int* stop, int* step,
            float* out_buffer, int* out_shape, int* out_strides, int* out_ndims)
{
    out_ndims = emalloc(sizeof(int));
    out_shape = emalloc(sizeof(int) * ndims);
    out_strides = emalloc(sizeof(int) * ndims);
    for (int i = 0; i < ndims; i++) {
        if (start[i] < 0) {
            start[i] += shape[i];
        }
        if (stop[i] < 0) {
            stop[i] += shape[i];
        }
        int length = (stop[i] - start[i] + step[i] - 1) / step[i];
        if (length <= 0) {
            length = 0;
        }
        out_shape[*out_ndims] = length;
        out_strides[*out_ndims] = strides[i] * step[i];
        *out_ndims++;
    }
    int64_t out_size = 1;
    for (int i = 0; i < *out_ndims; i++) {
        out_size *= out_shape[i];
    }
    float* data_ptr = buffer;
    if (out_size > 0) {
        data_ptr += start[ndims-1] * strides[ndims-1];
    }
    for (int i = ndims-1; i >= 0; i--) {
        if (out_shape[i] == 0) {
            out_strides[i] = 0;
        } else {
            break;
        }
    }
    out_buffer = data_ptr;
}

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
    return rtn;
}
