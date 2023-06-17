#include "ndarray_ops.h"
#include "ndarray.h"
#include "indexing.h"
#include "initializers.h"

/**
 * NDArray Slicing
 *
 * @param target
 * @param start
 * @param stop
 * @param step
 * @return
 */
NDArray*
NDArray_Slice(NDArray* target, int start, int stop, int step)
{
    NDArray* out;
    int* new_strides, *new_shape, *new_ndims;
    float* outdata;
    slice_float(
            NDArray_DATA(target),
            NDArray_NDIM(target),
            NDArray_SHAPE(target),
            NDArray_STRIDES(target),
            &start,
            &stop,
            &step,
            outdata,
            new_shape,
            new_strides,
            new_ndims);
    out = Create_NDArray(
            new_shape,
            *new_ndims,
            NDArray_TYPE(target)
            );
    return out;
}