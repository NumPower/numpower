#include <string.h>
#include "../manipulation.h"
#include "../initializers.h"
#include "calculation.h"
#include <Zend/zend.h>
#include "../../config.h"
#include "../ndarray.h"

static int
float_argmax(float *ip, int n, float *max_ind)
{
    int i;
    float mp = *ip;

    *max_ind = 0;

    if (isnanf(mp)) {
        /* nan encountered; it's maximal */
        return 0;
    }

    for (i = 1; i < n; i++) {
        ip++;

        /*
        * Propagate nans, similarly as max() and min()
        */
        if (!(*ip <= mp)) {  /* negated, for correct nan handling */
            mp = *ip;
            *max_ind = i;
            if (isnanf(mp)) {
                /* nan encountered, it's maximal */
                break;
            }
        }
    }
    return 0;
}

static int
float_argmin(float *ip, int n, float *min_ind)
{
    int i;
    float mp = *ip;
    *min_ind = 0;


    if (isnanf(mp)) {
        /* nan encountered; it's minimal */
        return 0;
    }

    for (i = 1; i < n; i++) {
        ip++;

        if (!_LESS_THAN_OR_EQUAL(mp, *ip)) {
            mp = *ip;
            *min_ind = i;
            if (isnanf(mp)) {
                break;
            }
        }
    }

    return 0;
}

/**
 * ArgMin and ArgMax common function
 *
 * This work is derived from NumPy
 *
 * @param op
 * @param axis
 * @param out
 * @param keepdims
 * @param is_argmax
 * @return
 */
NDArray *
NDArray_ArgMinMaxCommon(NDArray *op, int axis, int keepdims, bool is_argmax) {
    if (NDArray_DEVICE(op) == NDARRAY_DEVICE_GPU) {
        zend_throw_error(NULL, "GPU not supported.");
        return NULL;
    }

    NDArray *ap = NULL, *rp = NULL;
    NDArray_ArgFunc* arg_func = NULL;
    char *ip, *func_name;
    float *rptr;
    int i, n, m;
    int elsize;
    // Keep a copy because axis changes via call to NDArray_CheckAxis
    int axis_copy = axis;
    int _shape_buf[NDARRAY_MAX_DIMS];
    int *out_shape;
    // Keep the number of dimensions and shape of
    // original array. Helps when `keepdims` is True.
    int* original_op_shape = NDArray_SHAPE(op);
    int out_ndim = NDArray_NDIM(op);

    if ((ap = (NDArray *)NDArray_CheckAxis(op, &axis, 0)) == NULL) {
        zend_throw_error(NULL, "Invalid axis parameter");
        NDArray_FREE(op);
        return NULL;
    }

    if (axis != NDArray_NDIM(ap)-1) {
        NDArray_Dims newaxes;
        int dims[NDARRAY_MAX_DIMS];
        int j;

        newaxes.ptr = dims;
        newaxes.len = NDArray_NDIM(ap);
        for (j = 0; j < axis; j++) {
            dims[j] = j;
        }
        for (j = axis; j < NDArray_NDIM(ap) - 1; j++) {
            dims[j] = j + 1;
        }
        dims[NDArray_NDIM(ap) - 1] = axis;
        //@todo Use transpose permutation
        //op = NDArray_Transpose(ap, &newaxes);
        op = NDArray_Transpose(ap);

        NDArray_FREE(ap);
        if (op == NULL) {
            return NULL;
        }
    }
    else {
        op = ap;
    }

    // Will get native-byte order contiguous copy.
    NDArrayDescriptor *descr = NDArray_DESCRIPTOR(op);
    if (descr == NULL) {
        return NULL;
    }

    ap = NDArray_FromNDArray(op, 0, NULL, NULL, NULL);

    NDArray_FREE(op);
    if (ap == NULL) {
        return NULL;
    }

    // Decides the shape of the output array.
    if (!keepdims) {
        out_ndim = NDArray_NDIM(ap) - 1;
        out_shape = emalloc(sizeof(int) * NDArray_NDIM(ap));
        memcpy(out_shape, NDArray_SHAPE(ap), sizeof(int) * NDArray_NDIM(ap));
    }
    else {
        out_shape = _shape_buf;
        if (axis_copy == NDARRAY_MAX_DIMS) {
            for (int i = 0; i < out_ndim; i++) {
                out_shape[i] = 1;
            }
        }
        else {
            /*
             * While `ap` may be transposed, we can ignore this for `out` because the
             * transpose only reorders the size 1 `axis` (not changing memory layout).
             */
            memcpy(out_shape, original_op_shape, out_ndim * sizeof(int));
            out_shape[axis] = 1;
        }
    }

    if (is_argmax) {
        func_name = "argmax";
        if (NDArray_DEVICE(op) == NDARRAY_DEVICE_CPU) {
            arg_func = float_argmax;
        }
    }
    else {
        func_name = "argmin";
        if (NDArray_DEVICE(op) == NDARRAY_DEVICE_CPU) {
            arg_func = float_argmin;
        }
    }

    elsize = NDArray_ELSIZE(ap);
    m = NDArray_SHAPE(ap)[NDArray_NDIM(ap)-1];
    if (m == 0) {
        zend_throw_error(NULL, "attempt to get %s of an empty sequence", func_name);
        goto fail;
    }

    rp = NDArray_Zeros(out_shape, out_ndim, NDArray_TYPE(ap), NDArray_DEVICE(ap));

    if (rp == NULL) {
        goto fail;
    }

    n = NDArray_NUMELEMENTS(ap)/m;
    rptr = (float*)NDArray_DATA(rp);
    for (ip = NDArray_DATA(ap), i = 0; i < n; i++, ip += elsize*m) {
        arg_func((float*)ip, m, rptr);
        rptr += 1;
    }

    NDArray_FREE(ap);
    return rp;
fail:
    zend_throw_error(NULL, "%s fatal error", func_name);
    return NULL;
}
