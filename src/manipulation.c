#include <Zend/zend.h>
#include "manipulation.h"
#include "ndarray.h"
#include "initializers.h"
#include "../config.h"
#include "types.h"
#include <cblas.h>
#include "iterators.h"
#include "indexing.h"
#include "debug.h"

#ifdef HAVE_CUBLAS
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "ndmath/cuda/cuda_math.h"
#include "gpu_alloc.h"
#endif

int
multiply_int_vector(int *a, int size) {
    int total = 1, i;
    for (i = 0; i < size; i++) {
        total = total * a[i];
    }
    return total;
}

void reverse_copy(const int* src, int* dest, int size) {
    for (int i = size - 1; i >= 0; i--) {
        dest[i] = src[size - i - 1];
    }
}

void copy(const int* src, int* dest, unsigned int size) {
    for (int i = 0; i < size; i++) {
        dest[i] = src[i];
    }
}

static inline int
check_and_adjust_axis_msg(int *axis, int ndim) {
    if (axis == NULL) {
        return 0;
    }
    if (NDARRAY_UNLIKELY((*axis < -ndim) || (*axis >= ndim))) {
        zend_throw_error(NULL, "Axis is out of bounds for array dimension");
        return -1;
    }
    /* adjust negative indices */
    if (*axis < 0) {
        *axis += ndim;
    }
    return 0;
}

static inline int
check_and_adjust_axis(int *axis, int ndim) {
    return check_and_adjust_axis_msg(axis, ndim);
}


/**
 * @param a
 * @param permute
 * @return
 */
NDArray*
NDArray_Transpose(NDArray *a, NDArray_Dims *permute) {
    int *axes;
    int i, n;
    int permutation[NDARRAY_MAX_DIMS], reverse_permutation[NDARRAY_MAX_DIMS];
    NDArray *ret = NULL;

    if (permute == NULL) {
        n = NDArray_NDIM(a);
        for (i = 0; i < n; i++) {
            permutation[i] = n-1-i;
        }
    }
    else {
        n = permute->len;
        axes = permute->ptr;
        if (n != NDArray_NDIM(a)) {
            zend_throw_error(NULL, "axes don't match array");
            return NULL;
        }
        for (i = 0; i < n; i++) {
            reverse_permutation[i] = -1;
        }
        for (i = 0; i < n; i++) {
            int axis = axes[i];
            if (check_and_adjust_axis(&axis, NDArray_NDIM(a)) < 0) {
                zend_throw_error(NULL, "axes don't match array");
                return NULL;
            }
            if (reverse_permutation[axis] != -1) {
                zend_throw_error(NULL, "repeated axis in transpose");
                return NULL;
            }
            reverse_permutation[axis] = i;
            permutation[i] = axis;
        }
    }

    ret = NDArray_Copy(a, NDArray_DEVICE(a));
    if (ret == NULL) {
        return NULL;
    }

    /* fix the dimensions and strides of the return-array */
    for (i = 0; i < n; i++) {
        NDArray_SHAPE(ret)[i] = NDArray_SHAPE(a)[permutation[i]];
        NDArray_STRIDES(ret)[i] = NDArray_STRIDES(a)[permutation[i]];
    }
    NDArray_ENABLEFLAGS(ret, NDARRAY_ARRAY_F_CONTIGUOUS);

    if (NDArray_DEVICE(a) == NDARRAY_DEVICE_CPU || (NDArray_DEVICE(a) == NDARRAY_DEVICE_GPU && NDArray_NDIM(a) != 2)) {
        NDArray * contiguous_ret;
        contiguous_ret = NDArray_ToContiguous(ret);
        NDArray_FREE(ret);
        return contiguous_ret;
    } else {
#ifdef HAVE_CUBLAS
        cuda_float_transpose(32, 8, NDArray_FDATA(ret), NDArray_FDATA(ret), NDArray_SHAPE(a)[1], NDArray_SHAPE(a)[0]);
        return ret;
#endif
    }
}

/**
 * Reshape NDArray
 *
 * @param target
 * @param new_shape
 * @return
 */
NDArray*
NDArray_Reshape(NDArray *target, int *new_shape, int ndim) {
    int total_new_elements = 1;
    int i;
    if (new_shape == NULL) {
        zend_throw_error(NULL, "new shape cannot be null.");
        return NULL;
    }

    for (i = 0; i < ndim; i++) {
        total_new_elements = total_new_elements * new_shape[i];
    }

    if (total_new_elements != NDArray_NUMELEMENTS(target)) {
        zend_throw_error(NULL, "incompatible shape in reshape call.");
        return NULL;
    }
    NDArray *rtn = NDArray_Empty(new_shape, ndim, NDARRAY_TYPE_FLOAT32, NDArray_DEVICE(target));
    rtn->ndim = ndim;
    rtn->device = NDArray_DEVICE(target);
    NDArray_FREEDATA(rtn);
    rtn->data = target->data;
    rtn->base = target;
    NDArray_ADDREF(target);
    return rtn;
}

/**
 * @param target
 * @return
 */
NDArray*
NDArray_Flatten(NDArray *target) {
    NDArray *rtn = NDArray_Copy(target, NDArray_DEVICE(target));
    rtn->ndim = 1;
    if (NDArray_NDIM(target) == 0) {
        rtn->dimensions[0] = 1;
        rtn->strides[0] = NDArray_ELSIZE(target);
        return rtn;
    }
    if (NDArray_NDIM(target) == 1) {
        return rtn;
    }
    rtn->dimensions[0] = multiply_int_vector(NDArray_SHAPE(target), NDArray_NDIM(target));
    rtn->strides[0] = NDArray_ELSIZE(target);
    return rtn;
}

/**
 * @param array
 * @param indexes
 * @param num_indices
 * @param return_view
 * @return
 */
NDArray*
NDArray_Slice(NDArray* array, NDArray** indexes, int num_indices) {
    if (num_indices > NDArray_NDIM(array)) {
        zend_throw_error(NULL, "too many indices for array.");
        return NULL;
    }

    int new_strides[NDARRAY_MAX_DIMS];
    int new_shape[NDARRAY_MAX_DIMS];
    int i, start = 0, stop = 0, step = 0, n_steps = 0, new_dim = 0, orig_dim = 0;
    char *data_ptr = NDArray_DATA(array);

    SliceObject sliceobj;


    for (i = 0; i < num_indices; i++) {
        sliceobj.start = NULL;
        sliceobj.stop = NULL;
        sliceobj.step = NULL;
        if (NDArray_NUMELEMENTS(indexes[i]) >= 1) {
            sliceobj.start = emalloc(sizeof(int));
            sliceobj.start[0] = (int) NDArray_FDATA(indexes[i])[0];
        }
        if (NDArray_NUMELEMENTS(indexes[i]) >= 2) {
            sliceobj.stop = emalloc(sizeof(int));
            sliceobj.stop[0] = (int) NDArray_FDATA(indexes[i])[1];
        }
        if (NDArray_NUMELEMENTS(indexes[i]) == 3) {
            sliceobj.step = emalloc(sizeof(int));
            sliceobj.step[0] = (int) NDArray_FDATA(indexes[i])[2];
        }
        if(Slice_GetIndices(&sliceobj, NDArray_SHAPE(array)[orig_dim], &start, &stop, &step, &n_steps) < 0) {
            zend_throw_error(NULL, "Slicing error");
            goto failure;
        }
        if (n_steps <= 0) {
            n_steps = 0;
            step = 1;
            start = 0;
        }
        data_ptr += NDArray_STRIDES(array)[orig_dim] * start;
        new_strides[new_dim] = NDArray_STRIDES(array)[orig_dim] * step;
        new_shape[new_dim] = n_steps;
        new_dim += 1;
        orig_dim += 1;
        if (sliceobj.start != NULL) {
            efree(sliceobj.start);
        }
        if (sliceobj.stop != NULL) {
            efree(sliceobj.stop);
        }
        if (sliceobj.step != NULL) {
            efree(sliceobj.step);
        }
    }

    int *strides_ptr = emalloc(sizeof(int) * new_dim);
    memcpy(strides_ptr, NDArray_STRIDES(array), sizeof(int) * NDArray_NDIM(array));
    for (i = 0; i < new_dim; i++) {
        strides_ptr[i] = new_strides[i];
    }
    int *shape_ptr = emalloc(sizeof(int) * new_dim);
    memcpy(shape_ptr, NDArray_SHAPE(array), sizeof(int) * NDArray_NDIM(array));
    for (i = 0; i < new_dim; i++) {
        shape_ptr[i] = new_shape[i];
    }

    if (num_indices < NDArray_NDIM(array)) {
        new_dim = NDArray_NDIM(array);
    }

    NDArray *ret = NULL;
    NDArray *fret = NDArray_FromNDArrayBase(array, data_ptr, shape_ptr, strides_ptr, new_dim);

    if (num_indices > 1) {
        NDArray_ENABLEFLAGS(fret, NDARRAY_ARRAY_F_CONTIGUOUS);
        ret = NDArray_ToContiguous(fret);
        NDArray_FREE(fret);
    } else {
        ret = fret;
    }
    return ret;
failure:
    if (sliceobj.start != NULL) {
        efree(sliceobj.start);
    }
    if (sliceobj.stop != NULL) {
        efree(sliceobj.stop);
    }
    if (sliceobj.step != NULL) {
        efree(sliceobj.step);
    }
    return NULL;
}

NDArray*
NDArray_ConcatenateFlat(NDArray **arrays, int num_arrays)
{
    NDArray *rtn;
    int iarrays;
    int *shape = emalloc(sizeof(int));

    *shape = 0;
    if (num_arrays <= 0) {
        zend_throw_error(NULL,
                        "need at least one array to concatenate");
        return NULL;
    }

    for (iarrays = 0; iarrays < num_arrays; ++iarrays) {
        *shape += NDArray_NUMELEMENTS(arrays[iarrays]);
        if (*shape < 0) {
            zend_throw_error(NULL,
                            "total number of elements "
                            "too large to concatenate");
            return NULL;
        }
    }
    rtn = NDArray_Zeros(shape, 1, NDArray_TYPE(arrays[0]), NDArray_DEVICE(arrays[0]));
    int device = NDArray_DEVICE(rtn);

    int extra_bytes = 0;
    char *tmp_data = NULL;
    for (int i_array = 0; i_array < num_arrays; i_array++) {
        extra_bytes = i_array * (NDArray_NUMELEMENTS(arrays[i_array]) * NDArray_ELSIZE(arrays[0]));
        tmp_data = NDArray_DATA(rtn) + extra_bytes;

        switch (NDArray_DEVICE(rtn)) {
            case NDARRAY_DEVICE_GPU:
#ifdef HAVE_CUBLAS

#endif
                break;
            default:
                memcpy(tmp_data, NDArray_DATA(arrays[i_array]), NDArray_NUMELEMENTS(arrays[i_array]) * NDArray_ELSIZE(arrays[i_array]));
        }
    }
    return rtn;
}

/**
 * @param target
 * @return
 */
NDArray*
NDArray_Append(NDArray **arrays, int axis, int num_arrays) {
    NDArray *output = NULL;
    if (axis == -1) {
        output = NDArray_ConcatenateFlat(arrays, num_arrays);
    }
    return output;
}

/**
 * @param a
 * @return
 */
NDArray*
NDArray_ToContiguous(NDArray *a) {
    NDArray *ret = NDArray_EmptyLike(a);
    efree(ret->strides);
    ret->strides = Generate_Strides(NDArray_SHAPE(a), NDArray_NDIM(a), NDArray_ELSIZE(a));

    int index;
    int elsize = NDArray_ELSIZE(a);
    int ret_size = NDArray_NUMELEMENTS(ret);
    int a_size = NDArray_NUMELEMENTS(a);

    int ncopies = (ret_size / a_size);

    NDArrayIter *a_it = NDArray_NewElementWiseIter(a);
    NDArrayIter *ret_it = NDArray_NewElementWiseIter(ret);

    if (NDArray_DEVICE(a) == NDARRAY_DEVICE_CPU) {
        while (ncopies--) {
            index = a_size;
            while (index--) {
                memmove(ret_it->dataptr, a_it->dataptr, elsize);
                NDArray_ITER_NEXT(a_it);
                NDArray_ITER_NEXT(ret_it);
            }
            NDArray_ITER_RESET(a_it);
        }
    }
#ifdef HAVE_CUBLAS
    if (NDArray_DEVICE(a) == NDARRAY_DEVICE_GPU) {
        while (ncopies--) {
            index = a_size;
            while (index--) {
                vmemcpyd2d(a_it->dataptr, ret_it->dataptr, elsize);
                NDArray_ITER_NEXT(a_it);
                NDArray_ITER_NEXT(ret_it);
            }
            NDArray_ITER_RESET(a_it);
        }
    }
#endif
    efree(a_it);
    efree(ret_it);
    return ret;
}

static inline NDArray*
normalize_axis_vector(NDArray *axis, int ndim) {
    NDArray *output = NDArray_EmptyLike(axis);
    NDArray *axis_val;
    NDArrayIterator_REWIND(axis);
    int i = 0;
    while(!NDArrayIterator_ISDONE(axis)) {
        axis_val = NDArrayIterator_GET(axis);
        int axis_int_val = (int)(NDArray_FDATA(axis_val)[0]);
        if (check_and_adjust_axis(&axis_int_val, ndim) < 0) {
            NDArray_FREE(axis_val);
            NDArray_FREE(output);
            return NULL;
        }
        NDArray_FDATA(output)[i] = (float)axis_int_val;
        NDArrayIterator_NEXT(axis);
        NDArray_FREE(axis_val);
        i++;
    }
    return output;
}

/**
 * @param a
 * @param axis
 * @return
 */
NDArray*
NDArray_ExpandDim(NDArray *a, NDArray *axis) {
    NDArray *temp;
    bool free_axis = false;

    if (NDArray_NDIM(axis) == 0) {
        axis = NDArray_AtLeast1D(axis);
        free_axis = true;
    }

    int output_ndim = NDArray_NUMELEMENTS(axis) + NDArray_NDIM(a);
    int *output_shape = emalloc(sizeof(int) * output_ndim);
    if (NDArray_NDIM(axis) > 1) {
        zend_throw_error(NULL, "axis must be either a scalar or a vector. Found matrix with %d dimensions.",
                         NDArray_NDIM(axis));
    }

    NDArray *normalized_axis = normalize_axis_vector(axis, output_ndim);

    if (normalized_axis == NULL) {
        efree(output_shape);
        NDArray_FREE(normalized_axis);
        if (free_axis) {
            NDArray_FREE(axis);
        }
        zend_throw_error(NULL, "invalid axis or axes provided.");
        return NULL;
    }

    int found;
    int *a_shape = NDArray_SHAPE(a);
    int shape_it = 0;
    for (int ax = 0; ax < output_ndim; ax++) {
        found  = 0;
        for (int i = 0; i < NDArray_NUMELEMENTS(normalized_axis); i++) {
            if ((int)(NDArray_FDATA(normalized_axis)[i]) == ax) {
                found = 1;
                output_shape[ax] = 1;
                break;
            }
        }
        if (!found) {
            output_shape[ax] = a_shape[shape_it];
            shape_it++;
        }
    }

    NDArray_FREE(normalized_axis);
    NDArray *output = NDArray_Reshape(a, output_shape, output_ndim);

    if (output == NULL) {
        efree(output_shape);
        return NULL;
    }

    if (free_axis) {
        NDArray_FREE(axis);
    }

    return output;
}

/**
 * @param arr
 * @param axis
 * @param _flags
 * @return
 */
NDArray*
NDArray_CheckAxis(NDArray *arr, int *axis, int _flags)
{
    NDArray *temp1, *temp2;
    int n = NDArray_NDIM(arr);

    if (*axis == NDARRAY_MAX_DIMS || n == 0) {
        if (n != 1) {
            temp1 = NDArray_Flatten(arr);
            if (*axis == NDARRAY_MAX_DIMS) {
                *axis = NDArray_NDIM(temp1)-1;
            }
        }
        else {
            temp1 = arr;
            NDArray_ADDREF(temp1);
            *axis = 0;
        }
        if (!_flags && *axis == 0) {
            return temp1;
        }
    }
    else {
        temp1 = arr;
        NDArray_ADDREF(temp1);
    }
    temp2 = temp1;
    n = NDArray_NDIM(temp2);
    if (check_and_adjust_axis(axis, n) < 0) {
        return NULL;
    }
    return temp2;
}

NDArray*
NDArray_AtLeast1D(NDArray *a) {
    NDArray *output = NULL;
    if (NDArray_NDIM(a) == 0) {
        int *new_shape = emalloc(sizeof(int));
        new_shape[0] = 1;
        output = NDArray_Reshape(a, new_shape, 1);
    } else {
        int *strides = emalloc(sizeof(int) * NDArray_NDIM(a));
        int *new_shape = emalloc(sizeof(int) * NDArray_NDIM(a));

        memcpy(strides, NDArray_STRIDES(a), sizeof(int) * NDArray_NDIM(a));
        memcpy(new_shape, NDArray_SHAPE(a), sizeof(int) * NDArray_NDIM(a));

        output = NDArray_FromNDArrayBase(a, NDArray_DATA(a), new_shape, strides, NDArray_NDIM(a));
    }
    return output;
}

NDArray*
NDArray_AtLeast2D(NDArray *a) {
    NDArray *output = NULL;
    if (NDArray_NDIM(a) < 2) {
        int *new_shape = emalloc(sizeof(int) * 2);
        new_shape[0] = 1;
        new_shape[1] = NDArray_NUMELEMENTS(a);
        output = NDArray_Reshape(a, new_shape, 2);
    } else {
        int *strides = emalloc(sizeof(int) * NDArray_NDIM(a));
        int *new_shape = emalloc(sizeof(int) * NDArray_NDIM(a));
        memcpy(strides, NDArray_STRIDES(a), sizeof(int) * NDArray_NDIM(a));
        memcpy(new_shape, NDArray_SHAPE(a), sizeof(int) * NDArray_NDIM(a));
        output = NDArray_FromNDArrayBase(a, NDArray_DATA(a), new_shape, strides, NDArray_NDIM(a));
    }
    return output;
}

NDArray*
NDArray_AtLeast3D(NDArray *a) {
    NDArray *output = NULL;
    if (NDArray_NDIM(a) < 3) {
        int *new_shape = emalloc(sizeof(int) * 2);
        new_shape[0] = 1;
        if (NDArray_NDIM(a) < 2) {
            new_shape[1] = 1;
            new_shape[2] = NDArray_NUMELEMENTS(a);
        }
        if (NDArray_NDIM(a) == 2) {
            new_shape[1] = NDArray_SHAPE(a)[0];
            new_shape[2] = NDArray_SHAPE(a)[1];
        }
        output = NDArray_Reshape(a, new_shape, 3);
    } else {
        int *strides = emalloc(sizeof(int) * NDArray_NDIM(a));
        int *new_shape = emalloc(sizeof(int) * NDArray_NDIM(a));
        memcpy(strides, NDArray_STRIDES(a), sizeof(int) * NDArray_NDIM(a));
        memcpy(new_shape, NDArray_SHAPE(a), sizeof(int) * NDArray_NDIM(a));
        output = NDArray_FromNDArrayBase(a, NDArray_DATA(a), new_shape, strides, NDArray_NDIM(a));
    }
    return output;
}

int
NDArray_ConvertMultiAxis(NDArray *axis_in, int ndim, bool *out_axis_flags)
{
    /* NULL means all the axes */
    if (axis_in == NULL) {
        memset(out_axis_flags, 1, ndim);
        return 1;
    }

    if (NDArray_NDIM(axis_in) == 0) {
        int axis = (int)NDArray_FDATA(axis_in)[0];

        memset(out_axis_flags, 0, ndim);

        if (ndim == 0 && (axis == 0 || axis == -1)) {
            return 1;
        }

        if (check_and_adjust_axis(&axis, ndim) < 0) {
            return 0;
        }
        out_axis_flags[axis] = 1;
        return 1;
    }

    int i, naxes;

    memset(out_axis_flags, 0, ndim);

    naxes = NDArray_NUMELEMENTS(axis_in);
    if (naxes < 0) {
        return 0;
    }
    for (i = 0; i < naxes; ++i) {
        int axis = (int)NDArray_FDATA(axis_in)[i];
        if (check_and_adjust_axis(&axis, ndim) < 0) {
            return 0;
        }
        if (out_axis_flags[axis]) {
            zend_throw_error(NULL,
                             "duplicate value in 'axis'");
            return 0;
        }
        out_axis_flags[axis] = 1;
    }

    return 1;
}

void
NDArray_RemoveAxesInPlace(NDArray *arr, const bool *flags)
{
    int *shape = NDArray_SHAPE(arr), *strides = NDArray_STRIDES(arr);
    int idim, ndim = NDArray_NDIM(arr), idim_out = 0;

    /* Compress the dimensions and strides */
    for (idim = 0; idim < ndim; ++idim) {
        if (!flags[idim]) {
            shape[idim_out] = shape[idim];
            strides[idim_out] = strides[idim];
            ++idim_out;
        }
    }

    /* The final number of dimensions */
    arr->ndim = idim_out;
}

NDArray*
NDArray_SqueezeSelected(NDArray *self, bool *axis_flags)
{
    NDArray *ret;
    int idim, ndim, any_ones;
    int *shape;

    ndim = NDArray_NDIM(self);
    shape = NDArray_SHAPE(self);

    /* Verify that the axes requested are all size one */
    any_ones = 0;
    for (idim = 0; idim < ndim; ++idim) {
        if (axis_flags[idim] != 0) {
            if (shape[idim] == 1) {
                any_ones = 1;
            }
            else {
                zend_throw_error(NULL,
                                "cannot select an axis to squeeze out "
                                "which has size not equal to one");
                return NULL;
            }
        }
    }

    /* If there were no axes to squeeze out, return the same array */
    if (!any_ones) {
        NDArray_ADDREF(self);
        return self;
    }


    int *n_shape = emalloc(sizeof(int) * NDArray_NDIM(self));
    int *n_strides = emalloc(sizeof(int) * NDArray_NDIM(self));
    memcpy(n_shape, NDArray_SHAPE(self), sizeof(int) * NDArray_NDIM(self));
    memcpy(n_strides, NDArray_STRIDES(self), sizeof(int) * NDArray_NDIM(self));
    ret = NDArray_FromNDArrayBase(self, NDArray_DATA(self), n_shape, n_strides, NDArray_NDIM(self));
    if (ret == NULL) {
        return NULL;
    }

    NDArray_RemoveAxesInPlace(ret, axis_flags);
    return ret;
}

NDArray*
NDArray_Squeeze(NDArray *a, NDArray *axis)
{
    bool unit_dims[128];
    if (axis != NULL) {
        if (NDArray_ConvertMultiAxis(axis, NDArray_NDIM(a), unit_dims) != 1) {
            return NULL;
        }
        return NDArray_SqueezeSelected(a, unit_dims);
    }

    NDArray *ret;
    int idim, ndim, any_ones;
    int *shape;

    ndim = NDArray_NDIM(a);
    shape = NDArray_SHAPE(a);

    any_ones = 0;
    for (idim = 0; idim < ndim; ++idim) {
        if (shape[idim] == 1) {
            unit_dims[idim] = 1;
            any_ones = 1;
        }
        else {
            unit_dims[idim] = 0;
        }
    }

    if (!any_ones) {
        NDArray_ADDREF(a);
        return a;
    }

    int *n_shape = emalloc(sizeof(int) * NDArray_NDIM(a));
    int *n_strides = emalloc(sizeof(int) * NDArray_NDIM(a));
    memcpy(n_shape, NDArray_SHAPE(a), sizeof(int) * NDArray_NDIM(a));
    memcpy(n_strides, NDArray_STRIDES(a), sizeof(int) * NDArray_NDIM(a));
    ret = NDArray_FromNDArrayBase(a, NDArray_DATA(a), n_shape, n_strides, NDArray_NDIM(a));
    if (ret == NULL) {
        return NULL;
    }

    NDArray_RemoveAxesInPlace(ret, unit_dims);
    return ret;
}

NDArray*
NDArray_SwapAxes(NDArray *a, int axis1, int axis2)
{
    NDArray_Dims new_axes;
    int dims[NDARRAY_MAX_DIMS];
    int n = NDArray_NDIM(a);
    int i;

    if (check_and_adjust_axis_msg(&axis1, n) < 0) {
        return NULL;
    }
    if (check_and_adjust_axis_msg(&axis2, n) < 0) {
        return NULL;
    }

    for (i = 0; i < n; ++i) {
        dims[i] = i;
    }
    dims[axis1] = axis2;
    dims[axis2] = axis1;

    new_axes.ptr = dims;
    new_axes.len = n;

    return NDArray_Transpose(a, &new_axes);
}

NDArray*
NDArray_Rollaxis(NDArray *a, int axis, int start)
{
    NDArray *result;
    int n = NDArray_NDIM(a);

    if (check_and_adjust_axis_msg(&axis, n) < 0) {
        return NULL;
    }

    if (start < 0) start += n;

    if (!(0 <= start < n + 1)) {
        zend_throw_error(NULL, "'%s' arg requires %d <= %s < %d, but %d was passed in", "start", -n, "start", n+1, start);
        return NULL;
    }

    if (axis < start) start -= 1;

    if (axis == start) {
        return NDArray_Copy(a, NDArray_DEVICE(a));
    }

    int *axes = emalloc(sizeof(int) * n);
    for (int i = 0; i < n; i++) {
        axes[i] = i;
    }

    for (int i = n; i > start; i--) {
        axes[i] = axes[i - 1];
    }
    axes[start] = axis;

    NDArray_Dims dims;
    dims.len = n;
    dims.ptr = axes;

    result = NDArray_Transpose(a, &dims);
    efree(axes);
    return result;
}


NDArray*
NDArray_Moveaxis(NDArray *a, int* src, int* dest, int n_source, int n_dest)
{
    NDArray *result;
    int order[NDARRAY_MAX_DIMS];
    int n = NDArray_NDIM(a);

    if (check_and_adjust_axis_msg(src, n) < 0) {
        return NULL;
    }
    if (check_and_adjust_axis_msg(dest, n) < 0) {
        return NULL;
    }

    if (n_source != n_dest) {
        zend_throw_error(NULL, "`source` and `destination` must have the same number of elements.");
        return NULL;
    }

    int found;
    int count_order = 0;
    for (int i = 0; i < n; i++) {
        found = 0;
        for (int j = 0; j < n_source; j++) {
            if (src[j] == i) found = 1;
        }
        if (!found) {
            order[count_order] = i;
            count_order++;
        }
    }

    for (int i = 0; i < n_source; i++) {
        if (dest[i] < 0) {
            dest[i] = dest[i] + (count_order + 1);
        }
        order[dest[i]] = src[i];
        count_order++;
    }

    NDArray_Dims order_dims;
    order_dims.len = count_order;
    order_dims.ptr = order;
    return NDArray_Transpose(a, &order_dims);
}

NDArray*
NDArray_Flip(NDArray *a, NDArray *axis)
{

}