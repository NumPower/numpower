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

    /* Check that index is valid, taking into account negative indices */
    if (NDARRAY_UNLIKELY((*axis < -ndim) || (*axis >= ndim))) {
        //zend_throw_error(NULL, "Axis is out of bounds for array dimension");
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
NDArray_Transpose(NDArray *a) {
    NDArray *ret = NULL;
    NDArray *contiguous_ret = NULL;
    if (NDArray_NDIM(a) < 2) {
        int ndim = NDArray_NDIM(a);
        return NDArray_FromNDArray(a, 0, NULL, NULL, &ndim);
    }

    int *new_shape = emalloc(sizeof(int) * NDArray_NDIM(a));
    int *new_strides = emalloc(sizeof(int) * NDArray_NDIM(a));

    if (new_shape == NULL || new_strides == NULL) {
        zend_throw_error(NULL, "failed to allocate memory for new_shape and new_strides.");
        return NULL;
    }

    reverse_copy(NDArray_SHAPE(a), new_shape, NDArray_NDIM(a));
    reverse_copy(NDArray_STRIDES(a), new_strides, NDArray_NDIM(a));

    ret = NDArray_Copy(a, NDArray_DEVICE(a));
    efree(ret->strides);
    efree(ret->dimensions);
    ret->strides = new_strides;
    ret->dimensions = new_shape;
    NDArray_ENABLEFLAGS(ret, NDARRAY_ARRAY_F_CONTIGUOUS);
    contiguous_ret = NDArray_ToContiguous(ret);
    NDArray_FREE(ret);
    return contiguous_ret;
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
        zend_throw_error(NULL, "incompatible shape during reshape call.");
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

    NDArray *ret = NDArray_FromNDArrayBase(array, data_ptr, shape_ptr, strides_ptr, new_dim);
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

/**
 * @param target
 * @todo Append all dimensions with axis
 * @return
 */
NDArray*
NDArray_Append(NDArray *a, NDArray *b) {
    char *tmp_ptr;
    if (NDArray_DEVICE(a) != NDArray_DEVICE(b)) {
        zend_throw_error(NULL, "NDArrays must be on the same device.");
        return NULL;
    }

    if (NDArray_NDIM(a) != 1 || NDArray_NDIM(b) != 1) {
        zend_throw_error(NULL, "You can only append vectors.");
        return NULL;
    }

    int *shape = emalloc(sizeof(int));
    shape[0] = NDArray_NUMELEMENTS(a) + NDArray_NUMELEMENTS(b);
    NDArray* rtn = NDArray_Empty(shape, 1, NDArray_TYPE(a), NDArray_DEVICE(a));

    if (NDArray_DEVICE(a) == NDARRAY_DEVICE_GPU) {
#ifdef HAVE_CUBLAS
        NDArray_VMEMCPY_D2D(NDArray_DATA(a), NDArray_DATA(rtn), NDArray_ELSIZE(a) * NDArray_NUMELEMENTS(a));
        tmp_ptr = NDArray_DATA(rtn) + NDArray_ELSIZE(a) * NDArray_NUMELEMENTS(a);
        NDArray_VMEMCPY_D2D(NDArray_DATA(b), tmp_ptr, NDArray_ELSIZE(b) * NDArray_NUMELEMENTS(b));
#endif
    }

    if (NDArray_DEVICE(a) == NDARRAY_DEVICE_CPU) {
        memcpy(NDArray_DATA(rtn), NDArray_DATA(a), NDArray_ELSIZE(a) * NDArray_NUMELEMENTS(a));
        tmp_ptr = NDArray_DATA(rtn) + NDArray_ELSIZE(a) * NDArray_NUMELEMENTS(a);
        memcpy(tmp_ptr, NDArray_DATA(b), NDArray_ELSIZE(b) * NDArray_NUMELEMENTS(b));
    }

    return rtn;
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

    while(ncopies--) {
        index = a_size;
        while(index--) {
            memmove(ret_it->dataptr, a_it->dataptr, elsize);
            NDArray_ITER_NEXT(a_it);
            NDArray_ITER_NEXT(ret_it);
        }
        NDArray_ITER_RESET(a_it);
    }
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

