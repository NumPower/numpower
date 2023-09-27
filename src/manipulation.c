#include <Zend/zend.h>
#include "manipulation.h"
#include "ndarray.h"
#include "php.h"
#include "initializers.h"
#include "debug.h"
#include "../config.h"
#include "buffer.h"
#include "types.h"
#include <cblas.h>
#include "iterators.h"
#include "indexing.h"

#ifdef HAVE_CUBLAS
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "ndmath/cuda/cuda_math.h"
#include "gpu_alloc.h"
#endif

#ifdef HAVE_AVX2
#include <immintrin.h>
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

/**
 * @param a
 * @param permute
 * @return
 */
NDArray*
NDArray_Transpose(NDArray *a, NDArray_Dims *permute) {
    NDArray *ret = NULL;
    NDArray *contiguous_ret = NULL;
    if (NDArray_NDIM(a) < 2) {
        int ndim = NDArray_NDIM(a);
        return NDArray_FromNDArray(a, 0, NULL, NULL, &ndim);
    }

    int *new_shape = emalloc(sizeof(int) * NDArray_NDIM(a));
    int *new_strides = emalloc(sizeof(int) * NDArray_NDIM(a));
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

    for (i = 0; i < ndim; i++) {
        total_new_elements = total_new_elements * new_shape[i];
    }

    if (total_new_elements != NDArray_NUMELEMENTS(target)) {
        zend_throw_error(NULL, "NDArray Reshape: Incompatible shape");
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

void *
linearize_FLOAT_matrix(float *dst_in,
                       float *src_in,
                       NDArray * a) {
    float *src = (float *) src_in;
    float *dst = (float *) dst_in;

    if (dst) {
        int i, j;
        float* rv = dst;
        int columns = (int)NDArray_SHAPE(a)[1];
        int column_strides = NDArray_STRIDES(a)[1] / NDArray_ELSIZE(a);
        int one = 1;
        for (i = 0; i < NDArray_SHAPE(a)[0]; i++) {
            if (column_strides > 0) {
                if (NDArray_DEVICE(a) == NDARRAY_DEVICE_CPU) {
#ifdef HAVE_CBLAS
                    cblas_scopy(columns,
                                (float *) src, column_strides,
                                (float *) dst, one);
#endif
                }
                if (NDArray_DEVICE(a) == NDARRAY_DEVICE_GPU) {
#ifdef HAVE_CUBLAS
                    cublasHandle_t handle;
                    cublasCreate(&handle);
                    cublasScopy(handle, columns,
                                (const float*)src,
                                column_strides, dst, one);
#endif
                }
            } else if (column_strides < 0) {
                if (NDArray_DEVICE(a) == NDARRAY_DEVICE_CPU) {
#ifdef HAVE_CBLAS
                    cblas_scopy(columns,
                                (float *) ((float *) src + (columns - 1) * column_strides),
                                column_strides,
                                (float *) dst, one);
#endif
                }
                if (NDArray_DEVICE(a) == NDARRAY_DEVICE_GPU) {
#ifdef HAVE_CUBLAS
                    cublasHandle_t handle;
                    cublasCreate(&handle);
                    cublasScopy(handle, columns, (const float*)src,
                                column_strides / sizeof(float), dst, one);
#endif
                }
            } else {
                if (NDArray_DEVICE(a) == NDARRAY_DEVICE_CPU) {
                    for (j = 0; j < columns; ++j) {
                        memcpy((float *) dst + j, (float *) src, sizeof(float));
                    }
                }
            }

            src += NDArray_STRIDES(a)[0]/sizeof(float);
            dst += NDArray_SHAPE(a)[1];
        }
        return rv;
    } else {
        return src;
    }
}

/**
 * @param array
 * @param indexes
 * @param num_indices
 * @param return_view
 * @return
 */
NDArray*
NDArray_Slice(NDArray* array, NDArray** indexes, int num_indices, int return_view) {
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

/**
 * @param a
 * @param axis
 * @return
 */
NDArray*
NDArray_ExpandDim(NDArray *a, int axis) {
    int *output_shape = emalloc(sizeof(int) * (NDArray_NDIM(a) + 1));
    int output_ndim = NDArray_NDIM(a) + 1;

    // Calculate the total size of the input and output arrays in bytes
    size_t input_size = sizeof(float);
    size_t output_size = sizeof(float);

    for (int i = 0; i < NDArray_NDIM(a); i++) {
        input_size *= NDArray_SHAPE(a)[i];
        output_size *= (i < axis) ? NDArray_SHAPE(a)[i] : (i == axis) ? 1 : NDArray_SHAPE(a)[i - 1];
    }

    // Initialize the output shape and strides
    for (int i = 0; i <= NDArray_NDIM(a); i++) {
        if (i < axis) {
            output_shape[i] = NDArray_SHAPE(a)[i];
        } else if (i == axis) {
            output_shape[i] = 1; // Expand along this axis
        } else {
            output_shape[i] = NDArray_SHAPE(a)[i - 1];
        }
    }
    NDArray *rtn = NDArray_Empty(output_shape, output_ndim, NDARRAY_TYPE_FLOAT32, NDArray_DEVICE(a));

    // Copy data to the expanded output array
    if (NDArray_DEVICE(a) == NDARRAY_DEVICE_CPU) {
        memcpy(NDArray_FDATA(rtn), NDArray_FDATA(a), input_size);
    } else {
#ifdef HAVE_CUBLAS
        NDArray_VMEMCPY_D2D(NDArray_DATA(a), NDArray_DATA(rtn), input_size);
#endif
    }
    return rtn;
}