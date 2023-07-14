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

void
transposeMatrixFloat(float* matrix, float* output, int rows, int cols) {
    int i, j;
    for ( i = 0; i < rows; i++) {
        for ( j = 0; j < cols; j++) {
            output[j * rows + i] = matrix[i * cols + j];
        }
    }
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

    if (NDArray_NDIM(a) < 2) {
        int ndim = NDArray_NDIM(a);
        return NDArray_FromNDArray(a, 0, NULL, NULL, &ndim);
    }

    int *new_shape = emalloc(sizeof(int) * NDArray_NDIM(a));
    reverse_copy(NDArray_SHAPE(a), new_shape, NDArray_NDIM(a));
    ret = NDArray_Empty(new_shape, NDArray_NDIM(a), NDARRAY_TYPE_FLOAT32, NDArray_DEVICE(a));

    // @todo Implement N-dimensinal permutation
    if (NDArray_NDIM(a) != 2) {
        zend_throw_error(NULL, "must be a 2-d array");
        return NULL;
    }
    if (NDArray_DEVICE(a) == NDARRAY_DEVICE_GPU) {
#ifdef HAVE_CUBLAS
        cuda_float_transpose(NDArray_FDATA(a), NDArray_FDATA(ret), NDArray_SHAPE(a)[0], NDArray_SHAPE(a)[1]);
#endif
    } else {
        transposeMatrixFloat(NDArray_FDATA(a), NDArray_FDATA(ret), NDArray_SHAPE(a)[0], NDArray_SHAPE(a)[1]);
    }
    return ret;
}

/**
 * Reshape NDArray
 *
 * @param target
 * @param new_shape
 * @return
 */
NDArray*
NDArray_Reshape(NDArray *target, int *new_shape, int ndim)
{
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
NDArray_Flatten(NDArray *target)
{
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
                       NDArray * a)
{
    float *src = (float *) src_in;
    float *dst = (float *) dst_in;

    if (dst) {
        int i, j;
        float* rv = dst;
        int columns = (int)NDArray_SHAPE(a)[1];
        int column_strides = NDArray_STRIDES(a)[1]/sizeof(float);
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
                                (const float*)((const float*)src + (columns - 1) * column_strides),
                                column_strides, dst, one);
#endif
                }
            }
            else if (column_strides < 0) {
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
                    cublasScopy(handle, columns, (const float*)((const char*)src + (columns - 1) * column_strides),
                                column_strides / sizeof(float), dst, one);
#endif
                }
            }
            else {
                if (NDArray_DEVICE(a) == NDARRAY_DEVICE_CPU) {
                    /*
                     * Zero stride has undefined behavior in some BLAS
                     * implementations (e.g. OSX Accelerate), so do it
                     * manually
                     */
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

NDArray*
NDArray_Slice(NDArray* array, NDArray** indexes, int num_indices) {
    NDArray *slice, *rtn;
    int slice_ndim = NDArray_NDIM(array);
    int *slice_shape = emalloc(sizeof(int) * slice_ndim);
    int *slice_strides = emalloc(sizeof(int) * slice_ndim);
    int i, offset = 0;
    int start = 0, stop = 0, step = 0;

    if (num_indices > NDArray_NDIM(array)) {
        zend_throw_error(NULL, "too many indices for array");
        return NULL;
    }

    for (i = 0; i < num_indices; i++) {
        if (NDArray_NUMELEMENTS(indexes[i]) >= 1) {
            start = (int) NDArray_FDATA(indexes[i])[0];
        } else {
            start = 0;
        }
        if (NDArray_NUMELEMENTS(indexes[i]) >= 2) {
            stop  = (int)NDArray_FDATA(indexes[i])[1];
        } else {
            stop = NDArray_SHAPE(array)[i];
        }
        if (NDArray_NUMELEMENTS(indexes[i]) == 3) {
            step  = (int)NDArray_FDATA(indexes[i])[2];
        } else {
            step = 1;
        }
        if (NDArray_NUMELEMENTS(indexes[i]) > 3) {
            zend_throw_error(NULL, "Too many arguments for slicing indexes");
            return NULL;
        }
        slice_shape[i] = (int)floorf(((float)stop - (float)start) / (float)step);
        offset += start * NDArray_STRIDES(array)[i];
    }

    for (; i < slice_ndim; i++) {
        slice_shape[i] = NDArray_SHAPE(array)[i];
    }
    memcpy(slice_strides, NDArray_STRIDES(array), slice_ndim * sizeof(int));
    slice = NDArray_FromNDArray(array, offset, slice_shape, slice_strides, &slice_ndim);

    float *rtn_data;
    if (NDArray_DEVICE(array) == NDARRAY_DEVICE_CPU) {
        rtn_data = emalloc(NDArray_ELSIZE(array) * NDArray_NUMELEMENTS(slice));
    }
#ifdef HAVE_CUBLAS
    if (NDArray_DEVICE(array) == NDARRAY_DEVICE_GPU) {
        NDArray_VMALLOC((void**)&rtn_data, NDArray_ELSIZE(array) * NDArray_NUMELEMENTS(slice));
    }
#endif

    linearize_FLOAT_matrix(rtn_data, NDArray_FDATA(slice), slice);
    slice->data = (char*)rtn_data;
    slice->strides = Generate_Strides(slice_shape, slice_ndim, NDArray_ELSIZE(slice));
    slice->base = NULL;
    NDArray_FREE(array);
    efree(slice_strides);
    return slice;
}