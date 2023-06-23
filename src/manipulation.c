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

void copy(const int* src, int* dest, int size) {
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
    NDArray *rtn = NDArray_Zeros(new_shape, ndim, NDARRAY_TYPE_FLOAT32);
    efree(rtn->data);
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
