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

#ifdef HAVE_AVX2
#include <immintrin.h>
#endif


void transposeMatrix(float* matrix, float* output, int rows, int cols) {
    int i, j;
    for ( i = 0; i < rows; i++) {
        for ( j = 0; j < cols; j++) {
            output[j * rows + i] = matrix[i * cols + j];
        }
    }
}

NDArray*
NDArray_Transpose(NDArray *a, NDArray_Dims *permute) {
    NDArray *ret = NULL;
    int *new_shape = emalloc(sizeof(int) * NDArray_NDIM(a));
    memcpy(new_shape, NDArray_SHAPE(a), sizeof(int) * NDArray_NDIM(a));
    ret = NDArray_Zeros(new_shape, NDArray_NDIM(a), NDARRAY_TYPE_FLOAT32);
    // @todo Implement N-dimensinal permutation
    if (NDArray_NDIM(a) != 2) {
        zend_throw_error(NULL, "must be a 2-d array");
        return NULL;
    }
    transposeMatrix(NDArray_FDATA(a), NDArray_FDATA(ret), NDArray_SHAPE(a)[0], NDArray_SHAPE(a)[1]);
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
    target->ndim = ndim;

    for (i = 0; i < ndim; i++) {
        total_new_elements = total_new_elements * new_shape[i];
    }

    if (total_new_elements != NDArray_NUMELEMENTS(target)) {
        zend_throw_error(NULL, "NDArray Reshape: Incompatible shape");
        return NULL;
    }

    if (NDArray_NDIM(target) < ndim) {
        efree(target->dimensions);
        target->dimensions = emalloc(sizeof(int) * ndim);
        memcpy(target->dimensions, new_shape, sizeof(int) * ndim);
    }

    efree(target->strides);
    target->strides = Generate_Strides(new_shape, ndim, sizeof(float));

    efree(target->dimensions);
    target->dimensions = new_shape;
    return target;
}