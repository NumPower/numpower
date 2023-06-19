#include "logic.h"
#include "ndarray.h"
#include "iterators.h"
#include "../config.h"


#ifdef HAVE_CUBLAS
#include "ndmath/cuda/cuda_math.h"
#include "debug.h"

#endif

#ifdef HAVE_AVX2
#include <immintrin.h>
#endif

/**
 * Check if all values are not 0
 *
 * @todo Implement non-AVX2 logic
 * @param a
 * @return
 */
float
NDArray_All(NDArray *a) {
    int i;
    float *array = NDArray_FDATA(a);
#ifdef HAVE_AVX2
    __m256 zero = _mm256_set1_ps(0.0f);
    for (i = 0; i < NDArray_NUMELEMENTS(a) - 3; i += 4) {
        __m256 elements = _mm256_loadu_ps(&array[i]);
        __m256 comparison = _mm256_cmp_ps(elements, zero, _CMP_NEQ_OQ);

        // Perform horizontal OR operation on comparison results
        int mask = _mm256_movemask_ps(comparison);
        if (mask != 0x0F) {
            return 0;  // At least one element is zero
        }
    }

    // Check remaining elements (if any)
    for (; i < NDArray_NUMELEMENTS(a); i++) {
        if (array[i] == 0.0) {
            return 0;  // Element is zero
        }
    }

    return 1;  // All elements are non-zero
#else
    for (; i < NDArray_NUMELEMENTS(a); i++) {
        if (array[i] == 0.0) {
            return 0;  // Element is zero
        }
    }
    return 1;
#endif
}

/**
 *
 * @return
 */
int
_compare_ndarrays(NDArray *a, NDArray *b, int current_axis) {
    int diff = 1;

    if (NDArray_DEVICE(a) == NDARRAY_DEVICE_GPU && NDArray_DEVICE(b) == NDARRAY_DEVICE_GPU) {
#ifdef HAVE_CUBLAS
        diff = cuda_equal_float(NDArray_NUMELEMENTS(a), NDArray_FDATA(a), NDArray_FDATA(b), NDArray_NUMELEMENTS(a));
#endif
    } else {
        for (int i =0; i < NDArray_NUMELEMENTS(a); i++) {
            if (NDArray_FDATA(a)[i] != NDArray_FDATA(b)[i]) {
                diff = 0;
            }
        }
    }
    return diff;
}

/**
 * 1 if two arrays have the same shape and elements, 0 otherwise.
 *
 * @param a
 * @param b
 * @return
 */
int
NDArray_ArrayEqual(NDArray *a, NDArray *b)
{
    int i;
    if (NDArray_NDIM(a) != NDArray_NDIM(b)) {
        return 0;
    }

    for (i = 0; i < NDArray_NDIM(a); i++) {
        if (NDArray_SHAPE(a)[i] != NDArray_SHAPE(b)[i]) {
            return 0;
        }
    }

    return _compare_ndarrays(a, b, 0);
}