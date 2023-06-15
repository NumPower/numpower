#include "logic.h"
#include "ndarray.h"
#include "iterators.h"

#ifndef HAVE_AVX2
#include <immintrin.h>
#endif

/**
 * Check if all values are not 0
 *
 * @todo Implement non-AVX2 logic
 * @param a
 * @return
 */
double
NDArray_All(NDArray *a) {
    __m256d zero = _mm256_set1_pd(0.0);
    int i;
    double *array = NDArray_DDATA(a);
    for (i = 0; i < NDArray_NUMELEMENTS(a) - 3; i += 4) {
        __m256d elements = _mm256_loadu_pd(&array[i]);
        __m256d comparison = _mm256_cmp_pd(elements, zero, _CMP_NEQ_OQ);

        // Perform horizontal OR operation on comparison results
        int mask = _mm256_movemask_pd(comparison);
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
}

/**
 *
 * @return
 */
int
_compare_ndarrays(NDArray *a, NDArray *b, int current_axis) {
    int diff = 1;
    NDArray *slice_a, *slice_b;
    NDArrayIterator_REWIND(a);
    NDArrayIterator_REWIND(b);

    while(!NDArrayIterator_ISDONE(a)) {
        slice_a = NDArrayIterator_GET(a);
        slice_b = NDArrayIterator_GET(b);
        if (current_axis < NDArray_NDIM(a) - 1) {
            diff = _compare_ndarrays(slice_a, slice_b, current_axis);
            if (diff == 0) {
                NDArray_FREE(slice_a);
                NDArray_FREE(slice_b);
                diff = 0;
            }
        }
        if (current_axis == NDArray_NDIM(a) - 1) {
            if (NDArray_DDATA(slice_a)[0] != NDArray_DDATA(slice_b)[0]) {
                NDArray_FREE(slice_a);
                NDArray_FREE(slice_b);
                return 0;
            }
        }
        NDArray_FREE(slice_a);
        NDArray_FREE(slice_b);
        NDArrayIterator_NEXT(a);
        NDArrayIterator_NEXT(b);
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