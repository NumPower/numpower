#include "logic.h"
#include "ndarray.h"
#include "iterators.h"
#include "../config.h"
#include "initializers.h"
#include <Zend/zend.h>


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
 * Return if NDArray is equal element-wise
 *
 * @param nda
 * @param ndb
 * @return
 */
NDArray*
NDArray_Equal(NDArray* nda, NDArray* ndb) {
    if (NDArray_DEVICE(nda) != NDArray_DEVICE(ndb)) {
        zend_throw_error(NULL, "Devices mismatch in `equal` function");
        return NULL;
    }

    int i;
    int *rtn_shape = emalloc(sizeof(int) * NDArray_NDIM(nda));

    for (i = 0; i < NDArray_NDIM(nda); i++) {
        rtn_shape[i] = NDArray_SHAPE(nda)[i];
    }

    NDArray *result = NDArray_Empty(rtn_shape, NDArray_NDIM(nda), NDArray_TYPE(nda), NDArray_DEVICE(nda));
    if (NDArray_NUMELEMENTS(nda) != NDArray_NUMELEMENTS(ndb)) {
        zend_throw_error(NULL, "Incompatible shapes in `equal` function");
        return NULL;
    }

    if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_GPU) {
#ifdef HAVE_CUBLAS
        cuda_float_compare_equal(NDArray_SHAPE(nda)[0], NDArray_FDATA(nda), NDArray_FDATA(ndb), NDArray_FDATA(result),
                                 NDArray_NUMELEMENTS(nda));
#endif
    } else {
#if HAVE_AVX2
        // Process 8 elements at a time using AVX2
        i = 0;
        for (; i < NDArray_NUMELEMENTS(nda) - 7; i += 8) {
            // Load 8 elements from each array
            __m256 vec1 = _mm256_loadu_ps(&NDArray_FDATA(nda)[i]);
            __m256 vec2 = _mm256_loadu_ps(&NDArray_FDATA(ndb)[i]);

            // Compare elements for equality
            __m256 cmp = _mm256_cmp_ps(vec1, vec2, _CMP_EQ_OQ);

            // Convert comparison results to float (1.0 for true, 0.0 for false)
            __m256 resultVec = _mm256_cvtepi32_ps(_mm256_castps_si256(cmp));

            // Store the results
            _mm256_storeu_ps(&NDArray_FDATA(result)[i], resultVec);
        }

        // Process remaining elements using scalar operations
        for (; i < NDArray_NUMELEMENTS(nda); i++) {
            NDArray_FDATA(result)[i] = (NDArray_FDATA(nda)[i] == NDArray_FDATA(ndb)[i]) ? 1.0f : 0.0f;
        }
        return result;
#else
        for (i = 0; i < NDArray_NUMELEMENTS(nda); i++) {
            NDArray_FDATA(result)[i] = (fabs(NDArray_FDATA(nda)[i] - NDArray_FDATA(ndb)[i]) <= 0.0000001f) ? 1.0f : 0.0f;
        }
    }
    return result;
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