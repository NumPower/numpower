#include "logic.h"
#include "ndarray.h"
#include "../config.h"
#include "initializers.h"
#include <Zend/zend.h>
#include <php.h>

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
    for (i = 0; i < NDArray_NUMELEMENTS(a); i++) {
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
NDArray_Greater(NDArray* nda, NDArray* ndb) {
    if (NDArray_DEVICE(nda) != NDArray_DEVICE(ndb)) {
        zend_throw_error(NULL, "Devices mismatch in `equal` function");
        return NULL;
    }

    int i;
    int *rtn_shape = emalloc(sizeof(int) * NDArray_NDIM(nda));

    for (i = 0; i < NDArray_NDIM(nda); i++) {
        rtn_shape[i] = NDArray_SHAPE(nda)[i];
    }

    NDArray *broadcasted = NULL;
    NDArray *a_broad = NULL, *b_broad = NULL;

    if (NDArray_NUMELEMENTS(nda) < NDArray_NUMELEMENTS(ndb)) {
        broadcasted = NDArray_Broadcast(nda, ndb);
        a_broad = broadcasted;
        b_broad = ndb;
    } else if (NDArray_NUMELEMENTS(ndb) < NDArray_NUMELEMENTS(nda)) {
        broadcasted = NDArray_Broadcast(ndb, nda);
        b_broad = broadcasted;
        a_broad = nda;
    } else {
        b_broad = ndb;
        a_broad = nda;
    }

    NDArray *result = NDArray_Empty(rtn_shape, NDArray_NDIM(a_broad), NDArray_TYPE(a_broad), NDArray_DEVICE(a_broad));

    if (b_broad == NULL) {
        zend_throw_error(NULL, "Can't broadcast arrays.");
        return NULL;
    }
    if (NDArray_DEVICE(a_broad) == NDARRAY_DEVICE_GPU) {
#ifdef HAVE_CUBLAS
        cuda_float_compare_greater(NDArray_SHAPE(nda)[0], NDArray_FDATA(nda), NDArray_FDATA(ndb), NDArray_FDATA(result),
                                   NDArray_NUMELEMENTS(nda));
#endif
    } else {
#ifdef HAVE_AVX2
        // Process 8 elements at a time using AVX2
        i = 0;
        for (; i < NDArray_NUMELEMENTS(a_broad) - 7; i += 8) {
            // Load 8 elements from each array
            __m256 vec1 = _mm256_loadu_ps(&NDArray_FDATA(a_broad)[i]);
            __m256 vec2 = _mm256_loadu_ps(&NDArray_FDATA(b_broad)[i]);

            // Compare elements for equality
            __m256 cmp = _mm256_cmp_ps(vec1, vec2, _CMP_GT_OQ);

            // Convert comparison results to float (1.0 for true, 0.0 for false)
            __m256 resultVec = _mm256_and_ps(cmp, _mm256_set1_ps(1.0f));

            // Store the results
            _mm256_storeu_ps(&NDArray_FDATA(result)[i], resultVec);
        }

        // Process remaining elements using scalar operations
        for (; i < NDArray_NUMELEMENTS(a_broad); i++) {
            NDArray_FDATA(result)[i] = NDArray_FDATA(a_broad)[i] > NDArray_FDATA(b_broad)[i] ? 1.0f : 0.0f;
        }
#else
        for (i = 0; i < NDArray_NUMELEMENTS(a_broad); i++) {
            NDArray_FDATA(result)[i] = NDArray_FDATA(a_broad)[i] > NDArray_FDATA(b_broad)[i] ? 1.0f : 0.0f;
        }
#endif
    }
    if (broadcasted != NULL) {
        NDArray_FREE(broadcasted);
    }
    return result;
}

/**
 * @param nda
 * @param ndb
 * @return
 */
NDArray*
NDArray_Less(NDArray* nda, NDArray* ndb) {
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
        cuda_float_compare_less(NDArray_SHAPE(nda)[0], NDArray_FDATA(nda), NDArray_FDATA(ndb), NDArray_FDATA(result),
                                NDArray_NUMELEMENTS(nda));
#endif
    } else {
#ifdef HAVE_AVX2
        // Process 8 elements at a time using AVX2
        i = 0;
        for (; i < NDArray_NUMELEMENTS(nda) - 7; i += 8) {
            // Load 8 elements from each array
            __m256 vec1 = _mm256_loadu_ps(&NDArray_FDATA(nda)[i]);
            __m256 vec2 = _mm256_loadu_ps(&NDArray_FDATA(ndb)[i]);

            // Compare elements for equality
            __m256 cmp = _mm256_cmp_ps(vec1, vec2, _CMP_LT_OQ);

            // Convert comparison results to float (1.0 for true, 0.0 for false)
            __m256 resultVec = _mm256_and_ps(cmp, _mm256_set1_ps(1.0f));

            // Store the results
            _mm256_storeu_ps(&NDArray_FDATA(result)[i], resultVec);
        }

        // Process remaining elements using scalar operations
        for (; i < NDArray_NUMELEMENTS(nda); i++) {
            NDArray_FDATA(result)[i] = NDArray_FDATA(nda)[i] < NDArray_FDATA(ndb)[i] ? 1.0f : 0.0f;
        }
        return result;
#else
        for (i = 0; i < NDArray_NUMELEMENTS(nda); i++) {
            NDArray_FDATA(result)[i] = NDArray_FDATA(nda)[i] < NDArray_FDATA(ndb)[i] ? 1.0f : 0.0f;
        }
#endif
    }
    return result;
}

/**
 * @param nda
 * @param ndb
 * @return
 */
NDArray*
NDArray_LessEqual(NDArray* nda, NDArray* ndb) {
    if (NDArray_DEVICE(nda) != NDArray_DEVICE(ndb)) {
        zend_throw_error(NULL, "Devices mismatch in `equal` function");
        return NULL;
    }

    int i;
    int *rtn_shape = emalloc(sizeof(int) * NDArray_NDIM(nda));

    for (i = 0; i < NDArray_NDIM(nda); i++) {
        rtn_shape[i] = NDArray_SHAPE(nda)[i];
    }

    NDArray *broadcasted = NULL;
    NDArray *a_broad = NULL, *b_broad = NULL;

    if (NDArray_NUMELEMENTS(nda) < NDArray_NUMELEMENTS(ndb)) {
        broadcasted = NDArray_Broadcast(nda, ndb);
        a_broad = broadcasted;
        b_broad = ndb;
    } else if (NDArray_NUMELEMENTS(ndb) < NDArray_NUMELEMENTS(nda)) {
        broadcasted = NDArray_Broadcast(ndb, nda);
        b_broad = broadcasted;
        a_broad = nda;
    } else {
        b_broad = ndb;
        a_broad = nda;
    }

    NDArray *result = NDArray_Empty(rtn_shape, NDArray_NDIM(a_broad), NDArray_TYPE(a_broad), NDArray_DEVICE(a_broad));

    if (b_broad == NULL) {
        zend_throw_error(NULL, "Can't broadcast arrays.");
        return NULL;
    }

    if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_GPU) {
#ifdef HAVE_CUBLAS
        cuda_float_compare_less_equal(NDArray_SHAPE(nda)[0], NDArray_FDATA(nda), NDArray_FDATA(ndb), NDArray_FDATA(result),
                                      NDArray_NUMELEMENTS(nda));
#endif
    } else {
#ifdef HAVE_AVX2
        // Process 8 elements at a time using AVX2
        i = 0;
        for (; i < NDArray_NUMELEMENTS(a_broad) - 7; i += 8) {
            // Load 8 elements from each array
            __m256 vec1 = _mm256_loadu_ps(&NDArray_FDATA(a_broad)[i]);
            __m256 vec2 = _mm256_loadu_ps(&NDArray_FDATA(b_broad)[i]);

            // Compare elements for equality
            __m256 cmp = _mm256_cmp_ps(vec1, vec2, _CMP_LE_OQ);

            // Convert comparison results to float (1.0 for true, 0.0 for false)
            __m256 resultVec = _mm256_and_ps(cmp, _mm256_set1_ps(1.0f));

            // Store the results
            _mm256_storeu_ps(&NDArray_FDATA(result)[i], resultVec);
        }

        // Process remaining elements using scalar operations
        for (; i < NDArray_NUMELEMENTS(a_broad); i++) {
            NDArray_FDATA(result)[i] = NDArray_FDATA(a_broad)[i] <= NDArray_FDATA(b_broad)[i] ? 1.0f : 0.0f;
        }
#else
        for (i = 0; i < NDArray_NUMELEMENTS(a_broad); i++) {
            NDArray_FDATA(result)[i] = NDArray_FDATA(a_broad)[i] <= NDArray_FDATA(b_broad)[i] ? 1.0f : 0.0f;
        }
#endif
    }
    if (broadcasted != NULL) {
        NDArray_FREE(broadcasted);
    }
    return result;
}

/**
 * Return if NDArray is greater or equal element-wise
 *
 * @param nda
 * @param ndb
 * @return
 */
NDArray*
NDArray_GreaterEqual(NDArray* nda, NDArray* ndb) {
    if (NDArray_DEVICE(nda) != NDArray_DEVICE(ndb)) {
        zend_throw_error(NULL, "Devices mismatch in `equal` function");
        return NULL;
    }

    int i;
    int *rtn_shape = emalloc(sizeof(int) * NDArray_NDIM(nda));

    for (i = 0; i < NDArray_NDIM(nda); i++) {
        rtn_shape[i] = NDArray_SHAPE(nda)[i];
    }

    NDArray *broadcasted = NULL;
    NDArray *a_broad = NULL, *b_broad = NULL;

    if (NDArray_NUMELEMENTS(nda) < NDArray_NUMELEMENTS(ndb)) {
        broadcasted = NDArray_Broadcast(nda, ndb);
        a_broad = broadcasted;
        b_broad = ndb;
    } else if (NDArray_NUMELEMENTS(ndb) < NDArray_NUMELEMENTS(nda)) {
        broadcasted = NDArray_Broadcast(ndb, nda);
        b_broad = broadcasted;
        a_broad = nda;
    } else {
        b_broad = ndb;
        a_broad = nda;
    }

    NDArray *result = NDArray_Empty(rtn_shape, NDArray_NDIM(a_broad), NDArray_TYPE(a_broad), NDArray_DEVICE(a_broad));

    if (b_broad == NULL) {
        zend_throw_error(NULL, "Can't broadcast arrays.");
        return NULL;
    }

    if (NDArray_DEVICE(a_broad) == NDARRAY_DEVICE_GPU) {
#ifdef HAVE_CUBLAS
        cuda_float_compare_greater_equal(NDArray_SHAPE(nda)[0], NDArray_FDATA(nda), NDArray_FDATA(ndb), NDArray_FDATA(result),
                                         NDArray_NUMELEMENTS(nda));
#endif
    } else {
#ifdef HAVE_AVX2
        // Process 8 elements at a time using AVX2
        i = 0;
        for (; i < NDArray_NUMELEMENTS(a_broad) - 7; i += 8) {
            // Load 8 elements from each array
            __m256 vec1 = _mm256_loadu_ps(&NDArray_FDATA(a_broad)[i]);
            __m256 vec2 = _mm256_loadu_ps(&NDArray_FDATA(b_broad)[i]);

            // Compare elements for equality
            __m256 cmp = _mm256_cmp_ps(vec1, vec2, _CMP_GE_OS);

            // Convert comparison results to float (1.0 for true, 0.0 for false)
            __m256 resultVec = _mm256_and_ps(cmp, _mm256_set1_ps(1.0f));

            // Store the results
            _mm256_storeu_ps(&NDArray_FDATA(result)[i], resultVec);
        }

        // Process remaining elements using scalar operations
        for (; i < NDArray_NUMELEMENTS(a_broad); i++) {
            NDArray_FDATA(result)[i] = NDArray_FDATA(a_broad)[i] >= NDArray_FDATA(b_broad)[i] ? 1.0f : 0.0f;
        }
#else
        for (i = 0; i < NDArray_NUMELEMENTS(a_broad); i++) {
            NDArray_FDATA(result)[i] = NDArray_FDATA(a_broad)[i] >= NDArray_FDATA(b_broad)[i] ? 1.0f : 0.0f;
        }
#endif
    }
    if (broadcasted != NULL) {
        NDArray_FREE(broadcasted);
    }
    return result;
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
            __m256 resultVec = _mm256_and_ps(cmp, _mm256_set1_ps(1.0f));

            // Store the results
            _mm256_storeu_ps(&NDArray_FDATA(result)[i], resultVec);
        }

        // Process remaining elements using scalar operations
        for (; i < NDArray_NUMELEMENTS(nda); i++) {
            NDArray_FDATA(result)[i] = (fabsf(NDArray_FDATA(nda)[i] - NDArray_FDATA(ndb)[i]) <= 0.0000001f) ? 1.0f : 0.0f;
        }
        return result;
#else
        for (i = 0; i < NDArray_NUMELEMENTS(nda); i++) {
            NDArray_FDATA(result)[i] = (fabsf(NDArray_FDATA(nda)[i] - NDArray_FDATA(ndb)[i]) <= 0.0000001f) ? 1.0f : 0.0f;
        }
#endif
    }
    return result;
}

/**
 * Return if NDArray is not equal element-wise
 *
 * @param nda
 * @param ndb
 * @return
 */
NDArray*
NDArray_NotEqual(NDArray* nda, NDArray* ndb) {
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
        cuda_float_compare_not_equal(NDArray_SHAPE(nda)[0], NDArray_FDATA(nda), NDArray_FDATA(ndb), NDArray_FDATA(result),
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
            __m256 cmp = _mm256_cmp_ps(vec1, vec2, _CMP_NEQ_OQ);

            // Convert comparison results to float (1.0 for true, 0.0 for false)
            __m256 resultVec = _mm256_and_ps(cmp, _mm256_set1_ps(1.0f));

            // Store the results
            _mm256_storeu_ps(&NDArray_FDATA(result)[i], resultVec);
        }

        // Process remaining elements using scalar operations
        for (; i < NDArray_NUMELEMENTS(nda); i++) {
            NDArray_FDATA(result)[i] = (fabsf(NDArray_FDATA(nda)[i] - NDArray_FDATA(ndb)[i]) <= 0.0000001f) ? 0.0f : 1.0f;
        }
        return result;
#else
        for (i = 0; i < NDArray_NUMELEMENTS(nda); i++) {
            NDArray_FDATA(result)[i] = (fabsf(NDArray_FDATA(nda)[i] - NDArray_FDATA(ndb)[i]) <= 0.0000001f) ? 0.0f : 1.0f;
        }
#endif
    }
    return result;
}

/**
 *
 * @return
 */
int
compare_ndarrays(NDArray *a, NDArray *b) {
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
NDArray_ArrayEqual(NDArray *a, NDArray *b) {
    int i;
    if (NDArray_NDIM(a) != NDArray_NDIM(b)) {
        return 0;
    }

    for (i = 0; i < NDArray_NDIM(a); i++) {
        if (NDArray_SHAPE(a)[i] != NDArray_SHAPE(b)[i]) {
            return 0;
        }
    }

    return compare_ndarrays(a, b);
}

int
float_allclose(float* arr1, float* arr2, const int* shape,
               const int* strides_a, const int* strided_b,
               int ndim, float atol, float rtol) {
    int totalElements = 1;
    for (int i = 0; i < ndim; i++) {
        totalElements *= shape[i];
    }
    unsigned long index_a, index_b;
    for (int i = 0; i < totalElements; i++) {
        index_a = (i * sizeof(float)) + (i * strides_a[0]/sizeof(float));
        index_b = (i * sizeof(float)) + (i * strided_b[0]/sizeof(float));
        float diff = fabsf(arr1[(int)index_a] - arr2[(int)index_b]);
        float tolerance = atol + rtol * fabsf(arr2[index_b]);
        if (diff > tolerance) {
            return false;
        }
    }

    return true;
}

/**
 * NDArray::allclose
 *
 * @param a
 * @param b
 * @param rtol
 * @param atol
 * @return
 */
int
NDArray_AllClose(NDArray* a, NDArray *b, float rtol, float atol) {
    if (NDArray_DEVICE(a) == NDARRAY_DEVICE_GPU || NDArray_DEVICE(b) == NDARRAY_DEVICE_GPU)
    {
        zend_throw_error(NULL, "`allclose` is not compatible with GPU operations.");
    }

    if (NDArray_ShapeCompare(a, b) == 0) {
        zend_throw_error(NULL, "Shape mismatch");
        return -1;
    }

    if (NDArray_DEVICE(a) != NDArray_DEVICE(b)) {
        zend_throw_error(NULL, "NDArray::allclose() requires both arrays to be on the same device (CPU or GPU).");
        return -1;
    }

    if (NDArray_DEVICE(a) == NDARRAY_DEVICE_CPU) {
        return float_allclose(NDArray_FDATA(a), NDArray_FDATA(b), NDArray_SHAPE(a),
                              NDArray_STRIDES(a), NDArray_STRIDES(b),
                              NDArray_NDIM(a), atol, rtol);
    }
    return -1;
}