#include <php.h>
#include "Zend/zend_alloc.h"
#include "Zend/zend_API.h"
#include <string.h>
#include "arithmetics.h"
#include "../ndarray.h"
#include "../../config.h"
#include "../initializers.h"
#include "../iterators.h"
#include "../types.h"
#include "../debug.h"
#include "linalg.h"
#include "../manipulation.h"
#include "double_math.h"

#ifdef HAVE_CUBLAS
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "cuda/cuda_math.h"
#include "../gpu_alloc.h"

#endif

#ifdef HAVE_CBLAS
#include <cblas.h>
#endif

#ifdef HAVE_AVX2
#include <immintrin.h>
#endif

/**
 * Add elements of a and b element-wise
 *
 * @param a
 * @param b
 * @return
 */
double
NDArray_Sum_Double(NDArray* a) {
    double value = 0;

#ifdef HAVE_CBLAS
    value = cblas_dasum(NDArray_NUMELEMENTS(a), NDArray_DDATA(a), 1);
#else
    for (int i = 0; i < NDArray_NUMELEMENTS(a); i++) {
        value += NDArray_DDATA(a)[i];
    }
#endif
    return value;
}

/**
 * Product of array element-wise
 *
 * @param a
 * @param b
 * @return
 */
float
NDArray_Float_Prod(NDArray* a) {
    float value = 1;
    if (NDArray_DEVICE(a) == NDARRAY_DEVICE_GPU) {
#ifdef HAVE_CUBLAS
        cuda_prod_float(NDArray_NUMELEMENTS(a), NDArray_FDATA(a), &value, NDArray_NUMELEMENTS(a));
#endif
    } else {
        for (int i = 0; i < NDArray_NUMELEMENTS(a); i++) {
            value *= NDArray_FDATA(a)[i];
        }
    }
    return value;
}

/**
 * Add elements of a element-wise
 *
 * @param a
 * @param b
 * @return
 */
float
NDArray_Sum_Float(NDArray* a) {
    float value = 0;
    if (NDArray_DEVICE(a) == NDARRAY_DEVICE_GPU) {
#ifdef HAVE_CUBLAS
        cuda_sum_float(NDArray_NUMELEMENTS(a), NDArray_FDATA(a), &value, NDArray_NUMELEMENTS(a));
#endif
    } else {
        for (int i = 0; i < NDArray_NUMELEMENTS(a); i++) {
            value += NDArray_FDATA(a)[i];
        }
    }
    return value;
}

/**
 * Add elements of a element-wise
 *
 * @param a
 * @param b
 * @return
 */
float
NDArray_Mean_Float(NDArray* a) {
    float value = 0;
    if (NDArray_DEVICE(a) == NDARRAY_DEVICE_GPU) {
#ifdef HAVE_CUBLAS
        //cublasHandle_t handle;
        //cublasCreate(&handle);
        //cublasSasum(handle, NDArray_NUMELEMENTS(a), NDArray_FDATA(a), 1, &value);
        cuda_sum_float(NDArray_NUMELEMENTS(a), NDArray_FDATA(a), &value, NDArray_NUMELEMENTS(a));
        value = value / NDArray_NUMELEMENTS(a);
#endif
    } else {

#ifdef HAVE_CBLAS
        value = cblas_sasum(NDArray_NUMELEMENTS(a), NDArray_FDATA(a), 1);
        value = value / NDArray_NUMELEMENTS(a);
#else
        for (int i = 0; i < NDArray_NUMELEMENTS(a); i++) {
            value += NDArray_FDATA(a)[i];
        }
        value = value / NDArray_NUMELEMENTS(a);
#endif
    }
    return value;
}

// Comparison function for sorting
int compare(const void* a, const void* b) {
    float fa = *((const float*)a);
    float fb = *((const float*)b);
    return (fa > fb) - (fa < fb);
}

float calculate_median(float* matrix, int size) {
    // Copy matrix elements to a separate array
    float* temp = malloc(size * sizeof(float));
    if (temp == NULL) {
        // Handle memory allocation error
        fprintf(stderr, "Memory allocation failed.\n");
        exit(1);
    }
    memcpy(temp, matrix, size * sizeof(float));

    // Sort the array in ascending order
    qsort(temp, size, sizeof(float), compare);

    // Calculate the median value
    float median;
    if (size % 2 == 0) {
        // If the number of elements is even, average the two middle values
        median = (temp[size / 2 - 1] + temp[size / 2]) / 2.0;
    } else {
        // If the number of elements is odd, take the middle value
        median = temp[size / 2];
    }

    // Free the temporary array
    free(temp);

    return median;
}

/**
 * Add elements of a element-wise
 *
 * @todo Implement GPU support
 * @param a
 * @param b
 * @return
 */
float
NDArray_Median_Float(NDArray* a) {
    if (NDArray_DEVICE(a) == NDARRAY_DEVICE_GPU) {
#ifdef HAVE_CUBLAS
        zend_throw_error(NULL, "Median not available for GPU.");
        return -1;
#endif
    } else {
        return calculate_median(NDArray_FDATA(a), NDArray_NUMELEMENTS(a));
    }
}

NDArray*
NDArray_Add_Float(NDArray* a, NDArray* b) {
    NDArray *broadcasted = NULL;
    if (NDArray_DEVICE(a) != NDArray_DEVICE(b)) {
        zend_throw_error(NULL, "Device mismatch, both NDArray MUST be in the same device.");
        return NULL;
    }

    NDArray *a_broad = NULL, *b_broad = NULL;

    if (NDArray_NDIM(a) == 0 && NDArray_NDIM(b) == 0) {
        int* shape = ecalloc(1, sizeof(int));
        NDArray *rtn = NDArray_Zeros(shape, 0, NDARRAY_TYPE_FLOAT32, NDArray_DEVICE(a));
#ifdef HAVE_CUBLAS
        if (NDArray_DEVICE(a) == NDARRAY_DEVICE_GPU) {
            cuda_add_float(2, NDArray_FDATA(a), NDArray_FDATA(b), NDArray_FDATA(rtn), 1);
        } else {
#endif
            NDArray_FDATA(rtn)[0] = NDArray_FDATA(a)[0] + NDArray_FDATA(b)[0];
#ifdef HAVE_CUBLAS
        }
#endif
        return rtn;
    }

    if (NDArray_NUMELEMENTS(a) < NDArray_NUMELEMENTS(b)) {
        broadcasted = NDArray_Broadcast(a, b);
        a_broad = broadcasted;
        b_broad = b;
    } else if (NDArray_NUMELEMENTS(b) < NDArray_NUMELEMENTS(a)) {
        broadcasted = NDArray_Broadcast(b, a);
        b_broad = broadcasted;
        a_broad = a;
    } else {
        b_broad = b;
        a_broad = a;
    }

    if (b_broad == NULL || a_broad == NULL) {
        zend_throw_error(NULL, "Can't broadcast arrays.");
        return NULL;
    }

    // Check if the element size of the input arrays match
    if (a->descriptor->elsize != sizeof(float) || b->descriptor->elsize != sizeof(float)) {
        // Element size mismatch, return an error or handle it accordingly
        return NULL;
    }

    // Create a new NDArray to store the result
    NDArray* result = (NDArray*)emalloc(sizeof(NDArray));
    result->strides = (int*)emalloc(a_broad->ndim * sizeof(int));
    result->dimensions = (int*)emalloc(a_broad->ndim * sizeof(int));
    result->ndim = a_broad->ndim;
    if (NDArray_DEVICE(a_broad) == NDARRAY_DEVICE_GPU) {
#if HAVE_CUBLAS
        NDArray_VMALLOC((void **) &result->data, NDArray_NUMELEMENTS(a_broad) * sizeof(float));
        result->device = NDARRAY_DEVICE_GPU;
#endif
    } else {
        result->data = (char *) emalloc(a_broad->descriptor->numElements * sizeof(float));
    }
    result->base = NULL;
    result->flags = 0;  // Set appropriate flags
    result->descriptor = (NDArrayDescriptor*)emalloc(sizeof(NDArrayDescriptor));
    result->descriptor->type = NDARRAY_TYPE_FLOAT32;
    result->descriptor->elsize = sizeof(float);
    result->descriptor->numElements = a_broad->descriptor->numElements;
    result->refcount = 1;

    // Perform element-wise addition
    result->strides = memcpy(result->strides, a_broad->strides, a_broad->ndim * sizeof(int));
    result->dimensions = memcpy(result->dimensions, a_broad->dimensions, a_broad->ndim * sizeof(int));
    float* resultData = (float*)result->data;
    float* aData = (float*)a_broad->data;
    float* bData = (float*)b_broad->data;
    int numElements = a_broad->descriptor->numElements;
    NDArrayIterator_INIT(result);
    if (NDArray_DEVICE(a_broad) == NDARRAY_DEVICE_GPU && NDArray_DEVICE(b_broad) == NDARRAY_DEVICE_GPU) {
#if HAVE_CUBLAS
        cuda_add_float(NDArray_NUMELEMENTS(a_broad), NDArray_FDATA(a_broad), NDArray_FDATA(b_broad), NDArray_FDATA(result), NDArray_NUMELEMENTS(a_broad));
        result->device = NDARRAY_DEVICE_GPU;
#endif
    } else {
#ifdef HAVE_AVX2
        int i;
        __m256 vec1, vec2, mul;

        for (i = 0; i < NDArray_NUMELEMENTS(a) - 7; i += 8) {
            vec1 = _mm256_loadu_ps(&aData[i]);
            vec2 = _mm256_loadu_ps(&bData[i]);
            mul = _mm256_add_ps(vec1, vec2);
            _mm256_storeu_ps(&resultData[i], mul);
        }
        // Handle remaining elements if the length is not a multiple of 4
        for (; i < NDArray_NUMELEMENTS(a); i++) {
            resultData[i] = aData[i] + bData[i];
        }
#elif HAVE_CBLAS
        if (NDArray_NUMELEMENTS(a_broad) == NDArray_NUMELEMENTS(b_broad)) {
            memcpy(resultData, NDArray_FDATA(b_broad), NDArray_ELSIZE(b_broad) * NDArray_NUMELEMENTS(b_broad));
            cblas_saxpy(NDArray_NUMELEMENTS(a_broad), 1.0F, NDArray_FDATA(a_broad), 1, resultData,
                        1);
        } else {
            for (int i = 0; i < numElements; i++) {
                resultData[i] = aData[i] + bData[i];
            }
        }
#else
        for (int i = 0; i < numElements; i++) {
            resultData[i] = aData[i] + bData[i];
        }
#endif
    }

    if (broadcasted != NULL) {
        NDArray_FREE(broadcasted);
    }
    return result;
}

/**
 * Multiply elements of a and b element-wise
 *
 * @param a
 * @param b
 * @return
 */
NDArray*
NDArray_Multiply_Float(NDArray* a, NDArray* b) {
    NDArray *broadcasted = NULL;
    NDArray *a_temp = NULL, *b_temp = NULL;
    if (NDArray_DEVICE(a) != NDArray_DEVICE(b)) {
        zend_throw_error(NULL, "Device mismatch, both NDArray MUST be in the same device.");
        return NULL;
    }

    if (NDArray_NDIM(a) == 0 && NDArray_NDIM(b) == 0) {
        if (NDArray_DEVICE(a) == NDARRAY_DEVICE_GPU) {
#ifdef HAVE_CUBLAS
            int *shape = ecalloc(1, sizeof(int));
            NDArray *rtn = NDArray_Empty(shape, 0, NDARRAY_TYPE_FLOAT32, NDARRAY_DEVICE_GPU);
            cuda_multiply_float(1, NDArray_FDATA(a), NDArray_FDATA(b), NDArray_FDATA(rtn), 1);
            return rtn;
#endif
        } else {
            int *shape = ecalloc(1, sizeof(int));
            NDArray *rtn = NDArray_Empty(shape, 0, NDARRAY_TYPE_FLOAT32, NDARRAY_DEVICE_CPU);
            NDArray_FDATA(rtn)[0] = NDArray_FDATA(a)[0] * NDArray_FDATA(b)[0];
            return rtn;
        }
    }

    // If a or b are scalars, reshape
    if (NDArray_NDIM(a) == 0 && NDArray_NDIM(b) > 0) {
        a_temp = a;
        int *n_shape = emalloc(sizeof(int) * NDArray_NDIM(b));
        copy(NDArray_SHAPE(b), n_shape, NDArray_NDIM(b));
        a = NDArray_Zeros(n_shape, NDArray_NDIM(b), NDArray_TYPE(b), NDArray_DEVICE(b));
        a = NDArray_Fill(a, NDArray_FDATA(a_temp)[0]);
    } else if (NDArray_NDIM(b) == 0 && NDArray_NDIM(a) > 0) {
        b_temp = b;
        int *n_shape = emalloc(sizeof(int) * NDArray_NDIM(a));
        copy(NDArray_SHAPE(a), n_shape, NDArray_NDIM(a));
        b = NDArray_Zeros(n_shape, NDArray_NDIM(a), NDArray_TYPE(a), NDArray_DEVICE(a));
        b = NDArray_Fill(b, NDArray_FDATA(b_temp)[0]);
    }

    NDArray *a_broad = NULL, *b_broad = NULL;

    if (NDArray_NUMELEMENTS(a) < NDArray_NUMELEMENTS(b)) {
        broadcasted = NDArray_Broadcast(a, b);
        a_broad = broadcasted;
        b_broad = b;
    } else if (NDArray_NUMELEMENTS(b) < NDArray_NUMELEMENTS(a)) {
        broadcasted = NDArray_Broadcast(b, a);
        b_broad = broadcasted;
        a_broad = a;
    } else {
        b_broad = b;
        a_broad = a;
    }

    if (b_broad == NULL || a_broad == NULL) {
        zend_throw_error(NULL, "Can't broadcast arrays.");
        return NULL;
    }

    // Check if the element size of the input arrays match
    if (a->descriptor->elsize != sizeof(float) || b->descriptor->elsize != sizeof(float)) {
        // Element size mismatch, return an error or handle it accordingly
        return NULL;
    }

    // Create a new NDArray to store the result
    NDArray *result = (NDArray *) emalloc(sizeof(NDArray));
    result->strides = (int *) emalloc(a_broad->ndim * sizeof(int));
    result->dimensions = (int *) emalloc(a_broad->ndim * sizeof(int));
    result->ndim = a->ndim;
    result->device = NDArray_DEVICE(a_broad);
    if (NDArray_DEVICE(a_broad) == NDARRAY_DEVICE_GPU) {
#if HAVE_CUBLAS
        NDArray_VMALLOC((void **) &result->data, NDArray_NUMELEMENTS(a_broad) * sizeof(float));
        result->device = NDARRAY_DEVICE_GPU;
#endif
    } else {
        result->data = (char *) emalloc(a_broad->descriptor->numElements * sizeof(float));
    }
    result->base = NULL;
    result->flags = 0;  // Set appropriate flags
    result->descriptor = (NDArrayDescriptor *) emalloc(sizeof(NDArrayDescriptor));
    result->descriptor->type = NDARRAY_TYPE_FLOAT32;
    result->descriptor->elsize = sizeof(float);
    result->descriptor->numElements = a_broad->descriptor->numElements;
    result->refcount = 1;

    // Perform element-wise product
    result->strides = memcpy(result->strides, a_broad->strides, a_broad->ndim * sizeof(int));
    result->dimensions = memcpy(result->dimensions, a_broad->dimensions, a_broad->ndim * sizeof(int));
    float *resultData = (float *) result->data;
    float *aData = (float *) a_broad->data;
    float *bData = (float *) b_broad->data;
    int numElements = a_broad->descriptor->numElements;
    NDArrayIterator_INIT(result);
    if (NDArray_DEVICE(a_broad) == NDARRAY_DEVICE_GPU && NDArray_DEVICE(b_broad) == NDARRAY_DEVICE_GPU) {
#if HAVE_CUBLAS
        cuda_multiply_float(NDArray_NUMELEMENTS(a_broad), NDArray_FDATA(a_broad), NDArray_FDATA(b_broad), NDArray_FDATA(result),
                            NDArray_NUMELEMENTS(a_broad));
        result->device = NDARRAY_DEVICE_GPU;
#endif
    } else {
#ifdef HAVE_AVX2
        int i;
        __m256 vec1, vec2, mul;

        for (i = 0; i < NDArray_NUMELEMENTS(a) - 7; i += 8) {
            vec1 = _mm256_loadu_ps(&aData[i]);
            vec2 = _mm256_loadu_ps(&bData[i]);
            mul = _mm256_mul_ps(vec1, vec2);
            _mm256_storeu_ps(&resultData[i], mul);
        }

        // Handle remaining elements if the length is not a multiple of 4
        for (; i < NDArray_NUMELEMENTS(a); i++) {
            resultData[i] = aData[i] * bData[i];
        }
#else
        for (int i = 0; i < numElements; i++) {
            resultData[i] = aData[i] * bData[i];
        }
#endif
    }
    if (a_temp != NULL) {
        NDArray_FREE(a);
    }
    if (b_temp != NULL) {
        NDArray_FREE(b);
    }
    if (broadcasted != NULL) {
        NDArray_FREE(broadcasted);
    }
    return result;
}

/**
 * Subtract elements of a and b element-wise
 *
 * @param a
 * @param b
 * @return
 */
NDArray*
NDArray_Subtract_Float(NDArray* a, NDArray* b) {
    NDArray *a_temp = NULL, *b_temp = NULL;
    if (NDArray_DEVICE(a) != NDArray_DEVICE(b)) {
        zend_throw_error(NULL, "Device mismatch, both NDArray MUST be in the same device.");
        return NULL;
    }

    // If a or b are scalars, reshape
    if (NDArray_NDIM(a) == 0 && NDArray_NDIM(b) > 0) {
        a_temp = a;
        int *n_shape = emalloc(sizeof(int) * NDArray_NDIM(b));
        copy(NDArray_SHAPE(b), n_shape, NDArray_NDIM(b));
        a = NDArray_Zeros(n_shape, NDArray_NDIM(b), NDArray_TYPE(b), NDArray_DEVICE(b));
        a = NDArray_Fill(a, NDArray_FDATA(a_temp)[0]);
    } else if (NDArray_NDIM(b) == 0 && NDArray_NDIM(a) > 0) {
        b_temp = b;
        int *n_shape = emalloc(sizeof(int) * NDArray_NDIM(a));
        copy(NDArray_SHAPE(a), n_shape, NDArray_NDIM(a));
        b = NDArray_Zeros(n_shape, NDArray_NDIM(a), NDArray_TYPE(a), NDArray_DEVICE(a));
        b = NDArray_Fill(b, NDArray_FDATA(b_temp)[0]);
    }

    NDArray *broadcasted = NULL;
    NDArray *a_broad = NULL, *b_broad = NULL;

    if (NDArray_NUMELEMENTS(a) < NDArray_NUMELEMENTS(b)) {
        broadcasted = NDArray_Broadcast(a, b);
        a_broad = broadcasted;
        b_broad = b;
    } else if (NDArray_NUMELEMENTS(b) < NDArray_NUMELEMENTS(a)) {
        broadcasted = NDArray_Broadcast(b, a);
        b_broad = broadcasted;
        a_broad = a;
    } else {
        b_broad = b;
        a_broad = a;
    }

    if (b_broad == NULL || a_broad == NULL) {
        zend_throw_error(NULL, "Can't broadcast arrays.");
        return NULL;
    }

    // Check if the element size of the input arrays match
    if (a->descriptor->elsize != sizeof(float) || b->descriptor->elsize != sizeof(float)) {
        // Element size mismatch, return an error or handle it accordingly
        return NULL;
    }

    // Create a new NDArray to store the result
    NDArray *result = (NDArray *) emalloc(sizeof(NDArray));
    result->strides = (int *) emalloc(a_broad->ndim * sizeof(int));
    result->dimensions = (int *) emalloc(a_broad->ndim * sizeof(int));
    result->ndim = a_broad->ndim;
    if (NDArray_DEVICE(a_broad) == NDARRAY_DEVICE_GPU) {
#if HAVE_CUBLAS
        NDArray_VMALLOC((void **) &result->data, NDArray_NUMELEMENTS(a_broad) * sizeof(float));
        cudaDeviceSynchronize();
        result->device = NDARRAY_DEVICE_GPU;
#endif
    } else {
        result->data = (char *) emalloc(a_broad->descriptor->numElements * sizeof(float));
    }
    result->base = NULL;
    result->flags = 0;  // Set appropriate flags
    result->descriptor = (NDArrayDescriptor *) emalloc(sizeof(NDArrayDescriptor));
    result->descriptor->type = NDARRAY_TYPE_FLOAT32;
    result->descriptor->elsize = sizeof(float);
    result->descriptor->numElements = a_broad->descriptor->numElements;
    result->refcount = 1;
    result->device = NDArray_DEVICE(a_broad);

    // Perform element-wise subtraction
    result->strides = memcpy(result->strides, a_broad->strides, a_broad->ndim * sizeof(int));
    result->dimensions = memcpy(result->dimensions, a_broad->dimensions, a_broad->ndim * sizeof(int));
    float *resultData = (float *) result->data;
    float *aData = (float *) a_broad->data;
    float *bData = (float *) b_broad->data;
    int numElements = a_broad->descriptor->numElements;
    NDArrayIterator_INIT(result);
    if (NDArray_DEVICE(a_broad) == NDARRAY_DEVICE_GPU && NDArray_DEVICE(b_broad) == NDARRAY_DEVICE_GPU) {
#if HAVE_CUBLAS
        cuda_subtract_float(NDArray_NUMELEMENTS(a_broad), NDArray_FDATA(a_broad), NDArray_FDATA(b_broad), NDArray_FDATA(result),
                            NDArray_NUMELEMENTS(a_broad));
#endif
    } else {
#ifdef HAVE_AVX2
        int i;
        __m256 vec1, vec2, sub;

        for (i = 0; i < NDArray_NUMELEMENTS(a) - 7; i += 8) {
            vec1 = _mm256_loadu_ps(&aData[i]);
            vec2 = _mm256_loadu_ps(&bData[i]);
            sub = _mm256_sub_ps(vec1, vec2);
            _mm256_storeu_ps(&resultData[i], sub);
        }

        // Handle remaining elements if the length is not a multiple of 4
        for (; i < NDArray_NUMELEMENTS(a); i++) {
            resultData[i] = aData[i] - bData[i];
        }
#else
        for (int i = 0; i < numElements; i++) {
            resultData[i] = aData[i] - bData[i];
        }
#endif
    }
    if (a_temp != NULL) {
        NDArray_FREE(a);
    }
    if (b_temp != NULL) {
        NDArray_FREE(b);
    }
    if (broadcasted != NULL) {
        NDArray_FREE(broadcasted);
    }
    return result;
}

/**
 * Divide elements of a and b element-wise
 *
 * @param a
 * @param b
 * @return
 */
NDArray*
NDArray_Divide_Float(NDArray* a, NDArray* b) {
    NDArray *a_temp = NULL, *b_temp = NULL;
    php_printf("\n %d != %d ? \n", NDArray_DEVICE(a), NDArray_DEVICE(b));
    if (NDArray_DEVICE(a) != NDArray_DEVICE(b)) {
        zend_throw_error(NULL, "Device mismatch, both NDArray MUST be in the same device.");
        return NULL;
    }

    if (NDArray_NDIM(a) == 0 && NDArray_NDIM(b) == 0) {
        int *shape = ecalloc(1, sizeof(int));
        NDArray *rtn = NDArray_Zeros(shape, 0, NDARRAY_TYPE_FLOAT32, NDArray_DEVICE(a));
        NDArray_FDATA(rtn)[0] = NDArray_FDATA(a)[0] / NDArray_FDATA(b)[0];
        return rtn;
    }

    // If a or b are scalars, reshape
    if (NDArray_NDIM(a) == 0 && NDArray_NDIM(b) > 0) {
        a_temp = a;
        int *n_shape = emalloc(sizeof(int) * NDArray_NDIM(b));
        copy(NDArray_SHAPE(b), n_shape, NDArray_NDIM(b));
        a = NDArray_Zeros(n_shape, NDArray_NDIM(b), NDArray_TYPE(b), NDArray_DEVICE(b));
        a = NDArray_Fill(a, NDArray_FDATA(a_temp)[0]);
    } else if (NDArray_NDIM(b) == 0 && NDArray_NDIM(a) > 0) {
        b_temp = b;
        int *n_shape = emalloc(sizeof(int) * NDArray_NDIM(a));
        copy(NDArray_SHAPE(a), n_shape, NDArray_NDIM(a));
        b = NDArray_Zeros(n_shape, NDArray_NDIM(a), NDArray_TYPE(a), NDArray_DEVICE(a));
        b = NDArray_Fill(b, NDArray_FDATA(b_temp)[0]);
    }

    NDArray *broadcasted = NULL;
    NDArray *a_broad = NULL, *b_broad = NULL;

    if (NDArray_NUMELEMENTS(a) < NDArray_NUMELEMENTS(b)) {
        broadcasted = NDArray_Broadcast(a, b);
        a_broad = broadcasted;
        b_broad = b;
    } else if (NDArray_NUMELEMENTS(b) < NDArray_NUMELEMENTS(a)) {
        broadcasted = NDArray_Broadcast(b, a);
        b_broad = broadcasted;
        a_broad = a;
    } else {
        b_broad = b;
        a_broad = a;
    }

    if (b_broad == NULL || a_broad == NULL) {
        zend_throw_error(NULL, "Can't broadcast arrays.");
        return NULL;
    }

    // Check if the element size of the input arrays match
    if (a->descriptor->elsize != sizeof(float) || b->descriptor->elsize != sizeof(float)) {
        // Element size mismatch, return an error or handle it accordingly
        return NULL;
    }

    // Create a new NDArray to store the result
    NDArray *result = (NDArray *) emalloc(sizeof(NDArray));
    result->strides = (int *) emalloc(a_broad->ndim * sizeof(int));
    result->dimensions = (int *) emalloc(a_broad->ndim * sizeof(int));
    result->device = NDArray_DEVICE(a_broad);
    result->ndim = a_broad->ndim;
    if (NDArray_DEVICE(a_broad) == NDARRAY_DEVICE_GPU) {
#if HAVE_CUBLAS
        NDArray_VMALLOC((void **) &result->data, NDArray_NUMELEMENTS(a_broad) * sizeof(float));
        cudaDeviceSynchronize();
        result->device = NDARRAY_DEVICE_GPU;
#endif
    } else {
        result->data = (char *) emalloc(a_broad->descriptor->numElements * sizeof(float));
    }
    result->base = NULL;
    result->flags = 0;  // Set appropriate flags
    result->descriptor = (NDArrayDescriptor *) emalloc(sizeof(NDArrayDescriptor));
    result->descriptor->type = NDARRAY_TYPE_FLOAT32;
    result->descriptor->elsize = sizeof(float);
    result->descriptor->numElements = a_broad->descriptor->numElements;
    result->refcount = 1;

    // Perform element-wise division
    result->strides = memcpy(result->strides, a_broad->strides, a_broad->ndim * sizeof(int));
    result->dimensions = memcpy(result->dimensions, a_broad->dimensions, a_broad->ndim * sizeof(int));
    float *resultData = (float *) result->data;
    float *aData = (float *) a_broad->data;
    float *bData = (float *) b_broad->data;
    int numElements = a_broad->descriptor->numElements;
    NDArrayIterator_INIT(result);
    if (NDArray_DEVICE(a_broad) == NDARRAY_DEVICE_GPU && NDArray_DEVICE(b_broad) == NDARRAY_DEVICE_GPU) {
#if HAVE_CUBLAS
        cuda_divide_float(NDArray_NUMELEMENTS(a_broad), NDArray_FDATA(a_broad), NDArray_FDATA(b_broad), NDArray_FDATA(result),
                          NDArray_NUMELEMENTS(a_broad));
#endif
    } else {
#ifdef HAVE_AVX2
        int i;
        __m256 vec1, vec2, sub;

        for (i = 0; i < NDArray_NUMELEMENTS(a) - 7; i += 8) {
            vec1 = _mm256_loadu_ps(&aData[i]);
            vec2 = _mm256_loadu_ps(&bData[i]);
            sub = _mm256_div_ps(vec1, vec2);
            _mm256_storeu_ps(&resultData[i], sub);
        }

        // Handle remaining elements if the length is not a multiple of 4
        for (; i < NDArray_NUMELEMENTS(a); i++) {
            resultData[i] = aData[i] / bData[i];
        }
#else
        for (int i = 0; i < numElements; i++) {
            resultData[i] = aData[i] / bData[i];
        }
#endif
    }
    if (a_temp != NULL) {
        NDArray_FREE(a);
    }
    if (b_temp != NULL) {
        NDArray_FREE(b);
    }
    if (broadcasted != NULL) {
        NDArray_FREE(broadcasted);
    }
    return result;
}

/**
 * @param a
 * @param b
 * @return
 */
NDArray*
NDArray_Mod_Float(NDArray* a, NDArray* b) {
    NDArray *a_temp = NULL, *b_temp = NULL;
    if (NDArray_DEVICE(a) != NDArray_DEVICE(b)) {
        zend_throw_error(NULL, "Device mismatch, both NDArray MUST be in the same device.");
        return NULL;
    }

    if (NDArray_NDIM(a) == 0) {
        int* shape = ecalloc(1, sizeof(int));
        NDArray *rtn = NDArray_Zeros(shape, 0, NDARRAY_TYPE_FLOAT32, NDArray_DEVICE(a));
        NDArray_FDATA(rtn)[0] = NDArray_FDATA(a)[0] + NDArray_FDATA(b)[0];
        return rtn;
    }

    // If a or b are scalars, reshape
    if (NDArray_NDIM(a) == 0 && NDArray_NDIM(b) > 0) {
        a_temp = a;
        int *n_shape = emalloc(sizeof(int) * NDArray_NDIM(b));
        copy(NDArray_SHAPE(b), n_shape, NDArray_NDIM(b));
        a = NDArray_Zeros(n_shape, NDArray_NDIM(b), NDArray_TYPE(b), NDArray_DEVICE(b));
        a = NDArray_Fill(a, NDArray_FDATA(a_temp)[0]);
    } else if (NDArray_NDIM(b) == 0 && NDArray_NDIM(a) > 0) {
        b_temp = b;
        int *n_shape = emalloc(sizeof(int) * NDArray_NDIM(a));
        copy(NDArray_SHAPE(a), n_shape, NDArray_NDIM(a));
        b = NDArray_Zeros(n_shape, NDArray_NDIM(a), NDArray_TYPE(a), NDArray_DEVICE(a));
        b = NDArray_Fill(b, NDArray_FDATA(b_temp)[0]);
    }

    NDArray *broadcasted = NULL;
    NDArray *a_broad = NULL, *b_broad = NULL;
    if (NDArray_NUMELEMENTS(a) < NDArray_NUMELEMENTS(b)) {
        broadcasted = NDArray_Broadcast(a, b);
        a_broad = broadcasted;
        b_broad = b;
    } else if (NDArray_NUMELEMENTS(b) < NDArray_NUMELEMENTS(a)) {
        broadcasted = NDArray_Broadcast(b, a);
        b_broad = broadcasted;
        a_broad = a;
    } else {
        b_broad = b;
        a_broad = a;
    }

    if (b_broad == NULL || a_broad == NULL) {
        zend_throw_error(NULL, "Can't broadcast arrays.");
        return NULL;
    }

    // Check if the element size of the input arrays match
    if (a->descriptor->elsize != sizeof(float) || b->descriptor->elsize != sizeof(float)) {
        // Element size mismatch, return an error or handle it accordingly
        return NULL;
    }

    // Create a new NDArray to store the result
    NDArray* result = (NDArray*)emalloc(sizeof(NDArray));
    result->strides = (int*)emalloc(a_broad->ndim * sizeof(int));
    result->dimensions = (int*)emalloc(a_broad->ndim * sizeof(int));
    result->ndim = a_broad->ndim;
    if (NDArray_DEVICE(a_broad) == NDARRAY_DEVICE_GPU) {
#if HAVE_CUBLAS
        NDArray_VMALLOC((void **) &result->data, NDArray_NUMELEMENTS(a_broad) * sizeof(float));
        cudaDeviceSynchronize();
        result->device = NDARRAY_DEVICE_GPU;
#endif
    } else {
        result->data = (char *) emalloc(a_broad->descriptor->numElements * sizeof(float));
    }
    result->base = NULL;
    result->flags = 0;  // Set appropriate flags
    result->descriptor = (NDArrayDescriptor*)emalloc(sizeof(NDArrayDescriptor));
    result->descriptor->type = NDARRAY_TYPE_FLOAT32;
    result->descriptor->elsize = sizeof(float);
    result->descriptor->numElements = a_broad->descriptor->numElements;
    result->refcount = 1;
    result->device = NDArray_DEVICE(a_broad);

    // Perform element-wise subtraction
    result->strides = memcpy(result->strides, a_broad->strides, a_broad->ndim * sizeof(int));
    result->dimensions = memcpy(result->dimensions, a_broad->dimensions, a_broad->ndim * sizeof(int));
    float* resultData = (float*)result->data;
    float* aData = (float*)a_broad->data;
    float* bData = (float*)b_broad->data;
    int numElements = a_broad->descriptor->numElements;
    NDArrayIterator_INIT(result);
    if (NDArray_DEVICE(a_broad) == NDARRAY_DEVICE_GPU && NDArray_DEVICE(b_broad) == NDARRAY_DEVICE_GPU) {
#if HAVE_CUBLAS
        cuda_mod_float(NDArray_NUMELEMENTS(a_broad), NDArray_FDATA(a_broad), NDArray_FDATA(b_broad), NDArray_FDATA(result),
                       NDArray_NUMELEMENTS(a_broad));
#endif
    } else {
        for (int i = 0; i < numElements; i++) {
            resultData[i] = fmodf(aData[i], bData[i]);
        }
    }

    if (a_temp != NULL) {
        NDArray_FREE(a);
    }
    if (b_temp != NULL) {
        NDArray_FREE(b);
    }
    if (broadcasted != NULL) {
        NDArray_FREE(broadcasted);
    }
    return result;
}

/**
 * @param a
 * @param b
 * @return
 */
NDArray*
NDArray_Pow_Float(NDArray* a, NDArray* b) {
    NDArray *a_temp = NULL, *b_temp = NULL;
    if (NDArray_DEVICE(a) != NDArray_DEVICE(b)) {
        zend_throw_error(NULL, "Device mismatch, both NDArray MUST be in the same device.");
        return NULL;
    }

    if (NDArray_NDIM(a) == 0) {
        int *shape = ecalloc(1, sizeof(int));
        NDArray *rtn = NDArray_Zeros(shape, 0, NDARRAY_TYPE_FLOAT32, NDArray_DEVICE(a));
        NDArray_FDATA(rtn)[0] = NDArray_FDATA(a)[0] + NDArray_FDATA(b)[0];
        return rtn;
    }

    // If a or b are scalars, reshape
    if (NDArray_NDIM(a) == 0 && NDArray_NDIM(b) > 0) {
        a_temp = a;
        int *n_shape = emalloc(sizeof(int) * NDArray_NDIM(b));
        copy(NDArray_SHAPE(b), n_shape, NDArray_NDIM(b));
        a = NDArray_Zeros(n_shape, NDArray_NDIM(b), NDArray_TYPE(b), NDArray_DEVICE(b));
        a = NDArray_Fill(a, NDArray_FDATA(a_temp)[0]);
    } else if (NDArray_NDIM(b) == 0 && NDArray_NDIM(a) > 0) {
        b_temp = b;
        int *n_shape = emalloc(sizeof(int) * NDArray_NDIM(a));
        copy(NDArray_SHAPE(a), n_shape, NDArray_NDIM(a));
        b = NDArray_Zeros(n_shape, NDArray_NDIM(a), NDArray_TYPE(a), NDArray_DEVICE(a));
        b = NDArray_Fill(b, NDArray_FDATA(b_temp)[0]);
    }

    NDArray *broadcasted = NULL;
    NDArray *a_broad = NULL, *b_broad = NULL;

    if (NDArray_NUMELEMENTS(a) < NDArray_NUMELEMENTS(b)) {
        broadcasted = NDArray_Broadcast(a, b);
        a_broad = broadcasted;
        b_broad = b;
    } else if (NDArray_NUMELEMENTS(b) < NDArray_NUMELEMENTS(a)) {
        broadcasted = NDArray_Broadcast(b, a);
        b_broad = broadcasted;
        a_broad = a;
    } else {
        b_broad = b;
        a_broad = a;
    }

    if (b_broad == NULL || a_broad == NULL) {
        zend_throw_error(NULL, "Can't broadcast arrays.");
        return NULL;
    }

    // Check if the element size of the input arrays match
    if (a->descriptor->elsize != sizeof(float) || b->descriptor->elsize != sizeof(float)) {
        // Element size mismatch, return an error or handle it accordingly
        return NULL;
    }

    // Create a new NDArray to store the result
    NDArray *result = (NDArray *) emalloc(sizeof(NDArray));
    result->strides = (int *) emalloc(a_broad->ndim * sizeof(int));
    result->dimensions = (int *) emalloc(a_broad->ndim * sizeof(int));
    result->ndim = a_broad->ndim;
    if (NDArray_DEVICE(a_broad) == NDARRAY_DEVICE_GPU) {
#if HAVE_CUBLAS
        NDArray_VMALLOC((void **) &result->data, NDArray_NUMELEMENTS(a) * sizeof(float));
        cudaDeviceSynchronize();
        result->device = NDARRAY_DEVICE_GPU;
#endif
    } else {
        result->data = (char *) emalloc(a->descriptor->numElements * sizeof(float));
    }
    result->base = NULL;
    result->flags = 0;  // Set appropriate flags
    result->descriptor = (NDArrayDescriptor *) emalloc(sizeof(NDArrayDescriptor));
    result->descriptor->type = NDARRAY_TYPE_FLOAT32;
    result->descriptor->elsize = sizeof(float);
    result->descriptor->numElements = a_broad->descriptor->numElements;
    result->refcount = 1;
    result->device = NDArray_DEVICE(a_broad);

    // Perform element-wise
    result->strides = memcpy(result->strides, a_broad->strides, a_broad->ndim * sizeof(int));
    result->dimensions = memcpy(result->dimensions, a_broad->dimensions, a_broad->ndim * sizeof(int));
    float *resultData = (float *) result->data;
    float *aData = (float *) a_broad->data;
    float *bData = (float *) b_broad->data;
    int numElements = a->descriptor->numElements;
    NDArrayIterator_INIT(result);
    if (NDArray_DEVICE(a_broad) == NDARRAY_DEVICE_GPU && NDArray_DEVICE(b_broad) == NDARRAY_DEVICE_GPU) {
#if HAVE_CUBLAS
        cuda_pow_float(NDArray_NUMELEMENTS(a_broad), NDArray_FDATA(a_broad), NDArray_FDATA(b_broad), NDArray_FDATA(result),
                       NDArray_NUMELEMENTS(a_broad));
#endif
    } else {
        for (int i = 0; i < numElements; i++) {
            resultData[i] = powf(aData[i], bData[i]);
        }
    }
    if (a_temp != NULL) {
        NDArray_FREE(a);
    }
    if (b_temp != NULL) {
        NDArray_FREE(b);
    }
    if (broadcasted != NULL) {
        NDArray_FREE(broadcasted);
    }
    return result;
}

/**
 * NDArray::abs
 *
 * @param nda
 * @return
 */
NDArray*
NDArray_Abs(NDArray *nda) {
    NDArray *rtn = NULL;
    if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_CPU) {
        rtn = NDArray_Map(nda, float_abs);
    } else {
#ifdef HAVE_CUBLAS
        rtn = NDArrayMathGPU_ElementWise(nda, cuda_float_abs);
#else
        zend_throw_error(NULL, "GPU operations unavailable. CUBLAS not detected.");
#endif
    }
    return rtn;
}
