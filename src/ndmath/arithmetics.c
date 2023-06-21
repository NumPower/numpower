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
        //cublasHandle_t handle;
        //cublasCreate(&handle);
        //cublasSasum(handle, NDArray_NUMELEMENTS(a), NDArray_FDATA(a), 1, &value);
        cuda_sum_float(NDArray_NUMELEMENTS(a), NDArray_FDATA(a), &value, NDArray_NUMELEMENTS(a));
#endif
    } else {

#ifdef HAVE_CBLAS
        value = cblas_sasum(NDArray_NUMELEMENTS(a), NDArray_FDATA(a), 1);
#else
        for (int i = 0; i < NDArray_NUMELEMENTS(a); i++) {
            value += NDArray_FDATA(a)[i];
        }
#endif
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

NDArray*
NDArray_Add_Float(NDArray* a, NDArray* b) {
    if (NDArray_DEVICE(a) != NDArray_DEVICE(b)) {
        zend_throw_error(NULL, "Device mismatch, both NDArray MUST be in the same device.");
        return NULL;
    }
    // Check if the dimensions of the input arrays match
    if (a->ndim != b->ndim) {
        // Dimensions mismatch, return an error or handle it accordingly
        return NULL;
    }

    if (NDArray_NDIM(a) == 0) {
        int* shape = ecalloc(1, sizeof(int));
        NDArray *rtn = NDArray_Zeros(shape, 0, NDARRAY_TYPE_FLOAT32);
        NDArray_FDATA(rtn)[0] = NDArray_FDATA(a)[0] + NDArray_FDATA(b)[0];
        return rtn;
    }

    // Check if the shape of the input arrays match
    for (int i = 0; i < a->ndim; i++) {
        if (a->dimensions[i] != b->dimensions[i]) {
            // Shape mismatch, return an error or handle it accordingly
            zend_throw_error(NULL, "Shape mismatch");
            return NULL;
        }
    }

    // Check if the element size of the input arrays match
    if (a->descriptor->elsize != sizeof(float) || b->descriptor->elsize != sizeof(float)) {
        // Element size mismatch, return an error or handle it accordingly
        return NULL;
    }

    // Create a new NDArray to store the result
    NDArray* result = (NDArray*)emalloc(sizeof(NDArray));
    result->strides = (int*)emalloc(a->ndim * sizeof(int));
    result->dimensions = (int*)emalloc(a->ndim * sizeof(int));
    result->ndim = a->ndim;
    if (NDArray_DEVICE(a) == NDARRAY_DEVICE_GPU) {
#if HAVE_CUBLAS
        NDArray_VMALLOC((void **) &result->data, NDArray_NUMELEMENTS(a) * sizeof(float));
        cudaDeviceSynchronize();
#endif
    } else {
        result->data = (char *) emalloc(a->descriptor->numElements * sizeof(float));
    }
    result->base = NULL;
    result->flags = 0;  // Set appropriate flags
    result->descriptor = (NDArrayDescriptor*)emalloc(sizeof(NDArrayDescriptor));
    result->descriptor->type = NDARRAY_TYPE_FLOAT32;
    result->descriptor->elsize = sizeof(float);
    result->descriptor->numElements = a->descriptor->numElements;
    result->refcount = 1;
    result->device = NDArray_DEVICE(a);

    // Perform element-wise addition
    result->strides = memcpy(result->strides, a->strides, a->ndim * sizeof(int));
    result->dimensions = memcpy(result->dimensions, a->dimensions, a->ndim * sizeof(int));
    float* resultData = (float*)result->data;
    float* aData = (float*)a->data;
    float* bData = (float*)b->data;
    int numElements = a->descriptor->numElements;
    NDArrayIterator_INIT(result);
    if (NDArray_DEVICE(a) == NDARRAY_DEVICE_GPU && NDArray_DEVICE(b) == NDARRAY_DEVICE_GPU) {
#if HAVE_CUBLAS
        cuda_add_float(NDArray_NUMELEMENTS(a), NDArray_FDATA(a), NDArray_FDATA(b), NDArray_FDATA(result), NDArray_NUMELEMENTS(a));
        result->device = NDARRAY_DEVICE_GPU;
#endif
    } else {
#ifdef HAVE_AVX2
        int i;
        __m256 vec1, vec2, mul;

        for (i = 0; i < NDArray_NUMELEMENTS(a); i += 8) {
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

        if (NDArray_NUMELEMENTS(a) == NDArray_NUMELEMENTS(b)) {
            memcpy(resultData, NDArray_FDATA(b), NDArray_ELSIZE(b) * NDArray_NUMELEMENTS(b));
            cblas_saxpy(NDArray_NUMELEMENTS(a), 1.0F, NDArray_FDATA(a), 1, resultData,
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
    if (NDArray_DEVICE(a) != NDArray_DEVICE(b)) {
        zend_throw_error(NULL, "Device mismatch, both NDArray MUST be in the same device.");
        return NULL;
    }
    // Check if the dimensions of the input arrays match
    if (a->ndim != b->ndim) {
        // Dimensions mismatch, return an error or handle it accordingly
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
            NDArray *rtn = NDArray_Empty(shape, 0, NDARRAY_TYPE_FLOAT32, NDARRAY_DEVICE_GPU);
            NDArray_FDATA(rtn)[0] = NDArray_FDATA(a)[0] * NDArray_FDATA(b)[0];
            return rtn;
        }
    }

    // Check if the shape of the input arrays match
    for (int i = 0; i < a->ndim; i++) {
        if (a->dimensions[i] != b->dimensions[i]) {
            // Shape mismatch, return an error or handle it accordingly
            return NULL;
        }
    }

    // Check if the element size of the input arrays match
    if (a->descriptor->elsize != sizeof(float) || b->descriptor->elsize != sizeof(float)) {
        // Element size mismatch, return an error or handle it accordingly
        return NULL;
    }

    // Create a new NDArray to store the result
    NDArray *result = (NDArray *) emalloc(sizeof(NDArray));
    result->strides = (int *) emalloc(a->ndim * sizeof(int));
    result->dimensions = (int *) emalloc(a->ndim * sizeof(int));
    result->ndim = a->ndim;
    result->device = NDArray_DEVICE(a);
    if (NDArray_DEVICE(a) == NDARRAY_DEVICE_GPU) {
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
    result->descriptor->numElements = a->descriptor->numElements;
    result->refcount = 1;

    // Perform element-wise product
    result->strides = memcpy(result->strides, a->strides, a->ndim * sizeof(int));
    result->dimensions = memcpy(result->dimensions, a->dimensions, a->ndim * sizeof(int));
    float *resultData = (float *) result->data;
    float *aData = (float *) a->data;
    float *bData = (float *) b->data;
    int numElements = a->descriptor->numElements;
    NDArrayIterator_INIT(result);
    if (NDArray_DEVICE(a) == NDARRAY_DEVICE_GPU && NDArray_DEVICE(b) == NDARRAY_DEVICE_GPU) {
#if HAVE_CUBLAS
        cuda_multiply_float(NDArray_NUMELEMENTS(a), NDArray_FDATA(a), NDArray_FDATA(b), NDArray_FDATA(result),
                       NDArray_NUMELEMENTS(a));
#endif
    } else {
#ifdef HAVE_AVX2
        int i;
        __m256 vec1, vec2, mul;

        for (i = 0; i < NDArray_NUMELEMENTS(a); i += 8) {
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
    if (NDArray_DEVICE(a) != NDArray_DEVICE(b)) {
        zend_throw_error(NULL, "Device mismatch, both NDArray MUST be in the same device.");
        return NULL;
    }
    // Check if the dimensions of the input arrays match
    if (a->ndim != b->ndim) {
        // Dimensions mismatch, return an error or handle it accordingly
        return NULL;
    }

    if (NDArray_NDIM(a) == 0) {
        int *shape = ecalloc(1, sizeof(int));
        NDArray *rtn = NDArray_Zeros(shape, 0, NDARRAY_TYPE_FLOAT32);
        NDArray_FDATA(rtn)[0] = NDArray_FDATA(a)[0] + NDArray_FDATA(b)[0];
        return rtn;
    }

    // Check if the shape of the input arrays match
    for (int i = 0; i < a->ndim; i++) {
        if (a->dimensions[i] != b->dimensions[i]) {
            // Shape mismatch, return an error or handle it accordingly
            return NULL;
        }
    }

    // Check if the element size of the input arrays match
    if (a->descriptor->elsize != sizeof(float) || b->descriptor->elsize != sizeof(float)) {
        // Element size mismatch, return an error or handle it accordingly
        return NULL;
    }

    // Create a new NDArray to store the result
    NDArray *result = (NDArray *) emalloc(sizeof(NDArray));
    result->strides = (int *) emalloc(a->ndim * sizeof(int));
    result->dimensions = (int *) emalloc(a->ndim * sizeof(int));
    result->ndim = a->ndim;
    if (NDArray_DEVICE(a) == NDARRAY_DEVICE_GPU) {
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
    result->descriptor->numElements = a->descriptor->numElements;
    result->refcount = 1;
    result->device = NDArray_DEVICE(a);

    // Perform element-wise subtraction
    result->strides = memcpy(result->strides, a->strides, a->ndim * sizeof(int));
    result->dimensions = memcpy(result->dimensions, a->dimensions, a->ndim * sizeof(int));
    float *resultData = (float *) result->data;
    float *aData = (float *) a->data;
    float *bData = (float *) b->data;
    int numElements = a->descriptor->numElements;
    NDArrayIterator_INIT(result);
    if (NDArray_DEVICE(a) == NDARRAY_DEVICE_GPU && NDArray_DEVICE(b) == NDARRAY_DEVICE_GPU) {
#if HAVE_CUBLAS
        cuda_subtract_float(NDArray_NUMELEMENTS(a), NDArray_FDATA(a), NDArray_FDATA(b), NDArray_FDATA(result),
                            NDArray_NUMELEMENTS(a));
#endif
    } else {
#ifdef HAVE_AVX2
        int i;
        __m256 vec1, vec2, sub;

        for (i = 0; i < NDArray_NUMELEMENTS(a); i += 8) {
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
    if (NDArray_DEVICE(a) != NDArray_DEVICE(b)) {
        zend_throw_error(NULL, "Device mismatch, both NDArray MUST be in the same device.");
        return NULL;
    }
    // Check if the dimensions of the input arrays match
    if (a->ndim != b->ndim) {
        // Dimensions mismatch, return an error or handle it accordingly
        return NULL;
    }

    if (NDArray_NDIM(a) == 0) {
        int *shape = ecalloc(1, sizeof(int));
        NDArray *rtn = NDArray_Zeros(shape, 0, NDARRAY_TYPE_FLOAT32);
        NDArray_FDATA(rtn)[0] = NDArray_FDATA(a)[0] + NDArray_FDATA(b)[0];
        return rtn;
    }

    // Check if the shape of the input arrays match
    for (int i = 0; i < a->ndim; i++) {
        if (a->dimensions[i] != b->dimensions[i]) {
            // Shape mismatch, return an error or handle it accordingly
            return NULL;
        }
    }

    // Check if the element size of the input arrays match
    if (a->descriptor->elsize != sizeof(float) || b->descriptor->elsize != sizeof(float)) {
        // Element size mismatch, return an error or handle it accordingly
        return NULL;
    }

    // Create a new NDArray to store the result
    NDArray *result = (NDArray *) emalloc(sizeof(NDArray));
    result->strides = (int *) emalloc(a->ndim * sizeof(int));
    result->dimensions = (int *) emalloc(a->ndim * sizeof(int));
    result->device = NDArray_DEVICE(a);
    result->ndim = a->ndim;
    if (NDArray_DEVICE(a) == NDARRAY_DEVICE_GPU) {
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
    result->descriptor->numElements = a->descriptor->numElements;
    result->refcount = 1;

    // Perform element-wise subtraction
    result->strides = memcpy(result->strides, a->strides, a->ndim * sizeof(int));
    result->dimensions = memcpy(result->dimensions, a->dimensions, a->ndim * sizeof(int));
    float *resultData = (float *) result->data;
    float *aData = (float *) a->data;
    float *bData = (float *) b->data;
    int numElements = a->descriptor->numElements;
    NDArrayIterator_INIT(result);
    if (NDArray_DEVICE(a) == NDARRAY_DEVICE_GPU && NDArray_DEVICE(b) == NDARRAY_DEVICE_GPU) {
#if HAVE_CUBLAS
        cuda_divide_float(NDArray_NUMELEMENTS(a), NDArray_FDATA(a), NDArray_FDATA(b), NDArray_FDATA(result),
                            NDArray_NUMELEMENTS(a));
#endif
    } else {
#ifdef HAVE_AVX2
        int i;
        __m256 vec1, vec2, sub;

        for (i = 0; i < NDArray_NUMELEMENTS(a); i += 8) {
            vec1 = _mm256_loadu_ps(&aData[i]);
            vec2 = _mm256_loadu_ps(&bData[i]);
            sub = _mm256_div_ps(vec1, vec2);
            _mm256_storeu_ps(&resultData[i], sub);
        }

        // Handle remaining elements if the length is not a multiple of 4
        for (; i < NDArray_NUMELEMENTS(a); i++) {
            resultData[i] = aData[i] - bData[i];
        }
#else
        for (int i = 0; i < numElements; i++) {
            resultData[i] = aData[i] / bData[i];
        }
#endif
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
    if (NDArray_DEVICE(a) != NDArray_DEVICE(b)) {
        zend_throw_error(NULL, "Device mismatch, both NDArray MUST be in the same device.");
        return NULL;
    }
    // Check if the dimensions of the input arrays match
    if (a->ndim != b->ndim) {
        // Dimensions mismatch, return an error or handle it accordingly
        return NULL;
    }

    if (NDArray_NDIM(a) == 0) {
        int* shape = ecalloc(1, sizeof(int));
        NDArray *rtn = NDArray_Zeros(shape, 0, NDARRAY_TYPE_FLOAT32);
        NDArray_FDATA(rtn)[0] = NDArray_FDATA(a)[0] + NDArray_FDATA(b)[0];
        return rtn;
    }

    // Check if the shape of the input arrays match
    for (int i = 0; i < a->ndim; i++) {
        if (a->dimensions[i] != b->dimensions[i]) {
            // Shape mismatch, return an error or handle it accordingly
            return NULL;
        }
    }

    // Check if the element size of the input arrays match
    if (a->descriptor->elsize != sizeof(float) || b->descriptor->elsize != sizeof(float)) {
        // Element size mismatch, return an error or handle it accordingly
        return NULL;
    }

    // Create a new NDArray to store the result
    NDArray* result = (NDArray*)emalloc(sizeof(NDArray));
    result->strides = (int*)emalloc(a->ndim * sizeof(int));
    result->dimensions = (int*)emalloc(a->ndim * sizeof(int));
    result->ndim = a->ndim;
    if (NDArray_DEVICE(a) == NDARRAY_DEVICE_GPU) {
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
    result->descriptor = (NDArrayDescriptor*)emalloc(sizeof(NDArrayDescriptor));
    result->descriptor->type = NDARRAY_TYPE_FLOAT32;
    result->descriptor->elsize = sizeof(float);
    result->descriptor->numElements = a->descriptor->numElements;
    result->refcount = 1;
    result->device = NDArray_DEVICE(a);

    // Perform element-wise subtraction
    result->strides = memcpy(result->strides, a->strides, a->ndim * sizeof(int));
    result->dimensions = memcpy(result->dimensions, a->dimensions, a->ndim * sizeof(int));
    float* resultData = (float*)result->data;
    float* aData = (float*)a->data;
    float* bData = (float*)b->data;
    int numElements = a->descriptor->numElements;
    NDArrayIterator_INIT(result);
    if (NDArray_DEVICE(a) == NDARRAY_DEVICE_GPU && NDArray_DEVICE(b) == NDARRAY_DEVICE_GPU) {
#if HAVE_CUBLAS
        cuda_mod_float(NDArray_NUMELEMENTS(a), NDArray_FDATA(a), NDArray_FDATA(b), NDArray_FDATA(result),
                          NDArray_NUMELEMENTS(a));
#endif
    } else {
        for (int i = 0; i < numElements; i++) {
            resultData[i] = fmodf(aData[i], bData[i]);
        }
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
    if (NDArray_DEVICE(a) != NDArray_DEVICE(b)) {
        zend_throw_error(NULL, "Device mismatch, both NDArray MUST be in the same device.");
        return NULL;
    }
    // Check if the dimensions of the input arrays match
    if (a->ndim != b->ndim) {
        // Dimensions mismatch, return an error or handle it accordingly
        return NULL;
    }

    if (NDArray_NDIM(a) == 0) {
        int *shape = ecalloc(1, sizeof(int));
        NDArray *rtn = NDArray_Zeros(shape, 0, NDARRAY_TYPE_FLOAT32);
        NDArray_FDATA(rtn)[0] = NDArray_FDATA(a)[0] + NDArray_FDATA(b)[0];
        return rtn;
    }

    // Check if the shape of the input arrays match
    for (int i = 0; i < a->ndim; i++) {
        if (a->dimensions[i] != b->dimensions[i]) {
            // Shape mismatch, return an error or handle it accordingly
            return NULL;
        }
    }

    // Check if the element size of the input arrays match
    if (a->descriptor->elsize != sizeof(float) || b->descriptor->elsize != sizeof(float)) {
        // Element size mismatch, return an error or handle it accordingly
        return NULL;
    }

    // Create a new NDArray to store the result
    NDArray *result = (NDArray *) emalloc(sizeof(NDArray));
    result->strides = (int *) emalloc(a->ndim * sizeof(int));
    result->dimensions = (int *) emalloc(a->ndim * sizeof(int));
    result->ndim = a->ndim;
    if (NDArray_DEVICE(a) == NDARRAY_DEVICE_GPU) {
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
    result->descriptor->numElements = a->descriptor->numElements;
    result->refcount = 1;
    result->device = NDArray_DEVICE(a);

    // Perform element-wise subtraction
    result->strides = memcpy(result->strides, a->strides, a->ndim * sizeof(int));
    result->dimensions = memcpy(result->dimensions, a->dimensions, a->ndim * sizeof(int));
    float *resultData = (float *) result->data;
    float *aData = (float *) a->data;
    float *bData = (float *) b->data;
    int numElements = a->descriptor->numElements;
    NDArrayIterator_INIT(result);
    if (NDArray_DEVICE(a) == NDARRAY_DEVICE_GPU && NDArray_DEVICE(b) == NDARRAY_DEVICE_GPU) {
#if HAVE_CUBLAS
        cuda_pow_float(NDArray_NUMELEMENTS(a), NDArray_FDATA(a), NDArray_FDATA(b), NDArray_FDATA(result),
                       NDArray_NUMELEMENTS(a));
#endif
    } else {
        for (int i = 0; i < numElements; i++) {
            resultData[i] = powf(aData[i], bData[i]);
        }
    }
    return result;
}

