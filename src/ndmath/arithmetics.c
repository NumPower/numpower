#include <Zend/zend_alloc.h>
#include <string.h>
#include "arithmetics.h"
#include "../ndarray.h"
#include "../config.h"
#include "../initializers.h"
#include "../debug.h"

#ifdef HAVE_CUBLAS
#include <cuda_runtime.h>
#include <cublas_v2.h>
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
double NDArray_Sum_Double(NDArray* a) {
    double value = 0;

#ifndef HAVE_CBLAS
    value = cblas_dasum(NDArray_NUMELEMENTS(a), NDArray_DDATA(a), 1);
#else
    for (int i = 0; i < NDArray_NUMELEMENTS(a); i++) {
        value += NDArray_DDATA(a)[i];
    }
#endif
    return value;
}

/**
 * Add elements of a and b element-wise
 *
 * @param a
 * @param b
 * @return
 */
NDArray*
NDArray_Add_Double(NDArray* a, NDArray* b) {
    // Check if the dimensions of the input arrays match
    if (a->ndim != b->ndim) {
        // Dimensions mismatch, return an error or handle it accordingly
        return NULL;
    }

    if (NDArray_NDIM(a) == 0) {
        int* shape = ecalloc(1, sizeof(int));
        NDArray *rtn = NDArray_Zeros(shape, 0);
        NDArray_DDATA(rtn)[0] = NDArray_DDATA(a)[0] + NDArray_DDATA(b)[0];
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
    if (a->descriptor->elsize != sizeof(double) || b->descriptor->elsize != sizeof(double)) {
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
        cudaMalloc((void **) &result->data, NDArray_NUMELEMENTS(a) * sizeof(double));
        cudaMemcpy(result->data, NDArray_DDATA(b), NDArray_NUMELEMENTS(a) * sizeof(double), cudaMemcpyHostToDevice);
#endif
    } else {
        result->data = (char *) emalloc(a->descriptor->numElements * sizeof(double));
    }
    result->base = NULL;
    result->flags = 0;  // Set appropriate flags
    result->descriptor = (NDArrayDescriptor*)emalloc(sizeof(NDArrayDescriptor));
    result->descriptor->type = "d";
    result->descriptor->elsize = sizeof(double);
    result->descriptor->numElements = a->descriptor->numElements;
    result->refcount = 1;

    // Perform element-wise addition
    result->strides = memcpy(result->strides, a->strides, a->ndim * sizeof(int));
    result->dimensions = memcpy(result->dimensions, a->dimensions, a->ndim * sizeof(int));
    double* resultData = (double*)result->data;
    double* aData = (double*)a->data;
    double* bData = (double*)b->data;
    int numElements = a->descriptor->numElements;
    if (NDArray_DEVICE(a) == NDARRAY_DEVICE_GPU && NDArray_DEVICE(b) == NDARRAY_DEVICE_GPU) {
#if HAVE_CUBLAS
        double alpha = 1.0;
        cublasHandle_t handle;
        cublasCreate(&handle);
        result->device = NDARRAY_DEVICE_GPU;
        cublasDaxpy(handle, NDArray_NUMELEMENTS(a), &alpha, NDArray_DDATA(a), 1, NDArray_DDATA(result), 1);
        cublasDestroy(handle);
#endif
    } else {
#ifdef HAVE_AVX2
        int i;
        __m256d vec1, vec2, mul;

        for (i = 0; i < NDArray_NUMELEMENTS(a); i += 4) {
            vec1 = _mm256_loadu_pd(&aData[i]);
            vec2 = _mm256_loadu_pd(&bData[i]);
            mul = _mm256_add_pd(vec1, vec2);
            _mm256_storeu_pd(&resultData[i], mul);
        }
        // Handle remaining elements if the length is not a multiple of 4
        for (; i < NDArray_NUMELEMENTS(a); i++) {
            resultData[i] = aData[i] + bData[i];
        }
#elif HAVE_CBLAS
        if (NDArray_NUMELEMENTS(a) == NDArray_NUMELEMENTS(b)) {
            memcpy(resultData, NDArray_DDATA(b), NDArray_ELSIZE(b) * NDArray_NUMELEMENTS(b));
            cblas_daxpy(NDArray_NUMELEMENTS(a), 1.0, NDArray_DDATA(a), 1, resultData,
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
 * NDArray Product
 * @param a
 * @return
 */
NDArray*
NDArray_Double_Prod(NDArray* a) {
    double product = 1.0;
#ifdef HAVE_AVX2
    int remainder = NDArray_NUMELEMENTS(a) % 4;
    int newSize = NDArray_NUMELEMENTS(a) - remainder;
    __m256d productVec = _mm256_set1_pd(1.0);

    for (int i = 0; i < newSize; i += 4) {
        __m256d vectorVec = _mm256_loadu_pd(&NDArray_DDATA(a)[i]);
        productVec = _mm256_mul_pd(productVec, vectorVec);
    }


    double* productPtr = (double*)&productVec;

    for (int i = 0; i < 4; i++) {
        product *= productPtr[i];
    }

    for (int i = newSize; i < NDArray_NUMELEMENTS(a); i++) {
        product *= NDArray_DDATA(a)[i];
    }
#else
    for (int i = 0; i < NDArray_NUMELEMENTS(a); i++) {
        product *= NDArray_DDATA(a)[i];
    }
#endif
    return NDArray_CreateFromDoubleScalar(product);
}

/**
 * Multiply elements of a and b element-wise
 *
 * @param a
 * @param b
 * @return
 */
NDArray* NDArray_Multiply_Double(NDArray* a, NDArray* b) {
    // Check if the dimensions of the input arrays match
    if (a->ndim != b->ndim) {
        // Dimensions mismatch, return an error or handle it accordingly
        return NULL;
    }

    if (NDArray_NDIM(a) == 0) {
        int* shape = ecalloc(1, sizeof(int));
        NDArray *rtn = NDArray_Zeros(shape, 0);
        NDArray_DDATA(rtn)[0] = NDArray_DDATA(a)[0] * NDArray_DDATA(b)[0];
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
    if (a->descriptor->elsize != sizeof(double) || b->descriptor->elsize != sizeof(double)) {
        // Element size mismatch, return an error or handle it accordingly
        return NULL;
    }

    // Create a new NDArray to store the result
    NDArray* result = (NDArray*)emalloc(sizeof(NDArray));
    result->strides = (int*)emalloc(a->ndim * sizeof(int));
    result->dimensions = (int*)emalloc(a->ndim * sizeof(int));
    result->ndim = a->ndim;
    result->data = (char*)emalloc(a->descriptor->numElements * sizeof(double));
    result->base = NULL;
    result->flags = 0;  // Set appropriate flags
    result->descriptor = (NDArrayDescriptor*)emalloc(sizeof(NDArrayDescriptor));
    result->descriptor->type = "d";
    result->descriptor->elsize = sizeof(double);
    result->descriptor->numElements = a->descriptor->numElements;
    result->refcount = 1;

    // Perform element-wise product
    result->strides = memcpy(result->strides, a->strides, a->ndim * sizeof(int));
    result->dimensions = memcpy(result->dimensions, a->dimensions, a->ndim * sizeof(int));
    double* resultData = (double*)result->data;
    double* aData = (double*)a->data;
    double* bData = (double*)b->data;
    int numElements = a->descriptor->numElements;

#ifdef HAVE_AVX2
    int i;
    __m256d vec1, vec2, mul;

    for (i = 0; i < NDArray_NUMELEMENTS(a); i += 4) {
        vec1 = _mm256_loadu_pd(&aData[i]);
        vec2 = _mm256_loadu_pd(&bData[i]);
        mul = _mm256_mul_pd(vec1, vec2);
        _mm256_storeu_pd(&resultData[i], mul);
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
NDArray_Subtract_Double(NDArray* a, NDArray* b) {
    // Check if the dimensions of the input arrays match
    if (a->ndim != b->ndim) {
        // Dimensions mismatch, return an error or handle it accordingly
        return NULL;
    }

    if (NDArray_NDIM(a) == 0) {
        int* shape = ecalloc(1, sizeof(int));
        NDArray *rtn = NDArray_Zeros(shape, 0);
        NDArray_DDATA(rtn)[0] = NDArray_DDATA(a)[0] + NDArray_DDATA(b)[0];
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
    if (a->descriptor->elsize != sizeof(double) || b->descriptor->elsize != sizeof(double)) {
        // Element size mismatch, return an error or handle it accordingly
        return NULL;
    }

    // Create a new NDArray to store the result
    NDArray* result = (NDArray*)emalloc(sizeof(NDArray));
    result->strides = (int*)emalloc(a->ndim * sizeof(int));
    result->dimensions = (int*)emalloc(a->ndim * sizeof(int));
    result->ndim = a->ndim;
    result->data = (char*)emalloc(a->descriptor->numElements * sizeof(double));
    result->base = NULL;
    result->flags = 0;  // Set appropriate flags
    result->descriptor = (NDArrayDescriptor*)emalloc(sizeof(NDArrayDescriptor));
    result->descriptor->type = "d";
    result->descriptor->elsize = sizeof(double);
    result->descriptor->numElements = a->descriptor->numElements;
    result->refcount = 1;

    // Perform element-wise subtraction
    result->strides = memcpy(result->strides, a->strides, a->ndim * sizeof(int));
    result->dimensions = memcpy(result->dimensions, a->dimensions, a->ndim * sizeof(int));
    double* resultData = (double*)result->data;
    double* aData = (double*)a->data;
    double* bData = (double*)b->data;
    int numElements = a->descriptor->numElements;
#ifdef HAVE_AVX2
    int i;
    __m256d vec1, vec2, sub;

    for (i = 0; i < NDArray_NUMELEMENTS(a); i += 4) {
        vec1 = _mm256_loadu_pd(&aData[i]);
        vec2 = _mm256_loadu_pd(&bData[i]);
        sub = _mm256_sub_pd(vec1, vec2);
        _mm256_storeu_pd(&resultData[i], sub);
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
NDArray_Divide_Double(NDArray* a, NDArray* b) {
    // Check if the dimensions of the input arrays match
    if (a->ndim != b->ndim) {
        // Dimensions mismatch, return an error or handle it accordingly
        return NULL;
    }

    if (NDArray_NDIM(a) == 0) {
        int* shape = ecalloc(1, sizeof(int));
        NDArray *rtn = NDArray_Zeros(shape, 0);
        NDArray_DDATA(rtn)[0] = NDArray_DDATA(a)[0] + NDArray_DDATA(b)[0];
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
    if (a->descriptor->elsize != sizeof(double) || b->descriptor->elsize != sizeof(double)) {
        // Element size mismatch, return an error or handle it accordingly
        return NULL;
    }

    // Create a new NDArray to store the result
    NDArray* result = (NDArray*)emalloc(sizeof(NDArray));
    result->strides = (int*)emalloc(a->ndim * sizeof(int));
    result->dimensions = (int*)emalloc(a->ndim * sizeof(int));
    result->ndim = a->ndim;
    result->data = (char*)emalloc(a->descriptor->numElements * sizeof(double));
    result->base = NULL;
    result->flags = 0;  // Set appropriate flags
    result->descriptor = (NDArrayDescriptor*)emalloc(sizeof(NDArrayDescriptor));
    result->descriptor->type = "d";
    result->descriptor->elsize = sizeof(double);
    result->descriptor->numElements = a->descriptor->numElements;
    result->refcount = 1;

    // Perform element-wise subtraction
    result->strides = memcpy(result->strides, a->strides, a->ndim * sizeof(int));
    result->dimensions = memcpy(result->dimensions, a->dimensions, a->ndim * sizeof(int));
    double* resultData = (double*)result->data;
    double* aData = (double*)a->data;
    double* bData = (double*)b->data;
    int numElements = a->descriptor->numElements;
#ifdef HAVE_AVX2
    int i;
    __m256d vec1, vec2, sub;

    for (i = 0; i < NDArray_NUMELEMENTS(a); i += 4) {
        vec1 = _mm256_loadu_pd(&aData[i]);
        vec2 = _mm256_loadu_pd(&bData[i]);
        sub = _mm256_div_pd(vec1, vec2);
        _mm256_storeu_pd(&resultData[i], sub);
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
    return result;
}

/**
 * @param a
 * @param b
 * @return
 */
NDArray*
NDArray_Pow_Double(NDArray* a, NDArray* b) {
    // Check if the dimensions of the input arrays match
    if (a->ndim != b->ndim) {
        // Dimensions mismatch, return an error or handle it accordingly
        return NULL;
    }

    if (NDArray_NDIM(a) == 0) {
        int* shape = ecalloc(1, sizeof(int));
        NDArray *rtn = NDArray_Zeros(shape, 0);
        NDArray_DDATA(rtn)[0] = NDArray_DDATA(a)[0] + NDArray_DDATA(b)[0];
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
    if (a->descriptor->elsize != sizeof(double) || b->descriptor->elsize != sizeof(double)) {
        // Element size mismatch, return an error or handle it accordingly
        return NULL;
    }

    // Create a new NDArray to store the result
    NDArray* result = (NDArray*)emalloc(sizeof(NDArray));
    result->strides = (int*)emalloc(a->ndim * sizeof(int));
    result->dimensions = (int*)emalloc(a->ndim * sizeof(int));
    result->ndim = a->ndim;
    result->data = (char*)emalloc(a->descriptor->numElements * sizeof(double));
    result->base = NULL;
    result->flags = 0;  // Set appropriate flags
    result->descriptor = (NDArrayDescriptor*)emalloc(sizeof(NDArrayDescriptor));
    result->descriptor->type = "d";
    result->descriptor->elsize = sizeof(double);
    result->descriptor->numElements = a->descriptor->numElements;
    result->refcount = 1;

    // Perform element-wise subtraction
    result->strides = memcpy(result->strides, a->strides, a->ndim * sizeof(int));
    result->dimensions = memcpy(result->dimensions, a->dimensions, a->ndim * sizeof(int));
    double* resultData = (double*)result->data;
    double* aData = (double*)a->data;
    double* bData = (double*)b->data;
    int numElements = a->descriptor->numElements;

    for (int i = 0; i < numElements; i++) {
        resultData[i] = pow(aData[i], bData[i]);
    }
    return result;
}

/**
 * @param a
 * @param b
 * @return
 */
NDArray*
NDArray_Mod_Double(NDArray* a, NDArray* b) {
    // Check if the dimensions of the input arrays match
    if (a->ndim != b->ndim) {
        // Dimensions mismatch, return an error or handle it accordingly
        return NULL;
    }

    if (NDArray_NDIM(a) == 0) {
        int* shape = ecalloc(1, sizeof(int));
        NDArray *rtn = NDArray_Zeros(shape, 0);
        NDArray_DDATA(rtn)[0] = NDArray_DDATA(a)[0] + NDArray_DDATA(b)[0];
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
    if (a->descriptor->elsize != sizeof(double) || b->descriptor->elsize != sizeof(double)) {
        // Element size mismatch, return an error or handle it accordingly
        return NULL;
    }

    // Create a new NDArray to store the result
    NDArray* result = (NDArray*)emalloc(sizeof(NDArray));
    result->strides = (int*)emalloc(a->ndim * sizeof(int));
    result->dimensions = (int*)emalloc(a->ndim * sizeof(int));
    result->ndim = a->ndim;
    result->data = (char*)emalloc(a->descriptor->numElements * sizeof(double));
    result->base = NULL;
    result->flags = 0;  // Set appropriate flags
    result->descriptor = (NDArrayDescriptor*)emalloc(sizeof(NDArrayDescriptor));
    result->descriptor->type = "d";
    result->descriptor->elsize = sizeof(double);
    result->descriptor->numElements = a->descriptor->numElements;
    result->refcount = 1;

    // Perform element-wise subtraction
    result->strides = memcpy(result->strides, a->strides, a->ndim * sizeof(int));
    result->dimensions = memcpy(result->dimensions, a->dimensions, a->ndim * sizeof(int));
    double* resultData = (double*)result->data;
    double* aData = (double*)a->data;
    double* bData = (double*)b->data;
    int numElements = a->descriptor->numElements;

    for (int i = 0; i < numElements; i++) {
        resultData[i] = fmod(aData[i], bData[i]);
    }
    return result;
}
