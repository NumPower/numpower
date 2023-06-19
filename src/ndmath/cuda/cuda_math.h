#ifndef PHPSCI_NDARRAY_CUDAMATH_H
#define PHPSCI_NDARRAY_CUDAMATH_H


#include "../../ndarray.h"

#ifdef __cplusplus
extern "C" {
#endif

    typedef void (*ElementWiseFloatGPUOperation)(int, float *);
    NDArray* NDArrayMathGPU_ElementWise(NDArray *ndarray, ElementWiseFloatGPUOperation op);
    void cuda_float_abs(int nblocks, float *d_array);
    void cuda_float_expm1(int nblocks, float *d_array);
    void cuda_float_exp(int nblocks, float *d_array);
    void cuda_float_sqrt(int nblocks, float *d_array);
    void cuda_float_log(int nblocks, float *d_array);
    void cuda_float_logb(int nblocks, float *d_array);
    void cuda_float_log2(int nblocks, float *d_array);
    void cuda_float_log1p(int nblocks, float *d_array);
    void cuda_float_log10(int nblocks, float *d_array);
    void cuda_add_float(int nblocks, float *a, float *b, float *rtn, int nelements);
    void cuda_subtract_float(int nblocks, float *a, float *b, float *rtn, int nelements);
    void cuda_divide_float(int nblocks, float *a, float *b, float *rtn, int nelements);
    void cuda_multiply_float(int nblocks, float *a, float *b, float *rtn, int nelements);
    void cuda_mod_float(int nblocks, float *a, float *b, float *rtn, int nelements);
    int cuda_svd_float(float *d_A, float *d_U, float *d_V, float *d_S, int m, int n);
    float cuda_max_float(float *a, int nelements);
    float cuda_min_float(float *a, int nelements);
    void cuda_pow_float(int nblocks, float *a, float *b, float *rtn, int nelements);
    int cuda_equal_float(int nblocks, float *a, float *b, int nelements);
    void cuda_sum_float(int nblocks, float *a, float *rtn, int nelements);
    void cuda_matmul_float(int nblocks, float *a, float *b, float *rtn, int widthA, int heightA, int widthB);
    void cuda_fill_float(float *a, float value, int n);
    int cuda_det_float(float *a, float *result, int n);
#ifdef __cplusplus
}
#endif
#endif //PHPSCI_NDARRAY_CUDAMATH_H