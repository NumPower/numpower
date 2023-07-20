#ifndef PHPSCI_NDARRAY_CUDAMATH_H
#define PHPSCI_NDARRAY_CUDAMATH_H


#include "../../ndarray.h"

#ifdef __cplusplus
extern "C" {
#endif

    typedef void (*ElementWiseFloatGPUOperation)(int, float *);
    typedef void (*ElementWiseFloatGPUOperation2F)(int, float *, float, float);
    typedef void (*ElementWiseFloatGPUOperation1F)(int, float *, float);
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
    void cuda_float_sin(int nblocks, float *d_array);
    void cuda_float_cos(int nblocks, float *d_array);
    void cuda_float_tan(int nblocks, float *d_array);
    void cuda_float_arcsin(int nblocks, float *d_array);
    void cuda_float_arccos(int nblocks, float *d_array);
    void cuda_float_arctan(int nblocks, float *d_array);
    void cuda_float_degrees(int nblocks, float *d_array);
    void cuda_float_radians(int nblocks, float *d_array);
    void cuda_float_sinh(int nblocks, float *d_array);
    void cuda_float_cosh(int nblocks, float *d_array);
    void cuda_float_tanh(int nblocks, float *d_array);
    void cuda_float_arcsinh(int nblocks, float *d_array);
    void cuda_float_arccosh(int nblocks, float *d_array);
    void cuda_float_arctanh(int nblocks, float *d_array);
    void cuda_float_rint(int nblocks, float *d_array);
    void cuda_float_fix(int nblocks, float *d_array);
    void cuda_float_ceil(int nblocks, float *d_array);
    void cuda_float_floor(int nblocks, float *d_array);
    void cuda_float_sinc(int nblocks, float *d_array);
    void cuda_float_trunc(int nblocks, float *d_array);
    void cuda_float_negate(int nblocks, float *d_array);
    void cuda_float_sign(int nblocks, float *d_array);
    void cuda_float_clip(int nblocks, float *d_array, float minVal, float maxVal);
    void cuda_float_transpose(float *target, float *rtn, int rows, int cols);
    void cuda_float_multiply_matrix_vector(int nblocks, float *a_array, float *b_array, float *result, int rows, int cols);
    void cuda_float_compare_equal(int nblocks, float *a_array, float *b_array, float *result, int n);
    void cuda_matrix_float_l1norm(float *target, float *rtn, int rows, int cols);
    int cuda_matrix_float_l2norm(float *target, float *rtn, int rows, int cols);
    void cuda_matrix_float_inverse(float* matrix, int n);
    void cuda_float_lu(float *matrix, float *L, float *U, float *P, int size);
    void cuda_prod_float(int nblocks, float *a, float *rtn, int nelements);
    void cuda_float_round(int nblocks, float *d_array, float decimals);
    void cuda_calculate_outer_product(int m, int n, float *a_array, float *b_array, float *r_array);
    void cuda_float_compare_greater(int nblocks, float *a_array, float *b_array, float *result, int n);
    void cuda_float_compare_greater_equal(int nblocks, float *a_array, float *b_array, float *result, int n);
    void cuda_float_compare_less(int nblocks, float *a_array, float *b_array, float *result, int n);
    void cuda_float_compare_less_equal(int nblocks, float *a_array, float *b_array, float *result, int n);
    void cuda_float_compare_not_equal(int nblocks, float *a_array, float *b_array, float *result, int n);
    void cuda_matrix_eig_float(float* d_matrix, int n, float* rightEigenvectors);
    void cuda_lstsq_float(float* A, int m, int n, float* B, int k, float* X);
    float cuda_float_median_float(int nblocks, float *a_array, int n);
    NDArray* NDArrayMathGPU_ElementWise2F(NDArray* ndarray, ElementWiseFloatGPUOperation2F op, float val1, float val2);
    NDArray* NDArrayMathGPU_ElementWise1F(NDArray* ndarray, ElementWiseFloatGPUOperation1F op, float val1);
    void cuda_convolve2d_same_float(const float* a, const float* b,
                                   const int* shape_a, const int* shape_b,
                                   const int* strides_a, const int* strides_b,
                                   char boundary, float* output,
                                   float fill_value);

#ifdef __cplusplus
}
#endif
#endif //PHPSCI_NDARRAY_CUDAMATH_H