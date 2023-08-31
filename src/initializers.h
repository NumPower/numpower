#ifndef PHPSCI_NDARRAY_INITIALIZERS_H
#define PHPSCI_NDARRAY_INITIALIZERS_H

#include <Zend/zend_types.h>
#include "ndarray.h"

NDArray* Create_NDArray(int* shape, int ndim, const char* type, int device);
NDArray* Create_NDArray_FromZval(zval* php_object);
NDArray* NDArray_FromNDArray(NDArray *target, int buffer_offset, int* shape, int* strides, const int* ndim);
NDArray* NDArray_Zeros(int *shape, int ndim, const char *type, int device);
NDArray* NDArray_Ones(int *shape, int ndim, const char *type);
NDArray* NDArray_Identity(int size);
NDArray* NDArray_Normal(double loc, double scale, int* shape, int ndim);
NDArray* NDArray_StandardNormal(int* shape, int ndim);
NDArray* NDArray_Poisson(double lam, int* shape, int ndim);
NDArray* NDArray_Uniform(double low, double high, int* shape, int ndim);
NDArray* NDArray_Diag(NDArray *a);
NDArray* NDArray_Fill(NDArray *a, float fill_value);
NDArray* NDArray_Full(int *shape, int ndim,  double fill_value);
NDArray* NDArray_CreateFromDoubleScalar(double scalar);
NDArray* NDArray_CreateFromLongScalar(long scalar);
int* Generate_Strides(int* dimensions, int dimensions_size, int elsize);
NDArray* NDArray_CreateFromFloatScalar(float scalar);
NDArray* NDArray_Empty(int *shape, int ndim, const char *type, int device);
NDArray* NDArray_Arange(double start, double stop, double step);
NDArray* NDArray_Binominal(int *shape, int ndim, int n, float p);
#ifdef __cplusplus
extern "C" {
#endif
NDArray *NDArray_Copy(NDArray *a, int device);
#ifdef __cplusplus
}
#endif

#endif //PHPSCI_NDARRAY_INITIALIZERS_H
