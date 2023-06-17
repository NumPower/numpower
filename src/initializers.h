#ifndef PHPSCI_NDARRAY_INITIALIZERS_H
#define PHPSCI_NDARRAY_INITIALIZERS_H

#include <Zend/zend_types.h>
#include "ndarray.h"

NDArray* Create_NDArray(int* shape, int shape_size, const char* type);
NDArray* Create_NDArray_FromZval(zval* php_object);
NDArray* NDArray_FromNDArray(NDArray *target, int buffer_offset, int* shape, int* strides, const int* ndim);
NDArray* NDArray_Zeros(int *shape, int ndim, const char *type);
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

#endif //PHPSCI_NDARRAY_INITIALIZERS_H
