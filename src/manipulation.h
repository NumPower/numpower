#ifndef PHPSCI_NDARRAY_MANIPULATION_H
#define PHPSCI_NDARRAY_MANIPULATION_H

#include "ndarray.h"

NDArray* NDArray_Transpose(NDArray *a, NDArray_Dims *permute);
NDArray* NDArray_Reshape(NDArray *target, int *new_shape, int ndim);

#endif //PHPSCI_NDARRAY_MANIPULATION_H
