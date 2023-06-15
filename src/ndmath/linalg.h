#ifndef PHPSCI_NDARRAY_LINALG_H
#define PHPSCI_NDARRAY_LINALG_H

#include "../ndarray.h"

NDArray* NDArray_Matmul(NDArray *a, NDArray *b);
NDArray** NDArray_SVD(NDArray *target);

#endif //PHPSCI_NDARRAY_LINALG_H
