#ifndef PHPSCI_NDARRAY_LINALG_H
#define PHPSCI_NDARRAY_LINALG_H

#include "../ndarray.h"

NDArray* NDArray_Matmul(NDArray *a, NDArray *b);
NDArray** NDArray_SVD(NDArray *target);
NDArray* NDArray_Det(NDArray *a);
NDArray* NDArray_Dot(NDArray *nda, NDArray *ndb);

#endif //PHPSCI_NDARRAY_LINALG_H
