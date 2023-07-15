#ifndef PHPSCI_NDARRAY_LINALG_H
#define PHPSCI_NDARRAY_LINALG_H

#include "../ndarray.h"

NDArray* NDArray_Matmul(NDArray *a, NDArray *b);
NDArray** NDArray_SVD(NDArray *target);
NDArray* NDArray_Det(NDArray *a);
NDArray* NDArray_Dot(NDArray *nda, NDArray *ndb);
NDArray* NDArray_Inner(NDArray *nda, NDArray *ndb);
NDArray* NDArray_Norm(NDArray* target, int type);
NDArray* NDArray_L1Norm(NDArray* target);
NDArray* NDArray_L2Norm(NDArray* target);
NDArray* NDArray_Inverse(NDArray* target);
NDArray** NDArray_LU(NDArray* target);
NDArray* NDArray_MatrixRank(NDArray *target, float *tol);
NDArray* NDArray_Outer(NDArray *a, NDArray *b);
#endif //PHPSCI_NDARRAY_LINALG_H
