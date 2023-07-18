#ifndef PHPSCI_NDARRAY_LOGIC_H
#define PHPSCI_NDARRAY_LOGIC_H

#include "ndarray.h"

float NDArray_All(NDArray *a);
int NDArray_ArrayEqual(NDArray *a, NDArray *b);
NDArray* NDArray_Equal(NDArray* nda, NDArray* ndb);
int NDArray_AllClose(NDArray* a, NDArray *b, float rtol, float atol);
NDArray* NDArray_Greater(NDArray* nda, NDArray* ndb);
NDArray* NDArray_GreaterEqual(NDArray* nda, NDArray* ndb);
NDArray* NDArray_LessEqual(NDArray* nda, NDArray* ndb);
NDArray* NDArray_Less(NDArray* nda, NDArray* ndb);
NDArray* NDArray_NotEqual(NDArray* nda, NDArray* ndb);
#endif //PHPSCI_NDARRAY_LOGIC_H
