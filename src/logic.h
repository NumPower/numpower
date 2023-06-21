#ifndef PHPSCI_NDARRAY_LOGIC_H
#define PHPSCI_NDARRAY_LOGIC_H

#include "ndarray.h"

float NDArray_All(NDArray *a);
int NDArray_ArrayEqual(NDArray *a, NDArray *b);
NDArray* NDArray_Equal(NDArray* nda, NDArray* ndb);

#endif //PHPSCI_NDARRAY_LOGIC_H
