#ifndef PHPSCI_NDARRAY_INDEXING_H
#define PHPSCI_NDARRAY_INDEXING_H

#include "ndarray.h"

typedef struct {
    int *start;
    int *stop;
    int *step;
} SliceObject;

NDArray* NDArray_Diagonal(NDArray *target, int offset);
int Slice_GetIndices(SliceObject *r, int length, int *start, int *stop, int *step, int *slicelength);
#endif //PHPSCI_NDARRAY_INDEXING_H
