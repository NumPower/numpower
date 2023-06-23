#ifndef PHPSCI_NDARRAY_INDEXING_H
#define PHPSCI_NDARRAY_INDEXING_H

#include "ndarray.h"

void* slice_float(float* buffer, int ndims, int* shape, int* strides, int* start, int* stop, int* step, float* out_buffer, int* out_shape, int* out_strides, int* out_ndims);
NDArray* NDArray_Diagonal(NDArray *target, int offset);
#endif //PHPSCI_NDARRAY_INDEXING_H
