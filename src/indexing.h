#ifndef PHPSCI_NDARRAY_INDEXING_H
#define PHPSCI_NDARRAY_INDEXING_H

void* slice_float(float* buffer, int ndims, int* shape, int* strides, int* start, int* stop, int* step, float* out_buffer, int* out_shape, int* out_strides, int* out_ndims);
#endif //PHPSCI_NDARRAY_INDEXING_H
