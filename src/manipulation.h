#ifndef PHPSCI_NDARRAY_MANIPULATION_H
#define PHPSCI_NDARRAY_MANIPULATION_H

#include "ndarray.h"

NDArray* NDArray_Transpose(NDArray *a, NDArray_Dims *permute);
NDArray* NDArray_Reshape(NDArray *target, int *new_shape, int ndim);
NDArray* NDArray_Flatten(NDArray *target);
void reverse_copy(const int* src, int* dest, int size);
void copy(const int* src, int* dest, unsigned int size);
NDArray* NDArray_Slice(NDArray* array, NDArray** indexes, int num_indices, int return_view);
void *linearize_FLOAT_matrix(float *dst_in, float *src_in, NDArray * a);
NDArray* NDArray_Append(NDArray *a, NDArray *b);
NDArray* NDArray_ExpandDim(NDArray *a, int axis);
NDArray* NDArray_ToContiguous(NDArray *a);
NDArray* NDArray_CheckAxis(NDArray *arr, int *axis, int flags);
#endif //PHPSCI_NDARRAY_MANIPULATION_H
