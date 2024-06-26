#ifndef PHPSCI_NDARRAY_MANIPULATION_H
#define PHPSCI_NDARRAY_MANIPULATION_H

#include "ndarray.h"


NDArray* NDArray_Transpose(NDArray *a);
NDArray* NDArray_Reshape(NDArray *target, int *new_shape, int ndim);
NDArray* NDArray_Flatten(NDArray *target);
void reverse_copy(const int* src, int* dest, int size);
void copy(const int* src, int* dest, unsigned int size);
NDArray* NDArray_Slice(NDArray* array, NDArray** indexes, int num_indices);
NDArray* NDArray_Append(NDArray **arrays, int axis, int num_arrays);
NDArray* NDArray_ExpandDim(NDArray *a, NDArray *axis);
NDArray* NDArray_ToContiguous(NDArray *a);
NDArray* NDArray_CheckAxis(NDArray *arr, int *axis, int _flags);
NDArray* NDArray_AtLeast1D(NDArray *a);
NDArray* NDArray_AtLeast2D(NDArray *a);
NDArray* NDArray_AtLeast3D(NDArray *a);
NDArray* NDArray_ConcatenateFlat(NDArray **arrays, int num_arrays);
NDArray* NDArray_Flip(NDArray *a, NDArray *axis);
NDArray* NDArray_Squeeze(NDArray *a, NDArray *axis);
int NDArray_ConvertMultiAxis(NDArray *axis_in, int ndim, bool *out_axis_flags);
#endif //PHPSCI_NDARRAY_MANIPULATION_H
