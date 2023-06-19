#ifndef PHPSCI_NDARRAY_DEBUG_H
#define PHPSCI_NDARRAY_DEBUG_H

#include "ndarray.h"

#ifdef __cplusplus
extern "C" {
#endif
void NDArray_Dump(NDArray* array);
char* print_matrix(double* buffer, int ndims, int* shape, int* strides, int num_elements, int device);
char* print_matrix_float(float* buffer, int ndims, int* shape, int* strides, int num_elements, int device);
void NDArrayIterator_DUMP(NDArray *a);
void NDArray_DumpDevices();
#ifdef __cplusplus
}
#endif

#endif //PHPSCI_NDARRAY_DEBUG_H
