#ifndef PHPSCI_NDARRAY_ARITHMETICS_H
#define PHPSCI_NDARRAY_ARITHMETICS_H

#include "../ndarray.h"

NDArray* NDArray_Subtract_Float(NDArray* a, NDArray* b);
NDArray* NDArray_Add_Float(NDArray* a, NDArray* b);
NDArray* NDArray_Multiply_Float(NDArray* a, NDArray* b);
NDArray* NDArray_Divide_Float(NDArray* a, NDArray* b);
NDArray* NDArray_Pow_Float(NDArray* a, NDArray* b);
NDArray* NDArray_Mod_Float(NDArray* a, NDArray* b);
float NDArray_Sum_Float(NDArray* a);
float NDArray_Float_Prod(NDArray* a);
float NDArray_Mean_Float(NDArray* a);
NDArray* NDArray_Abs(NDArray *nda);
float NDArray_Median_Float(NDArray* a);
#endif //PHPSCI_NDARRAY_ARITHMETICS_H
