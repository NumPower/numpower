#ifndef PHPSCI_NDARRAY_ARITHMETICS_H
#define PHPSCI_NDARRAY_ARITHMETICS_H

#include "../ndarray.h"

NDArray* NDArray_Subtract_Double(NDArray* a, NDArray* b);
NDArray* NDArray_Add_Double(NDArray* a, NDArray* b);
NDArray* NDArray_Multiply_Double(NDArray* a, NDArray* b);
NDArray* NDArray_Divide_Double(NDArray* a, NDArray* b);
NDArray* NDArray_Pow_Double(NDArray* a, NDArray* b);
NDArray* NDArray_Mod_Double(NDArray* a, NDArray* b);
double NDArray_Sum_Double(NDArray* a);
NDArray* NDArray_Double_Prod(NDArray* a);
#endif //PHPSCI_NDARRAY_ARITHMETICS_H
