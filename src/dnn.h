#ifndef NUMPOWER_DNN_H
#define NUMPOWER_DNN_H

#include "ndarray.h"
#include "../config.h"

NDArray* NDArrayDNN_Conv2D_Forward(NDArray *x, NDArray *filters, int *kernel_size, char activation, int use_bias);
NDArray** NDArrayDNN_Conv2D_Backward(NDArray *input, NDArray *y, NDArray *filters, int kernel_size, char activation, int use_bias);

#endif //NUMPOWER_DNN_H