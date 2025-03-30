#ifndef PHPSCI_NDARRAY_SANITIZERS_H
#define PHPSCI_NDARRAY_SANITIZERS_H

#include <Zend/zend.h>
#include "ndarray.h"

int* zval_parameter_to_normalized_axis_argument(zval *arg, const char *name, int ndim, int *outsize);

#endif //PHPSCI_NDARRAY_SANITIZERS_H
