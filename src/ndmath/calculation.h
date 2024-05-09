#ifndef NUMPOWER_CALCULATION_H
#define NUMPOWER_CALCULATION_H

#include "../ndarray.h"

typedef int (NDArray_ArgFunc)(float*, int, float *);

NDArray * NDArray_ArgMinMaxCommon(NDArray *op, int axis, int keepdims, bool is_argmax);

#endif //NUMPOWER_CALCULATION_H
