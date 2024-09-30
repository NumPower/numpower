#ifndef NUMPOWER_CALCULATION_H
#define NUMPOWER_CALCULATION_H

#include "../ndarray.h"

typedef int (NDArray_ArgFunc)(float*, int, float *);

#define _LESS_THAN_OR_EQUAL(a,b) ((a) <= (b))

NDArray * NDArray_ArgMinMaxCommon(NDArray *op, int axis, bool keepdims, bool is_argmax);

#endif //NUMPOWER_CALCULATION_H
