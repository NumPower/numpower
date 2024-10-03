#ifndef NUMPOWER_STATISTICS_H
#define NUMPOWER_STATISTICS_H

#include "../ndarray.h"

NDArray* NDArray_Quantile(NDArray *target, NDArray *q);
NDArray* NDArray_Std(NDArray *a);
NDArray* NDArray_Variance(NDArray *a);
NDArray* NDArray_Average(NDArray *a, NDArray *weights);
NDArray* NDArray_cov(NDArray *a, bool rowvar);

#endif //NUMPOWER_STATISTICS_H
