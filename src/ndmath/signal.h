#ifndef NUMPOWER_SIGNAL_H
#define NUMPOWER_SIGNAL_H
/**
 * Copyright (c) 2001-2002 Enthought, Inc. 2003-2023, SciPy Developers.
 * All rights reserved.
 */

#include "../ndarray.h"

#define BOUNDARY_MASK 12
#define OUTSIZE_MASK 3
#define FLIP_MASK  16
#define TYPE_MASK  (32+64+128+256+512)
#define TYPE_SHIFT 5

#define FULL  2
#define SAME  1
#define VALID 0

#define CIRCULAR 8
#define REFLECT  4
#define PAD      0

#define MAXTYPES 21


NDArray * NDArray_Correlate2D(NDArray *a, NDArray *b, int mode, int boundary, NDArray* fill_value, int flip);

#endif //NUMPOWER_SIGNAL_H
