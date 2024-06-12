#ifndef PHPSCI_NDARRAY_DOUBLE_MATH_H
#define PHPSCI_NDARRAY_DOUBLE_MATH_H

#include "../../config.h"
#include "../ndarray.h"

float float_abs(float val);
float float_sqrt(float val);
float float_exp(float val);
float float_exp2(float val);
float float_expm1(float val);
float float_log(float val);
float float_log2(float val);
float float_log10(float val);
float float_log1p(float val);
float float_logb(float val);
float float_sin(float val);
float float_cos(float val);
float float_tan(float val);
float float_arcsin(float val);
float float_arccos(float val);
float float_arctan(float val);
float float_degrees(float val);
float float_radians(float val);
float float_sinh(float val);
float float_cosh(float val);
float float_tanh(float val);
float float_arcsinh(float val);
float float_arccosh(float val);
float float_arctanh(float val);
float float_rint(float val);
float float_fix(float val);
float float_floor(float val);
float float_ceil(float val);
float float_trunc(float val);
float float_sinc(float val);
float float_negate(float val);
float float_sign(float val);
float float_clip(float val, float min, float max);
float float_round(float number, float decimals);
float float_rsqrt(float val);
float float_arctan2(float x, float y);
#endif //PHPSCI_NDARRAY_DOUBLE_MATH_H
