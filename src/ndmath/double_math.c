#include "double_math.h"
#include <math.h>
#include "../../config.h"

/**
 * @param val
 * @return
 */
float
float_abs(float val) {
    return fabsf(val);
}

/**
 * @param val
 * @return
 */
float
float_sqrt(float val) {
    return sqrtf(val);
}

/**
 * @param val
 * @return
 */
float
float_exp(float val) {
    return expf(val);
}

/**
 * @param val
 * @return
 */
float
float_exp2(float val) {
    return exp2f(val);
}

/**
 * @param val
 * @return
 */
float
float_expm1(float val) {
    return expm1f(val);
}

/**
 * @param val
 * @return
 */
float
float_log(float val) {
    return logf(val);
}

/**
 * @param val
 * @return
 */
float
float_log10(float val) {
    return log10f(val);
}

/**
 * @param val
 * @return
 */
float
float_log1p(float val) {
    return log1pf(val);
}

/**
 * @param val
 * @return
 */
float
float_logb(float val) {
    return logbf(val);
}

/**
 * @param val
 * @return
 */
float
float_log2(float val) {
    return log2f(val);
}

/**
 * @param val
 * @return
 */
float float_sin(float val) {
    return sinf(val);
}

/**
 * @param val
 * @return
 */
float float_cos(float val) {
    return cosf(val);
}

/**
 * @param val
 * @return
 */
float float_tan(float val) {
    return tanf(val);
}

/**
 * @param val
 * @return
 */
float float_arcsin(float val) {
    return asinf(val);
}

float float_arccos(float val) {
    if (val < -1.0 || val > 1.0) {
        printf("RuntimeError: Invalid argument provided for arccos");
        exit(1);
    }
    return acosf(val);
}

float float_arctan(float val) {
    return atanf(val);
}

float float_degrees(float val) {
    return (float)(val * (180.0 / 3.1415926535));
}

float float_radians(float val) {
    return (float)(val * (3.1415926535 / 180.0));
}

float float_sinh(float val) {
    return sinhf(val);
}

float float_cosh(float val) {
    return coshf(val);
}

float float_tanh(float val) {
    return tanhf(val);
}

float float_arcsinh(float val) {
    return asinhf(val);
}

float float_arccosh(float val) {
    if (val < 1.0) {
        printf("RuntimeError: Invalid argument provided for arccosh");
        exit(1);
    }
    return acoshf(val);
}

float float_arctanh(float val) {
    if (fabsf(val) == 1.0f) {
        printf("RuntimeError: Invalid argument provided for arctanh. Division by zero detected");
        exit(1);
    }
    if (val < -1.0f || val > 1.0f) {
        printf("RuntimeError: Invalid argument provided for arctanh");
        exit(1);
    }
    return atanhf(val);
}

float float_rint(float val) {
    float rounded = rintf(val);
    int floorInt = (int)floorf(val);

    // Check if the rounded value is halfway between two integers
    if (rounded - floorInt == 0.5f && ((int)rounded % 2 != 0)) {
        rounded -= 1.0f;
    }

    return (int)rounded;
}

float float_fix(float val) {
    return truncf(val);
}

float float_floor(float val) {
    return floorf(val);
}

float float_ceil(float val) {
    return ceilf(val);
}

float float_trunc(float val) {
    return truncf(val);
}

float float_sinc(float val) {
    if (val == 0.0) {
        return 1.0;
    } else {
        return sinf(M_PI * val) / (M_PI * val);
    }
}

float float_negate(float val) {
    return -val;
}

float float_sign(float val) {
    return (val > 0) - (val < 0);
}