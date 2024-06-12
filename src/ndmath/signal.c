/**
 * This is a file derived from SciPy's _signaltools.py and modified to
 * operate on NDArrays instead of numpy arrays.
 *
 * https://github.com/scipy/scipy/blob/v1.11.2/scipy/signal/_signaltools.py
 *
 * Copyright (c) 2001-2002 Enthought, Inc. 2003-2023, SciPy Developers.
 * All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 *
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above
 *    copyright notice, this list of conditions and the following
 *    disclaimer in the documentation and/or other materials provided
 *    with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 *    contributors may be used to endorse or promote products derived
 *    from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
 * "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
 * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
 * A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT
 * OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL,
 * SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
 * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
 * DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY
 * THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 */

#include <Zend/zend.h>
#include "signal.h"
#include "../initializers.h"

typedef void (OneMultAddFunction) (char *, char *, int64_t, char **, int64_t);

#define MAKE_ONEMULTADD(fname, type) \
static void fname ## _onemultadd(char *sum, char *term1, int64_t str, char **pvals, int64_t n) { \
        type dsum = *(type*)sum; \
        for (int64_t k=0; k < n; k++) { \
          type tmp = *(type*)(term1 + k * str); \
          dsum += tmp * *(type*)pvals[k]; \
        } \
        *(type*)(sum) = dsum; \
}
MAKE_ONEMULTADD(FLOAT, float)

static OneMultAddFunction *OneMultAdd[]={FLOAT_onemultadd,
                                         NULL, NULL, NULL, NULL};

/**
 * @param mode 0 - full, 1 - valid, 2 - same
 * @param pInt
 * @param pInt1
 */
static
int _inputs_swap(int mode, NDArray *a, NDArray *b) {
    if (mode != 1) {
        return 0;
    }

    int i, oka = 0, okb = 0;
    for (i = 0; i < NDArray_NDIM(a); i++) {
        if (NDArray_SHAPE(a)[i] >= NDArray_SHAPE(b)[i]) {
            oka++;
            continue;
        }
        okb++;
    }
    if (!(oka || okb)) {
        return -1;
    }
    return !oka;
}

static int64_t
circular_wrap_index(int64_t j, int64_t m)
{
    // About the negative case: in C, -3 % 5 is -3, so that explains
    // the " + m" after j % m.  But -5 % 5 is 0, and so -5 % 5 + 5 is 5,
    // which we want to wrap around to 0.  That's why the second " % m" is
    // included in the expression.
    return (j >= 0) ? (j % m) : ((j % m + m) % m);
}

static int64_t
reflect_symm_index(int64_t j, int64_t m)
{
    // First map j to k in the interval [0, 2*m-1).
    // Then flip the k values that are greater than or equal to m.
    int64_t k = (j >= 0) ? (j % (2*m)) : (llabs(j + 1) % (2*m));
    return (k >= m) ? (2*m - k - 1) : k;
}


/**
 *
 * @param a
 * @param b
 * @param mode
 * @param boundary
 * @param value
 * @return
 */
int
_convolve2d(
        char  *in,        /* Input data Ns[0] x Ns[1] */
        int   *instr,     /* Input strides */
        char  *out,       /* Output data */
        int   *outstr,    /* Output strides */
        char  *hvals,     /* coefficients in filter */
        int   *hstr,      /* coefficients strides */
        int   *Nwin,     /* Size of kernel Nwin[0] x Nwin[1] */
        int   *Ns,        /* Size of image Ns[0] x Ns[1] */
        int   flag,       /* convolution parameters */
        char  *fillvalue
) {
    const int boundary = flag & BOUNDARY_MASK;  /* flag can be fill, reflecting, circular */
    const int outsize = flag & OUTSIZE_MASK;
    const int convolve = flag & FLIP_MASK;
    const int type_num = (flag & TYPE_MASK) >> TYPE_SHIFT;
    /*type_size*/
    OneMultAddFunction *mult_and_add = OneMultAdd[type_num];
    if (mult_and_add == NULL) return -5;  /* Not available for this type */

    if (type_num < 0 || type_num > MAXTYPES) return -4;  /* Invalid type */
    const int type_size = sizeof(float);

    int64_t Os[2];
    if (outsize == FULL) {Os[0] = Ns[0]+Nwin[0]-1; Os[1] = Ns[1]+Nwin[1]-1;}
    else if (outsize == SAME) {Os[0] = Ns[0]; Os[1] = Ns[1];}
    else if (outsize == VALID) {Os[0] = Ns[0]-Nwin[0]+1; Os[1] = Ns[1]-Nwin[1]+1;}
    else return -1; /* Invalid output flag */

    if ((boundary != PAD) && (boundary != REFLECT) && (boundary != CIRCULAR))
        return -2; /* Invalid boundary flag */

    char **indices = emalloc(Nwin[1] * sizeof(indices[0]));
    if (indices == NULL) return -3; /* No memory */

    /* Speed this up by not doing any if statements in the for loop.  Need 3*3*2=18 different
       loops executed for different conditions */

    for (int64_t m=0; m < Os[0]; m++) {
        /* Reposition index into input image based on requested output size */
        int64_t new_m;
        if (outsize == FULL) new_m = convolve ? m : (m-Nwin[0]+1);
        else if (outsize == SAME) new_m = convolve ? (m+((Nwin[0]-1)>>1)) : (m-((Nwin[0]-1) >> 1));
        else new_m = convolve ? (m+Nwin[0]-1) : m; /* VALID */

        for (int64_t n=0; n < Os[1]; n++) {  /* loop over columns */
            char * sum = out+m*outstr[0]+n*outstr[1];
            memset(sum, 0, type_size); /* sum = 0.0; */

            int64_t new_n;
            if (outsize == FULL) new_n = convolve ? n : (n-Nwin[1]+1);
            else if (outsize == SAME) new_n = convolve ? (n+((Nwin[1]-1)>>1)) : (n-((Nwin[1]-1) >> 1));
            else new_n = convolve ? (n+Nwin[1]-1) : n;

            /* Sum over kernel, if index into image is out of bounds
           handle it according to boundary flag */
            for (int64_t j=0; j < Nwin[0]; j++) {
                int64_t ind0 = convolve ? (new_m-j): (new_m+j);
                bool bounds_pad_flag = false;

                if ((ind0 < 0) || (ind0 >= Ns[0])) {
                    if (boundary == REFLECT) ind0 = reflect_symm_index(ind0, Ns[0]);
                    else if (boundary == CIRCULAR) ind0 = circular_wrap_index(ind0, Ns[0]);
                    else bounds_pad_flag = true;
                }

                const int64_t ind0_memory = ind0*instr[0];

                if (bounds_pad_flag) {
                    for (int64_t k=0; k < Nwin[1]; k++) {
                        indices[k] = fillvalue;
                    }
                }
                else  {
                    for (int64_t k=0; k < Nwin[1]; k++) {
                        int64_t ind1 = convolve ? (new_n-k) : (new_n+k);
                        if ((ind1 < 0) || (ind1 >= Ns[1])) {
                            if (boundary == REFLECT) ind1 = reflect_symm_index(ind1, Ns[1]);
                            else if (boundary == CIRCULAR) ind1 = circular_wrap_index(ind1, Ns[1]);
                            else bounds_pad_flag = true;
                        }

                        if (bounds_pad_flag) {
                            indices[k] = fillvalue;
                        }
                        else {
                            indices[k] = in+ind0_memory+ind1*instr[1];
                        }
                        bounds_pad_flag = false;
                    }
                }
                mult_and_add(sum, hvals+j*hstr[0], hstr[1], indices, Nwin[1]);
            }
        }
    }
    efree(indices);
    return 0;
}

/**
 *
 * @return
 */
NDArray *
NDArray_Correlate2D(NDArray *a, NDArray *b, int mode, int boundary, NDArray* fill_value, int flip) {
    if (NDArray_DEVICE(a) == NDARRAY_DEVICE_GPU || NDArray_DEVICE(b) == NDARRAY_DEVICE_GPU) {
        zend_throw_error(NULL, "correlate2d not implemented for GPU computation.");
        return NULL;
    }
    NDArray *temp, *afill;
    int i, flag;
    afill = fill_value;
    int  *aout_dimens=NULL;
    if (NDArray_NDIM(a) != 2 || NDArray_NDIM(b) != 2) {
        zend_throw_error(NULL, "NDArray::correlate2d inputs must be both 2-D arrays");
        return NULL;
    }

    int inputs_swap = _inputs_swap(mode, a, b);
    if (inputs_swap == -1) {
        zend_throw_error(NULL,
         "For 'valid' mode, one input must be at least as large as the other in every dimension"
        );
        return NULL;
    }
    if (inputs_swap) {
        temp = a;
        a = b;
        b = temp;
    }

    if ((boundary == PAD) & (afill != NULL)) {
        if (NDArray_NUMELEMENTS(afill) != 1) {
            zend_throw_error(NULL, "fillValue must be a scalar or an array with one element.");
            return NULL;
        }
    } else {
        afill = NDArray_CreateFromFloatScalar(0);
        assert(afill != NULL);
    }

    aout_dimens = emalloc(NDArray_NDIM(a)*sizeof(int));
    assert(aout_dimens != NULL);

    switch (mode & OUTSIZE_MASK) {
        case VALID:
            for (i = 0; i < NDArray_NDIM(a); i++) {
                aout_dimens[i] = NDArray_SHAPE(a)[i] - NDArray_SHAPE(b)[i] + 1;
                if (aout_dimens[i] < 0) {
                    zend_throw_error(NULL,
                     "no part of the output is valid, use option 1 (same) or 2 (full)."
                    );
                    return NULL;
                }
            }
            break;
        case SAME:
            for (i = 0; i < NDArray_NDIM(a); i++) {
                aout_dimens[i] = NDArray_SHAPE(a)[i];
            }
            break;
        case FULL:
            for (i = 0; i < NDArray_NDIM(a); i++) {
                aout_dimens[i] = NDArray_SHAPE(a)[i];
            }
            break;
        default:
            zend_throw_error(NULL, "invalid mode.");
            return NULL;
    }

    NDArray *rtn = NDArray_Empty(aout_dimens, NDArray_NDIM(a), NDArray_TYPE(a), NDArray_DEVICE(a));

    flag = mode + boundary + (flip != 0) * FLIP_MASK;

    int rtn_status = _convolve2d(
            NDArray_DATA(a),        /* Input data Ns[0] x Ns[1] */
            NDArray_STRIDES(a),     /* Input strides */
            NDArray_DATA(rtn),       /* Output data */
            NDArray_STRIDES(rtn),    /* Output strides */
            NDArray_DATA(b),     /* coefficients in filter */
            NDArray_STRIDES(b),      /* coefficients strides */
            NDArray_SHAPE(b),     /* Size of kernel Nwin[0] x Nwin[1] */
            NDArray_SHAPE(a),        /* Size of image Ns[0] x Ns[1] */
            flag,       /* convolution parameters */
            NDArray_DATA(afill)
            );
    NDArray_FREE(afill);
    return rtn;
}