#include <stdio.h>
#include "ndarray.h"
#include "debug.h"
#include "iterators.h"
#include "initializers.h"
#include "types.h"
#include <php.h>
#include "../config.h"
#include "Zend/zend_alloc.h"
#include "Zend/zend_API.h"
#include <Zend/zend_types.h>

#ifdef HAVE_AVX2
#include <immintrin.h>
#endif

#ifdef HAVE_CUBLAS
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "ndmath/cuda/cuda_math.h"
#include "gpu_alloc.h"
#endif

#ifdef HAVE_GD
#include "gd.h"

typedef struct _gd_ext_image_object {
    gdImagePtr image;
    zend_object std;
} php_gd_image_object;

static
gdImagePtr gdImagePtr_from_zobj_p(zend_object *obj) {
    return ((php_gd_image_object *) ((char *) (obj) - XtOffsetOf(php_gd_image_object, std)))->image;
}

static
zend_always_inline php_gd_image_object *php_gd_exgdimage_from_zobj_p(zend_object *obj) {
    return (php_gd_image_object *) ((char *) (obj) - XtOffsetOf(php_gd_image_object, std));
}

void
php_gd_assign_libgdimageptr_as_extgdimage(zval *val, gdImagePtr image) {
    zend_class_entry *gd_image_ce = zend_lookup_class(zend_string_init("GdImage", strlen("GdImage"), 1));
    object_init_ex(val, gd_image_ce);
    php_gd_exgdimage_from_zobj_p(Z_OBJ_P(val))->image = image;
}

gdImagePtr gdImageCreateTrueColor_(int sx, int sy) {
    int i;
    gdImagePtr im;

    im = (gdImage *) emalloc(sizeof(gdImage));
    memset(im, 0, sizeof(gdImage));
    im->tpixels = (int **) emalloc(sizeof(int *) * sy);
    im->polyInts = 0;
    im->polyAllocated = 0;
    im->brush = 0;
    im->tile = 0;
    im->style = 0;
    for (i = 0; i < sy; i++) {
        im->tpixels[i] = (int *) ecalloc(sx, sizeof(int));
    }
    im->sx = sx;
    im->sy = sy;
    im->transparent = (-1);
    im->interlace = 0;
    im->trueColor = 1;
    /* 2.0.2: alpha blending is now on by default, and saving of alpha is
     * off by default. This allows font antialiasing to work as expected
     * on the first try in JPEGs -- quite important -- and also allows
     * for smaller PNGs when saving of alpha channel is not really
     * desired, which it usually isn't!
     */
    im->saveAlphaFlag = 0;
    im->alphaBlendingFlag = 1;
    im->thick = 1;
    im->AA = 0;
    im->cx1 = 0;
    im->cy1 = 0;
    im->cx2 = im->sx - 1;
    im->cy2 = im->sy - 1;
    im->res_x = GD_RESOLUTION;
    im->res_y = GD_RESOLUTION;
    im->interpolation = NULL;
    im->interpolation_id = GD_BILINEAR_FIXED;
    return im;
}

NDArray *
NDArray_FromGD(zval *a, bool channel_last) {
    NDArray *rtn;
    int color_index, j, i;
    int offset_green, offset_red, offset_blue;
    int red, green, blue;
    int *i_shape = emalloc(sizeof(int) * 3);
    gdImagePtr img_ptr = gdImagePtr_from_zobj_p(Z_OBJ_P(a));
    if (!channel_last) {
        i_shape[0] = 3;
        i_shape[1] = (int) img_ptr->sy;
        i_shape[2] = (int) img_ptr->sx;
    } else {
        i_shape[1] = (int) img_ptr->sy;
        i_shape[0] = (int) img_ptr->sx;
        i_shape[2] = 3;
    }
    rtn = NDArray_Zeros(i_shape, 3, NDARRAY_TYPE_FLOAT32, NDARRAY_DEVICE_CPU);
    for (i = 0; i < img_ptr->sy; i++) {
        for (j = 0; j < img_ptr->sx; j++) {
            if (img_ptr->trueColor) {
                if (!channel_last) {
                    offset_red = (NDArray_STRIDES(rtn)[0] / NDArray_ELSIZE(rtn) * 0) +
                                 ((NDArray_STRIDES(rtn)[1] / NDArray_ELSIZE(rtn)) * i) +
                                 ((NDArray_STRIDES(rtn)[2] / NDArray_ELSIZE(rtn)) * j);
                    offset_green = ((NDArray_STRIDES(rtn)[0] / NDArray_ELSIZE(rtn)) * 1) +
                                   ((NDArray_STRIDES(rtn)[1] / NDArray_ELSIZE(rtn)) * i) +
                                   ((NDArray_STRIDES(rtn)[2] / NDArray_ELSIZE(rtn)) * j);
                    offset_blue = ((NDArray_STRIDES(rtn)[0] / NDArray_ELSIZE(rtn)) * 2) +
                                  ((NDArray_STRIDES(rtn)[1] / NDArray_ELSIZE(rtn)) * i) +
                                  ((NDArray_STRIDES(rtn)[2] / NDArray_ELSIZE(rtn)) * j);
                } else {
                    offset_red = (NDArray_STRIDES(rtn)[2] / NDArray_ELSIZE(rtn) * 0) +
                                ((NDArray_STRIDES(rtn)[1] / NDArray_ELSIZE(rtn)) * i) +
                                ((NDArray_STRIDES(rtn)[0] / NDArray_ELSIZE(rtn)) * j);
                    offset_green = ((NDArray_STRIDES(rtn)[2] / NDArray_ELSIZE(rtn)) * 1) +
                                   ((NDArray_STRIDES(rtn)[1] / NDArray_ELSIZE(rtn)) * i) +
                                   ((NDArray_STRIDES(rtn)[0] / NDArray_ELSIZE(rtn)) * j);
                    offset_blue = ((NDArray_STRIDES(rtn)[2] / NDArray_ELSIZE(rtn)) * 2) +
                                  ((NDArray_STRIDES(rtn)[1] / NDArray_ELSIZE(rtn)) * i) +
                                  ((NDArray_STRIDES(rtn)[0] / NDArray_ELSIZE(rtn)) * j);
                }
                color_index = img_ptr->tpixels[i][j];
                red = (color_index >> 16) & 0xFF;
                green = (color_index >> 8) & 0xFF;
                blue = color_index & 0xFF;
                NDArray_FDATA(rtn)[offset_red] = (float) red;
                NDArray_FDATA(rtn)[offset_blue] = (float) blue;
                NDArray_FDATA(rtn)[offset_green] = (float) green;
            } else {
                if (!channel_last) {
                    offset_red = (NDArray_STRIDES(rtn)[0] / NDArray_ELSIZE(rtn) * 0) +
                                 ((NDArray_STRIDES(rtn)[1] / NDArray_ELSIZE(rtn)) * j) +
                                 ((NDArray_STRIDES(rtn)[2] / NDArray_ELSIZE(rtn)) * i);
                    offset_green = ((NDArray_STRIDES(rtn)[0] / NDArray_ELSIZE(rtn)) * 1) +
                                   ((NDArray_STRIDES(rtn)[1] / NDArray_ELSIZE(rtn)) * j) +
                                   ((NDArray_STRIDES(rtn)[2] / NDArray_ELSIZE(rtn)) * i);
                    offset_blue = ((NDArray_STRIDES(rtn)[0] / NDArray_ELSIZE(rtn)) * 2) +
                                  ((NDArray_STRIDES(rtn)[1] / NDArray_ELSIZE(rtn)) * j) +
                                  ((NDArray_STRIDES(rtn)[2] / NDArray_ELSIZE(rtn)) * i);
                } else {
                    offset_red = (NDArray_STRIDES(rtn)[2] / NDArray_ELSIZE(rtn) * 0) +
                                 ((NDArray_STRIDES(rtn)[1] / NDArray_ELSIZE(rtn)) * j) +
                                 ((NDArray_STRIDES(rtn)[0] / NDArray_ELSIZE(rtn)) * i);
                    offset_green = ((NDArray_STRIDES(rtn)[2] / NDArray_ELSIZE(rtn)) * 1) +
                                   ((NDArray_STRIDES(rtn)[1] / NDArray_ELSIZE(rtn)) * j) +
                                   ((NDArray_STRIDES(rtn)[0] / NDArray_ELSIZE(rtn)) * i);
                    offset_blue = ((NDArray_STRIDES(rtn)[2] / NDArray_ELSIZE(rtn)) * 2) +
                                  ((NDArray_STRIDES(rtn)[1] / NDArray_ELSIZE(rtn)) * j) +
                                  ((NDArray_STRIDES(rtn)[0] / NDArray_ELSIZE(rtn)) * i);
                }
                color_index = gdImagePalettePixel(img_ptr, i, j);
                red = img_ptr->red[color_index];
                green = img_ptr->green[color_index];
                blue = img_ptr->blue[color_index];
                NDArray_FDATA(rtn)[offset_red] = (float) red;
                NDArray_FDATA(rtn)[offset_blue] = (float) blue;
                NDArray_FDATA(rtn)[offset_green] = (float) green;
            }
        }
    }
    return rtn;
}

void
NDArray_ToGD(NDArray *a, NDArray *n_alpha, zval *output) {
    if (NDArray_NDIM(a) != 3 || NDArray_SHAPE(a)[0] != 3) {
        zend_throw_error(NULL, "Incompatible shape for image");
        return;
    }
    int color_index, i, j;
    int offset_green, offset_red, offset_blue, offset_alpha;
    int red, green, blue, alpha;
    gdImagePtr im = gdImageCreateTrueColor_(NDArray_SHAPE(a)[2], NDArray_SHAPE(a)[1]);

#ifdef HAVE_AVX2
    __m256i alpha_values;
    int elsize = NDArray_ELSIZE(a);
    __m256i alpha_mask = _mm256_set_epi32(0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF, 0xFF);

    for (i = 0; i < im->sy; i++) {
        for (j = 0; j < im->sx - 7; j += 8) {
            offset_alpha = (NDArray_STRIDES(a)[0] / elsize * i) +
                           ((NDArray_STRIDES(a)[1] / elsize) * j);
            offset_red = (NDArray_STRIDES(a)[0] / elsize * 0) +
                         ((NDArray_STRIDES(a)[1] / elsize) * i) +
                         ((NDArray_STRIDES(a)[2] / elsize) * j);
            offset_green = ((NDArray_STRIDES(a)[0] / elsize) * 1) +
                           ((NDArray_STRIDES(a)[1] / elsize) * i) +
                           ((NDArray_STRIDES(a)[2] / elsize) * j);
            offset_blue = ((NDArray_STRIDES(a)[0] / elsize) * 2) +
                          ((NDArray_STRIDES(a)[1] / elsize) * i) +
                          ((NDArray_STRIDES(a)[2] / elsize) * j);

            __m256 red_values = _mm256_loadu_ps(&NDArray_FDATA(a)[offset_red]);
            __m256 green_values = _mm256_loadu_ps(&NDArray_FDATA(a)[offset_green]);
            __m256 blue_values = _mm256_loadu_ps(&NDArray_FDATA(a)[offset_blue]);

            __m256i red_int = _mm256_cvtps_epi32(red_values);
            __m256i green_int = _mm256_cvtps_epi32(green_values);
            __m256i blue_int = _mm256_cvtps_epi32(blue_values);

            __m256i color_indices;

            if (n_alpha != NULL) {
                alpha_values = _mm256_cvtps_epi32(_mm256_loadu_ps(&NDArray_FDATA(n_alpha)[offset_alpha]));
                alpha_values = _mm256_and_si256(alpha_values, alpha_mask);

                color_indices = _mm256_or_si256(_mm256_or_si256(_mm256_slli_epi32(alpha_values, 24),
                                                                _mm256_slli_epi32(red_int, 16)),
                                                _mm256_or_si256(_mm256_slli_epi32(green_int, 8), blue_int));
            } else {
                // Handle the case when n_alpha is NULL
                // Set color_indices using only red, green, and blue values
                color_indices = _mm256_or_si256(_mm256_or_si256(_mm256_slli_epi32(red_int, 16),
                                                                _mm256_slli_epi32(green_int, 8)),
                                                blue_int);
            }

            _mm256_storeu_si256((__m256i *) &im->tpixels[i][j], color_indices);
        }
        for (; j < im->sx; j++) {
            offset_alpha = (NDArray_STRIDES(a)[0] / NDArray_ELSIZE(a) * i) +
                           ((NDArray_STRIDES(a)[1] / NDArray_ELSIZE(a)) * j);
            offset_red = (NDArray_STRIDES(a)[0] / NDArray_ELSIZE(a) * 0) +
                         ((NDArray_STRIDES(a)[1] / NDArray_ELSIZE(a)) * i) +
                         ((NDArray_STRIDES(a)[2] / NDArray_ELSIZE(a)) * j);
            offset_green = ((NDArray_STRIDES(a)[0] / NDArray_ELSIZE(a)) * 1) +
                           ((NDArray_STRIDES(a)[1] / NDArray_ELSIZE(a)) * i) +
                           ((NDArray_STRIDES(a)[2] / NDArray_ELSIZE(a)) * j);
            offset_blue = ((NDArray_STRIDES(a)[0] / NDArray_ELSIZE(a)) * 2) +
                          ((NDArray_STRIDES(a)[1] / NDArray_ELSIZE(a)) * i) +
                          ((NDArray_STRIDES(a)[2] / NDArray_ELSIZE(a)) * j);
            red = NDArray_FDATA(a)[offset_red];
            blue = NDArray_FDATA(a)[offset_blue];
            green = NDArray_FDATA(a)[offset_green];
            if (n_alpha != NULL) {
                alpha = NDArray_FDATA(n_alpha)[offset_alpha];
                color_index = (alpha << 24) | (red << 16) | (green << 8) | blue;
            } else {
                color_index = (red << 16) | (green << 8) | blue;
            }
            im->tpixels[i][j] = color_index;
        }
    }
#else
    for (int i = 0; i < im->sy; i++) {
        for (int j = 0; j < im->sx; j++) {
            offset_alpha = (NDArray_STRIDES(a)[0]/ NDArray_ELSIZE(a) * i) +
                           ((NDArray_STRIDES(a)[1]/ NDArray_ELSIZE(a)) * j);
            offset_red = (NDArray_STRIDES(a)[0]/ NDArray_ELSIZE(a) * 0) +
                         ((NDArray_STRIDES(a)[1]/ NDArray_ELSIZE(a)) * i) +
                         ((NDArray_STRIDES(a)[2]/ NDArray_ELSIZE(a)) * j);
            offset_green = ((NDArray_STRIDES(a)[0]/ NDArray_ELSIZE(a)) * 1) +
                           ((NDArray_STRIDES(a)[1]/ NDArray_ELSIZE(a)) * i) +
                           ((NDArray_STRIDES(a)[2]/ NDArray_ELSIZE(a)) * j);
            offset_blue = ((NDArray_STRIDES(a)[0]/ NDArray_ELSIZE(a)) * 2) +
                          ((NDArray_STRIDES(a)[1]/ NDArray_ELSIZE(a)) * i) +
                          ((NDArray_STRIDES(a)[2]/ NDArray_ELSIZE(a)) * j);
            red = NDArray_FDATA(a)[offset_red];
            blue = NDArray_FDATA(a)[offset_blue];
            green = NDArray_FDATA(a)[offset_green];
            if (n_alpha != NULL) {
                alpha = NDArray_FDATA(n_alpha)[offset_alpha];
                color_index = (alpha << 24) | (red << 16) | (green << 8) | blue;
            } else {
                color_index = (red << 16) | (green << 8) | blue;
            }
            im->tpixels[i][j] = color_index;
        }
    }
#endif
    php_gd_assign_libgdimageptr_as_extgdimage(output, im);
}

#endif

char*
convert_shape_to_string(int n, int const *vals, char *endin)
{
    int i;
    if (n == 0) {
        return NULL;
    }
    char *value = emalloc(sizeof(char) * (n + 3));
    value[0] = '(';
    for (i = 1; i <= n; i++) {
        value[i] = (char)vals[i];
    }
    value[i + 1] = ')';
    value[i + 2] = *endin;
    return value;
}

int
broadcast_strides(int ndim, int const *shape,
                  int strides_ndim, int const *strides_shape, int const *strides,
                  char const *strides_name,
                  int *out_strides)
{
    int idim, idim_start = ndim - strides_ndim;

    /* Can't broadcast to fewer dimensions */
    if (idim_start < 0) {
        goto broadcast_error;
    }

    for (idim = ndim - 1; idim >= idim_start; --idim) {
        int strides_shape_value = strides_shape[idim - idim_start];
        /* If it doesn't have dimension one, it must match */
        if (strides_shape_value == 1) {
            out_strides[idim] = 0;
        }
        else if (strides_shape_value != shape[idim]) {
            goto broadcast_error;
        }
        else {
            out_strides[idim] = strides[idim - idim_start];
        }
    }

    /* New dimensions get a zero stride */
    for (idim = 0; idim < idim_start; ++idim) {
        out_strides[idim] = 0;
    }

    return 0;

broadcast_error: {
        char *shape1 = convert_shape_to_string(strides_ndim, strides_shape, "");
        if (shape1 == NULL) {
            return -1;
        }

        char *shape2 = convert_shape_to_string(ndim, shape, "");
        if (shape2 == NULL) {
            efree(shape1);
            return -1;
        }
        zend_throw_error(NULL,
                         "could not broadcast %s from shape %s into shape %s",
                         strides_name, shape1, shape2);
        efree(shape1);
        efree(shape2);
        return -1;
    }
}

void apply_reduce(NDArray *result, NDArray *target, NDArray *(*operation)(NDArray *, NDArray *)) {
    NDArray *temp = operation(result, target);
    if (NDArray_DEVICE(target) == NDARRAY_DEVICE_CPU) {
        memcpy(result->data, temp->data, result->descriptor->numElements * sizeof(float));
    } else {
#ifdef HAVE_CUBLAS
        vmemcpyd2d(NDArray_DATA(temp), NDArray_DATA(result), result->descriptor->numElements * sizeof(float));
#endif
    }
    NDArray_FREE(temp);
}

void apply_single_reduce(NDArray *result, NDArray *target, float (*operation)(NDArray *)) {
    float temp;
    float temp2;
    float tmp_result;
    int *tmp_shape = emalloc(sizeof(int));
    tmp_shape[0] = 2;

    NDArray *tmp = NDArray_Zeros(tmp_shape, 2, NDARRAY_TYPE_FLOAT32, NDArray_DEVICE(result));
    if (NDArray_NDIM(target) >= 1) {
        temp = operation(target);
        temp2 = operation(result);
    } else {
        temp = NDArray_FDATA(target)[0];
        temp2 = NDArray_FDATA(result)[0];
    }
    NDArray_FDATA(tmp)[0] = temp;
    NDArray_FDATA(tmp)[1] = temp2;

    tmp_result = operation(tmp);
    if (NDArray_DEVICE(target) == NDARRAY_DEVICE_CPU) {
        memcpy(result->data, &tmp_result, sizeof(float));
    }
}

void _reduce(int current_axis, int rtn_init, int *axis, NDArray *target, NDArray *rtn,
             NDArray *(*operation)(NDArray *, NDArray *)) {
    NDArray *slice;
    NDArray *rtn_slice;
    NDArrayIterator_REWIND(target);
    while (!NDArrayIterator_ISDONE(target)) {
        slice = NDArrayIterator_GET(target);
        if (axis != NULL && current_axis < *axis) {
            rtn_slice = NDArrayIterator_GET(rtn);
            _reduce(current_axis + 1, rtn_init, axis, slice, rtn_slice, operation);
            NDArrayIterator_NEXT(rtn);
            NDArrayIterator_NEXT(target);
            NDArray_FREE(rtn_slice);
            NDArray_FREE(slice);
            continue;
        }
        if (rtn_init == 0) {
            rtn_init = 1;
            if (NDArray_DEVICE(rtn) == NDARRAY_DEVICE_CPU) {
                memcpy(rtn->data, slice->data, rtn->descriptor->numElements * sizeof(float));
            }
#ifdef HAVE_CUBLAS
            if (NDArray_DEVICE(rtn) == NDARRAY_DEVICE_GPU) {
                vmemcpyd2d(NDArray_DATA(slice), NDArray_DATA(rtn),
                                    rtn->descriptor->numElements * sizeof(float));
            }
#endif
            NDArrayIterator_NEXT(target);
            NDArray_FREE(slice);
            continue;
        }
        apply_reduce(rtn, slice, operation);
        NDArrayIterator_NEXT(target);
        NDArray_FREE(slice);
    }
}

void _single_reduce(int current_axis, int rtn_init, int *axis, NDArray *target, NDArray *rtn,
                    float (*operation)(NDArray *)) {
    NDArray *slice;
    NDArray *rtn_slice;
    NDArrayIterator_REWIND(target);

    while (!NDArrayIterator_ISDONE(target)) {
        slice = NDArrayIterator_GET(target);
        if (axis != NULL && current_axis < *axis) {
            rtn_slice = NDArrayIterator_GET(rtn);
            _single_reduce(current_axis + 1, rtn_init, axis, slice, rtn_slice, operation);
            NDArrayIterator_NEXT(rtn);
            NDArrayIterator_NEXT(target);
            NDArray_FREE(rtn_slice);
            NDArray_FREE(slice);
            continue;
        }
        apply_single_reduce(rtn, slice, operation);
        NDArrayIterator_NEXT(target);
        NDArray_FREE(slice);
    }
}

/**
 * Single Reduce function
 *
 * @param array
 * @param shape
 * @param strides
 * @param ndim
 * @param axis
 * @return
 */
NDArray *
single_reduce(NDArray *array, int *axis, float (*operation)(NDArray *)) {
    char *exception_buffer[256];

    if (axis == NULL) {
        axis = emalloc(sizeof(int));
        *axis = 0;
    }

    if (*axis >= NDArray_NDIM(array)) {
        sprintf((char *) exception_buffer, "axis %d is out of bounds for array of dimension %d", *axis,
                NDArray_NDIM(array));
        zend_throw_error(NULL, "%s", (const char *) exception_buffer);
        return NULL;
    }

    // Calculate the size and strides of the reduced output
    int out_dim = 0;
    int out_ndim;

    for (int i = 0; i < NDArray_NDIM(array); i++) {
        if (i != *axis) {
            out_dim++;
        }
    }

    out_ndim = out_dim;
    int *out_shape = emalloc(sizeof(int) * out_ndim);

    int j = 0;
    for (int i = 0; i < NDArray_NDIM(array); i++) {
        if (i != *axis) {
            out_shape[j] = NDArray_SHAPE(array)[i];
            j++;
        }
    }

    // Calculate the size of the reduced buffer
    int reduced_buffer_size = 1;
    for (int i = 0; i < out_dim; i++) {
        reduced_buffer_size *= out_shape[i];
    }

    // Allocate memory for the reduced buffer
    NDArray *rtn = NDArray_Zeros(out_shape, out_ndim, NDARRAY_TYPE_FLOAT32, NDArray_DEVICE(array));
    _single_reduce(0, 0, axis, array, rtn, operation);
    return rtn;
}

/**
 * Reduce function
 *
 * @param array
 * @param shape
 * @param strides
 * @param ndim
 * @param axis
 * @return
 */
NDArray *
reduce(NDArray *array, int *axis, NDArray *(*operation)(NDArray *, NDArray *)) {
    char *exception_buffer[256];
    int null_axis = 0;

    if (axis == NULL) {
        null_axis = 1;
        axis = emalloc(sizeof(int));
        *axis = 0;
    }

    if (*axis >= NDArray_NDIM(array)) {
        sprintf((char *) exception_buffer, "axis %d is out of bounds for array of dimension %d", *axis,
                NDArray_NDIM(array));
        zend_throw_error(NULL, "%s", (const char *) exception_buffer);
        return NULL;
    }

    // Calculate the size and strides of the reduced output
    int out_dim = 0;
    int out_ndim;

    for (int i = 0; i < NDArray_NDIM(array); i++) {
        if (i != *axis) {
            out_dim++;
        }
    }

    out_ndim = out_dim;
    int *out_shape = emalloc(sizeof(int) * out_ndim);

    int j = 0;
    for (int i = 0; i < NDArray_NDIM(array); i++) {
        if (i != *axis) {
            out_shape[j] = NDArray_SHAPE(array)[i];
            j++;
        }
    }

    // Calculate the size of the reduced buffer
    int reduced_buffer_size = 1;
    for (int i = 0; i < out_dim; i++) {
        reduced_buffer_size *= out_shape[i];
    }

    // Allocate memory for the reduced buffer
    NDArray *rtn = NDArray_Zeros(out_shape, out_ndim, NDARRAY_TYPE_FLOAT32, NDArray_DEVICE(array));
    _reduce(0, 0, axis, array, rtn, operation);

    if (null_axis == 1) {
        efree(axis);
    }

    NDArrayIterator_REWIND(array);
    return rtn;
}

#pragma clang diagnostic push
#pragma ide diagnostic ignored "misc-no-recursion"
/**
 * Free NDArray
 *
 * @param array
 */
void
NDArray_FREE(NDArray *array) {
    if (array == NULL || array->refcount == -1) {
        return;
    }

    // Decrement the reference count
    if (array->refcount > 0) {
        NDArray_DELREF(array);
    }

    // If the reference count reaches zero, free the memory
    if (array->refcount == 0) {
        if (array->iterator != NULL) {
            NDArrayIterator_FREE(array);
        }

        if (array->strides != NULL) {
            efree(array->strides);
        }

        if (array->dimensions != NULL) {
            efree(array->dimensions);
        }

        if (array->data != NULL && array->base == NULL && array->descriptor->numElements > 0) {
            if (NDArray_DEVICE(array) == NDARRAY_DEVICE_CPU) {
                efree(array->data);
            } else {
#ifdef HAVE_CUBLAS
                vfree(array->data);
#endif
            }
        }
        if (array->base != NULL) {
            NDArray_FREE(array->base);
        }

        if (array->descriptor != NULL) {
            efree(array->descriptor);
        }
        array->refcount = -1;
        efree(array);
        array = NULL;
    }
}
#pragma clang diagnostic pop

/**
 * Free NDArray data buffer regardless of references
 *
 * @param target
 */
void
NDArray_FREEDATA(NDArray *target) {
    if (NDArray_DEVICE(target) == NDARRAY_DEVICE_CPU) {
        efree(target->data);
    }
#ifdef HAVE_CUBLAS
    if (NDArray_DEVICE(target) == NDARRAY_DEVICE_GPU) {
        vfree(target->data);
    }
#endif
    target->data = NULL;
}

/**
 * Print NDArray or return the print string
 *
 * @param array
 * @param do_return
 * @return
 */
char *
NDArray_Print(NDArray *array, int do_return) {
    assert(array != NULL);
    char *str;
    if (is_type(NDArray_TYPE(array), NDARRAY_TYPE_DOUBLE64)) {
        str = print_matrix(NDArray_DDATA(array), NDArray_NDIM(array), NDArray_SHAPE(array),
                           NDArray_STRIDES(array), NDArray_NUMELEMENTS(array), NDArray_DEVICE(array));
    }
    if (is_type(NDArray_TYPE(array), NDARRAY_TYPE_FLOAT32)) {
        str = print_matrix_float(NDArray_FDATA(array), NDArray_NDIM(array), NDArray_SHAPE(array),
                                 NDArray_STRIDES(array), NDArray_NUMELEMENTS(array), NDArray_DEVICE(array));
    }
    if (do_return == 0) {
        printf("%s", str);
        return NULL;
    }
    return str;
}

/**
 * @param array
 */
NDArray *
NDArray_Map(NDArray *array, ElementWiseDoubleOperation op) {
    NDArray *rtn;
    int i;
    int *new_shape = emalloc(sizeof(int) * NDArray_NDIM(array));
    memcpy(new_shape, NDArray_SHAPE(array), sizeof(int) * NDArray_NDIM(array));
    rtn = NDArray_Zeros(new_shape, NDArray_NDIM(array), NDARRAY_TYPE_FLOAT32, NDArray_DEVICE(array));
    for (i = 0; i < NDArray_NUMELEMENTS(array); i++) {
        NDArray_FDATA(rtn)[i] = op(NDArray_FDATA(array)[i]);
    }
    return rtn;
}

/**
 * @param array
 */
NDArray *
NDArray_Map1F(NDArray *array, ElementWiseFloatOperation1F op, float val1) {
    NDArray *rtn;
    int i;
    int *new_shape = emalloc(sizeof(int) * NDArray_NDIM(array));
    memcpy(new_shape, NDArray_SHAPE(array), sizeof(int) * NDArray_NDIM(array));
    rtn = NDArray_Zeros(new_shape, NDArray_NDIM(array), NDARRAY_TYPE_FLOAT32, NDArray_DEVICE(array));

    for (i = 0; i < NDArray_NUMELEMENTS(array); i++) {
        NDArray_FDATA(rtn)[i] = op(NDArray_FDATA(array)[i], val1);
    }
    return rtn;
}

/**
 * @param array
 */
NDArray *
NDArray_Map1ND(NDArray *array, ElementWiseFloatOperation1F op, NDArray *val1) {
    NDArray *rtn;
    int i;
    int *new_shape = emalloc(sizeof(int) * NDArray_NDIM(array));
    memcpy(new_shape, NDArray_SHAPE(array), sizeof(int) * NDArray_NDIM(array));
    rtn = NDArray_Zeros(new_shape, NDArray_NDIM(array), NDARRAY_TYPE_FLOAT32, NDArray_DEVICE(array));

    for (i = 0; i < NDArray_NUMELEMENTS(array); i++) {
        NDArray_FDATA(rtn)[i] = op(NDArray_FDATA(array)[i], NDArray_FDATA(val1)[i]);
    }
    return rtn;
}

/**
 * @param array
 */
NDArray *
NDArray_Map2F(NDArray *array, ElementWiseFloatOperation2F op, float val1, float val2) {
    NDArray *rtn;
    int i;
    int *new_shape = emalloc(sizeof(int) * NDArray_NDIM(array));
    memcpy(new_shape, NDArray_SHAPE(array), sizeof(int) * NDArray_NDIM(array));
    rtn = NDArray_Zeros(new_shape, NDArray_NDIM(array), NDARRAY_TYPE_FLOAT32, NDArray_DEVICE(array));

    for (i = 0; i < NDArray_NUMELEMENTS(array); i++) {
        NDArray_FDATA(rtn)[i] = op(NDArray_FDATA(array)[i], val1, val2);
    }
    return rtn;
}

/**
 * Return minimum value of NDArray
 *
 * @param target
 * @return
 */
float
NDArray_Min(NDArray *target) {
    float *array = NDArray_FDATA(target);
    int length = NDArray_NUMELEMENTS(target);
    float min;
    if (NDArray_DEVICE(target) == NDARRAY_DEVICE_GPU) {
#ifdef HAVE_CUBLAS
        return cuda_min_float(array, NDArray_NUMELEMENTS(target));
#else
        return -1.f;
#endif
    } else {
        min = array[0];
        for (int i = 1; i < length; i++) {
            if (array[i] < min) {
                min = array[i];
            }
        }
    }
    return min;
}

/**
 * Return maximum value of NDArray over a given axis
 *
 * @param target
 * @param axis
 * @return
 */
NDArray *
NDArray_MaxAxis(NDArray *target, int axis) {
    if (NDArray_DEVICE(target) == NDARRAY_DEVICE_GPU) {
        zend_throw_error(NULL, "GPU NDArray_MaxAxis not implemented");
        return NULL;
    }

    if (axis < 0 || axis >= NDArray_NDIM(target)) {
        // Invalid axis
        zend_throw_error(NULL, "Invalid axis.\n");
        return NULL;
    }
    int *output_shape = emalloc(sizeof(int) * (NDArray_NDIM(target) - 1));

    int axis_size = NDArray_SHAPE(target)[axis];
    int num_elements = 1;

    // Calculate the number of elements before the specified axis
    for (int i = 0; i < axis; i++) {
        num_elements *= NDArray_SHAPE(target)[i];
    }

    // Calculate the output shape (excluding the specified axis)
    int output_index = 0;
    for (int i = 0; i < NDArray_NDIM(target); i++) {
        if (i != axis) {
            output_shape[output_index++] = NDArray_SHAPE(target)[i];
        }
    }

    NDArray *rtn = NDArray_Empty(output_shape, NDArray_NDIM(target) - 1, NDArray_TYPE(target), NDArray_DEVICE(target));

    int axis_first_element_index, j, i, axis_step;
    if (axis == 0) {
        for (i = 0; i < NDArray_NUMELEMENTS(rtn); i++) {
            axis_step = (NDArray_STRIDES(target)[axis] / NDArray_ELSIZE(target));
            axis_first_element_index = i * (NDArray_STRIDES(target)[NDArray_NDIM(target) - 1] / NDArray_ELSIZE(target));
            NDArray_FDATA(rtn)[i] = NDArray_FDATA(target)[axis_first_element_index];
            for (j = 1; j < axis_size; j++) {
                float current_val = NDArray_FDATA(target)[axis_first_element_index + (j * axis_step)];
                if (current_val > NDArray_FDATA(rtn)[i]) {
                    NDArray_FDATA(rtn)[i] = current_val;
                }
            }
        }
    } else {
        for (i = 0; i < NDArray_NUMELEMENTS(rtn); i++) {
            axis_step = (NDArray_STRIDES(target)[axis] / NDArray_ELSIZE(target));
            if (NDArray_NDIM(target) - 1 == axis) {
                axis_first_element_index = i * (NDArray_STRIDES(target)[axis - 1] / NDArray_ELSIZE(target));
            } else {
                axis_first_element_index = i * (NDArray_STRIDES(target)[NDArray_NDIM(target) + 1] / NDArray_ELSIZE(target));
            }
            NDArray_FDATA(rtn)[i] = NDArray_FDATA(target)[axis_first_element_index];
            for (j = 1; j < axis_size; j++) {
                float current_val = NDArray_FDATA(target)[axis_first_element_index + (j * axis_step)];
                if (current_val > NDArray_FDATA(rtn)[i]) {
                    NDArray_FDATA(rtn)[i] = current_val;
                }
            }
        }
    }
    return rtn;
}


/**
 * @param a
 * @param b
 * @return
 */
NDArray *
NDArray_Maximum(NDArray *a, NDArray *b) {
    if (NDArray_DEVICE(a) == NDARRAY_DEVICE_GPU || NDArray_DEVICE(b) == NDARRAY_DEVICE_GPU) {
        zend_throw_error(NULL, "NDArray_Maximum not implemented for GPU");
        return NULL;
    }

    NDArray *broadcasted = NULL;
    NDArray *a_broad = NULL, *b_broad = NULL;
    if (NDArray_NUMELEMENTS(a) < NDArray_NUMELEMENTS(b)) {
        broadcasted = NDArray_Broadcast(a, b);
        a_broad = broadcasted;
        b_broad = b;
    } else if (NDArray_NUMELEMENTS(b) < NDArray_NUMELEMENTS(a)) {
        broadcasted = NDArray_Broadcast(b, a);
        b_broad = broadcasted;
        a_broad = a;
    } else {
        b_broad = b;
        a_broad = a;
    }

    if (b_broad == NULL || a_broad == NULL) {
        zend_throw_error(NULL, "Can't broadcast arrays.");
        return NULL;
    }

    NDArray *rtn = NDArray_EmptyLike(a_broad);
    for (int i = 0; i < NDArray_NUMELEMENTS(a); i++) {
        NDArray_FDATA(rtn)[i] = fmaxf(NDArray_FDATA(a_broad)[i], NDArray_FDATA(b_broad)[i]);
    }

    if (broadcasted != NULL) {
        NDArray_FREE(broadcasted);
    }
    return rtn;
}

/**
 * @param a
 * @param b
 * @return
 */
NDArray *
NDArray_Minimum(NDArray *a, NDArray *b) {
    if (NDArray_DEVICE(a) == NDARRAY_DEVICE_GPU || NDArray_DEVICE(b) == NDARRAY_DEVICE_GPU) {
        zend_throw_error(NULL, "NDArray_Minimum not implemented for GPU");
        return NULL;
    }

    NDArray *broadcasted = NULL;
    NDArray *a_broad = NULL, *b_broad = NULL;
    if (NDArray_NUMELEMENTS(a) < NDArray_NUMELEMENTS(b)) {
        broadcasted = NDArray_Broadcast(a, b);
        a_broad = broadcasted;
        b_broad = b;
    } else if (NDArray_NUMELEMENTS(b) < NDArray_NUMELEMENTS(a)) {
        broadcasted = NDArray_Broadcast(b, a);
        b_broad = broadcasted;
        a_broad = a;
    } else {
        b_broad = b;
        a_broad = a;
    }

    if (b_broad == NULL || a_broad == NULL) {
        zend_throw_error(NULL, "Can't broadcast arrays.");
        return NULL;
    }

    NDArray *rtn = NDArray_EmptyLike(a_broad);
    for (int i = 0; i < NDArray_NUMELEMENTS(a); i++) {
        NDArray_FDATA(rtn)[i] = fminf(NDArray_FDATA(a_broad)[i], NDArray_FDATA(b_broad)[i]);
    }

    if (broadcasted != NULL) {
        NDArray_FREE(broadcasted);
    }
    return rtn;
}

/**
 * Return maximum value of NDArray
 *
 * @param target
 * @return
 */
float
NDArray_Max(NDArray *target) {
    float max;
    float *array = NDArray_FDATA(target);
    int length = NDArray_NUMELEMENTS(target);
    if (NDArray_DEVICE(target) == NDARRAY_DEVICE_GPU) {
#ifdef HAVE_CUBLAS
        return cuda_max_float(array, NDArray_NUMELEMENTS(target));
#else
        return -1.f;
#endif
    } else {
        max = array[0];
        for (int i = 1; i < length; i++) {
            if (array[i] > max) {
                max = array[i];
            }
        }
    }
    return max;
}

#pragma clang diagnostic push
#pragma ide diagnostic ignored "misc-no-recursion"
/**
 * @param data
 * @param strides
 * @param dimensions
 * @param ndim
 * @return
 */
zval
convertToStridedArrayToPHPArray(float *data, int *strides, int *dimensions, int ndim, int elsize) {
    zval phpArray;
    int i;

    array_init_size(&phpArray, ndim);

    for (i = 0; i < dimensions[0]; i++) {
        if (ndim > 1) {
            int j;
            zval subArray;

            float *subData = data + (i * (strides[0] / elsize));
            int *subStrides = strides + 1;
            int *subDimensions = dimensions + 1;

            subArray = convertToStridedArrayToPHPArray(subData, subStrides, subDimensions, ndim - 1, elsize);

            add_index_zval(&phpArray, i, &subArray);
        } else {
            add_index_double(&phpArray, i, *(data + (i * (*strides / elsize))));
        }
    }
    return phpArray;
}
#pragma clang diagnostic pop

/**
 * Convert a NDArray to PHP Array
 *
 * @return
 */
zval
NDArray_ToPHPArray(NDArray *target) {
    zval phpArray;
    phpArray = convertToStridedArrayToPHPArray(NDArray_FDATA(target), NDArray_STRIDES(target),
                                               NDArray_SHAPE(target), NDArray_NDIM(target), NDArray_ELSIZE(target));
    return phpArray;
}

/**
 * @param nda
 * @return
 */
int *
NDArray_ToIntVector(NDArray *nda) {
    double *tmp_val = emalloc(sizeof(float));
    int *vector = emalloc(sizeof(int) * NDArray_NUMELEMENTS(nda));
    for (int i = 0; i < NDArray_NUMELEMENTS(nda); i++) {
        if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_GPU) {
#ifdef HAVE_CUBLAS
            cudaMemcpy(tmp_val, &NDArray_FDATA(nda)[i], sizeof(float), cudaMemcpyDeviceToHost);
            vector[i] = (int) *tmp_val;
            continue;
#endif
        }
        vector[i] = (int) NDArray_FDATA(nda)[i];
    }
    efree(tmp_val);
    return vector;
}

/**
 * Transfer NDArray to GPU and return a copy
 *
 * @param target
 */
NDArray *
NDArray_ToGPU(NDArray *target) {
#ifdef HAVE_CUBLAS
    float *tmp_gpu;
    int *new_shape;
    int n_ndim = NDArray_NDIM(target);

    if (NDArray_DEVICE(target) == NDARRAY_DEVICE_GPU) {
        return NDArray_Copy(target, NDARRAY_DEVICE_GPU);
    }

    new_shape = emalloc(sizeof(int) * NDArray_NDIM(target));
    memcpy(new_shape, NDArray_SHAPE(target), sizeof(int) * NDArray_NDIM(target));

    NDArray *rtn = NDArray_Zeros(new_shape, n_ndim, NDARRAY_TYPE_FLOAT32, NDArray_DEVICE(target));
    rtn->device = NDARRAY_DEVICE_GPU;

    vmalloc((void **) &tmp_gpu, NDArray_NUMELEMENTS(target) * sizeof(float));
    cudaMemcpy(tmp_gpu, NDArray_FDATA(target), NDArray_NUMELEMENTS(target) * sizeof(float), cudaMemcpyHostToDevice);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        zend_throw_error(NULL, "Error synchronizing: %s\n", cudaGetErrorString(err));
        return NULL;
    }
    efree(rtn->data);
    rtn->data = (char *) tmp_gpu;
    return rtn;
#else
    zend_throw_error(NULL, "Unable to detect a compatible device.");
    return NULL;
#endif
}

/**
 * Transfer NDArray to CPU and return a copy
 *
 * @param target
 */
NDArray *
NDArray_ToCPU(NDArray *target) {
    int *new_shape;
    int n_ndim = NDArray_NDIM(target);

    if (NDArray_DEVICE(target) == NDARRAY_DEVICE_CPU) {
        return NDArray_Copy(target, NDARRAY_DEVICE_CPU);
    }

    new_shape = emalloc(sizeof(int) * NDArray_NDIM(target));
    memcpy(new_shape, NDArray_SHAPE(target), sizeof(int) * NDArray_NDIM(target));

    NDArray *rtn = NDArray_Empty(new_shape, n_ndim, NDARRAY_TYPE_FLOAT32, NDARRAY_DEVICE_CPU);
    rtn->device = NDARRAY_DEVICE_CPU;
#ifdef HAVE_CUBLAS
    cudaMemcpy(rtn->data, NDArray_FDATA(target), NDArray_NUMELEMENTS(target) * sizeof(float), cudaMemcpyDeviceToHost);
#endif
    return rtn;
}

/**
 * Return 1 if a.shape == b.shape or 0
 *
 * @param a
 * @param b
 * @return
 */
int
NDArray_ShapeCompare(NDArray *a, NDArray *b) {
    if (NDArray_NDIM(a) != NDArray_NDIM(b)) {
        return 0;
    }

    for (int i = 0; i < NDArray_NDIM(a); i++) {
        if (NDArray_SHAPE(a)[i] != NDArray_SHAPE(b)[i]) {
            return 0;
        }
    }

    return 1;
}

/**
 * Check if two NDArray are broadcastable
 *
 * @param arr1
 * @param arr2
 * @return
 */
int
NDArray_IsBroadcastable(const NDArray *array1, const NDArray *array2) {
    if (NDArray_NDIM(array1) == 1 && NDArray_NDIM(array2) > 1) {
        if (NDArray_SHAPE(array1)[0] == NDArray_SHAPE(array2)[NDArray_NDIM(array2) - 1]) {
            return 1;
        } else {
            return 0;
        }
    }
    if (NDArray_NDIM(array1) > 1 && NDArray_NDIM(array2) == 1) {
        if (NDArray_SHAPE(array2)[0] == NDArray_SHAPE(array1)[NDArray_NDIM(array1) - 1]) {
            return 1;
        } else {
            return 0;
        }
    }

    // Determine the maximum number of dimensions
    int maxDims = (NDArray_NDIM(array1) > NDArray_NDIM(array2)) ? NDArray_NDIM(array1) : NDArray_NDIM(array2);

    // Pad the shape arrays with 1s if necessary
    int paddedShape1[maxDims];
    int paddedShape2[maxDims];

    for (int i = 0; i < maxDims; i++) {
        paddedShape1[i] = (i < NDArray_NDIM(array1)) ? NDArray_SHAPE(array1)[i] : 1;
        paddedShape2[i] = (i < NDArray_NDIM(array2)) ? NDArray_SHAPE(array2)[i] : 1;
    }

    // Check if the modified arrays are broadcastable
    for (int i = 0; i < maxDims; i++) {
        if (paddedShape1[i] != paddedShape2[i] && paddedShape1[i] != 1 && paddedShape2[i] != 1) {
            return 0;
        }
    }

    // Arrays are broadcastable
    return 1;
}

/**
 * Broadcast NDArrays
 *
 * @todo Implement ND broadcast
 * @param a
 * @param b
 * @return
 */
NDArray *
NDArray_Broadcast(NDArray *a, NDArray *b) {
    int i;
    NDArray *src, *dst, *rtn;
    src = a;
    dst = b;
    if (NDArray_NDIM(a) == NDArray_NDIM(b)) {
        int all_equal = 1;
        for (i = 0; i < NDArray_NDIM(a); i++) {
            if (NDArray_SHAPE(a)[i] != NDArray_SHAPE(b)[i]) {
                all_equal = 0;
            }
        }
        if (all_equal == 1) {
            return src;
        }
    }

    if (!NDArray_IsBroadcastable(src, dst)) {
        zend_throw_error(NULL, "Broadcast shape mismatch.");
        return NULL;
    }
    rtn = NDArray_EmptyLike(dst);
    char *rtn_p = NDArray_DATA(rtn);
    if (NDArray_NDIM(a) == 0 && NDArray_NDIM(b) > 0) {
        for (i = 0; i < NDArray_NUMELEMENTS(b); i++) {
            NDArray_FDATA(rtn)[i] = NDArray_FDATA(a)[0];
        }
    }

    if (NDArray_NDIM(src) == 1 && NDArray_NDIM(dst) > 1) {

        if (NDArray_SHAPE(src)[0] == NDArray_SHAPE(dst)[NDArray_NDIM(dst) - 2]
            || NDArray_SHAPE(src)[0] == NDArray_SHAPE(dst)[NDArray_NDIM(dst) - 1]) {
            if (NDArray_DEVICE(dst) == NDARRAY_DEVICE_CPU) {
                for (i = 0; i < NDArray_SHAPE(dst)[NDArray_NDIM(dst) - 2]; i++) {
                    memcpy(rtn_p,
                           NDArray_FDATA(src), sizeof(float) * NDArray_SHAPE(dst)[NDArray_NDIM(dst) - 1]);
                    rtn_p = rtn_p + (sizeof(float) * NDArray_SHAPE(src)[0]);
                }
            }
#ifdef HAVE_CUBLAS
            if (NDArray_DEVICE(dst) == NDARRAY_DEVICE_GPU) {
                for (i = 0; i < NDArray_SHAPE(dst)[NDArray_NDIM(dst) - 2]; i++) {
                    vmemcpyd2d(NDArray_DATA(src), rtn_p,
                                        sizeof(float) * NDArray_SHAPE(dst)[NDArray_NDIM(dst) - 1]);
                    rtn_p = rtn_p + (sizeof(float) * NDArray_SHAPE(src)[0]);
                }
            }
#endif
        }
    }
    int j;
    if (NDArray_NDIM(src) == 2 && NDArray_NDIM(dst) == 2) {
        if (NDArray_SHAPE(src)[NDArray_NDIM(dst) - 2] == NDArray_SHAPE(dst)[NDArray_NDIM(dst) - 2]) {
            if (NDArray_DEVICE(dst) == NDARRAY_DEVICE_CPU) {
                if (NDArray_NUMELEMENTS(src) != 1) {
                    for (i = 0; i < NDArray_SHAPE(dst)[NDArray_NDIM(dst) - 2]; i++) {
                        for (j = 0; j < NDArray_SHAPE(dst)[NDArray_NDIM(dst) - 1]; j++) {
                            NDArray_FDATA(rtn)[(i * NDArray_STRIDES(rtn)[NDArray_NDIM(rtn) - 2] / NDArray_ELSIZE(rtn)) +
                                               j] = NDArray_FDATA(src)[i];
                        }
                    }
                } else {
                    for (i = 0; i < NDArray_SHAPE(dst)[NDArray_NDIM(dst) - 2]; i++) {
                        for (j = 0; j < NDArray_SHAPE(dst)[NDArray_NDIM(dst) - 1]; j++) {
                            NDArray_FDATA(rtn)[(i * NDArray_STRIDES(rtn)[NDArray_NDIM(rtn) - 2] / NDArray_ELSIZE(rtn)) +
                                               j] = NDArray_FDATA(src)[0];
                        }
                    }
                    return rtn;
                }
            }
#ifdef HAVE_CUBLAS
            char *tmp_p;
            if (NDArray_DEVICE(dst) == NDARRAY_DEVICE_GPU) {
                if (NDArray_NUMELEMENTS(src) != 1) {
                    for (i = 0; i < NDArray_SHAPE(dst)[NDArray_NDIM(dst) - 2]; i++) {
                        for (j = 0; j < NDArray_SHAPE(dst)[NDArray_NDIM(dst) - 1]; j++) {
                            tmp_p = (char *) (NDArray_FDATA(src) + i);
                            rtn_p = (char *) (NDArray_FDATA(rtn) +
                                              (i * NDArray_STRIDES(rtn)[NDArray_NDIM(rtn) - 2] / NDArray_ELSIZE(rtn)) +
                                              j);
                            vmemcpyd2d(tmp_p, rtn_p, sizeof(float));
                        }
                    }
                } else {
                    for (i = 0; i < NDArray_SHAPE(dst)[NDArray_NDIM(dst) - 2]; i++) {
                        for (j = 0; j < NDArray_SHAPE(dst)[NDArray_NDIM(dst) - 1]; j++) {
                            tmp_p = (char *) (NDArray_FDATA(src));
                            rtn_p = (char *) (NDArray_FDATA(rtn) +
                                              (i * NDArray_STRIDES(rtn)[NDArray_NDIM(rtn) - 2] / NDArray_ELSIZE(rtn)) +
                                              j);
                            vmemcpyd2d(tmp_p, rtn_p, sizeof(float));
                        }
                    }
                    return rtn;
                }
            }
#endif
        }
        if (NDArray_SHAPE(src)[NDArray_NDIM(dst) - 1] == NDArray_SHAPE(dst)[NDArray_NDIM(dst) - 2]) {
            if (NDArray_DEVICE(dst) == NDARRAY_DEVICE_CPU) {
                for (i = 0; i < NDArray_SHAPE(dst)[NDArray_NDIM(dst) - 2]; i++) {
                    memcpy(rtn_p,
                           NDArray_FDATA(src), sizeof(float) * NDArray_SHAPE(dst)[NDArray_NDIM(dst) - 1]);
                    rtn_p = rtn_p + (sizeof(float) * NDArray_SHAPE(src)[NDArray_NDIM(dst) - 1]);
                }
            }
#ifdef HAVE_CUBLAS
            if (NDArray_DEVICE(dst) == NDARRAY_DEVICE_GPU) {
                for (i = 0; i < NDArray_SHAPE(dst)[NDArray_NDIM(dst) - 2]; i++) {
                    vmemcpyd2d(NDArray_DATA(src), rtn_p,
                                        sizeof(float) * NDArray_SHAPE(dst)[NDArray_NDIM(dst) - 1]);
                    rtn_p = (char *) (NDArray_FDATA(rtn) +
                                      (i * NDArray_STRIDES(rtn)[NDArray_NDIM(rtn) - 2] / NDArray_ELSIZE(rtn)) + j);
                }
            }
#endif
        }
    }
    return rtn;
}

/**
 * Get the scalar value of a NDArray
 *
 * @param a
 * @return
 */
float
NDArray_GetFloatScalar(NDArray *a) {
    if (NDArray_DEVICE(a) == NDARRAY_DEVICE_CPU) {
        return NDArray_FDATA(a)[0];
    }
#ifdef HAVE_CUBLAS
    return NDArray_VFLOAT(NDArray_DATA(a));
#endif
}

/**
 * Overwrite the values of one NDArray with the values
 * of another.
 *
 * @param target
 * @param values
 */
int
NDArray_Overwrite(NDArray *target, NDArray *values) {

    if (NDArray_NDIM(values) == 0) {
        NDArray_Fill(target, NDArray_GetFloatScalar(values));
        return 1;
    }

    if (NDArray_DEVICE(target) != NDArray_DEVICE(values)) {
        zend_throw_error(NULL, "Incompatible devices during NDArray overwrite.");
        return 0;
    }

    if (NDArray_ShapeCompare(target, values) == 0) {
        zend_throw_error(NULL, "Incompatible shapes during NDArray overwrite.");
        return 0;
    }

    if (NDArray_DEVICE(target) == NDARRAY_DEVICE_CPU) {
        memcpy(NDArray_FDATA(target), NDArray_FDATA(values),
               sizeof(NDArray_ELSIZE(values)) * NDArray_NUMELEMENTS(values));
        return 1;
    }
#ifdef HAVE_CUBLAS
    if (NDArray_DEVICE(target) == NDARRAY_DEVICE_GPU) {
        vmemcpyd2d(values->data, target->data,
                            sizeof(NDArray_ELSIZE(values)) * NDArray_NUMELEMENTS(values));
        return 1;
    }
#endif
    return 0;
}

/**
 * @param a
 */
void
NDArray_Save(NDArray *a, char * filename, int length)
{
    // Open file for writing
    FILE *file = fopen(filename, "wb");
    if (file == NULL) {
        fprintf(stderr, "Error opening file %s\n", filename);
        return;
    }
    fwrite(a, sizeof(NDArray), 1, file);
    fwrite(a->strides, sizeof(int), NDArray_NDIM(a), file);
    fwrite(a->dimensions, sizeof(int), NDArray_NDIM(a), file);
    fwrite(a->iterator, sizeof(NDArrayIterator), 1, file);
    fwrite(a->descriptor, sizeof(NDArrayDescriptor), 1, file);
    fwrite(&a->descriptor->numElements, sizeof(long), 1, file);
    fwrite(a->data, a->descriptor->elsize, a->descriptor->numElements, file);
    fclose(file);
}

/**
 * @param a
 */
NDArray*
NDArray_Load(char * filename)
{
    NDArray *out = emalloc(sizeof(NDArray));
    // Open file for writing
    FILE *file = fopen(filename, "r");
    if (file == NULL) {
        zend_throw_error(NULL, "Error opening file %s\n", filename);
        return NULL;
    }
    fread(out, sizeof(NDArray), 1, file);

    out->dimensions = emalloc(sizeof(int) * NDArray_NDIM(out));
    out->strides = emalloc(sizeof(int) * NDArray_NDIM(out));
    out->descriptor = emalloc(sizeof(NDArrayDescriptor));

    fread(out->strides, sizeof(int), NDArray_NDIM(out), file);
    fread(out->dimensions, sizeof(int), NDArray_NDIM(out), file);
    fread(out->iterator, sizeof(NDArrayIterator), 1, file);
    fread(out->descriptor, sizeof(NDArrayDescriptor), 1, file);
    // Write size of data buffer
    fread(&out->descriptor->numElements, sizeof(long), 1, file);
    // Write data buffer
    fread(out->data, out->descriptor->elsize, out->descriptor->numElements, file);
    // Close the file
    fclose(file);
    return out;
}

NDArray*
NDArray_AssignRawScalar(NDArray *dst, NDArray *src)
{
    if (NDArray_DEVICE(dst) == NDARRAY_DEVICE_CPU) {
        NDArray_FDATA(dst)[0] = NDArray_FDATA(src)[0];
    }
#ifdef HAVE_CUBLAS
    if (NDArray_DEVICE(dst) == NDARRAY_DEVICE_GPU) {
        vmemcpyd2d(NDArray_DATA(src), NDArray_DATA(dst), sizeof(float));
    }
#endif
    return dst;
}

int
NDArray_CompareLists(int const *l1, int const *l2, int n)
{
    int i;

    for (i = 0; i < n; i++) {
        if (l1[i] != l2[i]) {
            return 0;
        }
    }
    return 1;
}

int
raw_array_assign_array(int ndim, int const *shape,
                       NDArrayDescriptor *dst_dtype, char *dst_data, int const *dst_strides,
                       NDArrayDescriptor *src_dtype, char *src_data, int const *src_strides,
                       int device)
{
    int idim;
    int shape_it[NDARRAY_MAX_DIMS];
    int dst_strides_it[NDARRAY_MAX_DIMS];
    int src_strides_it[NDARRAY_MAX_DIMS];
    int coord[NDARRAY_MAX_DIMS];

    if (NDArray_PrepareTwoRawArrayIter(
            ndim, shape,
            dst_data, dst_strides,
            src_data, src_strides,
            &ndim, shape_it,
            &dst_data, dst_strides_it,
            &src_data, src_strides_it) < 0) {
        return -1;
    }

    /*
     * Overlap check for the 1D case. Higher dimensional arrays and
     * opposite strides cause a temporary copy before getting here.
     */
    if (ndim == 1 && src_data < dst_data &&
        src_data + shape_it[0] * src_strides_it[0] > dst_data) {
        src_data += (shape_it[0] - 1) * src_strides_it[0];
        dst_data += (shape_it[0] - 1) * dst_strides_it[0];
        src_strides_it[0] = -src_strides_it[0];
        dst_strides_it[0] = -dst_strides_it[0];
    }

    int strides[2] = {src_strides_it[0], dst_strides_it[0]};
    int size;
    NDARRAY_RAW_ITER_START(idim, ndim, coord, shape_it) {
        size = shape_it[0];
        for (int i = 0; i < size; i++) {
            if (device == NDARRAY_DEVICE_CPU) {
                memcpy(dst_data + (int) (i * dst_strides_it[0]), src_data + (int) (i * src_strides_it[0]),
                       sizeof(float));
            }
            if (device == NDARRAY_DEVICE_GPU) {
#ifdef HAVE_CUBLAS
                vmemcpyd2d(src_data + (int) (i * src_strides_it[0]), dst_data + (int) (i * dst_strides_it[0]), sizeof(float));
#endif
            }
        }
    } NDARRAY_RAW_ITER_TWO_NEXT(idim, ndim, coord, shape_it,
                            dst_data, dst_strides_it,
                            src_data, src_strides_it);
    return 0;
fail:
    return -1;
}

int
NDArray_AssignArray(NDArray *dst, NDArray *src)
{
    int copied_src = 0;

    int src_strides[NDARRAY_MAX_DIMS];

    if (NDArray_NDIM(src) == 0) {
        NDArray_AssignRawScalar(dst, NDArray_DATA(src));
        return 0;
    }

    if (NDArray_DATA(src) == NDArray_DATA(dst) &&
        NDArray_DESCRIPTOR(src) == NDArray_DESCRIPTOR(dst) &&
        NDArray_NDIM(src) == NDArray_NDIM(dst) &&
        NDArray_CompareLists(NDArray_SHAPE(src),
                             NDArray_SHAPE(dst),
                             NDArray_NDIM(src)) &&
        NDArray_CompareLists(NDArray_STRIDES(src),
                             NDArray_STRIDES(dst),
                             NDArray_NDIM(src))) {
        return 0;
    }

    if (NDArray_NDIM(src) > NDArray_NDIM(dst)) {
        int ndim_tmp = NDArray_NDIM(src);
        int *src_shape_tmp = NDArray_SHAPE(src);
        int *src_strides_tmp = NDArray_STRIDES(src);

        while (ndim_tmp > NDArray_NDIM(dst) && src_shape_tmp[0] == 1) {
            --ndim_tmp;
            ++src_shape_tmp;
            ++src_strides_tmp;
        }

        if (broadcast_strides(NDArray_NDIM(dst), NDArray_SHAPE(dst),
                              ndim_tmp, src_shape_tmp,
                              src_strides_tmp, "input array",
                              src_strides) < 0) {
            goto fail;
        }
    }
    else {
        if (broadcast_strides(NDArray_NDIM(dst), NDArray_SHAPE(dst),
                              NDArray_NDIM(src), NDArray_SHAPE(src),
                              NDArray_STRIDES(src), "input array",
                              src_strides) < 0) {
            goto fail;
        }
    }

    if (raw_array_assign_array(NDArray_NDIM(dst), NDArray_SHAPE(dst),
                               NDArray_DESCRIPTOR(dst), NDArray_DATA(dst), NDArray_STRIDES(dst),
                               NDArray_DESCRIPTOR(src), NDArray_DATA(src), src_strides, NDArray_DEVICE(src)) < 0) {
        goto fail;
    }

    if (copied_src) {
        NDArray_FREE(src);
    }
    return 0;

fail:
    if (copied_src) {
        NDArray_FREE(src);
    }
    return -1;
}

static inline int
s_intp_abs(int x)
{
return (x < 0) ? -x : x;
}

void
NDArray_CreateMultiSortedStridePerm(int narrays, NDArray **arrays,
                                    int ndim, int *out_strideperm)
{
    int i0, i1, ipos, ax_j0, ax_j1, iarrays;

    for (i0 = 0; i0 < ndim; ++i0) {
        out_strideperm[i0] = i0;
    }

    for (i0 = 1; i0 < ndim; ++i0) {
        ipos = i0;
        ax_j0 = out_strideperm[i0];
        for (i1 = i0 - 1; i1 >= 0; --i1) {
            int ambig = 1, shouldswap = 0;

            ax_j1 = out_strideperm[i1];

            for (iarrays = 0; iarrays < narrays; ++iarrays) {
                if (NDArray_SHAPE(arrays[iarrays])[ax_j0] != 1 &&
                    NDArray_SHAPE(arrays[iarrays])[ax_j1] != 1) {
                    if (s_intp_abs(NDArray_STRIDES(arrays[iarrays])[ax_j0]) <=
                        s_intp_abs(NDArray_STRIDES(arrays[iarrays])[ax_j1])) {
                        /*
                         * Set swap even if it's not ambiguous already,
                         * because in the case of conflicts between
                         * different operands, C-order wins.
                         */
                        shouldswap = 0;
                    }
                    else {
                        /* Only set swap if it's still ambiguous */
                        if (ambig) {
                            shouldswap = 1;
                        }
                    }

                    /*
                     * A comparison has been done, so it's
                     * no longer ambiguous
                     */
                    ambig = 0;
                }
            }
            /*
             * If the comparison was unambiguous, either shift
             * 'ipos' to 'i1' or stop looking for an insertion point
             */
            if (!ambig) {
                if (shouldswap) {
                    ipos = i1;
                }
                else {
                    break;
                }
            }
        }

        /* Insert out_strideperm[i0] into the right place */
        if (ipos != i0) {
            for (i1 = i0; i1 > ipos; --i1) {
                out_strideperm[i1] = out_strideperm[i1-1];
            }
            out_strideperm[ipos] = ax_j0;
        }
    }
}

/*
 * Sorts items so stride is descending, because C-order
 * is the default in the face of ambiguity.
 */
static int _nd_stride_sort_item_comparator(const void *a, const void *b)
{
    int astride = ((const ndarray_stride_sort_item *)a)->stride,
        bstride = ((const ndarray_stride_sort_item *)b)->stride;

    /* Sort the absolute value of the strides */
    if (astride < 0) {
        astride = -astride;
    }
    if (bstride < 0) {
        bstride = -bstride;
    }

    if (astride == bstride) {
        /*
         * Make the qsort stable by next comparing the perm order.
         * (Note that two perm entries will never be equal)
         */
        int aperm = ((const ndarray_stride_sort_item *)a)->perm,
                bperm = ((const ndarray_stride_sort_item *)b)->perm;
        return (aperm < bperm) ? -1 : 1;
    }
    if (astride > bstride) {
        return -1;
    }
    return 1;
}

void
NDArray_CreateSortedStridePerm(int ndim, int const *strides,
                               ndarray_stride_sort_item *out_strideperm)
{
    int i;

    /* Set up the strideperm values */
    for (i = 0; i < ndim; ++i) {
        out_strideperm[i].perm = i;
        out_strideperm[i].stride = strides[i];
    }

    /* Sort them */
    qsort(out_strideperm, ndim, sizeof(raw_array_assign_array),
          &_nd_stride_sort_item_comparator);
}