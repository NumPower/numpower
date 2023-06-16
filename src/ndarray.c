#include <stdio.h>
#include <Zend/zend_alloc.h>
#include "ndarray.h"
#include "debug.h"
#include "iterators.h"
#include "ndmath/arithmetics.h"
#include "initializers.h"
#include "types.h"
#include "logic.h"
#include <php.h>
#include "../config.h"
#include <Zend/zend_API.h>
#include <Zend/zend_types.h>

#ifndef HAVE_AVX2
#include <immintrin.h>
#endif

#ifdef HAVE_CUBLAS
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

void apply_reduce(NDArray* result, NDArray *target, NDArray* (*operation)(NDArray*, NDArray*)) {
    NDArray* temp = operation(result, target);
    memcpy(result->data, temp->data, result->descriptor->numElements * sizeof(double));
    NDArray_FREE(temp);
}

void apply_single_reduce(NDArray* result, NDArray *target, double (*operation)(NDArray*)) {
    double temp = operation(target);
    NDArray_DDATA(result)[0] = temp;
}

void _reduce(int current_axis, int rtn_init, int* axis, NDArray* target, NDArray* rtn, NDArray* (*operation)(NDArray*, NDArray*)) {
    NDArray* slice;
    NDArray* rtn_slice;
    NDArrayIterator_REWIND(target);
    while(!NDArrayIterator_ISDONE(target)) {
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
            memcpy(rtn->data, slice->data, rtn->descriptor->numElements * sizeof(double));
            NDArrayIterator_NEXT(target);
            NDArray_FREE(slice);
            continue;
        }
        apply_reduce(rtn, slice, operation);
        NDArrayIterator_NEXT(target);
        NDArray_FREE(slice);
    }
}

void _single_reduce(int current_axis, int rtn_init, int* axis, NDArray* target, NDArray* rtn, double (*operation)(NDArray*)) {
    NDArray* slice;
    NDArray* rtn_slice;
    NDArrayIterator_REWIND(target);
    while(!NDArrayIterator_ISDONE(target)) {
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
        if (rtn_init == 0) {
            rtn_init = 1;
            memcpy(rtn->data, slice->data, rtn->descriptor->numElements * sizeof(double));
            NDArrayIterator_NEXT(target);
            NDArray_FREE(slice);
            continue;
        }
        apply_single_reduce(rtn, slice, operation);
        NDArrayIterator_NEXT(target);
        NDArray_FREE(slice);
    }
}

void _single_reduce_axis(int axis, NDArray* target, NDArray* rtn, double (*operation)(NDArray*)) {
    int i = 0;
    NDArrayAxisIterator *iterator = NDArrayAxisIterator_INIT(target, axis);
    NDArrayAxisIterator_REWIND(iterator);

    while(!NDArrayAxisIterator_ISDONE(iterator)) {
        NDArray_DDATA(rtn)[i] = operation(NDArrayAxisIterator_GET(iterator));
        NDArrayAxisIterator_NEXT(iterator);
        i++;
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
NDArray*
single_reduce(NDArray* array, int* axis, double (*operation)(NDArray*)) {
    char* exception_buffer[256];
    int null_axis = 0;

    if (axis == NULL) {
        null_axis = 1;
        axis = emalloc(sizeof(int));
        *axis = 0;
    }


    if (axis != NULL) {
        if (*axis >= NDArray_NDIM(array)) {
            sprintf(exception_buffer, "axis %d is out of bounds for array of dimension %d", *axis,
                    NDArray_NDIM(array));
            zend_throw_error(NULL, exception_buffer);
            return NULL;
        }
    }

    // Calculate the size and strides of the reduced output
    int out_dim = 0;
    int out_ndim = 0;

    if (axis != NULL) {
        for (int i = 0; i < NDArray_NDIM(array); i++) {
            if (i != *axis) {
                out_dim++;
            }
        }
    } else {
        out_dim = 0;
    }

    out_ndim = out_dim;
    int* out_shape = emalloc(sizeof(int) * out_ndim);

    if (axis != NULL) {
        int j = 0;
        for (int i = 0; i < NDArray_NDIM(array); i++) {
            if (i != *axis) {
                out_shape[j] = NDArray_SHAPE(array)[i];
                j++;
            }
        }
    } else {
        out_shape[0] = 1;
    }

    // Calculate the size of the reduced buffer
    int reduced_buffer_size = 1;
    for (int i = 0; i < out_dim; i++) {
        reduced_buffer_size *= out_shape[i];
    }

    // Allocate memory for the reduced buffer
    NDArray* rtn = NDArray_Zeros(out_shape, out_ndim);
    //if (reduced_buffer == NULL) {
    //    fprintf(stderr, "Memory allocation failed.\n");
    //    return;
    //}
    _single_reduce_axis(*axis, array, rtn, operation);
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
NDArray*
reduce(NDArray* array, int* axis, NDArray* (*operation)(NDArray*, NDArray*)) {
    char* exception_buffer[256];
    int null_axis = 0;

    if (axis == NULL) {
        null_axis = 1;
        axis = emalloc(sizeof(int));
        *axis = 0;
    }


    if (axis != NULL) {
        if (*axis >= NDArray_NDIM(array)) {
            sprintf(exception_buffer, "axis %d is out of bounds for array of dimension %d", *axis,
                    NDArray_NDIM(array));
            zend_throw_error(NULL, exception_buffer);
            return NULL;
        }
    }

    // Calculate the size and strides of the reduced output
    int out_dim = 0;
    int out_ndim = 0;

    if (axis != NULL) {
        for (int i = 0; i < NDArray_NDIM(array); i++) {
            if (i != *axis) {
                out_dim++;
            }
        }
    } else {
        out_dim = 0;
    }

    out_ndim = out_dim;
    int* out_shape = emalloc(sizeof(int) * out_ndim);

    if (axis != NULL) {
        int j = 0;
        for (int i = 0; i < NDArray_NDIM(array); i++) {
            if (i != *axis) {
                out_shape[j] = NDArray_SHAPE(array)[i];
                j++;
            }
        }
    } else {
        out_shape[0] = 1;
    }

    // Calculate the size of the reduced buffer
    int reduced_buffer_size = 1;
    for (int i = 0; i < out_dim; i++) {
        reduced_buffer_size *= out_shape[i];
    }

    // Allocate memory for the reduced buffer
    NDArray* rtn = NDArray_Zeros(out_shape, out_ndim);
    //if (reduced_buffer == NULL) {
    //    fprintf(stderr, "Memory allocation failed.\n");
    //    return;
    //}
    _reduce(0, 0, axis, array, rtn, operation);
    return rtn;
}


void
test(NDArray* array)
{
    NDArray_All(array);
}

/**
 * Free NDArray
 *
 * @param array
 */
void
NDArray_FREE(NDArray* array) {
    if (array == NULL) {
        return;
    }
    // Decrement the reference count
    if (array->refcount > 0) {
        array->refcount--;
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

        if (array->data != NULL && array->base == NULL) {
            if (NDArray_DEVICE(array) == NDARRAY_DEVICE_CPU) {
                efree(array->data);
            } else {
#ifdef HAVE_CUBLAS
                cudaFree(array->data);
#endif
            }
        }

        if (array->base != NULL) {
            NDArray_FREE(array->base);
        }

        if (array->descriptor != NULL) {
            efree(array->descriptor);
        }

        efree(array);
    }
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
    char *str = print_matrix(NDArray_DDATA(array), NDArray_NDIM(array), NDArray_SHAPE(array),
                             NDArray_STRIDES(array), NDArray_NUMELEMENTS(array), NDArray_DEVICE(array));
    if (do_return == 0) {
        printf("%s", str);
        return NULL;
    }
    return str;
}

/**
 * NDArray Reduce
 *
 * @param array
 * @return
 */
NDArray*
NDArray_Reduce(NDArray *array, int axis, char* function) {

}

/**
 * Compare two NDArrays
 *
 * @param a
 * @param b
 * @return
 */
NDArray*
NDArray_Compare(NDArray *a, NDArray *b) {
    int i;
    int *rtn_shape = emalloc(sizeof(int) * NDArray_NDIM(a));
    memcpy(rtn_shape, NDArray_SHAPE(a), sizeof(int) * NDArray_NDIM(a));
    NDArray *rtn = NDArray_Zeros(rtn_shape, NDArray_NDIM(a));

    // Check if arrays have the same dimension
    if (NDArray_NDIM(a) != NDArray_NDIM(b)) {
        zend_throw_error(NULL, "Can't compare two different shape arrays");
        return NULL;
    }

    // Check if arrays are equal
    for (i = 0; i < NDArray_NDIM(a); i++) {
        if(NDArray_SHAPE(a)[i] != NDArray_SHAPE(b)[i]) {
            zend_throw_error(NULL, "Can't compare two different shape arrays");
            return NULL;
        }
    }

#ifdef PHP_HAVE_AVX2
    NDArrayIterator_REWIND(a);
    NDArrayIterator_REWIND(b);
    while(!NDArrayIterator_ISDONE(a)) {
        NDArrayIterator_NEXT(a);
        NDArrayIterator_NEXT(b);
    }
#else

#endif
    return rtn;
}


/**
 * Check whether the given array is stored contiguously
 **/
static void
_UpdateContiguousFlags(NDArray * array)
{
    int sd;
    int dim;
    int i;
    int is_c_contig = 1;

    sd = NDArray_ELSIZE(array);
    for (i = NDArray_NDIM(array) - 1; i >= 0; --i) {
        dim = NDArray_SHAPE(array)[i];

        if (NDArray_STRIDES(array)[i] != sd) {
            is_c_contig = 0;
            break;
        }
        /* contiguous, if it got this far */
        if (dim == 0) {
            break;
        }
        sd *= dim;
    }
    if (is_c_contig) {
        NDArray_ENABLEFLAGS(array, NDARRAY_ARRAY_C_CONTIGUOUS);
    }
    else {
        NDArray_CLEARFLAGS(array, NDARRAY_ARRAY_C_CONTIGUOUS);
    }
}

/**
 * Update CArray flags
 **/
void
NDArray_UpdateFlags(NDArray *array, int flagmask)
{
    if (flagmask & (NDARRAY_ARRAY_F_CONTIGUOUS | NDARRAY_ARRAY_C_CONTIGUOUS)) {
        _UpdateContiguousFlags(array);
    }
}

/**
 * @param array
 */
NDArray*
NDArray_Map(NDArray *array, ElementWiseDoubleOperation op) {
    NDArray *rtn;
    int i;
    int *new_shape = emalloc(sizeof(int) * NDArray_NDIM(array));
    memcpy(new_shape, NDArray_SHAPE(array), sizeof(int) * NDArray_NDIM(array));
    rtn = NDArray_Zeros(new_shape, NDArray_NDIM(array));

    #pragma omp parallel for private(i)
    for (i = 0; i < NDArray_NUMELEMENTS(array); i++) {
        NDArray_DDATA(rtn)[i] = op(NDArray_DDATA(array)[i]);
    }
    return rtn;
}

/**
 * Return minimum value of NDArray
 *
 * @param target
 * @return
 */
double
NDArray_Min(NDArray *target) {
    double* array = NDArray_DDATA(target);
    int length = NDArray_NUMELEMENTS(target);
    double min = array[0];
    #pragma omp parallel for
    for (int i = 1; i < length; i++) {
        if (array[i] < min) {
            min = array[i];
        }
    }
    return min;
}

/**
 * Return maximum value of NDArray
 *
 * @param target
 * @return
 */
double
NDArray_Max(NDArray *target) {
    double* array = NDArray_DDATA(target);
    int length = NDArray_NUMELEMENTS(target);
    double max = array[0];
    #pragma omp parallel for
    for (int i = 1; i < length; i++) {
        if (array[i] > max) {
            max = array[i];
        }
    }
    return max;
}

/**
 * @param data
 * @param strides
 * @param dimensions
 * @param ndim
 * @return
 */
zval
convertToStridedArrayToPHPArray(double* data, int* strides, int* dimensions, int ndim, int elsize) {
    zval phpArray;
    int i;

    // Create a new PHP array
    //phpArray = (zval*)emalloc(sizeof(zval));
    array_init_size(&phpArray, ndim);

    #pragma omp parallel for
    for (i = 0; i < dimensions[0]; i++) {
        // If it's not the innermost dimension, recursively convert the sub-array
        if (ndim > 1) {
            int j;
            zval subArray;

            // Calculate the pointer and strides for the sub-array
            double* subData = data + (i * (strides[0]/elsize));
            int* subStrides = strides + 1;
            int* subDimensions = dimensions + 1;

            // Convert the sub-array to a PHP array
            subArray = convertToStridedArrayToPHPArray(subData, subStrides, subDimensions, ndim - 1, elsize);

            // Add the sub-array to the main array
            add_index_zval(&phpArray, i, &subArray);
        } else {
            //printf("\nNDIM: %d\n", *strides);
            // Add the scalar values to the main array
            add_index_double(&phpArray, i, *(data + (i * (*strides/elsize))));
        }
    }
    return phpArray;
}

/**
 * Convert a NDArray to PHP Array
 *
 * @return
 */
zval
NDArray_ToPHPArray(NDArray *target) {
    zval phpArray;
    phpArray = convertToStridedArrayToPHPArray(NDArray_DATA(target), NDArray_STRIDES(target),
                                             NDArray_SHAPE(target), NDArray_NDIM(target), NDArray_ELSIZE(target));
    return phpArray;
}

/**
 * Reshape NDArray
 *
 * @param target
 * @param new_shape
 * @return
 */
NDArray*
NDArray_Reshape(NDArray *target, int *new_shape, int ndim)
{
    int i;
    target->ndim = ndim;
    if (NDArray_NDIM(target) < ndim) {
        efree(target->dimensions);
        target->dimensions = emalloc(sizeof(int) * ndim);
        memcpy(target->dimensions, new_shape, sizeof(int) * ndim);
    }

    efree(target->strides);
    target->strides = Generate_Strides(new_shape, ndim, sizeof(double));

    efree(target->dimensions);
    target->dimensions = new_shape;
    NDArray_Dump(target);
    return target;
}

/**
 * @param nda
 * @return
 */
int*
NDArray_ToIntVector(NDArray *nda) {
    double *tmp_val = emalloc(sizeof(double));
    int *vector = emalloc(sizeof(int) * NDArray_NUMELEMENTS(nda));
    for (int i = 0; i < NDArray_NUMELEMENTS(nda); i++){
        if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_GPU) {
#ifdef HAVE_CUBLAS
            cudaMemcpy(tmp_val, &NDArray_DDATA(nda)[i], sizeof(double), cudaMemcpyDeviceToHost);
            vector[i] = (int) *tmp_val;
            continue;
#endif
        }
        vector[i] = (int) NDArray_DDATA(nda)[i];
    }
    efree(tmp_val);
    return vector;
}

/**
 * Transfer NDArray to GPU
 *
 * @param target
 */
NDArray*
NDArray_ToGPU(NDArray *target)
{
    double *tmp_gpu;
    int *new_shape;
    int n_ndim = NDArray_NDIM(target);

    if (NDArray_DEVICE(target) == NDARRAY_DEVICE_GPU) {
        return target;
    }

    new_shape = emalloc(sizeof(int) * NDArray_NDIM(target));
    memcpy(new_shape, NDArray_SHAPE(target), sizeof(int) * NDArray_NDIM(target));

    NDArray *rtn = NDArray_Zeros(new_shape, n_ndim);
    rtn->device = NDARRAY_DEVICE_GPU;
#ifdef HAVE_CUBLAS
    cudaMalloc((void **) &tmp_gpu, NDArray_NUMELEMENTS(target) * sizeof(double));
    cudaMemcpy(tmp_gpu, NDArray_DDATA(target), NDArray_NUMELEMENTS(target) * sizeof(double), cudaMemcpyHostToDevice);
    efree(rtn->data);
    rtn->data = tmp_gpu;
#endif
    return rtn;
}

/**
 * Transfer NDArray to CPU
 *
 * @param target
 */
NDArray*
NDArray_ToCPU(NDArray *target)
{
    int *new_shape;
    int n_ndim = NDArray_NDIM(target);

    if (NDArray_DEVICE(target) == NDARRAY_DEVICE_CPU) {
        return target;
    }

    new_shape = emalloc(sizeof(int) * NDArray_NDIM(target));
    memcpy(new_shape, NDArray_SHAPE(target), sizeof(int) * NDArray_NDIM(target));

    NDArray *rtn = NDArray_Zeros(new_shape, n_ndim);
    rtn->device = NDARRAY_DEVICE_CPU;

    cudaMemcpy(rtn->data, NDArray_DDATA(target), NDArray_NUMELEMENTS(target) * sizeof(double), cudaMemcpyDeviceToHost);
    return rtn;
}