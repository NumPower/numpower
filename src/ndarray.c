#include <stdio.h>
#include "ndarray.h"
#include "debug.h"
#include "iterators.h"
#include "ndmath/arithmetics.h"
#include "initializers.h"
#include "types.h"
#include "logic.h"
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
            memcpy(rtn->data, slice->data, rtn->descriptor->numElements * sizeof(float));
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

void
_single_reduce_axis(int axis, NDArray* target, NDArray* rtn, float (*operation)(NDArray*)) {
    int i = 0;
    NDArray *slice;
    NDArrayAxisIterator *iterator = NDArrayAxisIterator_INIT(target, axis);
    NDArrayAxisIterator_REWIND(iterator);

    while(!NDArrayAxisIterator_ISDONE(iterator)) {
        slice = NDArrayAxisIterator_GET(iterator);
        NDArray_FDATA(rtn)[i] = operation(slice);
        NDArrayAxisIterator_NEXT(iterator);
        i++;
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
NDArray*
single_reduce(NDArray* array, int* axis, float (*operation)(NDArray*)) {
    char* exception_buffer[256];
    int null_axis = 0;

    if (axis == NULL) {
        null_axis = 1;
        axis = emalloc(sizeof(int));
        *axis = 0;
    }


    if (axis != NULL) {
        if (*axis >= NDArray_NDIM(array)) {
            sprintf((char *) exception_buffer, "axis %d is out of bounds for array of dimension %d", *axis,
                    NDArray_NDIM(array));
            zend_throw_error(NULL, "%s", (const char *) exception_buffer);
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
    NDArray* rtn = NDArray_Zeros(out_shape, out_ndim, NDARRAY_TYPE_FLOAT32);
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
            sprintf((char *) exception_buffer, "axis %d is out of bounds for array of dimension %d", *axis,
                    NDArray_NDIM(array));
            zend_throw_error(NULL, "%s", (const char *) exception_buffer);
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
    NDArray* rtn = NDArray_Zeros(out_shape, out_ndim, NDARRAY_TYPE_FLOAT32);

    //if (reduced_buffer == NULL) {
    //    fprintf(stderr, "Memory allocation failed.\n");
    //    return;
    //}
    _reduce(0, 0, axis, array, rtn, operation);

    if (null_axis == 1) {
        efree(axis);
    }
    return rtn;
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
    NDArray *rtn = NDArray_Zeros(rtn_shape, NDArray_NDIM(a), NDARRAY_TYPE_FLOAT32);

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

#ifdef HAVE_AVX2
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
    rtn = NDArray_Zeros(new_shape, NDArray_NDIM(array), NDARRAY_TYPE_FLOAT32);

    for (i = 0; i < NDArray_NUMELEMENTS(array); i++) {
        NDArray_FDATA(rtn)[i] = op(NDArray_FDATA(array)[i]);
    }
    return rtn;
}

/**
 * @param array
 */
NDArray*
NDArray_Map2F(NDArray *array, ElementWiseFloatOperation2F op, float val1, float val2) {
    NDArray *rtn;
    int i;
    int *new_shape = emalloc(sizeof(int) * NDArray_NDIM(array));
    memcpy(new_shape, NDArray_SHAPE(array), sizeof(int) * NDArray_NDIM(array));
    rtn = NDArray_Zeros(new_shape, NDArray_NDIM(array), NDARRAY_TYPE_FLOAT32);

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
    float* array = NDArray_FDATA(target);
    int length = NDArray_NUMELEMENTS(target);
    float min = 0.f;
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
 * Return maximum value of NDArray
 *
 * @param target
 * @return
 */
float
NDArray_Max(NDArray *target) {
    float max = 0.f;
    float* array = NDArray_FDATA(target);
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

/**
 * @param data
 * @param strides
 * @param dimensions
 * @param ndim
 * @return
 */
zval
convertToStridedArrayToPHPArray(float* data, int* strides, int* dimensions, int ndim, int elsize) {
    zval phpArray;
    int i;

    // Create a new PHP array
    //phpArray = (zval*)emalloc(sizeof(zval));
    array_init_size(&phpArray, ndim);

    for (i = 0; i < dimensions[0]; i++) {
        // If it's not the innermost dimension, recursively convert the sub-array
        if (ndim > 1) {
            int j;
            zval subArray;

            // Calculate the pointer and strides for the sub-array
            float* subData = data + (i * (strides[0]/elsize));
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
    phpArray = convertToStridedArrayToPHPArray(NDArray_FDATA(target), NDArray_STRIDES(target),
                                             NDArray_SHAPE(target), NDArray_NDIM(target), NDArray_ELSIZE(target));
    return phpArray;
}

/**
 * @param nda
 * @return
 */
int*
NDArray_ToIntVector(NDArray *nda) {
    double *tmp_val = emalloc(sizeof(float));
    int *vector = emalloc(sizeof(int) * NDArray_NUMELEMENTS(nda));
    for (int i = 0; i < NDArray_NUMELEMENTS(nda); i++){
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
NDArray*
NDArray_ToGPU(NDArray *target)
{
#ifdef HAVE_CUBLAS
    float *tmp_gpu;
    int *new_shape;
    int n_ndim = NDArray_NDIM(target);

    if (NDArray_DEVICE(target) == NDARRAY_DEVICE_GPU) {
        return NDArray_Copy(target, NDARRAY_DEVICE_GPU);
    }

    new_shape = emalloc(sizeof(int) * NDArray_NDIM(target));
    memcpy(new_shape, NDArray_SHAPE(target), sizeof(int) * NDArray_NDIM(target));

    NDArray *rtn = NDArray_Zeros(new_shape, n_ndim, NDARRAY_TYPE_FLOAT32);
    rtn->device = NDARRAY_DEVICE_GPU;

    cudaMalloc((void **) &tmp_gpu, NDArray_NUMELEMENTS(target) * sizeof(float));
    cudaMemcpy(tmp_gpu, NDArray_FDATA(target), NDArray_NUMELEMENTS(target) * sizeof(float), cudaMemcpyHostToDevice);
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        printf("Error synchronizing: %s\n", cudaGetErrorString(err));
        // Handle error...
    }
    efree(rtn->data);
    rtn->data = (char*)tmp_gpu;
    return rtn;
#else
    // @todo this must be a copy
    zend_throw_error(NULL, "Unable to detect a compatible device.");
    return NULL;
#endif
}

/**
 * Transfer NDArray to CPU and return a copy
 *
 * @param target
 */
NDArray*
NDArray_ToCPU(NDArray *target)
{
    int *new_shape;
    int n_ndim = NDArray_NDIM(target);

    if (NDArray_DEVICE(target) == NDARRAY_DEVICE_CPU) {
        return NDArray_Copy(target, NDARRAY_DEVICE_CPU);
    }

    new_shape = emalloc(sizeof(int) * NDArray_NDIM(target));
    memcpy(new_shape, NDArray_SHAPE(target), sizeof(int) * NDArray_NDIM(target));

    NDArray *rtn = NDArray_Zeros(new_shape, n_ndim, NDARRAY_TYPE_FLOAT32);
    rtn->device = NDARRAY_DEVICE_CPU;
#ifdef HAVE_CUBLAS
    cudaMemcpy(rtn->data, NDArray_FDATA(target), NDArray_NUMELEMENTS(target) * sizeof(float ), cudaMemcpyDeviceToHost);
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
NDArray_ShapeCompare(NDArray *a, NDArray *b)
{
    if (NDArray_NDIM(a) != NDArray_NDIM(b)) {
        return 0;
    }

    for(int i; i < NDArray_NDIM(a); i++) {
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
NDArray_IsBroadcastable(const NDArray* array1, const NDArray* array2) {
    // If either array has 0 dimensions, they're not broadcastable
    if (array1->ndim == 0 || array2->ndim == 0) {
        return 0;
    }

    // Start from trailing dimensions and move forward
    int i = array1->ndim - 1;
    int j = array2->ndim - 1;

    while (i >= 0 && j >= 0) {
        // If dimensions are not equal and neither is 1, arrays are not broadcastable
        if (array1->dimensions[i] != array2->dimensions[j] && array1->dimensions[i] != 1 && array2->dimensions[j] != 1) {
            return 0;
        }
        i--;
        j--;
    }

    // If we've passed all checks, arrays are broadcastable
    return 1;
}

/**
 * Broadcast NDArray *a to NDArray *b shape and strides
 *
 * @param a
 * @param b
 * @return
 */
NDArray*
NDArray_Broadcast(NDArray *a, NDArray *b) {
    if (!NDArray_IsBroadcastable(a, b)) {
        zend_throw_error(NULL, "Can´t broadcast array.");
        return NULL;
    }


}