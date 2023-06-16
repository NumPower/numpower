#include "initializers.h"
#include "ndarray.h"
#include "types.h"
#include "Zend/zend_alloc.h"
#include "../config.h"
#include "Zend/zend_hash.h"
#include "php.h"
#include "iterators.h"
#include <math.h>
#include <time.h>

#ifdef HAVE_CUBLAS
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

/**
 *
 * @param arr
 * @param shape
 * @param ndim
 */
void get_zend_array_shape(zend_array* arr, int* shape, int ndim) {
    int i;

    // Initialize shape array to zeros
    for (i = 0; i < ndim; i++) {
        shape[i] = 0;
    }

    // Traverse the array to get the shape
    zval* val;
    ZEND_HASH_FOREACH_VAL(arr, val) {
        if (Z_TYPE_P(val) == IS_ARRAY) {
            get_zend_array_shape(Z_ARRVAL_P(val), shape + 1, ndim - 1);
            shape[0]++;
        } else {
            shape[0]++;
        }
    } ZEND_HASH_FOREACH_END();
}

/**
 *
 * @param arr
 * @return
 */
int is_packed_zend_array(zend_array *arr) {
    if (arr->nNumUsed == arr->nNumOfElements) {
        return 1;  // the array is packed
    } else {
        return 0;  // the array is sparse
    }
}

/**
 * Get number of dimensions from php_array
 *
 * @param arr
 * @return
 */
int get_num_dims_from_zval(zval *arr) {
    int num_dims = 0;
    zval *val = zend_hash_index_find(Z_ARRVAL_P(arr), 0);
    while (val && Z_TYPE_P(val) == IS_ARRAY) {
        num_dims++;
        val = zend_hash_index_find(Z_ARRVAL_P(val), 0);
    }
    return num_dims+1;
}

/**
 * Create a new NDArray Descriptor
 *
 * @param numElements
 * @param elSize
 * @param type
 * @return
 */
NDArrayDescriptor* Create_Descriptor(int numElements, int elsize, const char* type)
{
    NDArrayDescriptor* ndArrayDescriptor = emalloc(sizeof(NDArrayDescriptor));
    ndArrayDescriptor->elsize = elsize;
    ndArrayDescriptor->numElements = numElements;
    ndArrayDescriptor->type = type;
    return ndArrayDescriptor;
}

/**
 * Generate Strides Vector
 *
 * @return
 */
int* Generate_Strides(int* dimensions, int dimensions_size, int elsize)
{
    if (dimensions_size == 0) {
        return NULL;
    }

    int i;
    int * strides;
    int * target_stride;
    target_stride = safe_emalloc(dimensions_size, sizeof(int), 0);

    for(i = 0; i < dimensions_size; i++) {
        target_stride[i] = 0;
    }

    target_stride[dimensions_size-1] = elsize;

    for(i = dimensions_size-2; i >= 0; i--) {
        target_stride[i] = dimensions[i+1] * target_stride[i+1];
    }

    return target_stride;
}

/**
 *
 * @param buffer
 * @param numElements
 * @param elsize
 */
void
NDArray_CreateBuffer(NDArray* array, int numElements, int elsize)
{
    array->data = emalloc(numElements * elsize);
}

/**
 * @param target_carray
 */
void
NDArray_CopyFromZendArray(NDArray* target, zend_array* target_zval, int * first_index)
{
    double tmp;
    zval * element;
    double * data_double;

    ZEND_HASH_FOREACH_VAL(target_zval, element) {
                ZVAL_DEREF(element);
                if (Z_TYPE_P(element) == IS_ARRAY) {
                    NDArray_CopyFromZendArray(target, Z_ARRVAL_P(element), first_index);
                }
                if (Z_TYPE_P(element) == IS_LONG) {
                    convert_to_long(element);
                    data_double = (double *) NDArray_DATA(target);
                    data_double[*first_index] = (double) zval_get_long(element);
                    *first_index = *first_index + 1;
                }
                if (Z_TYPE_P(element) == IS_DOUBLE) {
                    convert_to_double(element);
                    data_double = (double *) NDArray_DATA(target);
                    data_double[*first_index] = (double) zval_get_double(element);
                    *first_index = *first_index + 1;
                }
                if (Z_TYPE_P(element) == IS_STRING) {

                }
    } ZEND_HASH_FOREACH_END();
}

/**
 * Create NDArray from zend_array
 *
 * @param ht
 * @param ndim
 * @return
 */
NDArray* Create_NDArray_FromZendArray(zend_array* ht, int ndim)
{
    int last_index = 0;
    int* shape = emalloc(ndim * sizeof(int));
    if (!is_packed_zend_array(ht)) {
        return NULL;
    }
    get_zend_array_shape(ht, shape, ndim);
    int total_num_elements = shape[0];

    // Calculate number of elements
    for (int i = 1; i < ndim; i++) {
        total_num_elements = total_num_elements * shape[i];
    }
    NDArray* array = Create_NDArray(shape, ndim, NDARRAY_TYPE_DOUBLE64);
    NDArray_CreateBuffer(array, total_num_elements, get_type_size(NDARRAY_TYPE_DOUBLE64));
    NDArray_CopyFromZendArray(array, ht, &last_index);
    return array;
}

/**
 * Create NDArray from PHP Object (zval)
 *
 * @param php_object
 * @return
 */
NDArray* Create_NDArray_FromZval(zval* php_object)
{
    NDArray* new_array = NULL;
    if (Z_TYPE_P(php_object) == IS_ARRAY) {
        new_array = Create_NDArray_FromZendArray(Z_ARRVAL_P(php_object), get_num_dims_from_zval(php_object));
    }
    return new_array;
}


/**
 * Create basic NDArray from shape and type
 *
 * @param shape
 * @return
 */
NDArray*
Create_NDArray(int* shape, int ndim, const char* type)
{
    char* new_buffer;
    NDArray* rtn;
    NDArrayDescriptor* descriptor;
    int type_size = get_type_size(type);
    int total_size = 0;
    int total_num_elements = shape[0];

    if (ndim == 0) {
        total_num_elements = 1;
    }

    // Calculate number of elements
    for (int i = 1; i < ndim; i++) {
        total_num_elements = total_num_elements * shape[i];
    }

    // Calculate total size in bytes
    total_size = type_size * total_num_elements;

    rtn = emalloc(sizeof(NDArray));
    rtn->descriptor = Create_Descriptor(total_num_elements, type_size, type);
    rtn->flags = 0;
    rtn->ndim = ndim;
    rtn->dimensions = shape;
    rtn->refcount = 1;
    rtn->base = NULL;
    rtn->device = NDARRAY_DEVICE_CPU;
    rtn->strides = Generate_Strides(shape, ndim, type_size);
    NDArrayIterator_INIT(rtn);
    return rtn;
}


/**
 * Create an NDArray View from another NDArray
 *
 * @param target
 * @param buffer_offset
 * @param shape
 * @param strides
 * @param ndim
 * @return
 */
NDArray*
NDArray_FromNDArray(NDArray *target, int buffer_offset, int* shape, int* strides, int* ndim) {
    NDArray* rtn = emalloc(sizeof(NDArray));
    int total_num_elements = 1;
    int out_ndim;

    if (strides == NULL) {
        rtn->strides = emalloc(sizeof(int) * NDArray_NDIM(target));
        memcpy(NDArray_STRIDES(rtn), NDArray_STRIDES(target), sizeof(int) * NDArray_NDIM(target));
    }
    if (shape == NULL) {
        out_ndim = NDArray_NDIM(target);
        rtn->dimensions = emalloc(sizeof(int) * NDArray_NDIM(target));
        memcpy(NDArray_SHAPE(rtn), NDArray_SHAPE(target), sizeof(int) * NDArray_NDIM(target));
    }

    if (shape != NULL) {
        rtn->dimensions = shape;
        rtn->strides = strides;
        out_ndim = *ndim;
    }

    // Calculate number of elements
    for (int i = 0; i < out_ndim; i++) {
        total_num_elements = total_num_elements * NDArray_SHAPE(target)[i];
    }

    rtn->flags = 0;
    rtn->data = target->data + buffer_offset;
    rtn->base = target;
    rtn->ndim = out_ndim;
    rtn->device = NDArray_DEVICE(target);
    rtn->descriptor = Create_Descriptor(total_num_elements, sizeof(double), NDARRAY_TYPE_DOUBLE64);
    NDArray_ADDREF(target);
    return rtn;
}


/**
 * Initialize NDArray with zeros
 *
 * @param shape
 * @param ndim
 * @return
 */
NDArray*
NDArray_Zeros(int *shape, int ndim) {
    NDArray* rtn = Create_NDArray(shape, ndim, NDARRAY_TYPE_DOUBLE64);
    rtn->data = ecalloc(rtn->descriptor->numElements, sizeof(double));
    return rtn;
}

/**
 * Initialize NDArray with zeros
 *
 * @param shape
 * @param ndim
 * @return
 */
NDArray*
NDArray_Ones(int *shape, int ndim) {
    NDArray* rtn = NDArray_Zeros(shape, ndim);
    int i;
    for (i = 0; i < NDArray_NUMELEMENTS(rtn); i++)
    {
        NDArray_DDATA(rtn)[i] = 1.0;
    }
    return rtn;
}

/**
 * Identity Matrix
 *
 * @param size
 * @return
 */
NDArray*
NDArray_Identity(int size) {
    NDArray *rtn;
    int index;
    int *shape;
    double *buffer_ptr;

    if (size < 0) {
        zend_throw_error(NULL, "negative dimensions are not allowed");
        return NULL;
    }

    shape = emalloc(sizeof(int) * 2);
    shape[0] = size;
    shape[1] = size;
    rtn = NDArray_Zeros(shape, 2);

    buffer_ptr = NDArray_DDATA(rtn);
    // Set the diagonal elements to one with the specified stride
    for (int i = 0; i < size; i++) {
        index = ((i * size * sizeof(double)) + (i * sizeof(double))) / sizeof(double);
        buffer_ptr[index] = 1.0;
    }
    return rtn;
}

/**
 * Random samples from a Gaussian distribution.
 *
 * @param size
 * @return
 */
NDArray*
NDArray_Normal(double loc, double scale, int* shape, int ndim) {
    NDArray *rtn;
    rtn = NDArray_Zeros(shape, ndim);

    // Set the seed for random number generation
    srand(time(NULL));

    // Generate random samples from the normal distribution
    for (int i = 0; i < NDArray_NUMELEMENTS(rtn); i++) {
        // Box-Muller transform to generate standard normal samples
        double u1 = (double)rand() / RAND_MAX;
        double u2 = (double)rand() / RAND_MAX;
        double z = sqrt(-2.0 * log(u1)) * cos(2.0 * M_PI * u2);

        // Scale and shift the standard normal sample
        NDArray_DDATA(rtn)[i] = loc + scale * z;
    }

    return rtn;
}

/**
 * Random samples from a Gaussian distribution.
 *
 * @param size
 * @return
 */
NDArray*
NDArray_StandardNormal(int* shape, int ndim) {
    return NDArray_Normal(0.0, 1.0, shape, ndim);
}

/**
 * Random samples from a Poisson distribution.
 *
 * @param size
 * @return
 */
NDArray*
NDArray_Poisson(double lam, int* shape, int ndim) {
    NDArray *rtn;
    rtn = NDArray_Zeros(shape, ndim);

    // Set the seed for random number generation
    srand(time(NULL));

    // Generate random samples from the normal distribution
    for (int i = 0; i < NDArray_NUMELEMENTS(rtn); i++) {
        double L = exp(-lam);
        double p = 1.0;
        int k = 0;

        do {
            k++;
            double u = (double)rand() / RAND_MAX;
            p *= u;
        } while (p > L);
        NDArray_DDATA(rtn)[i] = k - 1.0;
    }

    return rtn;
}

/**
 * Random samples from a Poisson distribution.
 *
 * @param size
 * @return
 */
NDArray*
NDArray_Uniform(double low, double high, int* shape, int ndim) {
    NDArray *rtn;
    rtn = NDArray_Zeros(shape, ndim);

    // Set the seed for random number generation
    srand(time(NULL));

    // Generate random samples from the normal distribution
    for (int i = 0; i < NDArray_NUMELEMENTS(rtn); i++) {
        double u = (double)rand() / RAND_MAX;
        NDArray_DDATA(rtn)[i] = low + u * (high - low);
    }
    return rtn;
}

/**
 * @param a
 * @return
 */
NDArray*
NDArray_Diag(NDArray *a) {
    int i;
    int index;
    NDArray *rtn;
    if (NDArray_NDIM(a) != 1) {
        zend_throw_error(NULL, "Input array must be a vector");
        return NULL;
    }

    int *rtn_shape = emalloc(sizeof(int) * 2);
    rtn_shape[0] = NDArray_NUMELEMENTS(a);
    rtn_shape[1] = NDArray_NUMELEMENTS(a);
    rtn = NDArray_Zeros(rtn_shape, 2);

    for (i = 0; i < NDArray_NUMELEMENTS(a); i++) {
        index = ((i * NDArray_STRIDES(rtn)[0]) + (i * NDArray_STRIDES(rtn)[1])) / NDArray_ELSIZE(rtn);
        NDArray_DDATA(rtn)[index] = NDArray_DDATA(a)[i];
    }

    return rtn;
}

/**
 * Fill values in place
 *
 * @param a
 * @return
 */
NDArray*
NDArray_Fill(NDArray *a, double fill_value) {
    int i;

    for (i = 0; i < NDArray_NUMELEMENTS(a); i++) {
        NDArray_DDATA(a)[i] = fill_value;
    }
    return a;
}

/**
 * @param a
 * @return
 */
NDArray*
NDArray_Full(int *shape, int ndim,  double fill_value) {
    int *new_shape = emalloc(sizeof(int) * ndim);
    memcpy(new_shape, shape, sizeof(int) * ndim);
    NDArray *rtn = NDArray_Zeros(new_shape, ndim);
    return NDArray_Fill(rtn, fill_value);
}

/**
 * Create NDArray from double
 * @return
 */
NDArray*
NDArray_CreateFromDoubleScalar(double scalar) {
    NDArray *rtn = safe_emalloc(1, sizeof(NDArray), 0);

    rtn->ndim = 0;
    rtn->descriptor = emalloc(sizeof(NDArrayDescriptor));
    rtn->descriptor->numElements = 1;
    rtn->descriptor->elsize = sizeof(double);
    rtn->descriptor->type = NDARRAY_TYPE_DOUBLE64;
    rtn->data = (double*)emalloc(sizeof(double));
    rtn->device = NDARRAY_DEVICE_CPU;
    rtn->strides = NULL;
    rtn->dimensions = NULL;
    rtn->iterator = NULL;
    rtn->base = NULL;
    ((double*)rtn->data)[0] = scalar;

    return rtn;
}

/**
 * Create NDArray from long
 * @return
 */
NDArray*
NDArray_CreateFromLongScalar(long scalar) {
    NDArray *rtn = safe_emalloc(1, sizeof(NDArray), 0);

    rtn->ndim = 0;
    rtn->descriptor = emalloc(sizeof(NDArrayDescriptor));
    rtn->descriptor->numElements = 1;
    rtn->descriptor->elsize = sizeof(double);
    rtn->descriptor->type = NDARRAY_TYPE_DOUBLE64;
    rtn->data = (double*)emalloc(sizeof(double));
    rtn->device = NDARRAY_DEVICE_CPU;
    rtn->strides = NULL;
    rtn->dimensions = NULL;
    rtn->iterator = NULL;
    rtn->base = NULL;
    ((double*)rtn->data)[0] = (double)scalar;

    return rtn;
}

