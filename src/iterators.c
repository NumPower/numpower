#include <string.h>
#include <php.h>
#include "Zend/zend_alloc.h"
#include "Zend/zend_API.h"
#include "iterators.h"
#include "ndarray.h"
#include "initializers.h"
#include "debug.h"

/**
 * @param iterator
 */
int
NDArrayIterator_ISDONE(NDArray* array) {
    if (array->iterator->current_index >= NDArray_SHAPE(array)[0]) {
        return 1;
    }
    return 0;
}

/**
 * @param iterator
 */
void
NDArrayIterator_NEXT(NDArray* array) {
    array->iterator->current_index = array->iterator->current_index + 1;
}

/**
 * @param iterator
 */
void
NDArrayIterator_REWIND(NDArray* array) {
    array->iterator->current_index = 0;
}

/**
 * @param iterator
 */
int
NDArrayIteratorPHP_ISDONE(NDArray* array) {
    if (array->php_iterator->current_index >= NDArray_SHAPE(array)[0]) {
        return 1;
    }
    return 0;
}

/**
 * @param iterator
 */
void
NDArrayIteratorPHP_NEXT(NDArray* array) {
    array->php_iterator->current_index = array->php_iterator->current_index + 1;
}

/**
 * @param iterator
 */
void
NDArrayIteratorPHP_REWIND(NDArray* array) {
    array->php_iterator->current_index = 0;
}

/**
 * @param iterator
 */
NDArray*
NDArrayIteratorPHP_GET(NDArray* array) {
    NDArray_ADDREF(array);
    int output_ndim = array->ndim - 1;
    int* output_shape = emalloc(sizeof(int) * output_ndim);
    memcpy(output_shape, NDArray_SHAPE(array) + 1, sizeof(int) * output_ndim);
    NDArray* rtn = Create_NDArray(output_shape, output_ndim, NDArray_TYPE(array), NDArray_DEVICE(array));
    rtn->device = NDArray_DEVICE(array);
    rtn->data = array->data + (array->php_iterator->current_index * NDArray_STRIDES(array)[0]);
    rtn->base = array;
    return rtn;
}

/**
 * @param array
 */
void
NDArrayIterator_INIT(NDArray* array) {
    NDArrayIterator* iterator = (NDArrayIterator*)emalloc(sizeof(NDArrayIterator));
    NDArrayIterator* php_iterator = (NDArrayIterator*)emalloc(sizeof(NDArrayIterator));
    iterator->current_index = 0;
    php_iterator->current_index = 0;
    array->iterator = iterator;
    array->php_iterator = php_iterator;
}

/**
 * @param iterator
 */
NDArray*
NDArrayIterator_GET(NDArray* array) {
    NDArray_ADDREF(array);
    int output_ndim = array->ndim - 1;
    int* output_shape = emalloc(sizeof(int) * output_ndim);
    memcpy(output_shape, NDArray_SHAPE(array) + 1, sizeof(int) * output_ndim);
    NDArray* rtn = Create_NDArray(output_shape, output_ndim, NDArray_TYPE(array), NDArray_DEVICE(array));
    rtn->device = NDArray_DEVICE(array);
    rtn->data = array->data + (array->iterator->current_index * NDArray_STRIDES(array)[0]);
    rtn->base = array;
    return rtn;
}

/**
 * @param array
 */
void
NDArrayIterator_FREE(NDArray* array) {
    if (array->iterator != NULL) {
        efree(array->iterator);
        array->iterator = NULL;
    }
    if (array->php_iterator != NULL) {
        efree(array->php_iterator);
        array->php_iterator = NULL;
    }
}

/**
 * Initialize Axis Iterator
 *
 * @param array
 * @return
 */
NDArrayAxisIterator*
NDArrayAxisIterator_INIT(NDArray *array, int axis) {
    NDArrayAxisIterator *it;
    it = emalloc(sizeof(NDArrayAxisIterator));
    it->array = array;
    it->axis = axis;
    it->current_index = 0;
    return it;
}

int
NDArrayAxisIterator_ISDONE(NDArrayAxisIterator *it) {
    if (it->current_index < NDArray_SHAPE(it->array)[it->axis]) {
        return 0;
    }
    return 1;
}

void
NDArrayAxisIterator_REWIND(NDArrayAxisIterator *it) {
    it->current_index = 0;
}

void
NDArrayAxisIterator_NEXT(NDArrayAxisIterator *it) {
    it->current_index = it->current_index + 1;
}

NDArray*
NDArrayAxisIterator_GET(NDArrayAxisIterator *it) {
    int new_ndim = NDArray_NDIM(it->array) - 1;
    int *new_shape = emalloc(sizeof(int) * (NDArray_NDIM(it->array) - 1));
    int *new_strides = emalloc(sizeof(int) * (NDArray_NDIM(it->array) - 1));
    if ((NDArray_NDIM(it->array) - 1) > 0) {
        for(int i = 0; i < (NDArray_NDIM(it->array) - 1); i++) {
            new_shape[i] = NDArray_SHAPE(it->array)[i + 1];
            new_strides[i] = NDArray_STRIDES(it->array)[(NDArray_NDIM(it->array) - 1) - i];
            new_strides[0] = NDArray_STRIDES(it->array)[it->axis];
        }
    }
    return NDArray_FromNDArray(it->array, it->current_index * NDArray_STRIDES(it->array)[it->axis + 1], new_shape, new_strides,
                               &new_ndim);
}

void
NDArrayAxisIterator_FREE(NDArrayAxisIterator *it) {
    efree(it);
}