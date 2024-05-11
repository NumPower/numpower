#include <string.h>
#include <php.h>
#include "Zend/zend_alloc.h"
#include "iterators.h"
#include "ndarray.h"
#include "initializers.h"

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
    int* output_shape;

    if (output_ndim >= 1) {
         output_shape = emalloc(sizeof(int) * output_ndim);
    } else {
        output_shape = emalloc(sizeof(int));
    }
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

NDArrayIter*
NDArray_NewElementWiseIter(NDArray *target) {
    NDArrayIter *it;
    int i, nd;
    NDArray *ao = target;

    it = emalloc(sizeof(NDArrayIter));
    if (it == NULL) {
        return NULL;
    }

    nd = NDArray_NDIM(ao);
    it->contiguous = 1;
    if (NDArray_CHKFLAGS(target, NDARRAY_ARRAY_F_CONTIGUOUS)) {
        it->contiguous = 0;
    }
    it->ao = ao;
    it->size = NDArray_NUMELEMENTS(ao);
    it->nd_m1 = nd - 1;
    it->factors[nd-1] = 1;
    for (i = 0; i < nd; i++) {
        it->dims_m1[i] = NDArray_SHAPE(it->ao)[i] - 1;
        it->strides[i] = NDArray_STRIDES(it->ao)[i];
        it->backstrides[i] = it->strides[i] * it->dims_m1[i];
        if (i > 0) {
            it->factors[nd-i-1] = it->factors[nd-i] * it->ao->dimensions[nd-i];
        }
    }
    NDArray_ITER_RESET(it);
    return it;
}

