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

int
NDArray_PrepareTwoRawArrayIter(int ndim, int const *shape,
                               char *dataA, int const *stridesA,
                               char *dataB, int const *stridesB,
                               int *out_ndim, int *out_shape,
                               char **out_dataA, int *out_stridesA,
                               char **out_dataB, int *out_stridesB)
{
    ndarray_stride_sort_item strideperm[NDARRAY_MAX_DIMS];
    int i, j;

    /* Special case 0 and 1 dimensions */
    if (ndim == 0) {
        *out_ndim = 1;
        *out_dataA = dataA;
        *out_dataB = dataB;
        out_shape[0] = 1;
        out_stridesA[0] = 0;
        out_stridesB[0] = 0;
        return 0;
    }
    else if (ndim == 1) {
        int stride_entryA = stridesA[0], stride_entryB = stridesB[0];
        int shape_entry = shape[0];
        *out_ndim = 1;
        out_shape[0] = shape[0];
        /* Always make a positive stride for the first operand */
        if (stride_entryA >= 0) {
            *out_dataA = dataA;
            *out_dataB = dataB;
            out_stridesA[0] = stride_entryA;
            out_stridesB[0] = stride_entryB;
        }
        else {
            *out_dataA = dataA + stride_entryA * (shape_entry - 1);
            *out_dataB = dataB + stride_entryB * (shape_entry - 1);
            out_stridesA[0] = -stride_entryA;
            out_stridesB[0] = -stride_entryB;
        }
        return 0;
    }

    /* Sort the axes based on the destination strides */
    NDArray_CreateSortedStridePerm(ndim, stridesA, strideperm);
    for (i = 0; i < ndim; ++i) {
        int iperm = strideperm[ndim - i - 1].perm;
        out_shape[i] = shape[iperm];
        out_stridesA[i] = stridesA[iperm];
        out_stridesB[i] = stridesB[iperm];
    }

    /* Reverse any negative strides of operand A */
    for (i = 0; i < ndim; ++i) {
        int stride_entryA = out_stridesA[i];
        int stride_entryB = out_stridesB[i];
        int shape_entry = out_shape[i];

        if (stride_entryA < 0) {
            dataA += stride_entryA * (shape_entry - 1);
            dataB += stride_entryB * (shape_entry - 1);
            out_stridesA[i] = -stride_entryA;
            out_stridesB[i] = -stride_entryB;
        }
        /* Detect 0-size arrays here */
        if (shape_entry == 0) {
            *out_ndim = 1;
            *out_dataA = dataA;
            *out_dataB = dataB;
            out_shape[0] = 0;
            out_stridesA[0] = 0;
            out_stridesB[0] = 0;
            return 0;
        }
    }

    /* Coalesce any dimensions where possible */
    i = 0;
    for (j = 1; j < ndim; ++j) {
        if (out_shape[i] == 1) {
            /* Drop axis i */
            out_shape[i] = out_shape[j];
            out_stridesA[i] = out_stridesA[j];
            out_stridesB[i] = out_stridesB[j];
        }
        else if (out_shape[j] == 1) {
            /* Drop axis j */
        }
        else if (out_stridesA[i] * out_shape[i] == out_stridesA[j] &&
                 out_stridesB[i] * out_shape[i] == out_stridesB[j]) {
            /* Coalesce axes i and j */
            out_shape[i] *= out_shape[j];
        }
        else {
            /* Can't coalesce, go to next i */
            ++i;
            out_shape[i] = out_shape[j];
            out_stridesA[i] = out_stridesA[j];
            out_stridesB[i] = out_stridesB[j];
        }
    }
    ndim = i+1;

    *out_dataA = dataA;
    *out_dataB = dataB;
    *out_ndim = ndim;
    return 0;
}


