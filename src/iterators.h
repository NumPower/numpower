#ifndef PHPSCI_NDARRAY_ITERATORS_H
#define PHPSCI_NDARRAY_ITERATORS_H

#include "ndarray.h"

typedef struct NDArrayAxisIterator {
    int current_index;
    NDArray* array;
    int axis;
} NDArrayAxisIterator;

NDArray* NDArrayIterator_GET(NDArray* array);
void NDArrayIterator_INIT(NDArray* array);
void NDArrayIterator_REWIND(NDArray* array);
int NDArrayIterator_ISDONE(NDArray* array);
void NDArrayIterator_NEXT(NDArray* array);
void NDArrayIterator_FREE(NDArray* array);

NDArray* NDArrayIteratorPHP_GET(NDArray* array);
void NDArrayIteratorPHP_REWIND(NDArray* array);
int NDArrayIteratorPHP_ISDONE(NDArray* array);
void NDArrayIteratorPHP_NEXT(NDArray* array);


NDArray* NDArrayAxisIterator_GET(NDArrayAxisIterator *it);
void NDArrayAxisIterator_NEXT(NDArrayAxisIterator *it);
void NDArrayAxisIterator_FREE(NDArrayAxisIterator *it);
void NDArrayAxisIterator_REWIND(NDArrayAxisIterator *it);
int NDArrayAxisIterator_ISDONE(NDArrayAxisIterator *it);
NDArrayAxisIterator* NDArrayAxisIterator_INIT(NDArray *array, int axis);
#endif //PHPSCI_NDARRAY_ITERATORS_H
