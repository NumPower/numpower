#ifndef PHPSCI_NDARRAY_NDARRAY_H
#define PHPSCI_NDARRAY_NDARRAY_H

#ifdef __cplusplus
extern "C" {
#endif

#include "stddef.h"
#include <Zend/zend_types.h>

#define NDARRAY_ARRAY_C_CONTIGUOUS    0x0001
#define NDARRAY_ARRAY_F_CONTIGUOUS    0x0002

#define NDARRAY_UNLIKELY(x) (x)
#define NDArray_DATA(a) ((void *)((a)->data))
#define NDArray_DDATA(a) ((double *)((a)->data))
#define NDArray_FDATA(a) ((float *)((a)->data))
#define NDArray_NDIM(a) ((int)((a)->ndim))
#define NDArray_FLAGS(a) ((int)((a)->flags))
#define NDArray_SHAPE(a) ((int *)((a)->dimensions))
#define NDArray_STRIDES(a) ((int *)((a)->strides))
#define NDArray_TYPE(a) ((const char *)((a)->descriptor->type))
#define NDArray_UUID(a) ((int)((a)->uuid))
#define NDArray_NUMELEMENTS(a) ((int)((a)->descriptor->numElements))
#define NDArray_ELSIZE(a) ((int)((a)->descriptor->elsize))
#define NDArray_DEVICE(a) ((int)((a)->device))

#define NDArray_ADDREF(a) ((a)->refcount++)
#define NDArray_DELREF(a) ((a)->refcount--)

#define NDARRAY_DEVICE_CPU 0
#define NDARRAY_DEVICE_GPU 1

/**
 * NDArray Dims
 **/
typedef struct NDArray_Dims {
    int * ptr;
    int len;
} NDArray_Dims;

typedef struct NDArrayIterator {
    int current_index;
} NDArrayIterator;

/**
 * NDArray Descriptor
 */
typedef struct NDArrayDescriptor {
    const char* type;          // d = double
    int elsize;         // Datatype size
    int numElements;    // Number of elements
} NDArrayDescriptor;

/**
 * NDArray
 */
typedef struct NDArray {
    int uuid;            // Buffer UUID
    int* strides;       // Strides vector (number of bytes)
    int* dimensions;    // Dimensions size vector (Shape)
    int ndim;            // Number of Dimensions
    char* data;         // Data Buffer (contiguous strided)
    struct NDArray* base;      // Used when sharing memory from other NDArray (slices, etc)
    int flags;           // Describes NDArray memory approach (Memory related flags)
    NDArrayDescriptor* descriptor;    // NDArray data descriptor
    NDArrayIterator* iterator;
    int refcount;
    int device; // NDArray Device   0 = CPU     1 = GPU
} NDArray;

static inline int
check_and_adjust_axis_msg(int *axis, int ndim)
{
    if (axis == NULL) {
        return 0;
    }

    /* Check that index is valid, taking into account negative indices */
    if (NDARRAY_UNLIKELY((*axis < -ndim) || (*axis >= ndim))) {
        //zend_throw_error(NULL, "Axis is out of bounds for array dimension");
        return -1;
    }

    /* adjust negative indices */
    if (*axis < 0) {
        *axis += ndim;
    }
    return 0;
}

static inline int
check_and_adjust_axis(int *axis, int ndim)
{
    return check_and_adjust_axis_msg(axis, ndim);
}

/*
 * Enables the specified array flags.
 */
static void
NDArray_ENABLEFLAGS(NDArray * arr, int flags)
{
    (arr)->flags |= flags;
}

/*
 * Clears the specified array flags. Does no checking,
 * assumes you know what you're doing.
 */
static void
NDArray_CLEARFLAGS(NDArray *arr, int flags)
{
    (arr)->flags &= ~flags;
}


void NDArray_FREE(NDArray *array);

char *NDArray_Print(NDArray *array, int do_return);

NDArray *reduce(NDArray *array, int *axis, NDArray *(*operation)(NDArray *, NDArray *));

NDArray *single_reduce(NDArray *array, int *axis, float (*operation)(NDArray *));

NDArray *NDArray_Compare(NDArray *a, NDArray *b);

void NDArray_UpdateFlags(NDArray *array, int flagmask);

float NDArray_Min(NDArray *target);

float NDArray_Max(NDArray *target);

zval NDArray_ToPHPArray(NDArray *target);

NDArray *NDArray_Reshape(NDArray *target, int *new_shape, int ndim);

int *NDArray_ToIntVector(NDArray *nda);

NDArray *NDArray_ToGPU(NDArray *target);

NDArray *NDArray_ToCPU(NDArray *target);

int NDArray_ShapeCompare(NDArray *a, NDArray *b);

NDArray *NDArray_Broadcast(NDArray *a, NDArray *b);

int NDArray_IsBroadcastable(const NDArray *arr1, const NDArray *arr2);

#ifdef __cplusplus
}
#endif

typedef float (*ElementWiseDoubleOperation)(float);
NDArray* NDArray_Map(NDArray *array, ElementWiseDoubleOperation op);


#endif //PHPSCI_NDARRAY_NDARRAY_H
