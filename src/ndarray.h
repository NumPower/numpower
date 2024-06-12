#ifndef PHPSCI_NDARRAY_NDARRAY_H
#define PHPSCI_NDARRAY_NDARRAY_H

#ifdef __cplusplus
extern "C" {
#endif

#include "stddef.h"
#include <Zend/zend_types.h>
#include <stdbool.h>

#define NDARRAY_MAX_DIMS 128
#define NDARRAY_ARRAY_C_CONTIGUOUS    0x0001
#define NDARRAY_ARRAY_F_CONTIGUOUS    0x0002

#define NDARRAY_UNLIKELY(x) (x)
#define NDArray_DATA(a) ((void *)((a)->data))
#define NDArray_DESCRIPTOR(a) ((NDArrayDescriptor *)((a)->descriptor))
#define NDArray_DDATA(a) ((double *)((a)->data))
#define NDArray_FDATA(a) ((float *)((a)->data))
#define NDArray_NDIM(a) ((int)((a)->ndim))
#define NDArray_FLAGS(a) ((int)((a)->flags))
#define NDArray_SHAPE(a) ((int *)((a)->dimensions))
#define NDArray_STRIDES(a) ((int *)((a)->strides))
#define NDArray_TYPE(a) ((const char *)((a)->descriptor->type))
#define NDArray_UUID(a) ((int)((a)->uuid))
#define NDArray_NUMELEMENTS(a) ((long)((a)->descriptor->numElements))
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
    long numElements;    // Number of elements
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
    NDArrayIterator* php_iterator;
    int refcount;
    int device; // NDArray Device   0 = CPU     1 = GPU
} NDArray;

/*
 * Enables the specified array flags.
 */
static void
NDArray_ENABLEFLAGS(NDArray * arr, int flags) {
    (arr)->flags |= flags;
}

static inline int
NDArray_CHKFLAGS(const NDArray *arr, int flags)
{
    return ((arr)->flags & flags) == flags;
}

/*
 * Clears the specified array flags. Does no checking,
 * assumes you know what you're doing.
 */
static void
NDArray_CLEARFLAGS(NDArray *arr, int flags) {
    (arr)->flags &= ~flags;
}

void NDArray_FREE(NDArray *array);
char *NDArray_Print(NDArray *array, int do_return);
NDArray *reduce(NDArray *array, int *axis, NDArray *(*operation)(NDArray *, NDArray *));
NDArray *single_reduce(NDArray *array, int *axis, float (*operation)(NDArray *));
float NDArray_Min(NDArray *target);
float NDArray_Max(NDArray *target);
NDArray* NDArray_Maximum(NDArray *a, NDArray *b);
NDArray * NDArray_Minimum(NDArray *a, NDArray *b);
NDArray* NDArray_MaxAxis(NDArray* target, int axis);
zval NDArray_ToPHPArray(NDArray *target);
int *NDArray_ToIntVector(NDArray *nda);
NDArray *NDArray_ToGPU(NDArray *target);
NDArray *NDArray_ToCPU(NDArray *target);
int NDArray_ShapeCompare(NDArray *a, NDArray *b);
NDArray* NDArray_Broadcast(NDArray *a, NDArray *b);
int NDArray_IsBroadcastable(const NDArray *arr1, const NDArray *arr2);
float NDArray_GetFloatScalar(NDArray *a);
void NDArray_FREEDATA(NDArray *target);
int NDArray_Overwrite(NDArray *target, NDArray *values);
NDArray* NDArray_FromGD(zval *a, bool channel_last);
void NDArray_ToGD(NDArray *a, NDArray *n_alpha, zval *output);
void NDArray_Save(NDArray *a, char * filename, int length);
NDArray* NDArray_Load(char * filename);

#ifdef __cplusplus
}
#endif

typedef float (*ElementWiseDoubleOperation)(float);
typedef float (*ElementWiseFloatOperation2F)(float, float, float);
typedef float (*ElementWiseFloatOperation1F)(float, float);
NDArray* NDArray_Map(NDArray *array, ElementWiseDoubleOperation op);
NDArray* NDArray_Map2F(NDArray *array, ElementWiseFloatOperation2F op, float val1, float val2);
NDArray* NDArray_Map1F(NDArray *array, ElementWiseFloatOperation1F op, float val1);
NDArray * NDArray_Map1ND(NDArray *array, ElementWiseFloatOperation1F op, NDArray *val1);

#endif //PHPSCI_NDARRAY_NDARRAY_H
