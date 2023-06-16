#include <php.h>
#include <Zend/zend_alloc.h>
#include "debug.h"
#include "../config.h"
#include "ndarray.h"

#ifdef HAVE_CUBLAS
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif


/**
 * Dump NDArray
 */
void
NDArray_Dump(NDArray* array)
{
    int i;
    php_printf("\n=================================================");
    php_printf("\nCArray.uuid\t\t\t%d", array->uuid);
    php_printf("\nCArray.dims\t\t\t[");
    for(i = 0; i < array->ndim; i ++) {
        php_printf(" %d", array->dimensions[i]);
    }
    php_printf(" ]\n");
    php_printf("CArray.strides\t\t\t[");
    for(i = 0; i < array->ndim; i ++) {
        php_printf(" %d", array->strides[i]);
    }
    php_printf(" ]\n");
    php_printf("CArray.ndim\t\t\t%d\n", array->ndim);
    php_printf("CArray.device\t\t\t%d\n", array->device);
    php_printf("CArray.refcount\t\t\t%d\n", array->refcount);
    php_printf("CArray.descriptor.elsize\t%d\n", array->descriptor->elsize);
    php_printf("CArray.descriptor.numElements\t%d\n", array->descriptor->numElements);
    php_printf("CArray.descriptor.type\t\t%s", array->descriptor->type);
    php_printf("\n=================================================\n");
}

/**
 * @param buffer
 * @param ndims
 * @param shape
 * @param strides
 * @param cur_dim
 * @param index
 * @return
 */
char*
print_array(double* buffer, int ndims, int* shape, int* strides, int cur_dim, int* index, int num_elements) {
    char* str;
    int i, j;
    // Allocate memory for the string
    str = (char*)malloc(100000000 * sizeof(char));
    if (str == NULL) {
        fprintf(stderr, "Error: Failed to allocate memory for string.\n");
        exit(1);
    }

    if (ndims == 0) {
        sprintf(str, "%g\n", buffer[0]);
        return str;
    }

    if (cur_dim == ndims - 1) {
        // Print the opening bracket for this dimension
        sprintf(str, "[");

        // Print the elements of the array
        for (i = 0; i < shape[cur_dim]; i++) {
            // Update the index of this element
            index[cur_dim] = i;

            // Compute the offset of this element in the buffer
            int offset = 0;
            for (int k = 0; k < ndims; k++) {
                offset += index[k] * strides[k];
            }
            // Print the element
            sprintf(str + strlen(str), "%g", buffer[offset / sizeof(double)]);

            // Print a comma if this is not the last element in the dimension
            if (i < shape[cur_dim] - 1) {
                sprintf(str + strlen(str), ", ");
            }

        }

        // Print the closing bracket for this dimension
        sprintf(str + strlen(str), "]");
    } else {
        // Print the opening bracket for this dimension
        sprintf(str, "[");

        // Recursively print each element in the dimension
        for (i = 0; i < shape[cur_dim]; i++) {
            // Update the index of this element
            index[cur_dim] = i;

            char* child_str = print_array(buffer, ndims, shape, strides, cur_dim + 1, index, num_elements);

            // Add the child string to the parent string
            sprintf(str + strlen(str), "%s", child_str);

            // Free the child string
            free(child_str);

            // Print a comma and newline if this is not the last element in the dimension
            if (i < shape[cur_dim] - 1) {
                sprintf(str + strlen(str), ",\n ");
                for (j = 0; j < cur_dim; j++) {
                    sprintf(str + strlen(str), " ");
                }
            }
        }

        // Print the closing bracket for this dimension
        sprintf(str + strlen(str), "]");
    }

    // Add a newline if this is the outermost dimension
    if (cur_dim == 0) {
        sprintf(str + strlen(str), "\n");
    }

    return str;
}

/**
 * Print matrix to
 *
 * @param buffer
 * @param ndims
 * @param shape
 * @param strides
 */
char*
print_matrix(double* buffer, int ndims, int* shape, int* strides, int num_elements, int device) {
    double *tmp_buffer;
    int *index = emalloc(ndims * sizeof(int));
    if (device == NDARRAY_DEVICE_GPU) {
#ifdef HAVE_CUBLAS
        tmp_buffer = emalloc(num_elements * sizeof(double));
        cudaMemcpy(tmp_buffer, buffer, num_elements * sizeof(double), cudaMemcpyDeviceToHost);
#endif
    } else {
        tmp_buffer = buffer;
    }
    char* rtn = print_array(tmp_buffer, ndims, shape, strides, 0, index, num_elements);
    efree(index);
#ifdef HAVE_CUBLAS
    if (device == NDARRAY_DEVICE_GPU) {
        efree(tmp_buffer);
    }
#endif
    return rtn;
}

/**
 * @param a
 */
void
NDArrayIterator_DUMP(NDArray *a) {
    printf("\n====================================\n");
    printf("iterator.current_index:\t\t%d",a->iterator->current_index);
    printf("\n====================================\n");
}