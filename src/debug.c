#include <php.h>
#include "Zend/zend_alloc.h"
#include "Zend/zend_API.h"
#include "debug.h"
#include "../config.h"
#include "ndarray.h"

#ifdef HAVE_CUBLAS
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <nvml.h>
#endif


/**
 * Dump NDArray
 */
void
NDArray_Dump(NDArray* array)
{
    int i;
    printf("\n=================================================");
    printf("\nNDArray.uuid\t\t\t%d", array->uuid);
    printf("\nNDArray.dims\t\t\t[");
    for(i = 0; i < array->ndim; i ++) {
        printf(" %d", array->dimensions[i]);
    }
    printf(" ]\n");
    printf("NDArray.strides\t\t\t[");
    for(i = 0; i < array->ndim; i ++) {
        printf(" %d", array->strides[i]);
    }
    printf(" ]\n");
    printf("NDArray.ndim\t\t\t%d\n", array->ndim);
    if (NDArray_DEVICE(array) == NDARRAY_DEVICE_GPU) {
        printf("NDArray.device\t\t\t%s\n", "GPU");
    } else {
        printf("NDArray.device\t\t\t%s\n", "CPU");
    }
    printf("NDArray.refcount\t\t%d\n", array->refcount);
    printf("NDArray.descriptor.elsize\t%d\n", array->descriptor->elsize);
    printf("NDArray.descriptor.numElements\t%d\n", array->descriptor->numElements);
    printf("NDArray.descriptor.type\t\t%s\n", array->descriptor->type);
    printf("NDArray.iterator.current_index\t\t%d", array->iterator->current_index);
    printf("\n=================================================\n");
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
 * @param buffer
 * @param ndims
 * @param shape
 * @param strides
 * @param cur_dim
 * @param index
 * @return
 */
char*
print_array_float(float* buffer, int ndims, int* shape, int* strides, int cur_dim, int* index, int num_elements) {
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
            sprintf(str + strlen(str), "%g", buffer[offset / sizeof(float)]);

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

            char* child_str = print_array_float(buffer, ndims, shape, strides, cur_dim + 1, index, num_elements);

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
 * Print matrix to
 *
 * @param buffer
 * @param ndims
 * @param shape
 * @param strides
 */
char*
print_matrix_float(float* buffer, int ndims, int* shape, int* strides, int num_elements, int device) {
    float *tmp_buffer;
    int *index = emalloc(ndims * sizeof(int));
    if (device == NDARRAY_DEVICE_GPU) {
#ifdef HAVE_CUBLAS
        tmp_buffer = emalloc(num_elements * sizeof(float));
        cudaMemcpy(tmp_buffer, buffer, num_elements * sizeof(float), cudaMemcpyDeviceToHost);
#endif
    } else {
        tmp_buffer = buffer;
    }
    char* rtn = print_array_float(tmp_buffer, ndims, shape, strides, 0, index, num_elements);
    efree(index);
#ifdef HAVE_CUBLAS
    if (device == NDARRAY_DEVICE_GPU) {
        efree(tmp_buffer);
    }
#endif
    return rtn;
}

void
NDArray_DumpDevices() {
#ifdef HAVE_CUBLAS
    int deviceCount;
    cudaError_t err = cudaGetDeviceCount(&deviceCount);

    if (err != cudaSuccess) {
        printf("Failed to retrieve device count: %s\n", cudaGetErrorString(err));
        return;
    }

    printf("\n==============================================================================\n");
    printf("Number of CUDA devices: %d\n", deviceCount);
    for (int i = 0; i < deviceCount; ++i) {
        struct cudaDeviceProp deviceProp;
        err = cudaGetDeviceProperties(&deviceProp, i);

        if (err != cudaSuccess) {
            printf("Failed to get properties for device %d: %s\n", i, cudaGetErrorString(err));
            return;
        }
        printf("\n---------------------------------------------------------------------------");
        printf("\nDevice %d: %s\n", i, deviceProp.name);
        printf("  Compute capability: %d.%d\n", deviceProp.major, deviceProp.minor);
        printf("  Total global memory: %zu bytes\n", deviceProp.totalGlobalMem);
        printf("  Max threads per block: %d\n", deviceProp.maxThreadsPerBlock);
        printf("  Max threads in X-dimension of block: %d\n", deviceProp.maxThreadsDim[0]);
        printf("  Max threads in Y-dimension of block: %d\n", deviceProp.maxThreadsDim[1]);
        printf("  Max threads in Z-dimension of block: %d\n", deviceProp.maxThreadsDim[2]);
        printf("  Max grid size in X-dimension: %d\n", deviceProp.maxGridSize[0]);
        printf("  Max grid size in Y-dimension: %d\n", deviceProp.maxGridSize[1]);
        printf("  Max grid size in Z-dimension: %d\n", deviceProp.maxGridSize[2]);
        printf("  Max grid size in Z-dimension: %d\n", deviceProp.maxGridSize[2]);
        printf("  Max grid size in Z-dimension: %d\n", deviceProp.maxGridSize[2]);
        printf("  Max grid size in Z-dimension: %d\n", deviceProp.maxGridSize[2]);
        printf("---------------------------------------------------------------------------\n");
    }
    printf("\n==============================================================================\n");
#else
    php_printf("\nNo GPU devices available. CUDA not enabled.\n");
#endif
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