#include <php.h>
#include "Zend/zend_alloc.h"
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
NDArray_Dump(NDArray* array) {
    int i;
    printf("\n=================================================");
    printf("\nNDArray.uuid\t\t\t%d", array->uuid);
    printf("\nNDArray.ndim\t\t\t%d", array->ndim);
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
    if (NDArray_DEVICE(array) == NDARRAY_DEVICE_GPU) {
        printf("NDArray.device\t\t\t(%d) %s\n", NDArray_DEVICE(array), "GPU");
    } else if(NDArray_DEVICE(array) == NDARRAY_DEVICE_CPU) {
        printf("NDArray.device\t\t\t(%d) %s\n", NDArray_DEVICE(array), "CPU");
    } else {
        printf("NDArray.device\t\t\t(%d) %s\n", NDArray_DEVICE(array), "ERROR");
    }
    printf("NDArray.refcount\t\t%d\n", array->refcount);
    printf("NDArray.descriptor.elsize\t%d\n", array->descriptor->elsize);
    printf("NDArray.descriptor.numElements\t%d\n", array->descriptor->numElements);
    printf("NDArray.descriptor.type\t\t%s\n", array->descriptor->type);
    printf("NDArray.iterator.current_index\t%d", array->iterator->current_index);
    printf("\n=================================================\n");
}

char*
expand_str(char* str, unsigned int additional_size) {
    if (str == NULL) {
        return (char*)emalloc(additional_size * sizeof(char));
    }
    //reallocate memory for the string to accommodate extra characters
    char *new_str_pointer = (char*)erealloc(str, (strlen(str) + additional_size) * sizeof(char));
    if (new_str_pointer != str) {
        efree(str);
    }
    return new_str_pointer;
}

int string_size_of_float(float number) {
    char buf[32];
    int size = snprintf(buf, sizeof(buf), "%f", number);
    int actualSize = 0;
    while (actualSize < size && buf[actualSize] != '\0') {
        ++actualSize;
    }

    return actualSize;
}

#pragma clang diagnostic push
#pragma ide diagnostic ignored "misc-no-recursion"
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
print_array_float(float* buffer, int ndims, int* shape, int* strides, int cur_dim, int* index, int num_elements, int* padded) {
    char* str = NULL;
    int i, j, t;
    int reverse_run = 0;

    if (ndims == 0 && buffer != NULL) {
        str = expand_str(str, string_size_of_float(buffer[0]));
        sprintf(str, "%g", buffer[0]);
        return str;
    }

    if (index == NULL) {
        fprintf(stderr, "Error: print_array_float called with NULL index.\n");
        exit(1);
    }

    if (cur_dim == ndims - 1 && buffer != NULL) {
        // Print the opening bracket for this dimension
        str = expand_str(str, strlen("["));
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
            str = expand_str(str, string_size_of_float(buffer[offset / sizeof(float)]));
            sprintf(str + strlen(str), "%g", buffer[offset / sizeof(float)]);

            // Print a comma if this is not the last element in the dimension
            if (i < shape[cur_dim] - 1) {
                str = expand_str(str, strlen(", "));
                sprintf(str + strlen(str), ", ");
            }

            if ((i + 1) % 10 == 0 && i < shape[cur_dim] - 1) {
                str = expand_str(str, 1);
                sprintf(str + strlen(str), "\n");
                for (t = 0; t < ndims; t++) {
                    str = expand_str(str, strlen(" "));
                    sprintf(str + strlen(str), " ");
                }
            }

            if (shape[cur_dim] > 20) {
                if (i > 1 && reverse_run == 0) {
                    i = shape[cur_dim] - 4;
                    reverse_run = 1;
                    str = expand_str(str, strlen("... "));
                    sprintf(str + strlen(str), "... ");
                }
            }
        }

        // Print the closing bracket for this dimension
        sprintf(str + strlen(str), "]");
        if (index[cur_dim-1] < shape[ndims - 2] - 1) {
            str = expand_str(str, strlen("\n "));
            sprintf(str + strlen(str), "\n ");
        }
    } else {
        if (cur_dim != 0) {
            if (cur_dim == index[cur_dim - 1]) {
                for (t = cur_dim; t < ndims; t++) {
                    str = expand_str(str, strlen(" "));
                    sprintf(str + strlen(str), " ");
                }
            }
        }
        // Print the opening bracket for this dimension
        str = expand_str(str, strlen("["));
        sprintf(str, "[");

        // Recursively print each element in the dimension
        for (i = 0; i < shape[cur_dim]; i++) {
            // Update the index of this element
            index[cur_dim] = i;

            char* child_str = print_array_float(buffer, ndims, shape, strides, cur_dim + 1, index, num_elements, padded);

            // Add the child string to the parent string
            str = expand_str(str, strlen(child_str));
            sprintf(str + strlen(str), "%s", child_str);

            // Free the child string
            efree(child_str);

            // Print a comma and newline if this is not the last element in the dimension
            if (i < shape[cur_dim] - 1) {
                for (j = 0; j < cur_dim; j++) {
                    str = expand_str(str, strlen(" "));
                    sprintf(str + strlen(str), " ");
                }
            }

            if (ndims > 1) {
                if (shape[ndims - 1] * shape[ndims - 2] > 500 && shape[cur_dim] > 10) {
                    if(i >= 2 && reverse_run == 0) {
                        i = shape[cur_dim] - 4;
                        reverse_run = 1;
                        str = expand_str(str, strlen("...\n"));
                        sprintf(str + strlen(str), "...\n");
                        if (i < shape[cur_dim] - 1) {
                            for (j = 1; j < ndims; j++) {
                                sprintf(str + strlen(str), " ");
                            }
                        }
                        *padded = 1;
                    }
                }
            }
        }
        // Print the closing bracket for this dimension
        str = expand_str(str, strlen("]"));
        sprintf(str + strlen(str), "]");

        if (cur_dim != 0 && index[cur_dim-1] < shape[cur_dim-1] - 1) {
            str = expand_str(str, strlen("\n"));
            sprintf(str + strlen(str), "\n");
        }
    }

    // Add a newline if this is the outermost dimension
    if (cur_dim == 0) {
        str = expand_str(str, strlen("\n"));
        sprintf(str + strlen(str), "\n");
    }

    return str;
}
#pragma clang diagnostic pop

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
    float *tmp_buffer = NULL;
    int *index = emalloc(ndims * sizeof(int));
    if (device == NDARRAY_DEVICE_GPU) {
#ifdef HAVE_CUBLAS
        tmp_buffer = emalloc(num_elements * sizeof(float));
        cudaMemcpy(tmp_buffer, buffer, num_elements * sizeof(float), cudaMemcpyDeviceToHost);
#endif
    } else {
        tmp_buffer = buffer;
    }
    int padded = 0;
    char* rtn = print_array_float(tmp_buffer, ndims, shape, strides, 0, index, num_elements, &padded);
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