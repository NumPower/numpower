#include "statistics.h"
#include <php.h>
#include "string.h"
#include "../ndarray.h"
#include "../initializers.h"

// Comparison function for sorting
int compare_quantile(const void* a, const void* b) {
    float fa = *((const float*)a);
    float fb = *((const float*)b);
    return (fa > fb) - (fa < fb);
}

float calculate_quantile(float* vector, int num_elements, int stride, float quantile) {
    // Copy vector elements to a separate array
    float* temp = emalloc(num_elements * sizeof(float));
    if (temp == NULL) {
        // Handle memory allocation error
        fprintf(stderr, "Memory allocation failed.\n");
        exit(1);
    }

    // Populate the temporary array using the strided vector
    for (int i = 0; i < num_elements; i++) {
        temp[i] = *((float*)((char*)vector + i * stride));
    }

    // Sort the array in ascending order
    qsort(temp, num_elements, sizeof(float), compare_quantile);

    // Calculate the index of the desired quantile
    float index = (num_elements - 1) * quantile;

    // Calculate the lower and upper indices for interpolation
    int lower_index = (int)index;
    int upper_index = lower_index + 1;

    // Calculate the weight for interpolation
    float weight = index - lower_index;

    // Perform linear interpolation between the two adjacent values
    float lower_value = temp[lower_index];
    float upper_value = temp[upper_index];
    float quantile_value = (1 - weight) * lower_value + weight * upper_value;

    // Free the temporary array
    efree(temp);

    return quantile_value;
}

/**
 * NDArray::quantile
 *
 * @param target
 * @param q
 * @return
 */
NDArray*
NDArray_Quantile(NDArray *target, NDArray *q) {
    if (NDArray_NDIM(q) > 0) {
        zend_throw_error(NULL, "Q must be a scalar");
        return NULL;
    }

    if (NDArray_FDATA(q)[0] < 0 || NDArray_FDATA(q)[0] > 1) {
        zend_throw_error(NULL, "Q must be between 0 and 1");
        return NULL;
    }

    float result = calculate_quantile(NDArray_FDATA(target), NDArray_NUMELEMENTS(target), sizeof(float), NDArray_FDATA(q)[0]);
    return NDArray_CreateFromFloatScalar(result);
}
