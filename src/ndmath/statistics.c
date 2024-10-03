#include "Zend/zend_API.h"
#include "statistics.h"
#include "string.h"
#include "../initializers.h"
#include "arithmetics.h"
#include "../manipulation.h"
#include "linalg.h"

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
    float index = (float)(num_elements - 1) * quantile;

    // Calculate the lower and upper indices for interpolation
    int lower_index = (int)index;
    int upper_index = lower_index + 1;

    // Calculate the weight for interpolation
    float weight =index - (float)lower_index;

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
 * @todo Implement GPU
 * @param target
 * @param q
 * @return
 */
NDArray*
NDArray_Quantile(NDArray *target, NDArray *q) {
    if (NDArray_DEVICE(target) == NDARRAY_DEVICE_GPU || NDArray_DEVICE(q) == NDARRAY_DEVICE_GPU) {
        zend_throw_error(NULL, "Quantile not available for GPU device.");
        return NULL;
    }

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

/**
 * NDArray::std
 *
 * @todo Implement
 * @return
 */
NDArray *
NDArray_Std(NDArray *a) {
    if (NDArray_DEVICE(a) == NDARRAY_DEVICE_GPU) {
        zend_throw_error(NULL, "NDArray::std not available for GPU.");
        return NULL;
    }
    if (NDArray_DEVICE(a) == NDARRAY_DEVICE_CPU) {
        float mean = NDArray_Sum_Float(a) / NDArray_NUMELEMENTS(a);
        float sum = 0.0f;

        for (int i = 0; i < NDArray_NUMELEMENTS(a); i++) {
            sum += powf(NDArray_FDATA(a)[i] - mean, 2);
        }

        return NDArray_CreateFromFloatScalar(sqrtf(sum / (float)(NDArray_NUMELEMENTS(a))));
    }
}

/**
 * NDArray::variance
 *
 * @param a
 * @return
 */
NDArray*
NDArray_Variance(NDArray *a) {
    NDArray *mean = NDArray_CreateFromFloatScalar(NDArray_Sum_Float(a) / NDArray_NUMELEMENTS(a));
    NDArray *subtracted = NDArray_Subtract_Float(a, mean);
    NDArray_FREE(mean);
    NDArray *abs = NDArray_Abs(subtracted);
    NDArray_FREE(subtracted);
    NDArray *two = NDArray_CreateFromFloatScalar(2.0f);
    NDArray *pow = NDArray_Pow_Float(abs, two);
    NDArray_FREE(abs);
    NDArray_FREE(two);
    NDArray *x = NDArray_CreateFromFloatScalar(NDArray_Sum_Float(pow) / NDArray_NUMELEMENTS(pow));
    NDArray_FREE(pow);
    return x;
}

/**
 * NDArray::average
 *
 * @param a
 * @param weights
 * @return
 */
NDArray*
NDArray_Average(NDArray *a, NDArray *weights) {
    NDArray *rtn = NULL;

    if (weights == NULL) {
        return NDArray_CreateFromFloatScalar(NDArray_Sum_Float(a) / NDArray_NUMELEMENTS(a));
    }

    if (weights != NULL) {
        if (NDArray_DEVICE(a) != NDArray_DEVICE(weights)) {
            zend_throw_error(NULL, "All NDArrays used in a operation must be on the same device.");
            return NULL;
        }
        NDArray *m_weights = NDArray_Multiply_Float(a, weights);
        float s_weights = NDArray_Sum_Float(weights);
        float s_aweights = NDArray_Sum_Float(m_weights);
        rtn = NDArray_CreateFromFloatScalar(s_aweights / s_weights);
        NDArray_FREE(m_weights);
    }
    return rtn;
}

/**
 * NDArray::cov
 *
 * @param a
 * @return
 */
NDArray *NDArray_cov(NDArray *a, bool rowvar)
{
    if (!rowvar) {
        a = NDArray_Transpose(a, NULL);
    }
    
    if (a == NULL || NDArray_NUMELEMENTS(a) == 0)
    {
        zend_throw_error(NULL, "Input cannot be null or empty.");
        return NULL;
    }
    if (NDArray_NDIM(a) != 2 || NDArray_SHAPE(a)[1] == 1)
    {
        zend_throw_error(NULL, "Input must be a 2D NDArray.");
        return NULL;
    }

    int cols = NDArray_SHAPE(a)[0];
    int rows = NDArray_SHAPE(a)[1];

    int *indices_shape = emalloc(sizeof(int) * 2);
    indices_shape[0] = 2;
    indices_shape[1] = 1;

    NDArray** indices_axis = emalloc(sizeof(NDArray*) * 2);
    indices_axis[0] =  NDArray_Zeros(indices_shape, 1, NDArray_TYPE(a), NDArray_DEVICE(a));
    indices_axis[1] =  NDArray_Zeros(indices_shape, 1, NDArray_TYPE(a), NDArray_DEVICE(a));

    NDArray_FDATA(indices_axis[1])[0] = 0;
    NDArray_FDATA(indices_axis[1])[1] = rows;

    NDArray **centered_vectors = emalloc(sizeof(NDArray *) * cols);
    for (int i = 0; i < cols; i++)
    {
        NDArray_FDATA(indices_axis[0])[0] = i;
        NDArray_FDATA(indices_axis[0])[1] = i + 1;
        NDArray *col_vector = NDArray_Slice(a, indices_axis, 2);
        NDArray *centered = NDArray_Subtract_Float(col_vector, NDArray_CreateFromFloatScalar(NDArray_Sum_Float(col_vector) / NDArray_NUMELEMENTS(col_vector)));
        NDArray_FREE(col_vector);
        centered_vectors[i] = centered;
    }
    efree(indices_shape);
    efree(indices_axis[0]);
    efree(indices_axis[1]);
    efree(indices_axis);
    NDArray *centered_a = NDArray_Reshape(NDArray_ConcatenateFlat(centered_vectors, cols), NDArray_SHAPE(a), NDArray_NDIM(a));
    for (int i = 0; i < cols; i++)
    {
        NDArray_FREE(centered_vectors[i]);
    }
    efree(centered_vectors);
    NDArray *multiplied = NDArray_Dot(centered_a, NDArray_Transpose(centered_a, NULL));
    NDArray_FREE(centered_a);
    NDArray *rtn = NDArray_Divide_Float(multiplied, NDArray_CreateFromFloatScalar((float)rows - 1));
    NDArray_FREE(multiplied);
    return rtn;
}