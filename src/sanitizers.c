#include <Zend/zend.h>

/**
 * @brief Validates and converts an axis argument (integer or array) to normalized form.
 *
 * Processes a zval argument representing array axes, which can be either:
 * - A single integer (normalized to 0 <= axis < ndim)
 * - An array of integers (each normalized similarly)
 * 
 * @author  Henrique Borba
 * @author  Aleksei Nechaev <omfg.rus@gmail.com>
 *
 * @param[in]  arg      Input zval (must be IS_LONG or IS_ARRAY of IS_LONG)
 * @param[in]  name     Argument name for error messages
 * @param[in]  ndim     Number of dimensions in target array
 * @param[out] outsize  Output parameter for result array size 
 * @return              Array of normalized axes (allocated via emalloc), NULL on failure
 *
 * @note    Caller must efree() the returned array
 * @warning Throws PHP errors on invalid input (wrong type, out-of-bound values)
 */
int* zval_parameter_to_normalized_axis_argument(zval *arg, const char *name, int ndim, int *outsize) {
    int *result = NULL;
    
    // Initialize output size
    *outsize = 0;

    /* Case 1: Single integer axis */
    if (Z_TYPE_P(arg) == IS_LONG) {
        int axis = Z_LVAL_P(arg);
        
        // Validate axis bounds [-ndim, ndim)
        if (axis < -ndim || axis >= ndim) {
            zend_throw_error(NULL, "Axis %d is out of bounds for array with %d dimensions in argument `%s`.", 
                           axis, ndim, name);
            return NULL;
        }
        
        // Normalize negative axis
        if (axis < 0) {
            axis += ndim;
        }

        result = emalloc(sizeof(int));
        
        result[0] = axis;
        *outsize = 1;

        return result;
    }

    /* Case 2: Array of axes */
    if (Z_TYPE_P(arg) == IS_ARRAY) {
        int count = zend_hash_num_elements(Z_ARRVAL_P(arg));
        result = emalloc(sizeof(int) * count);

        int i = 0;
        zval *val;
        
        // Iterate through array elements
        ZEND_HASH_FOREACH_VAL(Z_ARRVAL_P(arg), val) {
            // Validate element type
            if (Z_TYPE_P(val) != IS_LONG) {
                efree(result);
                zend_throw_error(NULL, "All elements of argument `%s` must be integers.", name);

                return NULL;
            }

            int axis = Z_LVAL_P(val);
            
            // Validate bounds for each axis
            if (axis < -ndim || axis >= ndim) {
                efree(result);
                zend_throw_error(NULL, 
                    "Axis %d is out of bounds for array with %d dimensions in argument `%s`.",
                    axis, ndim, name);
                return NULL;
            }
            
            // Normalize negative indices
            if (axis < 0) {
                axis += ndim;
            }
            
            result[i++] = axis;
        } ZEND_HASH_FOREACH_END();

        *outsize = count;
        return result;
    }

    /* Invalid type case */
    zend_throw_error(NULL, "Argument `%s` must be either an integer or an array of integers.", name);
    return NULL;
}
