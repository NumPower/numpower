#include "types.h"
#include "string.h"


/**
 * Get size of a specific NDArray type
 *
 * @param type
 * @return
 */
int get_type_size(const char *type) {
    if (!strcmp(type, NDARRAY_TYPE_DOUBLE64)) {
        return sizeof(double);
    }
    if (!strcmp(type, NDARRAY_TYPE_FLOAT32)) {
        return sizeof(float);
    }
    return 0;
}

/**
 * @param type_a
 * @param type_b
 * @return
 */
int is_type(const char *type_a, const char *type_b) {
    if (!strcmp(type_a, type_b)) {
        return 1;
    }
    return 0;
}