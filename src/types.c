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
    return 0;
}