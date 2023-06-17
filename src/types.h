#ifndef PHPSCI_NDARRAY_TYPES_H
#define PHPSCI_NDARRAY_TYPES_H

static const char* NDARRAY_TYPE_DOUBLE64 = "double64";
static const char* NDARRAY_TYPE_FLOAT32 = "float32";

int get_type_size(const char *type);
int is_type(const char *type_a, const char *type_b);

#endif //PHPSCI_NDARRAY_TYPES_H
