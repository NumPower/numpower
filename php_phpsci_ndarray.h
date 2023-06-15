/* phpsci_ndarray extension for PHP */

#ifndef PHP_NDARRAY_H
# define PHP_NDARRAY_H

#include "config.h"

#ifdef HAVE_CUBLAS
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif


PHPAPI zend_class_entry *phpsci_ce_NDArray;

# define PHP_NDARRAY_VERSION "0.1.0"

# if defined(ZTS) && defined(COMPILE_DL_NDARRAY)
ZEND_TSRMLS_CACHE_EXTERN()
# endif

#endif	/* PHP_NDARRAY_H */
