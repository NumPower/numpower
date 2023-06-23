#ifdef HAVE_CONFIG_H
# include "config.h"
#endif

#include <Zend/zend_modules.h>
#include <Zend/zend_interfaces.h>
#include "php.h"
#include "ext/standard/info.h"
#include "numpower_arginfo.h"
#include "src/initializers.h"
#include "Zend/zend_alloc.h"
#include "Zend/zend_API.h"
#include "src/buffer.h"
#include "src/iterators.h"
#include "php_numpower.h"
#include "src/debug.h"
#include "src/ndmath/arithmetics.h"
#include "src/logic.h"
#include "src/manipulation.h"
#include "src/ndmath/double_math.h"
#include "src/ndmath/linalg.h"
#include "config.h"
#include "src/types.h"

#ifdef HAVE_CUBLAS
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "src/ndmath/cuda/cuda_math.h"
#include "src/gpu_alloc.h"
#endif

/* For compatibility with older PHP versions */
#ifndef ZEND_PARSE_PARAMETERS_NONE
#define ZEND_PARSE_PARAMETERS_NONE() \
	ZEND_PARSE_PARAMETERS_START(0, 0) \
	ZEND_PARSE_PARAMETERS_END()
#endif

static zend_object_handlers ndarray_object_handlers;



int
get_object_uuid(zval* obj) {
    return Z_LVAL_P(OBJ_PROP_NUM(Z_OBJ_P(obj), 0));
}

NDArray* ZVAL_TO_NDARRAY(zval* obj) {
    if (Z_TYPE_P(obj) == IS_ARRAY) {
        return Create_NDArray_FromZval(obj);
    }
    if (Z_TYPE_P(obj) == IS_LONG) {
        return NDArray_CreateFromLongScalar(Z_LVAL_P(obj));
    }
    if (Z_TYPE_P(obj) == IS_DOUBLE) {
        return NDArray_CreateFromDoubleScalar(Z_DVAL_P(obj));
    }
    if (Z_TYPE_P(obj) == IS_OBJECT) {
        return buffer_get(get_object_uuid(obj));
    }
    zend_throw_error(NULL, "Invalid object type");
    return NULL;
}

NDArray* ZVALUUID_TO_NDARRAY(zval* obj) {
    if (Z_TYPE_P(obj) == IS_LONG) {
        return buffer_get(Z_LVAL_P(obj));
    }
    if (Z_TYPE_P(obj) == IS_OBJECT) {
        return buffer_get(get_object_uuid(obj));
    }
    return NULL;
}

void CHECK_INPUT_AND_FREE(zval *a, NDArray *nda) {
    if (Z_TYPE_P(a) == IS_ARRAY || Z_TYPE_P(a) == IS_DOUBLE || Z_TYPE_P(a) == IS_LONG) {
        NDArray_FREE(nda);
    }
}

void RETURN_NDARRAY(NDArray* array, zval* return_value) {
    if (array == NULL) {
        RETURN_THROWS();
        return;
    }
    add_to_buffer(array, sizeof(NDArray));
    object_init_ex(return_value, phpsci_ce_NDArray);
    ZVAL_LONG(OBJ_PROP_NUM(Z_OBJ_P(return_value), 0), NDArray_UUID(array));
}

void RETURN_NDARRAY_NOBUFFER(NDArray* array, zval* return_value) {
    if (array == NULL) {
        RETURN_THROWS();
        return;
    }
    object_init_ex(return_value, phpsci_ce_NDArray);
    ZVAL_LONG(OBJ_PROP_NUM(Z_OBJ_P(return_value), 0), NDArray_UUID(array));
}

void RETURN_3NDARRAY(NDArray* array1, NDArray* array2, NDArray* array3, zval* return_value) {
    zval a, b, c;
    if (array1 == NULL) {
        RETURN_THROWS();
        return;
    }
    if (array2 == NULL) {
        RETURN_THROWS();
        return;
    }
    if (array3 == NULL) {
        RETURN_THROWS();
        return;
    }

    add_to_buffer(array1, sizeof(NDArray));
    add_to_buffer(array2, sizeof(NDArray));
    add_to_buffer(array3, sizeof(NDArray));

    object_init_ex(&a, phpsci_ce_NDArray);
    object_init_ex(&b, phpsci_ce_NDArray);
    object_init_ex(&c, phpsci_ce_NDArray);

    ZVAL_LONG(OBJ_PROP_NUM(Z_OBJ_P(&a), 0), NDArray_UUID(array1));
    ZVAL_LONG(OBJ_PROP_NUM(Z_OBJ_P(&b), 0), NDArray_UUID(array2));
    ZVAL_LONG(OBJ_PROP_NUM(Z_OBJ_P(&c), 0), NDArray_UUID(array3));

    array_init_size(return_value, 3);
    add_next_index_zval(return_value, &a);
    add_next_index_zval(return_value, &b);
    add_next_index_zval(return_value, &c);
    RETURN_ZVAL(return_value, 0, 0);
}


void
bypass_printr() {
    zend_string* functionToRename = zend_string_init("print_r", strlen("print_r"), 0);
    zend_function* functionEntry = zend_hash_find_ptr(EG(function_table), functionToRename);

    if (functionEntry != NULL) {
        zend_string* newFunctionName = zend_string_init("print_r_", strlen("print_r_"), 1);
        zend_string_release_ex(functionEntry->common.function_name, 0);
        functionEntry->common.function_name = newFunctionName;
        functionEntry->internal_function.handler = ZEND_FN(print_r_);
        zend_string_addref(functionEntry->common.function_name);
    }
    zend_string_release_ex(functionToRename, 0);
}

/* }}}*/
ZEND_BEGIN_ARG_INFO(arginfo_construct, 1)
    ZEND_ARG_INFO(0, obj_zval)
ZEND_END_ARG_INFO();
PHP_METHOD(NDArray, __construct)
{
    zend_object *obj = Z_OBJ_P(ZEND_THIS);
    zval *obj_zval;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ZVAL(obj_zval)
    ZEND_PARSE_PARAMETERS_END();
    NDArray* array = ZVAL_TO_NDARRAY(obj_zval);
    if (array == NULL) {
        return;
    }
    add_to_buffer(array, sizeof(NDArray));
    ZVAL_LONG(OBJ_PROP_NUM(obj, 0), NDArray_UUID(array));
}

ZEND_BEGIN_ARG_INFO(arginfo_fill, 1)
    ZEND_ARG_INFO(0, value)
ZEND_END_ARG_INFO();
PHP_METHOD(NDArray, fill)
{
    double value;
    NDArray *rtn;
    zval *obj_zval = getThis();
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_DOUBLE(value)
    ZEND_PARSE_PARAMETERS_END();
    NDArray* array = ZVAL_TO_NDARRAY(obj_zval);
    if (array == NULL) {
        return;
    }
    rtn = NDArray_Fill(array, (float)value);
    NDArray_ADDREF(array);
    RETURN_NDARRAY_NOBUFFER(rtn, return_value);
}

ZEND_BEGIN_ARG_INFO(arginfo_toArray, 0)
ZEND_END_ARG_INFO();
PHP_METHOD(NDArray, toArray)
{
    zval rtn;
    zval *obj_zval = getThis();
    ZEND_PARSE_PARAMETERS_START(0, 0)
    ZEND_PARSE_PARAMETERS_END();
    NDArray* array = ZVAL_TO_NDARRAY(obj_zval);
    if (array == NULL) {
        return;
    }
    if (NDArray_DEVICE(array) == NDARRAY_DEVICE_GPU) {
        zend_throw_error(NULL, "NDArray must be on CPU RAM before it can be converted to a PHP array.");
        return;
    }
    rtn = NDArray_ToPHPArray(array);
    NDArray_FREE(array);
    RETURN_ZVAL(&rtn, 0, 0);
}

ZEND_BEGIN_ARG_INFO(arginfo_gpu, 0)
ZEND_END_ARG_INFO();
PHP_METHOD(NDArray, gpu)
{
    NDArray *rtn;
    zval *obj_zval = getThis();
    ZEND_PARSE_PARAMETERS_START(0, 0)
    ZEND_PARSE_PARAMETERS_END();
#ifdef HAVE_CUBLAS
    NDArray* array = ZVAL_TO_NDARRAY(obj_zval);
    if (array == NULL) {
        return;
    }
    rtn = NDArray_ToGPU(array);
    RETURN_NDARRAY(rtn, return_value);
#else
    zend_throw_error(NULL, "No GPU device available or CUDA not enabled");
    RETURN_NULL();
#endif
}

ZEND_BEGIN_ARG_INFO(arginfo_cpu, 0)
ZEND_END_ARG_INFO();
PHP_METHOD(NDArray, cpu)
{
    NDArray *rtn;
    zval *obj_zval = getThis();
    ZEND_PARSE_PARAMETERS_START(0, 0)
    ZEND_PARSE_PARAMETERS_END();
    NDArray* array = ZVAL_TO_NDARRAY(obj_zval);
    if (array == NULL) {
        return;
    }
    rtn = NDArray_ToCPU(array);
    RETURN_NDARRAY(rtn, return_value);
}

ZEND_BEGIN_ARG_INFO(arginfo_dump, 0)
ZEND_END_ARG_INFO();
PHP_METHOD(NDArray, dump)
{
    zval rtn;
    zval *obj_zval = getThis();
    ZEND_PARSE_PARAMETERS_START(0, 0)
    ZEND_PARSE_PARAMETERS_END();
    NDArray* array = ZVAL_TO_NDARRAY(obj_zval);
    if (array == NULL) {
        return;
    }
    NDArray_Dump(array);
}

ZEND_BEGIN_ARG_INFO(arginfo_dump_devices, 0)
ZEND_END_ARG_INFO();
PHP_METHOD(NDArray, dumpDevices)
{
    ZEND_PARSE_PARAMETERS_START(0, 0)
    ZEND_PARSE_PARAMETERS_END();
    NDArray_DumpDevices();
}

ZEND_BEGIN_ARG_INFO(arginfo_reshape, 1)
    ZEND_ARG_INFO(0, shape_zval)
ZEND_END_ARG_INFO();
PHP_METHOD(NDArray, reshape)
{
    int *new_shape;
    zval *shape_zval;
    zval *current = getThis();
    NDArray *rtn;
    ZEND_PARSE_PARAMETERS_START(1, 1)
            Z_PARAM_ZVAL(shape_zval)
    ZEND_PARSE_PARAMETERS_END();
    NDArray* target = ZVAL_TO_NDARRAY(current);
    NDArray* shape = ZVAL_TO_NDARRAY(shape_zval);

    new_shape = NDArray_ToIntVector(shape);

    rtn = NDArray_Reshape(target, new_shape, NDArray_NUMELEMENTS(shape));

    if (rtn == NULL) {
        NDArray_FREE(shape);
        efree(new_shape);
        RETURN_NULL();
    }

    if (Z_TYPE_P(shape_zval) == IS_ARRAY) {
        NDArray_FREE(shape);
    }
    RETURN_NDARRAY(rtn, return_value);
}

PHP_FUNCTION(print_r_)
{
    zval *var;
    bool do_return = 0;
    NDArray *target;
    ZEND_PARSE_PARAMETERS_START(1, 2)
            Z_PARAM_ZVAL(var)
            Z_PARAM_OPTIONAL
            Z_PARAM_BOOL(do_return)
    ZEND_PARSE_PARAMETERS_END();

    if (do_return) {
        if (Z_TYPE_P(var) == IS_OBJECT) {
            zend_class_entry* classEntry = Z_OBJCE_P(var);
            if (!strcmp(classEntry->name->val, "NDArray")) {
                target = buffer_get(get_object_uuid(var));
                RETURN_STRING(NDArray_Print(target, 1));
            }
        }
        RETURN_STR(zend_print_zval_r_to_str(var, 0));
    } else {
        if (Z_TYPE_P(var) == IS_OBJECT) {
            zend_class_entry* classEntry = Z_OBJCE_P(var);
            if (!strcmp(classEntry->name->val, "NDArray")) {
                target = buffer_get(get_object_uuid(var));
                NDArray_Print(target, 0);
                RETURN_TRUE;
            }
        }
        zend_print_zval_r(var, 0);
        RETURN_TRUE;
    }
}

PHP_METHOD(NDArray, __destruct)
{
    zend_object *obj = Z_OBJ_P(ZEND_THIS);
    zval *obj_uuid = OBJ_PROP_NUM(obj, 0);
    buffer_ndarray_free(Z_LVAL_P(obj_uuid));
}

/**
 * NDArray::zeros
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO(arginfo_ndarray_zeros, 1)
    ZEND_ARG_INFO(0, shape_zval)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, zeros)
{
    NDArray *rtn = NULL;
    int *shape;
    zval *shape_zval;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ZVAL(shape_zval)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(shape_zval);
    if (nda == NULL) {
        return;
    }
    shape = emalloc(sizeof(int) * NDArray_NUMELEMENTS(nda));
    for (int i = 0; i < NDArray_NUMELEMENTS(nda); i++){
        shape[i] = (int) NDArray_FDATA(nda)[i];
    }
    rtn = NDArray_Zeros(shape, NDArray_NUMELEMENTS(nda), NDARRAY_TYPE_FLOAT32);
    NDArray_FREE(nda);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NDArray::equal
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO(arginfo_ndarray_equal, 2)
    ZEND_ARG_INFO(0, a)
    ZEND_ARG_INFO(0, b)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, equal)
{
    NDArray *nda, *ndb, *rtn = NULL;
    zval *a, *b;
    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_ZVAL(a)
        Z_PARAM_ZVAL(b)
    ZEND_PARSE_PARAMETERS_END();
    nda = ZVAL_TO_NDARRAY(a);
    ndb = ZVAL_TO_NDARRAY(b);

    if (nda == NULL) return;
    if (ndb == NULL) return;

    rtn = NDArray_Equal(nda, ndb);

    if (rtn == NULL) return;

    CHECK_INPUT_AND_FREE(a, nda);
    CHECK_INPUT_AND_FREE(b, ndb);
    RETURN_NDARRAY(rtn, return_value);
}


/**
 * NDArray::identity
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO(arginfo_ndarray_identity, 1)
    ZEND_ARG_INFO(0, size)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, identity)
{
    NDArray *rtn = NULL;
    int *shape;
    long size;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_LONG(size)
    ZEND_PARSE_PARAMETERS_END();
    rtn = NDArray_Identity((int)size);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NDArray::normal
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_normal, 0, 0, 1)
    ZEND_ARG_INFO(0, size)
    ZEND_ARG_INFO(0, loc)
    ZEND_ARG_INFO(0, scale)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, normal)
{
    NDArray *rtn = NULL;
    int *shape;
    zval* size;
    double loc = 0.0, scale = 1.0;
    ZEND_PARSE_PARAMETERS_START(1, 3)
            Z_PARAM_ZVAL(size)
            Z_PARAM_OPTIONAL
            Z_PARAM_DOUBLE(loc)
            Z_PARAM_DOUBLE(scale)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(size);
    if (nda == NULL) return;
    shape = emalloc(sizeof(int) * NDArray_NUMELEMENTS(nda));
    for (int i = 0; i < NDArray_NUMELEMENTS(nda); i++){
        shape[i] = (int) NDArray_DDATA(nda)[i];
    }
    rtn = NDArray_Normal(loc, scale, shape, NDArray_NUMELEMENTS(nda));
    NDArray_FREE(nda);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NDArray::standard_normal
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_standard_normal, 0, 0, 1)
                ZEND_ARG_INFO(0, size)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, standard_normal)
{
    NDArray *rtn = NULL;
    int *shape;
    zval* size;
    double loc = 0.0, scale = 1.0;
    ZEND_PARSE_PARAMETERS_START(1, 3)
            Z_PARAM_ZVAL(size)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(size);
    if (nda == NULL) return;
    shape = emalloc(sizeof(int) * NDArray_NUMELEMENTS(nda));
    for (int i = 0; i < NDArray_NUMELEMENTS(nda); i++){
        shape[i] = (int) NDArray_DDATA(nda)[i];
    }
    rtn = NDArray_StandardNormal(shape, NDArray_NUMELEMENTS(nda));
    NDArray_FREE(nda);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NDArray::poisson
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_poisson, 0, 0, 1)
    ZEND_ARG_INFO(0, size)
    ZEND_ARG_INFO(0, lam)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, poisson)
{
    NDArray *rtn = NULL;
    int *shape;
    zval* size;
    double lam = 1.0;
    ZEND_PARSE_PARAMETERS_START(1, 3)
            Z_PARAM_ZVAL(size)
            Z_PARAM_OPTIONAL
            Z_PARAM_DOUBLE(lam)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(size);
    if (nda == NULL) {
        return;
    }
    shape = emalloc(sizeof(int) * NDArray_NUMELEMENTS(nda));
    for (int i = 0; i < NDArray_NUMELEMENTS(nda); i++){
        shape[i] = (int) NDArray_DDATA(nda)[i];
    }
    rtn = NDArray_Poisson(lam, shape, NDArray_NUMELEMENTS(nda));
    NDArray_FREE(nda);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NDArray::poisson
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_uniform, 0, 0, 1)
                ZEND_ARG_INFO(0, size)
                ZEND_ARG_INFO(0, low)
                ZEND_ARG_INFO(0, high)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, uniform)
{
    NDArray *rtn = NULL;
    int *shape;
    zval* size;
    double low = 0.0, high = 1.0;
    ZEND_PARSE_PARAMETERS_START(1, 3)
            Z_PARAM_ZVAL(size)
            Z_PARAM_OPTIONAL
            Z_PARAM_DOUBLE(low)
            Z_PARAM_DOUBLE(high)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(size);
    if (nda == NULL) {
        return;
    }
    shape = emalloc(sizeof(int) * NDArray_NUMELEMENTS(nda));
    for (int i = 0; i < NDArray_NUMELEMENTS(nda); i++){
        shape[i] = (int) NDArray_FDATA(nda)[i];
    }
    rtn = NDArray_Uniform(low, high, shape, NDArray_NUMELEMENTS(nda));
    NDArray_FREE(nda);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NDArray::diag
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_diag, 0, 0, 1)
    ZEND_ARG_INFO(0, target)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, diag)
{
    NDArray *rtn = NULL;
    zval* target;
    ZEND_PARSE_PARAMETERS_START(1, 1)
            Z_PARAM_ZVAL(target)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(target);
    if (nda == NULL)  return;
    rtn = NDArray_Diag(nda);
    if (Z_TYPE_P(target) == IS_ARRAY) {
        NDArray_FREE(nda);
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NDArray::diag
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_full, 0, 0, 2)
    ZEND_ARG_INFO(0, shape)
    ZEND_ARG_INFO(0, fill_value)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, full)
{
    NDArray *rtn = NULL;
    zval* shape;
    double fill_value;
    int *new_shape;
    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_ZVAL(shape)
        Z_PARAM_DOUBLE(fill_value)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda_shape = ZVAL_TO_NDARRAY(shape);
    if (nda_shape == NULL) {
        return;
    }
    if (NDArray_NDIM(nda_shape) != 1) {
        zend_throw_error(NULL, "`shape` argument must be a vector");
        NDArray_FREE(nda_shape);
        return;
    }
    new_shape = emalloc(sizeof(int) * NDArray_NUMELEMENTS(nda_shape));
    for (int i = 0; i < NDArray_NUMELEMENTS(nda_shape); i++) {
        new_shape[i] = (int) NDArray_DDATA(nda_shape)[i];
    }
    rtn = NDArray_Full(new_shape, NDArray_NUMELEMENTS(nda_shape), fill_value);
    efree(new_shape);
    if (Z_TYPE_P(shape) == IS_ARRAY) {
        NDArray_FREE(nda_shape);
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NDArray::ones
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO(arginfo_ndarray_ones, 1)
    ZEND_ARG_INFO(0, shape_zval)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, ones)
{
    double *ptr;
    NDArray *rtn = NULL;
    int *shape;
    zval *shape_zval;
    ZEND_PARSE_PARAMETERS_START(1, 1)
            Z_PARAM_ZVAL(shape_zval)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(shape_zval);
    if (nda == NULL) {
        return;
    }
    shape = NDArray_ToIntVector(nda);
    rtn = NDArray_Ones(shape, NDArray_NUMELEMENTS(nda), NDARRAY_TYPE_FLOAT32);
    NDArray_FREE(nda);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NDArray::all
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_all, 0, 0, 1)
    ZEND_ARG_INFO(0, array)
    ZEND_ARG_INFO(0, axis)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, all)
{
    NDArray *rtn = NULL;
    zval *array;
    long axis;
    int axis_i;
    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_ZVAL(array)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(axis)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }
    axis_i = (int)axis;
    if (ZEND_NUM_ARGS() == 1) {
        RETURN_LONG(NDArray_All(nda));
        if (Z_TYPE_P(array) == IS_ARRAY) {
            NDArray_FREE(nda);
        }
    } else {
        if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_GPU) {
            zend_throw_error(NULL, "Axis not supported for GPU operation");
            return;
        }
        zend_throw_error(NULL, "Not implemented");
        return;
        rtn = single_reduce(nda, &axis_i, &NDArray_All);
        RETURN_NDARRAY(rtn, return_value);
    }
}

/**
 * NDArray::transpose
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_transpose, 0, 0, 1)
    ZEND_ARG_INFO(0, array)
    ZEND_ARG_INFO(0, axis)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, transpose)
{
    NDArray *rtn = NULL;
    zval *array;
    long axis;
    int axis_i;
    ZEND_PARSE_PARAMETERS_START(1, 1)
            Z_PARAM_ZVAL(array)
            Z_PARAM_OPTIONAL
            Z_PARAM_LONG(axis)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }
    axis_i = (int)axis;
    if (ZEND_NUM_ARGS() == 1) {
        rtn = NDArray_Transpose(nda, NULL);
        add_to_buffer(rtn, sizeof(NDArray));
        if (Z_TYPE_P(array) == IS_ARRAY) {
            NDArray_FREE(nda);
        }
        RETURN_NDARRAY(rtn, return_value);
    } else {
        if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_GPU) {
            zend_throw_error(NULL, "Axis not supported for GPU operation");
            return;
        }
        zend_throw_error(NULL, "Not implemented");
        return;
    }
}

/**
 * NDArray::copy
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_copy, 0, 0, 1)
                ZEND_ARG_INFO(0, array)
                ZEND_ARG_INFO(0, axis)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, copy)
{
    NDArray *rtn = NULL;
    zval *array;
    long axis;
    int axis_i;
    ZEND_PARSE_PARAMETERS_START(1, 1)
            Z_PARAM_ZVAL(array)
            Z_PARAM_OPTIONAL
            Z_PARAM_LONG(axis)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }
    axis_i = (int)axis;
    if (ZEND_NUM_ARGS() == 1) {
        rtn = NDArray_Transpose(nda, NULL);
        add_to_buffer(rtn, sizeof(NDArray));
        RETURN_NDARRAY(rtn, return_value);
    } else {
        if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_GPU) {
            zend_throw_error(NULL, "Axis not supported for GPU operation");
            return;
        }
        zend_throw_error(NULL, "Not implemented");
        return;
    }
}

/**
 * NDArray::atleast_1d
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_atleast_1d, 0, 0, 1)
    ZEND_ARG_INFO(0, array)
    ZEND_ARG_INFO(0, axis)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, atleast_1d)
{
    NDArray *rtn = NULL;
    zval *array;
    long axis;
    int axis_i;
    ZEND_PARSE_PARAMETERS_START(1, 1)
            Z_PARAM_ZVAL(array)
            Z_PARAM_OPTIONAL
            Z_PARAM_LONG(axis)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }
    axis_i = (int)axis;
    if (ZEND_NUM_ARGS() == 1) {
        rtn = NDArray_Transpose(nda, NULL);
        add_to_buffer(rtn, sizeof(NDArray));
        RETURN_NDARRAY(rtn, return_value);
    } else {
        if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_GPU) {
            zend_throw_error(NULL, "Axis not supported for GPU operation");
            return;
        }
        zend_throw_error(NULL, "Not implemented");
        return;
    }
}

/**
 * NDArray::atleast_2d
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_atleast_2d, 0, 0, 1)
    ZEND_ARG_INFO(0, array)
    ZEND_ARG_INFO(0, axis)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, atleast_2d)
{
    NDArray *rtn = NULL;
    zval *array;
    long axis;
    int axis_i;
    ZEND_PARSE_PARAMETERS_START(1, 1)
            Z_PARAM_ZVAL(array)
            Z_PARAM_OPTIONAL
            Z_PARAM_LONG(axis)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }
    axis_i = (int)axis;
    if (ZEND_NUM_ARGS() == 1) {
        rtn = NDArray_Transpose(nda, NULL);
        add_to_buffer(rtn, sizeof(NDArray));
        RETURN_NDARRAY(rtn, return_value);
    } else {
        if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_GPU) {
            zend_throw_error(NULL, "Axis not supported for GPU operation");
            return;
        }
        zend_throw_error(NULL, "Not implemented");
        return;
    }
}

/**
 * NDArray::atleast_3d
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_atleast_3d, 0, 0, 1)
    ZEND_ARG_INFO(0, array)
    ZEND_ARG_INFO(0, axis)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, atleast_3d)
{
    NDArray *rtn = NULL;
    zval *array;
    long axis;
    int axis_i;
    ZEND_PARSE_PARAMETERS_START(1, 1)
            Z_PARAM_ZVAL(array)
            Z_PARAM_OPTIONAL
            Z_PARAM_LONG(axis)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }
    axis_i = (int)axis;
    if (ZEND_NUM_ARGS() == 1) {
        rtn = NDArray_Transpose(nda, NULL);
        add_to_buffer(rtn, sizeof(NDArray));
        RETURN_NDARRAY(rtn, return_value);
    } else {
        if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_GPU) {
            zend_throw_error(NULL, "Axis not supported for GPU operation");
            return;
        }
        zend_throw_error(NULL, "Not implemented");
        return;
    }
}

/**
 * NDArray::shape
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_shape, 0, 0, 1)
    ZEND_ARG_INFO(0, array)
    ZEND_ARG_INFO(0, axis)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, shape)
{
    NDArray *rtn = NULL;
    zval *array;
    long axis;
    int axis_i;
    ZEND_PARSE_PARAMETERS_START(1, 1)
            Z_PARAM_ZVAL(array)
            Z_PARAM_OPTIONAL
            Z_PARAM_LONG(axis)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }
    axis_i = (int)axis;
    if (ZEND_NUM_ARGS() == 1) {
        rtn = NDArray_Transpose(nda, NULL);
        add_to_buffer(rtn, sizeof(NDArray));
        RETURN_NDARRAY(rtn, return_value);
    } else {
        if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_GPU) {
            zend_throw_error(NULL, "Axis not supported for GPU operation");
            return;
        }
        zend_throw_error(NULL, "Not implemented");
        return;
    }
}

/**
 * NDArray::flatten
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_flat, 0, 0, 0)
    ZEND_ARG_INFO(0, a)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, flatten)
{
    NDArray *rtn = NULL;
    zval *a;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ZVAL(a)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(a);
    if (nda == NULL) {
        return;
    }
    rtn = NDArray_Flatten(nda);
    CHECK_INPUT_AND_FREE(a, nda);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NDArray::abs
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_abs, 0, 0, 1)
    ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, abs)
{
    NDArray *rtn = NULL;
    zval *array;
    ZEND_PARSE_PARAMETERS_START(1, 1)
            Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }

    rtn = NDArray_Abs(nda);

    if (Z_TYPE_P(array) == IS_ARRAY) {
        NDArray_FREE(nda);
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NDArray::sin
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_sin, 0, 0, 1)
                ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, sin)
{
    NDArray *rtn = NULL;
    zval *array;
    ZEND_PARSE_PARAMETERS_START(1, 1)
            Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }

    if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_CPU) {
        rtn = NDArray_Map(nda, float_sin);
    } else {
#ifdef HAVE_CUBLAS
        rtn = NDArrayMathGPU_ElementWise(nda, cuda_float_sin);
#else
        zend_throw_error(NULL, "GPU operations unavailable. CUBLAS not detected.");
#endif
    }
    if (Z_TYPE_P(array) == IS_ARRAY) {
        NDArray_FREE(nda);
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NDArray::cos
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_cos, 0, 0, 1)
                ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, cos)
{
    NDArray *rtn = NULL;
    zval *array;
    ZEND_PARSE_PARAMETERS_START(1, 1)
            Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }

    if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_CPU) {
        rtn = NDArray_Map(nda, float_cos);
    } else {
#ifdef HAVE_CUBLAS
        rtn = NDArrayMathGPU_ElementWise(nda, cuda_float_cos);
#else
        zend_throw_error(NULL, "GPU operations unavailable. CUBLAS not detected.");
#endif
    }
    if (Z_TYPE_P(array) == IS_ARRAY) {
        NDArray_FREE(nda);
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NDArray::sin
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_tan, 0, 0, 1)
                ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, tan)
{
    NDArray *rtn = NULL;
    zval *array;
    ZEND_PARSE_PARAMETERS_START(1, 1)
            Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }

    if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_CPU) {
        rtn = NDArray_Map(nda, float_tan);
    } else {
#ifdef HAVE_CUBLAS
        rtn = NDArrayMathGPU_ElementWise(nda, cuda_float_tan);
#else
        zend_throw_error(NULL, "GPU operations unavailable. CUBLAS not detected.");
#endif
    }
    if (Z_TYPE_P(array) == IS_ARRAY) {
        NDArray_FREE(nda);
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NDArray::arcsin
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_arcsin, 0, 0, 1)
                ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, arcsin)
{
    NDArray *rtn = NULL;
    zval *array;
    ZEND_PARSE_PARAMETERS_START(1, 1)
            Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }

    if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_CPU) {
        rtn = NDArray_Map(nda, float_arcsin);
    } else {
#ifdef HAVE_CUBLAS
        rtn = NDArrayMathGPU_ElementWise(nda, cuda_float_arcsin);
#else
        zend_throw_error(NULL, "GPU operations unavailable. CUBLAS not detected.");
#endif
    }
    if (Z_TYPE_P(array) == IS_ARRAY) {
        NDArray_FREE(nda);
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NDArray::arccos
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_arccos, 0, 0, 1)
                ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, arccos)
{
    NDArray *rtn = NULL;
    zval *array;
    ZEND_PARSE_PARAMETERS_START(1, 1)
            Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }

    if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_CPU) {
        rtn = NDArray_Map(nda, float_arccos);
    } else {
#ifdef HAVE_CUBLAS
        rtn = NDArrayMathGPU_ElementWise(nda, cuda_float_arccos);
#else
        zend_throw_error(NULL, "GPU operations unavailable. CUBLAS not detected.");
#endif
    }
    if (Z_TYPE_P(array) == IS_ARRAY) {
        NDArray_FREE(nda);
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NDArray::arccos
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_arctan, 0, 0, 1)
                ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, arctan)
{
    NDArray *rtn = NULL;
    zval *array;
    ZEND_PARSE_PARAMETERS_START(1, 1)
            Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }

    if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_CPU) {
        rtn = NDArray_Map(nda, float_arctan);
    } else {
#ifdef HAVE_CUBLAS
        rtn = NDArrayMathGPU_ElementWise(nda, cuda_float_arctan);
#else
        zend_throw_error(NULL, "GPU operations unavailable. CUBLAS not detected.");
#endif
    }
    if (Z_TYPE_P(array) == IS_ARRAY) {
        NDArray_FREE(nda);
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NDArray::degrees
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_degrees, 0, 0, 1)
    ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, degrees)
{
    NDArray *rtn = NULL;
    zval *array;
    ZEND_PARSE_PARAMETERS_START(1, 1)
            Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }

    if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_CPU) {
        rtn = NDArray_Map(nda, float_degrees);
    } else {
#ifdef HAVE_CUBLAS
        rtn = NDArrayMathGPU_ElementWise(nda, cuda_float_degrees);
#else
        zend_throw_error(NULL, "GPU operations unavailable. CUBLAS not detected.");
#endif
    }
    if (Z_TYPE_P(array) == IS_ARRAY) {
        NDArray_FREE(nda);
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NDArray::sinh
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_sinh, 0, 0, 1)
                ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, sinh)
{
    NDArray *rtn = NULL;
    zval *array;
    ZEND_PARSE_PARAMETERS_START(1, 1)
            Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }

    if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_CPU) {
        rtn = NDArray_Map(nda, float_sinh);
    } else {
#ifdef HAVE_CUBLAS
        rtn = NDArrayMathGPU_ElementWise(nda, cuda_float_sinh);
#else
        zend_throw_error(NULL, "GPU operations unavailable. CUBLAS not detected.");
#endif
    }
    if (Z_TYPE_P(array) == IS_ARRAY) {
        NDArray_FREE(nda);
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NDArray::cosh
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_cosh, 0, 0, 1)
                ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, cosh)
{
    NDArray *rtn = NULL;
    zval *array;
    ZEND_PARSE_PARAMETERS_START(1, 1)
            Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }

    if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_CPU) {
        rtn = NDArray_Map(nda, float_cosh);
    } else {
#ifdef HAVE_CUBLAS
        rtn = NDArrayMathGPU_ElementWise(nda, cuda_float_cosh);
#else
        zend_throw_error(NULL, "GPU operations unavailable. CUBLAS not detected.");
#endif
    }
    if (Z_TYPE_P(array) == IS_ARRAY) {
        NDArray_FREE(nda);
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NDArray::tanh
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_tanh, 0, 0, 1)
    ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, tanh)
{
    NDArray *rtn = NULL;
    zval *array;
    ZEND_PARSE_PARAMETERS_START(1, 1)
            Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }

    if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_CPU) {
        rtn = NDArray_Map(nda, float_tanh);
    } else {
#ifdef HAVE_CUBLAS
        rtn = NDArrayMathGPU_ElementWise(nda, cuda_float_tanh);
#else
        zend_throw_error(NULL, "GPU operations unavailable. CUBLAS not detected.");
#endif
    }
    if (Z_TYPE_P(array) == IS_ARRAY) {
        NDArray_FREE(nda);
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NDArray::arcsinh
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_arcsinh, 0, 0, 1)
                ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, arcsinh)
{
    NDArray *rtn = NULL;
    zval *array;
    ZEND_PARSE_PARAMETERS_START(1, 1)
            Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }

    if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_CPU) {
        rtn = NDArray_Map(nda, float_arcsinh);
    } else {
#ifdef HAVE_CUBLAS
        rtn = NDArrayMathGPU_ElementWise(nda, cuda_float_arcsinh);
#else
        zend_throw_error(NULL, "GPU operations unavailable. CUBLAS not detected.");
#endif
    }
    if (Z_TYPE_P(array) == IS_ARRAY) {
        NDArray_FREE(nda);
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NDArray::arccosh
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_arccosh, 0, 0, 1)
                ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, arccosh)
{
    NDArray *rtn = NULL;
    zval *array;
    ZEND_PARSE_PARAMETERS_START(1, 1)
            Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }

    if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_CPU) {
        rtn = NDArray_Map(nda, float_arccosh);
    } else {
#ifdef HAVE_CUBLAS
        rtn = NDArrayMathGPU_ElementWise(nda, cuda_float_arccosh);
#else
        zend_throw_error(NULL, "GPU operations unavailable. CUBLAS not detected.");
#endif
    }
    if (Z_TYPE_P(array) == IS_ARRAY) {
        NDArray_FREE(nda);
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NDArray::arctanh
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_arctanh, 0, 0, 1)
        ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, arctanh)
{
    NDArray *rtn = NULL;
    zval *array;
    ZEND_PARSE_PARAMETERS_START(1, 1)
            Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }

    if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_CPU) {
        rtn = NDArray_Map(nda, float_arctanh);
    } else {
#ifdef HAVE_CUBLAS
        rtn = NDArrayMathGPU_ElementWise(nda, cuda_float_arctanh);
#else
        zend_throw_error(NULL, "GPU operations unavailable. CUBLAS not detected.");
#endif
    }
    if (Z_TYPE_P(array) == IS_ARRAY) {
        NDArray_FREE(nda);
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NDArray::rint
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_rint, 0, 0, 1)
    ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, rint)
{
    NDArray *rtn = NULL;
    zval *array;
    ZEND_PARSE_PARAMETERS_START(1, 1)
            Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }

    if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_CPU) {
        rtn = NDArray_Map(nda, float_rint);
    } else {
#ifdef HAVE_CUBLAS
        rtn = NDArrayMathGPU_ElementWise(nda, cuda_float_rint);
#else
        zend_throw_error(NULL, "GPU operations unavailable. CUBLAS not detected.");
#endif
    }
    if (Z_TYPE_P(array) == IS_ARRAY) {
        NDArray_FREE(nda);
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NDArray::fix
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_fix, 0, 0, 1)
    ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, fix)
{
    NDArray *rtn = NULL;
    zval *array;
    ZEND_PARSE_PARAMETERS_START(1, 1)
            Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }

    if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_CPU) {
        rtn = NDArray_Map(nda, float_fix);
    } else {
#ifdef HAVE_CUBLAS
        rtn = NDArrayMathGPU_ElementWise(nda, cuda_float_fix);
#else
        zend_throw_error(NULL, "GPU operations unavailable. CUBLAS not detected.");
#endif
    }
    if (Z_TYPE_P(array) == IS_ARRAY) {
        NDArray_FREE(nda);
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NDArray::trunc
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_trunc, 0, 0, 1)
    ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, trunc)
{
    NDArray *rtn = NULL;
    zval *array;
    ZEND_PARSE_PARAMETERS_START(1, 1)
            Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }

    if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_CPU) {
        rtn = NDArray_Map(nda, float_trunc);
    } else {
#ifdef HAVE_CUBLAS
        rtn = NDArrayMathGPU_ElementWise(nda, cuda_float_trunc);
#else
        zend_throw_error(NULL, "GPU operations unavailable. CUBLAS not detected.");
#endif
    }
    if (Z_TYPE_P(array) == IS_ARRAY) {
        NDArray_FREE(nda);
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NDArray::sinc
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_sinc, 0, 0, 1)
                ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, sinc)
{
    NDArray *rtn = NULL;
    zval *array;
    ZEND_PARSE_PARAMETERS_START(1, 1)
            Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }

    if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_CPU) {
        rtn = NDArray_Map(nda, float_sinc);
    } else {
#ifdef HAVE_CUBLAS
        rtn = NDArrayMathGPU_ElementWise(nda, cuda_float_sinc);
#else
        zend_throw_error(NULL, "GPU operations unavailable. CUBLAS not detected.");
#endif
    }
    if (Z_TYPE_P(array) == IS_ARRAY) {
        NDArray_FREE(nda);
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NDArray::negative
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_negative, 0, 0, 1)
                ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, negative)
{
    NDArray *rtn = NULL;
    zval *array;
    ZEND_PARSE_PARAMETERS_START(1, 1)
            Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }

    if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_CPU) {
        rtn = NDArray_Map(nda, float_negate);
    } else {
#ifdef HAVE_CUBLAS
        rtn = NDArrayMathGPU_ElementWise(nda, cuda_float_negate);
#else
        zend_throw_error(NULL, "GPU operations unavailable. CUBLAS not detected.");
#endif
    }
    if (Z_TYPE_P(array) == IS_ARRAY) {
        NDArray_FREE(nda);
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NDArray::sign
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_sign, 0, 0, 1)
                ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, sign)
{
    NDArray *rtn = NULL;
    zval *array;
    ZEND_PARSE_PARAMETERS_START(1, 1)
            Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }

    if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_CPU) {
        rtn = NDArray_Map(nda, float_sign);
    } else {
#ifdef HAVE_CUBLAS
        rtn = NDArrayMathGPU_ElementWise(nda, cuda_float_sign);
#else
        zend_throw_error(NULL, "GPU operations unavailable. CUBLAS not detected.");
#endif
    }
    if (Z_TYPE_P(array) == IS_ARRAY) {
        NDArray_FREE(nda);
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NDArray::clip
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_clip, 0, 0, 1)
    ZEND_ARG_INFO(0, array)
    ZEND_ARG_INFO(0, min)
    ZEND_ARG_INFO(0, max)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, clip)
{
    NDArray *rtn = NULL;
    zval *array;
    double min, max;
    ZEND_PARSE_PARAMETERS_START(3, 3)
            Z_PARAM_ZVAL(array)
            Z_PARAM_DOUBLE(min)
            Z_PARAM_DOUBLE(max)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }

    if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_CPU) {
        rtn = NDArray_Map2F(nda, float_clip, (float)min, (float)max);
    } else {
#ifdef HAVE_CUBLAS
        rtn = NDArrayMathGPU_ElementWise2F(nda, cuda_float_clip, (float)min, (float)max);
#else
        zend_throw_error(NULL, "GPU operations unavailable. CUBLAS not detected.");
#endif
    }
    if (Z_TYPE_P(array) == IS_ARRAY) {
        NDArray_FREE(nda);
    }
    RETURN_NDARRAY(rtn, return_value);
}


/**
 * NDArray::mean
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_mean, 0, 0, 1)
    ZEND_ARG_INFO(0, array)
    ZEND_ARG_INFO(0, axis)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, mean)
{
    NDArray *rtn = NULL;
    zval *array;
    long axis;
    int i_axis;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    i_axis = (int)axis;
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }

    if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_CPU) {
        RETURN_DOUBLE((NDArray_Sum_Float(nda) / NDArray_NUMELEMENTS(nda)));
    } else {
#ifdef HAVE_CUBLAS
        if (ZEND_NUM_ARGS() == 1) {
            RETURN_DOUBLE((NDArray_Sum_Float(nda) / NDArray_NUMELEMENTS(nda)));
        } else {
            rtn = single_reduce(nda, &i_axis, NDArray_Mean_Float);
        }
#else
        zend_throw_error(NULL, "GPU operations unavailable. CUBLAS not detected.");
#endif
    }
    if (Z_TYPE_P(array) == IS_ARRAY) {
        NDArray_FREE(nda);
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NDArray::median
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_median, 0, 0, 1)
    ZEND_ARG_INFO(0, array)
    ZEND_ARG_INFO(0, axis)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, median)
{
    NDArray *rtn = NULL;
    zval *array;
    long axis;
    int i_axis;
    ZEND_PARSE_PARAMETERS_START(1, 1)
            Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    i_axis = (int)axis;
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }

    if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_CPU) {
        RETURN_DOUBLE((NDArray_Sum_Float(nda) / NDArray_NUMELEMENTS(nda)));
    } else {
#ifdef HAVE_CUBLAS
        if (ZEND_NUM_ARGS() == 1) {
            RETURN_DOUBLE((NDArray_Sum_Float(nda) / NDArray_NUMELEMENTS(nda)));
        } else {
            rtn = single_reduce(nda, &i_axis, NDArray_Mean_Float);
        }
#else
        zend_throw_error(NULL, "GPU operations unavailable. CUBLAS not detected.");
#endif
    }
    if (Z_TYPE_P(array) == IS_ARRAY) {
        NDArray_FREE(nda);
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NDArray::std
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_std, 0, 0, 1)
    ZEND_ARG_INFO(0, array)
    ZEND_ARG_INFO(0, axis)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, std)
{
    NDArray *rtn = NULL;
    zval *array;
    long axis;
    int i_axis;
    ZEND_PARSE_PARAMETERS_START(1, 1)
            Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    i_axis = (int)axis;
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }

    if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_CPU) {
        RETURN_DOUBLE((NDArray_Sum_Float(nda) / NDArray_NUMELEMENTS(nda)));
    } else {
#ifdef HAVE_CUBLAS
        if (ZEND_NUM_ARGS() == 1) {
            RETURN_DOUBLE((NDArray_Sum_Float(nda) / NDArray_NUMELEMENTS(nda)));
        } else {
            rtn = single_reduce(nda, &i_axis, NDArray_Mean_Float);
        }
#else
        zend_throw_error(NULL, "GPU operations unavailable. CUBLAS not detected.");
#endif
    }
    if (Z_TYPE_P(array) == IS_ARRAY) {
        NDArray_FREE(nda);
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NDArray::std
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_average, 0, 0, 1)
    ZEND_ARG_INFO(0, array)
    ZEND_ARG_INFO(0, axis)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, average)
{
    NDArray *rtn = NULL;
    zval *array;
    long axis;
    int i_axis;
    ZEND_PARSE_PARAMETERS_START(1, 1)
            Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    i_axis = (int)axis;
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }

    if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_CPU) {
        RETURN_DOUBLE((NDArray_Sum_Float(nda) / NDArray_NUMELEMENTS(nda)));
    } else {
#ifdef HAVE_CUBLAS
        if (ZEND_NUM_ARGS() == 1) {
            RETURN_DOUBLE((NDArray_Sum_Float(nda) / NDArray_NUMELEMENTS(nda)));
        } else {
            rtn = single_reduce(nda, &i_axis, NDArray_Mean_Float);
        }
#else
        zend_throw_error(NULL, "GPU operations unavailable. CUBLAS not detected.");
#endif
    }
    if (Z_TYPE_P(array) == IS_ARRAY) {
        NDArray_FREE(nda);
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NDArray::variance
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_variance, 0, 0, 1)
    ZEND_ARG_INFO(0, array)
    ZEND_ARG_INFO(0, axis)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, variance)
{
    NDArray *rtn = NULL;
    zval *array;
    long axis;
    int i_axis;
    ZEND_PARSE_PARAMETERS_START(1, 1)
            Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    i_axis = (int)axis;
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }

    if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_CPU) {
        RETURN_DOUBLE((NDArray_Sum_Float(nda) / NDArray_NUMELEMENTS(nda)));
    } else {
#ifdef HAVE_CUBLAS
        if (ZEND_NUM_ARGS() == 1) {
            RETURN_DOUBLE((NDArray_Sum_Float(nda) / NDArray_NUMELEMENTS(nda)));
        } else {
            rtn = single_reduce(nda, &i_axis, NDArray_Mean_Float);
        }
#else
        zend_throw_error(NULL, "GPU operations unavailable. CUBLAS not detected.");
#endif
    }
    if (Z_TYPE_P(array) == IS_ARRAY) {
        NDArray_FREE(nda);
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NDArray::ceil
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_ceil, 0, 0, 1)
    ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, ceil)
{
    NDArray *rtn = NULL;
    zval *array;
    ZEND_PARSE_PARAMETERS_START(1, 1)
            Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }

    if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_CPU) {
        rtn = NDArray_Map(nda, float_ceil);
    } else {
#ifdef HAVE_CUBLAS
        rtn = NDArrayMathGPU_ElementWise(nda, cuda_float_ceil);
#else
        zend_throw_error(NULL, "GPU operations unavailable. CUBLAS not detected.");
#endif
    }
    if (Z_TYPE_P(array) == IS_ARRAY) {
        NDArray_FREE(nda);
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NDArray::floor
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_floor, 0, 0, 1)
    ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, floor)
{
    NDArray *rtn = NULL;
    zval *array;
    ZEND_PARSE_PARAMETERS_START(1, 1)
            Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }

    if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_CPU) {
        rtn = NDArray_Map(nda, float_floor);
    } else {
#ifdef HAVE_CUBLAS
        rtn = NDArrayMathGPU_ElementWise(nda, cuda_float_floor);
#else
        zend_throw_error(NULL, "GPU operations unavailable. CUBLAS not detected.");
#endif
    }
    if (Z_TYPE_P(array) == IS_ARRAY) {
        NDArray_FREE(nda);
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NDArray::arccos
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_radians, 0, 0, 1)
    ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, radians)
{
    NDArray *rtn = NULL;
    zval *array;
    ZEND_PARSE_PARAMETERS_START(1, 1)
            Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }

    if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_CPU) {
        rtn = NDArray_Map(nda, float_radians);
    } else {
#ifdef HAVE_CUBLAS
        rtn = NDArrayMathGPU_ElementWise(nda, cuda_float_radians);
#else
        zend_throw_error(NULL, "GPU operations unavailable. CUBLAS not detected.");
#endif
    }
    if (Z_TYPE_P(array) == IS_ARRAY) {
        NDArray_FREE(nda);
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NDArray::sqrt
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_sqrt, 0, 0, 1)
    ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, sqrt)
{
    NDArray *rtn = NULL;
    zval *array;
    ZEND_PARSE_PARAMETERS_START(1, 1)
            Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }
    if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_CPU) {
        rtn = NDArray_Map(nda, float_sqrt);
    } else {
#ifdef HAVE_CUBLAS
        rtn = NDArrayMathGPU_ElementWise(nda, cuda_float_sqrt);
#else
        zend_throw_error(NULL, "GPU operations unavailable. CUBLAS not detected.");
#endif
    }
    if (Z_TYPE_P(array) == IS_ARRAY) {
        NDArray_FREE(nda);
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NDArray::sqrt
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_square, 0, 0, 1)
    ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, square)
{
    NDArray *rtn = NULL;
    zval *array;
    ZEND_PARSE_PARAMETERS_START(1, 1)
            Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }
    rtn = NDArray_Multiply_Float(nda, nda);
    if (Z_TYPE_P(array) == IS_ARRAY) {
        NDArray_FREE(nda);
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NDArray::exp
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_exp, 0, 0, 1)
    ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, exp)
{
    NDArray *rtn = NULL;
    zval *array;
    ZEND_PARSE_PARAMETERS_START(1, 1)
            Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }
    if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_CPU) {
        rtn = NDArray_Map(nda, float_exp);
    } else {
#ifdef HAVE_CUBLAS
        rtn = NDArrayMathGPU_ElementWise(nda, cuda_float_exp);
#else
        zend_throw_error(NULL, "GPU operations unavailable. CUBLAS not detected.");
#endif
    }
    if (Z_TYPE_P(array) == IS_ARRAY) {
        NDArray_FREE(nda);
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NDArray::exp2
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_exp2, 0, 0, 1)
                ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, exp2)
{
    NDArray *rtn = NULL;
    zval *array;
    ZEND_PARSE_PARAMETERS_START(1, 1)
            Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }
    rtn = NDArray_Map(nda, float_exp2);
    if (Z_TYPE_P(array) == IS_ARRAY) {
        NDArray_FREE(nda);
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NDArray::exp2
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_expm1, 0, 0, 1)
    ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, expm1)
{
    NDArray *rtn = NULL;
    zval *array;
    ZEND_PARSE_PARAMETERS_START(1, 1)
            Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }

    if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_CPU) {
        rtn = NDArray_Map(nda, float_expm1);
    } else {
#ifdef HAVE_CUBLAS
        rtn = NDArrayMathGPU_ElementWise(nda, cuda_float_expm1);
#else
        zend_throw_error(NULL, "GPU operations unavailable. CUBLAS not detected.");
#endif
    }
    if (Z_TYPE_P(array) == IS_ARRAY) {
        NDArray_FREE(nda);
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NDArray::log
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_log, 0, 0, 1)
                ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, log)
{
    NDArray *rtn = NULL;
    zval *array;
    ZEND_PARSE_PARAMETERS_START(1, 1)
            Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }
    if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_CPU) {
        rtn = NDArray_Map(nda, float_log);
    } else {
#ifdef HAVE_CUBLAS
        rtn = NDArrayMathGPU_ElementWise(nda, cuda_float_log);
#else
        zend_throw_error(NULL, "GPU operations unavailable. CUBLAS not detected.");
#endif
    }
    if (Z_TYPE_P(array) == IS_ARRAY) {
        NDArray_FREE(nda);
    }
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NDArray::logb
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_logb, 0, 0, 1)
                ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, logb)
{
    NDArray *rtn = NULL;
    zval *array;
    ZEND_PARSE_PARAMETERS_START(1, 1)
            Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }
    if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_CPU) {
        rtn = NDArray_Map(nda, float_logb);
    } else {
#ifdef HAVE_CUBLAS
        rtn = NDArrayMathGPU_ElementWise(nda, cuda_float_logb);
#else
        zend_throw_error(NULL, "GPU operations unavailable. CUBLAS not detected.");
#endif
    }
    CHECK_INPUT_AND_FREE(array, nda);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NDArray::log10
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_log10, 0, 0, 1)
                ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, log10)
{
    NDArray *rtn = NULL;
    zval *array;
    ZEND_PARSE_PARAMETERS_START(1, 1)
            Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }
    if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_CPU) {
        rtn = NDArray_Map(nda, float_log10);
    } else {
#ifdef HAVE_CUBLAS
        rtn = NDArrayMathGPU_ElementWise(nda, cuda_float_log10);
#else
        zend_throw_error(NULL, "GPU operations unavailable. CUBLAS not detected.");
#endif
    }
    CHECK_INPUT_AND_FREE(array, nda);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NDArray::log1p
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_log1p, 0, 0, 1)
                ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, log1p)
{
    NDArray *rtn = NULL;
    zval *array;
    ZEND_PARSE_PARAMETERS_START(1, 1)
            Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }
    if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_CPU) {
        rtn = NDArray_Map(nda, float_log1p);
    } else {
#ifdef HAVE_CUBLAS
        rtn = NDArrayMathGPU_ElementWise(nda, cuda_float_log1p);
#else
        zend_throw_error(NULL, "GPU operations unavailable. CUBLAS not detected.");
#endif
    }
    CHECK_INPUT_AND_FREE(array, nda);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NDArray::log1p
 *
 * @param execute_data
 * @param return_value
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_log2, 0, 0, 1)
                ZEND_ARG_INFO(0, array)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, log2)
{
    NDArray *rtn = NULL;
    zval *array;
    ZEND_PARSE_PARAMETERS_START(1, 1)
            Z_PARAM_ZVAL(array)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(array);
    if (nda == NULL) {
        return;
    }
    if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_CPU) {
        rtn = NDArray_Map(nda, float_log2);
    } else {
#ifdef HAVE_CUBLAS
        rtn = NDArrayMathGPU_ElementWise(nda, cuda_float_log2);
#else
        zend_throw_error(NULL, "GPU operations unavailable. CUBLAS not detected.");
#endif
    }
    CHECK_INPUT_AND_FREE(array, nda);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NDArray::subtract
 */
ZEND_BEGIN_ARG_INFO(arginfo_ndarray_subtract, 0)
    ZEND_ARG_OBJ_INFO(0, a, NDArray, 0)
    ZEND_ARG_OBJ_INFO(0, b, NDArray, 0)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, subtract)
{
    NDArray *rtn = NULL;
    zval *a, *b;
    long axis;
    ZEND_PARSE_PARAMETERS_START(2, 2)
            Z_PARAM_ZVAL(a)
            Z_PARAM_ZVAL(b)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(a);
    NDArray *ndb = ZVAL_TO_NDARRAY(b);
    if (nda == NULL) {
        return;
    }
    if (ndb == NULL) {
        CHECK_INPUT_AND_FREE(a, nda);
        return;
    }
    if (!NDArray_ShapeCompare(nda, ndb)) {
        zend_throw_error(NULL, "Incompatible shapes");
        return;
    }
    rtn = NDArray_Subtract_Float(nda, ndb);
    CHECK_INPUT_AND_FREE(a, nda);
    CHECK_INPUT_AND_FREE(b, ndb);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NDArray::mod
 */
ZEND_BEGIN_ARG_INFO(arginfo_ndarray_mod, 0)
                ZEND_ARG_OBJ_INFO(0, a, NDArray, 0)
                ZEND_ARG_OBJ_INFO(0, b, NDArray, 0)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, mod)
{
    NDArray *rtn = NULL;
    zval *a, *b;
    long axis;
    ZEND_PARSE_PARAMETERS_START(2, 2)
            Z_PARAM_ZVAL(a)
            Z_PARAM_ZVAL(b)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(a);
    NDArray *ndb = ZVAL_TO_NDARRAY(b);
    if (nda == NULL) {
        return;
    }
    if (ndb == NULL) {
        CHECK_INPUT_AND_FREE(a, nda);
        return;
    }
    if (!NDArray_ShapeCompare(nda, ndb)) {
        zend_throw_error(NULL, "Incompatible shapes");
        return;
    }
    rtn = NDArray_Mod_Float(nda, ndb);
    CHECK_INPUT_AND_FREE(a, nda);
    CHECK_INPUT_AND_FREE(a, ndb);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NDArray::pow
 */
ZEND_BEGIN_ARG_INFO(arginfo_ndarray_pow, 0)
    ZEND_ARG_OBJ_INFO(0, a, NDArray, 0)
    ZEND_ARG_OBJ_INFO(0, b, NDArray, 0)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, pow)
{
    NDArray *rtn = NULL;
    zval *a, *b;
    long axis;
    ZEND_PARSE_PARAMETERS_START(2, 2)
            Z_PARAM_ZVAL(a)
            Z_PARAM_ZVAL(b)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(a);
    NDArray *ndb = ZVAL_TO_NDARRAY(b);
    if (nda == NULL) {
        return;
    }
    if (ndb == NULL) {
        CHECK_INPUT_AND_FREE(a, nda);
        return;
    }
    if (!NDArray_ShapeCompare(nda, ndb)) {
        zend_throw_error(NULL, "Incompatible shapes");
        return;
    }
    rtn = NDArray_Pow_Float(nda, ndb);
    CHECK_INPUT_AND_FREE(a, nda);
    CHECK_INPUT_AND_FREE(b, ndb);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NDArray::multiply
 */
ZEND_BEGIN_ARG_INFO(arginfo_ndarray_multiply, 0)
    ZEND_ARG_OBJ_INFO(0, a, NDArray, 0)
    ZEND_ARG_OBJ_INFO(0, b, NDArray, 0)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, multiply)
{
    NDArray *rtn = NULL;
    zval *a, *b;
    long axis;
    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_ZVAL(a)
        Z_PARAM_ZVAL(b)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(a);
    NDArray *ndb = ZVAL_TO_NDARRAY(b);
    if (nda == NULL) {
        return;
    }
    if (ndb == NULL) {
        CHECK_INPUT_AND_FREE(a, nda);
        return;
    }
    if (!NDArray_ShapeCompare(nda, ndb)) {
        zend_throw_error(NULL, "Incompatible shapes");
        return;
    }
    rtn = NDArray_Multiply_Float(nda, ndb);

    CHECK_INPUT_AND_FREE(a, nda);
    CHECK_INPUT_AND_FREE(b, ndb);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NDArray::divide
 */
ZEND_BEGIN_ARG_INFO(arginfo_ndarray_divide, 0)
        ZEND_ARG_OBJ_INFO(0, a, NDArray, 0)
        ZEND_ARG_OBJ_INFO(0, b, NDArray, 0)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, divide)
{
    NDArray *rtn = NULL;
    zval *a, *b;
    long axis;
    ZEND_PARSE_PARAMETERS_START(2, 2)
            Z_PARAM_ZVAL(a)
            Z_PARAM_ZVAL(b)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(a);
    NDArray *ndb = ZVAL_TO_NDARRAY(b);
    if (nda == NULL) {
        return;
    }
    if (ndb == NULL) {
        return;
    }
    if (!NDArray_ShapeCompare(nda, ndb)) {
        zend_throw_error(NULL, "Incompatible shapes");
        return;
    }
    rtn = NDArray_Divide_Float(nda, ndb);
    CHECK_INPUT_AND_FREE(a, nda);
    CHECK_INPUT_AND_FREE(b, ndb);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NDArray::add
 */
ZEND_BEGIN_ARG_INFO(arginfo_ndarray_add, 0)
    ZEND_ARG_INFO(0, a)
    ZEND_ARG_INFO(0, b)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, add)
{
    NDArray *rtn = NULL;
    zval *a, *b;
    long axis;
    ZEND_PARSE_PARAMETERS_START(2, 2)
            Z_PARAM_ZVAL(a)
            Z_PARAM_ZVAL(b)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(a);
    NDArray *ndb = ZVAL_TO_NDARRAY(b);
    if (nda == NULL) {
        return;
    }
    if (ndb == NULL) {
        CHECK_INPUT_AND_FREE(a, nda);
        return;
    }
    if (!NDArray_IsBroadcastable(nda, ndb)) {
        zend_throw_error(NULL, "Can´t broadcast array.");
    }
    rtn = NDArray_Add_Float(nda, ndb);
    CHECK_INPUT_AND_FREE(a, nda);
    CHECK_INPUT_AND_FREE(b, ndb);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NDArray::matmul
 */
ZEND_BEGIN_ARG_INFO(arginfo_ndarray_matmul, 0)
    ZEND_ARG_INFO(0, a)
    ZEND_ARG_INFO(0, b)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, matmul)
{
    NDArray *rtn = NULL;
    zval *a, *b;
    long axis;
    ZEND_PARSE_PARAMETERS_START(2, 2)
            Z_PARAM_ZVAL(a)
            Z_PARAM_ZVAL(b)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(a);
    NDArray *ndb = ZVAL_TO_NDARRAY(b);
    if (nda == NULL) {
        return;
    }
    if (ndb == NULL) {
        CHECK_INPUT_AND_FREE(a, nda);
        return;
    }
    rtn = NDArray_Matmul(nda, ndb);

    CHECK_INPUT_AND_FREE(a, nda);
    CHECK_INPUT_AND_FREE(b, ndb);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NDArray::inner
 */
ZEND_BEGIN_ARG_INFO(arginfo_ndarray_inner, 0)
    ZEND_ARG_INFO(0, a)
    ZEND_ARG_INFO(0, b)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, inner)
{
    NDArray *rtn = NULL;
    zval *a, *b;
    long axis;
    ZEND_PARSE_PARAMETERS_START(2, 2)
            Z_PARAM_ZVAL(a)
            Z_PARAM_ZVAL(b)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(a);
    NDArray *ndb = ZVAL_TO_NDARRAY(b);
    if (nda == NULL) {
        return;
    }
    if (ndb == NULL) {
        CHECK_INPUT_AND_FREE(a, nda);
        return;
    }
    rtn = NDArray_Inner(nda, ndb);

    CHECK_INPUT_AND_FREE(a, nda);
    CHECK_INPUT_AND_FREE(b, ndb);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NDArray::outer
 */
ZEND_BEGIN_ARG_INFO(arginfo_ndarray_outer, 0)
    ZEND_ARG_INFO(0, a)
    ZEND_ARG_INFO(0, b)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, outer)
{
    NDArray *rtn = NULL;
    zval *a, *b;
    long axis;
    ZEND_PARSE_PARAMETERS_START(2, 2)
            Z_PARAM_ZVAL(a)
            Z_PARAM_ZVAL(b)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(a);
    NDArray *ndb = ZVAL_TO_NDARRAY(b);
    if (nda == NULL) {
        return;
    }
    if (ndb == NULL) {
        CHECK_INPUT_AND_FREE(a, nda);
        return;
    }
    rtn = NDArray_Matmul(nda, ndb);

    CHECK_INPUT_AND_FREE(a, nda);
    CHECK_INPUT_AND_FREE(b, ndb);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NDArray::dot
 */
ZEND_BEGIN_ARG_INFO(arginfo_ndarray_dot, 0)
        ZEND_ARG_INFO(0, a)
        ZEND_ARG_INFO(0, b)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, dot)
{
    NDArray *rtn = NULL;
    zval *a, *b;
    long axis;
    ZEND_PARSE_PARAMETERS_START(2, 2)
            Z_PARAM_ZVAL(a)
            Z_PARAM_ZVAL(b)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(a);
    NDArray *ndb = ZVAL_TO_NDARRAY(b);
    if (nda == NULL) {
        return;
    }
    if (ndb == NULL) {
        CHECK_INPUT_AND_FREE(a, nda);
        return;
    }
    rtn = NDArray_Dot(nda, ndb);
    CHECK_INPUT_AND_FREE(a, nda);
    CHECK_INPUT_AND_FREE(b, ndb);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NDArray::trace
 */
ZEND_BEGIN_ARG_INFO(arginfo_ndarray_trace, 0)
        ZEND_ARG_INFO(0, a)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, trace)
{
    NDArray *rtn = NULL;
    zval *a, *b;
    long axis;
    ZEND_PARSE_PARAMETERS_START(1, 1)
            Z_PARAM_ZVAL(a)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(a);
    if (nda == NULL) {
        return;
    }

    CHECK_INPUT_AND_FREE(a, nda);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NDArray::eig
 */
ZEND_BEGIN_ARG_INFO(arginfo_ndarray_eig, 0)
        ZEND_ARG_INFO(0, a)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, eig)
{
    NDArray *rtn = NULL;
    zval *a, *b;
    long axis;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ZVAL(a)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(a);
    if (nda == NULL) {
        return;
    }

    CHECK_INPUT_AND_FREE(a, nda);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NDArray::cholesky
 */
ZEND_BEGIN_ARG_INFO(arginfo_ndarray_cholesky, 0)
    ZEND_ARG_INFO(0, a)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, cholesky)
{
    NDArray *rtn = NULL;
    zval *a, *b;
    long axis;
    ZEND_PARSE_PARAMETERS_START(1, 1)
            Z_PARAM_ZVAL(a)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(a);
    if (nda == NULL) {
        return;
    }

    CHECK_INPUT_AND_FREE(a, nda);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NDArray::solve
 */
ZEND_BEGIN_ARG_INFO(arginfo_ndarray_solve, 1)
    ZEND_ARG_INFO(0, a)
    ZEND_ARG_INFO(0, b)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, solve)
{
    NDArray *rtn = NULL;
    zval *a, *b;
    long axis;
    ZEND_PARSE_PARAMETERS_START(2, 2)
        Z_PARAM_ZVAL(a)
        Z_PARAM_ZVAL(b)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(a);
    if (nda == NULL) {
        return;
    }

    CHECK_INPUT_AND_FREE(a, nda);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NDArray::lstsq
 */
ZEND_BEGIN_ARG_INFO(arginfo_ndarray_lstsq, 1)
    ZEND_ARG_INFO(0, a)
    ZEND_ARG_INFO(0, b)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, lstsq)
{
    NDArray *rtn = NULL;
    zval *a, *b;
    long axis;
    ZEND_PARSE_PARAMETERS_START(2, 2)
            Z_PARAM_ZVAL(a)
            Z_PARAM_ZVAL(b)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(a);
    NDArray *ndb = ZVAL_TO_NDARRAY(b);
    if (nda == NULL) {
        return;
    }
    if (ndb == NULL) {
        CHECK_INPUT_AND_FREE(a, nda);
        return;
    }
    rtn = NDArray_Matmul(nda, ndb);

    CHECK_INPUT_AND_FREE(a, nda);
    CHECK_INPUT_AND_FREE(b, ndb);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NDArray::qr
 */
ZEND_BEGIN_ARG_INFO(arginfo_ndarray_qr, 0)
                ZEND_ARG_INFO(0, a)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, qr)
{
    NDArray **rtns;
    zval *a, *b;
    long axis;
    ZEND_PARSE_PARAMETERS_START(1, 1)
            Z_PARAM_ZVAL(a)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(a);
    if (nda == NULL) {
        return;
    }

    rtns = NDArray_SVD(nda);

    CHECK_INPUT_AND_FREE(a, nda);
    RETURN_3NDARRAY(rtns[0], rtns[1], rtns[2], return_value);
    efree(rtns);
}

/**
 * NDArray::lu
 */
ZEND_BEGIN_ARG_INFO(arginfo_ndarray_lu, 0)
    ZEND_ARG_INFO(0, a)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, lu)
{
    NDArray **rtns;
    zval *a, *b;
    long axis;
    ZEND_PARSE_PARAMETERS_START(1, 1)
            Z_PARAM_ZVAL(a)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(a);
    if (nda == NULL) {
        return;
    }

    rtns = NDArray_LU(nda);

    CHECK_INPUT_AND_FREE(a, nda);
    RETURN_3NDARRAY(rtns[0], rtns[1], rtns[2], return_value);
    efree(rtns);
}

/**
 * NDArray::norm
 */
ZEND_BEGIN_ARG_INFO(arginfo_ndarray_norm, 1)
    ZEND_ARG_INFO(0, a)
    ZEND_ARG_INFO(0, order)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, norm)
{
    NDArray *rtn;
    zval *a;
    long order = INT_MAX;
    ZEND_PARSE_PARAMETERS_START(1, 2)
            Z_PARAM_ZVAL(a)
            Z_PARAM_OPTIONAL
            Z_PARAM_LONG(order)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(a);
    if (nda == NULL) {
        return;
    }

    rtn = NDArray_Norm(nda, (int)order);

    CHECK_INPUT_AND_FREE(a, nda);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NDArray::cond
 */
ZEND_BEGIN_ARG_INFO(arginfo_ndarray_cond, 0)
    ZEND_ARG_INFO(0, a)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, cond)
{
    NDArray **rtns;
    zval *a, *b;
    long axis;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ZVAL(a)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(a);
    if (nda == NULL) {
        return;
    }

    rtns = NDArray_SVD(nda);

    CHECK_INPUT_AND_FREE(a, nda);
    RETURN_3NDARRAY(rtns[0], rtns[1], rtns[2], return_value);
    efree(rtns);
}

/**
 * NDArray::inv
 */
ZEND_BEGIN_ARG_INFO(arginfo_ndarray_inv, 0)
    ZEND_ARG_INFO(0, a)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, inv)
{
    NDArray *rtn;
    zval *a, *b;
    long axis;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ZVAL(a)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(a);
    if (nda == NULL) {
        return;
    }

    rtn = NDArray_Inverse(nda);

    CHECK_INPUT_AND_FREE(a, nda);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NDArray::svd
 */
ZEND_BEGIN_ARG_INFO(arginfo_ndarray_svd, 0)
    ZEND_ARG_INFO(0, a)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, svd)
{
    NDArray **rtns;
    zval *a, *b;
    long axis;
    ZEND_PARSE_PARAMETERS_START(1, 1)
            Z_PARAM_ZVAL(a)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(a);
    if (nda == NULL) {
        return;
    }
    rtns = NDArray_SVD(nda);
    if (rtns == NULL) {
        return;
    }
    CHECK_INPUT_AND_FREE(a, nda);
    RETURN_3NDARRAY(rtns[0], rtns[1], rtns[2], return_value);
    efree(rtns);
}

/**
 * NDArray::det
 */
ZEND_BEGIN_ARG_INFO(arginfo_ndarray_det, 0)
    ZEND_ARG_INFO(0, a)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, det)
{
    NDArray *rtn, *nda;
    zval *a;
    ZEND_PARSE_PARAMETERS_START(1, 1)
            Z_PARAM_ZVAL(a)
    ZEND_PARSE_PARAMETERS_END();
    nda = ZVAL_TO_NDARRAY(a);
    if (nda == NULL) {
        return;
    }
    rtn = NDArray_Det(nda);
    CHECK_INPUT_AND_FREE(a, nda);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NDArray::sum
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_sum, 0, 0, 1)
    ZEND_ARG_OBJ_INFO(0, a, NDArray, 0)
    ZEND_ARG_INFO(0, axis)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, sum)
{
    NDArray *rtn = NULL;
    zval *a;
    long axis;
    int axis_i;
    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_ZVAL(a)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(axis)
    ZEND_PARSE_PARAMETERS_END();
    axis_i = (int)axis;
    NDArray *nda = ZVAL_TO_NDARRAY(a);
    if (nda == NULL) {
        return;
    }
    if (ZEND_NUM_ARGS() == 2) {
        if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_GPU) {
            zend_throw_error(NULL, "Axis not supported for GPU operation");
            return;
        }
        rtn = reduce(nda, &axis_i, NDArray_Add_Float);
    } else {
        double value = NDArray_Sum_Float(nda);
        CHECK_INPUT_AND_FREE(a, nda);
        RETURN_DOUBLE(value);
        return;
    }
    CHECK_INPUT_AND_FREE(a, nda);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NDArray::min
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_min, 0, 0, 1)
                ZEND_ARG_OBJ_INFO(0, a, NDArray, 0)
                ZEND_ARG_INFO(0, axis)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, min)
{
    NDArray *rtn = NULL;
    zval *a;
    long axis;
    int axis_i;
    double value;
    ZEND_PARSE_PARAMETERS_START(1, 1)
            Z_PARAM_ZVAL(a)
            Z_PARAM_OPTIONAL
            Z_PARAM_LONG(axis)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(a);
    if (nda == NULL) {
        return;
    }
    if (ZEND_NUM_ARGS() == 2) {
        if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_GPU) {
            zend_throw_error(NULL, "Axis not supported for GPU operation");
            return;
        }
        axis_i = (int)axis;
        rtn = single_reduce(nda, &axis_i, NDArray_Min);
    } else {
        value = NDArray_Min(nda);
        CHECK_INPUT_AND_FREE(a, nda);
        RETURN_DOUBLE(value);
        return;
    }
    CHECK_INPUT_AND_FREE(a, nda);
    RETURN_NDARRAY(rtn, return_value);
}

/**
 * NDArray::max
 */
ZEND_BEGIN_ARG_INFO_EX(arginfo_ndarray_max, 0, 0, 1)
                ZEND_ARG_OBJ_INFO(0, a, NDArray, 0)
                ZEND_ARG_INFO(0, axis)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, max)
{
    NDArray *rtn = NULL;
    zval *a;
    long axis;
    int axis_i;
    double value;
    ZEND_PARSE_PARAMETERS_START(1, 1)
            Z_PARAM_ZVAL(a)
            Z_PARAM_OPTIONAL
            Z_PARAM_LONG(axis)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(a);
    if (nda == NULL) {
        return;
    }
    if (ZEND_NUM_ARGS() == 2) {
        if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_GPU) {
            zend_throw_error(NULL, "Axis not supported for GPU operation");
            return;
        }
        axis_i = (int)axis;
        rtn = single_reduce(nda, &axis_i, NDArray_Min);
    } else {
        value = NDArray_Max(nda);
        CHECK_INPUT_AND_FREE(a, nda);
        RETURN_DOUBLE(value);
        return;
    }
    CHECK_INPUT_AND_FREE(a, nda);
    RETURN_NDARRAY(rtn, return_value);
}

ZEND_BEGIN_ARG_INFO(arginfo_ndarray_prod, 0)
    ZEND_ARG_INFO(0, a)
    ZEND_ARG_INFO(0, axis)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, prod)
{
    NDArray *rtn = NULL;
    zval *a;
    long axis;
    int axis_i;
    ZEND_PARSE_PARAMETERS_START(1, 2)
            Z_PARAM_ZVAL(a)
            Z_PARAM_OPTIONAL
            Z_PARAM_LONG(axis)
    ZEND_PARSE_PARAMETERS_END();
    axis_i = (int)axis;
    NDArray *nda = ZVAL_TO_NDARRAY(a);
    if (nda == NULL) {
        return;
    }
    if (ZEND_NUM_ARGS() == 2) {
        if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_GPU) {
            zend_throw_error(NULL, "Axis not supported for GPU operation");
            return;
        }
        rtn = reduce(nda, &axis_i, NDArray_Multiply_Float);
    } else {
        rtn = NDArray_Float_Prod(nda);
        add_to_buffer(rtn, sizeof(NDArray));
    }

    CHECK_INPUT_AND_FREE(a, nda);
    RETURN_NDARRAY(rtn, return_value);
}

ZEND_BEGIN_ARG_INFO(arginfo_ndarray_array, 0)
    ZEND_ARG_INFO(0, a)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, array)
{
    NDArray *rtn = NULL;
    zval *a;
    long axis;
    ZEND_PARSE_PARAMETERS_START(1, 1)
        Z_PARAM_ZVAL(a)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(a);
    if (nda == NULL) {
        return;
    }
    RETURN_NDARRAY(nda, return_value);
}

 /**
  * @param execute_data
  * @param return_value
  */
ZEND_BEGIN_ARG_INFO(arginfo_ndarray_count, 0)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, count)
{
    zend_object *obj = Z_OBJ_P(ZEND_THIS);
    ZEND_PARSE_PARAMETERS_START(0, 0)
    ZEND_PARSE_PARAMETERS_END();
    ZVAL_LONG(OBJ_PROP_NUM(obj, 0), 1);
}

PHP_METHOD(NDArray, current)
{
    zend_object *obj = Z_OBJ_P(ZEND_THIS);
    ZEND_PARSE_PARAMETERS_START(0, 0)
    ZEND_PARSE_PARAMETERS_END();
    zval *obj_uuid = OBJ_PROP_NUM(obj, 0);
    NDArray* ndarray = ZVALUUID_TO_NDARRAY(obj_uuid);
    NDArray* result  = NDArrayIterator_GET(ndarray);
    add_to_buffer(result, sizeof(NDArray));
    RETURN_NDARRAY(result, return_value);
}

PHP_METHOD(NDArray, key)
{
    zend_object *obj = Z_OBJ_P(ZEND_THIS);
    ZEND_PARSE_PARAMETERS_START(0, 0)
    ZEND_PARSE_PARAMETERS_END();
}

PHP_METHOD(NDArray, next)
{
    zend_object *obj = Z_OBJ_P(ZEND_THIS);
    ZEND_PARSE_PARAMETERS_START(0, 0)
    ZEND_PARSE_PARAMETERS_END();
    zval *obj_uuid = OBJ_PROP_NUM(obj, 0);
    NDArray* ndarray = ZVALUUID_TO_NDARRAY(obj_uuid);
    NDArrayIterator_NEXT(ndarray);
}

PHP_METHOD(NDArray, rewind)
{
    zend_object *obj = Z_OBJ_P(ZEND_THIS);
    ZEND_PARSE_PARAMETERS_START(0, 0)
    ZEND_PARSE_PARAMETERS_END();
    zval *obj_uuid = OBJ_PROP_NUM(obj, 0);
    NDArray* ndarray = ZVALUUID_TO_NDARRAY(obj_uuid);
    NDArrayIterator_REWIND(ndarray);
}

PHP_METHOD(NDArray, valid)
{
    int is_valid = 0, is_done = 0;
    zend_object *obj = Z_OBJ_P(ZEND_THIS);
    ZEND_PARSE_PARAMETERS_START(0, 0)
    ZEND_PARSE_PARAMETERS_END();
    zval *obj_uuid = OBJ_PROP_NUM(obj, 0);
    NDArray* ndarray = ZVALUUID_TO_NDARRAY(obj_uuid);
    is_done = NDArrayIterator_ISDONE(ndarray);
    if (is_done == 0) {
        RETURN_LONG(1);
    }
    RETURN_LONG(0);
}

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_ndarray_prod___toString, 0, 0, IS_STRING, 0)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, __toString)
{
    zend_object *obj = Z_OBJ_P(ZEND_THIS);
    ZEND_PARSE_PARAMETERS_START(0, 0)
    ZEND_PARSE_PARAMETERS_END();
}

static int ndarray_do_operation_ex(zend_uchar opcode, zval *result, zval *op1, zval *op2) /* {{{ */
{
    NDArray *nda = ZVAL_TO_NDARRAY(op1);
    NDArray *ndb = ZVAL_TO_NDARRAY(op2);
    NDArray *rtn = NULL;
    switch(opcode) {
        case ZEND_ADD:
            rtn = NDArray_Add_Float(nda, ndb);
            break;
        case ZEND_SUB:
            rtn = NDArray_Subtract_Float(nda, ndb);
            break;
        case ZEND_MUL:
            rtn = NDArray_Multiply_Float(nda, ndb);
            break;
        case ZEND_DIV:
            rtn = NDArray_Divide_Float(nda, ndb);
            break;
        case ZEND_POW:
            rtn = NDArray_Pow_Float(nda, ndb);
            break;
        case ZEND_MOD:
            rtn = NDArray_Mod_Float(nda, ndb);
            break;
        default:
            return FAILURE;
    }
    RETURN_NDARRAY(rtn, result);
    if (rtn != NULL) {
        return SUCCESS;
    }
    return FAILURE;
}

static
int ndarray_do_operation(zend_uchar opcode, zval *result, zval *op1, zval *op2) /* {{{ */
{
    int retval;
    retval = ndarray_do_operation_ex(opcode, result, op1, op2);
    return retval;
}

static const zend_function_entry class_NDArray_methods[] = {
        ZEND_ME(NDArray, __construct, arginfo_construct, ZEND_ACC_PUBLIC)
        ZEND_ME(NDArray, __destruct, arginfo_ndarray_count, ZEND_ACC_PUBLIC)
        ZEND_ME(NDArray, dump, arginfo_dump, ZEND_ACC_PUBLIC)
        ZEND_ME(NDArray, dumpDevices, arginfo_dump_devices, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, gpu, arginfo_gpu, ZEND_ACC_PUBLIC)
        ZEND_ME(NDArray, cpu, arginfo_cpu, ZEND_ACC_PUBLIC)

        ZEND_ME(NDArray, min, arginfo_ndarray_min, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, max, arginfo_ndarray_max, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)

        // MANIPULATION
        ZEND_ME(NDArray, reshape, arginfo_reshape, ZEND_ACC_PUBLIC)
        ZEND_ME(NDArray, toArray, arginfo_toArray, ZEND_ACC_PUBLIC)
        ZEND_ME(NDArray, copy, arginfo_ndarray_copy, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, shape, arginfo_ndarray_shape, ZEND_ACC_PUBLIC)
        ZEND_ME(NDArray, flatten, arginfo_ndarray_flat, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, atleast_1d, arginfo_ndarray_atleast_1d, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, atleast_2d, arginfo_ndarray_atleast_2d, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, atleast_3d, arginfo_ndarray_atleast_3d, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, transpose, arginfo_ndarray_transpose, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)

        // INITIALIZERS
        ZEND_ME(NDArray, zeros, arginfo_ndarray_zeros, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, ones, arginfo_ndarray_ones, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, identity, arginfo_ndarray_identity, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, normal, arginfo_ndarray_normal, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, standard_normal, arginfo_ndarray_standard_normal, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, poisson, arginfo_ndarray_poisson, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, uniform, arginfo_ndarray_uniform, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, diag, arginfo_ndarray_diag, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, full, arginfo_ndarray_full, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, fill, arginfo_fill, ZEND_ACC_PUBLIC)
        ZEND_ME(NDArray, array, arginfo_ndarray_array, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)

        // LINALG
        ZEND_ME(NDArray, matmul, arginfo_ndarray_matmul, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, svd, arginfo_ndarray_svd, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, det, arginfo_ndarray_det, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, dot, arginfo_ndarray_dot, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, inner, arginfo_ndarray_inner, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, outer, arginfo_ndarray_outer, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, cholesky, arginfo_ndarray_cholesky, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, qr, arginfo_ndarray_qr, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, eig, arginfo_ndarray_eig, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, cond, arginfo_ndarray_cond, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, norm, arginfo_ndarray_norm, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, trace, arginfo_ndarray_trace, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, solve, arginfo_ndarray_solve, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, inv, arginfo_ndarray_inv, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, lstsq, arginfo_ndarray_lstsq, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, lu, arginfo_ndarray_lu, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)

        // LOGIC
        ZEND_ME(NDArray, all, arginfo_ndarray_all, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, equal, arginfo_ndarray_equal, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)

        // MATH
        ZEND_ME(NDArray, abs, arginfo_ndarray_abs, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, square, arginfo_ndarray_square, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, sqrt, arginfo_ndarray_sqrt, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, exp, arginfo_ndarray_exp, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, expm1, arginfo_ndarray_expm1, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, exp2, arginfo_ndarray_exp2, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, log, arginfo_ndarray_log, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, log2, arginfo_ndarray_log2, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, logb, arginfo_ndarray_logb, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, log10, arginfo_ndarray_log10, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, log1p, arginfo_ndarray_log1p, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, sin, arginfo_ndarray_sin, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, cos, arginfo_ndarray_cos, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, tan, arginfo_ndarray_tan, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, arcsin, arginfo_ndarray_arcsin, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, arccos, arginfo_ndarray_arccos, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, arctan, arginfo_ndarray_arctan, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, degrees, arginfo_ndarray_degrees, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, radians, arginfo_ndarray_radians, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, sinh, arginfo_ndarray_sinh, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, cosh, arginfo_ndarray_cosh, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, tanh, arginfo_ndarray_tanh, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, arcsinh, arginfo_ndarray_arcsinh, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, arccosh, arginfo_ndarray_arccosh, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, arctanh, arginfo_ndarray_arctanh, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, rint, arginfo_ndarray_rint, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, fix, arginfo_ndarray_fix, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, floor, arginfo_ndarray_floor, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, ceil, arginfo_ndarray_ceil, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, trunc, arginfo_ndarray_trunc, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, sinc, arginfo_ndarray_sinc, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, negative, arginfo_ndarray_negative, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, sign, arginfo_ndarray_sign, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, clip, arginfo_ndarray_clip, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)

        // STATISTICS
        ZEND_ME(NDArray, mean, arginfo_ndarray_mean, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, median, arginfo_ndarray_median, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, variance, arginfo_ndarray_variance, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, average, arginfo_ndarray_average, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, std, arginfo_ndarray_std, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)

        // ARITHMETICS
        ZEND_ME(NDArray, add, arginfo_ndarray_add, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, subtract, arginfo_ndarray_subtract, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, pow, arginfo_ndarray_pow, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, divide, arginfo_ndarray_divide, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, multiply, arginfo_ndarray_multiply, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, sum, arginfo_ndarray_sum, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, prod, arginfo_ndarray_prod, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, mod, arginfo_ndarray_mod, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)

        ZEND_ME(NDArray, count, arginfo_count, ZEND_ACC_PUBLIC)
        ZEND_ME(NDArray, current, arginfo_current, ZEND_ACC_PUBLIC)
        ZEND_ME(NDArray, key, arginfo_key, ZEND_ACC_PUBLIC)
        ZEND_ME(NDArray, next, arginfo_next, ZEND_ACC_PUBLIC)
        ZEND_ME(NDArray, rewind, arginfo_rewind, ZEND_ACC_PUBLIC)
        ZEND_ME(NDArray, valid, arginfo_valid, ZEND_ACC_PUBLIC)
        ZEND_ME(NDArray, __toString, arginfo_ndarray_prod___toString, ZEND_ACC_PUBLIC)
        ZEND_FE_END
};

static int ndarray_objects_compare(zval *obj1, zval *obj2)
{
    zval result;
    NDArray *a, *b, *c;

    a = ZVAL_TO_NDARRAY(obj1);
    b = ZVAL_TO_NDARRAY(obj2);

    if (NDArray_ArrayEqual(a, b)) {
        return 0;
    }
    return 1;
}

typedef struct {
    zend_object std;
    int value;
} NDArrayObject;

static void ndarray_objects_init(zend_class_entry *class_type)
{
    memcpy(&ndarray_object_handlers, &std_object_handlers, sizeof(zend_object_handlers));
    ndarray_object_handlers.clone_obj = NULL;
    ndarray_object_handlers.cast_object = NULL;
    ndarray_object_handlers.compare = ndarray_objects_compare;
    ndarray_object_handlers.do_operation = ndarray_do_operation;
    //ndarray_object_handlers.compare = ndarray_objects_compare;
}


static zend_object *ndarray_create_object(zend_class_entry *class_type) {
    NDArrayObject *intern = zend_object_alloc(sizeof(NDArrayObject), class_type);

    zend_object_std_init(&intern->std, class_type);
    object_properties_init(&intern->std, class_type);
    intern->std.handlers = &ndarray_object_handlers;

    return &intern->std;
}

static zend_class_entry *register_class_NDArray(zend_class_entry *class_entry_Iterator, zend_class_entry *class_entry_Countable) {
    zend_class_entry ce, *class_entry;
    INIT_CLASS_ENTRY(ce, "NDArray", class_NDArray_methods);
    ndarray_objects_init(&ce);
    ce.create_object = ndarray_create_object;
    class_entry = zend_register_internal_class(&ce);
    zend_class_implements(class_entry, 2, class_entry_Iterator, class_entry_Countable);

    zval property_id_default_value;
    ZVAL_UNDEF(&property_id_default_value);
    zend_string *property_id_name = zend_string_init("id", sizeof("id") - 1, 1);
    zend_declare_typed_property(class_entry, property_id_name, &property_id_default_value, ZEND_ACC_PUBLIC, NULL, (zend_type) ZEND_TYPE_INIT_MASK(MAY_BE_LONG));
    zend_string_release(property_id_name);

    return class_entry;
}

/**
 * MINIT
 */
PHP_MINIT_FUNCTION(ndarray)
{
    phpsci_ce_NDArray = register_class_NDArray(zend_ce_iterator, zend_ce_countable);
    //memcpy(&phpsci_ce_NDArray, &std_object_handlers, sizeof(zend_object_handlers));
    return SUCCESS;
}

PHP_RINIT_FUNCTION(ndarray)
{
    srand(time(NULL));
    bypass_printr();
    buffer_init(2);
#if defined(ZTS) && defined(COMPILE_DL_NDARRAY)
	ZEND_TSRMLS_CACHE_UPDATE();
#endif
	return SUCCESS;
}

PHP_MINFO_FUNCTION(ndarray)
{
	php_info_print_table_start();
	php_info_print_table_header(2, "support", "enabled");
	php_info_print_table_end();
}

PHP_RSHUTDOWN_FUNCTION(ndarray)
{
#ifdef HAVE_CUBLAS
    NDArray_VCHECK();
#endif
    buffer_free();
    return SUCCESS;
}

zend_module_entry ndarray_module_entry = {
	STANDARD_MODULE_HEADER,
	"NumPower",					    /* Extension name */
	ext_functions,					/* zend_function_entry */
    PHP_MINIT(ndarray),             /* PHP_MINIT - Module initialization */
    NULL,							/* PHP_MSHUTDOWN - Module shutdown */
	PHP_RINIT(ndarray),			    /* PHP_RINIT - Request initialization */
	PHP_RSHUTDOWN(ndarray), /* PHP_RSHUTDOWN - Request shutdown */
	PHP_MINFO(ndarray),			    /* PHP_MINFO - Module info */
	PHP_NDARRAY_VERSION,		    /* Version */
	STANDARD_MODULE_PROPERTIES
};
/* }}} */

#ifdef COMPILE_DL_NDARRAY
# ifdef ZTS
ZEND_TSRMLS_CACHE_DEFINE()
# endif
ZEND_GET_MODULE(ndarray)
#endif
