/* phpsci_ndarray extension for PHP */

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

#ifdef HAVE_CUBLAS
#include <cuda_runtime.h>
#include <cublas_v2.h>
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

zval* NDARRAY_TO_ZVAL(NDArray* ndarray) {
    zval* a = emalloc(sizeof(zval));
    object_init_ex(a, phpsci_ce_NDArray);
    ZVAL_LONG(OBJ_PROP_NUM(Z_OBJ_P(a), 0), NDArray_UUID(ndarray));
    return a;
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
    rtn = NDArray_ToPHPArray(array);
    RETURN_ZVAL(&rtn, 0, 0);
    NDArray_FREE(array);
}

ZEND_BEGIN_ARG_INFO(arginfo_gpu, 0)
ZEND_END_ARG_INFO();
PHP_METHOD(NDArray, gpu)
{
    NDArray *rtn;
    zval *obj_zval = getThis();
    ZEND_PARSE_PARAMETERS_START(0, 0)
    ZEND_PARSE_PARAMETERS_END();
    NDArray* array = ZVAL_TO_NDARRAY(obj_zval);
    if (array == NULL) {
        return;
    }
    rtn = NDArray_ToGPU(array);
    RETURN_NDARRAY(rtn, return_value);
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

ZEND_BEGIN_ARG_INFO(arginfo_reshape, 1)
    ZEND_ARG_INFO(0, shape_zval)
ZEND_END_ARG_INFO();
PHP_METHOD(NDArray, reshape)
{
    int *new_shape;
    zval *shape_zval;
    zval *current = getThis();
    ZEND_PARSE_PARAMETERS_START(1, 1)
            Z_PARAM_ZVAL(shape_zval)
    ZEND_PARSE_PARAMETERS_END();
    NDArray* target = ZVAL_TO_NDARRAY(current);
    NDArray* shape = ZVAL_TO_NDARRAY(shape_zval);

    new_shape = NDArray_ToIntVector(shape);
    NDArray_Reshape(target, new_shape, NDArray_NUMELEMENTS(shape));
    if (Z_TYPE_P(shape_zval) == IS_ARRAY) {
        NDArray_FREE(shape);
    }
    RETURN_ZVAL(current, 0, 0);
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
                RETURN_STR(NDArray_Print(target, 1));
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
        shape[i] = (int) NDArray_DDATA(nda)[i];
    }
    rtn = NDArray_Zeros(shape, NDArray_NUMELEMENTS(nda));
    NDArray_FREE(nda);
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
    if (nda == NULL) {
        return;
    }
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
    if (nda == NULL) {
        return;
    }
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
        shape[i] = (int) NDArray_DDATA(nda)[i];
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
    if (nda == NULL) {
        return;
    }
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
    rtn = NDArray_Ones(shape, NDArray_NUMELEMENTS(nda));
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
    } else {
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
        rtn = NDArray_Transpose(nda, NULL);
        add_to_buffer(rtn, sizeof(NDArray));
        RETURN_NDARRAY(rtn, return_value);
    } else {
        zend_throw_error(NULL, "Not implemented");
        return;
    }
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
    rtn = NDArray_Map(nda, double_abs);
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
    rtn = NDArray_Map(nda, double_sqrt);
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
    rtn = NDArray_Multiply_Double(nda, nda);
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
    rtn = NDArray_Map(nda, double_exp);
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
    rtn = NDArray_Map(nda, double_exp2);
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
    rtn = NDArray_Map(nda, double_expm1);
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
    rtn = NDArray_Map(nda, double_log);
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
    rtn = NDArray_Map(nda, double_logb);
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
    rtn = NDArray_Map(nda, double_log10);
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
    rtn = NDArray_Map(nda, double_log1p);
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
    rtn = NDArray_Map(nda, double_log2);
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
    rtn = NDArray_Subtract_Double(nda, ndb);
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
    rtn = NDArray_Mod_Double(nda, ndb);
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
    rtn = NDArray_Pow_Double(nda, ndb);
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
    rtn = NDArray_Multiply_Double(nda, ndb);

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
    rtn = NDArray_Divide_Double(nda, ndb);
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

    rtn = NDArray_Add_Double(nda, ndb);
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

    CHECK_INPUT_AND_FREE(a, nda);
    RETURN_3NDARRAY(rtns[0], rtns[1], rtns[2], return_value);
    efree(rtns);
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
    ZEND_PARSE_PARAMETERS_START(1, 2)
        Z_PARAM_ZVAL(a)
        Z_PARAM_OPTIONAL
        Z_PARAM_LONG(axis)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(a);
    if (nda == NULL) {
        return;
    }
    if (ZEND_NUM_ARGS() == 2) {
        rtn = reduce(nda, &axis, NDArray_Add_Double);
    } else {
        double value = NDArray_Sum_Double(nda);
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
    ZEND_ARG_OBJ_INFO(0, a, NDArray, 0)
    ZEND_ARG_INFO(0, axis)
ZEND_END_ARG_INFO()
PHP_METHOD(NDArray, prod)
{
    NDArray *rtn = NULL;
    zval *a;
    long axis;
    ZEND_PARSE_PARAMETERS_START(1, 2)
            Z_PARAM_ZVAL(a)
            Z_PARAM_OPTIONAL
            Z_PARAM_LONG(axis)
    ZEND_PARSE_PARAMETERS_END();
    NDArray *nda = ZVAL_TO_NDARRAY(a);
    if (nda == NULL) {
        return;
    }
    if (ZEND_NUM_ARGS() == 2) {
        rtn = reduce(nda, &axis, NDArray_Multiply_Double);
    } else {
        rtn = NDArray_Double_Prod(nda);
        add_to_buffer(rtn, sizeof(NDArray));
    }

    CHECK_INPUT_AND_FREE(a, nda);
    RETURN_NDARRAY(rtn, return_value);
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

static const zend_function_entry class_NDArray_methods[] = {
        ZEND_ME(NDArray, __construct, arginfo_construct, ZEND_ACC_PUBLIC)
        ZEND_ME(NDArray, __destruct, arginfo_ndarray_count, ZEND_ACC_PUBLIC)
        ZEND_ME(NDArray, toArray, arginfo_toArray, ZEND_ACC_PUBLIC)
        ZEND_ME(NDArray, reshape, arginfo_reshape, ZEND_ACC_PUBLIC)
        ZEND_ME(NDArray, dump, arginfo_dump, ZEND_ACC_PUBLIC)
        ZEND_ME(NDArray, gpu, arginfo_gpu, ZEND_ACC_PUBLIC)
        ZEND_ME(NDArray, cpu, arginfo_cpu, ZEND_ACC_PUBLIC)

        ZEND_ME(NDArray, min, arginfo_ndarray_min, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, max, arginfo_ndarray_max, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)

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

        // LINALG
        ZEND_ME(NDArray, matmul, arginfo_ndarray_matmul, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, svd, arginfo_ndarray_svd, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)

        // LOGIC
        ZEND_ME(NDArray, all, arginfo_ndarray_all, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)

        // MANIPULATION
        ZEND_ME(NDArray, transpose, arginfo_ndarray_transpose, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)

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

        // ARITHMETICS
        ZEND_ME(NDArray, add, arginfo_ndarray_add, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, subtract, arginfo_ndarray_subtract, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, pow, arginfo_ndarray_pow, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, divide, arginfo_ndarray_divide, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, multiply, arginfo_ndarray_multiply, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, sum, arginfo_ndarray_sum, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, prod, arginfo_ndarray_prod, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)
        ZEND_ME(NDArray, mod, arginfo_ndarray_mod, ZEND_ACC_PUBLIC | ZEND_ACC_STATIC)

        ZEND_ME(NDArray, count, arginfo_ndarray_count, ZEND_ACC_PUBLIC)
        ZEND_ME(NDArray, current, arginfo_ndarray_count, ZEND_ACC_PUBLIC)
        ZEND_ME(NDArray, key, arginfo_ndarray_count, ZEND_ACC_PUBLIC)
        ZEND_ME(NDArray, next, arginfo_ndarray_count, ZEND_ACC_PUBLIC)
        ZEND_ME(NDArray, rewind, arginfo_ndarray_count, ZEND_ACC_PUBLIC)
        ZEND_ME(NDArray, valid, arginfo_ndarray_count, ZEND_ACC_PUBLIC)
        ZEND_ME(NDArray, __toString, arginfo_ndarray_prod___toString, ZEND_ACC_PUBLIC)
        ZEND_FE_END
};

static int ndarray_objects_compare(zval *obj1, zval *obj2)
{
    zval result;
    NDArray *a, *b, *c;

    a = ZVAL_TO_NDARRAY(obj1);
    b = ZVAL_TO_NDARRAY(obj2);

    ZVAL_LONG(&result, NDArray_ArrayEqual(a, b));

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
