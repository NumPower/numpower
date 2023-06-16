/* This is a generated file, edit the .stub.php file instead.
 * Stub hash: 1df72bef182e6ee40d18127fe3e1ea56dd33cfc0 */

ZEND_BEGIN_ARG_WITH_RETURN_TYPE_INFO_EX(arginfo_print_r_, 0, 1, IS_VOID, 0)
	ZEND_ARG_TYPE_INFO(0, var, IS_MIXED, 0)
	ZEND_ARG_OBJ_INFO_WITH_DEFAULT_VALUE(0, do_return, boolean, 0, "false")
ZEND_END_ARG_INFO()


ZEND_FUNCTION(print_r_);

static const zend_function_entry ext_functions[] = {
	ZEND_FE(print_r_, arginfo_print_r_)
	ZEND_FE_END
};
