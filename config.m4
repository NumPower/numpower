dnl config.m4 for extension ndarray

dnl Comments in this file start with the string 'dnl'.
dnl Remove where necessary.

dnl If your extension references something external, use 'with':

dnl PHP_ARG_WITH([ndarray],
dnl   [for ndarray support],
dnl   [AS_HELP_STRING([--with-ndarray],
dnl     [Include ndarray support])])

dnl Otherwise use 'enable':

PHP_ARG_ENABLE([ndarray],
  [whether to enable ndarray support],
  [AS_HELP_STRING([--enable-ndarray],
    [Enable ndarray support])],
  [no])

PHP_CHECK_LIBRARY(cublas,cublasDgemm,
[
  AC_DEFINE(HAVE_CUBLAS,1,[ ])
  PHP_ADD_LIBRARY(cublas,,NDARRAY_SHARED_LIBADD)
  AC_MSG_RESULT([CUBLAS detected ])
  PHP_ADD_MAKEFILE_FRAGMENT($abs_srcdir/Makefile.frag, $abs_builddir)
  CFLAGS+=" -lcublas -lcudart "
],[
    AC_MSG_RESULT([wrong cublas version or library not found.])
    AC_CHECK_HEADER([immintrin.h],
        [
          AC_DEFINE(HAVE_AVX2,1,[Have AV2/SSE support])
          AC_MSG_RESULT([AVX2/SSE detected ])
          CFLAGS+=" -mavx2 "
        ],[
          AC_DEFINE(HAVE_AVX2,0,[Have AV2/SSE support])
          AC_MSG_RESULT([AVX2/SSE not found ])
        ], [

        ]
    )
])

PHP_CHECK_LIBRARY(cblas,cblas_sdot,
[
  AC_DEFINE(HAVE_CBLAS,1,[ ])
  PHP_ADD_LIBRARY(cblas,,NDARRAY_SHARED_LIBADD)
  AC_MSG_RESULT([CBlas detected ])
  CFLAGS+=" -lcblas "
],[
  PHP_CHECK_LIBRARY(openblas,cblas_sdot,
  [
    PHP_ADD_LIBRARY(openblas,,NDARRAY_SHARED_LIBADD)
    AC_MSG_RESULT([OpenBLAS detected ])
    AC_DEFINE(HAVE_CBLAS,1,[ ])
    CFLAGS+=" -lopenblas -lpthread "
  ],[
    AC_MSG_ERROR([wrong openblas/blas version or library not found.])
  ],[
    -lopenblas
  ])
],[
  -lcblas
])



PHP_CHECK_LIBRARY(lapack,dgesvd_,
[
  AC_DEFINE(HAVE_LAPACKE,1,[ ])
  PHP_ADD_LIBRARY(lapack,,NDARRAY_SHARED_LIBADD)
  AC_MSG_RESULT([LAPACKE detected ])
  CFLAGS+=" -llapack -llapacke "
],[
    AC_MSG_ERROR([wrong LAPACKE version or library not found.])
])

if test "$PHP_NDARRAY" != "no"; then
  dnl Write more examples of tests here...

  dnl Remove this code block if the library does not support pkg-config.
  dnl PKG_CHECK_MODULES([LIBFOO], [foo])
  dnl PHP_EVAL_INCLINE($LIBFOO_CFLAGS)
  dnl PHP_EVAL_LIBLINE($LIBFOO_LIBS, NDARRAY_SHARED_LIBADD)

  dnl If you need to check for a particular library version using PKG_CHECK_MODULES,
  dnl you can use comparison operators. For example:
  dnl PKG_CHECK_MODULES([LIBFOO], [foo >= 1.2.3])
  dnl PKG_CHECK_MODULES([LIBFOO], [foo < 3.4])
  dnl PKG_CHECK_MODULES([LIBFOO], [foo = 1.2.3])

  dnl Remove this code block if the library supports pkg-config.
  dnl --with-ndarray -> check with-path
  dnl SEARCH_PATH="/usr/local /usr"     # you might want to change this
  dnl SEARCH_FOR="/include/ndarray.h"  # you most likely want to change this
  dnl if test -r $PHP_NDARRAY/$SEARCH_FOR; then # path given as parameter
  dnl   NDARRAY_DIR=$PHP_NDARRAY
  dnl else # search default path list
  dnl   AC_MSG_CHECKING([for ndarray files in default path])
  dnl   for i in $SEARCH_PATH ; do
  dnl     if test -r $i/$SEARCH_FOR; then
  dnl       NDARRAY_DIR=$i
  dnl       AC_MSG_RESULT(found in $i)
  dnl     fi
  dnl   done
  dnl fi
  dnl
  dnl if test -z "$NDARRAY_DIR"; then
  dnl   AC_MSG_RESULT([not found])
  dnl   AC_MSG_ERROR([Please reinstall the ndarray distribution])
  dnl fi

  dnl Remove this code block if the library supports pkg-config.
  dnl --with-ndarray -> add include path
  dnl PHP_ADD_INCLUDE($NDARRAY_DIR/include)

  dnl Remove this code block if the library supports pkg-config.
  dnl --with-ndarray -> check for lib and symbol presence
  dnl LIBNAME=NDARRAY # you may want to change this
  dnl LIBSYMBOL=NDARRAY # you most likely want to change this

  dnl If you need to check for a particular library function (e.g. a conditional
  dnl or version-dependent feature) and you are using pkg-config:
  dnl PHP_CHECK_LIBRARY($LIBNAME, $LIBSYMBOL,
  dnl [
  dnl   AC_DEFINE(HAVE_NDARRAY_FEATURE, 1, [ ])
  dnl ],[
  dnl   AC_MSG_ERROR([FEATURE not supported by your ndarray library.])
  dnl ], [
  dnl   $LIBFOO_LIBS
  dnl ])

  dnl If you need to check for a particular library function (e.g. a conditional
  dnl or version-dependent feature) and you are not using pkg-config:
  dnl PHP_CHECK_LIBRARY($LIBNAME, $LIBSYMBOL,
  dnl [
  dnl   PHP_ADD_LIBRARY_WITH_PATH($LIBNAME, $NDARRAY_DIR/$PHP_LIBDIR, NDARRAY_SHARED_LIBADD)
  dnl   AC_DEFINE(HAVE_NDARRAY_FEATURE, 1, [ ])
  dnl ],[
  dnl   AC_MSG_ERROR([FEATURE not supported by your ndarray library.])
  dnl ],[
  dnl   -L$NDARRAY_DIR/$PHP_LIBDIR -lm
  dnl ])
  dnl
  dnl PHP_SUBST(NDARRAY_SHARED_LIBADD)

  dnl In case of no dependencies
  AC_DEFINE(HAVE_NDARRAY, 1, [ Have ndarray support ])
  PHP_NEW_EXTENSION(ndarray,
      numpower.c \
      src/initializers.c \
      src/ndmath/double_math.c \
      src/ndarray.c \
      src/debug.c \
      src/buffer.c \
      src/logic.c \
      src/gpu_alloc.c \
      src/ndmath/linalg.c \
      src/manipulation.c \
      src/iterators.c \
      src/ndarray_ops.c \
      src/indexing.c \
      src/ndmath/arithmetics.c \
      src/types.c,
      $ext_shared)
fi
