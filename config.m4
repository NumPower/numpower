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
          CFLAGS+=" -mavx2 -march=native "
        ],[
          AC_DEFINE(HAVE_AVX2,0,[Have AV2/SSE support])
          AC_MSG_RESULT([AVX2/SSE not found ])
        ], [

        ]
    )
])


if test "$PHP_GD" != "no"; then
    AC_DEFINE(HAVE_GD,1,[Have GD support])
    AC_MSG_RESULT([GD detected ])
    PHP_ADD_EXTENSION_DEP(ndarray, gd, true)
fi

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
      src/indexing.c \
      src/ndmath/arithmetics.c \
      src/ndmath/statistics.c \
      src/types.c,
      $ext_shared)
fi
