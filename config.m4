PHP_ARG_ENABLE([ndarray],
  [whether to enable ndarray support],
  [AS_HELP_STRING([--enable-ndarray],
    [Enable ndarray support])],
  [no])

PHP_ARG_WITH(cuda, for CUDA support,
[  --with-cuda           Include CUDA support], [no], [no])

if test "$PHP_CUDA" != "no"; then
    PHP_CHECK_LIBRARY(cublas,cublasDgemm,
    [
      AC_DEFINE(HAVE_CUBLAS,1,[ ])
      PHP_ADD_LIBRARY(cublas,,NDARRAY_SHARED_LIBADD)
      AC_MSG_RESULT([CUBLAS detected ])
      PHP_ADD_MAKEFILE_FRAGMENT($abs_srcdir/Makefile.frag, $abs_builddir)
      CFLAGS+=" -lcublas -lcudart "
      AC_CHECK_HEADER([immintrin.h],
              [
                AC_DEFINE(HAVE_AVX2,1,[Have AV2/SSE support])
                AC_MSG_RESULT([AVX2/SSE detected ])
                CXX+=" -mavx2 -march=native "
              ],[
                AC_DEFINE(HAVE_AVX2,0,[Have AV2/SSE support])
                AC_MSG_RESULT([AVX2/SSE not found ])
              ], [

              ]
          )
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
else
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
fi


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

PHP_CHECK_LIBRARY(mkl_rt,LAPACKE_sgesdd,
    [
      AC_DEFINE(HAVE_LAPACKE_MKL,1,[ ])
      PHP_ADD_LIBRARY(lapack,,NDARRAY_SHARED_LIBADD)
      AC_MSG_RESULT([LAPACKE (MKL) detected ])
      CFLAGS+=" -lmkl_rt "
    ],[
    PHP_CHECK_LIBRARY(lapacke,LAPACKE_sgesdd,
    [
      AC_DEFINE(HAVE_LAPACKE,1,[ ])
      PHP_ADD_LIBRARY(lapack,,NDARRAY_SHARED_LIBADD)
      AC_MSG_RESULT([LAPACKE detected ])
      CFLAGS+=" -llapack -llapacke "
    ],[
        AC_MSG_ERROR([wrong LAPACKE version or library not found. Try `apt install liblapacke-dev`])
    ])
])


PHP_CHECK_LIBRARY(cudnn, cudnnCreate,
    [
      AC_DEFINE(HAVE_CUDNN,1,[ ])
      PHP_ADD_LIBRARY(z,,NDARRAY_SHARED_LIBADD)
      AC_MSG_RESULT([cuDNN detected, enabling GPU DNN capabilities.])
      CFLAGS+=" -lz -lcudnn "
    ],[
    AC_MSG_RESULT([cuDNN not found. GPU DNN capabilities disabled.])
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
      src/dnn.c \
      src/iterators.c \
      src/indexing.c \
      src/ndmath/arithmetics.c \
      src/ndmath/calculation.c \
      src/ndmath/statistics.c \
      src/ndmath/signal.c \
      src/types.c,
      $ext_shared)
fi
