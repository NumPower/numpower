cmake_minimum_required(VERSION 3.26)

project(numpower)


set(CMAKE_CXX_STANDARD 14)

include_directories(/usr/include/)
include_directories(/usr/local/include/php/main/)
include_directories(/usr/local/include/php)
include_directories(/usr/local/include)
include_directories(.)
include_directories(src)
include_directories(src/ndmath)
include_directories(src/ndmath/cuda)

add_executable(numpower
        src/ndmath/cuda/cuda_math.cu
        src/ndmath/cuda/cuda_math.h
        src/ndmath/arithmetics.c
        src/ndmath/arithmetics.h
        src/ndmath/double_math.c
        src/ndmath/double_math.h
        src/ndmath/linalg.c
        src/ndmath/linalg.h
        src/ndmath/statistics.c
        src/ndmath/statistics.h
        src/buffer.c
        src/buffer.h
        src/debug.c
        src/debug.h
        src/gd.h
        src/gd_io.h
        src/gpu_alloc.c
        src/gpu_alloc.h
        src/indexing.c
        src/indexing.h
        src/initializers.c
        src/initializers.h
        src/iterators.c
        src/iterators.h
        src/logic.c
        src/logic.h
        src/manipulation.c
        src/manipulation.h
        src/ndarray.c
        src/ndarray.h
        src/types.c
        src/types.h
        config.h
        numpower.c
        numpower_arginfo.h
        php_numpower.h
        src/ndmath/signal.c
        src/ndmath/signal.h
        src/ndmath/calculation.c
        src/ndmath/calculation.h
        src/dnn.c
        src/dnn.h
        src/ndmath/cuda/cuda_dnn.cu
        src/ndmath/cuda/cuda_dnn.cuh
)