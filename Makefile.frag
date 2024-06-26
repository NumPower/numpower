################################## nvcc ###################################

# Location of the CUDA Toolkit
CUDA_PATH ?= /usr/local/cuda

##############################
# start deprecated interface #
##############################
ifeq ($(x86_64),1)
    $(info WARNING - x86_64 variable has been deprecated)
    $(info WARNING - please use TARGET_ARCH=x86_64 instead)
    TARGET_ARCH ?= x86_64
endif
ifeq ($(ARMv7),1)
    $(info WARNING - ARMv7 variable has been deprecated)
    $(info WARNING - please use TARGET_ARCH=armv7l instead)
    TARGET_ARCH ?= armv7l
endif
ifeq ($(aarch64),1)
    $(info WARNING - aarch64 variable has been deprecated)
    $(info WARNING - please use TARGET_ARCH=aarch64 instead)
    TARGET_ARCH ?= aarch64
endif
ifeq ($(ppc64le),1)
    $(info WARNING - ppc64le variable has been deprecated)
    $(info WARNING - please use TARGET_ARCH=ppc64le instead)
    TARGET_ARCH ?= ppc64le
endif
ifneq ($(GCC),)
    $(info WARNING - GCC variable has been deprecated)
    $(info WARNING - please use HOST_COMPILER=$(GCC) instead)
    HOST_COMPILER ?= $(GCC)
endif
ifneq ($(abi),)
    $(error ERROR - abi variable has been removed)
endif
############################
# end deprecated interface #
############################

# architecture
HOST_ARCH   := $(shell uname -m)
TARGET_ARCH ?= $(HOST_ARCH)

# operating system
HOST_OS   := $(shell uname -s 2>/dev/null | tr "[:upper:]" "[:lower:]")
TARGET_OS ?= $(HOST_OS)
HOST_COMPILER ?= g++



# internal flags
NVCCFLAGS   := -m${TARGET_SIZE}
CCFLAGS     :=
LDFLAGS     :=
NVCC     := nvcc -ccbin $(HOST_COMPILER)


#################################### nvcc end ##################################

COMMON_FLAGS = $(DEFS) $(INCLUDES) $(EXTRA_INCLUDES) $(CPPFLAGS) $(PHP_FRAMEWORKPATH)

######################################################

install-cuda:
	rm ./.libs -rf
	mkdir ./.libs
	$(NVCC)  -I. -I  $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS)  $(ALL_CCFLAGS) $(GENCODE_FLAGS)  -c $(builddir)./numpower.c -shared -Xcompiler -fPIC -o .libs/numpower.o
	$(NVCC)  -I. -I $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS)  $(ALL_CCFLAGS) $(GENCODE_FLAGS)  -c $(builddir)./src/buffer.c -shared -Xcompiler -fPIC -o .libs/buffer.o
	$(NVCC)  -I. -I $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS)  $(ALL_CCFLAGS) $(GENCODE_FLAGS)  -c $(builddir)./src/debug.c -shared -Xcompiler -fPIC -o .libs/debug.o
	$(NVCC)  -I. -I $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS)  $(ALL_CCFLAGS) $(GENCODE_FLAGS)  -c $(builddir)./src/indexing.c -shared -Xcompiler -fPIC -o .libs/indexing.o
	$(CC)    -I. -I $(CXX) $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS)  $(ALL_CCFLAGS) $(GENCODE_FLAGS)  -c $(builddir)./src/initializers.c -shared -fPIC -o .libs/initializers.o
	$(NVCC)  -I. -I $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS)  $(ALL_CCFLAGS) $(GENCODE_FLAGS)  -c $(builddir)./src/iterators.c -shared -Xcompiler -fPIC -o .libs/iterators.o
	$(CC)    -I. -I $(CXX) $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS)  $(ALL_CCFLAGS) $(GENCODE_FLAGS)  -c $(builddir)./src/logic.c -shared -fPIC -o .libs/logic.o
	$(CC)    -I. -I $(CXX) $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS)  $(ALL_CCFLAGS) $(GENCODE_FLAGS)  -c $(builddir)./src/manipulation.c -shared -fPIC -o .libs/manipulation.o
	$(CC)    -I. -I $(CXX) $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS)  $(ALL_CCFLAGS) $(GENCODE_FLAGS)  -c $(builddir)./src/ndarray.c -shared -fPIC -o .libs/ndarray.o
	$(NVCC)  -I. -I $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS)  $(ALL_CCFLAGS) $(GENCODE_FLAGS)  -c $(builddir)./src/types.c -shared -Xcompiler -fPIC -o .libs/types.o
	$(CC)    -I. -I $(CXX) $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS)  $(ALL_CCFLAGS) $(GENCODE_FLAGS)  -c $(builddir)./src/ndmath/arithmetics.c -shared -fPIC -o .libs/arithmetics.o
	$(NVCC)  -I. -I $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS)  $(ALL_CCFLAGS) $(GENCODE_FLAGS)  -c $(builddir)./src/ndmath/double_math.c -shared -Xcompiler -fPIC -o .libs/double_math.o
	$(CC)    -I. -I $(CXX) $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS)  $(ALL_CCFLAGS) $(GENCODE_FLAGS)  -c $(builddir)./src/ndmath/linalg.c -shared -fPIC -o .libs/linalg.o
	$(CC)    -I. -I $(CXX) $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS)  $(ALL_CCFLAGS) $(GENCODE_FLAGS)  -c $(builddir)./src/ndmath/signal.c -shared -fPIC -o .libs/signal.o
	$(CC)    -I. -I $(CXX) $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS)  $(ALL_CCFLAGS) $(GENCODE_FLAGS)  -c $(builddir)./src/ndmath/calculation.c -shared -fPIC -o .libs/calculation.o
	$(CC)    -I. -I $(CXX) $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS)  $(ALL_CCFLAGS) $(GENCODE_FLAGS)  -c $(builddir)./src/dnn.c -shared -fPIC -o .libs/dnn.o
	$(NVCC)  -I. -I $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS)  $(ALL_CCFLAGS) $(GENCODE_FLAGS)  -c $(builddir)./src/gpu_alloc.c -shared -Xcompiler -fPIC -o .libs/gpu_alloc.o
	$(NVCC)  -I. -I $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS)  $(ALL_CCFLAGS) $(GENCODE_FLAGS)  -c $(builddir)./src/ndmath/cuda/cuda_math.cu -shared -Xcompiler -fPIC -o .libs/cuda_math.o
	$(NVCC)  -I. -I $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS)  $(ALL_CCFLAGS) $(GENCODE_FLAGS)  -c $(builddir)./src/ndmath/cuda/cuda_dnn.cu -shared -Xcompiler -fPIC -o .libs/cuda_dnn.o
	$(NVCC)  -I. -I $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS)  $(ALL_CCFLAGS) $(GENCODE_FLAGS)  -c $(builddir)./src/ndmath/statistics.c -shared -Xcompiler -fPIC -o .libs/statistics.o
	$(NVCC)  -shared .libs/numpower.o .libs/signal.o .libs/initializers.o .libs/double_math.o .libs/ndarray.o .libs/debug.o .libs/statistics.o .libs/calculation.o .libs/buffer.o .libs/dnn.o .libs/cuda_dnn.o .libs/logic.o .libs/gpu_alloc.o .libs/linalg.o .libs/manipulation.o .libs/iterators.o .libs/indexing.o .libs/arithmetics.o .libs/types.o  .libs/cuda_math.o $(CFLAGS_CLEAN) -o .libs/ndarray.so
	cp ./.libs/ndarray.so $(phplibdir)/ndarray.so
	cp ./.libs/ndarray.so $(EXTENSION_DIR)/ndarray.so


