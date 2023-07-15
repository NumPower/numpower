

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
	$(NVCC) -I. -I  $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS)  $(ALL_CCFLAGS) $(GENCODE_FLAGS)  -c $(builddir)./numpower.c -shared -Xcompiler -fPIC -o .libs/numpower.o
	$(NVCC)  -I. -I $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS)  $(ALL_CCFLAGS) $(GENCODE_FLAGS)  -c $(builddir)./src/buffer.c -shared -Xcompiler -fPIC -o .libs/buffer.o
	$(NVCC)  -I. -I $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS)  $(ALL_CCFLAGS) $(GENCODE_FLAGS)  -c $(builddir)./src/debug.c -shared -Xcompiler -fPIC -o .libs/debug.o
	$(NVCC)  -I. -I $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS)  $(ALL_CCFLAGS) $(GENCODE_FLAGS)  -c $(builddir)./src/indexing.c -shared -Xcompiler -fPIC -o .libs/indexing.o
	$(NVCC)  -I. -I $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS)  $(ALL_CCFLAGS) $(GENCODE_FLAGS)  -c $(builddir)./src/initializers.c -shared -Xcompiler -fPIC -o .libs/initializers.o
	$(NVCC)  -I. -I $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS)  $(ALL_CCFLAGS) $(GENCODE_FLAGS)  -c $(builddir)./src/iterators.c -shared -Xcompiler -fPIC -o .libs/iterators.o
	$(NVCC)  -I. -I $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS)  $(ALL_CCFLAGS) $(GENCODE_FLAGS)  -c $(builddir)./src/logic.c -shared -Xcompiler -fPIC -o .libs/logic.o
	$(NVCC)  -I. -I $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS)  $(ALL_CCFLAGS) $(GENCODE_FLAGS)  -c $(builddir)./src/manipulation.c -shared -Xcompiler -fPIC -o .libs/manipulation.o
	$(NVCC)  -I. -I $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS)  $(ALL_CCFLAGS) $(GENCODE_FLAGS)  -c $(builddir)./src/ndarray.c -shared -Xcompiler -fPIC -o .libs/ndarray.o
	$(NVCC)  -I. -I $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS)  $(ALL_CCFLAGS) $(GENCODE_FLAGS)  -c $(builddir)./src/types.c -shared -Xcompiler -fPIC -o .libs/types.o
	$(NVCC)  -I. -I $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS)  $(ALL_CCFLAGS) $(GENCODE_FLAGS)  -c $(builddir)./src/ndmath/arithmetics.c -shared -Xcompiler -fPIC -o .libs/arithmetics.o
	$(NVCC)  -I. -I $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS)  $(ALL_CCFLAGS) $(GENCODE_FLAGS)  -c $(builddir)./src/ndmath/double_math.c -shared -Xcompiler -fPIC -o .libs/double_math.o
	$(NVCC)  -I. -I $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS)  $(ALL_CCFLAGS) $(GENCODE_FLAGS)  -c $(builddir)./src/ndmath/linalg.c -shared -Xcompiler -fPIC -o .libs/linalg.o
	$(NVCC)  -I. -I $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS)  $(ALL_CCFLAGS) $(GENCODE_FLAGS)  -c $(builddir)./src/gpu_alloc.c -shared -Xcompiler -fPIC -o .libs/gpu_alloc.o
	$(NVCC)  -I. -I $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS)  $(ALL_CCFLAGS) $(GENCODE_FLAGS)  -c $(builddir)./src/ndmath/cuda/cuda_math.cu -shared -Xcompiler -fPIC -o .libs/cuda_math.o
	$(NVCC)  -I. -I $(COMMON_FLAGS) $(CFLAGS_CLEAN) $(EXTRA_CFLAGS)  $(ALL_CCFLAGS) $(GENCODE_FLAGS)  -c $(builddir)./src/ndmath/statistics.c -shared -Xcompiler -fPIC -o .libs/statistics.o
	$(NVCC)  -shared .libs/numpower.o .libs/initializers.o .libs/double_math.o .libs/ndarray.o .libs/debug.o .libs/statistics.o .libs/buffer.o .libs/logic.o .libs/gpu_alloc.o .libs/linalg.o .libs/manipulation.o .libs/iterators.o .libs/indexing.o .libs/arithmetics.o .libs/types.o  .libs/cuda_math.o -lcublas -lcusolver -lcudart -llapack -llapacke -lopenblas -lpthread -lcuda  -o .libs/ndarray.so
	cp ./.libs/ndarray.so $(phplibdir)/ndarray.so
	cp ./.libs/ndarray.so $(EXTENSION_DIR)/ndarray.so

temp:
	echo $(COMMON_FLAGS)
	echo $(CFLAGS_CLEAN)
	echo $(EXTRA_CFLAGS)
	echo $(GENCODE_FLAGS)
	echo $(INCLUDES)