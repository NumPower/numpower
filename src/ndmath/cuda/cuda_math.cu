#include "cuda_math.h"
#include <cuda_runtime.h>
#include "../../ndarray.h"
#include "../../initializers.h"
#include "../../debug.h"
#include <float.h>
#include <cusolverDn.h>

#define CHECK_CUDA(func) do { \
  cudaError_t status = (func); \
  if (status != cudaSuccess) { \
    printf("CUDA API failed at line %d with error: %s\n", \
           __LINE__, cudaGetErrorString(status)); \
    return EXIT_FAILURE; \
  } \
} while (0)

#define CHECK_CUSOLVER(func) do { \
  cusolverStatus_t status = (func); \
  if (status != CUSOLVER_STATUS_SUCCESS) { \
    printf("cuSOLVER API failed at line %d with error: %d\n", \
           __LINE__, status); \
    return EXIT_FAILURE; \
  } \
} while (0)

#define CHECK_CUBLAS(func) do { \
  cusolverStatus_t status = (func); \
  if (status != CUSOLVER_STATUS_SUCCESS) { \
    printf("cuBLAS API failed at line %d with error: %d\n", \
           __LINE__, status); \
    return EXIT_FAILURE; \
  } \
} while (0)

__device__
float sinc(float number) {
    if (number == 0.0) {
        return 1.0;
    } else {
        return sinf(M_PI * number) / (M_PI * number);
    }
}

__global__
void sincFloatKernel(float* d_array, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {

        d_array[index] = sinc(d_array[index]);
    }
}

__global__
void truncFloatKernel(float* d_array, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        d_array[index] = truncf(d_array[index]);
    }
}

__device__
int roundFloatToNearestInt(float number) {
    float rounded = rintf(number);
    int floorInt = (int)floorf(number);

    // Check if the rounded value is halfway between two integers
    if (rounded - floorInt == 0.5 && ((int)rounded % 2 != 0)) {
        rounded -= 1.0;
    }

    return (int)rounded;
}

__global__
void rintFloatKernel(float* d_array, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        d_array[index] = roundFloatToNearestInt(d_array[index]);
    }
}

__global__
void fixFloatKernel(float* d_array, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        d_array[index] = truncf(d_array[index]);
    }
}

__global__
void ceilFloatKernel(float* d_array, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        d_array[index] = ceilf(d_array[index]);
    }
}

__global__
void floorFloatKernel(float* d_array, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        d_array[index] = floorf(d_array[index]);
    }
}

__global__
void arcsinhFloatKernel(float* d_array, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        d_array[index] = asinhf(d_array[index]);
    }
}

__global__
void arccoshFloatKernel(float* d_array, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        d_array[index] = acoshf(d_array[index]);
    }
}

__global__
void arctanhFloatKernel(float* d_array, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        d_array[index] = atanhf(d_array[index]);
    }
}

__global__
void sinhFloatKernel(float* d_array, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        d_array[index] = sinhf(d_array[index]);
    }
}

__global__
void coshFloatKernel(float* d_array, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        d_array[index] = coshf(d_array[index]);
    }
}

__global__
void tanhFloatKernel(float* d_array, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        d_array[index] = tanhf(d_array[index]);
    }
}

__global__
void degreesFloatKernel(float* d_array, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        d_array[index] = d_array[index] * (180.0 / 3.1415926535);
    }
}

__global__
void radiansFloatKernel(float* d_array, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        d_array[index] = d_array[index] * (3.1415926535 / 180.0);
    }
}

__global__
void arcsinFloatKernel(float* d_array, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        d_array[index] = asinf(d_array[index]);
    }
}

__global__
void arccosFloatKernel(float* d_array, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        d_array[index] = acosf(d_array[index]);
    }
}

__global__
void arctanFloatKernel(float* d_array, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        d_array[index] = atanf(d_array[index]);
    }
}

__global__
void absFloatKernel(float* d_array, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        d_array[index] = fabsf(d_array[index]);
    }
}

__global__
void sinFloatKernel(float* d_array, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        d_array[index] = sinf(d_array[index]);
    }
}

__global__
void cosFloatKernel(float* d_array, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        d_array[index] = cosf(d_array[index]);
    }
}

__global__
void tanFloatKernel(float* d_array, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        d_array[index] = tanf(d_array[index]);
    }
}

__global__
void expm1FloatKernel(float* d_array, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        d_array[index] = expm1f(d_array[index]);
    }
}

__global__
void expFloatKernel(float* d_array, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        d_array[index] = expf(d_array[index]);
    }
}

__global__
void sqrtFloatKernel(float* d_array, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        d_array[index] = sqrtf(d_array[index]);
    }
}

__global__
void logFloatKernel(float* d_array, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        d_array[index] = logf(d_array[index]);
    }
}

__global__
void logbFloatKernel(float* d_array, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        d_array[index] = logbf(d_array[index]);
    }
}

__global__
void log2FloatKernel(float* d_array, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        d_array[index] = log2f(d_array[index]);
    }
}

__global__
void log1pFloatKernel(float* d_array, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        d_array[index] = log1pf(d_array[index]);
    }
}

__global__
void log10FloatKernel(float* d_array, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        d_array[index] = log10f(d_array[index]);
    }
}

__global__ void
add_vectors_float_kernel(float *a, float *b, float *result, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        result[index] = a[index] + b[index];
    }
}

__global__ void
subtract_vectors_float_kernel(float *a, float *b, float *result, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        result[index] = a[index] - b[index];
    }
}

__global__ void
divide_vectors_float_kernel(float *a, float *b, float *result, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        result[index] = a[index] / b[index];
    }
}

__global__ void
multiply_vectors_float_kernel(float *a, float *b, float *result, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        result[index] = a[index] * b[index];
    }
}

__global__ void
fmodf_float_kernel(float *a, float *b, float *result, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        result[index] = fmodf(a[index], b[index]);
    }
}

__global__ void
pow_float_kernel(float *a, float *b, float *result, int size) {
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if (index < size) {
        result[index] = powf(a[index], b[index]);
    }
}

__global__ void
max_reduce_naive(float * d_out, float * d_in, int n) {
    extern __shared__ float sdata[];

    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int tid = threadIdx.x;

    // load shared mem from global mem
    sdata[tid] = d_in[myId];
    __syncthreads();            // make sure entire block is loaded!

    // do reduction in shared mem
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();        // make sure all adds at one stage are done!
    }

    // only thread 0 writes result for this block back to global mem
    if (tid == 0) {
        d_out[blockIdx.x] = sdata[0];
    }
}

__global__ void
min_reduce_naive(float * d_out, float * d_in, int n) {
    extern __shared__ float sdata[];

    int myId = threadIdx.x + blockDim.x * blockIdx.x;
    int tid = threadIdx.x;

    // load shared mem from global mem
    sdata[tid] = d_in[myId];
    __syncthreads();            // make sure entire block is loaded!

    // do reduction in shared mem
    for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = fminf(sdata[tid], sdata[tid + s]);
        }
        __syncthreads();        // make sure all adds at one stage are done!
    }

    // only thread 0 writes result for this block back to global mem
    if (tid == 0) {
        d_out[blockIdx.x] = sdata[0];
    }
}

__global__
void array_equals_float(float *a, float *b, int *result, int n) {
    int idx = threadIdx.x + blockDim.x * blockIdx.x;
    if (idx < n) {
        if (a[idx] != b[idx]) {
            atomicExch(result, 0); // If any element is not equal, set 'equal' to 0
        }
    }
}

__global__
void array_sum_float(float *a, float *result, int n) {
    extern __shared__ float sdata[];

    // each thread loads one element from global to shared mem
    unsigned int tid = threadIdx.x;
    unsigned int i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;

    float x = 0;
    if (i < n) x += a[i];
    if (i + blockDim.x < n) x += a[i + blockDim.x];
    sdata[tid] = x;
    __syncthreads();

    // do reduction in shared mem
    for (unsigned int s=blockDim.x/2; s>0; s>>=1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // write result for this block to global mem
    if (tid == 0) atomicAdd(result, sdata[0]);
}

// CUDA Kernel for Matrix Multiplication for non-square matrices
__global__ void
matmul_float_kernel(float* A, float* B, float* C, int widthA, int heightA, int widthB) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < heightA && col < widthB) {
        float sum = 0;
        for(int i = 0; i < widthA; ++i) {
            sum += A[row * widthA + i] * B[i * widthB + col];
        }
        C[row * widthB + col] = sum;
    }
}

__global__
void fill_float_kernel(float* array, int n, float value) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if(idx < n) {
        array[idx] = value;
    }
}

extern "C" {

    int
    cuda_det_float(float *a, float *result, int n) {
        int N = n;
        float *d_A = a;
        cusolverDnHandle_t cusolverH = NULL;
        cudaStream_t stream = NULL;
        cublasHandle_t cublasH = NULL;
        cusolverStatus_t cusolver_status = CUSOLVER_STATUS_SUCCESS;

        CHECK_CUSOLVER(cusolverDnCreate(&cusolverH));
        CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CHECK_CUSOLVER(cusolverDnSetStream(cusolverH, stream));
        cublasCreate(&cublasH);
        cublasSetStream(cublasH, stream);

        int* d_Ipiv; // pivot array
        int* d_info;  // info on success or failure
        float* d_U; // U matrix of LU decomposition

        CHECK_CUDA(cudaMalloc(&d_Ipiv, N*sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_info, sizeof(int)));
        CHECK_CUDA(cudaMalloc(&d_U, N*N*sizeof(float)));

        // copy A to U as cusolverDnSgetrf works in place
        CHECK_CUDA(cudaMemcpy(d_U, d_A, N*N*sizeof(float), cudaMemcpyDeviceToDevice));

        // LU decompose
        cusolver_status = cusolverDnSgetrf(cusolverH, N, N, d_U, N, NULL, d_Ipiv, d_info);
        if (cusolver_status != CUSOLVER_STATUS_SUCCESS) {
            // handle error
            printf("LU decomposition failed\n");
            exit(1);
        }

        // Find determinant by product of diagonal elements
        float det = 1.0f;
        for (int i = 0; i < N; i++) {
            float elem;
            CHECK_CUDA(cudaMemcpy(&elem, d_U + i * N + i, sizeof(float), cudaMemcpyDeviceToHost));
            // Check for potential overflow
            if (fabsf(elem) > FLT_MAX / fabsf(det)) {
                // Handle overflow here, e.g., return a special value or throw an error
                printf("Overflow detected in det\n");
                exit(1);
            }
            if (!isnan(elem) && !isinf(elem)) {
                det *= elem;
            }
        }

        // Analyze pivot array to calculate number of permutations
        int* h_Ipiv = new int[N];
        CHECK_CUDA(cudaMemcpy(h_Ipiv, d_Ipiv, N*sizeof(int), cudaMemcpyDeviceToHost));

        int numPermutations = 0;
        for(int i = 0; i < N; i++) {
            if(i+1 != h_Ipiv[i]) numPermutations++;
        }

        if(numPermutations % 2 != 0) det = -det;

        // Cleanup
        if (d_U) cudaFree(d_U);
        if (d_Ipiv) cudaFree(d_Ipiv);
        if (d_info) cudaFree(d_info);
        if (cublasH) cublasDestroy(cublasH);
        if (cusolverH) cusolverDnDestroy(cusolverH);
        if (stream) cudaStreamDestroy(stream);

        CHECK_CUDA(cudaMemcpy(result, &det, sizeof(float), cudaMemcpyHostToDevice));
        return 1;
    }

    void
    cuda_fill_float(float *a, float value, int n) {
        int blockSize = 256;
        int gridSize = (n + blockSize - 1) / blockSize;

        fill_float_kernel<<<gridSize, blockSize>>>(a, n, value);
        cudaDeviceSynchronize();
    }

    void
    cuda_matmul_float(int nblocks, float *a, float *b, float *rtn, int widthA, int heightA, int widthB) {
        dim3 blockSize(16, 16); // Use a block size appropriate for your hardware
        dim3 gridSize((widthB + blockSize.x - 1) / blockSize.x, (heightA + blockSize.y - 1) / blockSize.y);

        matmul_float_kernel<<<gridSize, blockSize>>>(a, b, rtn, widthA, heightA, widthB);
        cudaDeviceSynchronize();
    }

    void
    cuda_sum_float(int nblocks, float *a, float *rtn, int nelements) {
        float *d_sum;
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize * 2 - 1) / (blockSize * 2);  // Number of blocks in the grid.
        cudaMalloc((void **) &d_sum, sizeof(float));

        cudaMemcpy(d_sum, rtn, sizeof(float), cudaMemcpyHostToDevice);
        array_sum_float<<<numBlocks, blockSize, blockSize * sizeof(float)>>>(a, d_sum, nelements);
        cudaMemcpy(rtn, d_sum, sizeof(float), cudaMemcpyDeviceToHost);
        cudaDeviceSynchronize();
    }

    int
    cuda_svd_float(float *d_A, float *d_U, float *d_V, float *d_S, int m, int n) {
        cusolverDnHandle_t cusolverH = NULL;  // cuSOLVER handle
        cudaStream_t stream = NULL;  // CUDA stream
        gesvdjInfo_t gesvdj_params = NULL;  // configuration of gesvdj
        CHECK_CUSOLVER(cusolverDnCreate(&cusolverH));
        CHECK_CUDA(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        CHECK_CUSOLVER(cusolverDnSetStream(cusolverH, stream));
        CHECK_CUSOLVER(cusolverDnCreateGesvdjInfo(&gesvdj_params));

        // Set desired configuration of gesvdj
        CHECK_CUSOLVER(cusolverDnXgesvdjSetTolerance(
                gesvdj_params,
                1.e-7));
        CHECK_CUSOLVER(cusolverDnXgesvdjSetMaxSweeps(
                gesvdj_params,
                15));

        // Perform SVD
        // Note: This is just a skeleton code. Please handle CUDA errors appropriately
        int* devInfo = NULL;  // info on gesvdj convergence
        CHECK_CUDA(cudaMalloc((void**)&devInfo, sizeof(int)));
        int lwork = 0;
        float *d_work = NULL;
        CHECK_CUSOLVER(cusolverDnSgesvdj_bufferSize(
                cusolverH,
                CUSOLVER_EIG_MODE_VECTOR,  // compute eigenvectors
                0,  // number of singular values to compute, 0 for all
                m,
                n,
                d_A,
                m,  // leading dimension of A
                d_S,
                d_U,
                m,  // leading dimension of U
                d_V,
                n,  // leading dimension of V
                &lwork,
                gesvdj_params));

        CHECK_CUDA(cudaMalloc((void**)&d_work , sizeof(float) * lwork));
        CHECK_CUSOLVER(cusolverDnSgesvdj(
                cusolverH,
                CUSOLVER_EIG_MODE_VECTOR,  // compute eigenvectors
                0,  // number of singular values to compute, 0 for all
                m,
                n,
                d_A,
                m,  // leading dimension of A
                d_S,
                d_U,
                m,  // leading dimension of U
                d_V,
                n,  // leading dimension of V
                d_work,
                lwork,
                devInfo,
                gesvdj_params));

        // Synchronize to ensure computation is finished
        CHECK_CUDA(cudaDeviceSynchronize());
        if (devInfo) CHECK_CUDA(cudaFree(devInfo));
        if (cusolverH) CHECK_CUSOLVER(cusolverDnDestroy(cusolverH));
        if (stream) CHECK_CUDA(cudaStreamDestroy(stream));
        if (gesvdj_params) CHECK_CUSOLVER(cusolverDnDestroyGesvdjInfo(gesvdj_params));

        return 1;
    }

    float
    cuda_max_float(float *a, int nelements) {
        int size = nelements;
        float *d_out;
        int blockSize = 256;  // Number of threads per block. This is a typical choice.

        int current_size = size;
        float *d_current_in = a;
        while(current_size > 1) {
            int blocks = (current_size + blockSize - 1) / blockSize;
            cudaMalloc((void **) &d_out, blocks * sizeof(float));
            max_reduce_naive<<<blocks, blockSize, blockSize * sizeof(float)>>>(d_out, d_current_in, current_size);

            if (d_current_in != a) { // Free the intermediate input arrays
                cudaFree(d_current_in);
            }

            // Prepare for the next iteration
            d_current_in = d_out;
            current_size = blocks;
        }
        cudaDeviceSynchronize();

        // copy the result back to the host
        float max_value;
        cudaMemcpy(&max_value, d_out, sizeof(float), cudaMemcpyDeviceToHost);

        return max_value;
    }

    float
    cuda_min_float(float *a, int nelements) {
        int size = nelements;
        float *d_out;
        int blockSize = 256;  // Number of threads per block. This is a typical choice.

        int current_size = size;
        float *d_current_in = a;
        while(current_size > 1) {
            int blocks = (current_size + blockSize - 1) / blockSize;
            cudaMalloc((void **) &d_out, blocks * sizeof(float));
            min_reduce_naive<<<blocks, blockSize, blockSize * sizeof(float)>>>(d_out, d_current_in, current_size);

            if (d_current_in != a) { // Free the intermediate input arrays
                cudaFree(d_current_in);
            }

            // Prepare for the next iteration
            d_current_in = d_out;
            current_size = blocks;
        }
        cudaDeviceSynchronize();

        // copy the result back to the host
        float min_value;
        cudaMemcpy(&min_value, d_out, sizeof(float), cudaMemcpyDeviceToHost);

        return min_value;
    }

    int
    cuda_equal_float(int nblocks, float *a, float *b, int nelements) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int result = 1;
        int *d_equal;
        // Allocate GPU memory for the result
        cudaMalloc(&d_equal, sizeof(int));
        cudaMemcpy(d_equal, &result, sizeof(int), cudaMemcpyHostToDevice);
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        array_equals_float<<<numBlocks, blockSize>>>(a, b, d_equal, nelements);
        cudaDeviceSynchronize();
        cudaMemcpy(&result, d_equal, sizeof(int), cudaMemcpyDeviceToHost);
        cudaFree(d_equal);
        return result;
    }

    void
    cuda_pow_float(int nblocks, float *a, float *b, float *rtn, int nelements) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        pow_float_kernel<<<numBlocks, blockSize>>>(a, b, rtn, nelements);
        cudaDeviceSynchronize();
    }

    void
    cuda_mod_float(int nblocks, float *a, float *b, float *rtn, int nelements) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        fmodf_float_kernel<<<numBlocks, blockSize>>>(a, b, rtn, nelements);
        cudaDeviceSynchronize();
    }

    void
    cuda_multiply_float(int nblocks, float *a, float *b, float *rtn, int nelements) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        multiply_vectors_float_kernel<<<numBlocks, blockSize>>>(a, b, rtn, nelements);
        cudaDeviceSynchronize();
    }

    void
    cuda_divide_float(int nblocks, float *a, float *b, float *rtn, int nelements) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        divide_vectors_float_kernel<<<numBlocks, blockSize>>>(a, b, rtn, nelements);
        cudaDeviceSynchronize();
    }

    void
    cuda_subtract_float(int nblocks, float *a, float *b, float *rtn, int nelements) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        subtract_vectors_float_kernel<<<numBlocks, blockSize>>>(a, b, rtn, nelements);
        cudaDeviceSynchronize();
    }

    void
    cuda_add_float(int nblocks, float *a, float *b, float *rtn, int nelements) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        add_vectors_float_kernel<<<numBlocks, blockSize>>>(a, b, rtn, nelements);
        cudaDeviceSynchronize();
    }

    void
    cuda_float_log(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        logFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
        cudaDeviceSynchronize();
    }

    void
    cuda_float_logb(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        logbFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
        cudaDeviceSynchronize();
    }

    void
    cuda_float_log2(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        log2FloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
        cudaDeviceSynchronize();
    }

    void
    cuda_float_log1p(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        log1pFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
        cudaDeviceSynchronize();
    }

    void
    cuda_float_log10(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        log10FloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
        cudaDeviceSynchronize();
    }

    void
    cuda_float_sqrt(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        sqrtFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
        cudaDeviceSynchronize();
    }

    void
    cuda_float_exp(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        expFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
        cudaDeviceSynchronize();
    }

    void
    cuda_float_abs(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        absFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
        cudaDeviceSynchronize();
    }

    void
    cuda_float_expm1(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        expm1FloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
        cudaDeviceSynchronize();
    }

    void
    cuda_float_sin(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        sinFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
        cudaDeviceSynchronize();
    }

    void
    cuda_float_cos(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        cosFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
        cudaDeviceSynchronize();
    }

    void
    cuda_float_tan(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        tanFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
        cudaDeviceSynchronize();
    }

    void
    cuda_float_arcsin(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        arcsinFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
        cudaDeviceSynchronize();
    }

    void
    cuda_float_arctan(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        arctanFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
        cudaDeviceSynchronize();
    }

    void
    cuda_float_arccos(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        arccosFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
        cudaDeviceSynchronize();
    }

    void
    cuda_float_radians(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        radiansFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
        cudaDeviceSynchronize();
    }

    void
    cuda_float_degrees(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        degreesFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
        cudaDeviceSynchronize();
    }

    void
    cuda_float_sinh(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        sinhFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
        cudaDeviceSynchronize();
    }

    void
    cuda_float_cosh(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        coshFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
        cudaDeviceSynchronize();
    }

    void
    cuda_float_tanh(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        tanhFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
        cudaDeviceSynchronize();
    }

    void
    cuda_float_arcsinh(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        arcsinhFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
        cudaDeviceSynchronize();
    }

    void
    cuda_float_arccosh(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        arccoshFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
        cudaDeviceSynchronize();
    }

    void
    cuda_float_arctanh(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        arctanhFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
        cudaDeviceSynchronize();
    }

    void
    cuda_float_rint(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        rintFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
        cudaDeviceSynchronize();
    }

    void
    cuda_float_fix(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        fixFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
        cudaDeviceSynchronize();
    }

    void
    cuda_float_ceil(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        ceilFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
        cudaDeviceSynchronize();
    }

    void
    cuda_float_floor(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        floorFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
        cudaDeviceSynchronize();
    }

    void
    cuda_float_trunc(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        truncFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
        cudaDeviceSynchronize();
    }

    void
    cuda_float_sinc(int nblocks, float *d_array) {
        int blockSize = 256;  // Number of threads per block. This is a typical choice.
        int numBlocks = (nblocks + blockSize - 1) / blockSize;  // Number of blocks in the grid.
        sincFloatKernel<<<numBlocks, blockSize>>>(d_array, nblocks);
        cudaDeviceSynchronize();
    }

    NDArray*
    NDArrayMathGPU_ElementWise(NDArray* ndarray, ElementWiseFloatGPUOperation op) {
        NDArray *rtn = NDArray_Copy(ndarray, NDArray_DEVICE(ndarray));
        op(NDArray_NUMELEMENTS(rtn), NDArray_FDATA(rtn));
        return rtn;
    }

}