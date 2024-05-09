#include <php.h>
#include "Zend/zend_alloc.h"
#include "Zend/zend_API.h"
#include "linalg.h"
#include "../../config.h"
#include "../initializers.h"
#include "../types.h"
#include "../manipulation.h"
#include "arithmetics.h"
#include "../iterators.h"
#include "../gpu_alloc.h"
#include "../indexing.h"

#ifdef HAVE_LAPACKE
#include <lapacke.h>
#endif

#ifdef HAVE_CBLAS
#include <cblas.h>
#endif

#ifdef HAVE_CUBLAS
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include "cuda/cuda_math.h"
#include "../gpu_alloc.h"
#endif

#ifdef HAVE_AVX2
#include <immintrin.h>
#endif

/**
 * Double type (float64) matmul
 *
 * @param a
 * @param b
 * @return
 */
NDArray*
NDArray_FMatmul(NDArray *a, NDArray *b) {
    int* output_shape = emalloc(sizeof(int) * 2);
    output_shape[0] = NDArray_SHAPE(a)[0];
    output_shape[1] = NDArray_SHAPE(b)[1];

    NDArray* result = NDArray_Zeros(output_shape, 2, NDARRAY_TYPE_FLOAT32, NDArray_DEVICE(a));

    if (NDArray_DEVICE(a) == NDARRAY_DEVICE_GPU) {
        // Perform GPU matrix multiplication
#ifdef HAVE_CUBLAS
        NDArray* result_gpu = NDArray_ToGPU(result);
        NDArray_FREE(result);
        cuda_matmul_float(NDArray_NUMELEMENTS(a), NDArray_FDATA(a), NDArray_FDATA(b), NDArray_FDATA(result_gpu),
                          NDArray_SHAPE(a)[1], NDArray_SHAPE(a)[0], NDArray_SHAPE(b)[1]);
        return result_gpu;
#endif
    } else {
        // Perform CPU matrix multiplication
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    NDArray_SHAPE(a)[0], NDArray_SHAPE(b)[1], NDArray_SHAPE(a)[1],
                    1.0f, NDArray_FDATA(a), NDArray_SHAPE(a)[1],
                    NDArray_FDATA(b), NDArray_SHAPE(b)[1],
                    0.0f, NDArray_FDATA(result), NDArray_SHAPE(b)[1]);
    }
    return result;
}

void
computeSVDFloat(float* A, int m, int n, float* U, float* S, float* V) {
    int lda = n;
    int ldu = m;
    int ldvt = n;

    int info;

    info = LAPACKE_sgesdd(LAPACK_ROW_MAJOR, 'S', m, n, A, lda, S, U, ldu, V, ldvt);

    if (info > 0) {
        printf("SVD computation failed.\n");
        return;
    }

}

#ifdef HAVE_CUBLAS
void
computeSVDFloatGPU(float* A, int m, int n, float* U, float* S, float* V) {
    cuda_svd_float(A, U, V, S, m, n);
}
#endif

/**
 * @return
 */
NDArray**
NDArray_SVD(NDArray *target) {
    NDArray **rtns;
    NDArray *target_ptr = target;
    NDArray *rtn_s, *rtn_u, *rtn_v;
    double *U, *S, *V;
    float *output_data;
    float  *Uf, *Sf, *Vf;
    int *U_shape, *S_shape, *V_shape;
    int smallest_dim = -1;

    if (NDArray_NDIM(target) == 1) {
        zend_throw_error(NULL, "Array must be at least two-dimensional");
        return NULL;
    }

    rtns = emalloc(sizeof(NDArray*) * 3);

    for (int i = 0; i < NDArray_NDIM(target_ptr); i++) {
        if (smallest_dim == -1) {
            smallest_dim = NDArray_SHAPE(target_ptr)[i];
            continue;
        }
        if (smallest_dim > NDArray_SHAPE(target_ptr)[i]) {
            smallest_dim = NDArray_SHAPE(target_ptr)[i];
        }
    }
    if(NDArray_DEVICE(target_ptr) == NDARRAY_DEVICE_GPU) {
#ifdef HAVE_CUBLAS
        target_ptr = NDArray_Transpose(target, NULL);
        NDArray_VMALLOC((void**)&Sf, sizeof(float) * smallest_dim);
        NDArray_VMALLOC((void**)&Uf, sizeof(float) * NDArray_SHAPE(target)[0] * NDArray_SHAPE(target)[0]);
        NDArray_VMALLOC((void**)&Vf, sizeof(float) * NDArray_SHAPE(target)[1] * NDArray_SHAPE(target)[1]);
        NDArray_VMALLOC((void**)&output_data, sizeof(float) * NDArray_NUMELEMENTS(target));
        cudaMemcpy(output_data, NDArray_FDATA(target_ptr), sizeof(float) * NDArray_NUMELEMENTS(target), cudaMemcpyDeviceToDevice);
        cudaDeviceSynchronize();
#else
        return NULL;
#endif
    } else {
        Sf = (float *) emalloc(sizeof(float) * smallest_dim);
        Uf = (float *) emalloc(sizeof(float) * NDArray_SHAPE(target_ptr)[0] * NDArray_SHAPE(target_ptr)[0]);
        Vf = (float *) emalloc(sizeof(float) * NDArray_SHAPE(target_ptr)[1] * NDArray_SHAPE(target_ptr)[1]);
        output_data = emalloc(sizeof(float) * NDArray_NUMELEMENTS(target_ptr));
        memcpy(output_data, NDArray_FDATA(target_ptr), sizeof(float) * NDArray_NUMELEMENTS(target_ptr));
    }

    if(NDArray_DEVICE(target_ptr) == NDARRAY_DEVICE_GPU) {
#ifdef HAVE_CUBLAS
        computeSVDFloatGPU(output_data, NDArray_SHAPE(target)[0], NDArray_SHAPE(target)[1], Uf, Sf, Vf);
#else
        return NULL;
#endif
    } else {
        computeSVDFloat((float *) output_data, NDArray_SHAPE(target_ptr)[0], NDArray_SHAPE(target_ptr)[1], Uf, Sf, Vf);
    }

    if(NDArray_DEVICE(target_ptr) == NDARRAY_DEVICE_CPU) {
        efree(output_data);
    }
    U_shape = emalloc(sizeof(int) * NDArray_NDIM(target_ptr));
    V_shape = emalloc(sizeof(int) * NDArray_NDIM(target_ptr));
    S_shape = emalloc(sizeof(int));
    S_shape[0] = smallest_dim;

    memcpy(U_shape, NDArray_SHAPE(target_ptr), sizeof(int) * NDArray_NDIM(target_ptr));
    U_shape[1] = NDArray_SHAPE(target_ptr)[0];

    memcpy(V_shape, NDArray_SHAPE(target_ptr), sizeof(int) * NDArray_NDIM(target_ptr));
    V_shape[0] = NDArray_SHAPE(target_ptr)[1];

    rtn_u = Create_NDArray(U_shape, NDArray_NDIM(target_ptr), NDArray_TYPE(target_ptr), NDArray_DEVICE(target_ptr));
    rtn_s = Create_NDArray(S_shape, 1, NDArray_TYPE(target_ptr), NDArray_DEVICE(target_ptr));
    rtn_v = Create_NDArray(V_shape, NDArray_NDIM(target_ptr), NDArray_TYPE(target_ptr), NDArray_DEVICE(target_ptr));

    if(is_type(NDArray_TYPE(target_ptr), NDARRAY_TYPE_DOUBLE64)) {
        rtn_u->data = (char *) U;
        rtn_s->data = (char *) S;
        rtn_v->data = (char *) V;
    } else {
        rtn_u->data = (char *) Uf;
        rtn_s->data = (char *) Sf;
        rtn_v->data = (char *) Vf;
    }

    rtns[0] = rtn_u;
    rtns[1] = rtn_s;
    rtns[2] = rtn_v;

    if (NDArray_DEVICE(target_ptr) == NDARRAY_DEVICE_GPU) {
        rtn_u->device = NDARRAY_DEVICE_GPU;
        rtn_s->device = NDARRAY_DEVICE_GPU;
        rtn_v->device = NDARRAY_DEVICE_GPU;
        NDArray_VFREE(output_data);
        NDArray_FREE(target_ptr);
    }

    return rtns;
}

/**
 * @param a
 * @param b
 * @return
 */
NDArray*
NDArray_Matmul(NDArray *a, NDArray *b) {
    if (NDArray_NDIM(a) != NDArray_NDIM(b)) {
        zend_throw_error(NULL, "Arrays must have the same shape. Broadcasting not implemented.");
        return NULL;
    }

    if (NDArray_NDIM(a) == 0 && NDArray_NDIM(b) == 0) {
        return NDArray_Multiply_Float(a, b);
    }
    if (NDArray_NDIM(a) == 1 && NDArray_NDIM(b) == 1) {
        return NDArray_Dot(a, b);
    }

    if (NDArray_SHAPE(a)[NDArray_NDIM(a) - 1] != NDArray_SHAPE(b)[NDArray_NDIM(b) - 2]) {
        zend_throw_error(NULL, "Shape mismatch for matmul. cols(a) != rows(b)");
    }

    if (NDArray_NDIM(a) > 2 && NDArray_NDIM(b) > 2) {
        zend_throw_error(NULL, "Stack of matrices not allowed");
        return NULL;
    }
    return NDArray_FMatmul(a, b);
}

/**
 * NDArray determinant
 *
 * @param a
 * @return
 */
NDArray*
NDArray_Det(NDArray *a) {
    int *new_shape = emalloc(sizeof(int));
    NDArray *rtn = Create_NDArray(new_shape, 0, NDARRAY_TYPE_FLOAT32, NDArray_DEVICE(a));
    if (NDArray_DEVICE(a) == NDARRAY_DEVICE_GPU) {
#ifdef HAVE_CUBLAS
        rtn->device = NDARRAY_DEVICE_GPU;
        NDArray_VMALLOC((void **)&rtn->data, sizeof(float));
        cuda_det_float(NDArray_FDATA(a), NDArray_FDATA(rtn), NDArray_SHAPE(a)[0]);
#endif
    } else {
        int info;
        int N = NDArray_SHAPE(a)[0];
        int* ipiv = (int*) emalloc(N * sizeof(int));
        float *matrix = emalloc(sizeof(float) * NDArray_NUMELEMENTS(a));
        rtn->data = emalloc(sizeof(float));
        memcpy(matrix, NDArray_FDATA(a), sizeof(float) * NDArray_NUMELEMENTS(a));
        // LU Decomposition using LAPACKE interface
        sgetrf_(&N, &N, matrix, &N, ipiv, &info);

        if (info != 0) {
            if (info > 0) {
                NDArray_FDATA(rtn)[0] = 0.f;
                efree(ipiv);
                efree(matrix);
                return rtn;
            }
            printf("Error in LU decomposition. Code: %d\n", info);
            efree(ipiv);
            exit(1);
        }

        // Calculate determinant as product of diagonal elements
        float det = 1;
        for (int i = 0; i < N; i++) {
            det *= matrix[i* N + i];
        }

        // Account for the parity of the permutation
        int num_perm = 0;
        for (int i = 0; i < N; i++) {
            if (i + 1 != ipiv[i]) num_perm++;
        }
        if (num_perm % 2 != 0) det = -det;

        efree(ipiv);
        efree(matrix);
        NDArray_FDATA(rtn)[0] = det;
    }
    return rtn;
}

/**
 * @param nda
 * @param ndb
 * @return
 */
NDArray*
NDArray_Inner(NDArray *nda, NDArray *ndb) {
    NDArray *rtn = NULL;

    if (NDArray_NDIM(nda) == 0 && NDArray_NDIM(ndb) == 0) {
        return NDArray_Multiply_Float(nda, ndb);
    }

    int i;
    int last_dim_a, last_dim_b;
    if (NDArray_DEVICE(nda) != NDArray_DEVICE(ndb)) {
        zend_throw_error(NULL, "Device mismatch, both NDArray must be in the same device.");
        return NULL;
    }

    last_dim_a = NDArray_SHAPE(nda)[NDArray_NDIM(nda) - 1];
    last_dim_b = NDArray_SHAPE(ndb)[NDArray_NDIM(ndb) - 1];
    if (last_dim_a != last_dim_b) {
        zend_throw_error(NULL, "Shape is not aligned to perform the inner product.");
        return NULL;
    }

    NDArray *mul = NDArray_Multiply_Float(nda, ndb);
    rtn = NDArray_CreateFromFloatScalar(NDArray_Sum_Float(mul));
    NDArray_FREE(mul);
    if (NDArray_NDIM(nda) > 1) {
        rtn->ndim = NDArray_NDIM(nda);
        rtn->dimensions = emalloc(sizeof(int) * NDArray_NDIM(nda));
        rtn->strides = emalloc(sizeof(int) * NDArray_NDIM(nda));
        for (i = 0; i < NDArray_NDIM(rtn); i++) {
            NDArray_SHAPE(rtn)[i] = 1;
            NDArray_STRIDES(rtn)[i] = NDArray_ELSIZE(rtn);
        }
    }
    return rtn;
}

/**
 * NDArray dot product
 *
 * @param nda
 * @param ndb
 * @return
 */
NDArray*
NDArray_Dot(NDArray *nda, NDArray *ndb) {
    if (NDArray_DEVICE(nda) != NDArray_DEVICE(ndb)) {
        zend_throw_error(NULL, "Device mismatch, both NDArray MUST be in the same device.");
        return NULL;
    }

    if (NDArray_NDIM(nda) == 1 && NDArray_NDIM(ndb) == 1) {
        return NDArray_Inner(nda, ndb);
    } else if (NDArray_NDIM(nda) == 2 && NDArray_NDIM(ndb) == 2) {
        return NDArray_Matmul(nda, ndb);
    } else if (NDArray_NDIM(nda) == 0 || NDArray_NDIM(ndb) == 0) {
        return NDArray_Multiply_Float(nda, ndb);
    } else if (NDArray_NDIM(nda) > 0 && NDArray_NDIM(ndb) == 1) {
        if (NDArray_DEVICE(nda) == NDARRAY_DEVICE_GPU) {
#ifdef HAVE_CUBLAS
            int *rtn_shape = emalloc(sizeof(int) * (NDArray_NDIM(nda) - 1));
            copy(NDArray_SHAPE(nda), rtn_shape, NDArray_NDIM(nda) -1);
            NDArray *rtn = NDArray_Empty(rtn_shape, NDArray_NDIM(nda) - 1, NDARRAY_TYPE_FLOAT32, NDARRAY_DEVICE_GPU);
            cuda_float_multiply_matrix_vector(NDArray_SHAPE(nda)[NDArray_NDIM(nda) - 1], NDArray_FDATA(nda), NDArray_FDATA(ndb),
                                              NDArray_FDATA(rtn), NDArray_SHAPE(nda)[NDArray_NDIM(nda) - 2], NDArray_SHAPE(nda)[NDArray_NDIM(nda) - 1]);
            return rtn;
#endif
        } else {
#ifdef HAVE_CBLAS
            int *rtn_shape = emalloc(sizeof(int) * (NDArray_NDIM(nda) - 1));
            copy(NDArray_SHAPE(nda), rtn_shape, NDArray_NDIM(nda) -1);
            NDArray *rtn = NDArray_Empty(rtn_shape, NDArray_NDIM(nda) - 1, NDARRAY_TYPE_FLOAT32, NDARRAY_DEVICE_CPU);
            cblas_sgemv(CblasRowMajor, CblasNoTrans, NDArray_SHAPE(nda)[NDArray_NDIM(nda) - 2], NDArray_SHAPE(nda)[NDArray_NDIM(nda) - 1], 1.0f, NDArray_FDATA(nda), NDArray_SHAPE(nda)[NDArray_NDIM(nda) - 1],
                        NDArray_FDATA(ndb), 1, 0.0f, NDArray_FDATA(rtn), 1);
            return rtn;
#endif
        }
    } else if (NDArray_NDIM(nda) > 0 && NDArray_NDIM(ndb) >= 2) {
        // @todo Implement missing conditional
        zend_throw_error(NULL, "Not implemented");
        return NULL;
    }
    return NULL;
}

/**
 * L2 NORM
 *
 * @param target
 * @return
 */
NDArray*
NDArray_L2Norm(NDArray* target) {
    NDArray *rtn = NULL;
    NDArray **svd = NDArray_SVD(target);
    if (svd == NULL) {
        return NULL;
    }
    float max_svd = NDArray_Max(svd[1]);
    rtn = NDArray_CreateFromFloatScalar(max_svd);
    NDArray_FREE(svd[0]);
    NDArray_FREE(svd[1]);
    NDArray_FREE(svd[2]);
    efree(svd);
    return rtn;
}

/**
 * L1 NORM
 *
 * @param target
 * @return
 */
NDArray*
NDArray_L1Norm(NDArray* target) {
    NDArray *rtn = NULL;
    float max_value = FLT_MIN;
    float *results = emalloc(sizeof(float) * NDArray_SHAPE(target)[NDArray_NDIM(target) - 2]);
    NDArray *transposed = NDArray_Transpose(target);
    NDArray *ab = NDArray_Abs(transposed);
    NDArray_FREE(transposed);
    NDArray *slice;
    while(!NDArrayIterator_ISDONE(ab)) {
        slice = NDArrayIterator_GET(ab);
        results[ab->iterator->current_index] = NDArray_Sum_Float(slice);
        NDArray_FREE(slice);
        NDArrayIterator_NEXT(ab);
    }
    for (int i = 0; i < NDArray_SHAPE(target)[NDArray_NDIM(target) - 2]; i++) {
        if (max_value < results[i]) {
            max_value = results[i];
        }
    }
    efree(results);
    NDArray_FREE(ab);
    rtn = NDArray_CreateFromFloatScalar(max_value);
    return rtn;
}

/**
 * Matrix or vector norm
 *
 * Types
 *  INT_MAX - Frobenius norm
 *  0 - sum(x!=0) Only Vectors
 *  1 - max(sum(abs(x), axis=0))
 * -1 - min(sum(abs(x), axis=0))
 *  2 - 2-norm
 * -2 - smallest singular value
 * @param target
 * @return
 */
NDArray*
NDArray_Norm(NDArray* target, int type) {
    if (type == 1) {
        return NDArray_L1Norm(target);
    }
    if (type == 2) {
        return NDArray_L2Norm(target);
    }

    zend_throw_error(NULL, "NDArray_Norm: The provided norm `%d` is invalid", type);
    return NULL;
}

/**
 *
 * @param matrix
 * @param n
 * @return 1 if succeeded, 0 if failed
 */
int
matrixFloatInverse(float* matrix, int n) {
    int* ipiv = (int*)emalloc(n * sizeof(int)); // Pivot indices
    int info; // Status variable

    // Perform LU factorization
    sgetrf_(&n, &n, matrix, &n, ipiv, &info);
    if (info != 0) {
        zend_throw_error(NULL, "LU factorization failed. Unable to compute the matrix inverse.\n");
        efree(ipiv);
        return 0;
    }

    // Calculate the inverse
    int lwork = n * n;
    float work_query[lwork];
    sgetri_(&n, matrix, &n, ipiv, work_query, &lwork, &info);
    if (info != 0) {
        zend_throw_error(NULL, "Matrix inversion failed.\n");
        efree(ipiv);
        return 0;
    }

    efree(ipiv);
    return 1;
}

/**
 *
 * @param matrix
 * @param n
 * @return 1 if succeeded, 0 if failed
 */
int
matrixFloatLU(float* matrix, int n, float *p, float *l, float *u) {
    int i, j, k, maxIndex;
    float maxVal, tempVal;

    // Initialize L, U, and P matrices
    for (i = 0; i < n; i++) {
        for (j = 0; j < n; j++) {
            if (i == j) {
                l[i * n + j] = 1.0f;
                u[i * n + j] = matrix[i * n + j];
            } else {
                l[i * n + j] = 0.0f;
                u[i * n + j] = matrix[i * n + j];
            }
            p[i * n + j] = (i == j) ? 1.0f : 0.0f;
        }
    }

    // Perform LU decomposition with partial pivoting
    for (k = 0; k < n - 1; k++) {
        maxIndex = k;
        maxVal = u[k * n + k];

        // Find the row with the maximum value in the current column
        for (i = k + 1; i < n; i++) {
            if (u[i * n + k] > maxVal) {
                maxIndex = i;
                maxVal = u[i * n + k];
            }
        }

        // Swap rows in U matrix
        if (maxIndex != k) {
            for (j = 0; j < n; j++) {
                tempVal = u[k * n + j];
                u[k * n + j] = u[maxIndex * n + j];
                u[maxIndex * n + j] = tempVal;

                tempVal = p[k * n + j];
                p[k * n + j] = p[maxIndex * n + j];
                p[maxIndex * n + j] = tempVal;
            }
        }

        // Perform elimination in U matrix and store multipliers in L matrix
        for (i = k + 1; i < n; i++) {
            l[i * n + k] = u[i * n + k] / u[k * n + k];
            for (j = k; j < n; j++) {
                u[i * n + j] -= l[i * n + k] * u[k * n + j];
            }
        }
    }
}

/**
 * Calculate the inverse of a square NDArray
 *
 * @param target
 * @return
 */
NDArray*
NDArray_Inverse(NDArray* target) {
    int info;
    NDArray *rtn = NDArray_Copy(target, NDArray_DEVICE(target));
    if (NDArray_NDIM(target) != 2) {
        zend_throw_error(NULL, "Array must be at least two-dimensional");
        NDArray_FREE(rtn);
        return NULL;
    }

    if (NDArray_SHAPE(target)[0] != NDArray_SHAPE(target)[1]) {
        zend_throw_error(NULL, "Array must be square");
        NDArray_FREE(rtn);
        return NULL;
    }

    if (NDArray_DEVICE(target) == NDARRAY_DEVICE_CPU) {
        // CPU INVERSE CALL
        info = matrixFloatInverse(NDArray_FDATA(rtn), NDArray_SHAPE(rtn)[0]);
        if (!info) {
            NDArray_FREE(rtn);
            return NULL;
        }
    } else {
        // GPU INVERSE CALL
#ifdef HAVE_CUBLAS
        cuda_matrix_float_inverse(NDArray_FDATA(rtn), NDArray_SHAPE(rtn)[0]);
#endif
    }

    return rtn;
}

/**
 * Calculate the inverse of a square NDArray
 *
 * @param target
 * @return
 */
NDArray**
NDArray_LU(NDArray* target) {
    if (NDArray_NDIM(target) != 2) {
        zend_throw_error(NULL, "Array must be at least two-dimensional");
        return NULL;
    }
    if (NDArray_SHAPE(target)[0] != NDArray_SHAPE(target)[1]) {
        zend_throw_error(NULL, "Array must be square");
        return NULL;
    }
    NDArray **rtns = emalloc(sizeof(NDArray*) * 3);
    int info;
    int *new_shape_p = emalloc(sizeof(int) * NDArray_NDIM(target));
    int *new_shape_l = emalloc(sizeof(int) * NDArray_NDIM(target));
    int *new_shape_u = emalloc(sizeof(int) * NDArray_NDIM(target));
    memcpy(new_shape_p, NDArray_SHAPE(target), sizeof(int) * (int)NDArray_NDIM(target));
    memcpy(new_shape_l, NDArray_SHAPE(target), sizeof(int) * (int)NDArray_NDIM(target));
    memcpy(new_shape_u, NDArray_SHAPE(target), sizeof(int) * (int)NDArray_NDIM(target));
    NDArray *copied = NDArray_Copy(target, NDArray_DEVICE(target));
    NDArray *p = NDArray_Empty(new_shape_p, NDArray_NDIM(target), NDARRAY_TYPE_FLOAT32, NDArray_DEVICE(target));
    NDArray *l = NDArray_Empty(new_shape_l, NDArray_NDIM(target), NDARRAY_TYPE_FLOAT32, NDArray_DEVICE(target));
    NDArray *u = NDArray_Empty(new_shape_u, NDArray_NDIM(target), NDARRAY_TYPE_FLOAT32, NDArray_DEVICE(target));

    if (NDArray_DEVICE(target) == NDARRAY_DEVICE_CPU) {
        // CPU INVERSE CALL
        info = matrixFloatLU(NDArray_FDATA(copied),
                             NDArray_SHAPE(copied)[0],
                             NDArray_FDATA(p),
                             NDArray_FDATA(l),
                             NDArray_FDATA(u));
        if (!info) {
            NDArray_FREE(copied);
            return NULL;
        }
    } else {
        // GPU INVERSE CALL
#ifdef HAVE_CUBLAS
        cuda_float_lu(NDArray_FDATA(copied), NDArray_FDATA(l), NDArray_FDATA(u), NDArray_FDATA(p), NDArray_SHAPE(copied)[0]);
#endif
    }
    NDArray_FREE(copied);
    rtns[0] = p;
    rtns[1] = l;
    rtns[2] = u;
    return rtns;
}

/**
 * NDArray matrix rank
 *
 * @param target
 * @param tol
 * @return
 */
NDArray*
NDArray_MatrixRank(NDArray *target, float *tol) {
    float mtol;
    int rank = 0, i;
    NDArray *rtn;
    NDArray **svd = NDArray_SVD(target);
    float *singular_values;

    if (NDArray_DEVICE(target) == NDARRAY_DEVICE_CPU) {
        singular_values = NDArray_FDATA(svd[1]);
    } else {
#ifdef HAVE_CUBLAS
        singular_values = emalloc(sizeof(float) * NDArray_NUMELEMENTS(target));
        cudaMemcpy(singular_values, NDArray_FDATA(svd[1]), sizeof(float) * NDArray_NUMELEMENTS(svd[1]), cudaMemcpyDeviceToHost);
#endif
    }
    int minMN = (NDArray_SHAPE(target)[NDArray_NDIM(target) - 2] < (NDArray_SHAPE(target)[NDArray_NDIM(target) - 1])) ? (NDArray_SHAPE(target)[NDArray_NDIM(target) - 2]) : (NDArray_SHAPE(target)[NDArray_NDIM(target) - 1]);

    // Set the tolerance if not provided
    if (tol == NULL) {
        float maxSingularValue = singular_values[0];
        for (i = 1; i < minMN; i++) {
            if (singular_values[i] > maxSingularValue) {
                maxSingularValue = singular_values[i];
            }
        }
        mtol = maxSingularValue * fmaxf((float)NDArray_SHAPE(target)[NDArray_NDIM(target) - 2], (float)NDArray_SHAPE(target)[NDArray_NDIM(target) - 1]) * FLT_EPSILON;
    } else {
        mtol = *tol;
    }

    for (i = 0; i < minMN; i++) {
        if (singular_values[i] > mtol) {
            rank++;
        }
    }

    NDArray_FREE(svd[0]);
    NDArray_FREE(svd[1]);
    NDArray_FREE(svd[2]);
    efree(svd);
    rtn = NDArray_CreateFromLongScalar((int)rank);

    if (NDArray_DEVICE(target) == NDARRAY_DEVICE_GPU) {
        efree(singular_values);
    }

    return rtn;
}

/**
 * NDArray::outer
 *
 * @param a
 * @param b
 * @return
 */
NDArray*
NDArray_Outer(NDArray *a, NDArray *b) {
    if (NDArray_NDIM(a) != 1 || NDArray_NDIM(b) != 1) {
        zend_throw_error(NULL, "Invalid operation: NDArray::outer() requires both arrays to be 1-dimensional vectors.");
        return NULL;
    }

    if (NDArray_DEVICE(a) != NDArray_DEVICE(b)) {
        zend_throw_error(NULL, "NDArray::outer() requires both arrays to be on the same device (CPU or GPU).");
        return NULL;
    }
    int *output_shape = emalloc(sizeof(int) * 2);
    output_shape[0] = NDArray_NUMELEMENTS(a);
    output_shape[1] = NDArray_NUMELEMENTS(b);
    NDArray *rtn = NDArray_Zeros(output_shape, 2, NDArray_TYPE(a), NDArray_DEVICE(a));
    if (NDArray_DEVICE(a) == NDARRAY_DEVICE_CPU) {
#ifdef HAVE_CBLAS
        cblas_sger(CblasRowMajor, NDArray_NUMELEMENTS(a), NDArray_NUMELEMENTS(b), 1.0f, NDArray_FDATA(a), 1, NDArray_FDATA(b), 1,
                   NDArray_FDATA(rtn), NDArray_NUMELEMENTS(b));
#endif
    } else {
#ifdef HAVE_CUBLAS
        cuda_calculate_outer_product(NDArray_NUMELEMENTS(a), NDArray_NUMELEMENTS(b), NDArray_FDATA(a), NDArray_FDATA(b),
                                     NDArray_FDATA(rtn));
#endif
    }
    return rtn;
}

/**
 * NDArray::trace
 *
 * @return
 */
NDArray*
NDArray_Trace(NDArray *a) {
    NDArray* diagonal = NDArray_Diagonal(a, 0);
    if (diagonal == NULL) {
        return NULL;
    }
    float result = NDArray_Sum_Float(diagonal);
    NDArray_FREE(diagonal);
    return NDArray_CreateFromFloatScalar(result);
}

int
computeEigenvaluesAndEigenvectorsFloat(NDArray* array, NDArray* rightEigenvectors,
                                       NDArray* eigenvalues, NDArray *wivectors, NDArray *leftEigenvectors) {
    // Assuming 'array' contains the input square matrix
    int n = array->dimensions[0]; // Size of the square matrix

    // Compute eigenvalues and right eigenvectors using LAPACK function
    int info = LAPACKE_sgeev(LAPACK_ROW_MAJOR, 'N', 'V', n, NDArray_FDATA(array), n,
                             NDArray_FDATA(rightEigenvectors), NDArray_FDATA(wivectors), NDArray_FDATA(leftEigenvectors),
                             n, NDArray_FDATA(eigenvalues),n);

    // Check if the computation was successful (info == 0)
    if (info != 0) {
        zend_throw_error(NULL, "Error computing eigenvalues and eigenvectors.\n");
        return 0;
    }
    return 1;
}

/**
 * NDArray::eig
 *
 * @param a
 * @return
 */
NDArray**
NDArray_Eig(NDArray *a) {
    if (NDArray_NDIM(a) != 2 || NDArray_SHAPE(a)[0] != NDArray_SHAPE(a)[1]) {
        zend_throw_error(NULL, "Error: Input matrix is not square.\n");
        return NULL;
    }
    NDArray **rtn = emalloc(sizeof(NDArray*) * 2);
    NDArray* eigenvalues, *rightEigenvectors, *wivectors, *leftEigenvectors;
    int *eigenvalues_shape, *rightEigenvectors_shape, *wivectors_shape, *leftEigenvectors_shape;

    rightEigenvectors_shape = emalloc(sizeof(int));
    rightEigenvectors_shape[0] = NDArray_SHAPE(a)[0];

    rightEigenvectors = NDArray_Zeros(rightEigenvectors_shape, 1, NDArray_TYPE(a), NDArray_DEVICE(a));
    if (NDArray_DEVICE(a) == NDARRAY_DEVICE_CPU) {
        wivectors_shape = emalloc(sizeof(int));
        leftEigenvectors_shape = emalloc(sizeof(int));
        eigenvalues_shape = emalloc(sizeof(int) * NDArray_NDIM(a));
        wivectors_shape[0] = NDArray_SHAPE(a)[0];
        eigenvalues_shape[0] = NDArray_SHAPE(a)[0];
        eigenvalues_shape[1] = NDArray_SHAPE(a)[0];
        leftEigenvectors_shape[0] = NDArray_SHAPE(a)[0];
        eigenvalues = NDArray_Zeros(eigenvalues_shape, NDArray_NDIM(a), NDArray_TYPE(a), NDArray_DEVICE(a));
        wivectors = NDArray_Zeros(wivectors_shape, 1, NDArray_TYPE(a), NDArray_DEVICE(a));
        leftEigenvectors = NDArray_Zeros(leftEigenvectors_shape, 1, NDArray_TYPE(a), NDArray_DEVICE(a));
        if (!computeEigenvaluesAndEigenvectorsFloat(a, rightEigenvectors, eigenvalues, wivectors, leftEigenvectors)) {
            efree(rtn);
            return NULL;
        }
        NDArray_FREE(leftEigenvectors);
        NDArray_FREE(wivectors);
    } else {
#ifdef HAVE_CUBLAS
        efree(rtn);
        NDArray_FREE(rightEigenvectors);
        zend_throw_error(NULL, "GPU eig currently unavailable");
        return NULL;
        //eigenvalues = NDArray_Copy(a, NDArray_DEVICE(a));
        //cuda_matrix_eig_float(NDArray_FDATA(eigenvalues), NDArray_SHAPE(a)[0], NDArray_FDATA(rightEigenvectors));
#endif
    }
    rtn[0] = rightEigenvectors;
    rtn[1] = eigenvalues;
    return rtn;
}

/**
 * NDArray::lstsq
 * @todo Implement GPU
 *
 * @param a
 * @param b
 * @return
 */
NDArray*
NDArray_Lstsq(NDArray *a, NDArray *b) {
    if (NDArray_DEVICE(a) == NDARRAY_DEVICE_GPU || NDArray_DEVICE(b) == NDARRAY_DEVICE_GPU) {
        zend_throw_error(NULL, "ndarray::lstsq not implemented for GPU");
        return NULL;
    }

    // Check if input matrices have compatible dimensions
    if (NDArray_NDIM(a) != 2 || NDArray_NDIM(b) != 2 || NDArray_SHAPE(a)[0] != NDArray_SHAPE(b)[0]) {
        zend_throw_error(NULL, "Invalid dimensions to calculate lstsq, both arrays must have 2 dimensions and $b must contain the same amount of rows as $a");
        return NULL;
    }

    int m = a->dimensions[0]; // Number of rows of the coefficient matrix A
    int n = a->dimensions[1]; // Number of columns of the coefficient matrix A
    int nrhs = b->dimensions[1]; // Number of right-hand sides (columns of B)

    int *out_shape = (int*)emalloc(2 * sizeof(int));
    out_shape[0] = n;
    out_shape[1] = nrhs;
    NDArray *x = NDArray_Zeros(out_shape, 2, NDArray_TYPE(a), NDArray_DEVICE(a));
    if (NDArray_DEVICE(a) == NDARRAY_DEVICE_CPU) {
        // Allocate memory and copy data for the coefficient matrix A
        float *a_data = (float *) emalloc(m * n * sizeof(float));
        memcpy(a_data, a->data, m * n * sizeof(float));

        // Allocate memory and copy data for the right-hand side matrix B
        float *b_data = (float *) emalloc(m * nrhs * sizeof(float));
        memcpy(b_data, b->data, m * nrhs * sizeof(float));

        int info = LAPACKE_sgels(LAPACK_ROW_MAJOR, 'N', NDArray_SHAPE(a)[0], NDArray_SHAPE(a)[1], NDArray_SHAPE(b)[1],
                                 a_data,
                                 NDArray_SHAPE(a)[1], b_data, NDArray_SHAPE(b)[1]);

        if (info > 0) {
            zend_throw_error(NULL,
                             "The diagonal element %i of the triangular factor of $a is zero, so that $a does not have full rank.",
                             info);
            return NULL;
        }
        // Copy the result data to the output NDArray
        memcpy(NDArray_FDATA(x), b_data, n * nrhs * sizeof(float));
        efree(a_data);
        efree(b_data);
    } else {
#ifdef HAVE_CUBLAS
        cuda_lstsq_float(NDArray_FDATA(a), NDArray_SHAPE(a)[0], NDArray_SHAPE(a)[1], NDArray_FDATA(b), NDArray_SHAPE(b)[0],
                         NDArray_FDATA(x));
#endif
    }
    return x;
}

/**
 * NDArray::qr
 * @todo Implement GPU
 *
 * @param a
 * @return
 */
NDArray**
NDArray_Qr(NDArray *a) {
    if (NDArray_DEVICE(a) == NDARRAY_DEVICE_GPU) {
        zend_throw_error(NULL, "ndarray::qr not implemented for GPU");
        return NULL;
    }

    // Check if the input matrix is 2D
    if (a->ndim != 2) {
        return NULL;
    }

    int m = a->dimensions[0]; // Number of rows of the matrix A
    int n = a->dimensions[1]; // Number of columns of the matrix A

    // Ensure that m >= n for the QR factorization
    if (m < n) {
        return NULL;
    }

    // Allocate memory for the result matrices Q and R
    int *q_dimensions = (int*)emalloc(2 * sizeof(int));
    q_dimensions[0] = m;
    q_dimensions[1] = n;
    NDArray* q = NDArray_Zeros(q_dimensions, 2, NDArray_TYPE(a), NDArray_DEVICE(a));

    int *r_dimensions = (int*)emalloc(2 * sizeof(int));
    r_dimensions[0] = n;
    r_dimensions[1] = n;
    NDArray* r = NDArray_Zeros(r_dimensions, 2, NDArray_TYPE(a), NDArray_DEVICE(a));

    // Allocate memory and copy data for the matrix A
    float* a_data = (float*)emalloc(m * n * n * sizeof(float));
    memcpy(a_data, NDArray_FDATA(a), m * n * sizeof(float));

    // Allocate memory for the workspace
    float* tau = (float*)emalloc(n * sizeof(float));
    int info;

    // Query the optimal workspace size
    info = LAPACKE_sgeqrf(LAPACK_ROW_MAJOR, m, n, a_data, n, tau);

    // Extract the upper triangular part of the matrix A to R
    for (int i = 0; i < n; i++) {
        for (int j = i; j < n; j++) {
            ((float*)r->data)[i * (r->strides[0]/NDArray_ELSIZE(r)) + j * (r->strides[1]/NDArray_ELSIZE(r))] = ((float*)a_data)[i * m + j];
        }
    }

    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < m; j++) {
            ((float*)q->data)[i * (q->strides[0]/NDArray_ELSIZE(q)) + j * (q->strides[1]/NDArray_ELSIZE(q))] = ((float*)a_data)[i * m + j];
        }
    }

    memcpy(NDArray_FDATA(q), a_data, NDArray_NUMELEMENTS(a) * NDArray_ELSIZE(a));
    efree(tau);
    efree(a_data);
    NDArray **rtn = emalloc(sizeof(NDArray*) * 2);
    rtn[0] = q;
    rtn[1] = r;
    return rtn;
}

/**
 * NDArray::solve
 * @todo Implement GPU
 *
 * @param a
 * @param b
 * @return
 */
NDArray*
NDArray_Solve(NDArray *a, NDArray *b) {
    if (NDArray_DEVICE(a) != NDArray_DEVICE(b)) {
        zend_throw_error(NULL, "Both NDArray must be in the same device.");
        return NULL;
    }
    if (NDArray_DEVICE(a) == NDARRAY_DEVICE_GPU) {
        zend_throw_error(NULL, "ndarray::solve not implemented for GPU");
        return NULL;
    }
    // Check if input matrices are valid
    if (a == NULL || b == NULL) {
        return NULL;
    }

    // Check if input matrices are 2D and have compatible dimensions
    if (a->ndim != 2 || b->ndim != 2 || a->dimensions[0] != a->dimensions[1] || a->dimensions[0] != b->dimensions[0]) {
        zend_throw_error(NULL, "Incompatible shapes");
        return NULL;
    }

    int n = a->dimensions[0]; // Number of rows/columns of the square matrix A

    int *x_dimensions = (int*)emalloc(2 * sizeof(int));
    x_dimensions[0] = n;
    x_dimensions[1] = b->dimensions[1];
    NDArray *x = NDArray_Zeros(x_dimensions, 2, NDArray_TYPE(a), NDArray_DEVICE(a));

    // Allocate memory and copy data for the square matrix A
    float* a_data = (float*)emalloc(n * n * sizeof(float));
    memcpy(a_data, a->data, n * n * sizeof(float));

    // Allocate memory and copy data for the matrix B
    float* b_data = (float*)emalloc(n * x->dimensions[1] * sizeof(float));
    memcpy(b_data, b->data, n * x->dimensions[1] * sizeof(float));

    // Allocate memory for the pivot indices
    int* ipiv = (int*)emalloc(n * sizeof(int));

    LAPACKE_sgesv(LAPACK_ROW_MAJOR, n, NDArray_SHAPE(b)[1], a_data, NDArray_SHAPE(a)[0], ipiv, b_data, NDArray_SHAPE(b)[0]);

    // Copy the result data to the output NDArray
    memcpy(x->data, b_data, NDArray_NUMELEMENTS(b) * sizeof(float));

    efree(a_data);
    efree(b_data);
    efree(ipiv);
    return x;
}

/**
 * NDArray::cond
 *
 * @param a
 * @param b
 * @return
 */
NDArray*
NDArray_Cond(NDArray *a) {
    NDArray *a_norm = NDArray_L2Norm(a);
    NDArray *a_inv = NDArray_Inverse(a);
    NDArray *a_inv_norm = NDArray_L2Norm(a_inv);
    NDArray_FREE(a_inv);
    NDArray *rtn = NDArray_Multiply_Float(a_norm, a_inv_norm);
    NDArray_FREE(a_norm);
    NDArray_FREE(a_inv_norm);
    return rtn;
}

/**
 * NDArray::cholesky
 *
 * @todo Implement GPU
 * @param a
 * @return
 */
NDArray*
NDArray_Cholesky(NDArray *a) {
    if (NDArray_DEVICE(a) == NDARRAY_DEVICE_GPU) {
        zend_throw_error(NULL, "ndarray::cholesky not implemented for GPU");
        return NULL;
    }
    if (NDArray_NDIM(a) != 2 || NDArray_SHAPE(a)[0] != NDArray_SHAPE(a)[1]) {
        zend_throw_error(NULL, "NDArray_Cholesky: $a must be a square matrix.");
        return NULL;
    }

    NDArray *rtn = NDArray_Copy(a, NDArray_DEVICE(a));
    int info = LAPACKE_spotrf(LAPACK_ROW_MAJOR, 'L', NDArray_SHAPE(a)[0], NDArray_FDATA(rtn), NDArray_SHAPE(a)[0]);

    if (info > 0) {
        NDArray_FREE(rtn);
        zend_throw_error(NULL, "Error calculating the cholesky decomposition. (Is $a not positive definite?)");
        return NULL;
    }
#ifdef HAVE_AVX2
    int blockSize = 8; // AVX2 can process 8 single-precision floats at a time
    for (int i = 0; i < NDArray_SHAPE(a)[0]; i++) {
        // Perform AVX2 loop for blocks of 8 elements
        int j = i + 1;
        for (; j < NDArray_SHAPE(a)[0] - blockSize + 1; j += blockSize) {
            // Load 8 elements of the row (upper triangular elements) into an AVX register
            __m256 row_avx = _mm256_loadu_ps(&NDArray_FDATA(rtn)[i * NDArray_SHAPE(a)[0] + j]);
            // Set all elements of the AVX register to 0
            __m256 zero_avx = _mm256_setzero_ps();
            // Store the 0s back into the upper triangular elements of the row
            _mm256_storeu_ps(&NDArray_FDATA(rtn)[i * NDArray_SHAPE(a)[0] + j], zero_avx);
        }
        // Handle the remaining elements
        for (; j < NDArray_SHAPE(a)[0]; j++) {
            NDArray_FDATA(rtn)[i * NDArray_SHAPE(a)[0] + j] = 0.0f;
        }
    }
#else
    for (int i = 0; i < NDArray_SHAPE(a)[0]; i++) {
        for (int j = i + 1; j < NDArray_SHAPE(a)[1]; j++) {
            NDArray_FDATA(rtn)[i * NDArray_SHAPE(a)[0] + j] = 0.0f;
        }
    }
#endif

    return rtn;
}