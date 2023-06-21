#include <php.h>
#include "Zend/zend_alloc.h"
#include "Zend/zend_API.h"
#include "linalg.h"
#include "../../config.h"
#include "../ndarray.h"
#include "../initializers.h"
#include "../types.h"
#include "../debug.h"
#include "../manipulation.h"
#include "arithmetics.h"

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

/**
 * Double type (float64) matmul
 *
 * @param a
 * @param b
 * @return
 */
NDArray*
NDArray_FMatmul(NDArray *a, NDArray *b) {
    int *output_shape = emalloc(sizeof(int) * 2);

    output_shape[0] = NDArray_SHAPE(a)[0];
    output_shape[1] = NDArray_SHAPE(b)[1];

    NDArray *result = NDArray_Zeros(output_shape, 2, NDARRAY_TYPE_FLOAT32);
    if (NDArray_DEVICE(a) == NDARRAY_DEVICE_GPU) {
#ifdef HAVE_CUBLAS
        NDArray *result_gpu = NDArray_ToGPU(result);
        NDArray_FREE(result);
        cuda_matmul_float(NDArray_NUMELEMENTS(a), NDArray_FDATA(a), NDArray_FDATA(b), NDArray_FDATA(result_gpu),
                          NDArray_SHAPE(a)[0], NDArray_SHAPE(a)[1], NDArray_SHAPE(b)[0]);
        return result_gpu;
#endif
    } else {
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                    NDArray_SHAPE(b)[1], NDArray_SHAPE(a)[0], NDArray_SHAPE(a)[1],
                    1.0f, NDArray_FDATA(a), NDArray_SHAPE(a)[1], NDArray_FDATA(b), NDArray_SHAPE(b)[1],
                    0.0f, NDArray_FDATA(result), NDArray_SHAPE(b)[1]);
    }
    return result;
}


void
computeSVDFloat(float* A, int m, int n, float* U, float* S, float* V) {
    int lda = m;
    int ldu = m;
    int ldvt = n;

    int info;

    // Compute workspace size
    float work_size;
    int lwork = -1;  // query the workspace size
#ifdef LAPACK_FORTRAN_STRLEN_END
    sgesvd_("A", "A", &m, &n, A, &lda, S, U, &ldu, V, &ldvt, &work_size, &lwork, &info, 0, 0);
#else
    sgesvd_("A", "A", &m, &n, A, &lda, S, U, &ldu, V, &ldvt, &work_size, &lwork, &info);
#endif
    // Allocate workspace
    lwork = (int)work_size;
    float* work = (float*)emalloc(sizeof(float) * lwork);

    // Compute SVD
#ifdef LAPACK_FORTRAN_STRLEN_END
    sgesvd_("A", "A", &m, &n, A, &lda, S, U, &ldu, V, &ldvt, work, &lwork, &info, 0, 0);
#else
    sgesvd_("A", "A", &m, &n, A, &lda, S, U, &ldu, V, &ldvt, work, &lwork, &info);
#endif

    if (info > 0) {
        printf("SVD computation failed.\n");
        return;
    }

    // Free allocated memory
    efree(work);
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
    NDArray *rtn_s, *rtn_u, *rtn_v;
    double *U, *S, *V;
    float *output_data;
    float  *Uf, *Sf, *Vf;
    int *U_shape, *S_shape, *V_shape;
    int smallest_dim = -1;
    rtns = emalloc(sizeof(NDArray*) * 3);

    for (int i = 0; i < NDArray_NDIM(target); i++) {
        if (smallest_dim == -1) {
            smallest_dim = NDArray_SHAPE(target)[i];
            continue;
        }
        if (smallest_dim > NDArray_SHAPE(target)[i]) {
            smallest_dim = NDArray_SHAPE(target)[i];
        }
    }
    if(NDArray_DEVICE(target) == NDARRAY_DEVICE_GPU) {
#ifdef HAVE_CUBLAS
        NDArray_VMALLOC((void**)&Sf, sizeof(float) * smallest_dim);
        NDArray_VMALLOC((void**)&Uf, sizeof(float) * NDArray_SHAPE(target)[0] * NDArray_SHAPE(target)[0]);
        NDArray_VMALLOC((void**)&Vf, sizeof(float) * NDArray_SHAPE(target)[1] * NDArray_SHAPE(target)[1]);
        cudaDeviceSynchronize();
        output_data = NDArray_FDATA(target);
 #else
        return NULL;
#endif
    } else {
        Sf = (float *) emalloc(sizeof(float) * smallest_dim);
        Uf = (float *) emalloc(sizeof(float) * NDArray_SHAPE(target)[0] * NDArray_SHAPE(target)[0]);
        Vf = (float *) emalloc(sizeof(float) * NDArray_SHAPE(target)[1] * NDArray_SHAPE(target)[1]);
        output_data = emalloc(sizeof(float) * NDArray_NUMELEMENTS(target));
        memcpy(output_data, NDArray_FDATA(target), sizeof(float) * NDArray_NUMELEMENTS(target));
    }

    if(NDArray_DEVICE(target) == NDARRAY_DEVICE_GPU) {
#ifdef HAVE_CUBLAS

        computeSVDFloatGPU(output_data, NDArray_SHAPE(target)[0], NDArray_SHAPE(target)[1], Uf, Sf, Vf);
#else
        return NULL;
#endif
    } else {
        computeSVDFloat((float *) output_data, NDArray_SHAPE(target)[0], NDArray_SHAPE(target)[1], Uf, Sf, Vf);
    }

    if(NDArray_DEVICE(target) == NDARRAY_DEVICE_CPU) {
        efree(output_data);
    }
    U_shape = emalloc(sizeof(int) * NDArray_NDIM(target));
    V_shape = emalloc(sizeof(int) * NDArray_NDIM(target));
    S_shape = emalloc(sizeof(int));
    S_shape[0] = smallest_dim;

    memcpy(U_shape, NDArray_SHAPE(target), sizeof(int) * NDArray_NDIM(target));
    U_shape[1] = NDArray_SHAPE(target)[0];

    memcpy(V_shape, NDArray_SHAPE(target), sizeof(int) * NDArray_NDIM(target));
    V_shape[0] = NDArray_SHAPE(target)[1];

    rtn_u = Create_NDArray(U_shape, NDArray_NDIM(target), NDArray_TYPE(target));
    rtn_s = Create_NDArray(S_shape, 1, NDArray_TYPE(target));
    rtn_v = Create_NDArray(V_shape, NDArray_NDIM(target), NDArray_TYPE(target));

    if(is_type(NDArray_TYPE(target), NDARRAY_TYPE_DOUBLE64)) {
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

    if (NDArray_DEVICE(target) == NDARRAY_DEVICE_GPU) {
        rtn_u->device = NDARRAY_DEVICE_GPU;
        rtn_s->device = NDARRAY_DEVICE_GPU;
        rtn_v->device = NDARRAY_DEVICE_GPU;
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
    NDArray *rtn = Create_NDArray(new_shape, 0, NDARRAY_TYPE_FLOAT32);
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
    }
    else if (NDArray_NDIM(nda) == 0 || NDArray_NDIM(ndb) == 0) {
        return NDArray_Multiply_Float(nda, ndb);
    }
    else if (NDArray_NDIM(nda) > 0 && NDArray_NDIM(ndb) == 1) {
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
    }
    else if (NDArray_NDIM(nda) > 0 && NDArray_NDIM(ndb) >= 2) {
        // @todo Implement missing conditional
        zend_throw_error(NULL, "Not implemented");
        return NULL;
    }
    return NULL;
}


