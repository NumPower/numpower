#include "linalg.h"
#include "../../config.h"
#include "../ndarray.h"
#include "../initializers.h"
#include "../types.h"
#include "../debug.h"
#include <Zend/zend_alloc.h>

#ifdef HAVE_CBLAS
#include <cblas.h>
#include <lapacke.h>
#endif

#ifdef HAVE_CUBLAS
#include <cuda_runtime.h>
#include <cublas_v2.h>
#endif

/**
 * Double type (float64) matmul
 *
 * @param a
 * @param b
 * @return
 */
NDArray*
NDArray_DMatmul(NDArray *a, NDArray *b) {
    int *output_shape = emalloc(sizeof(int) * 2);

    output_shape[0] = NDArray_SHAPE(a)[0];
    output_shape[1] = NDArray_SHAPE(b)[1];

    NDArray *result = NDArray_Zeros(output_shape, 2, NDARRAY_TYPE_DOUBLE64);

    if (NDArray_DEVICE(a) == NDARRAY_DEVICE_GPU) {
#ifdef HAVE_CUBLAS
        cublasHandle_t handle;
        cublasCreate(&handle);

        double alpha = 1.0, beta = 0.0;

        cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    NDArray_SHAPE(a)[0], NDArray_SHAPE(b)[1], NDArray_SHAPE(a)[1],
                    &alpha, NDArray_DDATA(b), NDArray_SHAPE(b)[1], NDArray_DDATA(a), NDArray_SHAPE(a)[1], &beta,
                    NDArray_DDATA(result), NDArray_SHAPE(b)[1]);
        cublasDestroy(handle);
        return result;
#endif
    }
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                NDArray_SHAPE(b)[1], NDArray_SHAPE(a)[0], NDArray_SHAPE(a)[1],
                1.0, NDArray_DDATA(a), NDArray_SHAPE(a)[1], NDArray_DDATA(b), NDArray_SHAPE(b)[1],
                0.0, NDArray_DDATA(result), NDArray_SHAPE(b)[1]);
    return result;
}

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
        cublasHandle_t handle;
        cublasCreate(&handle);

        float alpha = 1.0, beta = 0.0;

        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N,
                    NDArray_SHAPE(a)[0], NDArray_SHAPE(b)[1], NDArray_SHAPE(a)[1],
                    &alpha, NDArray_FDATA(b), NDArray_SHAPE(b)[1], NDArray_FDATA(a), NDArray_SHAPE(a)[1], &beta,
                    NDArray_FDATA(result), NDArray_SHAPE(b)[1]);
        cublasDestroy(handle);
        return result;
#endif
    }
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                NDArray_SHAPE(b)[1], NDArray_SHAPE(a)[0], NDArray_SHAPE(a)[1],
                1.0f, NDArray_FDATA(a), NDArray_SHAPE(a)[1], NDArray_FDATA(b), NDArray_SHAPE(b)[1],
                0.0f, NDArray_FDATA(result), NDArray_SHAPE(b)[1]);
    return result;
}

void
computeSVD(double* A, int m, int n, double* U, double* S, double* V) {
    int lda = m;
    int ldu = m;
    int ldvt = n;

    int info;

    // Compute workspace size
    double work_size;
    int lwork = -1;  // query the workspace size
#ifdef LAPACK_FORTRAN_STRLEN_END
    dgesvd_("A", "A", &m, &n, A, &lda, S, U, &ldu, V, &ldvt, &work_size, &lwork, &info, 0, 0);
#else
    dgesvd_("A", "A", &m, &n, A, &lda, S, U, &ldu, V, &ldvt, &work_size, &lwork, &info);
#endif
    // Allocate workspace
    lwork = (int)work_size;
    double* work = (double*)emalloc(sizeof(double) * lwork);

    // Compute SVD
#ifdef LAPACK_FORTRAN_STRLEN_END
    dgesvd_("A", "A", &m, &n, A, &lda, S, U, &ldu, V, &ldvt, work, &lwork, &info, 0, 0);
#else
    dgesvd_("A", "A", &m, &n, A, &lda, S, U, &ldu, V, &ldvt, work, &lwork, &info);
#endif

    if (info > 0) {
        printf("SVD computation failed.\n");
        return;
    }

    // Free allocated memory
    efree(work);
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
    dgesvd_("A", "A", &m, &n, A, &lda, S, U, &ldu, V, &ldvt, work, &lwork, &info);
#endif

    if (info > 0) {
        printf("SVD computation failed.\n");
        return;
    }

    // Free allocated memory
    efree(work);
}

/**
 * @return
 */
NDArray**
NDArray_SVD(NDArray *target) {
    NDArray **rtns;
    NDArray *rtn_s, *rtn_u, *rtn_v;
    double *U, *S, *V;
    char *output_data;
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

    if(is_type(NDArray_TYPE(target), NDARRAY_TYPE_DOUBLE64)) {
        S = (double *) emalloc(sizeof(double) * smallest_dim);
        U = (double *) emalloc(sizeof(double) * NDArray_SHAPE(target)[0] * NDArray_SHAPE(target)[0]);
        V = (double *) emalloc(sizeof(double) * NDArray_SHAPE(target)[1] * NDArray_SHAPE(target)[1]);
        output_data = emalloc(sizeof(double) * NDArray_NUMELEMENTS(target));
        memcpy(output_data, NDArray_DDATA(target), sizeof(double) * NDArray_NUMELEMENTS(target));
    } else {
        Sf = (float *) emalloc(sizeof(float) * smallest_dim);
        Uf = (float *) emalloc(sizeof(float) * NDArray_SHAPE(target)[0] * NDArray_SHAPE(target)[0]);
        Vf = (float *) emalloc(sizeof(float) * NDArray_SHAPE(target)[1] * NDArray_SHAPE(target)[1]);
        output_data = emalloc(sizeof(float) * NDArray_NUMELEMENTS(target));
        memcpy(output_data, NDArray_FDATA(target), sizeof(float) * NDArray_NUMELEMENTS(target));
    }

    if(is_type(NDArray_TYPE(target), NDARRAY_TYPE_DOUBLE64)) {
        computeSVD((double*)output_data, NDArray_SHAPE(target)[0], NDArray_SHAPE(target)[1], U, S, V);
    } else {
        computeSVDFloat((float *)output_data, NDArray_SHAPE(target)[0], NDArray_SHAPE(target)[1], Uf, Sf, Vf);
    }

    efree(output_data);
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