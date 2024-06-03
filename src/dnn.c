#include <Zend/zend.h>
#include "ndarray.h"
#include "initializers.h"
#include "dnn.h"
#include "../config.h"

#ifdef HAVE_CUDNN
#include "ndmath/cuda/cuda_dnn.cuh"
#include "manipulation.h"

#endif

#ifdef HAVE_CBLAS
#include <cblas.h>
#endif

void gemm_nn(int M, int N, int K, float ALPHA,
             float *A, int lda,
             float *B, int ldb,
             float *C, int ldc)
{
    int i,j,k;
#pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float A_PART = ALPHA*A[i*lda+k];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}


void gemm_nt(int M, int N, int K, float ALPHA,
             float *A, int lda,
             float *B, int ldb,
             float *C, int ldc)
{
    int i,j,k;
#pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i*lda+k]*B[j*ldb + k];
            }
            C[i*ldc+j] += sum;
        }
    }
}

void gemm_tn(int M, int N, int K, float ALPHA,
             float *A, int lda,
             float *B, int ldb,
             float *C, int ldc)
{
    int i,j,k;
#pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(k = 0; k < K; ++k){
            register float A_PART = ALPHA*A[k*lda+i];
            for(j = 0; j < N; ++j){
                C[i*ldc+j] += A_PART*B[k*ldb+j];
            }
        }
    }
}

void gemm_tt(int M, int N, int K, float ALPHA,
             float *A, int lda,
             float *B, int ldb,
             float *C, int ldc)
{
    int i,j,k;
#pragma omp parallel for
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            register float sum = 0;
            for(k = 0; k < K; ++k){
                sum += ALPHA*A[i+k*lda]*B[k+j*ldb];
            }
            C[i*ldc+j] += sum;
        }
    }
}


void gemm_cpu(int TA, int TB, int M, int N, int K, float ALPHA,
              float *A, int lda,
              float *B, int ldb,
              float BETA,
              float *C, int ldc)
{
    //printf("cpu: %d %d %d %d %d %f %d %d %f %d\n",TA, TB, M, N, K, ALPHA, lda, ldb, BETA, ldc);
    int i, j;
    for(i = 0; i < M; ++i){
        for(j = 0; j < N; ++j){
            C[i*ldc + j] *= BETA;
        }
    }
    if(!TA && !TB)
        gemm_nn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(TA && !TB)
        gemm_tn(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else if(!TA && TB)
        gemm_nt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
    else
        gemm_tt(M, N, K, ALPHA,A,lda, B, ldb,C,ldc);
}

void gemm(int TA, int TB, int M, int N, int K, float ALPHA,
          float *A, int lda,
          float *B, int ldb,
          float BETA,
          float *C, int ldc)
{
    gemm_cpu( TA,  TB,  M, N, K, ALPHA,A,lda, B, ldb,BETA,C,ldc);
}

float im2col_get_pixel(float *im, int height, int width, int channels,
                       int row, int col, int channel, int pad)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
        row >= height || col >= width) return 0;
    return im[col + width*(row + height*channel)];
}

//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE
void im2col_cpu(float* data_im,
                int channels,  int height,  int width,
                int ksize,  int stride, int pad, float* data_col)
{
    int c,h,w;
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                data_col[col_index] = im2col_get_pixel(data_im, height, width, channels,
                                                       im_row, im_col, c_im, pad);
            }
        }
    }
}

void col2im_add_pixel(float *im, int height, int width, int channels,
                      int row, int col, int channel, int pad, float val)
{
    row -= pad;
    col -= pad;

    if (row < 0 || col < 0 ||
        row >= height || col >= width) return;
    im[col + width*(row + height*channel)] += val;
}

void col2im_cpu(float* data_col,
                int channels,  int height,  int width,
                int ksize,  int stride, int pad, float* data_im)
{
    int c,h,w;
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                double val = data_col[col_index];
                col2im_add_pixel(data_im, height, width, channels,
                                 im_row, im_col, c_im, pad, val);
            }
        }
    }
}


float* dnn_conv2d_forward(float *input, int num_filters, int *input_shape,
                          float *kernel, int *kernel_shape, int *strides,
                          char padding, int *dilation_rate, int batch_size,
                          int num_channels, int height, int width, int kernel_size,
                          int num_elements, int *output_shape)
{
    int stride = 1;
    int pad = 0;
    int im_index = 0;

    int m = num_filters;
    int k = kernel_size*kernel_size*num_channels;

    int out_w = (width  + 2 * pad - kernel_size) / stride + 1;
    int out_h = (height + 2 * pad - kernel_size) / stride + 1;
    int n = out_w*out_h;

    float *workspace = emalloc(sizeof(float) * num_channels * out_h * out_w * kernel_size * kernel_size);
    float *output = ecalloc(batch_size * num_filters * n, sizeof(float));

    output_shape[0] = batch_size;
    output_shape[1] = num_filters;
    output_shape[2] = out_h;
    output_shape[3] = out_w;

    for (int i = 0; i < batch_size; i++) {
        float *a = kernel;
        float *b = workspace;
        float *c = output + i * n * m;

        float *im = input + i * num_channels * height * width;
        im2col_cpu(im, num_channels, height, width, kernel_size, stride, pad, b);
        gemm(0,0,m,n,k,1,a,k,b,n,1,c,n);
    }
    efree(workspace);
    return output;
}

float** backward_convolutional_layer(int num_filters, float *grad, int kernel_size, int num_channels, int out_w, int out_h,
                                    int nweights, int input_w, int input_h, float *input, int stride,
                                    int pad, int batch_size, float *weights) {
    float **output = emalloc(sizeof(float *) * 2);
    int i, j;
    int m = num_filters;
    int n = kernel_size*kernel_size*num_channels;
    int k = out_w*out_h;

    float *workspace = ecalloc(batch_size * num_channels * input_w * input_h, sizeof(float));
    float *weight_updates = ecalloc(batch_size * num_channels * input_w * input_h, sizeof(float));

    float *d_weights = ecalloc(batch_size * num_channels * input_w * input_h, sizeof(float));
    for(i = 0; i < batch_size; ++i){
        float *a = grad + i*m*k;
        float *b = workspace;
        float *c = weight_updates;

        float *im = input+(i*num_channels*input_h*input_w);

        im2col_cpu(im, num_channels, input_h, input_w,
                   kernel_size, stride, pad, b);
        // compute gradients w.r.t. weights
        gemm(0,1,m,n,k,1,a,k,b,k,1,c,n);

        a = weights;
        b = grad;
        c = workspace;

        gemm(1,0,n,k,m,1,a,n,b,k,0,c,k);
        col2im_cpu(workspace, num_channels, input_h, input_w, kernel_size, stride,
                   pad, d_weights + (i*num_channels*input_h*input_w));
    }
    output[0] = weight_updates;
    output[1] = d_weights;
    efree(workspace);
    return output;
}

NDArray*
NDArrayDNN_Conv2D_Forward(NDArray *x, NDArray *filters, int *kernel_size, char activation, int use_bias)
{
    NDArray *rtn = NULL;
    if (NDArray_DEVICE(x) == NDARRAY_DEVICE_CPU) {
        int *output_shape = emalloc(sizeof(int) * 4);;
        int nkernel = NDArray_SHAPE(filters)[2];
        int num_channels = NDArray_SHAPE(x)[1];
        int num_filters = NDArray_SHAPE(filters)[3];
        float *data = dnn_conv2d_forward(NDArray_FDATA(x),num_filters, NULL, NDArray_FDATA(filters), NULL, NULL, 'v', NULL, NDArray_SHAPE(x)[0], num_channels,
                                              NDArray_SHAPE(x)[2], NDArray_SHAPE(x)[3], nkernel, NDArray_NUMELEMENTS(x), output_shape);
        rtn = NDArray_Empty(output_shape, 4, NDArray_TYPE(x), NDArray_DEVICE(x));
        NDArray_FREEDATA(rtn);
        rtn->data = (void*)data;
    }

    if (NDArray_DEVICE(x) == NDARRAY_DEVICE_GPU) {
#ifdef HAVE_CUDNN
        int *output_shape = emalloc(sizeof(int) * 4);
        float *output = cuda_dnn_conv2d_float32(
                NDArray_FDATA(x),
                NDArray_SHAPE(x)[3],
                NDArray_NUMELEMENTS(x),
                NDArray_SHAPE(x)[0],
                NDArray_SHAPE(x)[1],
                NDArray_SHAPE(x)[2],
                output_shape,
                4,
                'v');
        rtn = NDArray_Empty(output_shape, 4, NDArray_TYPE(x), NDArray_DEVICE(x));
        rtn->data = (void*)output;
#else
        zend_throw_error(NULL, "DNN features for GPU not enabled. You must compile the NumPower extension in an environment with the cuDNN library available.");
        return NULL;
#endif
    }
    return rtn;
}

NDArray**
NDArrayDNN_Conv2D_Backward(NDArray *input, NDArray *y, NDArray *filters, int kernel_size, char activation, int use_bias)
{
    NDArray **rtn = emalloc(sizeof(NDArray*) * 2);
    NDArray *rtn_dw, *rtn_dinput = NULL, *rtn_temp;
    if (NDArray_DEVICE(input) == NDARRAY_DEVICE_CPU) {
        int stride = 1;
        int pad = 0;
        int nkernel = NDArray_SHAPE(filters)[2];
        int num_channels = NDArray_SHAPE(input)[1];
        int num_filters = NDArray_SHAPE(filters)[0];
        int batch_size = NDArray_SHAPE(input)[0];
        int out_w = (NDArray_SHAPE(input)[2]  + 2 * pad - kernel_size) / stride + 1;
        int out_h = (NDArray_SHAPE(input)[3] + 2 * pad - kernel_size) / stride + 1;
        int input_h = NDArray_SHAPE(input)[2];
        int input_w = NDArray_SHAPE(input)[3];
        int nweights = NDArray_NUMELEMENTS(filters);

        float **outputs = backward_convolutional_layer(num_filters, NDArray_FDATA(y),nkernel,
                                     num_channels, out_w, out_h,
                                     nweights, input_w, input_h, NDArray_FDATA(input),
                                     stride, pad, batch_size, NDArray_FDATA(filters));

        // Build dW
        int *output_shape_dw = emalloc(sizeof(int) * 4);
        memcpy(output_shape_dw, NDArray_SHAPE(filters), sizeof(int) * 4);
        rtn_temp = NDArray_Empty(output_shape_dw, 4, NDArray_TYPE(input), NDArray_DEVICE(input));
        NDArray_FREEDATA(rtn_temp);
        rtn_temp->data = (void*)outputs[0];
        rtn_dw = NDArray_Transpose(rtn_temp);
        NDArray_FREE(rtn_temp);

        // Build dW
        int *output_shape = emalloc(sizeof(int) * 4);
        memcpy(output_shape, NDArray_SHAPE(input), sizeof(int) * 4);
        rtn_dinput = NDArray_Empty(output_shape, 4, NDArray_TYPE(input), NDArray_DEVICE(input));
        NDArray_FREEDATA(rtn_dinput);
        rtn_dinput->data = (void*)outputs[1];

        efree(outputs);
    }

    if (NDArray_DEVICE(input) == NDARRAY_DEVICE_GPU) {
#ifdef HAVE_CUDNN
        int *output_shape = emalloc(sizeof(int) * 4);
        memcpy(output_shape, NDArray_SHAPE(y), NDArray_NDIM(y) * sizeof(int));
        rtn_dw = NDArray_Empty(output_shape, 4, NDArray_TYPE(y), NDArray_DEVICE(y));
        rtn_dw->data = (void*)cuda_dnn_conv2d_float32_backward(NDArray_FDATA(input), NDArray_FDATA(y), NDArray_FDATA(filters), 1, 0, 10, 32, 32, 3, 4, 'v');
#else
        zend_throw_error(NULL, "DNN features for GPU not enabled. You must compile the NumPower extension in an environment with the cuDNN library available.");
        return NULL;
#endif
    }
    rtn[0] = rtn_dw;
    rtn[1] = rtn_dinput;
    return rtn;
}
