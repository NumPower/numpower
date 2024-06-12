#include "../../../config.h"
#include "cuda_dnn.cuh"
#include <cuda_runtime.h>
#include <cuda.h>
#include <vector>
#include <stdio.h>

#ifdef HAVE_CUDNN

#include <cudnn.h>

#define checkCUDNN(expression)                               \
  {                                                          \
    cudnnStatus_t status = (expression);                     \
    if (status != CUDNN_STATUS_SUCCESS) {                    \
      printf("\nError on line %d (%d)", __LINE__, status);     \
    }                                                        \
  }

float*
cuda_dnn_conv2d_float32(float *input, int num_channels, int num_elements, int batch_size, int height, int width,
                        int *output_shape, int kernel_size, char padding) {
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
            /*format=*/CUDNN_TENSOR_NHWC,
            /*dataType=*/CUDNN_DATA_FLOAT,
            /*batch_size=*/batch_size,
            /*channels=*/num_channels,
            /*image_height=*/height,
            /*image_width=*/width));

    cudnnFilterDescriptor_t kernel_descriptor;
    checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
    checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
            /*dataType=*/CUDNN_DATA_FLOAT,
            /*format=*/CUDNN_TENSOR_NCHW,
            /*out_channels=*/kernel_size,
            /*in_channels=*/num_channels,
            /*kernel_height=*/3,
            /*kernel_width=*/3));

    int pad_height;
    int pad_width;
    switch (padding) {
        case 'v':
            pad_height = 0;
            pad_width = 0;
            break;
        case 's':
            pad_height = 1;
            pad_width = 1;
            break;
        default:
            pad_height = 0;
            pad_width = 0;
    }

    cudnnConvolutionDescriptor_t convolution_descriptor;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
    checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
            /*pad_height=*/pad_height,
            /*pad_width=*/pad_width,
            /*vertical_stride=*/1,
            /*horizontal_stride=*/1,
            /*dilation_height=*/1,
            /*dilation_width=*/1,
            /*mode=*/CUDNN_CROSS_CORRELATION,
            /*computeType=*/CUDNN_DATA_FLOAT));

    int new_batch_size, new_channels, new_height, new_width;
    checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convolution_descriptor,
                                                     input_descriptor,
                                                     kernel_descriptor,
                                                     &new_batch_size,
                                                     &new_channels,
                                                     &new_height,
                                                     &new_width));

    cudnnTensorDescriptor_t output_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
            /*format=*/CUDNN_TENSOR_NCHW,
            /*dataType=*/CUDNN_DATA_FLOAT,
            /*batch_size=*/batch_size,
            /*channels=*/kernel_size,
            /*image_height=*/new_height,
            /*image_width=*/new_width));

    cudnnConvolutionFwdAlgoPerf_t *convolution_algorithm = (cudnnConvolutionFwdAlgoPerf_t*)malloc(sizeof(cudnnConvolutionFwdAlgoPerf_t));
    int returned_algo_count = 0;
    checkCUDNN(
            cudnnGetConvolutionForwardAlgorithm_v7(cudnn,
                                                input_descriptor,
                                                kernel_descriptor,
                                                convolution_descriptor,
                                                output_descriptor,
                                                1,
                                                &returned_algo_count,
                                                convolution_algorithm));

    size_t workspace_bytes;
    checkCUDNN(cudnnGetConvolutionForwardWorkspaceSize(cudnn,
                                                       input_descriptor,
                                                       kernel_descriptor,
                                                       convolution_descriptor,
                                                       output_descriptor,
                                                       convolution_algorithm[0].algo,
                                                       &workspace_bytes));

    void* d_workspace = NULL;
    if (workspace_bytes > 0) {
        cudaMalloc(&d_workspace, workspace_bytes);
    }

    float *d_input = input;

    int image_bytes = batch_size * new_channels * new_height * new_width * sizeof(float);
    float* d_output{nullptr};
    cudaMalloc(&d_output, image_bytes);
    cudaMemset(d_output, 0, image_bytes);

    // Mystery kernel
    const float kernel_template[3][3] = {
            {0, -1, 0},
            {-1, 4, -1},
            {0,  -1, 0}
    };

    float h_kernel[kernel_size][3][3][3];
    for (int kernel = 0; kernel < kernel_size; ++kernel) {
        for (int channel = 0; channel < 3; ++channel) {
            for (int row = 0; row < 3; ++row) {
                for (int column = 0; column < 3; ++column) {
                    h_kernel[kernel][channel][row][column] = kernel_template[row][column];
                }
            }
        }
    }


    float* d_kernel{nullptr};
    cudaMalloc(&d_kernel, sizeof(h_kernel));
    cudaMemcpy(d_kernel, h_kernel, sizeof(h_kernel), cudaMemcpyHostToDevice);

    const float alpha = 1, beta = 0;
    checkCUDNN(cudnnConvolutionForward(cudnn,
                                       &alpha,
                                       input_descriptor,
                                       d_input,
                                       kernel_descriptor,
                                       d_kernel,
                                       convolution_descriptor,
                                       convolution_algorithm[0].algo,
                                       d_workspace,
                                       workspace_bytes,
                                       &beta,
                                       output_descriptor,
                                       d_output));
    output_shape[0] = batch_size;
    output_shape[1] = new_height;
    output_shape[2] = new_width;
    output_shape[3] = new_channels;
    return d_output;
}


float*
cuda_dnn_conv2d_float32_backward(float *data_output,
                                 float *data_input,
                                 float *data_filter,
                                 float alpha,
                                 float beta,
                                 int batch_size,
                                 int height,
                                 int width,
                                 int num_channels,
                                 int kernel_size,
                                 char padding)
{
    cudnnHandle_t cudnn;
    cudnnCreate(&cudnn);

    cudnnConvolutionBwdDataAlgo_t algo = CUDNN_CONVOLUTION_BWD_DATA_ALGO_1;

    void *workSpace = 0;
    size_t workSpaceSize;

    cudnnTensorDescriptor_t input_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&input_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(input_descriptor,
            /*format=*/CUDNN_TENSOR_NHWC,
            /*dataType=*/CUDNN_DATA_FLOAT,
            /*batch_size=*/batch_size,
            /*channels=*/num_channels,
            /*image_height=*/height,
            /*image_width=*/width));

    cudnnFilterDescriptor_t kernel_descriptor;
    checkCUDNN(cudnnCreateFilterDescriptor(&kernel_descriptor));
    checkCUDNN(cudnnSetFilter4dDescriptor(kernel_descriptor,
            /*dataType=*/CUDNN_DATA_FLOAT,
            /*format=*/CUDNN_TENSOR_NCHW,
            /*out_channels=*/kernel_size,
            /*in_channels=*/num_channels,
            /*kernel_height=*/3,
            /*kernel_width=*/3));

    int pad_height;
    int pad_width;
    switch (padding) {
        case 'v':
            pad_height = 0;
            pad_width = 0;
            break;
        case 's':
            pad_height = 1;
            pad_width = 1;
            break;
        default:
            pad_height = 0;
            pad_width = 0;
    }

    cudnnConvolutionDescriptor_t convolution_descriptor;
    checkCUDNN(cudnnCreateConvolutionDescriptor(&convolution_descriptor));
    checkCUDNN(cudnnSetConvolution2dDescriptor(convolution_descriptor,
            /*pad_height=*/pad_height,
            /*pad_width=*/pad_width,
            /*vertical_stride=*/1,
            /*horizontal_stride=*/1,
            /*dilation_height=*/1,
            /*dilation_width=*/1,
            /*mode=*/CUDNN_CONVOLUTION,
            /*computeType=*/CUDNN_DATA_FLOAT));

    int new_batch_size, new_channels, new_height, new_width;
    checkCUDNN(cudnnGetConvolution2dForwardOutputDim(convolution_descriptor,
                                                     input_descriptor,
                                                     kernel_descriptor,
                                                     &new_batch_size,
                                                     &new_channels,
                                                     &new_height,
                                                     &new_width));

    cudnnTensorDescriptor_t output_descriptor;
    checkCUDNN(cudnnCreateTensorDescriptor(&output_descriptor));
    checkCUDNN(cudnnSetTensor4dDescriptor(output_descriptor,
            /*format=*/CUDNN_TENSOR_NCHW,
            /*dataType=*/CUDNN_DATA_FLOAT,
            /*batch_size=*/batch_size,
            /*channels=*/kernel_size,
            /*image_height=*/new_height,
            /*image_width=*/new_width));

    checkCUDNN ( cudnnGetConvolutionBackwardDataWorkspaceSize(cudnn, kernel_descriptor, output_descriptor, convolution_descriptor,
                                                              input_descriptor, algo, &workSpaceSize) );

    if (workSpaceSize > 0) {
        cudaMalloc(&workSpace, workSpaceSize);
    }
    checkCUDNN ( cudnnConvolutionBackwardData (cudnn,
                                                  (void*)(&alpha),
                                                  kernel_descriptor, data_filter,
                                                  output_descriptor, data_output,
                                                  convolution_descriptor,
                                                  algo,
                                                  workSpace, workSpaceSize,
                                                  (void*)(&beta),
                                                  input_descriptor, data_input) );
    if (workSpace) {
        cudaFree(workSpace);
        workSpace = 0;
    }
    return data_input;
}

#endif