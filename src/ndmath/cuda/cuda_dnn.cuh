#ifndef NUMPOWER_CUDA_DNN_CUH
#define NUMPOWER_CUDA_DNN_CUH

#ifdef __cplusplus
extern "C" {
#endif

float* cuda_dnn_conv2d_float32(float *input, int num_channels, int num_elements, int batch_size, int height, int width,
                        int *output_shape, int kernel_size, char padding) ;

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
                                 char padding);
#ifdef __cplusplus
}
#endif
#endif //NUMPOWER_CUDA_DNN_CUH
