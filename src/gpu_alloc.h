#ifndef NUMPOWER_GPU_ALLOC_H
#define NUMPOWER_GPU_ALLOC_H

#ifdef __cplusplus
extern "C" {
#endif

void vmalloc(void **target, unsigned int size);
void vfree(void *target);
void vmemcheck();
void vmemcpyd2d(char* target, char* dst, unsigned int size);
void vmemcpyh2d(char* target, char* dst, unsigned int size);

float NDArray_VFLOAT(char *target);
float NDArray_VFLOATF_I(float *target, int index);

#ifdef __cplusplus
}
#endif

#endif //NUMPOWER_GPU_ALLOC_H
