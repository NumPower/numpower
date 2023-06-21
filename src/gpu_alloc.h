#ifndef NUMPOWER_GPU_ALLOC_H
#define NUMPOWER_GPU_ALLOC_H

void NDArray_VMALLOC(void** target, unsigned int size);
void NDArray_VFREE(void* target);
void NDArray_VCHECK();

#endif //NUMPOWER_GPU_ALLOC_H
