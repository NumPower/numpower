> 🚧 **UNDER CONSTRUCTION** 🚧

## Requirements
- LAPACKE
- OpenBLAS
- **Optional (GPU)**: CUBLAS, CUDA

## GPU Support

If you have an NVIDIA graphics card with CUDA support, you can use your graphics card 
to perform operations. To do this, just copy your array to the GPU memory.

```php 
$x = NDArray::ones([10, 10]);
$y = NDArray::ones([10, 10]);

$x_gpu = $x->gpu();   // Copy $x from RAM to VRAM
$y_gpu = $y->gpu();   // Copy $y from RAM to VRAM

$r = NDArray::matmul($x_gpu, $y_gpu); // Matmul is performed using CUDA
```

Both GPU and CPU memory management are done automatically by NumPower, so the memory of both devices will be 
automatically freed by the garbage collector.  You can also bring arrays back from VRAM into RAM:

```php 
$x_cpu = $x->cpu();
```

> **You must explicitly copy the arrays you want to use in your devices**. Cross-array operations (like adding) will 
> raise an exception if the arrays used are on different devices.

## Benchmark (Preview)
**Benchmark without much credibility (without using specific tools or standards)**. PHPBench will be used instead.

| **Method**         | **Rubix Tensor** | **NumPower (AVX2)** | 
|--------------------|------------------|---------------------|
| add    (4096x4096) | 0.382526863s     | 0.037545547s        | 
| matmul (512x512)   | 2.399778589s     | 0.021467702s        |
| log1p  (2048x2018) | 0.102784519s     | 0.03342665s         |
| svd    (1024x1024) | --               | 0.3374907970s       |
i7 9900k 32GB
