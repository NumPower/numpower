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

| **Method**         | **Rubix Tensor** | **NumPower (AVX2)** | **NumPower (GPU)** | 
|--------------------|------------------|---------------------|--------------------|
| add    (4096x4096) | 0.382526863s     | 0.03754547s         | 0.0004980564s      | 
| sum    (2048x2048) | 0.016164064s     | 0.00074601s         | 0.0001361370s      |
| abs    (2048x2048) | 0.092210054s     | 0.01689696s         | 0.0003008842s      |
| matmul (1024x1024) | 19.03744602s     | 0.02015590s         | 0.0051181316s      |   
| log1p  (2048x2018) | 0.102784519s     | 0.03342665s         | 0.0003099441s      |   
| svd    (4096x4096) | --               | 32.0257899s         | 0.3561830520s      |
| max    (4096x4096) | 0.054986000s     | 0.03053498s         | 0.0007658004s      |
| equals (4096x4096) | 0.422759056s     | 0.03017616s         | 0.0004699230s      |
| fill   (4096x4096) | 0.000167846s     | 0.02678418s         | 0.0002639293s      |

>i7 9700k 32GB RTX 2070 SUPER 8GB VRAM
