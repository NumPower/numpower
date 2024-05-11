<p align="center">
  <img src="https://github.com/NumPower/numpower/assets/1107499/ea2e8895-a1ab-4212-bd91-033e9afa711b" width="200" height="200">
</p>
<h1 align="center">NumPower</h1>
<p align="center">
<a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.demolab.com?font=Fira+Code&duration=2000&pause=500&multiline=true&width=435&height=100&separator=%3C&lines=%24a+%3D+nd%3A%3Aarray(%5B%5B1%2C+2%5D%2C+%5B3%2C+4%5D%5D);%3C%24b+%3D+%24a+*+2;%3Cprint_r(%24b);" alt="Typing SVG" /></a>
</p>

Inspired by NumPy, the NumPower library was created to provide the foundation for efficient scientific computing in PHP, as well as leverage the machine learning tools and libraries that already exist and can benefit from it.
<p></p>
This C extension developed for PHP can be used to considerably speed up mathematical operations on large datasets and facilitate the manipulation, creation and operation of N-dimensional tensors.
<p></p>
NumPower was designed from the ground up to utilize AVX2 and the GPU to further improve performance. With the use of contiguous single precision arrays, slices, buffer sharing and a specific GC engine, 
NumPower aims to manage memory more efficiently than a matrix in PHP arrays

<p></p>
<p align="center">
<a href="https://git.io/typing-svg"><img src="https://readme-typing-svg.demolab.com?font=Fira+Code&duration=2000&pause=500&multiline=true&width=435&height=100&separator=%3C&lines=%24a_gpu+%3D+%24a-%3Egpu();%3C%24b_gpu+%3D+%24b-%3Egpu();%3C%24m+%3D+nd%3A%3Amatmul(%24a_gpu%2C+%24b_gpu);" alt="Typing SVG" /></a>
</p>

## Documentation & Installation

- Website: https://numpower.org
- Documentation: https://numpower.org/docs/intro
- API: https://numpower.org/api/intro
- Docker: https://hub.docker.com/r/numpower/numpower

## Requirements
- PHP 8.x
- LAPACKE
- OpenBLAS
- **Optional (GPU)**: CUBLAS, CUDA Build Toolkit
- **Optional (Image)**: PHP-GD

## GPU Support

If you have an NVIDIA graphics card with CUDA support, you can use your graphics card 
to perform operations. To do this, just copy your array to the GPU memory.

```php
use \NDArray as nd;

$x = nd::ones([10, 10]);
$y = nd::ones([10, 10]);

$x_gpu = $x->gpu();   // Copy $x from RAM to VRAM
$y_gpu = $y->gpu();   // Copy $y from RAM to VRAM

$r = nd::matmul($x_gpu, $y_gpu); // Matmul is performed using CUDA
```

Both GPU and CPU memory management are done automatically by NumPower, so the memory of both devices will be 
automatically freed by the garbage collector.  You can also bring arrays back from VRAM into RAM:

```php 
$x_cpu = $x->cpu();
```

> **You must explicitly copy the arrays you want to use in your devices**. Cross-array operations (like adding) will 
> raise an exception if the arrays used are on different devices.
