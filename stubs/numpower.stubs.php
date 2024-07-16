<?php
final class NDArray {

    /**
     * Copy the NDArray to the CPU for computation. If the NDArray is already in RAM, a copy will still be made.
     *
     * @return void
     */
    public function cpu(): void {}

    /**
     * Copy the NDArray to the GPU for computation. If the NDArray is already in VRAM, a copy will still be made.
     *
     * @return void
     */
    public function gpu(): void {}

    /**
     * Check if the NDArray is stored on the GPU.
     *
     * @return bool
     */
    public function isGPU(): bool {}

    /**
     * Specifies which GPU device to use by ID. By default, all operations are performed on GPU id = 0.
     * Use the dumpDevices method if you want to check the ID in a multi-GPU environment.
     *
     * @param int $deviceId
     * @return void
     */
    public static function setDevice(int $deviceId): void {}

    /**
     * Add arguments element-wise
     *
     * Same as $a + $b
     *
     * @param NDArray|array|float|int $a The array to be added
     * @param NDArray|array|float|int $b The array to be added
     * @return NDArray|float|int The sum of $a and $b
     */
    public static function add(NDArray|array|float|int $a, NDArray|array|float|int $b): NDArray|float|int {}

    /**
     * Return the division between two arrays element-wise
     *
     * Same as $a / $b
     *
     * @param NDArray|array|float|int $a Dividend
     * @param NDArray|array|float|int $b Divisor
     * @return NDArray|float|int Array with the division between $a and $b element-wise
     */
    public static function divide(NDArray|array|float|int $a, NDArray|array|float|int $b): NDArray|float|int {}


    /**
     * Performs element-wise modulo operation between two arrays and returns a new array
     * containing the result. The modulo operation calculates the remainder after division.
     *
     * Same as $a @ $b
     *
     * @param NDArray|array|float|int $a
     * @param NDArray|array|float|int $b
     * @return NDArray|float|int
     */
    public static function mod(NDArray|array|float|int $a, NDArray|array|float|int $b): NDArray|float|int {}

    /**
     * Multiply arrays element-wise
     *
     * Same as $a * $b
     *
     * @param NDArray|array|float|int $a The arrays to be multiplied.
     * @param NDArray|array|float|int $b The arrays to be multiplied.
     * @return NDArray|float|int The multiplication of $a and $b element-wise
     */
    public static function multiply(NDArray|array|float|int $a, NDArray|array|float|int $b): NDArray|float|int {}

    /**
     * Computes the element-wise negation (unary minus) of an array, returning a new array with the negation of each element.
     *
     * Same as -$a
     *
     * @param NDArray|array|float|int $a Input array
     * @return NDArray|float|int The multiplication of $a * -1
     */
    public static function negative(NDArray|array|float|int $a): NDArray|float|int {}

    /**
     * Numerical positive, element-wise.
     *
     * @param NDArray|array|float|int $a Input array
     * @return NDArray|float|int
     */
    public static function positive(NDArray|array|float|int $a): NDArray|float|int {}

    /**
     * Return the reciprocal of the argument, element-wise.
     *
     * Calculates `1 / $a`
     *
     * @param NDArray|array|float|int $a Input array
     * @return NDArray|float|int
     */
    public static function reciprocal(NDArray|array|float|int $a): NDArray|float|int {}

    /**
     * Raises each element of an array $a to a specified power $b and returns a new array containing the result.
     *
     * Same as $a ** $b;
     *
     * @param NDArray|array|float|int $a Input array
     * @param NDArray|array|float|int $b Input array
     * @return NDArray|float|int $a ** $b
     */
    public static function pow(NDArray|array|float|int $a, NDArray|array|float|int $b): NDArray|float|int {};

    /**
     * Subtract two arrays element-wise
     *
     * @param NDArray|array|float|int $a Input array
     * @param NDArray|array|float|int $b Input array
     * @return NDArray|float|int $a - $b
     */
    public static function subtract(NDArray|array|float|int $a, NDArray|array|float|int $b): NDArray|float|int {}

    /**
     * Computes the element-wise exponential function of an array, returning
     * a new array with each element raised to the power of $array.
     *
     * @param NDArray|array|float|int $array Input array
     * @return NDArray|float|int
     */
    public static function exp(NDArray|array|float|int $array): NDArray|float|int {}

    /**
     * Computes the element-wise 2 raised to the power of an array,
     * returning a new array with each element raised to the power of 2.
     *
     * @param NDArray|array|float|int $array Input array
     * @return NDArray|float|int
     */
    public static function exp2(NDArray|array|float|int $array): NDArray|float|int {}

    /**
     * Calculates the element-wise exponential minus one function, returning
     * a new array with each element raised to the power of `$array` - 1.
     *
     * @param NDArray|array|float|int $array Input array
     * @return NDArray|float|int
     */
    public static function expm1(NDArray|array|float|int $array): NDArray|float|int {}

    /**
     * Calculates the element-wise natural logarithm of an array, returning
     * a new array with the natural logarithm of each element.
     *
     * @param NDArray|array|float|int $array Input array
     * @return NDArray|float|int
     */
    public static function log(NDArray|array|float|int $array): NDArray|float|int {}

    /**
     * Computes the element-wise logarithm of one plus an array, returning a new array with
     * the natural logarithm of each element plus one.
     *
     * @param NDArray|array|float|int $array Input array
     * @return NDArray|float|int
     */
    public static function log1p(NDArray|array|float|int $array): NDArray|float|int {}

    /**
     * Computes the element-wise base-2 logarithm of an array,
     * returning a new array with the base-2 logarithm of each element
     *
     * @param NDArray|array|float|int $array Input array
     * @return NDArray|float|int
     */
    public static function log2(NDArray|array|float|int $array): NDArray|float|int {}

    /**
     * Calculates the element-wise base-10 logarithm of an array,
     * returning a new array with the base-10 logarithm of each element.
     *
     * @param NDArray|array|float|int $array Input array
     * @return NDArray|float|int
     */
    public static function log10(NDArray|array|float|int $array): NDArray|float|int {}

    /**
     * Computes the element-wise logarithm base b of an array, returning
     * a new array with the logarithm base b of each element.
     *
     * @param NDArray|array|float|int $array Input array
     * @return NDArray|float|int
     */
    public static function logb(NDArray|array|float|int $array): NDArray|float|int {}

    /**
     * Finds the maximum value in the array.
     *
     * @param NDArray|array|float|int $a Input array
     * @return float|int
     */
    public static function max(NDArray|array|float|int $a): float|int {}

    /**
     * Finds the minimum value in the array.
     *
     * @param NDArray|array|float|int $a Input array
     * @return float|int
     */
    public static function min(NDArray|array|float|int $a): float|int {}

    /**
     * Calculates the element-wise inverse hyperbolic cosine (arccosineh) of an array,
     * returning a new array with the inverse hyperbolic cosine of each element.
     *
     * @param NDArray|array|float|int $array Input array
     * @return NDArray|float|int
     */
    public static function arccosh(NDArray|array|float|int $array): NDArray|float|int {}

    /**
     * Computes the element-wise inverse hyperbolic sine (arcsineh) of an array, returning a new
     * array with the inverse hyperbolic sine of each element.
     *
     * @param NDArray|array|float|int $array Input array
     * @return NDArray|float|int
     */
    public static function arcsinh(NDArray|array|float|int $array): NDArray|float|int {}

    /**
     * Computes the element-wise inverse hyperbolic tangent (arctangenth) of an array,
     * returning a new array with the inverse hyperbolic tangent of each element.
     *
     * @param NDArray|array|float|int $array Input array
     * @return NDArray|float|int
     */
    public static function arctanh(NDArray|array|float|int $array): NDArray|float|int {}

    /**
     * Computes the element-wise hyperbolic cosine of an array,
     * returning a new array with the hyperbolic cosine of each element.
     *
     * @param NDArray|array|float|int $array Input array
     * @return NDArray|float|int
     */
    public static function cosh(NDArray|array|float|int $array): NDArray|float|int {}

    /**
     * Calculates the element-wise inverse hyperbolic cosine (arccosineh) of an array,
     * returning a new array with the inverse hyperbolic cosine of each element.
     *
     * @param NDArray|array|float|int $array Input array
     * @return NDArray|float|int
     */
    public static function sinh(NDArray|array|float|int $array): NDArray|float|int {}

    /**
     * Calculates the element-wise hyperbolic tangent of an array,
     * returning a new array with the hyperbolic tangent of each element.
     *
     * @param NDArray|array|float|int $array Input array
     * @return NDArray|float|int
     */
    public static function tanh(NDArray|array|float|int $array): NDArray|float|int {}

    /**
     * Computes the element-wise absolute value of an array, returning a new array with non-negative elements.
     *
     * @param NDArray|array|float|int $array Input array
     * @return NDArray|float|int
     */
    public static function abs(NDArray|array|float|int $array): NDArray|float|int {}

    /**
     * Clips the values of an array between a minimum and maximum value, returning
     * a new array with the values clipped within the specified range.
     *
     * @param NDArray|array|float|int $array Input array
     * @param float $min Minimum value
     * @param float $max Maximum value
     * @return NDArray|float|int clipped $array
     */
    public static function clip(NDArray|array|float|int $array, float $min, float $max): NDArray|float|int {}

    /**
     * Computes the element-wise sign of an array, returning a new array
     * with the sign of each element (1 for positive, -1 for negative, 0 for zero).
     *
     * @param NDArray|array|float|int $array Input array
     * @return NDArray|float|int
     */
    public static function sign(NDArray|array|float|int $array): NDArray|float|int {}

    /**
     * Calculates the element-wise sinc function of an array,
     * returning a new array with the sinc function evaluated for each element.
     *
     * @param NDArray|array|float|int $array Input array
     * @return NDArray|float|int
     */
    public static function sinc(NDArray|array|float|int $array): NDArray|float|int {}

    /**
     * Calculates the element-wise square root of an array, returning a new array
     * with the positive square root of each element.
     *
     * @param NDArray|array|float|int $array Input array
     * @return NDArray|float|int
     */
    public static function sqrt(NDArray|array|float|int $array): NDArray|float|int {}

    /**
     * Computes the element-wise square of an array, returning a new array with each element squared.
     *
     * @param NDArray|array|float|int $array Input array
     * @return NDArray|float|int
     */
    public static function square(NDArray|array|float|int $array): NDArray|float|int {}

    /**
     * Rounds the elements of an array to the nearest integer greater than or equal to the element,
     * returning a new array with the elements rounded upwards.
     *
     * @param NDArray|array|float|int $array Input array
     * @return NDArray|float|int
     */
    public static function ceil(NDArray|array|float|int $array): NDArray|float|int {}

    /**
     * Rounds the elements of an array towards zero, returning a new array with the elements rounded towards zero.
     *
     * @param NDArray|array|float|int $array Input array
     * @return NDArray|float|int
     */
    public static function fix(NDArray|array|float|int $array): NDArray|float|int {}

    /**
     * Rounds the elements of an array to the nearest integer less than or equal to the element,
     * returning a new array with the elements rounded downwards.
     *
     * @param NDArray|array|float|int $array Input array
     * @return NDArray|float|int
     */
    public static function floor(NDArray|array|float|int $array): NDArray|float|int {}

    /**
     * Rounds the elements of an array to the nearest integer,
     * returning a new array with the elements rounded to the nearest integer.
     *
     * @param NDArray|array|float|int $array Input array
     * @return NDArray|float|int
     */
    public static function rint(NDArray|array|float|int $array): NDArray|float|int {}

    /**
     * Rounds the elements of an array to the nearest integer,
     * returning a new array with the elements rounded to the nearest integer.
     *
     * @param NDArray|array|float|int $array Input array
     * @return NDArray|float|int
     */
    public static function round(NDArray|array|float|int $array): NDArray|float|int {}

    /**
     * Truncates the elements of an array towards zero, returning a
     * new array with the elements truncated towards zero.
     *
     * @param NDArray|array|float|int $array Input array
     * @return NDArray|float|int
     */
    public static function trunc(NDArray|array|float|int $array): NDArray|float|int {}

    /**
     * Calculates the product of all elements in the array over a given axis
     * along which a product is performed. The default, axis=NULL,
     * will calculate the product of all the elements in the input array.
     *
     * @param NDArray|array|float|int $a Input array
     * @param int|null $axis The axis to perform the product. If `$axis` is NULL, will calculate the product of all the elements of `$a`.
     * @return NDArray|float|int The product of `$a`. If `$axis` is not NULL, the specified axis is removed.
     */
    public static function prod(NDArray|array|float|int $a, ?int $axis = NULL): NDArray|float|int {}

    /**
     * Calculates the sum of all elements in the array over a given axis
     * along which a sum is performed. The default, axis=NULL,
     * will calculate the product of all the elements in the input array.
     *
     * @param NDArray|array|float|int $a Input array
     * @param int|null $axis Specifies the axis along which the sum is performed. By default, ($axis=NULL),
     * the function sums all elements of the input array.
     * @return NDArray|float|int The function returns the summed array along the
     * specified axis, resulting in an array with the same shape as the input array,
     * but with the specified axis removed. If the input array is 0-dimensional
     * or if axis=NULL, a scalar value is returned.
     */
    public static function sum(NDArray|array|float|int $a, ?int $axis = NULL): NDArray|float|int {}

    /**
     * Calculates the element-wise inverse cosine (arccosine) of an array,
     * returning a new array with the arccosine of each element.
     *
     * @param NDArray|array|float|int $array Input array
     * @return NDArray|float|int
     */
    public static function arccos(NDArray|array|float|int $array): NDArray|float|int {}

    /**
     * Computes the element-wise inverse sine (arcsine) of an array,
     * returning a new array with the arcsine of each element.
     *
     * @param NDArray|array|float|int $array Input array
     * @return NDArray|float|int
     */
    public static function arcsin(NDArray|array|float|int $array): NDArray|float|int {}

    /**
     * Computes the element-wise inverse tangent (arctangent) of an array,
     * returning a new array with the arctangent of each element.
     *
     * @param NDArray|array|float|int $array Input array
     * @return NDArray|float|int
     */
    public static function arctan(NDArray|array|float|int $array): NDArray|float|int {}

    /**
     * Computes the element-wise cosine of an array, returning
     * a new array with the cosine of each element.
     *
     * @param NDArray|array|float|int $array Input array
     * @return NDArray|float|int
     */
    public static function cos(NDArray|array|float|int $array): NDArray|float|int {}

    /**
     * Converts the element-wise angle from radians to degrees,
     * returning a new array with the angles converted to degrees.
     *
     * @param NDArray|array|float|int $array Input array
     * @return NDArray|float|int
     */
    public static function degrees(NDArray|array|float|int $array): NDArray|float|int {}

    /**
     * Converts the element-wise angle from degrees to radians,
     * returning a new array with the angles converted to radians.
     *
     * @param NDArray|array|float|int $array Input array
     * @return NDArray|float|int
     */
    public static function radians(NDArray|array|float|int $array): NDArray|float|int {}

    /**
     * Calculates the element-wise sine of an array,
     * returning a new array with the sine of each element.
     *
     * @param NDArray|array|float|int $array Input array
     * @return NDArray|float|int
     */
    public static function sin(NDArray|array|float|int $array): NDArray|float|int {}

    /**
     * Generates an array of random numbers from a normal distribution. The normal distribution, also
     * known as the Gaussian distribution, is a continuous probability distribution that is symmetric and bell-shaped.
     *
     * @param array $size
     * @param float $loc
     * @param float $scale
     * @return NDArray
     */
    public static function normal(array $size, float $loc = 0.0, float $scale = 1.0): NDArray {}

    /**
     * Generates an array of random integers from a Poisson distribution.
     * The Poisson distribution models the number of events occurring in fixed intervals of time
     * or space, given the average rate of occurrence.
     *
     * @param array $size
     * @param float $lam
     * @return NDArray
     */
    public static function poisson(array $size, float $lam = 1.0): NDArray {}

    /**
     * Generates an array of random numbers from the standard normal distribution.
     * The standard normal distribution is a special case of the normal distribution with mean (μ) equal to
     * 0 and standard deviation (σ) equal to 1.
     *
     * @param array $size
     * @return NDArray
     */
    public static function standard_normal(array $size): NDArray {}

    /**
     * Generates an array of random numbers from a uniform distribution. The uniform distribution provides an equal
     * probability for each value within a specified range.
     *
     * @param array $size
     * @param float $low
     * @param float $high
     * @return NDArray
     */
    public static function uniform(array $size, float $low = 0.0, float $high = 1.0): NDArray {}

    /**
     * Convolve two 2-dimensional arrays.
     *
     * Convolve `$a` and `$b` with output size determined by `$mode`, and
     * boundary conditions determined by `$boundary` and `$fill_value`.
     *
     * #### $mode options
     *
     * - **full** - Full discrete linear convolution of the inputs
     * - **valid** - The output consists only of those elements that do not rely on the zero-padding. In ‘valid’ mode, either `$a` or `$b` must be at least as large as the other in every dimension.
     * - **same** - The output is the same size as `$a`, centered with respect to the ‘full’ output.
     *
     * #### $boundary options
     *
     * - **fill** - Pad input arrays with `$fill_value`
     * - **wrap** - Circular boundary
     * - **symm** - Symmetrical boundary
     *
     * @param NDArray|array $a The array to perform the convolution.
     * @param NDArray|array $b The kernel.
     * @param string $mode The size of the output. Can be: full, valid and same
     * @param string $boundary A flag indicating how to handle boundaries. Can be: fill, wrap and symm
     * @param float $fill_value Fill value (when $boundary = 'fill')
     * @return NDArray
     */
    public static function convolve2d(NDArray|array $a, NDArray|array $b, string $mode, string $boundary, float $fill_value = 0.0): NDArray {}

    /**
     * The weighted average of the elements in the array. It allows the user to specify weights
     * for each element to be considered in the computation of the average.
     *
     * @param NDArray|array|float|int $array
     * @param NDArray|array|float|int|null $weights
     * @return float|int
     */
    public static function average(NDArray|array|float|int $array, NDArray|array|float|int|null $weights = NULL): float|int {}

    /**
     * The arithmetic mean of the elements in the array. It computes the sum
     * of all values and then divides it by the total number of elements in the array.
     *
     * Same as calling `nd::sum($a) / $a->size()`
     *
     * @param NDArray|array|float|int $a
     * @return float|int
     */
    public static function mean(NDArray|array|float|int $a): float|int {}

    /**
     * The median of the elements in the array. It sorts the array, and if the number of elements is odd,
     * it returns the middle value; if the number of elements is even, it returns the average of the two middle values
     *
     * @param NDArray|array|float|int $a
     * @return float|int
     */
    public static function median(NDArray|array|float|int $a): float|int {}

    /**
     * Computes the specified quantile of the elements in the array. A quantile represents a
     * particular value below which a given percentage of data falls. For example,
     * the median is the 50th quantile.
     *
     * @param NDArray|array|float|int $a
     * @param float|int $q
     * @return float|int
     */
    public static function quantile(NDArray|array|float|int $a, float|int $q): float|int {}

    /**
     * Calculates the standard deviation of the elements in the array. It is the
     * square root of the variance and provides a measure of the amount of variation
     * or dispersion in the data.
     *
     * @param NDArray|array|float|int $a
     * @return float|int
     */
    public static function std(NDArray|array|float|int $a): float|int {}

    /**
     * Calculates the variance of the elements in the array. It measures the average of the
     * squared differences between each element and the mean.
     *
     * @param NDArray|array|float|int $array
     * @return float|int
     */
    public static function variance(NDArray|array|float|int $array): float|int {}

    /**
     * Convert inputs to arrays with at least one dimension.
     *
     * Scalar inputs are converted to 1-dimensional arrays, whilst higher-dimensional inputs are preserved.
     *
     * @param NDArray|array|float|int $array
     * @return NDArray
     */
    public static function atleast_1d(NDArray|array|float|int $array): NDArray {}

    /**
     * Convert inputs to arrays with at least two dimensions.
     *
     * @param NDArray|array|float|int $array
     * @return NDArray
     */
    public static function atleast_2d(NDArray|array|float|int $array): NDArray {}

    /**
     * Convert inputs to arrays with at least three dimensions.
     *
     * @param NDArray|array|float|int $array
     * @return NDArray
     */
    public static function atleast_3d(NDArray|array|float|int $array): NDArray {}

    /**
     * Create a copy of array `$a`.
     *
     * #### $device options
     * Default = NULL, copy on the same device as `$a`
     *
     * - **0** - CPU copy (same as `$a->cpu()`);
     * - **1** - GPU copy (same as `$a->gpu()`);
     *
     * @param NDArray|array|float|int $a
     * @param int|null $device NULL = Same device, 0 - copy to CPU, 1 - copy to GPU
     * @return NDArray copy of $a
     */
    public static function copy(NDArray|array|float|int $a, int $device = NULL): NDArray {}

    /**
     * Adds a new axis to the array at the specified position, thereby expanding its shape.
     *
     * @param NDArray|array|float|int $target Target array.
     * @param int|int[]|null $axis This parameter specifies the position where the new axis (or axes) will be inserted within the expanded array.
     * @return NDArray
     */
    public static function expand_dims(NDArray|array|float|int $target, int|array $axis = NULL): NDArray {}

    /**
     * Return a **copy** of the array `$a` into one dimension.
     *
     * @param NDArray|array|float|int $a Target array
     * @return NDArray A copy of `$a`, with dimensions collapsed to 1-d, in the same device.
     */
    public static function flatten(NDArray|array|float|int $a): NDArray {}

    /**
     * Changes the shape of the NDArray.
     *
     * @param NDArray|array|float|int $target Target array
     * @param int[] $shape Target shape
     * @return NDArray|float|int A copy of `$a`, with dimensions collapsed to 1-d, in the same device.
     */
    public static function reshape(NDArray|array|float|int $target, array $shape): NDArray|float|int {}

    /**
     * Return a PHP array representing the shape of the NDArray
     *
     * @return int[] Shape of the NDArray
     */
    public function shape(): array {}

    /**
     * Return the total number of elements in the NDArray.
     *
     * @return int Total number of elements in the NDArray
     */
    public function size(): int {}

    /**
     * Return a PHP array with the same shape and a **copy** of values of the NDArray.
     *
     * @return array PHP array with same shape and a copy of the elements of the NDArray
     */
    public function toArray(): array {}

    /**
     * Return the transpose of matrix `$a`
     *
     * @param NDArray|array|float|int $a Target array
     * @param array|null $axes For an n-D array, if $axes are given, their order indicates how the axes are permuted
     * @return NDArray $a transposed
     */
    public static function transpose(NDArray|array|float|int $a, ?array $axes): NDArray {}

    /**
     * Interchange two axes of an array.
     *
     * @param NDArray|array|float|int $a Target array
     * @param int $axis1 First axis
     * @param int $axis2 Second axis
     * @return NDArray
     */
    public static function swapaxes(NDArray|array|float|int $a, int $axis1, int $axis2): NDArray {}

    /**
     * Roll the specified axis backwards, until it lies in a given position.
     *
     * @param NDArray|array|float|int $a Target array
     * @param int $axis
     * @param int $start
     * @return NDArray
     */
    public static function rollaxis(NDArray|array|float|int $a, int $axis, int $start = 0): NDArray {}

    /**
     * Move axes of an array to new positions.
     *
     * @param NDArray|array|float|int $a Target array
     * @param int|array $source
     * @param int|array $destination
     * @return NDArray
     */
    public static function moveaxis(NDArray|array|float|int $a, int|array $source, int|array $destination): NDArray {}

    /**
     * Stack arrays in sequence vertically (row wise).
     *
     * @param NDArray[] $arrays
     * @return NDArray
     */
    public static function vstack(array $arrays): NDArray {}

    /**
     * Stack arrays in sequence horizontally (column wise).
     *
     * @param NDArray[] $arrays
     * @return NDArray
     */
    public static function hstack(array $arrays): NDArray {}

    /**
     * Stack arrays in sequence depth wise (along third axis).
     *
     * @param NDArray[] $arrays
     * @return NDArray
     */
    public static function dstack(array $arrays): NDArray {}

    /**
     * Join a sequence of arrays along an existing axis.
     *
     * @param NDArray[] $arrays
     * @param int|null $axis
     * @return NDArray
     */
    public static function concatenate(array $arrays, ?int $axis = 0): NDArray {}

    /**
     * Append values to the end of an array.
     *
     * @param NDArray|array $array
     * @param NDArray|array $values
     * @param int|null $axis
     * @return NDArray
     */
    public static function append(NDArray|array $array, NDArray|array $values, ?int $axis): NDArray {}

    /**
     * Stack 1-D arrays as columns into a 2-D array.
     *
     * @param NDArray[] $arrays
     * @return NDArray
     */
    public static function column_stack(array $arrays): NDArray {}

    /**
     * Creates a new NDArray from a PHP array.
     *
     * It is the equivalent of `new NDArray($array);`
     *
     * @param array|float|int $array
     * @return NDArray
     */
    public static function array(array|float|int $array): NDArray {}

    /**
     * This function returns a square array, where the main diagonal consists of ones and all other
     * elements are zeros. It takes a parameter `$size` which determines the number of rows and columns
     * in the output array.
     *
     * @param int $size Number of rows and columns of the new square array of size `($size, $size)`
     * @return NDArray Return a new square array of size `($size, $size)`
     */
    public static function identity(int $size): NDArray {}

    /**
     * The function creates a new NDArray with the specified shape, filled with ones.
     *
     * @param int[] $shape
     * @return NDArray
     */
    public static function ones(array $shape): NDArray {}

    /**
     * The function creates a new NDArray with the specified shape, filled with zeros.
     *
     * @param int[] $shape
     * @return NDArray
     */
    public static function zeros(array $shape): NDArray {}

    /**
     * Dumps the internal information of the NDArray.
     *
     * @return void
     */
    public function dump() : void {}

    /**
     * Dumps information about available devices for GPU computation.
     *
     * @return void
     */
    public static function dumpDevices(): void {}

    /**
     * @param NDArray|array|float|int $a
     * @return bool
     */
    public static function all(NDArray|array|float|int $a): bool {}

    /**
     * Checks if all elements in two arrays are approximately equal within a specified tolerance element-wise.
     *
     * @param NDArray|array|float|int $a
     * @param NDArray|array|float|int $b
     * @param float $rtol
     * @param float $atol
     * @return NDArray
     */
    public static function allclose(NDArray|array|float|int $a, NDArray|array|float|int $b, float $rtol = 1e-05, float $atol = 1e-08): NDArray {}

    /**
     * Performs an element-wise equality comparison between two arrays and returns a
     * new array of the same shape. The result will be 1 where the
     * elements are equal and 0 where they are not.
     *
     * @param NDArray|array|float|int $a
     * @param NDArray|array|float|int $b
     * @return NDArray
     */
    public static function equal(NDArray|array|float|int $a, NDArray|array|float|int $b): NDArray {}

    /**
     * Performs an element-wise greater-than comparison between two arrays and returns a new array of the same shape.
     * The result will be 1 where the elements in the first array are greater than the corresponding elements in the
     * second array, and 0 otherwise.
     *
     * @param NDArray|array|float|int $a
     * @param NDArray|array|float|int $b
     * @return NDArray
     */
    public static function greater(NDArray|array|float|int $a, NDArray|array|float|int $b): NDArray {}

    /**
     * Performs an element-wise greater-than-or-equal-to comparison between two arrays and returns a new
     * array of the same shape. The result will be 1 where the elements in the first array are greater than
     * or equal to the corresponding elements in the second array, and 0 otherwise.
     *
     * @param NDArray|array|float|int $a
     * @param NDArray|array|float|int $b
     * @return NDArray
     */
    public static function greater_equal(NDArray|array|float|int $a, NDArray|array|float|int $b): NDArray {}

    /**
     * Performs an element-wise less-than comparison between two arrays and returns a new array of the same shape.
     * The result will be 1 where the elements in the first array are less than the corresponding elements in the second
     * array, and 0 otherwise.
     *
     * @param NDArray|array|float|int $a
     * @param NDArray|array|float|int $b
     * @return NDArray
     */
    public static function less(NDArray|array|float|int $a, NDArray|array|float|int $b): NDArray {}

    /**
     * Performs an element-wise less-than-or-equal-to comparison between two arrays and returns a
     * new array of the same shape. The result will be 1 where the elements in the first array
     * are less than or equal to the corresponding elements in the second array, and 0 otherwise.
     *
     * @param NDArray|array|float|int $a
     * @param NDArray|array|float|int $b
     * @return NDArray
     */
    public static function less_equal(NDArray|array|float|int $a, NDArray|array|float|int $b): NDArray {}

    /**
     * Performs an element-wise inequality comparison between two arrays and returns a new array of the same shape.
     * The result will be 1 where the elements are not equal and 0 where they are equal.
     *
     * @param NDArray|array|float|int $a
     * @param NDArray|array|float|int $b
     * @return NDArray
     */
    public static function not_equal(NDArray|array|float|int $a, NDArray|array|float|int $b): NDArray {}

    /**
     * Computes the sum of the diagonal elements of a square array, also known as the trace of the array.
     *
     * @param NDArray|array $a
     * @return float
     */
    public static function trace(NDArray|array $a): float {}

    /**
     * Calculates the Singular Value Decomposition (SVD) of an array, which decomposes the array
     * into three separate arrays: U, Sigma, and V^T.
     *
     * @param NDArray|array $a
     * @return array PHP array containing the Unitary Arrays (U) `[0]`, the vector(s) with the singular values (S) `[1]` and the unitary arrays (Vh) `[2]`
     */
    public static function svd(NDArray|array $a): array {}

    /**
     * Solves a linear system of equations for `x`, where `Ax = b`, and `A` and `b` are given arrays.
     *
     * @param NDArray|array $a
     * @param NDArray|array $b
     * @return NDArray
     */
    public static function solve(NDArray|array $a, NDArray|array $b): NDArray {}

    /**
     * Calculates the QR decomposition of an array, which expresses it as the product of an orthogonal matrix (Q)
     * and an upper triangular matrix (R).
     *
     * @param NDArray|array $a
     * @return array
     */
    public static function qr(NDArray|array $a): array {}

    /**
     * Computes the outer product of two vectors, which results in a higher-dimensional array with dimensions
     * calculated from the input arrays' dimensions.
     *
     * @param NDArray|array $a
     * @param NDArray|array $b
     * @return NDArray
     */
    public static function outer(NDArray|array $a, NDArray|array $b): NDArray {}

    /**
     * Calculates different norms (e.g., L1 norm, L2 norm) of an array,
     * providing various measures of its magnitude.
     *
     * #### $order options
     *
     * - **1** - L1-Norm
     * - **2** - L2-Norm
     *
     * @param NDArray|array $a
     * @param int $order (1) L1-Norm, (2) L2-Norm
     * @return float
     */
    public static function norm(NDArray|array $a, int $order = 2): float {}

    /**
     * Calculates the numerical rank of a matrix, number of singular
     * values of the array that are greater than tol.
     *
     * @param NDArray|array $a
     * @param float $tol
     * @return NDArray
     */
    public static function matrix_rank(NDArray|array $a, float $tol = 1e-6): NDArray {}

    /**
     * Performs matrix multiplication between two arrays and returns the result as a new array.
     *
     * @param NDArray|array $a Input array
     * @param NDArray|array $b Input array
     * @return NDArray The matrix product of the inputs. This is a scalar only when both x1, x2 are 1-d vectors.
     */
    public static function matmul(NDArray|array $a, NDArray|array $b): NDArray {}

    /**
     * Computes the LU factorization of a matrix
     *
     * @param NDArray|array $a
     * @return NDArray
     */
    public static function lu(NDArray|array $a): NDArray {}

    /**
     * Performs the least-squares solution to a linear matrix equation `Ax = b`,
     * where `$a` is a given array and `$b` is the target array.
     *
     * @param NDArray|array $a
     * @param NDArray|array $b
     * @return NDArray
     */
    public static function lstsq(NDArray|array $a, NDArray|array $b): NDArray {}

    /**
     * Compute the inverse of a matrix, such that `$a * nd::inv($a) = np::identity($a->shape())`.
     *
     * @param NDArray|array $a
     * @return NDArray
     */
    public static function inv(NDArray|array $a): NDArray {}

    /**
     * Calculates the inner product of two arrays. This operation involves multiplying corresponding
     * elements of the arrays and summing them up.
     *
     * - When dealing with N-D arrays, the inner product is computed by taking a sum product over the last axes of the arrays.
     *
     * @param NDArray|array|float|int $a Input array
     * @param NDArray|array|float|int $b Input array
     * @return NDArray|float If both `$a` and `$b` are scalars or 1-D arrays, the function will return a scalar value.
     *                       Otherwise, if the input arrays have more than one dimension, an array will be returned.
     */
    public static function inner(NDArray|array|float|int $a, NDArray|array|float|int $b): NDArray|float {}

    /**
     * Computes the eigenvalues and eigenvectors of a square array.
     *
     * @param NDArray|array $a
     * @return array
     */
    public static function eig(NDArray|array $a): array {}

    /**
     * The `dot` function performs the dot product of two arrays. The behaviour of the function
     * varies depending on the dimensions and shapes of the input arrays:
     *
     * - If both `$a` and `$b` are 1-D arrays, the dot product is computed as the inner product of the vectors.
     * - If both `$a` and `$b` are 2-D arrays, the dot product is computed as the matrix multiplication
     * (See <a href="#">NDArray::matmul</a>).
     * - If either `$a` or `$b` is a 0-D (scalar) array, the dot product is equivalent to element-wise multiplication.
     * (See <a href="#">NDArray::multiply</a>).
     * - If `$a` is an N-D array and `$b` is a 1-D array, the dot product is computed as the
     * sum product over the last axis of `$a` and `$b`.
     *
     * @param NDArray|array|float|int $a Input array
     * @param NDArray|array|float|int $b Input array
     * @return NDArray
     */
    public static function dot(NDArray|array|float|int $a, NDArray|array|float|int $b): NDArray {}

    /**
     * Computes the determinant of a square array, which represents the scaling factor of the volume
     * of the array transformation.
     *
     * @param NDArray|array $a
     * @return float
     */
    public static function det(NDArray|array $a): float {}

    /**
     * Computes the condition number of an array.
     *
     * @param NDArray|array $a
     * @return float
     */
    public static function cond(NDArray|array $a): float {}

    /**
     * Calculates the Cholesky decomposition of a positive-definite array, decomposing
     * it into a lower triangular matrix and its conjugate transpose.
     *
     * @param NDArray|array $a
     * @return NDArray
     */
    public static function cholesky(NDArray|array $a): NDArray {}

    /**
     * Return evenly spaced values within a given interval.
     *
     * @param float|int $stop
     * @param float|int $start
     * @param float|int $step
     * @return NDArray
     */
    public static function arange(float|int $stop, float|int $start = 0, float|int $step = 1): NDArray {}

    /**
     * Remove axes of length one from $a.
     *
     * @param NDArray|array $a Input array
     * @param int|int[] $axis Selects a subset of the entries of length one in the shape.
     * If an axis is selected with shape entry greater than one, an error is raised.
     * @return NDArray|float The input array, but with all or a subset of the dimensions of length 1 removed.
     * This is always $a itself or $a view into $a. Note that if all axes are squeezed,
     * the result is a 0d array and not a scalar.
     */
    public static function squeeze(NDArray|array $a, int|array $axis): NDArray|float {}

    /**
     * Extract a diagonal or construct a diagonal array.
     *
     * @param NDArray|array $a
     * @return NDArray
     */
    public static function diag(NDArray|array $a): NDArray {}

    /**
     * Return a new array of given shape and type, filled with $fill_value.
     *
     * @param int[] $shape Shape of the new array
     * @param float|int $fill_value Fill value
     * @return NDArray
     */
    public static function full(array $shape, float|int $fill_value): NDArray {}

    /**
     * Fill the array with a scalar value.
     *
     * @param float|int $fill_value Fill value
     * @return NDArray
     */
    public function fill(float|int $fill_value): NDArray {}

    /**
     * Returns the indices of the minimum values along an axis.
     *
     * @param NDArray|array $a Target array
     * @param int|null $axis If NULL, the index is into the flattened array, otherwise along the specified axis.
     * @param bool $keepdims
     * @return NDArray Array of indices into the array. It has the same shape as $a with the dimension along $axis removed.
     */
    public static function argmin(NDArray|array $a, ?int $axis, bool $keepdims = false): NDArray {}

    /**
     * Returns the indices of the maximum values along an axis.
     *
     * @param NDArray|array $a Target array
     * @param int|null $axis If NULL, the index is into the flattened array, otherwise along the specified axis.
     * @param bool $keepdims
     * @return NDArray Array of indices into the array. It has the same shape as $a with the dimension along axis removed.
     */
    public static function argmax(NDArray|array $a, ?int $axis, bool $keepdims = false): NDArray {}

    /**
     * Array slicing, each argument represents a slice of a dimension.
     *
     * Empty arrays represent all values of a dimension, arrays with values are treated in
     * the format [start, stop, step], when only one value exists, it is automatically
     * assigned to stop, the default value of start is 0 and step is 1.
     *
     * When instead of an array, a number is passed, it is also assigned to the
     * stop of that dimension.
     *
     * Ex: Get the first row of a matrix:
     * $array->slice(0)
     *
     * Ex: Get the last column of a matrix:
     * $array->slice([], -1);
     *
     * Ex: Get the first two columns of the first row:
     * $array->slice(0, [0,2]);
     *
     * @param array ...$indices
     * @return NDArray|float
     */
    public function slice(...$indices): NDArray|float {};
}