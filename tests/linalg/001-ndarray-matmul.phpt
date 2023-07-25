--TEST--
NDArray::matmul
--FILE--
<?php
$a = \NDArray::array([[1, 2], [3, 4]]);
$b = \NDArray::array([[5, 6], [7, 8]]);
$c = \NDArray::array([9, 10]);
$d = \NDArray::array([[9], [10]]);
print_r(\NDArray::matmul($a, $b)->toArray());
print_r(\NDArray::matmul([[1, 2], [3, 4]], [[5, 6], [7, 8]])->toArray());
print_r(\NDArray::matmul($a, $d)->toArray());
print_r(\NDArray::matmul($a, $c)->toArray());
?>
--EXPECT--
Array
(
    [0] => Array
        (
            [0] => 19
            [1] => 22
        )

    [1] => Array
        (
            [0] => 43
            [1] => 50
        )

)
Array
(
    [0] => Array
        (
            [0] => 19
            [1] => 22
        )

    [1] => Array
        (
            [0] => 43
            [1] => 50
        )

)
Array
(
    [0] => Array
        (
            [0] => 29
        )

    [1] => Array
        (
            [0] => 67
        )

)

Fatal error: Uncaught Error: Arrays must have the same shape. Broadcasting not implemented. in /src/tests/linalg/001-ndarray-matmul.php:9
Stack trace:
#0 /src/tests/linalg/001-ndarray-matmul.php(9): NDArray::matmul(Object(NDArray), Object(NDArray))
#1 {main}
  thrown in /src/tests/linalg/001-ndarray-matmul.php on line 9