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