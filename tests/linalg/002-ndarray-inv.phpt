--TEST--
NDArray::inv
--FILE--
<?php
$a = \NDArray::array([[1, 2], [3, 4]]);
$b = \NDArray::array([[5, 6], [7, 8]]);
$c = \NDArray::array([9, 10]);
$d = \NDArray::array([[9], [10]]);
print_r(\NDArray::inv($a)->toArray());
print_r(\NDArray::inv([[1, 2], [3, 4]])->toArray());
?>
--EXPECT--
Array
(
    [0] => Array
        (
            [0] => -2
            [1] => 1
        )

    [1] => Array
        (
            [0] => 1.5
            [1] => -0.5
        )

)
Array
(
    [0] => Array
        (
            [0] => -2
            [1] => 1
        )

    [1] => Array
        (
            [0] => 1.5
            [1] => -0.5
        )

)