--TEST--
NDArray::cumprod
--FILE--
<?php
$empty = \NDArray::array([[]]);
$single = \NDArray::array([[3]]);
$a = \NDArray::array([[1, 2, 3], [4, 5, 6]]);
print_r(\NDArray::cumprod($empty)->toArray());
print_r(\NDArray::cumprod($single)->toArray());
print_r(\NDArray::cumprod($a)->toArray());
print_r(\NDArray::cumprod($a, 0)->toArray());
print_r(\NDArray::cumprod($a, 1)->toArray());
?>
--EXPECT--
Array
(
    [0] => Array
        (
        )

)
Array
(
    [0] => Array
        (
            [0] => 3
        )

)
Array
(
    [0] => Array
        (
            [0] => 1
            [1] => 2
            [2] => 6
            [3] => 24
            [4] => 120
            [5] => 720
        )

)
Array
(
    [0] => Array
        (
            [0] => 1
            [1] => 2
            [2] => 3
        )

    [1] => Array
        (
            [0] => 4
            [1] => 10
            [2] => 18
        )

)
Array
(
    [0] => Array
        (
            [0] => 1
            [1] => 2
            [2] => 6
        )

    [1] => Array
        (
            [0] => 4
            [1] => 20
            [2] => 120
        )

)