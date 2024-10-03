--TEST--
NDArray::cumsum
--FILE--
<?php
$empty = \NDArray::array([[]]);
$single = \NDArray::array([[3]]);
$a = \NDArray::array([[1, 2, 3], [4, 5, 6]]);
print_r(\NDArray::cumsum($empty)->toArray());
print_r(\NDArray::cumsum($single)->toArray());
print_r(\NDArray::cumsum($a)->toArray());
print_r(\NDArray::cumsum($a, 0)->toArray());
print_r(\NDArray::cumsum($a, 1)->toArray());
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
            [1] => 3
            [2] => 6
            [3] => 10
            [4] => 15
            [5] => 21
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
            [0] => 5
            [1] => 7
            [2] => 9
        )

)
Array
(
    [0] => Array
        (
            [0] => 1
            [1] => 3
            [2] => 6
        )

    [1] => Array
        (
            [0] => 4
            [1] => 9
            [2] => 15
        )

)