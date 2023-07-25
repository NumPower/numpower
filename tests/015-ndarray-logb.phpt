--TEST--
NDArray::logb
--FILE--
<?php
$a = \NDArray::array([[1, 2], [3, 4]]);
print_r(\NDArray::logb($a)->toArray());
print_r(\NDArray::logb($a[0])->toArray());
print_r(\NDArray::logb([[1],[2]])->toArray());
?>
--EXPECT--
Array
(
    [0] => Array
        (
            [0] => 0
            [1] => 1
        )

    [1] => Array
        (
            [0] => 1
            [1] => 2
        )

)
Array
(
    [0] => 0
    [1] => 1
)
Array
(
    [0] => Array
        (
            [0] => 0
        )

    [1] => Array
        (
            [0] => 1
        )

)