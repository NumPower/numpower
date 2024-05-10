--TEST--
NDArray::exp2
--FILE--
<?php
$a = \NDArray::array([[1, 2], [3, 4]]);
print_r(\NDArray::exp2($a)->toArray());
print_r(\NDArray::exp2($a[0])->toArray());
print_r(\NDArray::exp2([[1],[2]])->toArray());
?>
--EXPECT--
Array
(
    [0] => Array
        (
            [0] => 2
            [1] => 4
        )

    [1] => Array
        (
            [0] => 8
            [1] => 16
        )

)
Array
(
    [0] => 2
    [1] => 4
)
Array
(
    [0] => Array
        (
            [0] => 2
        )

    [1] => Array
        (
            [0] => 4
        )

)