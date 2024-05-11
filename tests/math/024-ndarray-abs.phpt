--TEST--
NDArray::abs
--FILE--
<?php
$a = \NDArray::array([[0, -0.5], [0, -0.5]]);
print_r(\NDArray::abs($a)->toArray());
print_r(\NDArray::abs($a[0])->toArray());
print_r(\NDArray::abs([[0],[-0.5]])->toArray());
?>
--EXPECT--
Array
(
    [0] => Array
        (
            [0] => 0
            [1] => 0.5
        )

    [1] => Array
        (
            [0] => 0
            [1] => 0.5
        )

)
Array
(
    [0] => 0
    [1] => 0.5
)
Array
(
    [0] => Array
        (
            [0] => 0
        )

    [1] => Array
        (
            [0] => 0.5
        )

)