--TEST--
NDArray::sign
--FILE--
<?php
$a = \NDArray::array([[-156, 150], [19, -39]]);
print_r(\NDArray::sign($a)->toArray());
print_r(\NDArray::sign($a[0])->toArray());
print_r(\NDArray::sign([[0],[-0.5]])->toArray());
?>
--EXPECT--
Array
(
    [0] => Array
        (
            [0] => -1
            [1] => 1
        )

    [1] => Array
        (
            [0] => 1
            [1] => -1
        )

)
Array
(
    [0] => -1
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
            [0] => -1
        )

)