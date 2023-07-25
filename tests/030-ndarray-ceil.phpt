--TEST--
NDArray::ceil
--FILE--
<?php
$a = \NDArray::array([[-156.50, 150.525435], [0, -39.151414]]);
print_r(\NDArray::ceil($a)->toArray());
print_r(\NDArray::ceil($a[0])->toArray());
print_r(\NDArray::ceil([[0.12],[-0.513124]])->toArray());
?>
--EXPECT--
Array
(
    [0] => Array
        (
            [0] => -156
            [1] => 151
        )

    [1] => Array
        (
            [0] => 0
            [1] => -39
        )

)
Array
(
    [0] => -156
    [1] => 151
)
Array
(
    [0] => Array
        (
            [0] => 1
        )

    [1] => Array
        (
            [0] => -0
        )

)