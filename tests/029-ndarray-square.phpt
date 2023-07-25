--TEST--
NDArray::square
--FILE--
<?php
$a = \NDArray::array([[-156, 150], [19, -39]]);
print_r(\NDArray::square($a)->toArray());
print_r(\NDArray::square($a[0])->toArray());
print_r(\NDArray::square([[0],[-0.5]])->toArray());
?>
--EXPECT--
Array
(
    [0] => Array
        (
            [0] => 24336
            [1] => 22500
        )

    [1] => Array
        (
            [0] => 361
            [1] => 1521
        )

)
Array
(
    [0] => 24336
    [1] => 22500
)
Array
(
    [0] => Array
        (
            [0] => 0
        )

    [1] => Array
        (
            [0] => 0.25
        )

)