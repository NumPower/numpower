--TEST--
NDArray::sqrt
--FILE--
<?php
$a = \NDArray::array([[-156, 150], [19, -39]]);
print_r(\NDArray::sqrt($a)->toArray());
print_r(\NDArray::sqrt($a[0])->toArray());
print_r(\NDArray::sqrt([[0],[-0.5]])->toArray());
?>
--EXPECT--
Array
(
    [0] => Array
        (
            [0] => NAN
            [1] => 12.247448921204
        )

    [1] => Array
        (
            [0] => 4.3588991165161
            [1] => NAN
        )

)
Array
(
    [0] => NAN
    [1] => 12.247448921204
)
Array
(
    [0] => Array
        (
            [0] => 0
        )

    [1] => Array
        (
            [0] => NAN
        )

)