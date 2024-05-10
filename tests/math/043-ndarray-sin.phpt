--TEST--
NDArray::sin
--FILE--
<?php
$a = \NDArray::array([[0, -0.5], [0, -0.5]]);
print_r(\NDArray::sin($a)->toArray());
print_r(\NDArray::sin($a[0])->toArray());
print_r(\NDArray::sin([[0],[-0.5]])->toArray());
?>
--EXPECT--
Array
(
    [0] => Array
        (
            [0] => 0
            [1] => -0.47942554950714
        )

    [1] => Array
        (
            [0] => 0
            [1] => -0.47942554950714
        )

)
Array
(
    [0] => 0
    [1] => -0.47942554950714
)
Array
(
    [0] => Array
        (
            [0] => 0
        )

    [1] => Array
        (
            [0] => -0.47942554950714
        )

)