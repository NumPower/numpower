--TEST--
NDArray::tanh
--FILE--
<?php
$a = \NDArray::array([[0, -0.5], [0, -0.5]]);
print_r(\NDArray::tanh($a)->toArray());
print_r(\NDArray::tanh($a[0])->toArray());
print_r(\NDArray::tanh([[0],[-0.5]])->toArray());
?>
--EXPECT--
Array
(
    [0] => Array
        (
            [0] => 0
            [1] => -0.46211716532707
        )

    [1] => Array
        (
            [0] => 0
            [1] => -0.46211716532707
        )

)
Array
(
    [0] => 0
    [1] => -0.46211716532707
)
Array
(
    [0] => Array
        (
            [0] => 0
        )

    [1] => Array
        (
            [0] => -0.46211716532707
        )

)