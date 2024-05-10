--TEST--
NDArray::arctanh
--FILE--
<?php
$a = \NDArray::array([[0, -0.5], [0, -0.5]]);
print_r(\NDArray::arctanh($a)->toArray());
print_r(\NDArray::arctanh($a[0])->toArray());
print_r(\NDArray::arctanh([[0],[-0.5]])->toArray());
?>
--EXPECT--
Array
(
    [0] => Array
        (
            [0] => 0
            [1] => -0.5493061542511
        )

    [1] => Array
        (
            [0] => 0
            [1] => -0.5493061542511
        )

)
Array
(
    [0] => 0
    [1] => -0.5493061542511
)
Array
(
    [0] => Array
        (
            [0] => 0
        )

    [1] => Array
        (
            [0] => -0.5493061542511
        )

)